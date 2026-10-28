"""R2 fine-tune of the student WoWmayo (DATA_PLAN_R2.md 7, recipe confirmed by the user 10/27 "제안대로 가자"; do not change
the shares, the row density, the offsets or the selection rule without the user): one joint stage from s2 final.

MixtureDataset: sources (MoraiDataset each) at fixed sample shares, one epoch, the length counted on the NEW data:
  new rows (A, B, D) ~4 Hz = the 2 Hz rows of the earlier stages (i % frame_stride == 0, frame_stride 5) plus, per 0.5 s
  period, the row 0.2 or 0.3 s after it (offset 2 or 3 grid rows, a hash of the episode and period): twice as dense, no
  sample twice (DATA_PLAN_R2 7 "길이", corrected 10/27; the sample count equals "2 Hz rows, 2 passes");
  total = (new ~4 Hz rows) / (share of the new sources),   count_s = share_s x total;
  old sources (recovery, curve, start relabel, replay) draw from their 2 Hz rows only, so the replay rows are s2's own
  and carry s2's rule check.
Every source draws its count without replacement from those rows (a random subset when it has more). A new source with
fewer ~4 Hz rows than its share asks for (the bundles' row counts are not exactly 2:1:1) continues with its other,
still unused grid rows (distinct frames, offsets 1 and 4); only when every row is used, or for an old source past its
2 Hz rows, rows repeat. The summary (start log) counts both per source. The (source, row, tier) schedule is fixed by the
seed and shuffled once: the shares hold over the whole run, and a resumed run sees the same order.
Each item also carries, for StudentVLA.forward:
  gt_row     the source's loss kind (gt: fm + traj CE + ViT KD; distill: every distillation loss, vkd on rule_ok rows)
  hist_drop  GT rows only: the ego history reaches the LLM as zero displacements with probability hist_dropout (a hash of
             the schedule position: the same draw on every rank and after a resume)
  hist_synth sources with `hist: thirds` (the recovery rows: R2 D, old recovery; user 10/28): the same hash picks the real,
             the zero-displacement or the lane-shaped history (MoraiDataset synth; hist_xyz / hist_rot) a third each; a row
             without a lane-shaped history keeps its real one. R2 D start rows (synth_only) always use theirs.
  vkd_ok     replay rows: s2's soft-target rule check (MoraiDataset rule_ok); GT rows False
FtSelect: final/ selection. Every `every_steps` (and at step 0 = s2 final, and at the last step) the expert's trajectory
ADE (early_stop.expert_ade: 6.4 s, mean of K samples, LiDAR on, seed 0 per sample) on the validation sets (general,
dynamic, static, curve); final/ = the last measurement whose `ref` set ADE is not more than rel_worse above the step-0
value (the last checkpoint when it holds; step 0, i.e. s2 final, if none does). Snapshots of the trainable weights per
measurement as in early_stop.TrajEarlyStop; state in <output_dir>/ft_select.json, so a resumed run continues.
Records (user 10/27 "메트릭을 철저히 기록"): per set also FDE and the mean |lateral| / |longitudinal| error in the GT path
frame (ft_select.json `metrics`), every row in <output_dir>/ft_val_rows.jsonl; scripts/ft_metrics.py makes CSVs and figures.
Stop switch (user 10/28): reads `stop_ref` = the ref set without the rows of `stop_exclude` (general_wo_relabel: the
start-relabelled rows keep their old "wait" GT in the gate set); the selection and the gate keep `ref`.
"""
import hashlib, json, logging, math, time
from pathlib import Path

import numpy as np, pandas as pd, torch
from safetensors.torch import load_file, save_file
from transformers.trainer_callback import TrainerCallback

from morai_vla.data import MoraiDataset
from morai_vla.early_stop import expert_ade
from morai_vla.processor import get_preprocess_fn

log = logging.getLogger(__name__)
LOSSES = ("gt", "distill")
FIRST, OTHER, REPEAT = 0, 1, 2                 # schedule tiers: the source's own row set, other unused grid rows, repeats


def mix(*key):
    """uniform [0, 1) from a key, well mixed (data.unit_hash is CRC-32, which is linear: keys differing only in their
    last digits, e.g. consecutive periods of one episode, give correlated values)."""
    return int.from_bytes(hashlib.blake2b("/".join(map(str, key)).encode(), digest_size=8).digest(), "little") / 2**64


def dense_rows(ep, i, stride=5, extra=(2, 3), seed=0):
    """the new data's ~4 Hz rows (bool per row): the 2 Hz lattice plus, per period of `stride` grid rows, the row
    extra[0] or extra[1] rows after it (picked by a hash of episode and period)."""
    o = i % stride; out = o == 0
    m = np.isin(o, extra)
    pick = np.array([extra[0] if mix("offset", seed, e, b) < 0.5 else extra[1] for e, b in zip(ep[m], (i - o)[m])], int)
    out[np.flatnonzero(m)] = o[m] == pick
    return out


def plan(rows, shares, new, stride=5, extra=(2, 3), seed=0, epochs=1):
    """rows {source: (episode array, grid row i array) of its usable rows}, shares {source: share}, new: the sources the
    length is counted on -> (schedule (N, 3) int64 of (source position, row position, tier), summary). epochs: passes over
    the new sources' ~4 Hz rows (each pass a new permutation, tier FIRST, so rows_used 2.0 = two passes); total samples =
    new rows x epochs / new share, the other sources draw their share of that once from their 2 Hz rows."""
    names = list(rows); tot_share = sum(shares.values())
    w = {s: shares[s] / tot_share for s in names}
    first = {s: dense_rows(*rows[s], stride, extra, seed) if s in new else rows[s][1] % stride == 0 for s in names}
    n_new = sum(int(first[s].sum()) for s in new)
    assert n_new > 0, f"no rows in the new sources {new}"
    total = int(round(n_new * epochs / sum(w[s] for s in new)))
    rng, parts = np.random.default_rng(seed), []
    summary = {"total": total, "epochs": epochs, "new_rows_4hz": n_new, "new_rows_2hz": sum(int((rows[s][1] % stride == 0).sum()) for s in new), "sources": {}}
    for j, s in enumerate(names):
        want = int(round(w[s] * total)); a = np.flatnonzero(first[s]); b = np.flatnonzero(~first[s]) if s in new else a[:0]
        assert want == 0 or len(a), f"source {s} has no rows"
        tiers = [(FIRST, rng.permutation(a)) for _ in range(epochs if s in new else 1)] + [(OTHER, rng.permutation(b))]
        while sum(len(x) for _, x in tiers) < want: tiers.append((REPEAT, rng.permutation(a)))
        got, left = [], want
        for t, x in tiers:
            x = x[:left]; left -= len(x)
            got.append(np.stack([np.full(len(x), j), x, np.full(len(x), t)], 1))
        sched = np.concatenate(got); parts.append(sched)
        n = {t: int((sched[:, 2] == t).sum()) for t in (FIRST, OTHER, REPEAT)}
        summary["sources"][s] = {"share": round(w[s], 4), "samples": int(len(sched)), "rows": int(len(a)),
                                 "rows_used": round(n[FIRST] / max(len(a), 1), 3), "other_offsets": n[OTHER], "repeated": n[REPEAT]}
        if n[OTHER] or n[REPEAT]:
            log.warning("MixtureDataset %s: share %.3f needs %d samples, %d rows: %d from other offsets, %d repeated",
                        s, w[s], want, len(a), n[OTHER], n[REPEAT])
    sched = np.concatenate(parts)
    return sched[rng.permutation(len(sched))], summary


class MixtureDataset(torch.utils.data.Dataset):
    """sources: {name: {share, loss: gt | distill, new: bool, dataset: MoraiDataset kwargs}}; common: MoraiDataset kwargs
    shared by every source (lidar_dropout, jitter, augment). Sources read every grid row; frame_stride and extra_offsets
    define the 2 Hz / ~4 Hz row sets (module docstring)."""

    def __init__(self, sources, frame_stride=5, extra_offsets=(2, 3), hist_dropout=0.5, seed=0, common=None, model_config=None,
                 vla_preprocess_args=None, epochs=1):
        self.names = list(sources); self.hist_dropout, self.seed, self.stride = hist_dropout, seed, frame_stride
        self.loss = [sources[s]["loss"] for s in self.names]
        self.thirds = [sources[s].get("hist") == "thirds" for s in self.names]
        self.real_synth = [sources[s].get("hist") == "real_synth" for s in self.names]   # 10/28 dyn ft: stop rows real / synth 1/2, no zero
        assert set(self.loss) <= set(LOSSES), self.loss
        self.parts = [MoraiDataset(**{**(common or {}), **sources[s]["dataset"], "frame_stride": 1}, seed=seed, model_config=model_config,
                                   vla_preprocess_args=vla_preprocess_args) for s in self.names]
        self.schedule, self.summary = plan({s: (p.rows.episode.astype(str).to_numpy(), p.rows.i.to_numpy()) for s, p in zip(self.names, self.parts)},
                                           {s: float(sources[s]["share"]) for s in self.names},
                                           [s for s in self.names if sources[s].get("new")], frame_stride, tuple(extra_offsets), seed, epochs)
        drop, syn = map(np.array, zip(*(self._hist_mode(k) for k in range(len(self.schedule))))) if len(self.schedule) else ([], [])
        gt = np.array([self.loss[j] == "gt" for j in self.schedule[:, 0]], bool)
        self.summary.update(gt_share=round(float(gt.mean()), 4), hist_drop_share=round(float(np.mean(drop)), 4),
                            hist_synth_share=round(float(np.mean(syn)), 4))   # synth: before the per-row availability check
        log.info("MixtureDataset: %s", json.dumps(self.summary))

    def __len__(self):
        return len(self.schedule)

    def _hist_mode(self, k):
        """(zero-displacement history, lane-shaped history) of schedule position k, before the per-row checks."""
        j, r = int(self.schedule[k, 0]), int(self.schedule[k, 1]); u = mix("hist", self.seed, k)
        rows = self.parts[j].rows
        if self.loss[j] != "gt" or bool(rows.synth_only.iloc[r] if "synth_only" in rows else False): return False, False
        if self.real_synth[j]: return False, u < 0.5
        return (u < 1 / 3, 1 / 3 <= u < 2 / 3) if self.thirds[j] else (u < self.hist_dropout, False)

    def __getitem__(self, k):
        j, r, t = (int(x) for x in self.schedule[k])
        item = self.parts[j][r]
        gt = self.loss[j] == "gt"
        drop, syn = self._hist_mode(k)
        sx, sr = item.pop("synth_xyz", None), item.pop("synth_rot", None); syn = syn and sx is not None   # no lane history: real
        item.update(gt_row=gt, hist_drop=drop, hist_synth=syn, hist_xyz=sx if syn else torch.zeros(1, 16, 3),
                    hist_rot=sr if syn else torch.eye(3).expand(1, 16, 3, 3).clone(),
                    vkd_ok=not gt and bool(item.get("vkd_ok", False)), source=self.names[j], ft_tier=t,
                    ft_offset=int(self.parts[j].rows.i.iloc[r]) % self.stride)
        return item


class FtValSamples(torch.utils.data.Dataset):
    """the generation prompts (LiDAR on, nominal frames) of one validation set: parts [{n, MoraiDataset kwargs}], each
    the first n rows of MoraiDataset(augment False, seed 1, max_samples n) (n 1000 on val + val_asset, frame_stride 10 =
    the gate / early-stop set); rows k::world of the whole set."""

    def __init__(self, parts, model_config, rank=0, world=1):
        self.ds, self.idx = [], []
        for j, p in enumerate(parts):
            p = dict(p); n = p.pop("n")
            ds = MoraiDataset(**p, max_samples=n, augment=False, seed=1)
            ds.rows = ds.rows.iloc[:n].reset_index(drop=True); self.ds.append(ds)
            self.idx += [(j, r) for r in range(len(ds.rows))]
        self.idx = self.idx[rank::world]
        self.fn = get_preprocess_fn(model_config, generation_mode=True)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        j, r = self.idx[k]
        d = dict(self.ds[j].sample(r)); d["cot"] = None; d["tokenized_data"] = self.fn(data=d)
        return {key: v for key, v in d.items() if key not in ("camera_frames", "bev_frames")}


def choose(history, ref="general", rel_worse=0.02):
    """the selection rule on [{step, ade: {set: value}}] (step 0 first): the last measurement whose ref ADE is at most
    (1 + rel_worse) x the step-0 value; rel_worse None = the last measurement (user 10/27 21h)."""
    base = next(h for h in history if h["step"] == 0)["ade"][ref]
    ok = [h for h in history if rel_worse is None or h["ade"][ref] <= (1 + rel_worse) * base]
    return max(ok, key=lambda h: h["step"]), base


class FtSelect(TrainerCallback):
    def __init__(self, sets, every_steps=500, ref="general", rel_worse=0.02, num_samples=6, num_workers=4, stop_rel_worse=None,
                 stop_patience=2, batch=1, stop_ref=None, stop_exclude=None, stop_metric="ade", stop_base=None):
        assert ref in sets, (ref, list(sets))
        # user 10/28: the stop switch reads the ref set without the start-relabelled rows (their GT is the old "wait" label)
        self.stop_ref, self.stop_exclude = stop_ref or ref, stop_exclude
        assert self.stop_ref in sets or stop_exclude, (stop_ref, "needs stop_exclude to derive it from the ref set")
        self.sets, self.every, self.ref, self.rel, self.k, self.workers = sets, every_steps, ref, rel_worse, num_samples, num_workers
        self.batch = batch                                            # rows per prefill (early_stop.expert_ade; 1 = one at a time)
        self.stop, self.patience = stop_rel_worse, stop_patience      # stop switch: run control only, not the selection
        self.stop_metric = stop_metric                                # user 10/28: ade | lat (|lateral error|, GT heading frame) | fde | lon
        self.stop_base = stop_base                                    # fixed reference (10/28 stop round: s2's value, not step 0's)
        self.samples = self.st = self._skip = None

    def _file(self, args):
        return Path(args.output_dir) / "ft_select.json"

    def _save(self, args, state):
        if state.is_world_process_zero: self._file(args).write_text(json.dumps(self.st, indent=1))

    def on_train_begin(self, args, state, control, model=None, **kw):
        f = self._file(args); s = state.global_step                              # > 0 when resumed
        self.st = json.loads(f.read_text()) if f.exists() else {"history": []}
        self.st["history"] = [h for h in self.st["history"] if h["step"] <= s]
        self.st.update(every_steps=self.every, ref=self.ref, rel_worse=self.rel, max_steps=state.max_steps, stop_ref=self.stop_ref,
                       stop_rel_worse=self.stop, stop_metric=self.stop_metric, stop_base=self.stop_base)
        if not any(h["step"] == 0 and self.stop_ref in h["ade"] for h in self.st["history"]):   # an older step 0 lacks stop_ref
            self._measure(args, state, model, "start")                           # s2 final: the reference
        self._save(args, state)

    def _measure(self, args, state, model, tag):
        dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank, world = (torch.distributed.get_rank(), torch.distributed.get_world_size()) if dist else (0, 1)
        m = getattr(model, "module", model); t = time.time()
        if self.samples is None:
            self.samples = {name: FtValSamples(parts, m.config, rank, world) for name, parts in self.sets.items()}
        ade, n, met = {}, {}, {}
        mean = lambda rows: {k: round(float(np.mean([e[k] for _, e in rows])), 5) if rows else math.nan for k in ("ade", "fde", "lat", "lon")}
        for name, samples in self.samples.items():
            rows = expert_ade(m, samples, self.k, self.workers, detail=True, batch=self.batch)
            if dist:
                parts = [None] * world; torch.distributed.all_gather_object(parts, rows); rows = [r for p in parts for r in p]
            met[name] = mean(rows); ade[name], n[name] = met[name]["ade"], len(rows)
            if name == self.ref and self.stop_ref not in self.sets:               # e.g. general_wo_relabel
                if self._skip is None:
                    x = pd.read_parquet(self.stop_exclude, columns=["episode", "i"]); self._skip = set(x.episode.astype(str) + "/" + x.i.astype(str))
                keep = [r for r in rows if r[0] not in self._skip]
                met[self.stop_ref] = mean(keep); ade[self.stop_ref], n[self.stop_ref] = met[self.stop_ref]["ade"], len(keep)
            if state.is_world_process_zero:                                       # per-row record (scripts/ft_metrics.py)
                with open(Path(args.output_dir) / "ft_val_rows.jsonl", "a") as f:
                    f.writelines(json.dumps({"step": state.global_step, "set": name, "key": k, **e}) + "\n" for k, e in rows)
        snap = Path(args.output_dir) / "ft_snapshots" / f"step-{state.global_step}.safetensors"
        if state.is_world_process_zero:
            snap.parent.mkdir(parents=True, exist_ok=True)
            save_file({k: p.detach().cpu().contiguous() for k, p in m.named_parameters() if p.requires_grad}, str(snap))
        h = dict(step=state.global_step, tag=tag, ade=ade, n=n, metrics=met, seconds=round(time.time() - t, 1), snapshot=str(snap.relative_to(args.output_dir)))
        self.st["history"] = [x for x in self.st["history"] if x["step"] != h["step"]] + [h]; self._save(args, state)
        if state.is_world_process_zero: print("ft_select " + json.dumps(h), flush=True)
        if dist: torch.distributed.barrier()                                      # rank 0 wrote the snapshot

    def on_step_end(self, args, state, control, model=None, **kw):
        s = state.global_step
        if s % self.every == 0 or s >= state.max_steps:
            self._measure(args, state, model, "end" if s >= state.max_steps else "main")
            if self.stop is not None and self._stop_hit():
                control.should_training_stop = control.should_save = True       # checkpoint at the stop point, then stop
                if state.is_world_process_zero:
                    (Path(args.output_dir) / "ft_stop.json").write_text(json.dumps({"step": s, "stop_ref": self.stop_ref, "stop_rel_worse": self.stop, "stop_metric": self.stop_metric, "stop_base": self.stop_base,
                        "patience": self.patience, "history": self.st["history"]}, indent=1))
                log.warning("ft_select: stop switch at step %d: %s %s > %.0f %% above s2 final %d times in a row", s, self.stop_ref,
                            self.stop_metric, 100 * self.stop, self.patience)
        return control

    def _stop_hit(self):
        v = lambda x: x["metrics"][self.stop_ref][self.stop_metric] if self.stop_metric != "ade" else x["ade"][self.stop_ref]
        h = sorted(self.st["history"], key=lambda x: x["step"]); base = self.stop_base if self.stop_base is not None else v(h[0])
        last = [x for x in h if x["step"] > 0][-self.patience:]
        return len(last) == self.patience and all(v(x) > (1 + self.stop) * base for x in last)

    def finalize(self, model, output_dir):
        """load the chosen measurement's weights into model (train.py, before final/ is saved); ft_choice.json (rank 0)."""
        out = Path(output_dir); hist = json.loads((out / "ft_select.json").read_text())["history"]
        best, base = choose(hist, self.ref, self.rel)
        m = getattr(model, "module", model)
        sd = load_file(str(out / best["snapshot"]), device=str(next(m.parameters()).device))
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not unexpected and not set(sd) & set(missing), (unexpected, set(sd) & set(missing))
        sel = {"chosen_step": best["step"], "chosen_tag": best["tag"], "chosen_ade": best["ade"], "ref": self.ref,
               "ref_start": base, "limit": None if self.rel is None else round((1 + self.rel) * base, 5), "last_step": max(h["step"] for h in hist), "history": hist}
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            (out / "ft_choice.json").write_text(json.dumps(sel, indent=1))
        log.info("ft_select: final = step %d (%s), %s ADE %.4f (s2 final %.4f, limit %s)", best["step"], best["tag"], self.ref,
                 best["ade"][self.ref], base, sel["limit"])
        return sel
