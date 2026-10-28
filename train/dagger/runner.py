"""WoWmayo DAgger runner (train/dagger/README.md): one resident process group that keeps training a BF16 LoRA on the
frozen delivered model (QAD final/: NVFP4/FP8 fake quantization, the actor engine's numbers) and publishes it every few
minutes; actors pick up latest.json for their next episode.

Every optimizer step = `batch` samples over all ranks, split by configs/dagger.yaml (whole-batch shares, users 10/28):
  replay  dagger.replay: a MixtureDataset of R1 general, the R1 signal-stop pool, R2 A/B/D, start relabel, recovery, curve
          (dagger/pools.py; R1 sources without the rows whose GT future knows the light turns green), read in its fixed
          schedule order through StudentVLA.forward unchanged (R2 ft's per-row losses: GT rows fm + traj_ce, R1 general
          + lm, history dropout)
  DAgger  dagger.share of the batch: 4 Hz rows of the actors' drives with the shadow labels (dagger/spool.py), loss
          dagger/loss.py x the row weight of its tags (signal approach, lane departure); the oldest untrained rows first
          (every row is trained on), the rest drawn from the rows already trained on (the aggregate) when there are not
          enough new ones. No staleness gate (labels come from the expert): each row keeps the LoRA version it was driven
          with (metrics).
LoRA (--lora): expert (q k v o gate up down of every expert layer) or llm+expert (the same on every LLM layer too); the
camera and LiDAR ViTs, embeddings, LM head and action projections stay frozen. Files as train/rl (vNNNNN.safetensors +
latest.json, keys <module>.lora_A / lora_B = the delivered module names, metadata scale, version).
First update once --min_rows DAgger rows are in. Before every publish: fixed validation rows (R2 ft's val rows + the
first DAgger rows) for non-finite values and blow-ups (loss > val_max_ratio x version 0); a failed check publishes
nothing and alerts; --max_rejects in a row stop the run. dagger.val_sets (signal stop, lane) are recorded beside it, and
after each publish (and for version 0) the history dependence on dagger.history_sets (dagger/history.py, records only). Metrics in <out>/metrics/ (steps, events, val, spool,
summary), pushed to B2 by run.sh. Same command = resume (<out>/state.pt). Data parallel over torchrun ranks (NCCL, gloo
on CPU): every rank holds the whole model, LoRA gradients are all-reduced (mean) before each step.
Usage (cwd train/): torchrun --nproc_per_node N -m dagger.runner --base <deliver>/final --spool <dir> --out <dir> \
    --data_root <R1 step-0> --r2_root <R2 step-0> --e2e_server <eval repo src/e2e_driver/server> \
    --lr ... --rank ... --alpha ... --batch ... --publish_min ... --min_rows ...      (the starting values need the user)
"""
import argparse, json, logging, math, os, shlex, subprocess, sys, time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np, torch, torch.distributed as dist

from dagger import history, spool
from dagger.loss import checkpoint_llm, dagger_forward, row_seed
from rl.episodes import e2e, write_json
from rl.policy import LLM, TARGET, LoRA

SCOPES = {"expert": (TARGET,), "llm+expert": (LLM, TARGET)}
CONFIGS = Path(__file__).resolve().parents[1] / "configs"
log = logging.getLogger("dagger")

# ------------------------------------------------------------------ data workers (image decode, prompts)
_W = {}


def _init(replay, model_config, server):
    torch.set_num_threads(1); _W.update(replay=replay, cfg=model_config, server=server)


def _load(key):
    kind, x = key
    return _W["replay"][x] if kind == "replay" else spool.sample(x, _W["server"], _W["cfg"])


def to_dev(x, dev):
    if isinstance(x, torch.Tensor): return x.to(dev, non_blocking=True)
    if isinstance(x, dict): return {k: to_dev(v, dev) for k, v in x.items()}
    return x


def val_set(parts, model_config):
    """recorded validation rows: parts [{n, MoraiDataset kwargs}] (as FtSelect's sets), training prompts, augment off, the
    first n of MoraiDataset(max_samples n, seed 1)."""
    from morai_vla.data import MoraiDataset
    out = []
    for p in parts:
        p = dict(p); n = p.pop("n")
        ds = MoraiDataset(**p, max_samples=n, augment=False, seed=1, model_config=model_config,
                          vla_preprocess_args={"_target_": "morai_vla.processor.get_preprocess_fn"})
        if len(ds): out.append(ds)
    return torch.utils.data.ConcatDataset(out) if out else []


# ------------------------------------------------------------------ planner (rank 0): which DAgger rows each step trains
class Planner:
    def __init__(self, spool_dir, server, scan_s, seed, tags=spool.TAGS, weights=None):
        self.dir, self.wire, self.scan_s, self.seed, self.tags, self.weights = Path(spool_dir), e2e(server)[0], scan_s, seed, tags, weights
        self.rows, self.episodes, self.bad, self.fifo, self.t_scan, self.t_new = [], [], [], 0, 0.0, time.time()
        self.uses = []                                               # times each row was trained on (user 10/28: overfit check)

    def add(self, d, events):
        try:
            res = spool.episode_rows(d, self.wire, self.tags, self.weights)
        except Exception as e:                                       # another session's files: never stop the run on one
            self.bad.append(str(d)); events.append({"event": "bad_episode", "episode": str(d), "error": repr(e)[:300]}); return None
        if res is None: return None                                  # still syncing: next scan
        rows, st = res
        self.rows += rows; self.episodes.append(str(d)); self.uses += [0] * len(rows)
        if rows: self.t_new = time.time()
        return st

    def scan(self, events, force=False):
        if not force and time.time() - self.t_scan < self.scan_s: return []
        self.t_scan, known, out = time.time(), set(self.episodes) | set(self.bad), []
        for d in spool.episodes(self.dir):
            if str(d) not in known and (st := self.add(d, events)) is not None: out.append(st)
        return out

    def take(self, n, step):
        """n rows: the oldest untrained first, then uniform draws from the trained ones -> (rows, new count)."""
        old = self.fifo; new = list(range(old, min(old + n, len(self.rows)))); self.fifo += len(new)
        rng = np.random.default_rng([self.seed, step])
        fill = rng.integers(0, old, n - len(new)).tolist() if n > len(new) and old else []
        for i in new + fill: self.uses[i] += 1
        return [self.rows[i] for i in new + fill], len(new), new + fill

    def reuse(self, idx):
        """per-row training counts: of this step's rows, and over every row trained so far."""
        u = np.asarray(self.uses[:self.fifo]); s = np.asarray([self.uses[i] for i in idx])
        return {"uses_step_mean": round(float(s.mean()), 3) if len(s) else 0.0, "uses_step_max": int(s.max()) if len(s) else 0,
                "uses_mean": round(float(u.mean()), 3) if len(u) else 0.0, "uses_max": int(u.max()) if len(u) else 0,
                "rows_used_2plus": int((u >= 2).sum()), "rows_used_5plus": int((u >= 5).sum())}

    def state(self, fifo, pending=()):
        """as of the last executed step: the plans made ahead (pending) give their rows back."""
        uses = list(self.uses)
        for q in pending:
            for i in q["idx"]: uses[i] -= 1
        return {"episodes": self.episodes, "bad": self.bad, "fifo": fifo, "uses": uses}

    def restore(self, st):
        ev = []
        for d in st["episodes"]:
            assert self.add(Path(d), ev) is not None, f"resume: {d} no longer readable ({ev})"
        self.bad, self.fifo = list(st["bad"]), st["fifo"]
        u = list(st.get("uses", [])); self.uses = (u + [0] * len(self.rows))[:len(self.rows)]


# ------------------------------------------------------------------ run
def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="delivered QAD final/ (frozen, fake-quantized): R2 delivery or B2 runs/stageC/deliver/final")
    p.add_argument("--spool", required=True); p.add_argument("--out", required=True)
    p.add_argument("--data_root", required=True, help="R1 step-0 root (/data/step0)"); p.add_argument("--r2_root", required=True)
    p.add_argument("--e2e_server", default=os.environ.get("E2E_SERVER"), required=not os.environ.get("E2E_SERVER"))
    p.add_argument("--override", action="append", default=[], help="Hydra override of configs/dagger.yaml (repeatable)")
    p.add_argument("--lora", choices=SCOPES, default="expert", help="user 10/28: expert (llm+expert on hold, kept working)")
    # starting values: proposals in README, the user approves them (no defaults on purpose)
    p.add_argument("--lr", type=float, required=True); p.add_argument("--rank", type=int, required=True)
    p.add_argument("--alpha", type=float, required=True); p.add_argument("--batch", type=int, required=True, help="samples per optimizer step, all ranks")
    p.add_argument("--publish_min", type=float, required=True); p.add_argument("--min_rows", type=int, required=True)
    p.add_argument("--warmup", type=int, default=20, help="linear LR warm-up steps, then constant")
    p.add_argument("--micro", type=int, default=1, help="samples per forward (1 = R2 ft; > 1 needs --attn sdpa)")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--device", default="cuda"); p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn", default="sdpa", help="sdpa | flash_attention_2 (H100, micro 1)")
    p.add_argument("--grad_ckpt", action="store_true", help="checkpoint the LLM layers (llm+expert on small GPUs)")
    p.add_argument("--compress", action="store_true", help="ModelOpt mtq.compress the frozen NVFP4/FP8 weights (13.3 -> 5.4 GiB; "
                   "NVFP4 exact, FP8 through a real FP8 GEMM, ~2 %% max output change): 16 GB cards only, not the training box")
    p.add_argument("--workers", type=int, default=6, help="data processes per rank")
    p.add_argument("--val_n", type=int, default=64, help="R2 ft val rows in the publish check"); p.add_argument("--val_dagger", type=int, default=64)
    p.add_argument("--val_max_ratio", type=float, default=1.5); p.add_argument("--max_rejects", type=int, default=3)
    p.add_argument("--scan_s", type=float, default=30.0); p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--stale_stop_min", type=float, default=30.0, help="stop when no new DAgger row arrived for this long (0 = never)")
    p.add_argument("--max_steps", type=int, default=0, help="stop after this many steps of this process (0 = run on)")
    p.add_argument("--publish_at_end", action="store_true", help="checked publish when stopping (dry run)")
    p.add_argument("--b2", help="b2://bucket/prefix: each published version uploaded there (file first, then latest.json)")
    p.add_argument("--alert_cmd", help="shell command run with the alert text as $1 (e.g. a Discord DM)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


class Runner:
    def __init__(self, a):
        self.a = a
        self.world = int(os.environ.get("WORLD_SIZE", 1)); self.rank = int(os.environ.get("RANK", 0))
        cuda = a.device == "cuda"
        if self.world > 1: dist.init_process_group("nccl" if cuda else "gloo")
        self.dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if cuda else torch.device("cpu")
        if cuda: torch.cuda.set_device(self.dev)
        from omegaconf import OmegaConf
        cfg = self.config(); dg = cfg.dagger
        shares = {k: float(v.share) for k, v in dg.replay.sources.items()}
        assert 0 < dg.share <= 1 and abs(sum(shares.values()) + dg.share - 1) < 1e-6, f"dagger.share {dg.share} + replay shares {shares} != 1"
        self.n_dag = round(a.batch * dg.share); self.n_rep = a.batch - self.n_dag
        assert abs(self.n_dag - a.batch * dg.share) < 1e-6 and self.n_dag % (self.world * a.micro) == 0 and self.n_rep % (self.world * a.micro) == 0, \
            f"batch {a.batch} x share {dg.share}: DAgger and replay counts must be multiples of ranks x micro ({self.world} x {a.micro})"
        assert a.min_rows >= self.n_dag, "the first step needs a full DAgger part of new rows"
        self.per_r, self.per_d = self.n_rep // self.world, self.n_dag // self.world   # samples per rank per step
        self.out = Path(a.out); self.met = self.out / "metrics"
        if self.rank == 0: (self.out / "lora").mkdir(parents=True, exist_ok=True); self.met.mkdir(exist_ok=True)
        torch.manual_seed(a.seed + self.rank)                        # fm noise differs per rank; LoRA init comes from rank 0

        from morai_vla.student import StudentVLA
        m = StudentVLA.build(init=str(a.base), attn_implementation=a.attn, dtype=a.dtype, device_map={"": self.dev.index or 0} if cuda else None)
        m.to(self.dev).requires_grad_(False)
        if not cuda:                                                 # ModelOpt NVFP4 fake quant is CUDA only (as rl.tiny_test)
            from rl.tiny_test import quant_off
            quant_off(m)
        m.loss_weights.update(OmegaConf.to_container(cfg.model.loss_weights))           # R2 ft's (no T*: kv vit vkd 0)
        if a.grad_ckpt: checkpoint_llm(m)
        if a.compress:                                               # 16 GB cards only (5080 measurement, README)
            import modelopt.torch.quantization as mtq
            mtq.compress(m); torch.cuda.empty_cache()
        self.m = m
        self.lora = LoRA(m, a.rank, a.alpha, SCOPES[a.lora]).to(self.dev)
        self.params = list(self.lora.parameters())
        self.opt = torch.optim.AdamW(self.params, lr=a.lr, weight_decay=0.0)

        import hydra.utils as hyu
        spec = OmegaConf.to_container(dg.replay, resolve=True)
        spec["sources"] = {k: v for k, v in spec["sources"].items() if v["share"] > 0}     # share 0: not loaded
        if spec["sources"] and not any(v.get("new") for v in spec["sources"].values()):   # MixtureDataset counts its length on
            for v in spec["sources"].values(): v["new"] = True                           # "new" sources; the runner wraps anyway
        self.replay = hyu.instantiate(spec, _convert_="partial", model_config=m.config) if self.n_rep else None
        self.shares = {"dagger": float(dg.share), **{k: v["share"] for k, v in spec["sources"].items()}}
        self.val = hyu.instantiate(cfg.data.val_dataset, _convert_="partial", model_config=m.config, max_samples=a.val_n)
        self.val_sets = {name: val_set(parts, m.config) for name, parts in OmegaConf.to_container(dg.val_sets, resolve=True).items()}
        self.hist_sets, self.hist_steps = history.sets(OmegaConf.to_container(dg.history_sets, resolve=True)), int(dg.history_steps)
        self.pool = ProcessPoolExecutor(a.workers, initializer=_init, initargs=(self.replay, m.config, a.e2e_server))
        self.planner = Planner(a.spool, a.e2e_server, a.scan_s, a.seed, dict(dg.tags), dict(dg.weights)) if self.rank == 0 else None
        self.step = self.version = self.replay_pos = self.rejects = 0
        self.val0 = self.val_rows = None; self.upload = None
        self.events = []

    def config(self):
        from hydra import compose, initialize_config_dir
        over = [f"data.root={self.a.data_root}", f"paths.r2_root={self.a.r2_root}", "paths.s2=unused", *self.a.override]
        with initialize_config_dir(str(CONFIGS), version_base=None):
            return compose("dagger", overrides=over)

    # -------------------------------------------------------------- collectives and records
    def bcast(self, obj):
        if self.world == 1: return obj
        o = [obj]; dist.broadcast_object_list(o, 0); return o[0]

    def gather(self, obj):
        if self.world == 1: return [obj]
        o = [None] * self.world; dist.all_gather_object(o, obj); return o

    def record(self, name, row):
        if self.rank == 0:
            with open(self.met / f"{name}.jsonl", "a") as f: f.write(json.dumps(row) + "\n")

    def event(self, kind, **kw):
        row = {"t": round(time.time(), 1), "step": self.step, "version": self.version, "event": kind, **kw}
        self.record("events", row)
        if self.rank == 0: log.info("event %s", json.dumps(row))

    def alert(self, text):
        self.event("alert", text=text)
        if self.rank == 0 and self.a.alert_cmd:
            subprocess.run(["bash", "-c", self.a.alert_cmd, "_", text], timeout=60, check=False)

    # -------------------------------------------------------------- state
    def save(self, fifo, replay_pos):
        sums = self.gather(float(sum(q.detach().double().sum() for q in self.params)))
        assert len(set(sums)) == 1, f"LoRA differs between ranks: {sums}"            # data-parallel sync check, every save
        if self.rank != 0: return
        tmp = self.out / "state.pt.tmp"
        torch.save({"lora": self.lora.state_dict(), "opt": self.opt.state_dict(), "step": self.step, "version": self.version,
                    "replay_pos": replay_pos, "planner": self.planner.state(fifo, getattr(self, "pending", ())), "val0": self.val0, "val_rows": self.val_rows,
                    "args": vars(self.a)}, tmp)
        tmp.replace(self.out / "state.pt")

    def resume(self):
        f = self.out / "state.pt"
        if not f.exists(): return False
        s = torch.load(f, map_location=self.dev, weights_only=False)
        for k in ("lora", "rank", "alpha"):
            assert s["args"][k] == vars(self.a)[k], f"resume with --{k} {s['args'][k]} (state.pt), not {vars(self.a)[k]}"
        self.lora.load_state_dict(s["lora"]); self.opt.load_state_dict(s["opt"])
        self.step, self.version, self.replay_pos, self.val0, self.val_rows = s["step"], s["version"], s["replay_pos"], s["val0"], s["val_rows"]
        if self.rank == 0: self.planner.restore(s["planner"])
        return True

    # -------------------------------------------------------------- one step
    def plan(self):
        """rank 0 -> every rank: the next step's DAgger rows (global) and replay positions, or None (too few rows yet)."""
        p = None
        if self.rank == 0:
            ev = []
            for st in self.planner.scan(ev):
                self.record("spool", st)
            for e in ev: self.event(e.pop("event"), **e)
            pl = self.planner
            if self.step > 0 or len(pl.rows) >= self.a.min_rows:
                first = pl.rows[:self.a.val_dagger] if self.val_rows is None else None    # fixed for the whole run
                rows, new, idx = pl.take(self.n_dag, self.step + len(self.pending))
                p = {"dagger": rows, "new": new, "idx": idx, "fifo": pl.fifo, "backlog": len(pl.rows) - pl.fifo, "rows_total": len(pl.rows), **pl.reuse(idx),
                     "episodes": len(pl.episodes), "stale_s": round(time.time() - pl.t_new, 1), "val_rows": first}
        p = self.bcast(p)
        if p is None: return None
        first = p.pop("val_rows")
        if first is not None: self.val_rows = first
        p["replay"] = [(self.replay_pos + k) % len(self.replay) for k in range(self.n_rep)]
        self.replay_pos += self.n_rep
        p["replay_pos"] = self.replay_pos
        mine = lambda xs, n: xs[self.rank * n:(self.rank + 1) * n]
        keys = [("replay", i) for i in mine(p["replay"], self.per_r)] + [("dagger", r) for r in mine(p["dagger"], self.per_d)]
        p["futures"] = [self.pool.submit(_load, k) for k in keys]
        return p

    def lr(self):
        return self.a.lr * min(1.0, (self.step + 1) / max(self.a.warmup, 1))

    def train_step(self, p):
        from morai_vla.processor import collate_fn
        t0, a, m, lr = time.time(), self.a, self.m, self.lr()
        for g in self.opt.param_groups: g["lr"] = lr
        m.train(); m.loss_sums.clear()
        items, wait = [], 0.0
        for f in p["futures"]:
            tw = time.time(); items.append(f.result()); wait += time.time() - tw
        rep, dag = items[:self.per_r], items[self.per_r:]
        n = self.per_r + self.per_d
        dsum = {}
        for i in range(0, max(self.per_r, self.per_d), a.micro):
            if rep[i:i + a.micro]:
                b = to_dev(collate_fn(rep[i:i + a.micro], model_config=m.config), self.dev)
                (m(**b).loss * len(rep[i:i + a.micro]) / n).backward()
            if dag[i:i + a.micro]:
                b = to_dev(collate_fn(dag[i:i + a.micro], model_config=m.config), self.dev)
                loss, per = dagger_forward(m, b)
                (loss * len(dag[i:i + a.micro]) / n).backward()
                for s, tg, fm, cs, cn in zip(b["label_source"], b["tag"], per["fm"].tolist(), per["ce_sum"].tolist(), per["ce_n"].tolist()):
                    names = ["dagger", f"dagger_{spool.SOURCES[int(s)]}"] + ["dagger_signal"] * (int(tg) & 1) + ["dagger_lane"] * (int(tg) >> 1)
                    for name in names:
                        d = dsum.setdefault(name, [0.0, 0, 0.0, 0.0]); d[0] += fm; d[1] += 1; d[2] += cs; d[3] += cn
        flat = torch.cat([(q.grad if q.grad is not None else torch.zeros_like(q)).flatten() for q in self.params])
        if self.world > 1: dist.all_reduce(flat); flat /= self.world
        o = 0
        for q in self.params:
            q.grad = flat[o:o + q.numel()].view_as(q).clone(); o += q.numel()
        gnorm = float(torch.nn.utils.clip_grad_norm_(self.params, a.max_grad_norm))
        self.opt.step(); self.opt.zero_grad(set_to_none=True)
        self.step += 1
        rs = {k: [float(s), c] for k, (s, c) in m.loss_sums.items()}; m.loss_sums.clear()
        tot, dt = {}, time.time() - t0
        for r, d in self.gather((rs, dsum)):
            for k, (s, c) in r.items(): x = tot.setdefault(f"replay/{k}", [0.0, 0]); x[0] += s; x[1] += c
            for k, (fm, c, cs, cn) in d.items():
                for name, s, cc in ((f"{k}/fm", fm, c), (f"{k}/traj_ce", cs, cn)): x = tot.setdefault(name, [0.0, 0]); x[0] += s; x[1] += cc
        row = {"step": self.step, "version": self.version, "lr": lr, "grad_norm": round(gnorm, 5),
               **{k: round(s / max(c, 1), 5) for k, (s, c) in sorted(tot.items())},
               "step_s": round(dt, 3), "samples_per_s": round(a.batch / dt, 3), "dagger_rows_per_s": round(self.n_dag / dt, 3),
               "data_wait_s": round(wait, 3), "new_rows": p["new"], "reused_rows": self.n_dag - p["new"],
               "tag_signal_rows": sum(x["signal"] for x in p["dagger"]), "tag_lane_rows": sum(x["lane"] for x in p["dagger"]),
               **{k: p[k] for k in ("uses_step_mean", "uses_step_max", "uses_mean", "uses_max", "rows_used_2plus", "rows_used_5plus")},
               "consumed_rows": p["fifo"], "backlog": p["backlog"], "rows_total": p["rows_total"], "episodes": p["episodes"],
               "lag_versions": [self.version - v for v in sorted({x["version"] for x in p["dagger"]})][:8], "replay_pos": p["replay_pos"]}
        if self.dev.type == "cuda":
            row["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(self.dev) / 2**30, 2); torch.cuda.reset_peak_memory_stats(self.dev)
        self.record("steps", row)
        if self.rank == 0 and (self.step <= 3 or self.step % 10 == 0): log.info("step %s", json.dumps(row))
        return row

    # -------------------------------------------------------------- validation and publish
    @torch.no_grad()
    def validate(self):
        """fixed rows (R2 ft val via StudentVLA.forward in eval mode: noise from the sample keys; the run's first DAgger rows
        with seeded noise), sharded over ranks -> {name: mean}."""
        from morai_vla.processor import collate_fn
        m = self.m; m.eval(); m.loss_sums.clear()
        sums = {}
        for i in range(self.rank, len(self.val), self.world):
            m(**to_dev(collate_fn([self.val[i]], model_config=m.config), self.dev))
        for k, (s, c) in m.loss_sums.items(): sums[f"val/{k}"] = [float(s), c]
        m.loss_sums.clear(); m.eval_sums.clear()
        for name, ds in self.val_sets.items():                        # recorded only (check() gates val/ and val_dagger/)
            for i in range(self.rank, len(ds), self.world):
                m(**to_dev(collate_fn([ds[i]], model_config=m.config), self.dev))
            for k, (s, c) in m.loss_sums.items(): sums[f"val_{name}/{k}"] = [float(s), c]
            m.loss_sums.clear(); m.eval_sums.clear()
        rows = self.val_rows[self.rank::self.world]
        for i in range(0, len(rows), self.a.micro):
            items = list(self.pool.map(_load, [("dagger", r) for r in rows[i:i + self.a.micro]]))
            b = to_dev(collate_fn(items, model_config=m.config), self.dev)
            _, per = dagger_forward(m, b, seed=row_seed([f"{r['ep']}/{r['j']}" for r in rows[i:i + self.a.micro]]))
            for name, v, c in (("val_dagger/fm", per["fm"].sum(), len(items)), ("val_dagger/traj_ce", per["ce_sum"].sum(), per["ce_n"].sum())):
                x = sums.setdefault(name, [0.0, 0]); x[0] += float(v); x[1] += float(c)
        tot = {}
        for part in self.gather(sums):
            for k, (s, c) in part.items(): x = tot.setdefault(k, [0.0, 0]); x[0] += s; x[1] += c
        return {k: s / c if c else float("nan") for k, (s, c) in sorted(tot.items())}

    def check(self, v):
        """-> list of failures: non-finite values or LoRA, R2 ft val fm / traj_ce above val_max_ratio x version 0."""
        bad = [k for k, x in v.items() if not math.isfinite(x)]
        if not all(torch.isfinite(q).all() for q in self.params): bad.append("lora weights")
        for k in ("val/fm", "val/traj_ce"):
            if k in v and k in self.val0 and v[k] > self.a.val_max_ratio * self.val0[k]:
                bad.append(f"{k} {v[k]:.4f} > {self.a.val_max_ratio} x {self.val0[k]:.4f}")
        return bad

    def publish(self, fifo, replay_pos):
        t0 = time.time(); v = self.validate()
        bad = self.check(v) if self.rank == 0 else None
        bad = self.bcast(bad)
        self.record("val", {"step": self.step, "candidate": self.version + 1, **{k: round(x, 5) for k, x in v.items()}, "failed": bad,
                            "seconds": round(time.time() - t0, 1)})
        if bad:
            self.rejects += 1
            self.alert(f"DAgger {self.out.name}: version {self.version + 1} not published at step {self.step}: {'; '.join(bad)} "
                       f"({self.rejects} in a row)")
            return False
        self.rejects = 0; self.version += 1
        if self.rank == 0: self.write_lora()
        self.save(fifo, replay_pos)
        self.event("publish", seconds=round(time.time() - t0, 1), val={k: round(x, 5) for k, x in v.items()})
        self.history()                                                 # after the files are out: never delays the actors
        return True

    def history(self):
        """history dependence of the current version on the fixed rows -> metrics/history.jsonl (per set) and
        history_rows.jsonl (per row)."""
        if not self.hist_sets: return
        t0 = time.time()
        rows = [r for part in self.gather(history.measure(self.m, self.lora, self.hist_sets, self.hist_steps, self.dev, self.rank, self.world))
                for r in part]
        for r in rows: self.record("history_rows", {"version": self.version, **r})
        self.record("history", {"version": self.version, "step": self.step, "seconds": round(time.time() - t0, 1), "sets": history.summary(rows)})

    def write_lora(self):
        from rl.learner import publish
        publish(SimpleNamespace(lora=self.lora), self.out, self.version)
        if not self.a.b2: return
        if self.upload is not None and self.upload.poll() is None: self.upload.wait()
        if self.upload is not None and self.upload.returncode: self.alert(f"B2 upload of a LoRA version failed (rc {self.upload.returncode})")
        bucket, _, prefix = self.a.b2.removeprefix("b2://").partition("/")
        up = lambda f: f"b2 file upload --no-progress {shlex.quote(bucket)} {shlex.quote(str(self.out / 'lora' / f))} {shlex.quote(prefix.rstrip('/') + '/lora/' + f)}"
        self.upload = subprocess.Popen(["bash", "-c", f"{up(f'v{self.version:05d}.safetensors')} && {up('latest.json')}"],
                                       stdout=subprocess.DEVNULL, stderr=open(self.met / "b2_upload.log", "a"))

    # -------------------------------------------------------------- loop
    def run(self):
        a = self.a; self.pending = deque()
        if self.resume():
            self.event("resume", replay_pos=self.replay_pos, rows=len(self.planner.rows) if self.planner else None)
        else:
            if self.world > 1:                                         # one LoRA init for every rank
                for q in self.params: dist.broadcast(q.data, 0)
            self.val_rows = []
            self.val0 = self.validate()                                # version 0 = the delivered model (no DAgger rows yet)
            self.val_rows = None
            self.event("start", val0=self.val0, args=vars(a), world=self.world)
            if self.rank == 0: self.write_lora()                       # v00000 = zero LoRA = the delivery
            self.save(0, 0)
            self.history()                                             # the reference: the starting delivery
        t_pub, done, last, waiting = time.time(), 0, None, False
        while True:
            while len(self.pending) < 2:
                p = self.plan()
                if p is None: break
                self.pending.append(p)
            if not self.pending:                                       # before the first update: too few DAgger rows
                if not waiting: self.event("waiting_rows", min_rows=a.min_rows); waiting = True
                stale = self.bcast(self.rank == 0 and bool(a.stale_stop_min) and time.time() - self.planner.t_new > a.stale_stop_min * 60)
                if stale: self.alert(f"DAgger {self.out.name}: no DAgger rows for {a.stale_stop_min} min, stopping"); break
                time.sleep(5); continue
            if self.step == 0: self.event("first_update", rows=self.pending[0]["rows_total"])
            waiting = False
            p = self.pending.popleft(); last = self.train_step(p); done += 1
            fifo, rpos = p["fifo"], p["replay_pos"]                     # state as of this step (the next plan already ran)
            ctl = None
            if self.rank == 0:                                         # stale: nothing new for a while and nothing left to train
                stale = bool(a.stale_stop_min) and p["backlog"] == 0 and p["stale_s"] > a.stale_stop_min * 60
                ctl = {"publish": time.time() - t_pub >= a.publish_min * 60, "stale": stale,
                       "stop": bool(a.max_steps and done >= a.max_steps) or (self.out / "STOP").exists() or stale}
            ctl = self.bcast(ctl)
            if ctl["stale"]: self.alert(f"DAgger {self.out.name}: no new DAgger row for {a.stale_stop_min} min and none left, stopping")
            if ctl["publish"] or (ctl["stop"] and a.publish_at_end):
                self.publish(fifo, rpos); t_pub = time.time()
                if self.rejects >= a.max_rejects:
                    self.alert(f"DAgger {self.out.name}: {self.rejects} rejected publishes in a row, stopping"); self.save(fifo, rpos); return 3
            elif self.step % a.save_every == 0:
                self.save(fifo, rpos)
            if ctl["stop"]:
                self.save(fifo, rpos); break
        self.summary()
        return 0

    def summary(self):
        if self.rank != 0 or not (self.met / "steps.jsonl").exists(): return
        rows = [json.loads(x) for x in open(self.met / "steps.jsonl")]
        steady = rows[3:] or rows
        med = lambda k: float(np.median([r[k] for r in steady if k in r])) if any(k in r for r in steady) else None
        s = {"steps": len(rows), "world": self.world, "lora": self.a.lora, "batch": self.a.batch, "micro": self.a.micro, "shares": self.shares,
             "grad_ckpt": self.a.grad_ckpt, "step_s_median": med("step_s"), "samples_per_s": med("samples_per_s"),
             "dagger_rows_per_s": med("dagger_rows_per_s"), "data_wait_s_median": med("data_wait_s"),
             "peak_mem_gb_max": max((r.get("peak_mem_gb", 0) for r in rows), default=None), "version": self.version,
             "lora_mb": round(sum(q.numel() for q in self.params) * 2 / 2**20, 1),
             # one actor (real time, 5090, RTF ~0.84) makes 4 x 0.84 = 3.4 DAgger rows per wall second; user 10/28: size the
             # actors on the 8-GPU, expert-LoRA dry run (dagger/dryrun.sh defaults), no cap
             "actors_at_balance": round(med("dagger_rows_per_s") / 3.36, 2) if med("dagger_rows_per_s") else None,
             "actors_basis": f"{self.world} ranks, lora {self.a.lora}, batch {self.a.batch}, DAgger share {self.shares['dagger']}",
             "val_sets": {k: len(v) for k, v in self.val_sets.items()}, "history_sets": {k: len(v) for k, v in self.hist_sets.items()}}
        write_json(self.met / "summary.json", s); log.info("summary %s", json.dumps(s))

    def close(self):
        self.pool.shutdown(cancel_futures=True)
        if self.upload is not None: self.upload.wait()
        if self.world > 1: dist.destroy_process_group()


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    r = Runner(parse(argv))
    try:
        rc = r.run()
    finally:
        r.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
