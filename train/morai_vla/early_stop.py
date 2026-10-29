"""Trajectory-ADE early stop (user 10/25; the numbers are the user's, do not change them):
  1. the planned epochs are the cap;
  2. every `every_steps` (~20 min) the expert's continuous trajectories are scored on the fixed 1,000 val + val_asset samples
     (check_release / eval_stage1 / t1t selection set): ADE over 6.4 s, mean of K samples, LiDAR on, per-sample seed 0;
  3. no stop in the first 0.5 epoch; after it, a measurement not more than 2 % below the best so far is "no improvement",
     and 3 in a row stop the main run;
  4. from the stop point the LR goes linearly to 0 in `decay_steps` (same process, the scheduler's lambda is replaced);
  5. final/ = the lowest-ADE measurement (the best so far or the one after the decay); <output_dir>/ade_select.json;
  6. every measurement keeps its own copy of the trainable weights (<output_dir>/ade_snapshots/step-N.safetensors),
     outside HF's checkpoint rotation.
Scoring is sharded over the training ranks (each rank holds full weights under DDP and ZeRO-2) and gathered. State lives in
<output_dir>/ade_early_stop.json, so a resumed run continues the count (measurements past the resume step are dropped).
Shared by Stage 2 (configs/stage2.yaml) and the student's stage 1: a model only needs TeacherVLA.sample_trajectories;
override TrajEarlyStop.score for another trajectory source. train.py calls finalize() before saving final/."""
import json, logging, math, time
from pathlib import Path

import numpy as np, torch
from safetensors.torch import load_file, save_file
from transformers.trainer_callback import TrainerCallback

from morai_vla.data import DT, MoraiDataset
from morai_vla.processor import collate_fn, get_preprocess_fn

log = logging.getLogger(__name__)


class TrajSamples(torch.utils.data.Dataset):
    """rows k::world of the first n of check_release's set; items are collate-ready generation prompts (LiDAR on)."""
    def __init__(self, root, model_config, n=1000, rank=0, world=1):
        self.ds = MoraiDataset(root, ["val", "val_asset"], frame_stride=10, max_samples=n, augment=False, seed=1)
        self.ds.rows = self.ds.rows.iloc[:n].iloc[rank::world].reset_index(drop=True)
        self.fn = get_preprocess_fn(model_config, generation_mode=True)

    def __len__(self):
        return len(self.ds.rows)

    def __getitem__(self, k):
        d = dict(self.ds.sample(k)); d["cot"] = None; d["tokenized_data"] = self.fn(data=d)       # as check_release "traj"
        return {key: v for key, v in d.items() if key not in ("camera_frames", "bev_frames")}


def brake_drops(hist_xy, pred_xy, gt_xy, window=1.0):
    """braking onset of one row (reports/lwf_dynstop_recipe_20261029.md section 6): v0 = the last history step / DT (history[-1]
    is the origin, points DT apart); a path's drop = v0 - its lowest speed over the first `window` s (speed of each DT step,
    the first one from the origin). hist_xy (16, 2), pred_xy (K, 64, 2), gt_xy (64, 2) -> {v0, pred_drop, gt_drop}.
    pred_drop = the mean of the K per-sample drops (= v0 - the mean of the per-sample minima): a braking share of the samples
    shows as a partial drop instead of one outlier sample deciding it."""
    n, v0 = round(window / DT), float((hist_xy[-1] - hist_xy[-2]).norm()) / DT
    low = lambda xy: (torch.diff(xy[..., :n, :], dim=-2, prepend=torch.zeros_like(xy[..., :1, :])).norm(dim=-1) / DT).amin(-1)
    return {"v0": v0, "pred_drop": v0 - float(low(pred_xy).mean()), "gt_drop": v0 - float(low(gt_xy))}


def expert_ade(model, samples, num_samples=6, num_workers=4, detail=False, batch=1, brake_s=1.0):
    """[(key, ADE over 6.4 s averaged over num_samples)]: check_release's traj_ade per row. detail: [(key, {ade, fde, lat,
    lon, v0, pred_drop, gt_drop})], lat / lon = mean |error| across / along the GT path (GT heading per point,
    scripts/lateral_error.split_error); v0 / pred_drop / gt_drop = brake_drops over the first brake_s s.
    batch > 1: rows share one prefill and one Euler loop (left padded); each row starts from the noise it draws alone
    (seed 0), so only the padding can move the numbers (README "R2 속도 최적화": checked against batch 1 on a GPU)."""
    m = getattr(model, "module", model); was_training = m.training; m.eval(); out = []
    device = next(m.parameters()).device                                          # the training GPU (CPU in the tiny CPU check)
    dev = lambda x: {k: dev(v) if isinstance(v, dict) else v.to(device) if torch.is_tensor(v) else v for k, v in x.items()}
    try:
        for items in torch.utils.data.DataLoader(samples, batch_size=batch, num_workers=num_workers, collate_fn=lambda s: s):
            b = dev(collate_fn(items, model_config=m.config)); torch.manual_seed(0)
            x0 = None if batch == 1 else torch.randn(num_samples, *m.action_space.get_action_space_dims(), device=device).repeat(len(items), 1, 1)
            xyz, _ = m.sample_trajectories(b["tokenized_data"], b["ego_history_xyz"], b["ego_history_rot"], num_samples=num_samples, noise=x0)
            for j, item in enumerate(items):
                d = xyz[j, ..., :2].float() - b["ego_future_xyz"][j, 0, :, :2].float(); err = d.norm(dim=-1)   # (K, 64)
                if not detail:
                    out.append((item["key"], float(err.mean()))); continue
                r = b["ego_future_rot"][j, 0].float(); c, s = r[:, 0, 0], r[:, 1, 0]                        # GT heading cos, sin (64,)
                out.append((item["key"], {"ade": float(err.mean()), "fde": float(err[:, -1].mean()),
                                          "lat": float((c * d[..., 1] - s * d[..., 0]).abs().mean()), "lon": float((c * d[..., 0] + s * d[..., 1]).abs().mean()),
                                          **brake_drops(b["ego_history_xyz"][j, 0, :, :2].float(), xyz[j, ..., :2].float(),
                                                        b["ego_future_xyz"][j, 0, :, :2].float(), brake_s)}))
    finally:
        m.train(was_training)
    return out


def stop_now(history, min_step, rel=0.02, patience=3):
    """rule 3 on the main-run measurements [(step, ade)] in order: True once `patience` in a row past min_step are not more
    than rel below the best before them."""
    streak = 0
    for j, (step, ade) in enumerate(history):
        if step < min_step or j == 0: continue
        streak = streak + 1 if ade > (1 - rel) * min(a for _, a in history[:j]) else 0
        if streak >= patience: return True
    return False


class TrajEarlyStop(TrainerCallback):
    def __init__(self, root, every_steps, decay_steps, n=1000, min_epochs=0.5, rel_improve=0.02, patience=3, num_samples=6,
                 num_workers=4):
        self.root, self.every, self.decay, self.n = root, every_steps, decay_steps, n
        self.min_epochs, self.rel, self.patience, self.k, self.workers = min_epochs, rel_improve, patience, num_samples, num_workers
        self.samples = self.st = None; self.patched = False

    # ------------------------------------------------------------------ state
    def _file(self, args):
        return Path(args.output_dir) / "ade_early_stop.json"

    def _save_state(self, args, state):
        if state.is_world_process_zero: self._file(args).write_text(json.dumps(self.st, indent=1))

    def on_train_begin(self, args, state, control, train_dataloader=None, **kw):
        f = self._file(args)
        self.st = json.loads(f.read_text()) if f.exists() else {"history": [], "stop_step": None, "decay_end": None, "lr_factor": None}
        s = state.global_step                                                     # > 0 when resumed
        self.st["history"] = [h for h in self.st["history"] if h["step"] <= s]
        if self.st["stop_step"] is not None and self.st["stop_step"] > s: self.st.update(stop_step=None, decay_end=None, lr_factor=None)
        per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        self.st.update(min_step=math.ceil(self.min_epochs * per_epoch), steps_per_epoch=per_epoch, max_steps=state.max_steps,
                       every_steps=self.every, decay_steps=self.decay, rule=dict(rel_improve=self.rel, patience=self.patience, min_epochs=self.min_epochs))
        self.patched = False; self._save_state(args, state)

    # ------------------------------------------------------------------ measurement
    def score(self, model, rank, world):
        """per-row (key, ADE) of this rank's shard."""
        m = getattr(model, "module", model)
        if self.samples is None: self.samples = TrajSamples(self.root, m.config, self.n, rank, world)
        return expert_ade(m, self.samples, self.k, self.workers)

    def _measure(self, args, state, model, tag):
        dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank, world = (torch.distributed.get_rank(), torch.distributed.get_world_size()) if dist else (0, 1)
        t = time.time(); rows = self.score(model, rank, world)
        if dist:
            parts = [None] * world; torch.distributed.all_gather_object(parts, rows); rows = [r for p in parts for r in p]
        m, snap = getattr(model, "module", model), Path(args.output_dir) / "ade_snapshots" / f"step-{state.global_step}.safetensors"
        if state.is_world_process_zero:
            snap.parent.mkdir(parents=True, exist_ok=True)
            save_file({n: p.detach().cpu().contiguous() for n, p in m.named_parameters() if p.requires_grad}, str(snap))
        h = dict(step=state.global_step, ade=round(float(np.mean([a for _, a in rows])), 5), n=len(rows), tag=tag,
                 seconds=round(time.time() - t, 1), snapshot=str(snap.relative_to(args.output_dir)))
        self.st["history"].append(h); self._save_state(args, state)
        if state.is_world_process_zero: print("traj_early_stop " + json.dumps(h), flush=True)
        if dist: torch.distributed.barrier()                                      # rank 0 wrote the snapshot

    def _patch(self, lr_scheduler):
        """rule 4: lambda(step) = f * (S + D - step) / D from the stop step S, f = the scheduled factor there."""
        S, E, f = self.st["stop_step"], self.st["decay_end"], self.st["lr_factor"]
        assert isinstance(lr_scheduler, torch.optim.lr_scheduler.LambdaLR), type(lr_scheduler)
        lam = lambda step: f * max(0.0, (E - step) / max(E - S, 1))
        lr_scheduler.lr_lambdas = [lam] * len(lr_scheduler.lr_lambdas); self.patched = True

    def on_step_begin(self, args, state, control, lr_scheduler=None, **kw):
        if self.st["stop_step"] is not None and not self.patched: self._patch(lr_scheduler)      # resumed inside the decay

    def on_step_end(self, args, state, control, model=None, lr_scheduler=None, **kw):
        s, st = state.global_step, self.st
        if st["stop_step"] is not None:                                           # decaying
            if s >= st["decay_end"]:
                self._measure(args, state, model, "after_decay"); control.should_training_stop = True
            return
        if s % self.every == 0:
            self._measure(args, state, model, "main")
            main = [(h["step"], h["ade"]) for h in st["history"] if h["tag"] == "main"]
            if stop_now(main, st["min_step"], self.rel, self.patience):
                st.update(stop_step=s, decay_end=min(s + self.decay, state.max_steps), lr_factor=lr_scheduler.lr_lambdas[0](s))
                self._save_state(args, state); self._patch(lr_scheduler)
                log.info("traj_early_stop: stop at %d, LR factor %.4g -> 0 by step %d", s, st["lr_factor"], st["decay_end"])
                if st["decay_end"] == s: control.should_training_stop = True
        elif s >= state.max_steps:                                                # rule 1: the planned epochs ended first
            self._measure(args, state, model, "end")

    # ------------------------------------------------------------------ selection (train.py, before final/ is saved)
    def finalize(self, model, output_dir):
        """rule 5: load the lowest-ADE measurement's weights into model; writes ade_select.json (rank 0)."""
        out = Path(output_dir); hist = json.loads((out / "ade_early_stop.json").read_text())["history"]
        if not hist: return None                                                  # no measurement (short runs): weights as trained
        best = min(hist, key=lambda h: h["ade"])
        m = getattr(model, "module", model)
        sd = load_file(str(out / best["snapshot"]), device=str(next(m.parameters()).device))
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not unexpected and not set(sd) & set(missing), (unexpected, set(sd) & set(missing))
        sel = {"chosen_step": best["step"], "chosen_ade": best["ade"], "chosen_tag": best["tag"], "history": hist}
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            (out / "ade_select.json").write_text(json.dumps(sel, indent=1))
        log.info("traj_early_stop: final = step %d (%s), ADE %.4f", best["step"], best["tag"], best["ade"])
        return sel
