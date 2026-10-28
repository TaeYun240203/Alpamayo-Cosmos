"""Before/after measurements of the R2 fine-tune (DATA_PLAN_R2 7 "측정"), on a BF16 student checkpoint or on the delivered
5080 engine (real NVFP4/FP8 GEMMs, the eval repo's EngineBackend). Every row uses the same initial noise in every
condition (seed = row position), so condition differences are the input's effect.
  history  history dependence on fixed held-out rows (default: the first n of the gate set, val + val_asset, frame_stride
           10, seed 1). Conditions: base; swap = the ego history TOKENS of another row (row k reads row (k + n/2) % n's 16
           points; the action space keeps the row's own history for v0 and the integration, as the fine-tune's dropout
           and the deployment do); nocam = all 16 camera frames black (LiDAR kept). Per condition ADE, FDE and |lateral|
           at 1/2/3 s in the GT path frame (scripts/lateral_error.py); and the change from base: mean distance between
           the two trajectories, |lateral| difference at 2/3 s. History dependence = how far swap moves the plan, camera
           use = how far nocam moves it.
  static   static obstacles (assets/rule_context.npz boxes, old data only): on rows whose GT body passes within 3 m of a
           scenario obstacle in 6.4 s, the body clearance along the predicted vs the GT trajectory (min, share < 0.3 m,
           overlaps), ADE; on every row of the static section (C2 pool) the mean speed over 6.4 s, predicted vs GT.
Usage (cwd train/): python scripts/ft_measure.py history|static --root <step-0 root> --out <dir> [--n 300]
         [--keep <parquet>] [--split val val_asset] [--stride 10] [--k 1] (--ckpt <student dir> | --engine <deliver dir>
         --server <eval repo>/src/e2e_driver/server) [--device cuda:0]
static defaults: --keep assets/ft/old_static.parquet --stride 5 --n 600. Writes <out>/rows.jsonl and <out>/summary.json."""
import argparse, json, sys
from pathlib import Path

import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from morai_vla.data import MoraiDataset                                          # noqa: E402
from morai_vla.processor import collate_fn, get_preprocess_fn                    # noqa: E402
from scripts.lateral_error import split_error                                    # noqa: E402

AT = {"1s": 9, "2s": 19, "3s": 29}
DROP = ("camera_frames", "bev_frames", "camera_times", "bev_times")


class BF16:
    """a saved student (StudentVLA.build) and its own inference; the (a, kappa) sample_trajectories integrates is read off
    diffusion.sample and re-integrated from the row's real history (so only the history tokens change in `swap`)."""

    def __init__(self, ckpt, device, vlm="Qwen/Qwen3-VL-4B-Instruct"):
        from morai_vla.student import StudentVLA
        self.m = StudentVLA.build(init=ckpt, vlm_name_or_path=vlm, attn_implementation="sdpa", dtype="bfloat16" if "cuda" in device else "float32",
                                  device_map={"": device}).eval()
        self.device, self.config, self.last = device, self.m.config, {}
        orig = self.m.diffusion.sample
        self.m.diffusion.sample = lambda *a, **k: self.last.__setitem__("action", orig(*a, **k)) or self.last["action"]

    def __call__(self, sample, hist, seed):
        d = {k: v for k, v in sample.items()}; td = get_preprocess_fn(self.config, generation_mode=True)(data=d)
        b = collate_fn([{k: v for k, v in d.items() if k not in DROP} | {"tokenized_data": td}], model_config=self.config)
        dev = lambda x: {k: dev(v) if isinstance(v, dict) else v.to(self.device) if torch.is_tensor(v) else v for k, v in x.items()}
        b = dev(b); hx, hr = (h[None].to(self.device) for h in hist)
        torch.manual_seed(seed)
        self.m.sample_trajectories(dict(b["tokenized_data"]), hx, hr, num_samples=1)
        xyz, _ = self.m.action_space.action_to_traj(self.last.pop("action").float(), b["ego_history_xyz"][:, 0].float(), b["ego_history_rot"][:, 0].float())
        return xyz[0, :, :2].float().cpu().numpy()


class Engine:
    """the delivered QAD student on the 5080 (eval repo server/backends.EngineBackend, FastEngine.run): the history
    tokens come from `hist`, (a, kappa) are integrated from the row's real history on the CPU, as EngineBackend.run."""

    def __init__(self, deliver, server, device):
        sys.path[:0] = [server, str(Path(server).parent / "src")]
        from backends import EngineBackend
        self.b = EngineBackend(deliver, device=device); self.device = device

    def __call__(self, sample, hist, seed):
        (b, keys, lay), _ = self.b.prepare(dict(sample))
        b = {k: ({kk: vv if kk == "pixel_values" else vv.to(self.device) for kk, vv in v.items()} if isinstance(v, dict)
                 else v.to(self.device) if torch.is_tensor(v) else v) for k, v in b.items()}
        hx, hr = (h[None].to(self.device) for h in hist)
        x0 = torch.randn(1, 64, 2, generator=torch.Generator().manual_seed(seed)).to(self.device)
        act, _ = self.b.fast.run(b["tokenized_data"], hx, hr, x0, keys, lay)
        xyz = self.b.sp_cpu.action_to_traj(act.cpu(), b["ego_history_xyz"][:, 0].cpu(), b["ego_history_rot"][:, 0].cpu())[0]
        return xyz[0, :, :2].float().numpy()


def rows_of(a, model_config=None):
    ds = MoraiDataset(a.root, a.split, frame_stride=a.stride, max_samples=a.n, augment=False, seed=1, keep=a.keep)
    ds.rows = ds.rows.iloc[:a.n].reset_index(drop=True)
    return ds


def gt_of(s):
    r = s["ego_future_rot"][0].numpy()
    return s["ego_future_xyz"][0, :, :2].numpy(), np.arctan2(r[:, 1, 0], r[:, 0, 0])


def errs(pred, gt, yaw):
    lat, _ = split_error(pred[None], gt[None], yaw[None]); e = np.linalg.norm(pred - gt, axis=-1)
    return {"ade": float(e.mean()), "fde": float(e[-1])} | {f"lat_{h}": float(abs(lat[0, i])) for h, i in AT.items()}


def history(a, model, ds):
    n = len(ds.rows); samples = [ds.sample(k) for k in range(n)]; out = []
    for k, s in enumerate(samples):
        gt, yaw = gt_of(s); own = (s["ego_history_xyz"], s["ego_history_rot"])
        o = samples[(k + n // 2) % n]; other = (o["ego_history_xyz"], o["ego_history_rot"])
        black = dict(s, camera_frames=[torch.zeros_like(f) for f in s["camera_frames"]])
        r = {"key": s["key"], "cell": str(ds.rows.cell.iloc[k]), "swap_from": o["key"]}
        for j in range(a.k):
            seed = 1000 * k + j
            base = model(s, own, seed)
            for name, pred in (("base", base), ("swap", model(s, other, seed)), ("nocam", model(black, own, seed))):
                for m, v in errs(pred, gt, yaw).items(): r[f"{name}_{m}"] = r.get(f"{name}_{m}", 0.0) + v / a.k
                if name != "base":
                    lat, _ = split_error(pred[None], base[None], yaw[None])
                    for m, v in {"move": float(np.linalg.norm(pred - base, axis=-1).mean()), "dlat_2s": float(abs(lat[0, AT["2s"]])),
                                 "dlat_3s": float(abs(lat[0, AT["3s"]]))}.items():
                        r[f"{name}_{m}"] = r.get(f"{name}_{m}", 0.0) + v / a.k
        out.append(r)
        if k % 25 == 0: print(json.dumps(r), flush=True)
    keys = [x for x in out[0] if x not in ("key", "cell", "swap_from")]
    summ = {x: {"mean": round(float(np.mean([r[x] for r in out])), 4), "p90": round(float(np.percentile([r[x] for r in out], 90)), 4)} for x in keys}
    return out, summ


def path_speed(xy):
    """mean speed (m/s) over the 6.4 s horizon: path length from t0."""
    return float(np.linalg.norm(np.diff(np.r_[[[0.0, 0.0]], xy], axis=0), axis=1).sum() / 6.4)


def clearance(rules, ego, rows, traj):
    """(N, 64, 2) t0-frame paths -> min body-to-obstacle distance (m, 0 when they overlap) over 6.4 s, (N,) nan if the
    episode has no static obstacle; overlap flag (N,)."""
    j = rules.ep_index.get(ego["episode"]); o = rules.obs[rules.obs[:, 0] == j][:, 1:] if j is not None else np.zeros((0, 5))
    if not len(o): return np.full(len(rows), np.nan), np.zeros(len(rows), bool)
    x0, y0, h0 = (ego[k][rows][:, None] for k in ("x", "y", "yaw"))
    tx, ty = traj[..., 0], traj[..., 1]
    X, Y = x0 + np.cos(h0) * tx - np.sin(h0) * ty, y0 + np.sin(h0) * tx + np.cos(h0) * ty
    gx, gy = np.gradient(tx, axis=1), np.gradient(ty, axis=1); moving = np.hypot(gx, gy) > 0.02     # heading along the path,
    last = np.maximum.accumulate(np.where(moving, np.arange(tx.shape[1])[None], 0), axis=1)          # held while standing
    H = h0 + np.take_along_axis(np.where(moving, np.arctan2(gy, gx), 0.0), last, 1)
    rear, front, hw = rules.z["body"]
    body = np.array([[rear, -hw], [rear, hw], [front, hw], [front, -hw]])
    best = np.full(X.shape, np.inf)
    for ox, oy, oh, ol, ow in o:
        # body corners in the obstacle frame -> distance to the obstacle rectangle; obstacle corners in the body frame
        c, s = np.cos(H)[..., None], np.sin(H)[..., None]
        bx, by = X[..., None] + c * body[:, 0] - s * body[:, 1], Y[..., None] + s * body[:, 0] + c * body[:, 1]
        lx, ly = np.cos(oh) * (bx - ox) + np.sin(oh) * (by - oy), -np.sin(oh) * (bx - ox) + np.cos(oh) * (by - oy)
        d1 = np.hypot(np.maximum(np.abs(lx) - ol, 0), np.maximum(np.abs(ly) - ow, 0)).min(-1)
        oc = np.array([[ol, ow], [ol, -ow], [-ol, ow], [-ol, -ow]])
        ex, ey = ox + np.cos(oh) * oc[:, 0] - np.sin(oh) * oc[:, 1], oy + np.sin(oh) * oc[:, 0] + np.cos(oh) * oc[:, 1]
        qx = np.cos(H)[..., None] * (ex - X[..., None]) + np.sin(H)[..., None] * (ey - Y[..., None])
        qy = -np.sin(H)[..., None] * (ex - X[..., None]) + np.cos(H)[..., None] * (ey - Y[..., None])
        mid, half = (front + rear) / 2, (front - rear) / 2
        d2 = np.hypot(np.maximum(np.abs(qx - mid) - half, 0), np.maximum(np.abs(qy) - hw, 0)).min(-1)
        best = np.minimum(best, np.minimum(d1, d2))
    hit = rules.obstacle(ego, X[:, None], Y[:, None], H[:, None])[:, 0]
    return np.where(hit, 0.0, best.min(1)), hit


def static(a, model, ds):
    from scripts.rule_check import Rules, ego_of
    rules = Rules(); out = []
    for k in range(len(ds.rows)):
        s = ds.sample(k); gt, yaw = gt_of(s); ep, i = s["key"].rsplit("/", 1); i = int(i)
        ego = ego_of(a.root, ep)
        preds = [model(s, (s["ego_history_xyz"], s["ego_history_rot"]), 1000 * k + j) for j in range(a.k)]
        r = {"key": s["key"], "cell": str(ds.rows.cell.iloc[k])}
        gcl, ghit = clearance(rules, ego, np.array([i]), gt[None])
        r |= {"gt_clear": float(gcl[0]), "gt_speed": path_speed(gt)}
        for p in preds:
            cl, hit = clearance(rules, ego, np.array([i]), p[None])
            m = errs(p, gt, yaw) | {"clear": float(cl[0]), "overlap": float(hit[0]), "speed": path_speed(p)}
            for x, v in m.items(): r[f"pred_{x}"] = r.get(f"pred_{x}", 0.0) + v / len(preds)
        out.append(r)
        if k % 25 == 0: print(json.dumps(r), flush=True)
    near = [r for r in out if np.isfinite(r["gt_clear"]) and r["gt_clear"] < 3.0]
    mean = lambda rs, x: round(float(np.mean([r[x] for r in rs])), 4) if rs else None
    summ = {"rows": len(out), "speed_pred": mean(out, "pred_speed"), "speed_gt": mean(out, "gt_speed"), "ade": mean(out, "pred_ade"),
            "near_obstacle": {"n": len(near), "ade": mean(near, "pred_ade"), "clear_pred": mean(near, "pred_clear"), "clear_gt": mean(near, "gt_clear"),
                              "clear_pred_p10": round(float(np.percentile([r["pred_clear"] for r in near], 10)), 4) if near else None,
                              "share_pred_clear_lt_0.3m": mean([dict(x=float(r["pred_clear"] < 0.3)) for r in near], "x"),
                              "overlap_rate": mean(near, "pred_overlap")}}
    return out, summ


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["history", "static"]); ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int); ap.add_argument("--keep"); ap.add_argument("--split", nargs="+", default=["val", "val_asset"])
    ap.add_argument("--stride", type=int); ap.add_argument("--k", type=int, default=1); ap.add_argument("--ckpt"); ap.add_argument("--engine")
    ap.add_argument("--server"); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if a.what == "static":
        a.keep = a.keep or str(Path(__file__).resolve().parents[1] / "assets/ft/old_static.parquet"); a.stride = a.stride or 5; a.n = a.n or 600
    a.stride, a.n = a.stride or 10, a.n or 300
    model = BF16(a.ckpt, a.device) if a.ckpt else Engine(a.engine, a.server, a.device)
    ds = rows_of(a)
    out, summ = (history if a.what == "history" else static)(a, model, ds)
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    with open(o / "rows.jsonl", "w") as f:
        for r in out: f.write(json.dumps(r) + "\n")
    summ = {"what": a.what, "backend": "bf16:" + str(a.ckpt) if a.ckpt else "engine:" + str(a.engine), "n": len(out), "k": a.k,
            "root": a.root, "keep": a.keep, "split": a.split, "stride": a.stride} | summ
    (o / "summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
