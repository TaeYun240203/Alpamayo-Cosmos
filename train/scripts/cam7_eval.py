"""Go/stay at a stop line: val + val_asset rows standing (v < 1 m/s) between --dmin and --near m (route distance from the
ego reference point to the line) before the judged stop lines in --stops, GT either staying for 3 s ("wait") or leaving
("go": over 0.5 m within 2 s AND at least GO_M over 6.4 s, so a creep toward the line is neither), stratified by stop line
x wait/go x fog (all candidates when there are at most --n). Discrete-token greedy trajectories for the inputs in --variants:
recorded, and optionally cam7_gray / front_gray (that camera's frames flat gray).
Reports per variant: red-light violations (the predicted front axle crosses the stop line at a future row whose GT light
is red; unknown counted apart; the GT light already encodes 4356's left arrow), late departures (GT goes, prediction stays
within 0.5 m for 2 s), agreement and departure time, by stop line; Front + Cam7 thumbnails of violations (<out>.png).
Trajectories are appended to <out>.jsonl as they come (a restart skips done rows); summary <out>.json.
Usage (cwd train/): python scripts/cam7_eval.py <checkpoint> <step-0 root> --out logs/stageC/cam7 [--n 150] [--nf4]
       [--stops 4188 --dmin 25 --near 45]"""
import argparse, json
from pathlib import Path

import numpy as np, pandas as pd, torch
from transformers import BitsAndBytesConfig

from morai_vla.data import MoraiDataset
from morai_vla.model import TeacherVLA
from morai_vla.processor import CAMERAS, collate_fn
from morai_vla.stage1_eval import discrete_trajectory
from scripts.signal_eval import CTX, crosses, stop_lines

GO_M = 5.0                                                                     # a departure moves at least this far in 6.4 s
VARIANTS = {"recorded": None, "cam7_gray": 3, "front_gray": 0}                  # camera index to gray out (CAMERAS order)
assert CAMERAS[3][0] == "Front upward camera" and CAMERAS[0][0] == "Front camera"


class Grayed(MoraiDataset):
    gray = None

    def sample(self, idx, rng=None):
        s = super().sample(idx, rng)
        if self.gray is not None: s["camera_frames"][self.gray] = torch.full_like(s["camera_frames"][self.gray], 128)
        return s


def cross_step(z, ego, i, traj, poly):
    """first future index (0-63) where the predicted front axle crosses the polyline, or None (signal_eval.crosses' geometry)."""
    for t in range(1, len(traj) + 1):
        if crosses(z, ego, i, traj[:t], poly): return t - 1
    return None


def candidates(root, z, near, keep_stops, dmin=0.0):
    idx = pd.read_parquet(Path(root) / "_index/index.parquet", columns=["episode", "i", "split"])
    idx = idx[idx.split.isin(["val", "val_asset"]) & (idx.i % 2 == 0)]
    stops, rows = stop_lines(z), []
    for ep, g in idx.groupby("episode", observed=True):
        f = Path(root) / "_ego" / f"{ep}.npz"
        if not f.exists(): continue
        e = np.load(f); i = g.i.to_numpy(); i = i[(i + 64 < len(e["v"]))]
        x, y, v = e["x"], e["y"], e["v"]
        for sid, _, ss in stops:
            if sid not in keep_stops: continue
            d = ss - e["path_s"][i].astype(float); m = (d >= dmin) & (d <= near) & (v[i] < 1.0)
            for k in i[m]:
                fut = np.hypot(x[k + 1:k + 65] - x[k], y[k + 1:k + 65] - y[k])
                kind = "wait" if fut[:30].max() < 0.5 else "go" if fut[:20].max() > 0.5 and fut.max() >= GO_M else None
                if kind: rows.append((str(ep), int(k), sid, float(ss - e["path_s"][k]), kind, bool(e["fog"][k]), int(e["signal"][k])))
    return pd.DataFrame(rows, columns=["episode", "i", "stop", "dist", "kind", "fog", "light"]).drop_duplicates(["episode", "i"])


def allocate(sizes, n):
    """rows per stratum: ceil(n / strata) each, capped at the stratum size; what small strata leave goes to the others
    (all candidates when there are at most n)."""
    take = [min(s, -(-n // len(sizes))) for s in sizes]
    while sum(take) < n and any(t < s for t, s in zip(take, sizes)):
        k = min((t, j) for j, (t, s) in enumerate(zip(take, sizes)) if t < s)[1]; take[k] += 1
    return take


def summarize(rows):
    df = pd.DataFrame(rows); out = {}
    for var, g in df.groupby("variant"):
        gt_go, pr_go = g.kind == "go", g.pred_2s > 0.5
        def s(h, a, b):
            return {"n": len(h), "agree": float((a == b).mean()), "pred_go_when_gt_wait": int((b & ~a).sum()), "pred_wait_when_gt_go": int((~b & a).sum()),
                    "gt_depart_s_median": float(np.median(h.gt_dep[a] + 1) / 10) if a.any() else None,
                    "pred_depart_s_median_when_gt_go": float(np.median(h.pred_dep[a] + 1) / 10) if a.any() else None}
        out[var] = {"all": s(g, gt_go, pr_go), "by_stop": {int(k): s(h, h.kind == "go", h.pred_2s > 0.5) for k, h in g.groupby("stop")},
                    "by_kind_fog": {f"{k}|{'fog' if f else 'clear'}": s(h, h.kind == "go", h.pred_2s > 0.5) for (k, f), h in g.groupby(["kind", "fog"])}}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("root"); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=150); ap.add_argument("--near", type=float, default=10.0); ap.add_argument("--dmin", type=float, default=0.0); ap.add_argument("--nf4", action="store_true")
    ap.add_argument("--batch", type=int, default=2); ap.add_argument("--stops", default="6254"); ap.add_argument("--variants", default="recorded")
    a = ap.parse_args(); z = np.load(CTX)
    cand = candidates(a.root, z, a.near, {int(x) for x in a.stops.split(",")}, a.dmin); print(cand.groupby(["stop", "kind", "fog"]).size().to_string(), flush=True)
    groups = [g for _, g in cand.groupby(["stop", "kind", "fog"])]
    pick = pd.concat([g.sample(t, random_state=0) for g, t in zip(groups, allocate([len(g) for g in groups], a.n))]).reset_index(drop=True)
    done = set()
    if Path(a.out + ".jsonl").exists():
        done = {(r["key"], r["variant"]) for r in map(json.loads, open(a.out + ".jsonl"))}
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                           llm_int8_skip_modules=["visual", "lm_head", "action_in_proj", "action_out_proj"]) if a.nf4 else None
    m = TeacherVLA.build(a.ckpt, attn_implementation="sdpa", quantization_config=q, device_map={"": 0}).eval()
    ds = Grayed(a.root, ["val", "val_asset"], model_config=m.config, frame_stride=2, augment=False,
                vla_preprocess_args={"_target_": "morai_vla.processor.get_preprocess_fn", "generation_mode": True})
    ds.rows = ds.rows.astype({"episode": str}).merge(pick[["episode", "i"]], on=["episode", "i"]).reset_index(drop=True)
    pick = ds.rows[["episode", "i"]].merge(pick, on=["episode", "i"])
    dev = lambda x: {k: dev(v) if isinstance(v, dict) else v.cuda() if torch.is_tensor(v) else v for k, v in x.items()}
    with open(a.out + ".jsonl", "a") as f:
        for var, cam in ((v, VARIANTS[v]) for v in a.variants.split(",")):
            ds.gray = cam
            todo = [k for k in range(len(ds.rows)) if (f"{pick.episode[k]}/{pick.i[k]}", var) not in done]
            for j in range(0, len(todo), a.batch):
                ks = todo[j:j + a.batch]; b = dev(collate_fn([ds[k] for k in ks], model_config=m.config))
                P = discrete_trajectory(m, b["tokenized_data"], b["ego_history_xyz"], b["ego_history_rot"])[..., :2].numpy()
                G = b["ego_future_xyz"][:, 0, :, :2].float().cpu().numpy()
                for k, p, g in zip(ks, P, G):
                    r = pick.iloc[k]
                    f.write(json.dumps({"key": f"{r.episode}/{r.i}", "variant": var, "stop": int(r.stop), "dist": r.dist, "kind": r.kind, "fog": bool(r.fog),
                                        "light": int(r.light), "pred": p.round(3).tolist(), "gt": g.round(3).tolist()}) + "\n")
                f.flush(); print(var, j + len(ks), "/", len(todo), flush=True)
    rows, polys, egos = [], {sid: poly for sid, poly, _ in stop_lines(z)}, {}
    for r in map(json.loads, open(a.out + ".jsonl")):
        p, g = np.array(r.pop("pred")), np.array(r.pop("gt")); dp, dg = np.linalg.norm(p, axis=1), np.linalg.norm(g, axis=1)
        ep, i = r["key"].rsplit("/", 1); i = int(i); eg = egos.setdefault(ep, dict(np.load(Path(a.root) / "_ego" / f"{ep}.npz")))
        t = cross_step(z, eg, i, p, polys[r["stop"]]); light = int(eg["signal"][min(i + t + 1, len(eg["signal"]) - 1)]) if t is not None else None
        rows.append(r | {"pred_2s": float(dp[:20].max()), "gt_2s": float(dg[:20].max()), "pred_dep": int(np.argmax(dp > 0.5)) if (dp > 0.5).any() else 64,
                         "gt_dep": int(np.argmax(dg > 0.5)) if (dg > 0.5).any() else 64, "pred_cross_s": None if t is None else (t + 1) / 10,
                         "light_at_cross": light, "violation": light == 1, "unknown_at_cross": light == 0})
    df = pd.DataFrame(rows)
    res = summarize(rows) | {"nf4": a.nf4, "ckpt": a.ckpt, "n_rows": len(pick)}
    for var, g in df.groupby("variant"):
        go = g.kind == "go"
        res[var] |= {"red_violations": int(g.violation.sum()), "crossings_on_unknown": int(g.unknown_at_cross.sum()), "n": len(g),
                     "late_departure_on_go": int((go & (g.pred_2s < 0.5)).sum()), "n_go": int(go.sum()),
                     "violations": g[g.violation][["key", "stop", "dist", "kind", "fog", "pred_cross_s"]].to_dict("records")}
    viol = df[(df.variant == "recorded") & df.violation]
    if len(viol):
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(len(viol), 2, figsize=(9, 3 * len(viol)), squeeze=False)
        for r, (_, row) in enumerate(viol.iterrows()):
            k = int(np.nonzero((pick.episode + "/" + pick.i.astype(str)).to_numpy() == row.key)[0][0]); ds.gray = None; s = ds.sample(k)
            for c, cam in enumerate((0, 3)):
                ax[r, c].imshow(s["camera_frames"][cam][-1].permute(1, 2, 0).numpy()); ax[r, c].axis("off")
            ax[r, 0].set_title(f"{row.key} stop {row.stop} {row.dist:.1f} m {row.kind} fog={row.fog}: crosses at {row.pred_cross_s:.1f} s on red", fontsize=8)
        fig.tight_layout(); fig.savefig(a.out + ".png", dpi=80)
    json.dump(res, open(a.out + ".json", "w"), indent=1); print(json.dumps({k: v["all"] for k, v in res.items() if isinstance(v, dict) and "all" in v}, indent=1))
