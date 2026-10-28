"""Does the predicted trajectory collapse to a point when the car should stay stopped, and does it leave when the car
should leave? From saved trajectories (scripts/lateral_error.py / signal_eval.py npz: key, pred, gt in the t0 ego frame);
CPU only. travel = max |position| over the horizon (or its first 3 s).
- GT stopped: (a) GT travel over 6.4 s < 0.5 m, (b) GT travel over the first 3 s < 0.5 m. Prediction travel median /
  p90 / max, share above 0.5 / 1 / 3 m in the first 3 s, worst cases.
- Late departure: standing at t0 (v0 < 0.5 m/s) and GT travel over the first 2 s > 0.5 m; share where the prediction
  stays within 0.5 m for 2 s, and the departure time lag (first step with travel > 0.5 m).
Usage (cwd train/): python scripts/stationary_eval.py <step-0 root> <npz> [<npz> ...] --out logs/stageC/stationary"""
import argparse, json
from pathlib import Path

import numpy as np, pandas as pd


def travel(x, upto=None):
    return np.linalg.norm(x[:, :upto], axis=-1).max(1)


def depart_step(x, thr=0.5):
    """first index (0.1 s steps from t0 + 0.1) whose distance from t0 exceeds thr, 64 = never."""
    m = np.linalg.norm(x, axis=-1) > thr
    return np.where(m.any(1), m.argmax(1), 64)


def stats(p):
    return {"n": int(len(p)), "median": float(np.median(p)) if len(p) else None, "p90": float(np.percentile(p, 90)) if len(p) else None,
            "max": float(p.max()) if len(p) else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("npz", nargs="+"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    keys, P, G = [], [], []
    for f in a.npz:
        d = np.load(f, allow_pickle=True); keys += list(d["key"]); P.append(d["pred"]); G.append(d["gt"])
    P, G = np.concatenate(P), np.concatenate(G)
    _, u = np.unique(keys, return_index=True); u = np.sort(u); keys = np.asarray(keys)[u]; P, G = P[u], G[u]   # shared rows once
    idx = pd.read_parquet(Path(a.root) / "_index/index.parquet", columns=["episode", "i", "cell", "nav_text", "fog"]).astype({"episode": str})
    df = pd.DataFrame({"episode": [k.rsplit("/", 1)[0] for k in keys], "i": [int(k.rsplit("/", 1)[1]) for k in keys]}).merge(idx, on=["episode", "i"], how="left")
    df["v0"] = [float(np.load(Path(a.root) / "_ego" / f"{e}.npz")["v"][i]) for e, i in zip(df.episode, df.i)]
    df["gt_6s"], df["gt_3s"], df["gt_2s"] = travel(G), travel(G, 30), travel(G, 20)
    df["pred_6s"], df["pred_3s"], df["pred_2s"] = travel(P), travel(P, 30), travel(P, 20)
    df["gt_dep"], df["pred_dep"] = depart_step(G), depart_step(P)
    res = {"n_rows": len(df)}
    for name, m in (("gt_stopped_6.4s", df.gt_6s < 0.5), ("gt_stopped_first3s", df.gt_3s < 0.5)):
        g = df[m]
        res[name] = {"pred_travel_6.4s": stats(g.pred_6s.to_numpy()), "pred_travel_first3s": stats(g.pred_3s.to_numpy()),
                     **{f"share_first3s_gt_{t}m": float((g.pred_3s > t).mean()) for t in (0.5, 1, 3)},
                     "by_cell_share_first3s_gt_1m": {c: float((h.pred_3s > 1).mean()) for c, h in g.groupby("cell")},
                     "worst": g.sort_values("pred_3s", ascending=False).head(6)[["episode", "i", "cell", "nav_text", "fog", "v0", "pred_3s", "pred_6s"]].to_dict("records")}
    dep = df[(df.v0 < 0.5) & (df.gt_2s > 0.5)]
    res["late_departure"] = {"n_gt_departs_within_2s": int(len(dep)), "share_pred_stays_2s": float((dep.pred_2s < 0.5).mean()) if len(dep) else None,
                             "gt_depart_s_median": float(np.median(dep.gt_dep + 1) / 10) if len(dep) else None,
                             "pred_depart_s_median": float(np.median(dep.pred_dep + 1) / 10) if len(dep) else None,
                             "lag_s_median": float(np.median(dep.pred_dep - dep.gt_dep) / 10) if len(dep) else None,
                             "worst": dep.assign(lag=(dep.pred_dep - dep.gt_dep) / 10).sort_values("lag", ascending=False).head(6)[
                                 ["episode", "i", "cell", "nav_text", "fog", "lag", "pred_2s", "gt_2s"]].to_dict("records")}
    df.to_csv(a.out + ".csv", index=False); json.dump(res, open(a.out + ".json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))
