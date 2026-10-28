"""Signal reading of a checkpoint (MODEL_PLAN 4 step 3.5, T* signal check): val + val_asset rows whose next judged stop line
is 5-80 m ahead on the route, stratified by the GT light at t0 (red / yellow / green) x fog, up to --n rows. The discrete-token
greedy trajectory and the GT future are each classified "passes the line" (front axle crosses that stop line within
6.4 s) or "stops before it". Reports agreement by distance bin x fog x light, red-run rate (GT stops, prediction passes)
and false-stop rate (GT passes, prediction stops); saves trajectories (<out>.npz), tables (<out>.json) and the worst
red-run cases with front-camera thumbnails (<out>.png).
Not separated: 4356's arrow state (the _ego signal code has red / yellow / green / unknown only).
Usage (cwd train/): python scripts/signal_eval.py <checkpoint> <step-0 root> --out logs/signal_eval [--n 400] [--batch 16]"""
import argparse, functools, json
from pathlib import Path

import numpy as np, pandas as pd, torch

from morai_vla.data import MoraiDataset
from morai_vla.model import TeacherVLA
from morai_vla.processor import collate_fn
from morai_vla.stage1_eval import discrete_trajectory

CTX = Path(__file__).resolve().parents[1] / "assets/rule_context.npz"
LIGHT = {1: "red", 2: "yellow", 3: "green"}
BINS = [(5, 20), (20, 40), (40, 60), (60, 80)]


def stop_lines(z):
    """[(stop id, (P, 2) polyline, route s of its midpoint)]"""
    out = []
    for j, sid in enumerate(z["stop_ids"]):
        P = z["stop_pts"][z["stop_off"][j]:z["stop_off"][j + 1]]
        out.append((int(sid), P, float(z["path_s"][np.argmin(np.hypot(*(z["path"] - P.mean(0)).T))])))
    return out


def candidates(root, z, stride=5):
    idx = pd.read_parquet(Path(root) / "_index/index.parquet", columns=["episode", "i", "cell", "split"])
    idx = idx[idx.split.isin(["val", "val_asset"]) & (idx.i % stride == 0)]
    stops, rows = stop_lines(z), []
    for ep, g in idx.groupby("episode", observed=True):
        f = Path(root) / "_ego" / f"{ep}.npz"
        if not f.exists(): continue
        e = np.load(f); i = g.i.to_numpy(); s = e["path_s"][i].astype(float)
        for sid, _, ss in stops:
            d = ss - s; m = (d >= 5) & (d <= 80) & np.isin(e["signal"][i], list(LIGHT))
            for k in np.nonzero(m)[0]:
                rows.append((ep, int(i[k]), str(g.cell.iloc[k]), sid, float(d[k]), int(e["signal"][i[k]]), bool(e["fog"][i[k]])))
    df = pd.DataFrame(rows, columns=["episode", "i", "cell", "stop", "dist", "light", "fog"])
    return df.sort_values("dist").drop_duplicates(["episode", "i"])            # the nearest judged stop line ahead


def crosses(z, ego, i, traj, stop_poly):
    """traj (64, 2) in the t0 ego frame of row i -> front axle crosses the polyline within 6.4 s."""
    x0, y0, h0 = ego["x"][i], ego["y"][i], ego["yaw"][i]
    p = np.vstack([[0, 0], traj]); h, a = np.zeros(len(p)), 0
    for t in range(1, len(p)):                  # heading changes only after 0.5 m of travel: a standing trajectory's
        d = p[t] - p[a]                         # jitter (up to 0.25 m, 7294) would swing the front axle across the line
        h[t] = np.arctan2(d[1], d[0]) if np.hypot(*d) > 0.5 else h[t - 1]
        if np.hypot(*d) > 0.5: a = t
    X, Y = x0 + np.cos(h0) * p[:, 0] - np.sin(h0) * p[:, 1], y0 + np.sin(h0) * p[:, 0] + np.cos(h0) * p[:, 1]
    fa = float(z["front_axle"]); FX, FY = X + fa * np.cos(h0 + h), Y + fa * np.sin(h0 + h)
    cr = lambda o, u, v: (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])
    for a, b in zip(stop_poly[:-1], stop_poly[1:]):
        for t in range(len(FX) - 1):
            pt, qt = (FX[t], FY[t]), (FX[t + 1], FY[t + 1])
            if cr(a, b, pt) * cr(a, b, qt) < 0 and cr(pt, qt, a) * cr(pt, qt, b) < 0: return True
    return False


def tables(df):
    def stat(g):
        return {"n": len(g), "agree": float((g.pred_pass == g.gt_pass).mean()), "red_run": int(((g.light == 1) & ~g.gt_pass & g.pred_pass).sum()),
                "gt_stop_red": int(((g.light == 1) & ~g.gt_pass).sum()), "false_stop": int((g.gt_pass & ~g.pred_pass).sum()), "gt_pass": int(g.gt_pass.sum())}
    df = df.assign(bin=pd.cut(df.dist, [b[0] for b in BINS] + [BINS[-1][1]], labels=[f"{a}-{b}m" for a, b in BINS], include_lowest=True),
                   light_name=df.light.map(LIGHT), weather=np.where(df.fog, "fog", "clear"))
    out = {"all": stat(df), "by_light": {k: stat(g) for k, g in df.groupby("light_name")},
           "by_bin": {str(k): stat(g) for k, g in df.groupby("bin", observed=True)}, "by_weather": {k: stat(g) for k, g in df.groupby("weather")},
           "by_bin_weather_light": {f"{b}|{w}|{l}": stat(g) for (b, w, l), g in df.groupby(["bin", "weather", "light_name"], observed=True)}}
    a = out["all"]; out["red_run_rate"] = a["red_run"] / max(a["gt_stop_red"], 1); out["false_stop_rate"] = a["false_stop"] / max(a["gt_pass"], 1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("root"); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=400); ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args(); z = np.load(CTX)
    cand = candidates(a.root, z)
    per = int(np.ceil(a.n / cand.groupby(["light", "fog"]).ngroups))
    pick = pd.concat([g.sample(min(per, len(g)), random_state=0) for _, g in cand.groupby(["light", "fog"])])   # keeps light/fog columns
    m = TeacherVLA.build(a.ckpt, attn_implementation="sdpa", device_map={"": 0}).eval()
    ds = MoraiDataset(a.root, ["val", "val_asset"], model_config=m.config, frame_stride=5, augment=False,
                      vla_preprocess_args={"_target_": "morai_vla.processor.get_preprocess_fn", "generation_mode": True})
    ds.rows = ds.rows.astype({"episode": str}).merge(pick[["episode", "i"]].astype({"episode": str}), on=["episode", "i"]).reset_index(drop=True)
    pick = ds.rows[["episode", "i"]].merge(pick.astype({"episode": str}), on=["episode", "i"])      # usable rows, dataset order
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=8, collate_fn=functools.partial(collate_fn, model_config=m.config))
    dev = lambda x: {k: dev(v) if isinstance(v, dict) else v.cuda() if torch.is_tensor(v) else v for k, v in x.items()}
    P, G = [], []
    for b in dl:
        b = dev(b)
        P.append(discrete_trajectory(m, b["tokenized_data"], b["ego_history_xyz"], b["ego_history_rot"])[..., :2].numpy())
        G.append(b["ego_future_xyz"][:, 0, :, :2].float().cpu().numpy()); print(sum(len(x) for x in P), flush=True)
    P, G = np.concatenate(P), np.concatenate(G)
    polys = {sid: poly for sid, poly, _ in stop_lines(z)}
    eg = {ep: dict(np.load(Path(a.root) / "_ego" / f"{ep}.npz")) for ep in pick.episode.unique()}
    pick["pred_pass"] = [crosses(z, eg[e], i, P[k], polys[s]) for k, (e, i, s) in enumerate(zip(pick.episode, pick.i, pick.stop))]
    pick["gt_pass"] = [crosses(z, eg[e], i, G[k], polys[s]) for k, (e, i, s) in enumerate(zip(pick.episode, pick.i, pick.stop))]
    pick["pred_travel"], pick["gt_travel"] = np.hypot(*np.diff(np.concatenate([np.zeros((len(P), 1, 2)), P], 1), axis=1).T).sum(0), \
        np.hypot(*np.diff(np.concatenate([np.zeros((len(G), 1, 2)), G], 1), axis=1).T).sum(0)
    np.savez(a.out + ".npz", pred=P, gt=G, key=(pick.episode + "/" + pick.i.astype(str)).to_numpy())
    pick.to_csv(a.out + ".csv", index=False)
    res = tables(pick); json.dump(res, open(a.out + ".json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("all", "by_light", "by_bin", "by_weather", "red_run_rate", "false_stop_rate")}, indent=1))
    # worst red runs: GT stops on red, prediction passes; largest prediction travel beyond the stop distance
    rr = pick[(pick.light == 1) & ~pick.gt_pass & pick.pred_pass].assign(over=lambda d: d.pred_travel - d.dist).sort_values("over", ascending=False).head(8)
    if len(rr):
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        raw = MoraiDataset(a.root, ["val", "val_asset"], frame_stride=5, augment=False); raw.rows = raw.rows.astype({"episode": str}).merge(rr[["episode", "i"]], on=["episode", "i"]).reset_index(drop=True)
        fig, ax = plt.subplots(len(rr), 2, figsize=(10, 3.2 * len(rr)), squeeze=False, gridspec_kw={"width_ratios": [1.75, 1]})
        for r, (_, row) in enumerate(rr.iterrows()):
            k = int(np.nonzero((raw.rows.episode == row.episode).to_numpy() & (raw.rows.i == row.i).to_numpy())[0][0]); s = raw.sample(k)
            ax[r, 0].imshow(s["camera_frames"][0][-1].permute(1, 2, 0).numpy()); ax[r, 0].axis("off")
            ax[r, 0].set_title(f"{row.episode}/{row.i} {row.cell} stop {row.stop} {row.dist:.0f} m red, fog={row.fog}", fontsize=8)
            j = int(np.nonzero((pick.episode == row.episode).to_numpy() & (pick.i == row.i).to_numpy())[0][0])
            ax[r, 1].plot(-G[j, :, 1], G[j, :, 0], "k", label="GT"); ax[r, 1].plot(-P[j, :, 1], P[j, :, 0], "C3", label="pred")
            ax[r, 1].axhline(row.dist, color="C1", ls="--", lw=1); ax[r, 1].set_aspect("equal"); ax[r, 1].legend(fontsize=7)
            ax[r, 1].set_title(f"GT {row.gt_travel:.0f} m, pred {row.pred_travel:.0f} m (line at {row.dist:.0f} m ahead)", fontsize=8)
        fig.tight_layout(); fig.savefig(a.out + ".png", dpi=80)
