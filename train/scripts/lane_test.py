"""Lane-centre return test of a BF16 student (10/28, for the DAgger train parts): started off the lane centre, does the
model see the lane with its cameras and plan back to it? Three steps, the map work on the Mac, the model on a GPU box:
  select  (Mac) rows of the R2 D held-out scenes (index split val / val_asset, assets/ft/r2_D.parquet: the scenes of the
          fine-tune's "curve" R2 300-row set, which at frame_stride 10 has only ~74 rows, so here every grid row): all rows
          with |offset| >= 1 m and --n-ctl control rows with |offset| < 0.3 m. Index rows are trainable rows, i.e. no noise
          flag in their 6.4 s future (tools/step0/r2_relabel_gaps.py): the GT is the collection driver's own return.
          offset o = signed distance (left +) of the ego position to the route lane centerline, the map match of the
          fine-tune's lane-shaped histories (ft_synth_hist.project). Writes sel.parquet (episode, i, o, yaw_err, v0, bin).
  run     (GPU) per row, same initial noise in every condition (seed = row position), ft_measure.BF16:
            real      the row's real history
            straight  history tokens = v0 constant along the current heading (ft_synth_hist's straight fallback): no
                      offset cue left in the history
            nocam     straight + all 16 camera frames black (LiDAR kept; ft_measure nocam)
          only the history tokens change; v0 and the integration use the real history (ft_measure, fine-tune, deployment).
          Writes preds.npz (key, real / straight / nocam: (N, 64, 2) t0-frame xy) and run.json.
  report  (Mac) lateral position to the lane centre of GT and each prediction over 6.4 s; per condition x bin (ctl < 0.3,
          1-2, 2-3.5, >= 3.5 m, all >= 1 m): return 1 - |d(t)| / |o| at 2 s and 3 s (mean, median), slope of d(3 s) - o on
          o through the origin (-1 = back on the centre, 0 = kept the offset), episodes per bin, and the >= 1 m rows by t0
          heading (toward the centre / parallel within 5 deg / away); figures. Baseline cv: v0 straight on along the t0
          heading (what the heading alone gives). Without --preds: GT and cv only (CPU check of the map side).
Usage (cwd train/):
  python scripts/lane_test.py select --root <dir with _index/index.parquet and _ego/> --out sel.parquet [--n-ctl 80]
  python scripts/lane_test.py run --root <step0_r2> --sel sel.parquet --ckpt <ft/final> --out <dir> [--device cuda:0]
  python scripts/lane_test.py report --ego <step0_r2>/_ego --sel sel.parquet [--preds <dir>/preds.npz] --out <dir> [--figs 8]"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
BINS = (("ctl", 0.0, 0.3), ("1-2", 1.0, 2.0), ("2-3.5", 2.0, 3.5), (">=3.5", 3.5, np.inf))
AT = {"2s": 19, "3s": 29}
CONDS = ("real", "straight", "nocam")


def bin_of(o):
    a = abs(o)
    return next((n for n, lo, hi in BINS if lo <= a < hi), None)


def lane_offsets(P, S, z, i, xy):
    """(T, 2) positions in the ego frame of _ego row i -> signed offset (left +) of each to the route lane centerline;
    each point is matched near the row's route progress plus the distance driven to it."""
    from scripts.ft_synth_hist import project
    x0, y0, h0, s0 = (float(z[k][i]) for k in ("x", "y", "yaw", "path_s"))
    c, s = np.cos(h0), np.sin(h0); X, Y = x0 + c * xy[:, 0] - s * xy[:, 1], y0 + s * xy[:, 0] + c * xy[:, 1]
    along = np.cumsum(np.hypot(*np.diff(np.r_[[[0.0, 0.0]], xy], axis=0).T))
    return np.array([project(P, S, X[k], Y[k], s0 + along[k])[1] for k in range(len(xy))])


def check_map(P, S):
    """self-check: points 2 m left / 1.5 m right of the centerline, given in the ego frame of one of them, read back as
    +2 / -1.5 m (frame transform and progress matching)."""
    d = np.diff(P, axis=0); H = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    for s0, lat in ((95.0, 2.0), (190.0, -1.5), (590.0, 3.0)):
        s = s0 + 0.8 * np.arange(64); k = np.clip(np.searchsorted(S, s) - 1, 0, len(P) - 2); u = (s - S[k]) / (S[k + 1] - S[k])
        h = H[k]; x = P[k, 0] + u * d[k, 0] - lat * np.sin(h); y = P[k, 1] + u * d[k, 1] + lat * np.cos(h)
        c, sn = np.cos(h[0]), np.sin(h[0]); xy = np.c_[c * (x - x[0]) + sn * (y - y[0]), -sn * (x - x[0]) + c * (y - y[0])]
        got = lane_offsets(P, S, dict(x=x, y=y, yaw=h, path_s=s), 0, xy)
        assert np.abs(got - lat).max() < 0.05, (s0, lat, got.min(), got.max())


def select(a):
    from scripts.ft_synth_hist import path_arrays, project
    P, S, _ = path_arrays(); check_map(P, S)
    idx = pd.read_parquet(Path(a.root) / "_index/index.parquet", columns=["episode", "i", "split"]).astype({"episode": str})
    D = set(pd.read_parquet(Path(a.pools) / "r2_D.parquet").episode.astype(str))
    v = idx[idx.episode.isin(D) & idx.split.isin(["val", "val_asset"])]
    rows = []
    for ep, g in v.groupby("episode"):
        z = np.load(Path(a.root) / "_ego" / f"{ep}.npz")
        for i in g.i.to_numpy():
            _, o, h = project(P, S, float(z["x"][i]), float(z["y"][i]), float(z["path_s"][i]))
            rows.append(dict(episode=ep, i=int(i), o=o, yaw_err=float((z["yaw"][i] - h + np.pi) % (2 * np.pi) - np.pi), v0=float(z["v"][i]), bin=bin_of(o)))
    df = pd.DataFrame(rows); off = df[df.bin.notna() & (df.bin != "ctl")]; ctl = df[df.bin == "ctl"]
    ctl = ctl.iloc[np.sort(np.random.default_rng(0).choice(len(ctl), min(a.n_ctl, len(ctl)), replace=False))]
    sel = pd.concat([off, ctl]).sort_values(["episode", "i"]).reset_index(drop=True); sel.to_parquet(a.out, index=False)
    print(json.dumps({"val_rows": len(df), "episodes": int(df.episode.nunique()), "bins_all": df.bin.fillna("0.3-1").value_counts().to_dict(),
                      "selected": sel.groupby("bin").agg(n=("i", "size"), eps=("episode", "nunique")).to_dict("index")}, indent=1))


def run(a):
    import torch
    from morai_vla.data import MoraiDataset, hist_tensors
    from scripts.ft_measure import BF16
    sel = pd.read_parquet(a.sel).astype({"episode": str})
    ds = MoraiDataset(a.root, ["val", "val_asset"], frame_stride=1, augment=False, seed=1, keep=a.sel)
    ds.rows = ds.rows.astype({"episode": str}).merge(sel[["episode", "i", "v0"]], on=["episode", "i"]).sort_values(["episode", "i"]).reset_index(drop=True)
    model = BF16(a.ckpt, a.device); out = {c: [] for c in CONDS}; keys = []; t0 = time.time()
    for k in range(len(ds.rows)):
        s = ds.sample(k); seed = 1000 * k
        straight = hist_tensors(-float(ds.rows.v0.iloc[k]) * 0.1 * np.arange(15, -1, -1), np.zeros(16), np.zeros(16))
        black = dict(s, camera_frames=[torch.zeros_like(f) for f in s["camera_frames"]])
        out["real"].append(model(s, (s["ego_history_xyz"], s["ego_history_rot"]), seed))
        out["straight"].append(model(s, straight, seed)); out["nocam"].append(model(black, straight, seed)); keys.append(s["key"])
        if k % 20 == 0: print(f"{k + 1}/{len(ds.rows)} {s['key']} {(time.time() - t0) / (k + 1):.2f} s/row", flush=True)
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    np.savez(o / "preds.npz", key=np.array(keys), **{c: np.stack(v).astype(np.float32) for c, v in out.items()})
    (o / "run.json").write_text(json.dumps({"ckpt": a.ckpt, "root": a.root, "selected": len(sel), "run": len(keys),
                                            "s_per_row": round((time.time() - t0) / max(len(keys), 1), 3)}, indent=1))


def stats(g, c):
    """one condition (or gt) on the rows g of a bin."""
    o = g.o.to_numpy(); e = {"n": len(g), "eps": int(g.episode.nunique()), "abs_o": round(float(np.abs(o).mean()), 3)}
    for h in AT:
        d = g[f"{c}_{h}"].to_numpy(); e[f"abs_d_{h}"] = round(float(np.abs(d).mean()), 3)
        if np.abs(o).min() >= 1.0:
            r = 1 - np.abs(d) / np.abs(o); e[f"ret_{h}"] = round(float(r.mean()), 3); e[f"ret_{h}_med"] = round(float(np.median(r)), 3)
    e["slope_3s"] = round(float(((g[f"{c}_3s"].to_numpy() - o) * o).sum() / (o ** 2).sum()), 3)
    return e


def report(a):
    from morai_vla.data import ego_trajectory
    from scripts.ft_synth_hist import path_arrays
    P, S, _ = path_arrays(); sel = pd.read_parquet(a.sel).astype({"episode": str})
    preds = dict(np.load(a.preds)) if a.preds else None; conds = ("gt", "cv") + (CONDS if preds else ())
    if preds:
        pos = {k: j for j, k in enumerate(preds["key"])}
        sel = sel[[f"{e}/{i}" in pos for e, i in zip(sel.episode, sel.i)]].reset_index(drop=True)
    curves = {c: [] for c in conds}; ego = {}
    for e, i in zip(sel.episode, sel.i):
        z = ego.setdefault(e, dict(np.load(Path(a.ego) / f"{e}.npz")))
        curves["gt"].append(lane_offsets(P, S, z, i, ego_trajectory(z["v"], z["yaw"], i)[2][0, :, :2].numpy()))
        cv = np.c_[float(z["v"][i]) * 0.1 * np.arange(1, 65), np.zeros(64)]           # baseline: v0 straight on along the t0 heading
        curves["cv"].append(lane_offsets(P, S, z, i, cv))
        for c in conds[2:]: curves[c].append(lane_offsets(P, S, z, i, preds[c][pos[f"{e}/{i}"]]))
    curves = {c: np.stack(v) for c, v in curves.items()}
    for c in conds:
        for h, j in AT.items(): sel[f"{c}_{h}"] = curves[c][:, j]
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True); sel.to_csv(o / "rows.csv", index=False)
    np.savez(o / "lateral.npz", key=(sel.episode + "/" + sel.i.astype(str)).to_numpy(str), o=sel.o.to_numpy(), **curves)
    tw = -np.sign(sel.o) * np.degrees(sel.yaw_err); off = sel.bin != "ctl"          # t0 heading toward the centre (+)
    groups = [(n, sel.bin == n) for n, _, _ in BINS] + [("all>=1", off), ("all>=1 heading toward >5deg", off & (tw > 5)),
              ("all>=1 heading parallel", off & (tw.abs() <= 5)), ("all>=1 heading away >5deg", off & (tw < -5))]
    summ = {c: {n: stats(sel[m], c) for n, m in groups if m.any()} for c in conds}
    (o / "summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))
    if a.figs: figures(sel, curves, conds, o, a.figs)


def figures(sel, curves, conds, o, n_figs):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    col = {"gt": "k", "cv": "tab:gray", "real": "tab:blue", "straight": "tab:green", "nocam": "tab:red"}
    fig, ax = plt.subplots(1, len(conds), figsize=(3.6 * len(conds), 3.6), sharex=True, sharey=True, squeeze=False)
    for p, c in zip(ax[0], conds):
        p.scatter(sel.o, sel[f"{c}_3s"], s=6, c=col[c]); lim = [sel.o.min() - 0.3, sel.o.max() + 0.3]
        p.plot(lim, lim, "k:", lw=0.8, label="kept offset"); p.axhline(0, color="gray", lw=0.8, label="lane centre")
        p.set_title(c); p.set_xlabel("offset at t0 (m, left +)"); p.grid(alpha=0.3)
    ax[0, 0].set_ylabel("offset at 3 s (m)"); ax[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(o / "scatter_3s.png", dpi=110); plt.close(fig)
    t = 0.1 * np.arange(1, 65); picks = []
    for n, _, _ in BINS[1:][::-1]:                                   # largest offset of each episode, big bins first
        g = sel[sel.bin == n].assign(a=sel.o.abs()).sort_values("a", ascending=False).drop_duplicates("episode")
        picks += list(g.index[:max(1, n_figs // 3)])
    for k in picks[:n_figs]:
        r = sel.loc[k]; fig, p = plt.subplots(figsize=(6, 3.4))
        for c in conds: p.plot(np.r_[0, t], np.r_[r.o, curves[c][k]], color=col[c], lw=1.6 if c == "gt" else 1.2, label=c)
        p.axhline(0, color="gray", lw=0.8); p.set_xlabel("t (s)"); p.set_ylabel("offset to lane centre (m, left +)")
        p.set_title(f"{r.episode} i={r.i} o={r.o:+.2f} m yaw err {np.degrees(r.yaw_err):+.0f} deg v0 {r.v0:.1f} m/s", fontsize=9)
        p.grid(alpha=0.3); p.legend(fontsize=7); fig.tight_layout(); fig.savefig(o / f"lat_{r.episode}_{r.i}.png", dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="what", required=True)
    p = sp.add_parser("select"); p.add_argument("--root", required=True); p.add_argument("--out", required=True)
    p.add_argument("--pools", default=str(Path(__file__).resolve().parents[1] / "assets/ft")); p.add_argument("--n-ctl", type=int, default=80)
    p = sp.add_parser("run"); [p.add_argument(k, required=True) for k in ("--root", "--sel", "--ckpt", "--out")]; p.add_argument("--device", default="cuda:0")
    p = sp.add_parser("report"); [p.add_argument(k, required=True) for k in ("--ego", "--sel", "--out")]; p.add_argument("--preds"); p.add_argument("--figs", type=int, default=8)
    a = ap.parse_args(); {"select": select, "run": run, "report": report}[a.what](a)


if __name__ == "__main__":
    main()
