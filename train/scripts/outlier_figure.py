"""One row per sample: Front / Front-left / Front-right camera at t0 (model input sizes) and a top-down plot in the t0 ego
frame (history, GT future, predicted future, the route centerline from the rule context), titled with key, cell, nav
text, GT speed and |lateral| at 1/2/3 s. Samples: C5 rows whose max |lateral| over the first 2 s exceeds 0.5 m, plus the
--worst worst rows overall, from scripts/lateral_error.py's npz.
Usage (cwd train/): python scripts/outlier_figure.py <lateral.npz> <step-0 root> --out logs/stageC/c5_outliers.png"""
import argparse
from pathlib import Path

import numpy as np

from morai_vla.data import MoraiDataset
from scripts.lateral_error import split_error

CTX = Path(__file__).resolve().parents[1] / "assets/rule_context.npz"

ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("root"); ap.add_argument("--out", required=True)
ap.add_argument("--worst", type=int, default=3); ap.add_argument("--cell", default="C5")
a = ap.parse_args(); d = np.load(a.npz, allow_pickle=True); route = np.load(CTX)["path"]
lat, _ = split_error(d["pred"], d["gt"], d["gt_yaw"]); m2 = np.abs(lat[:, :20]).max(1)
sel = list(np.nonzero((d["cell"] == a.cell) & (m2 > 0.5))[0])
sel += [k for k in np.argsort(-m2) if k not in sel][:a.worst]
raw = MoraiDataset(a.root, ["val", "val_asset"], frame_stride=10, max_samples=1000, augment=False, seed=1)
raw.rows = raw.rows.iloc[:len(d["pred"])].reset_index(drop=True)
assert all(f"{e}/{i}" == k for e, i, k in zip(raw.rows.episode, raw.rows.i, d["key"])), "npz rows != dataset rows"
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(len(sel), 4, figsize=(20, 3.6 * len(sel)), squeeze=False, gridspec_kw={"width_ratios": [1.75, 1.375, 1.375, 1.2]})
for r, k in enumerate(sel):
    s = raw.sample(int(k)); ep, i = raw.rows.episode.iloc[k], int(raw.rows.i.iloc[k]); e = np.load(Path(a.root) / "_ego" / f"{ep}.npz")
    for c in range(3):
        ax[r, c].imshow(s["camera_frames"][c][-1].permute(1, 2, 0).numpy()); ax[r, c].axis("off")
    x0, y0, h0 = e["x"][i], e["y"][i], e["yaw"][i]; dx, dy = route[:, 0] - x0, route[:, 1] - y0
    rx, ry = np.cos(h0) * dx + np.sin(h0) * dy, -np.sin(h0) * dx + np.cos(h0) * dy; near = (np.abs(rx) < 80) & (np.abs(ry) < 40)
    hx = s["ego_history_xyz"][0, :, :2].numpy()
    A = ax[r, 3]; A.plot(-ry[near], rx[near], ".", ms=1, color="0.7", label="route")
    A.plot(-hx[:, 1], hx[:, 0], "C0", label="history"); A.plot(-d["gt"][k, :, 1], d["gt"][k, :, 0], "k", label="GT")
    A.plot(-d["pred"][k, :, 1], d["pred"][k, :, 0], "C3", label="pred"); A.set_aspect("equal"); A.legend(fontsize=7)
    L = max(15.0, float(np.abs(np.r_[d["gt"][k].ravel(), d["pred"][k].ravel()]).max()) * 1.1); A.set_xlim(-L / 2, L / 2); A.set_ylim(-0.25 * L, L)
    ax[r, 0].set_title(f"{ep}/{i} (tau {e['tau'][i]:.1f} s) {d['cell'][k]}  v={e['v'][i] * 3.6:.0f} km/h  |lat| 1/2/3 s: "
                       f"{abs(lat[k, 9]):.2f}/{abs(lat[k, 19]):.2f}/{abs(lat[k, 29]):.2f} m\n{s['nav_text']}", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(a.out, dpi=70); print("rows:", [str(d["key"][k]) for k in sel])
