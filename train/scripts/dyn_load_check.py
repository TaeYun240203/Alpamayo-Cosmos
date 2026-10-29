"""CPU check after scripts/dyn_stage.py (box or winpc): the unchanged MoraiDataset reads n random staged virtual episodes of
_dyn_relabel/; their future path follows the synthetic speed (dyn_future.parquet), the history equals the original row's,
camera frames and BEVs load through the links. 10/28 on winpc: 35 of 48 rows usable (the rest fail MoraiDataset.usable
frame gaps, as ~20 % of all R2 A rows do), path error <= 1.5e-5 m. Run from train/ (morai_vla importable).
Usage: python scripts/dyn_load_check.py R2_ROOT [n] [--relabel-dir _dyn_relabel]   (pop-out v2 labels: _dyn_relabel_v2)
"""
import sys, tempfile
from pathlib import Path

import numpy as np, pandas as pd

from morai_vla.data import MoraiDataset, ego_trajectory


def main():
    rd = sys.argv[sys.argv.index("--relabel-dir") + 1] if "--relabel-dir" in sys.argv else "_dyn_relabel"
    pos = [x for k, x in enumerate(sys.argv[1:], 1) if x != "--relabel-dir" and sys.argv[k - 1] != "--relabel-dir"]
    root, n = Path(pos[0]), int(pos[1]) if len(pos) > 1 else 40
    vi = pd.read_parquet(root / rd / "dyn_index.parquet")
    vi = vi[[(root / e / "cams.bin").exists() for e in vi.episode]]
    if "kind" in vi: vi = vi[vi.kind != "restart_hold"]   # pop-out restart rows: their future is only in their ego table (dyn_future = the stop label)
    pick = vi.sample(min(n, len(vi)), random_state=0)
    keep = Path(tempfile.mkdtemp()) / "keep.parquet"; pick[["episode", "i"]].to_parquet(keep, index=False)
    ds = MoraiDataset(root, split=["train", "val", "val_asset"], index=f"{rd}/dyn_index.parquet", keep=str(keep), augment=False)
    F = pd.read_parquet(root / rd / "dyn_future.parquet").set_index(["episode", "i"])
    src = dict(zip(vi.episode, vi.src_episode)); worst = 0.0
    print(f"{len(ds)} of {len(pick)} staged rows usable")
    assert len(ds), "no usable virtual row"
    for k in range(len(ds)):
        r = ds.rows.iloc[k]; i = int(r.i); s = ds.sample(k); f = F.loc[(src[r.episode], i)]
        z = np.load(root / "_ego" / f"{src[r.episode]}.npz")
        pts = np.r_[np.zeros((1, 3)), s["ego_future_xyz"][0].numpy()]; v = np.r_[float(z["v"][i]), f.fut_v]
        err = np.abs(np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1)) - np.cumsum(np.abs(0.5 * (v[1:] + v[:-1]) * 0.1))).max()
        h = ego_trajectory(z["v"].astype(float), z["yaw"], i)[0][0].numpy()
        assert err < 1e-3, (r.episode, err)
        assert np.allclose(s["ego_history_xyz"][0].numpy(), h, atol=1e-5), (r.episode, "history differs")
        assert len(s["camera_frames"]) == 4 and ("bev_frames" in s or s["lidar_dropped"]), (r.episode, "sensors")
        worst = max(worst, err)
    print(f"OK: {len(ds)} virtual rows through the unchanged loader, path error max {worst:.1e} m, history and sensors as the original")


if __name__ == "__main__":
    main()
