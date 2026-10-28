"""A tiny synthetic step-0 root in the real layout (data.py docstring) for CPU checks of the R2 fine-tune: old and R2
episodes, flat-colour JPEGs at the real camera sizes, empty-ish LiDAR scans, _ego tables, _index/index.parquet, and the
fine-tune pools (scripts/ft_pools.py names) + a start relabel parquet + an s2-style rule_ok parquet.
Old episodes stand still for 3 s, then accelerate to 8 m/s on a gentle left curve; R2 episodes start moving at once.
Usage: python tests/ft_synth.py <out root>   (also importable: make(root))"""
import io, sys, zlib
from pathlib import Path

import numpy as np, pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from morai_vla.processor import CAMERAS                                         # noqa: E402

N = 120                                                                           # grid rows per episode (12 s)
EPISODES = {"old-001": ("old", "train"), "old-002": ("old", "train"), "old-003": ("old", "train"), "old-004": ("old", "val"),
            "r2-a-001": ("ra", "train"), "r2-a-002": ("ra", "train"), "r2-a-003": ("ra", "val"),
            "r2-b-001": ("rb", "train"), "r2-b-002": ("rb", "val"), "r2-d-001": ("rd", "train"), "r2-d-002": ("rd", "val")}
STAND = 30                                                                        # old episodes: rows standing still


def jpeg(w, h, c):
    b = io.BytesIO(); Image.new("RGB", (w, h), tuple(int(x) for x in c)).save(b, "JPEG", quality=40); return b.getvalue()


def episode(d, kind, seed):
    rng = np.random.default_rng(seed); d.mkdir(parents=True, exist_ok=True)
    tau = np.arange(N) * 0.1
    v = np.clip((np.arange(N) - (STAND if kind == "old" else -40)) * 0.2, 0, 8.0)
    yaw = np.cumsum(v * 0.1 * 0.01)                                              # curvature 0.01 1/m
    cam, ct, off, ln, blob = [], [], [], [], bytearray()
    for k in range(N):
        for c, (_, w, h) in enumerate(CAMERAS):
            j = jpeg(w, h, rng.integers(0, 255, 3)); cam.append(c); ct.append(tau[k] + 0.003 * c); off.append(len(blob)); ln.append(len(j)); blob += j
    (d / "cams.bin").write_bytes(bytes(blob))
    np.savez(d / "cams_index.npz", cam=np.array(cam), tau=np.array(ct), offset=np.array(off), length=np.array(ln))
    soff, sln, sblob = [], [], bytearray()
    for k in range(N):
        s = np.zeros((2, 320, 160), np.uint8); s[:, rng.integers(0, 320, 50), rng.integers(0, 160, 50)] = 30
        z = zlib.compress(s.tobytes()); soff.append(len(sblob)); sln.append(len(z)); sblob += z
    (d / "lidar.bin").write_bytes(bytes(sblob))
    pose = np.stack([np.cumsum(v * 0.1 * np.cos(yaw)), np.cumsum(v * 0.1 * np.sin(yaw)), yaw], 1)
    np.savez(d / "lidar_index.npz", tau=tau + 0.05, offset=np.array(soff), length=np.array(sln), pose=pose)
    return dict(tau=tau, v=v.astype(np.float32), yaw=yaw)


def make(root):
    root = Path(root); (root / "_ego").mkdir(parents=True, exist_ok=True); (root / "_index").mkdir(exist_ok=True)
    rows = []
    for j, (ep, (kind, split)) in enumerate(EPISODES.items()):
        np.savez(root / "_ego" / f"{ep}.npz", **episode(root / ep, kind, j))
        for i in range(15, N - 64):                                               # 1.5 s history, 6.4 s future
            rows.append((ep, i, i * 0.1, {"old": "C0", "ra": "C3", "rb": "C6", "rd": "C0"}[kind], split, "Follow the road. Speed limit 60 km/h"))
    df = pd.DataFrame(rows, columns=["episode", "i", "tau", "cell", "split", "nav_text"])
    for c in ("episode", "cell", "split", "nav_text"): df[c] = df[c].astype("category")
    df.to_parquet(root / "_index/index.parquet", index=False)
    ft = root / "_ft"; ft.mkdir(exist_ok=True)
    eps = lambda kinds: pd.DataFrame({"episode": [e for e, (k, _) in EPISODES.items() if k in kinds]})
    for name, kinds in (("r2_A", ["ra"]), ("r2_B", ["rb"]), ("r2_D", ["rd"]), ("r2_AB", ["ra", "rb"])): eps(kinds).to_parquet(ft / f"{name}.parquet")
    old = df[df.episode.astype(str).str.startswith("old")]
    old[old.i % 3 == 0][["episode", "i"]].to_parquet(ft / "old_recovery.parquet")
    old[old.i % 2 == 0][["episode", "i"]].to_parquet(ft / "old_curve.parquet")
    old[["episode", "i"]].to_parquet(ft / "old_static.parquet")
    st = old[old.i < STAND][["episode", "i"]].assign(src_i=STAND)                # standing rows -> the drive from row STAND
    st.to_parquet(ft / "start_relabel.parquet")
    old.assign(rule_ok=(old.i % 10) < 7)[["episode", "i", "rule_ok"]].to_parquet(root / "rule_ok.parquet")
    # lane-shaped histories (scripts/ft_synth_hist.py layout): straight at the row's speed for the old recovery rows and
    # r2-d-001, plus D start rows i 3..14 (extra) that the index leaves out
    sy, eg = [], {e: np.load(root / "_ego" / f"{e}.npz") for e in {*old.episode.astype(str), "r2-d-001"}}
    back = 0.1 * np.arange(15, -1, -1, dtype=np.float32)
    for e, i, x in [(e, i, False) for e, i in zip(old[old.i % 3 == 0].episode.astype(str), old[old.i % 3 == 0].i)] + \
                   [("r2-d-001", i, False) for i in range(15, N - 64)] + [("r2-d-001", i, True) for i in range(3, 15)]:
        r = dict(episode=e, i=int(i), method="straight", hist_x=-float(eg[e]["v"][i]) * back, hist_y=np.zeros(16, np.float32),
                 hist_yaw=np.zeros(16, np.float32), extra=x)
        sy.append(r | (dict(tau=i * 0.1, cell="C0", split="train", nav_text="Follow the road. Speed limit 60 km/h") if x else {}))
    pd.DataFrame(sy).to_parquet(ft / "synth_hist.parquet", index=False)
    return root


if __name__ == "__main__":
    print(make(sys.argv[1]))
