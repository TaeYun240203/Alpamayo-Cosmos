"""Fake DAgger spool for tests and dry runs (train/dagger/README.md): episodes in the actor's layout (rl_episode.py) with
engine wire frames at the real camera / BEV sizes (random JPEGs and scans, as rl.tiny_test: real token counts) and a
fake labeler's dagger_labels.npz. Decision times: 10 Hz of simulator time with jitter and, now and then, a late decision
(gap 0.3-0.5 s); every 5th decision has no label (some 4 Hz rows among them), 1 in 10 labels is the route fallback
(source 1), valid masks lose a random tail, the second 4 Hz row's label has no valid point. critic_obs (rl_critic.FIELDS
length 30): v 10 m/s, lane_offset 0.8 m on every 4th decision; labels carry signal (red with stop_d 30 m on every 6th row,
else green / none) and stop_d.
Usage (cwd train/): python -m dagger.fake <spool> --e2e_server <eval repo src/e2e_driver/server> [--actors 2 --episodes 3 --decisions 40]
"""
import argparse, io, json, os, time
from pathlib import Path

import numpy as np
from PIL import Image

from morai_vla.processor import BEV, CAMERAS
from dagger.spool import rows_4hz
from rl.episodes import e2e, write_json


def jpeg(rng, w, h):
    b = io.BytesIO(); Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(b, "JPEG", quality=90); return b.getvalue()


def episode(d, server, n=40, seed=0, version=0):
    """one fake episode dir; returns the number of decisions."""
    wire, _ = e2e(server)
    rng = np.random.default_rng(seed); d = Path(d); d.mkdir(parents=True, exist_ok=True)
    cams = [[jpeg(rng, w, h) for _ in range(n + 3)] for _, w, h in CAMERAS]
    scans = [(rng.integers(0, 2, (BEV[2], BEV[1])) * rng.integers(0, 255, (BEV[2], BEV[1]))).astype(np.uint8) for _ in range(2 * (n + 3))]
    dt = np.where(rng.random(n) < 0.08, rng.uniform(0.3, 0.5, n), rng.uniform(0.09, 0.11, n)); dt[0] = 0.0
    t = 100.0 + np.cumsum(dt)
    with open(d / "steps.jsonl", "w") as f:
        for i in range(n):
            req = dict(t0=float(t[i]), camera_jpegs=[cams[c][i:i + 4] for c in range(4)], camera_times=np.tile([-0.3, -0.2, -0.1, 0.0], (4, 1)),
                       nav_text="Follow the road. Speed limit 60 km/h", scan_height=np.stack(scans[2 * i:2 * i + 8:2]),
                       scan_density=np.stack(scans[2 * i + 1:2 * i + 8:2]), scan_pose=np.array([[j * 1.0, 0.0, 0.0] for j in range(i, i + 4)]),
                       bev_times=np.array([-0.1, 0.0]), ego_v=np.full(16, 10.0), ego_yaw=np.zeros(16), id=i, t_wall=time.time(), world=1)
            (d / f"{i:05d}.req.bin").write_bytes(wire.pack(req))
            (d / f"{i:05d}.out.bin").write_bytes(wire.pack(dict(expert=0, lora_version=None if version == 0 else str(version))))
            obs = [10.0] + [0.0] * 6 + [0.8 if i % 4 == 0 else 0.1] + [0.0] * 22
            f.write(json.dumps(dict(k=i, t_sim=round(float(t[i]), 3), req=f"{i:05d}.req.bin", out=f"{i:05d}.out.bin", r_time=-0.1,
                                    critic_obs=obs)) + "\n")
    write_json(d / "done.json", {"terminated": True, "truncated": False, "end": "goal"})
    step = np.array([i for i in range(n) if i % 5], np.int32)
    tt = np.arange(64) * 0.1
    ak = np.stack([np.stack([0.8 * np.sin(tt / 2 + s) * rng.uniform(0.2, 1.5), 0.01 * np.cos(tt / 3 + s)], -1) for s in step]).astype(np.float32)
    valid = np.arange(64)[None] < rng.integers(20, 65, len(step))[:, None]
    valid[np.searchsorted(step, next(i for i in rows_4hz(t)[1:] if i % 5))] = False
    src = (rng.random(len(step)) < 0.1).astype(np.int8)
    tmp = d / "dagger_labels.tmp.npz"                                    # the labeler writes it last, atomically
    sig = np.where(np.arange(len(step)) % 6 == 0, 1, 3).astype(np.int8); stop_d = np.where(sig == 1, 30.0, np.inf).astype(np.float32)
    np.savez(tmp, step=step, ak=ak, valid=valid, source=src, signal=sig, stop_d=stop_d); os.replace(tmp, d / "dagger_labels.npz")
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("spool"); p.add_argument("--e2e_server", default=os.environ.get("E2E_SERVER"), required=not os.environ.get("E2E_SERVER"))
    p.add_argument("--actors", type=int, default=2); p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--decisions", type=int, default=40); p.add_argument("--seed", type=int, default=0); p.add_argument("--start", type=int, default=0)
    a = p.parse_args()
    for k in range(a.actors):
        for e in range(a.start, a.start + a.episodes):
            episode(Path(a.spool) / f"actor{k}" / f"ep{e:05d}", a.e2e_server, a.decisions, a.seed * 1000 + 100 * k + e)
    print(f"{a.actors * a.episodes} fake episodes in {a.spool}")


if __name__ == "__main__":
    main()
