"""Step 0, Mac side: per-episode ego table on the restored sim clock (10 Hz tau grid) as plain npz for the winpc
workers (numpy 1.17 in the Noetic chroot cannot read parquet). Usage: python ego_tables.py OUT_DIR [--jobs 10]"""
import argparse, glob, os, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coc_labeler"))
from coc_labeler import data   # noqa: E402

SIG = {"red": 1, "yellow": 2, "green": 3}


def one(args):
    ep, out = args
    Path(out).mkdir(parents=True, exist_ok=True); f = Path(out) / f"{ep}.npz"
    if f.exists(): return ep, "skip"
    try:
        E = data.load_episode(ep); G = E.G
        env = [E.env(i) for i in range(0, len(G), 50)]                         # weather/hour switch every few minutes
        envg = np.repeat(env, 50)[:len(G)]
        np.savez(f.with_suffix(".tmp.npz"), tau=G.tau.to_numpy(np.float64), bag_ns=G.bag_ns.to_numpy(np.int64),
                 x=G.x.to_numpy(np.float64), y=G.y.to_numpy(np.float64), yaw=G.yaw.to_numpy(np.float64),
                 v=G.v.to_numpy(np.float32), a=G.a.to_numpy(np.float32), yaw_rate=G.yaw_rate.to_numpy(np.float32),
                 path_s=G.path_s.to_numpy(np.float32), d=G.d.to_numpy(np.float32), section=G.route_section.to_numpy(np.int16),
                 signal=np.array([SIG.get(str(s), 0) for s in G.signal], np.int8), gps_blackout=G.gps_blackout.to_numpy(bool),
                 clean=G.clean.to_numpy(bool), fog=np.array([e.startswith("F") for e in envg]), hour=np.array([int(e[1:]) if e[1:].isdigit() else -1 for e in envg], np.int8),
                 scene=np.array(E.scene_id or ""), worker=np.array(E.worker))
        os.replace(f.with_suffix(".tmp.npz"), f)
        return ep, "ok"
    except Exception as e:
        return ep, f"FAILED {type(e).__name__}: {e}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--only", help="file with the episode names to do (R2: the QA-ok episodes)")
    a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    eps = sorted(os.path.basename(f).split(".")[0] for f in glob.glob(str(data.LABELS / "*" / "*.labels.parquet")))
    if a.only:
        keep = set(open(a.only).read().split()); eps = [e for e in eps if e in keep]
    with Pool(a.jobs) as p:
        for k, (ep, st) in enumerate(p.imap_unordered(one, [(e, a.out) for e in eps])):
            if st != "skip": print(k, ep, st, flush=True)
