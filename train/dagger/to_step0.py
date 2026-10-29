"""DAgger spool -> step-0 rows for the regular fine-tune (morai_vla.data.MoraiDataset), so shadow-labelled rows can join a
fine-tune / LwF mix as a GT source by config only (reports/ring_dagger_plan_20261029.md). The online runner reads the
spool itself (spool.py); this is the offline path. Rows are spool.episode_rows' (4 Hz, labelled), kept only when the
label is valid at all 64 points (the GT loss has no valid mask).

  ROOT/<ep>/                         cams.bin + cams_index.npz, lidar.bin + lidar_index.npz, meta.json: every camera
                                     frame and LiDAR scan of the episode's engine requests (what the model saw), one per
                                     stamp, at the requests' own stamps (the request t0 clock)
  ROOT/_dagger/ego/<ep>_dg<j>.npz    per row: the 10 Hz grid tau, v, yaw at rows 0..79 with t0 at row I0 = 15; history =
                                     the request's ego_v / ego_yaw (the actor's real history), future = the label's
                                     (a, kappa) integrated as the action space does (future())
  ROOT/_dagger/rows/<ep>.parquet     the episode's index rows; ROOT/_dagger/dagger_index.parquet = all of them: episode
                                     (virtual <ep>_dg<j>), i = I0, cell, split (by episode hash), nav_text, src_episode, j,
                                     s (route s), source, tau
  ROOT/<ep>_dg<j>/, ROOT/_ego/<ep>_dg<j>.npz   links to ROOT/<ep>/ and _dagger/ego/ (as scripts/dyn_stage.py; B2 keeps
                                     no links: --stage re-makes them after a download)
Source config: dataset: {root: ROOT, index: _dagger/dagger_index.parquet}, loss gt.
Usage (cwd train/): python -m dagger.to_step0 ROOT EP_DIR|SPOOL [...] --e2e_server <eval repo src/e2e_driver/server>
                        [--s-range 850 935] [--cell dagger_ring] [--val 0.05]
                    python -m dagger.to_step0 ROOT --stage
"""
import argparse, hashlib, json, os, zlib
from pathlib import Path

import numpy as np, pandas as pd

from dagger.spool import episode_rows, episodes, ready

N_HIST, N_FUT, DT, I0 = 16, 64, 0.1, 15                         # I0 % 5 == 0: fine-tune sources keep 2 Hz rows only
SENSORS = ("cams.bin", "cams_index.npz", "lidar.bin", "lidar_index.npz", "meta.json")


def future(ak, v0, yaw0, dt=DT):
    """(a m/s^2, kappa 1/m) (64, 2) -> v, yaw at t0 + 0.1 .. 6.4 s, integrated as UnicycleAccelCurvatureActionSpace.action_to_traj."""
    a, k = np.asarray(ak, np.float64).T
    v = v0 + np.cumsum(a * dt)
    return v, yaw0 + np.cumsum(k * np.r_[v0, v[:-1]] * dt + k * a * dt * dt / 2)


def name_of(d):
    return f"dg_{Path(d).parent.name}_{Path(d).name}"


def sensors(d, steps, js, wire, out):
    """the requests of decisions js -> out/ sensor files; -> {j: request header (t0, nav_text, ego_v, ego_yaw)}."""
    out.mkdir(parents=True, exist_ok=True)
    cams, scans, head = {}, {}, {}
    with open(out / "cams.bin", "wb") as fc, open(out / "lidar.bin", "wb") as fl:
        for j in js:
            r = wire.unpack((d / steps[j]["req"]).read_bytes()); t0 = float(r["t0"])
            head[j] = dict(t0=t0, nav_text=str(r["nav_text"]), ego_v=np.asarray(r["ego_v"], np.float64), ego_yaw=np.asarray(r["ego_yaw"], np.float64))
            stamps = r["camera_stamps"] if r.get("camera_stamps") is not None else t0 + np.asarray(r["camera_times"], np.float64)
            for c, (jp, ts) in enumerate(zip(r["camera_jpegs"], stamps)):
                for b, t in zip(jp, ts):
                    if (c, round(float(t), 4)) not in cams:
                        cams[(c, round(float(t), 4))] = (float(t), fc.tell(), len(b)); fc.write(b)
            if "scan_height" in r:
                ss, known = r.get("scan_stamps"), [True] * 4
                if ss is None:   # shortcut: requests before scan_stamps carry each pair's newer scan only; the older one is put half the pair spacing before it
                    bt = t0 + np.asarray(r["bev_times"], np.float64); h = (bt[1] - bt[0]) / 2
                    ss, known = [bt[0] - h, bt[0], bt[1] - h, bt[1]], [False, True, False, True]
                for h, dd, p, t, kn in zip(r["scan_height"], r["scan_density"], r["scan_pose"], ss, known):
                    raw = np.ascontiguousarray(np.stack([h, dd]), np.uint8).tobytes(); key = hashlib.sha1(raw).digest()   # one scan, one entry
                    if key not in scans:
                        b = zlib.compress(raw); scans[key] = [float(t), fl.tell(), len(b), np.asarray(p, np.float64), kn]; fl.write(b)
                    elif kn and not scans[key][4]:
                        scans[key][0], scans[key][4] = float(t), True
    cv, sv = list(cams.items()), sorted(scans.values(), key=lambda x: x[0])
    np.savez(out / "cams_index.npz", cam=np.array([k[0] for k, _ in cv], np.int8), tau=np.array([v[0] for _, v in cv]),
             offset=np.array([v[1] for _, v in cv], np.int64), length=np.array([v[2] for _, v in cv], np.int64))
    np.savez(out / "lidar_index.npz", tau=np.array([s[0] for s in sv]), offset=np.array([s[1] for s in sv], np.int64),
             length=np.array([s[2] for s in sv], np.int64), pose=np.array([s[3] for s in sv]).reshape(-1, 3))
    (out / "meta.json").write_text(json.dumps(dict(source=str(d), frames=len(cv), scans=len(sv), by="dagger.to_step0")))
    return head


def convert(d, root, wire, s_range=None, cell="dagger", val=0.05):
    """one ready spool episode -> ROOT files (module docstring); -> stats."""
    d, root = Path(d), Path(root)
    steps = [json.loads(line) for line in open(d / "steps.jsonl")]
    got = episode_rows(d, wire)
    if got is None:
        return dict(episode=str(d), skipped="req files still syncing")
    rows, st = got
    st["partial_valid"] = sum(not r["valid"].all() for r in rows)
    rows = [r for r in rows if r["valid"].all() and (s_range is None or s_range[0] <= steps[r["j"]].get("s", np.nan) <= s_range[1])]
    st["kept"] = len(rows)
    if not rows:
        return st
    ep = name_of(d); t = np.array([s["t_sim"] for s in steps])
    near = np.flatnonzero(np.min(np.abs(t[:, None] - t[[r["j"] for r in rows]][None]), 1) <= 0.6)   # frames around every row
    head = sensors(d, steps, near, wire, root / ep)
    (root / "_dagger/ego").mkdir(parents=True, exist_ok=True); (root / "_dagger/rows").mkdir(exist_ok=True)
    split = "val" if int(hashlib.sha1(ep.encode()).hexdigest()[:8], 16) / 16 ** 8 < val else "train"
    out = []
    for r in rows:
        h = head[r["j"]]; v, yaw = future(r["ak"], h["ego_v"][-1], h["ego_yaw"][-1])
        name = f"{ep}_dg{r['j']:05d}"
        np.savez_compressed(root / "_dagger/ego" / f"{name}.npz", tau=h["t0"] + DT * (np.arange(N_HIST + N_FUT) - I0),
                            v=np.r_[h["ego_v"], v], yaw=np.r_[h["ego_yaw"], yaw])
        out.append(dict(episode=name, i=I0, cell=cell, split=split, nav_text=h["nav_text"], src_episode=ep, j=r["j"],
                        s=float(steps[r["j"]].get("s", np.nan)), source=r["source"], tau=h["t0"]))
    pd.DataFrame(out).to_parquet(root / "_dagger/rows" / f"{ep}.parquet", index=False)
    return st


def stage(root):
    """links of every virtual episode in the index (idempotent); -> number staged."""
    root = Path(root); (root / "_ego").mkdir(exist_ok=True)
    vi = pd.read_parquet(root / "_dagger/dagger_index.parquet", columns=["episode", "src_episode"])
    for v, src in zip(vi.episode, vi.src_episode):
        (root / v).mkdir(exist_ok=True)
        for f in SENSORS:
            if (root / src / f).exists() and not (root / v / f).is_symlink(): (root / v / f).symlink_to(Path("..") / src / f)
        e = root / "_ego" / f"{v}.npz"
        if not e.is_symlink(): e.symlink_to(Path("../_dagger/ego") / f"{v}.npz")
    return len(vi)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root"); p.add_argument("eps", nargs="*", help="spool episode dirs or spool roots (<spool>/<actor>/epK)")
    p.add_argument("--e2e_server", default=os.environ.get("E2E_SERVER")); p.add_argument("--stage", action="store_true")
    p.add_argument("--s-range", nargs=2, type=float, help="route s window of the rows kept (e.g. the ring entry 850 935)")
    p.add_argument("--cell", default="dagger"); p.add_argument("--val", type=float, default=0.05, help="episode share in split val")
    a = p.parse_args()
    if not a.stage:
        from rl.episodes import e2e
        wire = e2e(a.e2e_server)[0]
        for e in a.eps:
            for d in ([Path(e)] if ready(e) else episodes(e)):
                print(json.dumps(convert(d, a.root, wire, a.s_range, a.cell, a.val), default=int), flush=True)
        parts = sorted((Path(a.root) / "_dagger/rows").glob("*.parquet"))
        assert parts, "no trainable rows"
        pd.concat([pd.read_parquet(f) for f in parts]).to_parquet(Path(a.root) / "_dagger/dagger_index.parquet", index=False)
    print(f"to_step0: {stage(a.root)} rows staged in {a.root}")


if __name__ == "__main__":
    main()
