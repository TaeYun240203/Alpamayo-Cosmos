"""Synthetic labels for the near pop-out collection (reports/popout_collect_plan_20261029.md), on the box per batch.
The driver followed the route only (no yield/AEB) and the episode ended at a collision, so every label here is synthetic:
  stop rows (soft/hard/hold, popout_* when the mover started behind a parked car): dyn_relabel.one() unchanged (braking
    from the first row where the mover is moving, seen and on a collision course; GT start of motion, not the trigger;
    standing rows keep the original drive = "do not stop for a person who just stands").
    Tracks: the scene's straight constant-speed path (scenario_variants popout) with tau_start = the first Object Info frame
    the mover is 0.2 m off its start (objects.json.gz), not the manifest + r2.log of R2.
    The future past a collision (or the episode end) is the route at the last speed (user 10/29 "RDDF"): the _ego table
    is extended by 64 rows on the route before labelling, sensors stay up to t0.
  Seen (--onset): lidar (default, user 10/29 "LiDAR 기준으로"): dyn_relabel's 2D LiDAR-ray test. camera (the 10/29 audit,
    reports/popout_audit_20261029.md: the ray test calls a mover seen when one box corner clears the parked car): the mover's
    Object Info box on the Front input (448x256, nearer vehicle / obstacle boxes cut out, the audit's projection) >= --vis-px
    unoccluded px and >= --vis-h px tall in the frame nearest the row, after >= --vis-moving s of motion; braking starts at
    the first such row on a collision course, earlier rows keep the original drive, a mover never seen gives no stop rows,
    an event whose first stop row is not visible (hidden again) starts at its first visible stop row.
    Unavoidable events (a_need > A_MAX) keep the A_MAX label (user 10/29).
  nav_text / cell / fog / hour of a virtual row: build_index's rules on the source row's own _ego row (no fallback).
  restart_hold (user 10/29 "정지이력에서 재출발"; off since the 10/29 audit, --restart): rows where the mover is 2-10 m ahead
    of the nose and in or entering the corridor (route-frame |lat - d| <= 1.8 m): history standing (v 0 at i-15 .. i), label
    0 until the mover is out of the corridor + RESTART_WAIT, then RESTART_A up to the row's original speed; the mover's
    future from its planned path.
Out (OUT): dyn_relabel_index.parquet (rows + kind), dyn_future.parquet, dyn_index.parquet (virtual episodes' index rows,
src_episode, kind), ego/<virtual>.npz, keep.parquet (the batch's original index rows minus the relabelled ones),
summary.json. Merge of batches and dyn_stage/dyn_load_check: the _dyn_relabel layout (popout_box.sh).
summary.json episodes[ep].first_stop_i: the first stop row dyn_relabel.one gave (before any row filter; popout_mix cruise /
pre_onset rows end before it).
Usage: python train/scripts/popout_relabel.py S0 INDEX_DIR OUT EP... [--onset lidar|camera --vis-px 30 --vis-h 8
  --vis-moving 0.3 --restart] (S0: step0_popout with _ego/, _popout/<ep>/, camera: <ep>/cams_index.npz); all episodes:
  scripts/popout_rebuild.sh
"""
import argparse, gzip, json, math, sys
from pathlib import Path

import numpy as np, pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import dyn_relabel as dr  # noqa: E402
from morai_vla.build_index import maneuvers, nav_text  # noqa: E402

dr.self_check = lambda *a: None                           # rows are checked below (one bad row must not drop the episode)
dr.A_MAX = 6.96                                           # R2 driver AEB median (reports/dyn_relabel_20261028.md, summary.json)
N, DT, CORR = dr.N, dr.DT, dr.HALF_W + dr.PAD + dr.R_OBJ  # corridor half width 1.8 m
COLS = ["episode", "i", "tau", "cell", "split", "nav_text", "fog", "hour", "scene"]
W, H = 448, 256                                           # Front input; 1280x720 HFOV 90 resized as step 0 (popout audit a3_vis.py)
FX, FY = 640 * W / 1280, 640 * H / 720
CX_M, CZ_M, PITCH = 1.90, 1.20, math.radians(2.0)         # tools/start_relabel/samples.py (bridge config + MORAI GUI)


def _box(x, y, hdg_deg, L, Wd, Ht):
    h = math.radians(hdg_deg); u = np.array([math.cos(h), math.sin(h)]); v = np.array([-u[1], u[0]])
    return np.array([np.r_[np.array([x, y]) + a * L / 2 * u + b * Wd / 2 * v, zz] for a in (-1, 1) for b in (-1, 1) for zz in (0.0, Ht)])


def _project(P, cam):
    cx, cy, yaw = cam; dx, dy = P[:, 0] - cx, P[:, 1] - cy
    fwd = math.cos(yaw) * dx + math.sin(yaw) * dy; left = -math.sin(yaw) * dx + math.cos(yaw) * dy; up = P[:, 2] - CZ_M
    depth = fwd * math.cos(PITCH) - up * math.sin(PITCH); upc = fwd * math.sin(PITCH) + up * math.cos(PITCH)
    return np.stack([W / 2 - FX * left / depth, H / 2 - FY * upc / depth], 1), depth


def _mask(uv):
    from matplotlib.path import Path as MPath           # camera onset only (boxes need no matplotlib for lidar)
    from scipy.spatial import ConvexHull
    lo = np.clip(np.floor(uv.min(0)).astype(int), 0, [W, H]); hi = np.clip(np.ceil(uv.max(0)).astype(int), 0, [W, H])
    m = np.zeros((H, W), bool)
    if (hi <= lo).any(): return m
    try: hull = uv[ConvexHull(uv).vertices]
    except Exception: return m
    sub = np.stack(np.meshgrid(np.arange(lo[0], hi[0]) + 0.5, np.arange(lo[1], hi[1]) + 0.5), -1).reshape(-1, 2)
    m[lo[1]:hi[1], lo[0]:hi[0]] = MPath(hull).contains_points(sub).reshape(hi[1] - lo[1], hi[0] - lo[0])
    return m


def front_vis(ep, S0, z, rec):
    """(mover n, ego tau) -> (unoccluded px, height px) of the mover in the Front frame nearest tau (the row's newest frame):
    its Object Info box projected into the Front input, nearer vehicle / obstacle boxes (Object Info types 1, 2) cut out."""
    fr = json.load(gzip.open(S0 / "_popout" / ep / "objects.json.gz", "rt"))["frames"]
    te = np.interp(np.array([f[1] for f in fr], float) * 1e9, z["bag_ns"].astype(float), z["tau"])
    ci = np.load(S0 / ep / "cams_index.npz"); names = [str(x) for x in ci["cam_names"]]
    ft = np.sort(ci["tau"][ci["cam"] == (names.index("front") if "front" in names else 0)].astype(float))
    yaw_u = np.unwrap(z["yaw"].astype(float)); gt = []
    for a in (x for x in rec["scene_meta"]["actors"] if x.get("ptype") == "popout"):   # the mover's id as tracks_from
        x0, y0 = a["xy"]; oid, g = None, []
        for t, (_, _, objs) in zip(te, fr):
            if not objs: continue
            if oid is None:
                c = min(objs, key=lambda q: math.hypot(q[2] - x0, q[3] - y0))
                if math.hypot(c[2] - x0, c[3] - y0) > 1.5: continue
                oid = c[0]
            q = next((q for q in objs if q[0] == oid), None)
            if q is not None: g.append((t, q[2], q[3], q[5], q[6], q[7], q[8]))
        gt.append((oid, np.array(g) if g else np.zeros((0, 7))))

    def vis(n, tau):
        oid, g = gt[n]
        if not len(g) or len(ft) < 2: return 0, 0
        j = int(np.clip(np.searchsorted(ft, tau), 1, len(ft) - 1)); t = ft[j - 1] if abs(ft[j - 1] - tau) <= abs(ft[j] - tau) else ft[j]
        yaw = float(np.interp(t, z["tau"], yaw_u))
        cam = (float(np.interp(t, z["tau"], z["x"])) + CX_M * math.cos(yaw), float(np.interp(t, z["tau"], z["y"])) + CX_M * math.sin(yaw), yaw)
        b = g[int(np.argmin(np.abs(g[:, 0] - t)))]
        uv, dep = _project(_box(b[1], b[2], b[3], max(b[4], 0.3), max(b[5], 0.3), b[6] if b[6] > 0.2 else 1.7), cam)
        if (dep <= 0.3).any(): return 0, 0
        m, dist = _mask(uv), math.hypot(b[1] - cam[0], b[2] - cam[1])
        for q in fr[int(np.argmin(np.abs(te - t)))][2]:
            if q[0] == oid or q[1] not in (1, 2): continue
            uvb, depb = _project(_box(q[2], q[3], q[5], q[6], q[7], q[8]), cam); d_b = math.hypot(q[2] - cam[0], q[3] - cam[1])
            if (depb <= 0.3).any() or d_b > dist + 2 or depb.min() >= dep.max(): continue
            if d_b < dist: m &= ~_mask(uvb)
        rows = np.flatnonzero(m.any(1))
        return int(m.sum()), int(rows[-1] - rows[0] + 1) if len(rows) else 0
    return vis


def collision_ns(L, rec):
    """host-clock ns of the first scorer collision with a real object (main 10/29: some boxes report object -100 contacts
    from the start while the car drives on; those are no collision), None without one."""
    t = [e["t_start"] for e in rec.get("scorer", {}).get("events", []) if e["type"] == "collision" and str(e.get("object")) != "-100"]
    return float(np.interp(min(t), L.t.to_numpy(float), L.bag_time_ns.to_numpy(float))) if t else None


def extend(z, n_add):
    """the _ego table + n_add rows on the route at the last speed (path_s advances, d held, heading of the route)."""
    z = {k: np.asarray(v) for k, v in z.items()}
    n = len(z["tau"]); v = float(max(z["v"][-1], 0.0)); ds = v * DT * np.arange(1, n_add + 1)
    s = float(z["path_s"][-1]) + ds; d = float(z["d"][-1])
    pts = [dr.data.route_point(q, 0.0) for q in s]
    h = np.unwrap(np.r_[z["yaw"][-1], [p[2] for p in pts]])[1:]
    add = dict(tau=z["tau"][-1] + DT * np.arange(1, n_add + 1), v=np.full(n_add, v), a=np.zeros(n_add), yaw_rate=np.zeros(n_add),
               path_s=s, d=np.full(n_add, d), yaw=h, x=np.array([p[0] for p in pts]), y=np.array([p[1] for p in pts]),
               bag_ns=z["bag_ns"][-1] + (1e8 * np.arange(1, n_add + 1)).astype(np.int64))
    out = {}
    for k, a in z.items():
        if a.ndim == 1 and len(a) == n:
            out[k] = np.r_[a, add[k].astype(a.dtype)] if k in add else np.r_[a, np.repeat(a[-1:], n_add)]
        else:
            out[k] = a
    return out


def tracks_from(ep, S0, z):
    """scene movers -> dyn_relabel track dicts; tau_start from Object Info (first frame 0.2 m off the start)."""
    rec = json.load(open(S0 / "_popout" / ep / "episode.json"))
    fr = json.load(gzip.open(S0 / "_popout" / ep / "objects.json.gz", "rt"))["frames"]
    out = []
    for n, a in enumerate(x for x in rec["scene_meta"]["actors"] if x.get("ptype") == "popout"):
        x0, y0 = a["xy"]; t0 = None; oid = None
        for tau, tw, objs in fr:
            if not objs: continue
            if oid is None:
                o = min(objs, key=lambda o: math.hypot(o[2] - x0, o[3] - y0))
                if math.hypot(o[2] - x0, o[3] - y0) > 1.5: continue
                oid = o[0]
            o = next((o for o in objs if o[0] == oid), None)
            if o is not None and math.hypot(o[2] - x0, o[3] - y0) > 0.2:
                t0 = float(np.interp(tw * 1e9, z["bag_ns"], z["tau"])) - 0.2 / (a["kph"] / 3.6); break
        out.append(dict(episode=ep, n=n, disp_k=None, cls=a["cls"], ptype="behind_parked" if a["start"] == "hidden" else a["start"],
                        timing=a["timing"], s0=a["s"], l0=a["lat"], s1=a["s_end"], l1=a["lat_end"], kph=a["kph"], move=a["move"],
                        active=a["active"], tau_start=t0, start=a["start"], zone=a["zone"]))
    return out, rec


def restart_rows(ep, z, n0, trk, idx_ep):
    """restart_hold rows and labels (see the module doc)."""
    res, fut = [], []
    v, yaw, s_e, d_e = (z[k].astype(float) for k in ("v", "yaw", "path_s", "d"))
    tp = DT * np.arange(1, N + 1)
    for o in trk:
        if o["tau_start"] is None: continue
        for i in range(15, n0, 2):                                     # every other row: neighbours are near duplicates
            s, l, _, _, started = (q[0] for q in dr.obj_state(o, np.array([z["tau"][i]])))
            gap = s - dr.R_OBJ - (s_e[i] + dr.FRONT)
            if not started or not 2.0 <= gap <= 10.0: continue
            sf, df = s_e[i:i + N + 1], d_e[i:i + N + 1]
            sp, lp, _, _, _ = dr.obj_state(o, z["tau"][i] + tp)
            inc = np.abs(lp - np.interp(sp, sf, df)) <= CORR
            l_now = abs(l - float(np.interp(s, sf, df)))
            if not (l_now <= CORR or (inc.any() and int(np.argmax(inc)) < 10)): continue   # in, or entering within 1 s
            k_leave = int(np.flatnonzero(inc)[-1]) + 1 if inc.any() else 0
            wait, v_go, vk, out = k_leave + round(dr.RESTART_WAIT / DT), float(v[i]), 0.0, np.zeros(N)
            for k in range(N):
                if k >= wait: vk = min(vk + dr.RESTART_A * DT, v_go)
                out[k] = vk
            xl = np.r_[0, np.cumsum(0.5 * (np.r_[0.0, out][1:] + np.r_[0.0, out][:-1]) * DT)][1:]
            xo = np.r_[0, np.cumsum(0.5 * (v[i:i + N + 1][1:] + v[i:i + N + 1][:-1]) * DT)]
            yl = np.interp(xl, xo, yaw[i:i + N + 1])
            res.append(dict(episode=ep, i=i, kind="restart_hold", pass_type="", obj=o["n"], cls=o["cls"], ptype=o["ptype"],
                            timing=o["timing"], v0=0.0, a_need=0.0, d_stop=round(float(gap), 2), seen_i=-1,
                            restart_s=round(wait * DT, 1), v_min=0.0, orig_vmin=round(float(v[i:i + N + 1].min()), 3)))
            fut.append(dict(episode=ep, i=i, fut_v=out.astype(np.float32), fut_yaw=yl.astype(np.float32)))
    return res, fut


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("s0"); ap.add_argument("index_dir"); ap.add_argument("out"); ap.add_argument("eps", nargs="+")
    ap.add_argument("--onset", choices=("lidar", "camera"), default="lidar")
    ap.add_argument("--vis-px", type=int, default=30); ap.add_argument("--vis-h", type=int, default=8); ap.add_argument("--vis-moving", type=float, default=0.3)
    ap.add_argument("--restart", action="store_true", help="make restart_hold rows (off since the 10/29 audit)")
    arg = ap.parse_args(); S0, idx_dir, out, eps = Path(arg.s0), Path(arg.index_dir), Path(arg.out), arg.eps
    (out / "ego").mkdir(parents=True, exist_ok=True); tmp = out / "_ext"; (tmp / "_ego").mkdir(parents=True, exist_ok=True)
    index = pd.read_parquet(idx_dir / "index.parquet") if (idx_dir / "index.parquet").exists() else pd.DataFrame(columns=COLS)
    index["episode"] = index.episode.astype(str)
    M, lap, exempt, cells = maneuvers(), float(dr.data.path_xy()[1][-1]), dr.data.kmap().speed_exempt_s, dr.data.ROUTE["cells"]
    labs, futs, vrows, keep, summ = [], [], [], [], {}
    for ep in eps:
        z0 = dict(np.load(S0 / "_ego" / f"{ep}.npz")); n0 = len(z0["tau"]); had_ext = "ext" in z0
        if had_ext: z0 = {k: a[:int(z0["ext"])] if a.ndim == 1 and len(a) > int(z0["ext"]) else a for k, a in z0.items() if k != "ext"}
        # main 10/29: the original future must end at the collision (the RDDF Ego drove into the mover): cut the table at the
        # first of a > 3 m/s drop in one step or the scorer's collision, the route at the last speed from there on
        L = pd.read_parquet(S0 / "_popout" / ep / "labels.parquet")
        cut, ns = len(z0["tau"]), collision_ns(L, json.load(open(S0 / "_popout" / ep / "episode.json")))
        if ns is not None: cut = min(cut, int(np.searchsorted(z0["tau"], np.interp(ns, z0["bag_ns"], z0["tau"]))))
        drop = np.flatnonzero(np.diff(z0["v"]) < -3.0)
        if len(drop): cut = min(cut, int(drop[0]) + 1)
        z0 = {k: a[:cut] if a.ndim == 1 and len(a) == len(z0["tau"]) else a for k, a in z0.items()}
        n0 = len(z0["tau"])
        z = extend(z0, N + 2); np.savez(tmp / "_ego" / f"{ep}.npz", **z)
        if not had_ext: np.savez(S0 / "_ego" / f"{ep}.npz", **z, ext=np.array(n0))   # the original table with the route tail: loader rows need i + 64
        trk, rec = tracks_from(ep, S0, z)
        vis, memo = front_vis(ep, S0, z, rec) if arg.onset == "camera" else None, {}

        def seen(o, t):                                                # the Front rule (module doc), per mover and row
            if (o["n"], t) not in memo:
                px, h = vis(o["n"], t)
                memo[o["n"], t] = t - o["tau_start"] >= arg.vis_moving and px >= arg.vis_px and h >= arg.vis_h
            return memo[o["n"], t]
        dr.SEEN = seen if arg.onset == "camera" else None
        sc = dict(rec["scene_meta"], actors=[dict(a, zone="ra_parked") if a.get("zone") == "rb_parked" else a
                                             for a in rec["scene_meta"]["actors"]])
        dr.tracks = lambda *_a, _t=trk: _t                             # Object Info start times instead of r2.log
        res, fut, _ = dr.one((ep, sc, tmp, None, range(15, n0)))       # rows with sensors only (i < n0)
        first_stop = min((r["i"] for r in res if r["kind"] in dr.STOP_KINDS), default=None)
        r2, f2 = restart_rows(ep, z, n0, trk, None) if arg.restart else ([], [])
        ok = []
        for r, f in zip(res, fut):                                     # dyn_relabel.self_check's A_MAX rule, per row
            full = np.r_[z["v"][r["i"]], f["fut_v"]]; dev = np.abs(f["fut_v"] - z["v"][r["i"] + 1:r["i"] + N + 1]) > 1e-6
            ok.append(r["kind"] == "pass" or not (np.diff(full) / DT < -dr.A_MAX - 1e-3).any())   # any step, deviating or not; 1e-3:
            # fut_v is float32 (1e-6 dropped A_MAX labels, the unavoidable events' first rows, 10/29)
        summ_bad = len(ok) - sum(ok)
        res = [r for r, k in zip(res, ok) if k]; fut = [f for f, k in zip(fut, ok) if k]
        first = {}                                                     # an event starts at its first stop row the Front rule sees
        for r in sorted(res, key=lambda r: r["i"]):
            if r["kind"] in dr.STOP_KINDS and r["obj"] not in first and (arg.onset == "lidar" or seen(trk[r["obj"]], float(z["tau"][r["i"]]))):
                first[r["obj"]] = r["i"]
        pre = [r["kind"] in dr.STOP_KINDS and r["i"] < first.get(r["obj"], np.inf) for r in res]
        res = [r for r, p in zip(res, pre) if not p]; fut = [f for f, p in zip(fut, pre) if not p]
        res += r2; fut += f2
        ie = index[index.episode == ep].set_index("i")
        for r, f in zip(res, fut):
            if r["kind"] not in dr.STOP_KINDS + ["restart_hold"]: continue
            i = r["i"]; name = f"{ep}_{'rh' if r['kind'] == 'restart_hold' else 'dyn'}{i:05d}"
            zz = {k: (a.copy() if a.ndim == 1 and len(a) == len(z["tau"]) else a) for k, a in z.items()}
            zz["v"][i + 1:i + N + 1], zz["yaw"][i + 1:i + N + 1] = f["fut_v"], f["fut_yaw"]
            if r["kind"] == "restart_hold":                            # standing history: v 0, pose held at row i
                zz["v"][i - 15:i + 1] = 0.0
                for k in ("x", "y", "yaw", "path_s", "d"): zz[k][i - 15:i] = zz[k][i]
            np.savez_compressed(out / "ego" / f"{name}.npz", **zz)
            vrows.append(dict(episode=name, i=i, tau=float(z["tau"][i]), cell=cells.get(int(z["section"][i]), "C?"), split="train",   # popout_mix.merge splits
                              nav_text=nav_text(float(z["path_s"][i]), M, lap, exempt), fog=bool(z["fog"][i]), hour=int(z["hour"][i]),
                              scene=str(rec["scene"]), src_episode=ep, kind=r["kind"],
                              section=int(z["section"][i]), cls=r["cls"], timing=r["timing"], start=next(o["start"] for o in trk if o["n"] == r["obj"])))
        relab = {r["i"] for r in res if r["kind"] != "pass"}
        keep.append(ie.reset_index().loc[lambda d: ~d.i.isin(relab), ["episode", "i"]] if len(ie) else pd.DataFrame(columns=["episode", "i"]))
        for r, f in zip(res, fut): f["kind"] = r["kind"]                  # (episode, i) may hold a stop and a restart_hold row
        labs += res; futs += fut
        summ[ep] = dict(dropped_amax=summ_bad, dropped_pre_vis=int(sum(pre)), first_stop_i=first_stop, n=n0, end=rec.get("end"), starts=[o["start"] for o in trk], tau_start=[o["tau_start"] for o in trk])
    lab = pd.DataFrame(labs); vi = pd.DataFrame(vrows)
    lab.to_parquet(out / "dyn_relabel_index.parquet", index=False)
    pd.DataFrame(futs).to_parquet(out / "dyn_future.parquet", index=False)
    vi.to_parquet(out / "dyn_index.parquet", index=False)
    pd.concat(keep).to_parquet(out / "keep.parquet", index=False)
    kinds = lab.kind.value_counts().to_dict() if len(lab) else {}
    json.dump(dict(episodes=summ, kinds=kinds, virtual=len(vi)), open(out / "summary.json", "w"), indent=1, default=str)
    print(json.dumps(dict(episodes=len(eps), kinds=kinds, virtual=len(vi))))


if __name__ == "__main__":
    main()
