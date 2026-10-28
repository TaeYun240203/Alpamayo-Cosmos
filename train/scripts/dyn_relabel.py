"""Synthetic dynamic-obstacle stop labels for R2 A (user 10/28: "갑자기 튀어나오면 급정거가 반드시 필요하다"). CPU, no model.
Doc: reports/dyn_relabel_20261028.md.

The bags hold no Object Info topic (tools/scorer/label_check_expert.py says the same), so each moving object's track is
rebuilt from its scene manifest (route-frame path (s, lat) -> (s_end, lat_end), speed kph, move m; scenario_variants.r2_actor
walks that straight path at constant speed) and the collection driver's r2.log `disp` lines (GT displacement from the start,
every 2 s): the start time is tau - disp / v of the samples taken mid-walk. Object Info would give a scaled static prop its
unscaled size (memory "/objects unscaled bbox"); irrelevant here: only people and moving props, all ~0.7 m.

Per index row t0 (R2 A dynamic zone + R2 B tunnel, person / other movers; car props and cut-in cars out):
  seen      object moving at t0, within 50 m and 100 deg of the heading, not hidden by a parked car (label_check.visible:
            2D rays from the LiDAR); stays seen once seen
  conflict  the object at its t0 velocity over 6.4 s enters the Ego corridor (route-frame lateral within half width 0.95 +
            0.5 m + 0.35 m of the Ego's planned lateral path) ahead of the nose, before the Ego's rear is past it (+1 s)
  label     from the first seen+conflict row: stop 3 m before the object (nose to its near edge). a_need = v^2 / 2(D - 3);
            a_need <= 2.5: the original drive until a SOFT_A envelope bites (max(SOFT_A, a_need)), else that decel from t0,
            at most A_MAX (R2 driver AEB, measured); stays stopped while the object (its true track) is in the corridor.
            Restart (user 10/28, no loss mask): RESTART_WAIT after the object has left the corridor, RESTART_A up to the
            original speed, then the original drive; a label that never deviated is the original drive
  counter   seen, no conflict, the original future does not brake: the original label, kind "pass"
Kinds: soft, hard, hold (standing at t0), pass (counter example), popout = soft/hard of a behind_parked object.
Replay exclusions (b): rows of each R2 A AEB / yield stop window (stop_pools.r2_rows rule) whose stop an eligible object
caused, before its first seen+conflict row; all pool rows are excluded from the replay too (one input, one label).
Stop windows by cause (r2.log yield / brake targets -> tracks): cutin, dynamic, car_prop, static (a disp entry that is no
moving track: parked cars, static obstacles); dyn_exclude.py keeps the cut-in-only windows.
Preprocessed for the unchanged loader (user 10/28 "그냥 전처리 해서 데이터셋 넘기면 되는거 아니야?"): every stop row becomes a
virtual episode <ep>_dyn<i:05d> whose ego/<name>.npz is the original _ego table with v and yaw at i+1 .. i+64 replaced by
the label; its sensor files are links to the original episode (scripts/dyn_stage.py makes them on the box).
Out (OUT): dyn_relabel_index.parquet (index columns + kind, obj, cls, ptype, timing, v0, a_need, d_stop, seen_i, restart_s,
v_min, orig_vmin), dyn_future.parquet (episode, i, fut_v 64, fut_yaw 64 absolute unwrapped), dyn_index.parquet (virtual
episodes' index rows + src_episode, kind), ego/<virtual>.npz, dyn_val.parquet (val + val_asset virtual rows, every 5th),
dyn_keep_pass.parquet (train pass rows of the original episodes), tracks.parquet, r2_prevision_exclude.parquet,
r2_stop_windows.parquet (episode, i, cause, src), summary.json.
Usage (cwd = workspace): python train/scripts/dyn_relabel.py R2_ROOT LOGS OUT [--jobs 8] --episodes train/assets/ft/r2_A.parquet
  R2_ROOT: _ego/, _index/ (B2 step0_r2/_index: the 10/28 --glitch net index, 121,523 rows), _meta/; LOGS: */<ep>.r2.log
"""
import argparse, glob, json, math, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np, pandas as pd

WS = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WS / "tools/coc_labeler"), str(WS / "tools/scorer")]
from coc_labeler import data  # noqa: E402
from label_check import visible  # noqa: E402

FRONT, REAR, HALF_W = 3.9, 0.8, 0.95                     # label_check.yaml body_m (base_link = rear axle)
PAD, R_OBJ, STOP_GAP = 0.5, 0.35, 3.0                     # corridor pad, object half width (r2_actor half_w), stop gap
SOFT_MAX, SOFT_A = 2.5, 2.0                               # m/s2: "soft" if a_need <= SOFT_MAX; the soft envelope's decel
RESTART_WAIT, RESTART_A = 0.5, 1.5                        # s, m/s2: user 10/28 (the object leaving is visible: a predictable start)
A_MAX = None                                              # set in main from the R2 driver AEB stops
SEE_M, SEE_DEG, H, N, DT = 50.0, 100.0, 6.4, 64, 0.1
VIS_CFG = dict(bev_m=[20.0, 60.0, 20.0], lidar_x_m=1.4, occluder_m=2.0)
CAR_M = (4.6, 1.9)
STOP_KINDS = ["soft", "hard", "hold", "popout_soft", "popout_hard"]


def disp_series(path):
    """r2.log -> (unix times, list of {k: (type, disp)})."""
    T, D = [], []
    for line in open(path, errors="replace"):
        p = line.split()
        if len(p) > 2 and p[1] == "disp":
            T.append(float(p[0])); D.append({int(q.split(":")[0]): (q.split(":")[1], float(q.split(":")[2])) for q in p[2:]})
    return np.array(T), D


def targets(path):
    """r2.log -> [(unix, object number)] of GT yield-on and AEB brake lines (the number is the disp line's)."""
    out = []
    for line in open(path, errors="replace") if path else []:
        p = line.split()
        if len(p) > 3 and p[1] == "yield" and p[2] == "on" and ":" in p[3]: out.append((float(p[0]), int(p[3].split(":")[0])))
        elif len(p) > 3 and p[1] == "brake" and p[2] == "object": out.append((float(p[0]), int(p[3])))
    return out


def tracks(ep, sc, z, log):
    """moving actors of the scene with their start tau (None: never moved) and route-frame path."""
    bag_s, tau = z["bag_ns"] / 1e9, z["tau"]
    T, D = disp_series(log) if log else (np.zeros(0), [])
    keys = sorted({k for d in D for k in d})
    dmax = {k: max(d.get(k, ("", 0.0))[1] for d in D) for k in keys}
    used, out = set(), []
    for n, a in enumerate(sc.get("actors", [])):
        if not str(a.get("zone", "")).endswith("_dynamic"): continue
        typ = "pedestrian" if a.get("ped") else "obstacle"
        cand = [k for k in keys if k not in used and next(d[k][0] for d in D if k in d) == typ] or [k for k in keys if k not in used]
        k = min(cand, key=lambda k: abs(dmax[k] - a["move"])) if cand else None
        if k is not None and dmax[k] > a["move"] + 1.5: k = None             # moved beyond its path: not this actor (cut-ins and
        # walkers cut off by the episode end move less than their path)
        if k is not None: used.add(k)
        v_o = a["kph"] / 3.6; t0 = None
        if k is not None and dmax[k] > 0.3:
            ds = np.array([d[k][1] if k in d else np.nan for d in D]); tt = np.interp(T, bag_s, tau); ok = np.isfinite(ds); ds, tt = ds[ok], tt[ok]
            mid = (ds > 0.2) & (ds < dmax[k] - 0.2)
            if mid.any(): t0 = float(np.median(tt[mid] - ds[mid] / v_o))
            else:                                          # walked the whole path between two 2 s samples
                j = int(np.argmax(ds > 0.2)); t0 = float(tt[j] - min(ds[j], a["move"]) / v_o - (tt[j] - tt[max(j - 1, 0)]) / 2)
        out.append(dict(episode=ep, n=n, disp_k=k, cls=a["cls"], ptype=a["ptype"], timing=a["timing"], s0=a["s"],
                        l0=a["lat"], s1=a["s_end"], l1=a["lat_end"], kph=a["kph"], move=float(dmax[k]) if k is not None and dmax[k] > 0.3 else a["move"],
                        active=a["active"], tau_start=t0))
    return out


def obj_state(o, t):
    """route-frame position and velocity of a track at times t (array)."""
    L = math.hypot(o["s1"] - o["s0"], o["l1"] - o["l0"]); v_o = o["kph"] / 3.6
    us, ul = ((o["s1"] - o["s0"]) / L, (o["l1"] - o["l0"]) / L) if L > 1e-6 else (0.0, 0.0)
    if o["tau_start"] is None or not np.isfinite(o["tau_start"]): arc = np.zeros_like(t); moving = np.zeros(t.shape, bool)
    else:
        arc = np.clip(v_o * (t - o["tau_start"]), 0, min(o["move"], L)); moving = (arc > 0) & (arc < min(o["move"], L))
    s, l = o["s0"] + us * arc, o["l0"] + ul * arc
    return s, l, np.where(moving, v_o * us, 0.0), np.where(moving, v_o * ul, 0.0), arc > 0


def label(vo, lim, a_need, k_leave):
    """synthetic 64-step speed (t0+0.1 ... t0+6.4): the original drive vo[1:] capped by the stop envelope sqrt(2 A room),
    room = lim[k] - travel (lim: nose travel allowed, inf = free); braking beyond A_MAX is impossible (max brake held).
    From step k_leave (the object out of the corridor) a braked label waits RESTART_WAIT, then gains RESTART_A up to vo."""
    A, wait = min(max(SOFT_A, a_need), A_MAX), k_leave + round(RESTART_WAIT / DT)
    v, x, out, braking = vo[0], 0.0, np.zeros(N), False
    for k in range(N):
        if braking and k >= k_leave:                                           # restart
            vk = max(min(vo[k + 1], v if k < wait else v + RESTART_A * DT), 0.0)
            braking = vk < vo[k + 1] - 1e-6                                    # caught up: the original drive again
        else:
            env = math.sqrt(2 * A * max(lim[k] - x, 0.0)) if np.isfinite(lim[k]) else np.inf
            vk = vo[k + 1] if env >= vo[k + 1] else min(vo[k + 1], max(env, v - A_MAX * DT, 0.0))
            if braking and np.isfinite(lim[k]): vk = max(min(vk, v), 0.0)     # no creeping on while the object is there (and no
            braking |= vk < vo[k + 1] - 1e-6                                   # GT standstill noise of -0.03 m/s: no reversing)
        x += 0.5 * (v + vk) * DT; v = vk; out[k] = vk
    return out


def one(args):
    ep, sc, R, log, rows = args
    z = np.load(Path(R) / "_ego" / f"{ep}.npz")
    tau, v, yaw, s_e, d_e = (z[k].astype(float) for k in ("tau", "v", "yaw", "path_s", "d"))
    x_e, y_e = z["x"].astype(float), z["y"].astype(float)
    trk = tracks(ep, sc, z, log)
    occl = []
    for k, a in enumerate(a for a in sc.get("actors", []) if a.get("zone") in ("ra_parked", "static", "tunnel_static") and "s" in a):
        px, py, ph = data.route_point(a["s"], a.get("lat", 0.0)); car = a["zone"] == "ra_parked" or a.get("id") in data.VEHICLE_PROPS
        sc_ = a["scale"][:2] if "scale" in a else (1.0, 1.0)                     # dyn_exclude.footprint (Object Info: unscaled)
        occl.append((2000 + k, "vehicle" if car else "obstacle", px, py, ph, 0.0, 0.0, (CAR_M[0] if car else 1.0) * sc_[0], (CAR_M[1] if car else 1.0) * sc_[1]))
    elig = [o for o in trk if o["cls"] in ("person", "other") and o["ptype"] != "cutin" and o["tau_start"] is not None]
    n = len(tau); fut_t = DT * np.arange(1, N + 1); tp = np.r_[0.0, fut_t]; corr = HALF_W + PAD + R_OBJ
    st = [obj_state(o, tau) for o in elig]; o_cls = [o["cls"] for o in elig]
    res, fut, seen_at, rowset = [], [], {}, set(int(r) for r in rows)
    for i in range(15, n - N - 1):                                                # every grid row: "seen" is sticky
        cand, why = [], []
        sf, df = s_e[i:i + N + 1], d_e[i:i + N + 1]
        for m, o in enumerate(elig):
            s, l, vs, vl, started = (q[i] for q in st[m])
            if not started or (o["cls"] != "person" and vs == 0 and vl == 0): continue   # a prop at rest = standing: avoid, never stop
            if m not in seen_at:
                ce, se = math.cos(yaw[i]), math.sin(yaw[i]); ox, oy, _ = data.route_point(s, l)
                fx, fy = ce * (ox - x_e[i]) + se * (oy - y_e[i]), -se * (ox - x_e[i]) + ce * (oy - y_e[i])
                if (math.hypot(fx - 1.4, fy) <= SEE_M and abs(math.degrees(math.atan2(fy, fx - 1.4))) <= SEE_DEG
                        and visible(x_e[i], y_e[i], yaw[i], [(1, "pedestrian", ox, oy, 0.0, 0.0, 0.0, 0.7, 0.7)] + occl, VIS_CFG)[0]):
                    seen_at[m] = i
            if m not in seen_at or i not in rowset: continue
            sp, lp = s + vs * tp, l + vl * tp                                     # t0 velocity held over 6.4 s
            gap = sp - R_OBJ - (s_e[i] + FRONT)
            t_rear = (sp + R_OBJ - (s_e[i] - REAR)) / max(v[i], 0.5)
            hit = (np.abs(lp - np.interp(sp, sf, df)) <= corr) & (gap > -0.5) & (gap < SEE_M) & (tp <= t_rear + 1.0)
            if hit.any(): cand.append(m)
            elif 0 < gap[0] < SEE_M: why.append(m)
        if not cand and not why: continue
        vo = v[i:i + N + 1]
        xo = np.r_[0, np.cumsum(0.5 * (vo[1:] + vo[:-1]) * DT)]                 # original travel per step (0 = t0)
        clean_go = vo[0] >= 2.0 and vo[1:].min() >= max(vo[0] - 1.5, 0.6 * vo[0])
        k_leave, restart = N, float("nan")
        if cand:
            lim, k_leave = np.full(N, np.inf), 0                                  # true tracks: nose travel allowed per step
            for m in cand:
                s, l = st[m][0][i + 1:i + N + 1], st[m][1][i + 1:i + N + 1]
                g = s - R_OBJ - (s_e[i] + FRONT)
                inc = (np.abs(l - np.interp(s, sf, df)) <= corr) & (g > -0.5)
                if o_cls[m] != "person": inc &= (st[m][2][i + 1:i + N + 1] != 0) | (st[m][3][i + 1:i + N + 1] != 0)   # a prop that came to rest no longer holds the Ego
                if not inc.any(): continue
                last = int(np.flatnonzero(inc)[-1]); k_leave = max(k_leave, last + 1)
                gl = np.where(inc, g - STOP_GAP, np.inf)
                sm = np.minimum.accumulate(gl[:last + 1][::-1])[::-1]             # stay stoppable until it has left
                lim[:last + 1] = np.minimum(lim[:last + 1], sm)
            D = float(lim[0]) if np.isfinite(lim[0]) else min(float(st[m][0][i] - R_OBJ - (s_e[i] + FRONT) - STOP_GAP) for m in cand)
            a_need = v[i] ** 2 / (2 * D) if D > 0.05 else np.inf
            vl_ = label(vo, lim, a_need, k_leave)
            dev = np.abs(vl_ - vo[1:]) > 0.3
            if not dev.any() and v[i] >= 0.5 and vl_.min() >= 0.5:
                if not clean_go: continue                                          # driver slowed anyway: no clean "go"
                kind, vl_, k_leave = "pass", vo[1:].copy(), N                      # the object clears in time
            else:                                                                  # stops (label = original where it did)
                if k_leave == 0: continue                                          # not truly in the corridor
                kind = "hold" if v[i] < 0.5 else "soft" if a_need <= SOFT_MAX else "hard"
                if k_leave < N: restart = round(k_leave * DT, 1)
            m = min(cand, key=lambda m: st[m][0][i])
        else:
            if not clean_go: continue
            m = why[0]; kind, vl_, a_need, D = "pass", vo[1:].copy(), 0.0, float("nan")
        o = elig[m]
        ptype = ""
        if kind == "pass":                                                         # along / will cross after the Ego / has crossed
            s_, l_ = st[m][0][i], st[m][1][i]; dp = float(np.interp(s_, sf, df))
            crossed = np.sign(l_ - dp) != np.sign(o["l0"] - dp) and abs(l_ - dp) > corr
            ptype = "along" if o["ptype"] in ("along", "curb_stop", "short_cross") else "crossed" if crossed else "late_cross"
        self_check(vo, vl_, kind, a_need, k_leave)
        if kind in ("soft", "hard") and o["ptype"] == "behind_parked": kind = "popout_" + kind
        xl = np.r_[0, np.cumsum(0.5 * (np.r_[vo[0], vl_][1:] + np.r_[vo[0], vl_][:-1]) * DT)][1:]
        yl = np.interp(xl, xo, yaw[i:i + N + 1])                                    # same path, heading by distance
        res.append(dict(episode=ep, i=int(i), kind=kind, pass_type=ptype, obj=int(o["n"]), cls=o["cls"], ptype=o["ptype"], timing=o["timing"],
                        v0=round(float(v[i]), 3), a_need=round(float(min(a_need, 99.0)), 3), d_stop=round(D, 2), seen_i=int(seen_at[m]),
                        restart_s=restart, v_min=round(float(vl_.min()), 3), orig_vmin=round(float(vo[1:].min()), 3)))
        fut.append(dict(episode=ep, i=int(i), fut_v=vl_.astype(np.float32), fut_yaw=yl.astype(np.float32)))
    for m, o in enumerate(elig): o["seen_i"] = seen_at.get(m)
    return res, fut, trk


def self_check(vo, vl, kind, a_need, k_leave):
    """the rules: never faster than the original, own braking within A_MAX, no creeping while held, the restart only after
    RESTART_WAIT and at most RESTART_A, soft/hard by a_need, pass = the original."""
    full, dev = np.r_[vo[0], vl], np.abs(vl - vo[1:]) > 1e-6
    acc, w = np.diff(full) / DT, min(k_leave + round(RESTART_WAIT / DT), N)
    assert (vl <= np.maximum(vo[1:], 0.0) + 1e-6).all(), "label faster than the original drive"
    assert (acc >= -A_MAX - 1e-6)[dev].all(), "own braking beyond A_MAX"
    f = int(np.argmax(dev)) + 1 if dev.any() else w                        # the envelope's first cap may still rise
    assert (acc[f:w] <= 1e-6)[dev[f:w]].all(), "speeding up while held or before RESTART_WAIT"
    assert (acc[w:] <= RESTART_A + 1e-6)[dev[w:]].all(), "restart faster than RESTART_A"
    assert kind != "soft" or a_need <= SOFT_MAX
    assert kind != "hard" or a_need > SOFT_MAX
    assert kind != "pass" or not dev.any()


def a_max_from_aeb(R, meta):
    """median mean decel of R2 driver AEB stops from >= 4 m/s (onset to v < 0.3): the driver's real full brake."""
    out = []
    for r in meta.itertuples():
        z = np.load(Path(R) / "_ego" / f"{r.episode}.npz"); v, tau = z["v"], z["tau"]
        for t0, _ in (r.aeb_tau if r.aeb_tau is not None else []):
            j = int(np.searchsorted(tau, t0))
            if j >= len(v) or v[j] < 4.0: continue
            k = np.flatnonzero(v[j:] < 0.3)
            if len(k) and k[0] > 0: out.append(v[j] / (k[0] * DT))
    return np.array(out)


def virtual(R, out, lab, fut):
    """stop rows -> virtual episodes: ego/<ep>_dyn<i>.npz (the original table, v and yaw at i+1 .. i+64 = the label) and their
    index rows (the original row's columns, episode = the virtual name, src_episode = the original)."""
    st = lab[lab.kind.isin(STOP_KINDS)].merge(fut, on=["episode", "i"])
    (out / "ego").mkdir(exist_ok=True); rows = []
    for ep, g in st.groupby("episode"):
        z = dict(np.load(Path(R) / "_ego" / f"{ep}.npz"))
        for r in g.itertuples():
            name, i = f"{ep}_dyn{r.i:05d}", int(r.i)
            zz = dict(z); zz["v"], zz["yaw"] = z["v"].copy(), z["yaw"].copy()
            zz["v"][i + 1:i + N + 1], zz["yaw"][i + 1:i + N + 1] = r.fut_v, r.fut_yaw
            np.savez_compressed(out / "ego" / f"{name}.npz", **zz)
            rows.append(dict(episode=name, i=i, tau=r.tau, cell=r.cell, split=r.split, nav_text=r.nav_text, fog=r.fog, hour=r.hour,
                             scene=r.scene, src_episode=ep, kind=r.kind, section=r.section, cls=r.cls, timing=r.timing))
    return pd.DataFrame(rows)


def main():
    global A_MAX
    ap = argparse.ArgumentParser(); ap.add_argument("r2"); ap.add_argument("logs"); ap.add_argument("out"); ap.add_argument("--jobs", type=int, default=8); ap.add_argument("--limit", type=int)
    ap.add_argument("--episodes", help="parquet (episode): the A / B episodes allowed in training (assets/ft/r2_AB.parquet: moving-prop-only out)")
    a = ap.parse_args(); R, out = Path(a.r2), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(R / "_meta/r2_episodes.parquet"); A = meta[meta.kind.isin(["A", "B"])]   # dynamic zone + tunnel
    if a.episodes: A = A[A.episode.isin(set(pd.read_parquet(a.episodes).episode.astype(str)))]
    A = A.iloc[:a.limit]
    dec = a_max_from_aeb(R, meta); A_MAX = float(np.median(dec))   # onset times lead the brake: higher values include pre-braking
    idx = pd.read_parquet(R / "_index/index.parquet").astype({"episode": str})
    logs = {Path(f).name.split(".")[0]: f for f in glob.glob(f"{a.logs}/*/*.r2.log")}
    jobs = []
    for r in A.itertuples():
        _, sc = data.scene_of(r.episode)
        rows = idx.i[idx.episode == r.episode].to_numpy()
        if sc and len(rows): jobs.append((r.episode, sc, str(R), logs.get(r.episode), rows))
    with Pool(a.jobs, initializer=_init, initargs=(A_MAX,)) as p: parts = p.map(one, jobs, chunksize=4)
    lab = pd.DataFrame([x for r, _, _ in parts for x in r]); fut = pd.DataFrame([x for _, f, _ in parts for x in f])
    trk = pd.DataFrame([x for _, _, t in parts for x in t])
    lab = lab.merge(idx, on=["episode", "i"]).merge(meta[["episode", "kind"]].rename(columns={"kind": "section"}), on="episode")   # A dynamic zone, B tunnel
    fut = fut.merge(lab[["episode", "i"]], on=["episode", "i"])
    lab.to_parquet(out / "dyn_relabel_index.parquet", index=False); fut.to_parquet(out / "dyn_future.parquet", index=False)
    trk.to_parquet(out / "tracks.parquet", index=False)
    vi = virtual(R, out, lab, fut); vi.to_parquet(out / "dyn_index.parquet", index=False)
    vi[vi.split.isin(["val", "val_asset"]) & (vi.i % 5 == 0)][["episode", "i"]].to_parquet(out / "dyn_val.parquet", index=False)
    lab[(lab.kind == "pass") & (lab.split == "train")][["episode", "i"]].to_parquet(out / "dyn_keep_pass.parquet", index=False)
    win = stop_windows(A, R, trk, logs)
    win.merge(idx[["episode", "i"]], on=["episode", "i"]).to_parquet(out / "r2_stop_windows.parquet", index=False)
    ex = prevision(win, idx, trk, lab, R)
    ex.to_parquet(out / "r2_prevision_exclude.parquet", index=False)
    tr = lab[lab.split == "train"]; stp = lab[lab.kind.isin(STOP_KINDS)]
    w1 = win.drop_duplicates(["episode", "j", "src"])
    s = dict(a_max=round(A_MAX, 2), aeb_decel_pct={p: round(float(np.percentile(dec, p)), 2) for p in (50, 75, 90, 95)}, aeb_stops=len(dec),
             episodes=len(jobs), tracks=len(trk), tracks_moved=int(trk.tau_start.notna().sum()),
             tracks_no_disp=int(trk.disp_k.isna().sum()), rows=len(lab), rows_train=len(tr),
             by_kind=lab.kind.value_counts().to_dict(), by_kind_train=tr.kind.value_counts().to_dict(),
             by_section_kind_train={f"{a}/{k}": n for (a, k), n in tr.groupby(["section", "kind"]).size().items()},
             by_kind_cls={f"{k}/{c}": n for (k, c), n in lab.groupby(["kind", "cls"]).size().items()},
             stop_rows_with_restart=int(stp.restart_s.notna().sum()), restart_s_median=float(stp.restart_s.median()),
             virtual_episodes=len(vi), virtual_by_split=vi.split.value_counts().to_dict(), dyn_val=int((vi.split.isin(["val", "val_asset"]) & (vi.i % 5 == 0)).sum()),
             stop_windows_by_cause=w1.cause.value_counts().to_dict(), prevision_excluded=len(ex), split=lab.split.value_counts().to_dict())
    json.dump(s, open(out / "summary.json", "w"), indent=1, default=int); print(json.dumps(s, indent=1, default=int))


def stop_windows(A, R, trk, logs):
    """AEB / yield stop windows as stop_pools.r2_rows (stop j = first v < 0.5 within 8 s of the interval start; rows
    [j - 64, j]) with their cause from the r2.log yield-on / brake targets in the interval: cutin, dynamic (person / other
    track), car_prop (a moving car prop), static (a disp entry that is no moving track: parked car, static obstacle),
    unknown (no target line); several causes joined by '+'."""
    out = []
    for r in A.itertuples():
        z = np.load(Path(R) / "_ego" / f"{r.episode}.npz"); tau, v, bag_s = z["tau"], z["v"], z["bag_ns"] / 1e9
        tg = [(float(np.interp(t, bag_s, tau)), k) for t, k in targets(logs.get(r.episode))]
        tk = trk[trk.episode == r.episode]; kind = {int(k): (c, p) for k, c, p in zip(tk.disp_k, tk.cls, tk.ptype) if pd.notna(k)}
        for src, runs in (("aeb", r.aeb_tau), ("yield", r.yield_tau)):
            for t0, t1 in (runs if runs is not None else []):
                k = np.flatnonzero((tau >= t0 - 0.05) & (tau <= max(t1, t0) + 8) & (v < 0.5))
                if not len(k): continue
                j = int(k[0]); cs = set()
                for t, n in tg:
                    if t0 - 0.3 <= t <= t1 + 0.3:
                        c, p = kind.get(n, ("static", ""))
                        cs.add("cutin" if p == "cutin" else "dynamic" if c in ("person", "other") else "car_prop" if c == "car" else "static")
                cause = "+".join(sorted(cs)) or "unknown"
                out += [(r.episode, i, j, cause, src) for i in range(max(j - N, 0), j + 1)]
    return pd.DataFrame(out, columns=["episode", "i", "j", "cause", "src"])


def prevision(win, idx, trk, lab, R):
    """(b): rows of AEB / yield stop windows whose stop an eligible object caused, before it was first seen+conflicting
    (its first pool row of a stop kind); attribution = an eligible track in the corridor ahead (nose gap < 25 m) at the
    stop row j."""
    first = lab[lab.kind.isin(STOP_KINDS)].groupby(["episode", "obj"]).i.min()
    out = []
    for (ep, j), g in win.groupby(["episode", "j"]):
        z = np.load(Path(R) / "_ego" / f"{ep}.npz"); tau, s_e, d_e = z["tau"], z["path_s"], z["d"]
        tk = trk[(trk.episode == ep) & trk.cls.isin(["person", "other"]) & (trk.ptype != "cutin") & trk.tau_start.notna()]
        cause = None
        for o in tk.to_dict("records"):
            s, l, _, _, _ = obj_state(o, np.array([tau[j]]))
            if abs(l[0] - d_e[j]) <= HALF_W + PAD + R_OBJ and -0.5 < s[0] - R_OBJ - (s_e[j] + FRONT) < 25: cause = o; break
        if cause is None: continue
        f = first.get((ep, cause["n"]), j + 1)
        out += [(ep, i) for i in range(max(j - N, 0), min(f, j + 1))]
    ex = pd.DataFrame(out, columns=["episode", "i"]).drop_duplicates()
    return ex.merge(idx[["episode", "i"]], on=["episode", "i"])


def _init(a_max):
    global A_MAX
    A_MAX = a_max


if __name__ == "__main__":
    main()
