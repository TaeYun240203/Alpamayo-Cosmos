"""DAgger replay pools (configs/dagger.yaml, users 10/28) as MoraiDataset keep / exclude parquets. Mac, CPU: the judged stop
lines come from the scorer's map, as tools/start_relabel/relabel_stage1.py reads them. R1 only: the R2 _ego tables have
no red or yellow (signal 0 / 3), so the green rule has nothing to find there and the no-GT rule would only drop rows.
  signal_approach.parquet    episode, i (every split; the dataset picks its split), user 10/28 "감속·접근": rows at a red
                             light (_ego signal 1) that brake, or still move (v0 >= 0.5) with the next judged stop line
                             <= APPROACH_M ahead; motion tags as scripts/ft_regression.py: v0 = v[i], dv3 = v[i+30] - v0,
                             v64 = v[i+64]; "stays stopped" v0 < 0.5 and v64 < 0.5, else "accel" dv3 > 1, else "brake"
                             dv3 < -1; minus the future_green rows
  signal_hold.parquet        "정지 유지": red-light rows tagged "stays stopped", minus the future_green rows
  highway.parquet            episode, i: rows of the 60 kph exemption (고주로, nav_text "No speed limit"), the high-speed
                             set of the history-dependence records (dagger/history.py)
  future_green.parquet       episode, i, reason: rows whose 6.4 s GT future knows that the light turns green (user 10/28
                             "미래의 초록불 정보가 6.4초 미래 궤적에 들어가면 안됀다"). Where the next judged stop line is within
                             the label's reach + 50 m (relabel_stage1 SIG_MARGIN; front axle 3 m ahead of base_link, lines up
                             to 5 m behind it count), the window signal[i .. i+64] goes from red / yellow to green ("to_green")
                             or has no GT ("unknown": 0), as relabel_stage1's signal_change / signal_unknown
  r1_replay_exclude.parquet  the R1 general replay's exclude: the start relabel rows (old "wait" GT, as student_ft's replay)
                             + future_green
  r2_A_dyn.parquet, r2_B_dyn.parquet   (`r2`, user 10/28 "동적장애물 장면의 경우 실제로 급제동하는 경우 위주로. 아예정지하지
                             않는 경우는 제외") episode, i of the R2 A / B pools (assets/ft r2_A / r2_B) whose 6.4 s GT future
                             meets an object event of the R2 driver (r2_episodes.parquet aeb_tau: GT AEB brake, yield_tau:
                             yield on..off; tools/step0/r2_meta.py) that really slowed the car: v 3 s later - v <= -2 m/s
                             starting in [start - 3 s, end], or v < 0.5 in [start, end + 3 s]; rows tau_i in [start - 6.4 s,
                             end + RESTART_S] (the stand and the restart once the object has passed). Only episodes with a
                             moving actor (actor_kph > 0). From the --glitch net R2 index; + its rescued rows (in the net index,
                             not in --old-index, user 10/28) whose 6.4 s GT future brakes hard (v 3 s later - v <= -2 m/s)
  r2_unseen.parquet          (`r2 --r2logs`, user 10/28) episode, i, reason of r2_A_dyn / r2_B_dyn rows whose GT future
                             brakes for an object the model could not see at t0: per object event of the r2.log (brake object
                             k, yield on k:type .. off; effective as above), the object's t0 position is rebuilt from the scene
                             manifest (start s/lat -> end s_end/lat_end, straight) and the log's 2 s `disp` (metres moved),
                             the other actors as they stand; tools/scorer/label_check.visible() (the shadow / label checker
                             rule: inside the LiDAR BEV, no vehicle or >= 2 m object in the line from the LiDAR) on the _ego
                             pose; rows from start - 6.4 s up to the first visible grid time are "unseen". r2.log object k ->
                             manifest actor by type (asset 5xxxxxxx pedestrian, else obstacle) and order within the type;
                             an episode whose types or movements do not match is "unjudgeable" (kept, counted). Rows kept
                             only by the hard-braking rule have no event to judge (kept, counted).
                             shortcut: object sizes are class defaults (R2 bags have no Object Info); use the recorded
                             sizes once the actors record them (objects.json.gz)
  pools.json                 counts per split (r1), per kind (r2)
Usage (cwd train/): python -m dagger.pools r1 --index ~/coc_runs/index_v3/index.parquet --ego ~/.cache/step0_ego \
    --start-relabel <step0>/_start_relabel/start_relabel_index.parquet --out assets/dagger
                    python -m dagger.pools r2 --index <step0_r2>/_index/index.parquet --ego <step0_r2>/_ego \
    --meta <step0_r2>/_meta/r2_episodes.parquet --pools assets/ft --old-index <the step-rule R2 index> --out assets/dagger \
    [--r2logs <dir of the collection's *.r2.log>]   (Mac: needs the scene manifests and the scorer map)
"""
import argparse, json, math
from pathlib import Path

import numpy as np, pandas as pd

FUT, DT, FRONT_AXLE, SIG_MARGIN, BEHIND = 64, 0.1, 3.0, 50.0, 5.0       # relabel_stage1 values
RED, YELLOW, GREEN = 1, 2, 3                                               # tools/step0/ego_tables.py SIG
APPROACH_M = 60.0                                                          # = configs/dagger.yaml dagger.tags.signal_stop_d
BRAKE_DV, BRAKE_S, STOP_V, RESTART_S = -2.0, 3.0, 0.5, 3.0                 # r2: hard braking, standing, restart margin


def motion(v, i):
    """scripts/ft_regression.py tags at rows i."""
    n = len(v); v0 = v[i]; dv3 = v[np.minimum(i + 30, n - 1)] - v0; v64 = v[np.minimum(i + 64, n - 1)]
    return np.select([(v0 < 0.5) & (v64 < 0.5), dv3 > 1.0, dv3 < -1.0], ["stays stopped", "accel", "brake"], "cruise")


def next_line(path_s, i, stops):
    """front axle -> next judged stop line at rows i (m, inf if none; relabel_stage1: lines with st + 5 - fs > 0 count)."""
    st = np.asarray(sorted(stops), float); fs = (path_s[i] + FRONT_AXLE)[:, None]
    return np.where(st[None] + BEHIND - fs > 0, st[None] - fs, np.inf).min(1)


def future_green(v, sig, path_s, i, stops):
    """reason per row i: "to_green", "unknown" or "" (kept)."""
    n = len(v)
    dist = np.r_[0, np.cumsum(0.5 * (v[1:] + v[:-1]) * DT)]
    j = np.minimum(i[:, None] + np.arange(FUT + 1), n - 1)
    w = sig[j]                                                                  # (rows, 65) window of the 6.4 s label
    arc = dist[j[:, -1]] - dist[i]
    d = next_line(path_s, i, stops)
    zone = d <= arc + SIG_MARGIN
    ry = np.isin(w, (RED, YELLOW))
    first = np.where(ry.any(1), ry.argmax(1), FUT + 1)
    to_green = ((w == GREEN) & (np.arange(FUT + 1)[None] > first[:, None])).any(1)
    return np.where(zone & to_green, "to_green", np.where(zone & (w == 0).any(1), "unknown", ""))


def build(index, ego_dir, stops):
    idx = pd.read_parquet(index, columns=["episode", "i", "split"]).astype({"episode": str})
    sig_rows, fg_rows = [], []
    for ep, g in idx.groupby("episode", sort=True):
        f = Path(ego_dir) / f"{ep}.npz"
        if not f.exists(): continue
        z = np.load(f); v, sig, s = z["v"].astype(np.float64), z["signal"], z["path_s"].astype(np.float64)
        i = g.i.to_numpy()
        reason = future_green(v, sig, s, i, stops)
        mo = motion(v, i)
        fg_rows.append(pd.DataFrame({"episode": ep, "i": i, "reason": reason, "split": g.split.to_numpy()})[reason != ""])
        red = (sig[i] == RED) & (reason == "")
        approach = red & ((mo == "brake") | ((v[i] >= 0.5) & (mo != "stays stopped") & (next_line(s, i, stops) <= APPROACH_M)))
        hold = red & (mo == "stays stopped")
        for name, keep in (("approach", approach), ("hold", hold)):
            sig_rows.append(pd.DataFrame({"episode": ep, "i": i[keep], "motion": mo[keep], "pool": name, "split": g.split.to_numpy()[keep]}))
    return pd.concat(sig_rows, ignore_index=True), pd.concat(fg_rows, ignore_index=True), idx


def effective(tau, v, intervals):
    """the object events that really slowed the car (module docstring) -> [(start, end)]."""
    out = []
    for a, b in intervals:
        m = (tau >= a - BRAKE_S) & (tau <= b)
        hard = bool(((np.interp(tau[m] + BRAKE_S, tau, v) - v[m]) <= BRAKE_DV).any())
        stop = bool((v[(tau >= a) & (tau <= b + RESTART_S)] < STOP_V).any())
        if hard or stop: out.append((float(a), float(b)))
    return out


def r2_rows(tau_i, events):
    keep = np.zeros(len(tau_i), bool)
    for a, b in events: keep |= (tau_i >= a - FUT * DT) & (tau_i <= b + RESTART_S)
    return keep


def unseen_of(a, kind, d):
    """reasons for the kept rows d of one bundle (Mac: scene manifests, scorer map)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/scorer"))
    from label_check import visible
    import yaml
    from morai_vla import build_index as bi
    vis_cfg = yaml.safe_load(open(Path(__file__).resolve().parents[2] / "tools/scorer/label_check.yaml"))["visibility"]
    logs = {f.name[:-len(".r2.log")]: f for f in Path(a.r2logs).glob("*/*.r2.log")}
    parts = []
    for ep, g in d.groupby("episode"):
        z = np.load(Path(a.ego) / f"{ep}.npz"); acts = (bi.data.scene_of(ep)[1] or {}).get("actors", [])
        r = unseen(ep, z["tau"][g.i.to_numpy()].astype(float), acts, logs[ep], z, bi.data, vis_cfg, visible) if ep in logs and acts \
            else np.full(len(g), "unjudgeable", object)
        parts.append(g.assign(reason=r))
    return pd.concat(parts) if parts else d.assign(reason=[])


def brakes(tau, v, i):
    """rows i whose 6.4 s GT future has v(t + 3 s) - v(t) <= BRAKE_DV."""
    t = tau[i][:, None] + np.arange(0, FUT * DT - BRAKE_S + 1e-6, DT)[None]
    return ((np.interp(t + BRAKE_S, tau, v) - np.interp(t, tau, v)) <= BRAKE_DV).any(1)


SIZES = {"vehicle": (4.6, 1.9), "pedestrian": (0.6, 0.6), "obstacle": (1.0, 0.6)}   # shortcut: class defaults (docstring)
VEHICLE_PROPS = {40100008, 40100009, 40100010, 40100011}                             # car-shaped props (coc_pilot/priv.py)


def log_kind(asset):
    return "pedestrian" if int(asset) // 10**7 == 5 else "obstacle"


def read_r2log(path):
    """-> (disp {k: (unix t (n,), metres (n,), kind)}, events [(t_on, t_off, k)]: brake lines merged per object when < 1 s
    apart, yield on .. off)."""
    disp, brakes, events, on = {}, {}, [], {}
    for line in open(path, errors="replace"):
        p = line.split()
        if len(p) < 3: continue
        try: t = float(p[0])
        except ValueError: continue
        if p[1] == "disp":
            for x in p[2:]:
                k, kind, d = x.split(":"); disp.setdefault(int(k), ([], [], kind)); disp[int(k)][0].append(t); disp[int(k)][1].append(float(d))
        elif p[1] == "brake" and p[2] == "object":
            brakes.setdefault(int(p[3]), []).append(t)
        elif p[1] == "yield" and p[2] == "on":
            k = int(p[3].split(":")[0]); on.setdefault(k, t); on["last"] = k
        elif p[1] == "yield" and p[2] == "off" and "last" in on:
            k = on.pop("last"); events.append((on.pop(k, t), t, k))
    for k, ts in brakes.items():
        for a_, b_ in _runs_of(sorted(ts)): events.append((a_, b_, k))
    return {k: (np.array(t), np.array(d), kind) for k, (t, d, kind) in disp.items()}, events


def _runs_of(ts, gap=1.0):
    out = []
    for t in ts:
        if out and t - out[-1][1] <= gap: out[-1][1] = t
        else: out.append([t, t])
    return out


def actor_map(acts, disp):
    """log object k -> manifest actor index, by kind and order; None when kinds or movements do not match."""
    kinds = [log_kind(x["id"]) for x in acts]
    if sorted(kinds) != sorted(v[2] for v in disp.values()): return None
    m = {}
    for kind in ("pedestrian", "obstacle"):
        for j, k in zip([j for j, x in enumerate(kinds) if x == kind], sorted(k for k, v in disp.items() if v[2] == kind)):
            moving = acts[j].get("kph", 0) > 0 and acts[j].get("move", 0) > 0.5
            if moving != (disp[k][1].max() > 0.3): return None
            m[k] = j
    return m


def actor_boxes(acts, data):
    """per actor: (label, start xy, end xy, heading, size, path length)."""
    out = []
    for x in acts:
        a = np.array(data.route_point(x["s"], x["lat"])[:2]); b = np.array(data.route_point(x.get("s_end", x["s"]), x.get("lat_end", x["lat"]))[:2])
        veh = int(x["id"]) in VEHICLE_PROPS or x.get("cls") == "car"
        label = "vehicle" if veh else log_kind(x["id"])
        h = math.atan2(*(b - a)[::-1]) if np.hypot(*(b - a)) > 0.3 else data.route_heading(x["s"])
        out.append((label, a, b, h, SIZES[label], float(np.hypot(*(b - a)))))
    return out


def unseen(ep, rows_tau, acts, logf, ego, data, vis_cfg, visible):
    """-> reason per kept row ("unseen", "no_event", "unjudgeable", "") of one episode."""
    disp, events = read_r2log(logf)
    m = actor_map(acts, disp) if disp else None
    if m is None: return np.full(len(rows_tau), "unjudgeable", object)
    tau, bag_s = ego["tau"].astype(float), ego["bag_ns"] / 1e9
    v = ego["v"].astype(float); boxes = actor_boxes(acts, data)
    to_tau = lambda t: float(np.interp(t, bag_s, tau))
    pos_t = {k: (np.interp(disp[k][0], bag_s, tau), disp[k][1]) for k in m}
    def objects_at(t):
        ob = []
        for j, (label, a, b, h, (sx, sy), L) in enumerate(boxes):
            k = next((k for k, jj in m.items() if jj == j), None)
            f = 0.0 if k is None or L < 0.3 else float(np.clip(np.interp(t, *pos_t[k]) / L, 0, 1))
            c = a + f * (b - a); ob.append((j, label, c[0], c[1], h, 0.0, 0.0, sx, sy))
        return ob
    reason = np.full(len(rows_tau), "no_event", object)
    for t_on, t_off, k in events:
        if k not in m: continue
        ea, eb = to_tau(t_on), to_tau(t_off)
        if not effective(tau, v, [(ea, eb)]): continue
        inside = (rows_tau >= ea - FUT * DT) & (rows_tau <= eb + RESTART_S)
        reason[inside & (reason == "no_event")] = ""
        first = None
        for g in np.flatnonzero((tau >= ea - FUT * DT - 1e-6) & (tau <= eb)):
            if visible(float(ego["x"][g]), float(ego["y"][g]), float(ego["yaw"][g]), objects_at(tau[g]), vis_cfg)[m[k]]:
                first = tau[g]; break
        cut = inside & (rows_tau < (first if first is not None else eb + RESTART_S + 1))
        reason[cut] = "unseen"
    return reason


def r2(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    idx = pd.read_parquet(a.index, columns=["episode", "i", "split"]).astype({"episode": str})
    old = pd.read_parquet(a.old_index, columns=["episode", "i"]).astype({"episode": str})
    idx["rescued"] = ~pd.MultiIndex.from_frame(idx[["episode", "i"]]).isin(pd.MultiIndex.from_frame(old))
    meta = pd.read_parquet(a.meta).astype({"episode": str}).set_index("episode")
    cnt, unseen_parts = {}, []
    for kind in ("A", "B"):
        eps = set(pd.read_parquet(Path(a.pools) / f"r2_{kind}.parquet").episode.astype(str))
        rows, st = [], {"episodes": len(eps), "with_moving_actor": 0, "events": 0, "effective_events": 0, "episodes_kept": 0,
                        "rescued_rows": 0, "rescued_kept": 0, "rescued_kept_by_brake_only": 0}
        for ep, g in idx[idx.episode.isin(eps)].groupby("episode"):
            mrow = meta.loc[ep]
            st["rescued_rows"] += int(g.rescued.sum())
            if not any(k > 0 for k in mrow.actor_kph): continue
            st["with_moving_actor"] += 1
            z = np.load(Path(a.ego) / f"{ep}.npz"); tau, v = z["tau"].astype(float), z["v"].astype(float)
            ivs = [tuple(x) for x in list(mrow.aeb_tau) + list(mrow.yield_tau)]
            ev = effective(tau, v, ivs); st["events"] += len(ivs); st["effective_events"] += len(ev)
            i = g.i.to_numpy(); by_event = r2_rows(tau[i], ev); res = g.rescued.to_numpy() & brakes(tau, v, i)
            keep = by_event | res
            st["rescued_kept"] += int((keep & g.rescued.to_numpy()).sum()); st["rescued_kept_by_brake_only"] += int((res & ~by_event).sum())
            if keep.any(): st["episodes_kept"] += 1; rows.append(g[keep])
        d = pd.concat(rows) if rows else idx.iloc[:0]
        d[["episode", "i"]].to_parquet(out / f"r2_{kind}_dyn.parquet", index=False)
        if a.r2logs:                                                    # rows whose braking object was not visible at t0
            u = unseen_of(a, kind, d); unseen_parts.append(u[u.reason == "unseen"])
            st.update({f"rows_{r or 'judged_seen'}": int((u.reason == r).sum()) for r in ("unseen", "no_event", "unjudgeable", "")},
                      train_rows_unseen=int(((u.reason == "unseen") & (d.split.to_numpy() == "train")).sum()))
        all_ = idx[idx.episode.isin(eps)]
        cnt[kind] = {**st, "rows": len(d), "rows_of_pool": len(all_), "train_rows": int((d.split == "train").sum()),
                     "train_rows_2hz": int(((d.split == "train") & (d.i % 5 == 0)).sum()), "pool_train_rows": int((all_.split == "train").sum())}
    if a.r2logs:
        pd.concat(unseen_parts)[["episode", "i", "reason"]].to_parquet(out / "r2_unseen.parquet", index=False)
    f = out / "pools.json"; j = json.load(open(f)) if f.exists() else {}
    j["r2_dynamic"] = cnt; json.dump(j, open(f, "w"), indent=1)
    print(json.dumps(cnt, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["r1", "r2"])
    for k in ("--index", "--ego", "--out"): p.add_argument(k, required=True)
    for k in ("--start-relabel", "--meta", "--pools", "--old-index", "--r2logs"): p.add_argument(k)
    a = p.parse_args()
    if a.what == "r2": return r2(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from morai_vla import build_index as bi                                    # scorer map (judged stop lines), as relabel_stage1
    cfg = bi.data.load_config(); stops = [bi.data.kmap().route_stops[sid] for sid in cfg["signal"]["stop_lines"]]
    sig, fg, idx = build(a.index, a.ego, stops)
    hw = pd.read_parquet(a.index, columns=["episode", "i", "split", "nav_text"]).astype({"episode": str})
    hw = hw[hw.nav_text.str.contains("No speed limit", regex=False)]
    hw[["episode", "i"]].to_parquet(out / "highway.parquet", index=False)
    rel = pd.read_parquet(a.start_relabel, columns=["episode", "i"]).astype({"episode": str})
    for name in ("approach", "hold"):
        sig[sig.pool == name][["episode", "i", "motion"]].to_parquet(out / f"signal_{name}.parquet", index=False)
    fg[["episode", "i", "reason"]].to_parquet(out / "future_green.parquet", index=False)
    pd.concat([rel, fg[["episode", "i"]]]).drop_duplicates().to_parquet(out / "r1_replay_exclude.parquet", index=False)
    two = lambda d: d[d.i % 5 == 0]                                             # the 2 Hz rows old sources draw from
    cnt = {"stop_lines_s": sorted(round(float(x), 1) for x in stops), "index_rows": len(idx)}
    for split in ("train", "val", "val_asset"):
        s, f, x = sig[sig.split == split], fg[fg.split == split], idx[idx.split == split]
        cnt[split] = {"rows": len(x), "rows_2hz": len(two(x)),
                      **{f"signal_{k}": len(s[s.pool == k]) for k in ("approach", "hold")},
                      **{f"signal_{k}_2hz": len(two(s[s.pool == k])) for k in ("approach", "hold")},
                      "signal_approach_motion": s[s.pool == "approach"].motion.value_counts().to_dict(),
                      "future_green": f.reason.value_counts().to_dict(), "future_green_2hz": two(f).reason.value_counts().to_dict()}
    cnt["start_relabel_rows"] = len(rel)
    cnt["highway"] = hw.split.value_counts().to_dict()
    f = out / "pools.json"; old = json.load(open(f)) if f.exists() else {}
    json.dump({**cnt, **({"r2_dynamic": old["r2_dynamic"]} if "r2_dynamic" in old else {})}, open(f, "w"), indent=1)
    print(json.dumps(cnt, indent=1))


if __name__ == "__main__":
    main()
