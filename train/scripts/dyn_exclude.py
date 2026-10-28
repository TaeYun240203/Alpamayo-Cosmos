"""Replay exclusions for the dynamic-stop fine-tune (student_dyn.yaml). CPU, no model. One parquet (episode, i) per reason
and their union with the stop round's replay_exclude (start relabel + ring waits > 5 s) and every dyn_relabel pool row
(one input, one label). R1 and R2 episode names never collide, so one list serves both roots.
  ring_crawl  (a) index rows crawling 0.5-3.2 m/s inside s 880-925 now: the lanelet seams (s 885.1 11 deg, s 914.8 7 deg)
              cap roundabout_merge's curvature speed at 2.3-2.8 m/s. Optional (not in the union): ring_crawl_future, rows
              at s 855-930 whose next 6.4 s crawls there >= 1 s (99 % of the ring_entry pool)
  ring_stop   (a) standing (v < 0.5) inside the ring, s 887-925 (past the entry line s 879-885). No Object Info in the
              bags: a stop behind a ring car cannot be told apart; optional, not in the union
  prevision   (b) dyn_relabel r2_prevision_exclude.parquet
  late_lc     (c) highway (s 1118.7-1741.7) lane changes (|d(t+3 s) - d(t)| > 2.5 m) with a speed drop >= 20 kph from
              >= 40 kph (not to a stop): rows from 6.4 s before the top speed ahead of the drop to the end of the change
  prop_close  (d) tunnel_static props passed with < 0.3 m between the Ego box and the scaled footprint (1 m x scale),
              frames bridging a GT jump / respawn (|ds/dt - v| >= 5 m/s, the 10 Hz grid interpolates across it) not counted
  static_stop optional (not in the union): standing (v < 0.5) with a lane-intruding static prop 0-30 m ahead of the nose
Pools written next to OUT (in DYN_OUT, user 10/28):
  static_avoid.parquet  (episode, i, zone, clearance m, side: the Ego passes left / right of the prop) R1 + R2 rows avoiding a static prop (static, tunnel_static, ra_parked) that intrudes on the lane
              (the lane-centre path would pass within 0.3 m of its real footprint) with >= 0.3 m clearance, from 6.4 s
              before the Ego box reaches it to the last frame beside it; rows whose next 6.4 s (up to passing) stand or
              crawl (v < 1.0 m/s) out, then the union out ((c) late lane changes, (d) close passes ...); episodes the box downloads only
              (the stop round's R1 replay_episodes, assets/ft/r2_AB.parquet)
  r2_cutin_stop.parquet R2 A AEB / yield stop windows caused by cut-in cars only (dyn_relabel r2_stop_windows) minus the union
Footprint of a static prop: 1 x 1 m (vehicles and parked cars 4.6 x 1.9 m) x its scale (R1 `scale` [along, across, height];
R2 `scale_z` is the height). Object Info reports the unscaled size, which is why the collection driver passed some too close.
Usage (cwd = workspace): python train/scripts/dyn_exclude.py R1_ROOT R2_ROOT DYN_OUT STOP_REPLAY_EXCLUDE OUT
"""
import json, sys
from pathlib import Path

import numpy as np, pandas as pd

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "tools/coc_labeler"))
from coc_labeler import data  # noqa: E402

N, FRONT, REAR, HALF_W = 64, 3.9, 0.8, 0.95
OPTIONAL = ("ring_crawl_future", "ring_stop", "static_stop")        # own lists, not in the union: the user decides
STATIC = ("static", "tunnel_static", "ra_parked")
CAR_M = (4.6, 1.9)


def footprint(p):
    """(along the route, across) metres of a static prop (module docstring)."""
    bx, by = CAR_M if p.get("zone") == "ra_parked" or p.get("id") in data.VEHICLE_PROPS else (1.0, 1.0)
    sx, sy = p["scale"][:2] if "scale" in p else (1.0, 1.0)
    return bx * sx, by * sy


def windows(k, lo=N, hi=0):
    """rows i with some flagged frame in [i, i + lo] (i in [k - lo, k + hi])."""
    return np.unique(np.concatenate([np.arange(max(a - lo, 0), a + hi + 1) for a in k])) if len(k) else np.zeros(0, int)


def episode(root, ep, sc):
    z = np.load(Path(root) / "_ego" / f"{ep}.npz"); s, v, d = z["path_s"], z["v"], z["d"]; n = len(v)
    out = {}
    crawl = (s >= 880) & (s <= 925) & (v >= 0.5) & (v <= 3.2)
    c = np.r_[0, np.cumsum(crawl)]; i = np.arange(n); j = np.minimum(i + N, n - 1)
    out["ring_crawl"] = i[crawl]                                                   # crawling now
    out["ring_crawl_future"] = i[(s >= 855) & (s <= 930) & (c[j + 1] - c[i] >= 10) & ~crawl]   # optional: hits ~all ring_entry rows
    out["ring_stop"] = i[(s >= 887) & (s <= 925) & (v < 0.5)]
    hw = (s >= 1118.7) & (s <= 1741.7) & (np.abs(d) < 8)
    k = i[hw & (i + 30 < n)]; k = k[np.abs(d[np.minimum(k + 30, n - 1)] - d[k]) > 2.5]
    lc = []
    for e in np.split(k, np.flatnonzero(np.diff(k) > 30) + 1) if len(k) else []:
        a, b = int(e[0]), int(min(e[-1] + 30, n - 1)); w0 = max(a - 60, 0)
        km = int(np.argmin(v[w0:b + 1])) + w0
        st = int(np.argmax(v[w0:km + 1])) + w0; vb = v[st]                         # the drop starts at the top speed before it
        if vb * 3.6 >= 40 and (vb - v[km]) * 3.6 >= 20 and v[km] * 3.6 >= 5:
            lc.append((st, b, round(float(vb * 3.6), 1), round(float(v[km] * 3.6), 1), round(float(s[a]), 1)))
    out["late_lc"] = np.unique(np.concatenate([np.arange(max(st - N, 0), b + 1) for st, b, *_ in lc])) if lc else np.zeros(0, int)
    tau = z["tau"]; ds = np.abs(np.diff(s)) / np.maximum(np.diff(tau), 1e-3)
    real = np.r_[True, np.abs(ds - v[1:]) < 5.0] & np.r_[np.abs(ds - v[:-1]) < 5.0, True]   # not a GT jump / respawn bridge
    near, avoid, events = np.zeros(n, bool), {}, []                                 # avoid: row -> (zone, clearance, side)
    stand = np.zeros(n, bool)
    for p in (sc or {}).get("actors", []):
        if p.get("zone") not in STATIC or "s" not in p: continue
        sx, sy = footprint(p); hx, hy, lat = 0.5 * sx, 0.5 * sy, p.get("lat", 0.0)
        over = (s + FRONT >= p["s"] - hx) & (s - REAR <= p["s"] + hx) & real
        gap = np.abs(d - lat) - HALF_W - hy
        if p["zone"] == "tunnel_static": near |= over & (gap < 0.3)
        if abs(lat) - hy >= HALF_W + 0.3: continue                                 # beside the lane: no avoiding needed
        ahead = p["s"] - hx - (s + FRONT)
        stand |= (v < 0.5) & (ahead > 0) & (ahead < 30) & real
        k = np.flatnonzero(over)
        for run in np.split(k, np.flatnonzero(np.diff(k) > 1) + 1) if len(k) else []:
            c = float(gap[run].min())
            events.append((p["zone"], round(c, 2), c >= 0.3))
            if c < 0.3: continue
            kr, kp = int(run[0]), int(run[-1]); side = "left" if float(np.mean(d[run] - lat)) > 0 else "right"   # Ego passes on that side
            for i_ in range(max(kr - N, 0), kp + 1):
                if v[i_:min(i_ + N, kp) + 1].min() >= 1.0 and (i_ not in avoid or c < avoid[i_][1]):   # no standing or crawling
                    avoid[i_] = (p["zone"], round(c, 2), side)
    out["prop_close"] = windows(np.flatnonzero(near))
    out["static_stop"] = i[stand]
    out["static_avoid"] = avoid
    return out, lc, int(near.sum()), events


def main():
    r1, r2, dyn, stop_ex, out = map(Path, sys.argv[1:6]); out.mkdir(parents=True, exist_ok=True)
    rows = {k: [] for k in ("ring_crawl", "ring_crawl_future", "ring_stop", "late_lc", "prop_close", "static_stop", "static_avoid")}
    lcs, near_frames, events = [], 0, []
    for root in (r1, r2):
        idx = pd.read_parquet(root / "_index/index.parquet", columns=["episode", "i"]).astype({"episode": str})
        for ep, g in idx.groupby("episode"):
            _, sc = data.scene_of(ep)
            o, lc, nf, ev = episode(root, ep, sc); near_frames += nf; events += [(ep, *e) for e in ev]
            lcs += [(ep, *x) for x in lc]
            have = set(g.i)
            av = o.pop("static_avoid"); rows["static_avoid"] += [(ep, int(x), *av[x]) for x in sorted(av) if int(x) in have]
            for k, ii in o.items(): rows[k] += [(ep, int(x)) for x in ii if int(x) in have]
    avoid = pd.DataFrame(rows.pop("static_avoid"), columns=["episode", "i", "zone", "clearance", "side"])
    parts = {k: pd.DataFrame(v, columns=["episode", "i"]) for k, v in rows.items()}
    parts["prevision"] = pd.read_parquet(dyn / "r2_prevision_exclude.parquet")[["episode", "i"]]
    parts["dyn_pool"] = pd.read_parquet(dyn / "dyn_relabel_index.parquet")[["episode", "i"]]
    parts["stop_round"] = pd.read_parquet(stop_ex)[["episode", "i"]]
    for k, d in parts.items(): d.astype({"episode": str}).to_parquet(out / f"ex_{k}.parquet", index=False)
    allx = pd.concat([d for k, d in parts.items() if k not in OPTIONAL]).astype({"episode": str}).drop_duplicates()
    allx.to_parquet(out / "replay_exclude_dyn.parquet", index=False)
    pd.DataFrame(lcs, columns=["episode", "i_start", "i_end", "kph_before", "kph_min", "s"]).to_parquet(out / "late_lc_events.parquet", index=False)
    mx = pd.MultiIndex.from_frame(allx)
    ok = set(pd.read_parquet(stop_ex.parent / "replay_episodes.parquet").episode.astype(str))      # what the box downloads: the
    ok |= set(pd.read_parquet(WS / "train/assets/ft/r2_AB.parquet").episode.astype(str))            # stop round's R1 + allowed R2 A/B
    avoid = avoid[avoid.episode.isin(ok) & ~pd.MultiIndex.from_frame(avoid[["episode", "i"]]).isin(mx)]; avoid.to_parquet(dyn / "static_avoid.parquet", index=False)
    w = pd.read_parquet(dyn / "r2_stop_windows.parquet"); cut = w[w.cause == "cutin"][["episode", "i"]].drop_duplicates()
    cut = cut[~pd.MultiIndex.from_frame(cut.astype({"episode": str})).isin(mx)]; cut.to_parquet(dyn / "r2_cutin_stop.parquet", index=False)
    ev = pd.DataFrame(events, columns=["episode", "zone", "clearance", "clean"]); ev.to_parquet(out / "static_pass_events.parquet", index=False)
    split = pd.concat([pd.read_parquet(r / "_index/index.parquet", columns=["episode", "i", "split"]).astype({"episode": str}) for r in (r1, r2)])
    def brk(d):                                                                    # rows by data (R1 / R2) and split
        m = d.astype({"episode": str}).merge(split, on=["episode", "i"]); r2_ = m.episode.str.startswith("r2")
        return dict(total=len(d), r1=int((~r2_).sum()), r2=int(r2_.sum()), r1_train=int((~r2_ & (m.split == "train")).sum()),
                    r2_train=int((r2_ & (m.split == "train")).sum()))
    s = {k: brk(d) for k, d in parts.items()} | dict(union=brk(allx), late_lc_events=len(lcs), prop_close_frames=near_frames,
                                                     static_avoid=brk(avoid[["episode", "i"]]), r2_cutin_stop=brk(cut),
                                                     static_pass_events={f"{z}/{'clean' if c else 'close'}": int(n) for (z, c), n in ev.groupby(["zone", "clean"]).size().items()},
                                                     index_rows=dict(r1=int((~split.episode.str.startswith("r2")).sum()), r2=int(split.episode.str.startswith("r2").sum())))
    json.dump(s, open(out / "exclude_summary.json", "w"), indent=1); print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
