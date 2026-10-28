"""Keyframes from privileged ground truth, following AR1 Sec. 4.2 / Tab. 3:
reactive events -> one keyframe 0.5 s before the ego's behavior change; proactive ones -> the start of the range.
Each keyframe carries a closed-set decision (Tab. 1, from ego GT) and the privileged critical components (Tab. 2)
that will later be checked for visibility in the 2 s history window."""
import numpy as np

from . import data

HIST, FUT = 20, 60            # ticks (10 Hz): 2 s history, 6 s future
LEAD = 5                      # 0.5 s before the behavior change
LON = dict(set_speed="Set speed tracking", lead="Lead obstacle following", adapt="Speed adaptation (road events)",
           gap="Gap-searching", yield_="Yield (agent right-of-way)", stop="Stop for static constraints")
LAT = dict(keep="Lane keeping & centering", merge="Merge / Split", out_nudge="Out-of-lane nudge", in_nudge="In-lane nudge",
           lc="Lane change", turn="Turn")


def runs(mask):
    m = np.r_[False, np.asarray(mask, bool), False]; e = np.flatnonzero(np.diff(m.astype(int)))
    return list(zip(e[::2], e[1::2]))


def back_while(i, cond):
    while i > 0 and cond(i - 1): i -= 1
    return i


def kf(ep, kind, t0, lon, lat, direction=None, comps=(), verify=(), npc=None, **extra):
    t0 = max(int(t0), 0)                       # onsets near the bag start; _clean drops windows that do not fit
    return dict(episode=ep.name, kind=kind, t0=int(t0), decision=dict(longitudinal=lon, lateral=lat, direction=direction),
                components=list(comps), verify=list(verify), npc=npc, src={"lon": kind, "lat": kind}, **extra)


def signal_component(ep, feat, i, state=None):
    sid = feat.get("signal")
    judged = data.ROUTE["judged_signals"].get(sid) if sid not in (None, "none", "unjudged") else None
    cfg = data.load_config()["signal"]["stop_lines"].get(sid) if judged else None
    light = data.traffic_lights().get(cfg["lights"][0]) if cfg else None
    if light is None:
        x, y, _ = data.route_point(feat["stop_s"] + 6.0); light = np.array([x, y])
    G = ep.G; i = int(np.clip(i, 0, len(G) - 1))
    dist = float(np.hypot(*(light - (G.x[i], G.y[i]))))
    return dict(id="signal", category="Traffic lights", stop_line=sid, movement=feat["movement"],
                state=state if judged else "unknown", judged=bool(judged), four_lamp=bool(judged and judged.get("four_lamp")),
                distance_m=round(dist, 1))


def detect(ep):
    G = ep.G; n = len(G); v = G.v.to_numpy(); a = G.a.to_numpy(); s = G.path_s.to_numpy()
    front = s + data.BUMPER; out = []
    stopped = v < 0.3
    stops = [(k, e) for k, e in runs(stopped) if e - k >= 10]

    # ---- stops and departures at signals / stop lines / the roundabout yield line
    for k, e in stops:
        spawn = k == 0
        feat = next((f for f in data.ROUTE["features"] if f["type"] == "intersection" and -2 <= f["stop_s"] - front[k] <= 60), None)
        rbt = data.ROUTE["features"][[f["id"] for f in data.ROUTE["features"]].index("rbt")]
        at_rbt = -3 <= rbt["entry_s"] - front[k] <= 45
        dec = back_while(k, lambda i: a[i] < -0.2 and v[i] > 0.3)
        acc = e if e < n else None
        if feat and feat.get("signal") not in (None, "none"):
            judged = feat["signal"] != "unjudged"
            st = str(G.signal[k]) if judged else "unknown"
            if not spawn and (not judged or st in ("red", "yellow")):
                change = max([i for i in range(max(0, dec - 100), k) if G.signal[i] in ("red", "yellow") and G.signal[max(0, i - 1)] not in ("red", "yellow")], default=-1) if judged else -1
                t0 = max(dec - LEAD, change, 0)
                out.append(kf(ep, "signal_stop", t0, LON["stop"], LAT["keep"], comps=[signal_component(ep, feat, t0, str(G.signal[t0]) if judged else None)],
                              verify=["signal"], feature=feat["id"]))
            if acc is not None and not spawn:
                green = next((i for i in range(k, min(n, e + 30)) if G.signal[i] == "green"), None) if judged else None
                if judged and green is None: continue
                t0 = max(acc - LEAD, min(green + LEAD, acc)) if green is not None else acc - LEAD
                turn = feat["movement"] in ("left", "right") and abs(G.yaw[min(n - 1, t0 + 80)] - G.yaw[t0]) > 0.8
                out.append(kf(ep, "signal_go", t0, LON["set_speed"], LAT["turn"] if turn else LAT["keep"], feat["movement"] if turn else None,
                              comps=[signal_component(ep, feat, t0, "green" if judged else None)], verify=["signal"], feature=feat["id"]))
        elif at_rbt:
            if not spawn:
                out.append(kf(ep, "rbt_stop", dec - LEAD, LON["yield_"], LAT["keep"],
                              comps=[dict(id="rbt", category="Yield/Stop control", type="roundabout entrance yield line")],
                              npc="roundabout", feature="rbt"))
            if acc is not None and not spawn:
                out.append(kf(ep, "rbt_enter", acc - LEAD, LON["set_speed"], LAT["turn"], "right",
                              comps=[dict(id="rbt", category="Road events", type="roundabout")], npc="roundabout", feature="rbt"))
        elif data.feature_at(front[k], ("toll_plaza",)) and not spawn:
            out.append(kf(ep, "toll_stop", dec - LEAD, LON["stop"], LAT["keep"], comps=[dict(id="toll", category="Road events", type="toll gate")],
                          verify=["toll"], feature="toll"))
        elif data.feature_at(front[k], ("merge",)) and not spawn:
            out.append(kf(ep, "merge_wait", dec - LEAD, LON["gap"], LAT["keep"], comps=[dict(id="merge", category="Road events", type="acceleration lane")],
                          npc="merge", feature="merge"))
        elif feat and feat.get("signal") == "none" and not spawn:
            out.append(kf(ep, "stopline_stop", dec - LEAD, LON["stop"], LAT["keep"],
                          comps=[dict(id="stopline", category="Yield/Stop control", type="stop line", distance_m=round(feat["stop_s"] - front[k], 1))],
                          feature=feat["id"]))
        elif not spawn:
            out.append(kf(ep, "stop_unexplained", dec - LEAD, LON["stop"], LAT["keep"], npc="any"))

    # ---- passing a signalized intersection without stopping (proactive range start; 1 s before the line keeps the head in view)
    for f in data.ROUTE["features"]:
        if f["type"] != "intersection" or f.get("signal") in (None, "none"): continue
        idx = np.flatnonzero((front[:-1] < f["stop_s"]) & (front[1:] >= f["stop_s"]))
        for i in idx:
            if v[i] < 2.0 or any(k < i and i - e < 300 and f["stop_s"] - 62 <= front[k] <= f["stop_s"] for k, e in stops): continue
            t0 = max(0, int(i - 10))
            judged = f["signal"] != "unjudged"; st = str(G.signal[i]) if judged else None
            if judged and st not in ("green", "yellow"): continue
            # 0.5 rad as in qa.gt_check: a slow right turn at int551 reaches only ~0.7 rad within 6 s
            turn = f["movement"] in ("left", "right") and abs(G.yaw[min(n - 1, i + 60)] - G.yaw[i]) > 0.5
            out.append(kf(ep, "signal_pass", t0, LON["set_speed"], LAT["turn"] if turn else LAT["keep"], f["movement"] if turn else None,
                          comps=[signal_component(ep, f, t0, st)], verify=["signal"], feature=f["id"]))

    # ---- unsignalized turn (int73) and curves: decel onset for the geometry, else curve entry
    for f in data.ROUTE["features"]:
        if not (f["type"] == "curve" or (f["type"] == "intersection" and f.get("signal") == "none" and f["movement"] != "straight")): continue
        prev_end = max([g["s"][1] for g in data.ROUTE["features"] if g["s"][1] <= f["s"][0]], default=-1e9)
        inside = np.flatnonzero((s >= max(f["s"][0] - 60, prev_end)) & (s <= f["s"][0] + 10))
        if len(inside) == 0: continue
        seg = inside[np.r_[True, np.diff(inside) > 1]]          # one pass per lap
        for i0 in seg:
            win = [i for i in range(i0, min(n, i0 + 200)) if max(f["s"][0] - 60, prev_end) <= s[i] <= f["s"][0] + 10]
            dec = next((i for i in win if a[i] < -0.4 and v[i] > 2 and all(a[j] < -0.2 for j in range(i, min(n, i + 5)))), None)
            if dec is not None and v[dec] - v[dec:min(n, dec + 50)].min() < 1.0: dec = None   # a dip, not a slowdown
            enter = next((i for i in range(i0, min(n, i0 + 300)) if s[i] >= f["s"][0]), None)
            if f["type"] == "intersection":
                st = next((i for i in range(i0, min(n, i0 + 300)) if abs(G.yaw_rate[i]) > 0.08 and s[i] >= f["s"][0]), None)
                t0 = (dec - LEAD) if dec is not None else (st - LEAD if st is not None else None)
                if t0 is None: continue
                out.append(kf(ep, "turn", t0, LON["adapt"] if dec is not None else LON["set_speed"], LAT["turn"], f["movement"],
                              comps=[dict(id="junction", category="Road events", type="intersection", movement=f["movement"])], feature=f["id"]))
            else:
                t0 = (dec - LEAD) if dec is not None else enter
                if t0 is None: continue
                out.append(kf(ep, "curve", t0, LON["adapt"], LAT["keep"], f["dir"],
                              comps=[dict(id="curve", category="Road events", type="sharp curve" if f.get("sharp") else ("ramp curve" if f.get("ramp") else "curve"),
                                          dir=f["dir"], slows=dec is not None)], feature=f["id"]))

    # ---- roundabout: approach slowdown without a stop, exit turn
    rbt = next(f for f in data.ROUTE["features"] if f["id"] == "rbt")
    for i0 in _passes(s, rbt["entry_s"] - 60, rbt["entry_s"]):
        dec = next((i for i in range(i0, min(n, i0 + 200)) if s[i] <= rbt["entry_s"] and a[i] < -0.4 and v[i] > 2), None)
        if dec is not None and not any(k - 60 <= dec <= k for k, _ in stops):
            out.append(kf(ep, "rbt_slow", dec - LEAD, LON["adapt"], LAT["keep"], comps=[dict(id="rbt", category="Road events", type="roundabout ahead")],
                          npc="roundabout", feature="rbt"))
    for i0 in _passes(s, 905, rbt["s"][1]):
        st = next((i for i in range(i0, min(n, i0 + 100)) if G.yaw_rate[i] < -0.12), None)
        if st is not None:
            out.append(kf(ep, "rbt_exit", st - LEAD, LON["set_speed"], LAT["turn"], "right", comps=[dict(id="rbt", category="Road events", type="roundabout exit")], feature="rbt"))

    # ---- lane-line crossings: highway merge, construction lane shift, other lane changes
    P = np.c_[G.cx, G.cy]
    for i, side in data.crossings(P):
        f = data.feature_at(s[i])
        onset = _approach_onset(G, i)
        t0 = onset - LEAD
        slow = v[max(0, onset - 30):i + 1].min() < v[max(0, onset - 30)] - 1.5
        if f and f["type"] == "merge":
            out.append(kf(ep, "merge", t0, LON["gap"] if slow else LON["set_speed"], LAT["merge"], side,
                          comps=[dict(id="merge", category="Road events", type="on-ramp acceleration lane ending", dir=side)], npc="merge", feature="merge", cross_i=int(i)))
        elif any(abs(s[i] - act["s"]) < 40 for act in (ep.scene or {}).get("actors", []) if act["zone"] in ("static", "tunnel_static")):
            continue                                    # part of an obstacle bypass, labeled by _actors
        elif f and f["type"] == "toll_plaza":
            out.append(kf(ep, "toll_lane", t0, LON["adapt"] if slow else LON["set_speed"], LAT["lc"], side,
                          comps=[dict(id="toll", category="Routing intent", type="toll gate lanes ahead")], verify=["toll"], feature="toll", cross_i=int(i)))
        elif f and f["type"] == "construction":
            out.append(kf(ep, "work_shift", t0, LON["adapt"] if slow else LON["set_speed"], LAT["lc"], side,
                          comps=[dict(id="work", category="ODD constraints", type="construction drums closing the right lane")], verify=["work"], feature="work", cross_i=int(i)))
        elif f and f["id"] in data.ROUTE["crossing_exempt"]:
            continue
        else:
            out.append(kf(ep, "crossing_unexplained", t0, None, LAT["lc"], side, npc="any", cross_i=int(i)))

    # ---- construction zone / toll plaza slowdowns
    for fid, comp, kind in (("work", dict(id="work", category="ODD constraints", type="construction zone with drums and barriers"), "work_slow"),
                            ("toll", dict(id="toll", category="Road events", type="toll gate ahead"), "toll_slow")):
        f = next(x for x in data.ROUTE["features"] if x["id"] == fid)
        for i0 in _passes(s, f["s"][0] - 80, f["s"][1]):
            dec = next((i for i in range(i0, min(n, i0 + 300)) if f["s"][0] - 80 <= s[i] <= f["s"][1] and a[i] < -0.5 and v[i] > 3
                        and all(a[j] < -0.3 for j in range(i, min(n, i + 8)))), None)
            if dec is not None:
                out.append(kf(ep, kind, dec - LEAD, LON["adapt"], LAT["keep"], comps=[comp], verify=[fid], feature=fid))

    # ---- scenario actors (manifest): static obstacles and pedestrians
    if ep.scene:
        out += _actors(ep, stops)

    # ---- highway slowdowns not explained by the route: lead vehicle candidates
    hw = G.route_section.between(10, 12).to_numpy()
    for k, e in runs((a < -0.6) & hw & (v > 3)):
        if e - k < 8 or data.feature_at(s[k], ("curve", "construction", "toll_plaza")): continue
        if any(abs(o["t0"] - (k - LEAD)) < 20 for o in out): continue
        if data.feature_at(s[k], ("merge",)):          # slowing in the acceleration lane: letting mainline traffic pass
            out.append(kf(ep, "merge_slow", k - LEAD, LON["gap"], LAT["keep"], npc="merge", feature="merge"))
        else:
            out.append(kf(ep, "hw_slow", k - LEAD, LON["lead"], LAT["keep"], npc="lead"))

    return _clean(ep, out)


def _passes(s, lo, hi):
    idx = np.flatnonzero((s >= lo) & (s <= hi))
    return list(idx[np.r_[True, np.diff(idx) > 1]]) if len(idx) else []


def _approach_onset(G, i):
    """first tick of the steady lateral approach that ends in the crossing at i (|heading change| or |v_lat| rising)."""
    vl = G.v_lat.to_numpy(); sgn = np.sign(np.nanmean(vl[max(0, i - 3):i + 4])) or 1.0
    j = i
    while j > 0 and i - j < 60 and sgn * vl[j - 1] > 0.15: j -= 1
    return j


def _actors(ep, stops):
    G = ep.G; n = len(G); v = G.v.to_numpy(); a = G.a.to_numpy(); cam = G.cam_s.to_numpy()
    dev = np.abs(G.d.to_numpy()); out = []; used = set()
    for act in sorted(ep.scene.get("actors", []), key=lambda q: q["s"]):
        s_a = act["s"]; tunnel = act["zone"].startswith("tunnel")
        if act["zone"] in ("static", "tunnel_static"):
            aid = act.get("id", int(s_a * 10))          # early manifests (rs2020-) carry no DataID
            obj = "parked vehicle" if aid in data.VEHICLE_PROPS else "obstacle"
            side = "right" if act["lat"] < 0 else ("left" if act["lat"] > 0.15 else "centre")
            ahead = (s_a - cam > 0) & (s_a - cam < 45)
            cand = [i for i in range(1, n) if ahead[i] and dev[i] > 0.3 and dev[i] > dev[i - 1] and v[i] > 0.3]
            comp = dict(id=f"actor{aid}", category="Critical objects", type=obj, asset_id=act.get("id"), side=side,
                        motion="stationary", tunnel=tunnel, s=s_a, lat=act["lat"])
            if cand:
                on = back_while(cand[0], lambda i: dev[i] > 0.1 and dev[i] > dev[max(0, i - 1)])
                if any(abs(on - u) < 50 for u in used): continue
                used.add(on)
                mx = float(dev[on:on + 60].max()); dirn = "left" if np.nansum(G.d[on:on + 60]) > 0 else "right"
                slow = v[min(n - 1, on + 30)] < v[on] - 1.0
                comp["distance_m"] = round(float(s_a - cam[on - LEAD]), 1)
                out.append(kf(ep, "bypass", on - LEAD, LON["adapt"] if slow else LON["set_speed"], LAT["out_nudge"] if mx > 1.0 else LAT["in_nudge"], dirn,
                              comps=[comp], verify=[comp["id"]], actor=aid))
            for k, e in stops:      # held behind the obstacle (e.g. waiting for room to pass)
                if 0 < s_a - (G.path_s[k] + data.BUMPER) < 15:
                    dec = back_while(k, lambda i: a[i] < -0.2 and v[i] > 0.3)
                    c = dict(comp, distance_m=round(float(s_a - cam[max(0, dec - LEAD)]), 1))
                    out.append(kf(ep, "obstacle_stop", dec - LEAD, LON["stop"], LAT["keep"], comps=[c], verify=[c["id"]], actor=aid))
        else:
            trig = [i for i in range(n) if 0 < s_a - (cam[i] + 1.9) <= act["active"]]
            if not trig: continue
            dec = next((i for i in range(trig[0], min(n, trig[0] + 60)) if a[i] < -0.5 and 0 < s_a - cam[i] < 40), None)
            if dec is None: continue
            t0 = back_while(dec, lambda i: a[i] < -0.2) - LEAD
            stops_ = bool((v[t0:t0 + 60] < 0.3).any())
            aid = act.get("id", int(s_a * 10))
            comp = dict(id=f"actor{aid}", category="Critical objects", type=data.PED_NAMES.get(aid, "pedestrian"), asset_id=act.get("id"),
                        motion="crossing ahead", tunnel=tunnel, s=s_a, distance_m=round(float(s_a - cam[t0]), 1))
            out.append(kf(ep, "ped_yield", t0, LON["yield_"], LAT["keep"], comps=[comp], verify=[comp["id"]], actor=aid, stops=stops_))
    return out


LON_RANK = [LON["stop"], LON["yield_"], LON["lead"], LON["gap"], LON["adapt"], LON["set_speed"], None]


STOPS = {"signal_stop", "rbt_stop", "obstacle_stop", "stopline_stop", "toll_stop", "merge_wait", "stop_unexplained"}
GOES = {"signal_go", "rbt_enter"}


def _merge(a, b):
    """fold b into a: the stronger longitudinal decision and any lateral maneuver win; components are pooled.
    A departure right after a short stop is dropped: the first action after the keyframe is the stop (AR1 Tab. 1)."""
    ka, kb = set(a["kind"].split("+")), set(b["kind"].split("+"))
    if (ka & STOPS and kb & GOES) or (kb & STOPS and ka & GOES):
        return a if ka & STOPS else b
    if ka & (GOES | {"signal_pass"}) and kb & {"bypass"}:     # departure, then the bypass a few seconds later:
        a["decision"]["lateral"], a["decision"]["direction"] = b["decision"]["lateral"], b["decision"]["direction"]   # lon stays the departure
        a["src"]["lat"] = b["src"]["lat"]; a["components"] += [c for c in b["components"] if c["id"] not in {x["id"] for x in a["components"]}]
        a["verify"] = sorted(set(a["verify"]) | set(b["verify"])); a["kind"] = "+".join(sorted(ka | kb)); a["actor"] = b.get("actor")
        return a
    da, db = a["decision"], b["decision"]
    if LON_RANK.index(db["longitudinal"]) < LON_RANK.index(da["longitudinal"]): da["longitudinal"] = db["longitudinal"]; a["src"]["lon"] = b["src"]["lon"]
    if da["lateral"] == LAT["keep"] and db["lateral"] not in (None, LAT["keep"]):
        da["lateral"], da["direction"] = db["lateral"], db["direction"]; a["src"]["lat"] = b["src"]["lat"]
    a["components"] += [c for c in b["components"] if c["id"] not in {x["id"] for x in a["components"]}]
    a["verify"] = sorted(set(a["verify"]) | set(b["verify"])); a["npc"] = a["npc"] or b["npc"]
    a["kind"] = "+".join(sorted(set(a["kind"].split("+")) | set(b["kind"].split("+"))))
    return a


def _clean(ep, out):
    """drop keyframes without a full 2 s + 6 s window or inside the post-spawn settle; fold keyframes <=1.5 s apart."""
    n = len(ep.G); G = ep.G; keep = []
    for k in sorted(out, key=lambda o: o["t0"]):
        if k["t0"] < max(HIST, 30) or k["t0"] + FUT >= n: continue
        i = k["t0"]
        cam = float(G.cam_s[i])
        for c in k["components"]:
            f = data.ROUTE_BY_ID.get(c["id"] if c["id"] in ("work", "toll", "merge", "rbt") else k.get("feature") if c["id"] in ("curve", "junction") else None)
            if f is not None:
                c["feature_dist_m"] = round((f.get("devices_s") or f.get("gate_s") or f["s"][0]) - cam, 1)
                if f.get("drums_s"): c["drums_dist_m"] = round(f["drums_s"] - cam, 1)
                c["behind_tunnel_exit"] = bool(1868 <= cam < 1986 <= f["s"][0])
                if c["id"] == "curve":              # the bend visible in the next 40 m may turn the other way first
                    dh = data.route_heading(cam + 40) - data.route_heading(cam)
                    c["visible_bend"] = "left" if dh > 0.1 else "right" if dh < -0.1 else "straight"
        k.update(tau0=round(float(G.tau[i]), 2), path_s=round(float(G.path_s[i]), 1), section=int(G.route_section[i]), standing=bool(G.v[i] < 0.3),
                 cell=data.cell_of(G.route_section[i]), env=ep.env(i), clean=bool(G.clean[i]),
                 frames_bag_ns=[int(G.bag_ns[i + 5 * q]) for q in range(-4, 13)])
        near = next((o for o in keep if abs(o["t0"] - k["t0"]) <= 15
                     or ("bypass" in k["kind"] and set(o["kind"].split("+")) & (GOES | {"signal_pass"}) and 0 <= k["t0"] - o["t0"] <= 50)), None)
        if near is None: keep.append(k)
        elif _merge(near, k) is k: keep[keep.index(near)] = k     # a stop replaces a departure listed first
    return keep
