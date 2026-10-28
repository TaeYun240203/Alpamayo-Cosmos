"""LLM-free CoC labeler from privileged info (labels parquet + scenario manifest + map), AR1 Sec. 4 rules:
keyframe 0.5 s before the behavior change (Tab. 3), closed-set decision (Tab. 1), critical components (Tab. 2) gated to
what is observable in the 2 s history (causal locality), template trace in the release-prompt example style.
shortcut: covers signal stop/resume, static-obstacle bypass and pedestrian yield only (C1-C3, C6); curves/turns not labeled."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/taeyunkim/Library/Mobile Documents/com~apple~CloudDocs/Workspaces/Alpamayo-Cosmos/tools/scorer")
sys.path.insert(0, "/Users/taeyunkim/Library/Mobile Documents/com~apple~CloudDocs/Workspaces/Alpamayo-Cosmos/tools")
from scorer import load_config, KMap
from ledger import env_table, env_at
import prep

H = os.path.dirname(os.path.abspath(__file__))
CFG = load_config(); KM = KMap.from_files(CFG)
STOPS = {sid: KM.route_stops[sid] for sid in CFG["signal"]["stop_lines"]}
TL = {t["idx"]: np.asarray(t["point"][:2], float) for t in json.load(open(
    "/Users/taeyunkim/Library/Mobile Documents/com~apple~CloudDocs/Workspaces/lanelet_editor/outputs/lanelet-map-repair/map/source/traffic_light_set.json"))}
VEHICLE_PROPS = {40100008, 40100009, 40100010, 40100011}
EPS = {"run1021i-w1-006": "rs2035_044", "run1021i-w2-010": "rs2035_019", "run1021h-w2-001": "rs2035_023",
       "run1021v-w1-018": "rs2040_040", "run1021s-w1-004": "rs2039_038"}
CAM_FWD = 2.0   # m from base_link (rear axle) to the front camera, for distances


def lateral_offset(x, y):
    """signed distance to the global path (left +)."""
    P = KM.path[np.r_[True, np.hypot(*np.diff(KM.path, axis=0).T) > 1e-6]]; out = np.empty(len(x))
    for k, p in enumerate(np.c_[x, y]):
        i = int(np.argmin(np.hypot(*(P[:-1] - p).T))); a, b = P[i], P[i + 1]
        u = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1); f = a + u * (b - a)
        out[k] = np.sign((b - a)[0] * (p - f)[1] - (b - a)[1] * (p - f)[0]) * np.hypot(*(p - f))
    return out


def episode_frame(ep):
    G, _ = prep.episode(ep)
    w = ep.split("-")[1]
    d = pd.read_parquet(os.path.join(os.path.expanduser("~/.claude-morai-mac/labels-cache"), "mac-win-1" if w == "w1" else "mac-win-2", ep + ".labels.parquet"))
    i = np.clip(np.searchsorted(d.t.to_numpy(), G.t.to_numpy()), 0, len(d) - 1)
    G["path_s"] = np.interp(G.t, d.t, d.path_s.ffill().bfill()); G["signal"] = d.signal.to_numpy()[i]
    cx, cy = G.x + 1.4 * np.cos(G.yaw), G.y + 1.4 * np.sin(G.yaw)
    G["d"] = lateral_offset(cx.to_numpy(), cy.to_numpy())
    G["a"] = np.convolve(np.gradient(G.v, 0.1), np.ones(5) / 5, "same")
    fog = env_at(env_table(w), d.bag_time_ns.iloc[0] / 1e9).startswith("F")
    return G, fog


def onset_back(G, n, cond):
    """walk back from tick n while cond holds; returns the first tick of the run."""
    while n > 0 and cond(n - 1): n -= 1
    return n


def label_episode(ep, scene):
    G, fog = episode_frame(ep)
    sc = json.load(open(os.path.join(H, scene[:6] + "-manifest.json")))[scene]
    rng = np.random.default_rng(abs(hash(ep)) % 2 ** 32)
    vis_obj, vis_tl = (40.0, 60.0) if fog else (60.0, 80.0)
    cam_s = G.path_s.to_numpy() + CAM_FWD; v = G.v.to_numpy(); a = G.a.to_numpy(); n = len(G)
    out = []

    def add(t0, lon, lat, comps, text, cause_dist, vis_lim, kind):
        if t0 - 20 < 0 or t0 + 60 >= n: return
        observed = cause_dist is not None and 0 < cause_dist <= vis_lim
        out.append(dict(ep=ep, kind=kind, t0=int(t0), tau0=round(float(G.tau[t0]), 2), section=int(G.sec[t0]), fog=fog,
                        decision={"longitudinal": lon, "lateral": lat}, critical_components=comps,
                        observed_in_history=bool(observed), cause_dist_m=None if cause_dist is None else round(cause_dist, 1),
                        coc=text if observed else "UNOBSERVED"))

    # (a) stop / (b) resume at the judged signals
    stopped = v < 0.3
    for sid, s_stop in STOPS.items():
        light = TL.get(CFG["signal"]["stop_lines"][sid]["lights"][0]); left = CFG["signal"]["stop_lines"][sid]["movement"] == "left"
        near = (s_stop - (cam_s + 1.9) > -1) & (s_stop - (cam_s + 1.9) < 60)  # driver holds ~30-40 m early
        k = 0
        while k < n:
            if not (stopped[k] and near[k]): k += 1; continue
            e = k
            while e < n and stopped[e]: e += 1
            if e - k >= 10 and G.signal[k] in ("red", "yellow"):
                dec = onset_back(G, k, lambda i: a[i] < -0.2 and v[i] > 0.3)
                change = max([i for i in range(max(0, dec - 100), k) if G.signal[i] in ("red", "yellow") and G.signal[max(0, i - 1)] not in ("red", "yellow")], default=-1)
                t0 = max(dec - 5, change)
                st = G.signal[t0] if G.signal[t0] in ("red", "yellow") else "red"
                dl = float(np.hypot(*(light - (G.x[t0], G.y[t0])))) if light is not None else None
                add(t0, "Stop for static constraints", "Lane keeping & centering",
                    [{"category": "Traffic lights", "state": st, "distance_m": round(dl, 1) if dl else None}, {"category": "Yield/Stop control", "type": "stop line"}],
                    rng.choice([f"Stop for the {st} traffic light ahead.", f"Decelerate to a stop because the traffic light ahead is {st}.",
                                f"Slow down and stop before the intersection for the {st} traffic light."]), dl, vis_tl, "stop_tl")
                if e < n:  # resume
                    green = min([i for i in range(k, min(n, e + 1)) if G.signal[i] == "green"], default=e)
                    t1 = max(e - 5, green)
                    dl = float(np.hypot(*(light - (G.x[t1], G.y[t1])))) if light is not None else None
                    add(t1, "Set speed tracking", "Turn" if left else "Lane keeping & centering",
                        [{"category": "Traffic lights", "state": "green arrow" if left else "green", "distance_m": round(dl, 1) if dl else None}],
                        ("Turn left at the intersection as the left-turn arrow turns green." if left else
                         rng.choice(["Resume speed through the intersection because the light has turned green.",
                                     "Accelerate toward the intersection as the traffic light turns green."])),
                        dl, vis_tl, "resume_tl")
            k = e

    done_onsets = set()
    for act in sorted(sc["actors"], key=lambda q: q["s"]):
        s_a = act["s"]; tunnel = act["zone"].startswith("tunnel"); where = " in the tunnel" if tunnel else ""
        if act["zone"] in ("static", "tunnel_static"):
            obj = "parked vehicle" if act["id"] in VEHICLE_PROPS else "obstacle"
            side = "right" if act["lat"] < 0 else "left"
            ahead = (s_a - cam_s > 0) & (s_a - cam_s < 45)
            dev = np.abs(G.d.to_numpy())
            cand = [i for i in range(1, n) if ahead[i] and dev[i] > 0.3 and dev[i] > dev[i - 1] and not stopped[i]]
            if not cand: continue
            on = onset_back(G, cand[0], lambda i: dev[i] > 0.1 and dev[i] > dev[max(0, i - 1)])
            if any(abs(on - q) < 50 for q in done_onsets): continue
            done_onsets.add(on)
            t0 = on - 5; mx = float(G.d[on:on + 60].abs().max()); sgn = np.sign(G.d[on:on + 60].sum())
            dirn = "left" if sgn > 0 else "right"
            slow = v[min(n - 1, t0 + 30)] - v[t0] < -1.0
            lat = "Out-of-lane nudge" if mx > 1.0 else "In-lane nudge"
            lon = "Speed adaptation (road events)" if slow else "Set speed tracking"
            pre = "Slow down and nudge" if slow else "Nudge"
            text = (f"{pre} {dirn} out of the lane to pass the {obj} on the {side}{where}." if mx > 1.0 else
                    f"{pre} {dirn} within the lane to keep clearance from the {obj} on the {side}{where}.")
            add(t0, lon, lat + f" ({dirn})", [{"category": "Critical objects", "type": obj, "pose": f"in-path, {side} side", "motion": "stationary"}],
                text, float(s_a - cam_s[t0]), vis_obj, "bypass")
        else:   # pedestrian
            trig = [i for i in range(n) if 0 < s_a - (cam_s[i] + 1.9) <= act["active"]]
            if not trig: continue
            dec = [i for i in range(trig[0], min(n, trig[0] + 60)) if a[i] < -0.5 and 0 < s_a - cam_s[i] < 40]
            if not dec: continue
            t0 = onset_back(G, dec[0], lambda i: a[i] < -0.2) - 5
            stops = bool(stopped[t0:t0 + 60].any())
            add(t0, "Yield (agent right-of-way)", "Lane keeping & centering",
                [{"category": "Critical objects", "type": "pedestrian", "pose": "ahead, crossing", "motion": "crossing"}],
                rng.choice([f"Stop to yield to the pedestrian crossing the road ahead{where}." if stops else f"Decelerate to yield to the pedestrian crossing ahead{where}.",
                            f"Yield to the pedestrian crossing in front of the ego vehicle{where}."]),
                float(s_a - cam_s[t0]), vis_obj, "yield_ped")
    return G, out


if __name__ == "__main__":
    allk = []
    for ep, scene in EPS.items():
        G, ks = label_episode(ep, scene)
        for k in ks:
            idx = [k["t0"] + 5 * q for q in range(-4, 13)]; rel = [0.5 * q for q in range(-4, 13)]
            k["bag_ns"] = [int(G.bag_ns[i]) for i in idx]
            k["meta"] = "\nMeta-actions by sampled timestamp (relative to event, unit: seconds):\n" + "\n".join(
                f"(t={r:+.1f}s) longitudinal: {G.lon[i]}, lateral: {G.lat[i]}, lane: {G.lane[i]}" for r, i in zip(rel, idx))
            k["ego"] = ("Ego speed time series (relative to event time t=0.0s):\n- Longitudinal (m/s): [" + ", ".join(f"(t={r:.1f}s, v_long={G.v[i]:.3f} m/s)" for r, i in zip(rel, idx)) +
                        "]\n- Lateral (m/s): [" + ", ".join(f"(t={r:.1f}s, v_lat={G.v_lat[i]:.3f} m/s)" for r, i in zip(rel, idx)) + "]\nPositive lateral speed means rightward motion, negative means leftward motion.")
            print(ep[-10:], k["kind"], "s%d" % k["section"], "tau0", k["tau0"], "fog" if k["fog"] else "sun", "dist", k["cause_dist_m"], "|", k["coc"])
        allk += ks
    json.dump(allk, open(os.path.join(H, "priv_keyframes.json"), "w"), indent=1, default=str)
    print(len(allk), "keyframes,", sum(k["observed_in_history"] for k in allk), "observed")
