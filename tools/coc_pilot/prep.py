"""Pilot keyframes for C4/C5 CoC: sim-clock tau, autolabeler-style meta-actions, 17-frame windows (2 s hist + 6 s fut @ 2 Hz).
shortcut: lane meta-actions use MGeo (road_id, ego_lane) changes + curvature only, no slight-shift detection; fine for a pilot."""
import json, os, sys
import numpy as np, pandas as pd

CACHE = os.path.expanduser("~/.claude-morai-mac/labels-cache")
LINKS = "/Users/taeyunkim/Library/Mobile Documents/com~apple~CloudDocs/Workspaces/lanelet_editor/outputs/lanelet-map-repair/map/source/link_set.json"
EPS = ["run1021v-w1-018", "run1021s-w1-004", "run1021i-w1-007", "run1021v-w1-144"]
PER_EP = 5
OUT = os.path.join(os.path.dirname(__file__), "keyframes.json")

# --- map: link segments with (road_id, ego_lane)
segs = []
for L in json.load(open(LINKS)):
    p = np.asarray(L["points"], float)[:, :2]
    if len(p) < 2: continue
    for a, b in zip(p[:-1], p[1:]):
        segs.append((*a, *b, int(L["road_id"]) if str(L["road_id"]).isdigit() else -1, L["ego_lane"] or 0))
S = np.asarray(segs, float)
SA, SB, SH = S[:, 0:2], S[:, 2:4], np.arctan2(S[:, 3] - S[:, 1], S[:, 2] - S[:, 0])


def lane_of(xy, yaw):
    d = SB - SA; L2 = (d ** 2).sum(1) + 1e-9
    u = np.clip(((xy - SA) * d).sum(1) / L2, 0, 1)
    dist = np.hypot(*(SA + u[:, None] * d - xy).T)
    dist[np.abs(np.angle(np.exp(1j * (SH - yaw)))) > np.radians(45)] = np.inf
    i = int(np.argmin(dist))
    return (int(S[i, 4]), int(S[i, 5])), float(dist[i])


def smooth_runs(lab, min_len):
    """Runs shorter than their min length take the previous run's label (autolabeler min_len_ticks)."""
    lab = list(lab)
    i = 0
    while i < len(lab):
        j = i
        while j < len(lab) and lab[j] == lab[i]: j += 1
        if j - i < min_len.get(lab[i], 8) and i > 0:
            lab[i:j] = [lab[i - 1]] * (j - i)
        i = j
    return lab


def episode(ep):
    w = ep.split("-")[1]
    f = os.path.join(CACHE, "mac-win-1" if w == "w1" else "mac-win-2", ep + ".labels.parquet")
    d = pd.read_parquet(f)
    p = d[["x", "y"]].to_numpy(); v = d.speed_kph.to_numpy() / 3.6; t = d.t.to_numpy()
    step = np.r_[np.hypot(*np.diff(p, axis=0).T), 0]; vdt = v * np.r_[np.diff(t), 0.1]
    k = np.ones(51)
    num, den = np.convolve(step, k, "same"), np.convolve(vdt, k, "same")
    rho = np.where(den > 2.0, num / np.maximum(den, 1e-9), np.nan)
    rho = pd.Series(rho).interpolate(limit_direction="both").to_numpy()
    rho = np.clip(rho, 0.5, 1.05)
    tau = np.r_[0, np.cumsum(rho[:-1] * np.diff(t))]
    g = np.arange(0, tau[-1], 0.1)
    yaw_u = np.unwrap(d.yaw.to_numpy())
    G = pd.DataFrame({"tau": g, **{c: np.interp(g, tau, d[c].to_numpy()) for c in ("x", "y", "t")},
                      "yaw": np.interp(g, tau, yaw_u), "v": np.interp(g, tau, v),
                      "bag_ns": np.interp(g, tau, d.bag_time_ns.to_numpy().astype(float)),
                      "sec": d.route_section.to_numpy()[np.clip(np.searchsorted(tau, g), 0, len(d) - 1)]})
    # motion in sim time
    a = np.convolve(np.gradient(G.v, 0.1), np.ones(5) / 5, "same")
    dx, dy = np.gradient(G.x, 0.1), np.gradient(G.y, 0.1)
    G["v_lat"] = -(-np.sin(G.yaw) * dx + np.cos(G.yaw) * dy)  # positive = rightward
    dyaw = np.r_[np.diff(G.yaw), 0]
    kappa = np.where(G.v > 1.0, dyaw / np.maximum(G.v * 0.1, 1e-6), 0.0)
    lon = np.select([G.v < 0.3, a > 1.25, a > 0.1, a < -1.25, a < -0.1],
                    ["Stop", "Strong Acceleration", "Gentle Acceleration", "Strong Deceleration", "Gentle Deceleration"],
                    "Maintain Speed")
    lon = smooth_runs(lon, {"Maintain Speed": 12})
    lat = np.select([np.abs(dyaw) < 0.001, dyaw >= 0.05, dyaw <= -0.05, dyaw > 0], ["Go Straight", "Sharp Steer Left", "Sharp Steer Right", "Steer Left"], "Steer Right")
    lat = np.where(G.v < 0.3, "Go Straight", lat)
    lat = smooth_runs(lat, {k: 10 for k in set(lat)})
    # lane: curvature runs -> Turn / Follow Curve, (road_id, ego_lane) change on the same road -> lane change
    lane = np.full(len(G), "Keep Lane", dtype=object)
    curv = np.where(kappa > 0.01, 1, np.where(kappa < -0.01, -1, 0))
    i = 0
    while i < len(G):
        j = i
        while j < len(G) and curv[j] == curv[i]: j += 1
        if curv[i] != 0 and j - i >= 6:
            tot = abs(G.yaw[j - 1] - G.yaw[i])
            side = "Left" if curv[i] > 0 else "Right"
            lane[i:j] = f"Turn {side}" if tot > 1.0 else (f"Follow Curve {side}" if tot > 0.2 else "Keep Lane")
        i = j
    ids, dist = zip(*[lane_of(np.array([x, y]), yw) for x, y, yw in zip(G.x, G.y, G.yaw)])
    for n in range(1, len(G)):
        (r0, l0), (r1, l1) = ids[n - 1], ids[n]
        if r0 == r1 and l0 != l1 and r0 > 0 and dist[n] < 2.5 and dist[n - 1] < 2.5:
            # direction from lateral motion over the maneuver, not lane numbering
            side = "Right" if G.v_lat[max(0, n - 10):n + 10].mean() > 0 else "Left"
            lane[max(0, n - 15):n + 15] = f"{side} Lane Change"
    G["lon"], G["lat"], G["lane"] = lon, lat, lane
    return G, float(np.median(rho))


def pick(G, ep):
    m = G.sec.between(9, 12).to_numpy()
    key = (G.lon + "|" + G.lane).to_numpy()
    ch = [n for n in range(1, len(G)) if m[n] and key[n] != key[n - 1]]
    lc = [n for n in ch if "Lane Change" in G.lane[n] and "Lane Change" not in G.lane[n - 1]]
    order = lc + [n for n in ch if n not in lc]
    chosen = []
    for n in order:
        t0 = n - 5  # change lands at +0.5 s, inside the 0-2 s decision window
        if t0 - 20 < 0 or t0 + 60 >= len(G): continue
        if all(abs(t0 - c) >= 80 for c in chosen): chosen.append(t0)
        if len(chosen) == PER_EP: break
    out = []
    for t0 in sorted(chosen):
        idx = [t0 + 5 * k for k in range(-4, 13)]
        rel = [0.5 * k for k in range(-4, 13)]
        out.append({
            "ep": ep, "tau0": round(float(G.tau[t0]), 2), "section": int(G.sec[t0]), "path_change": G.lon[t0 + 5] + " / " + G.lane[t0 + 5],
            "bag_ns": [int(G.bag_ns[i]) for i in idx],
            "meta": "\nMeta-actions by sampled timestamp (relative to event, unit: seconds):\n" + "\n".join(
                f"(t={r:+.1f}s) longitudinal: {G.lon[i]}, lateral: {G.lat[i]}, lane: {G.lane[i]}" for r, i in zip(rel, idx)),
            "ego": "Ego speed time series (relative to event time t=0.0s):\n"
                   "- Longitudinal (m/s): [" + ", ".join(f"(t={r:.1f}s, v_long={G.v[i]:.3f} m/s)" for r, i in zip(rel, idx)) + "]\n"
                   "- Lateral (m/s): [" + ", ".join(f"(t={r:.1f}s, v_lat={G.v_lat[i]:.3f} m/s)" for r, i in zip(rel, idx)) + "]\n"
                   "Positive lateral speed means rightward motion, negative means leftward motion.",
        })
    return out


if __name__ == "__main__":
    allk = []
    for ep in EPS:
        G, rho = episode(ep)
        ks = pick(G, ep)
        print(ep, "rho_med=%.3f" % rho, "lane-change ticks:", int(G.lane.str.contains("Lane Change").sum()),
              [(k["tau0"], k["section"], k["path_change"]) for k in ks])
        allk += ks
    json.dump(allk, open(OUT, "w"), indent=1)
    print(len(allk), "keyframes ->", OUT)
