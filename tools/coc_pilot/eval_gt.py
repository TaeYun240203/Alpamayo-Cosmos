"""Score CoC outputs against ego-motion GT (stop/speed/lane-line crossings) and C4/C5 scene GT (only NPC vehicles exist)."""
import json, re, sys, xml.etree.ElementTree as ET
import numpy as np, pandas as pd
from pathlib import Path
import prep

H = Path(__file__).parent
OSM = "/Users/taeyunkim/Library/Mobile Documents/com~apple~CloudDocs/Workspaces/MORAI-Autonomous-Driving-System/src/map/kcity_lanelet2.osm"

# every painted lane line (dashed included): crossing one = a real lateral lane transition
root = ET.parse(OSM).getroot()
tg = lambda e: {t.get("k"): t.get("v") for t in e.findall("tag")}
nodes = {n.get("id"): (float(tg(n)["local_x"]), float(tg(n)["local_y"])) for n in root.iter("node")}
A, B = [], []
for w in root.iter("way"):
    t = tg(w)
    if t.get("type") in ("line_thin", "line_thick"):
        p = np.array([nodes[r.get("ref")] for r in w.findall("nd")])
        A += list(p[:-1]); B += list(p[1:])
A, B = np.array(A), np.array(B)


def crossings(P):
    """[(i, side)] where segment P[i]->P[i+1] crosses a lane line; side = which way the ego moved across it
    (line oriented along travel: ego goes from its right side to its left side = 'left'). Works for taper lines too."""
    lo, hi = P.min(0) - 5, P.max(0) + 5
    m = ((np.minimum(A, B) <= hi) & (np.maximum(A, B) >= lo)).all(1)
    a, b = A[m], B[m]; out = []
    cr = lambda o, p, q: (p[..., 0] - o[..., 0]) * (q[..., 1] - o[..., 1]) - (p[..., 1] - o[..., 1]) * (q[..., 0] - o[..., 0])
    for i in range(len(P) - 1):
        p, q = P[i], P[i + 1]
        if np.hypot(*(q - p)) < 1e-3: continue
        d1, d2 = cr(a, b, p[None]), cr(a, b, q[None]); d3, d4 = cr(p[None], q[None], a), cr(p[None], q[None], b)
        hit = (d1 * d2 < 0) & (d3 * d4 < 0)
        if np.any(hit):
            k = int(np.flatnonzero(hit)[0]); la, lb = a[k], b[k]
            if np.dot(lb - la, q - p) < 0: la, lb = lb, la
            out.append((i, "left" if cr(la[None], lb[None], q[None])[0] > 0 else "right"))
    return out


def claims(s):
    s = s.lower()
    side = lambda pat: ("left" if re.search(pat + r"[^.]{0,25}left|left[^.]{0,12}" + pat, s) else "") + ("right" if re.search(pat + r"[^.]{0,25}right|right[^.]{0,12}" + pat, s) else "")
    return {"lc": bool(re.search(r"lane change|change lanes?|changes lanes|merge", s)), "lc_side": side(r"(lane change|change lanes?|merge)"),
            "turn": bool(re.search(r"\bturn (right|left)", s)), "stop": bool(re.search(r"\bstop\b|\bstop (for|at|behind)", s)), "yield": "yield" in s,
            "decel": bool(re.search(r"decelerat|slow", s)), "accel": bool(re.search(r"accelerat|resume|build speed|speed up", s)), "gap": "gap" in s,
            "vehicle": bool(re.search(r"vehicle|\bcar\b|sedan|suv|truck|traffic", s)),
            "halluc": [k for k, p in {"pedestrian": r"pedestrian", "cyclist": r"cyclist|bicycl", "light": r"traffic light|red light|green light|yellow",
                                        "construction": r"construction|cone|work zone", "barrier": r"barrier", "bump": r"speed bump", "cutin": r"cut(ting)? ?in",
                                        "stopsign": r"stop sign"}.items() if re.search(p, s)]}


R = [json.loads(l) for l in open(H / "results.jsonl")]
G = {ep: prep.episode(ep)[0] for ep in {r["ep"] for r in R}}
rows = []
for r in R:
    g = G[r["ep"]]; t0 = int(np.argmin(np.abs(g.tau - r["tau0"])))
    w = lambda a, b: slice(max(0, t0 + int(a * 10)), min(len(g), t0 + int(b * 10) + 1))
    v = g.v.to_numpy(); P = np.c_[g.x + 1.4 * np.cos(g.yaw), g.y + 1.4 * np.sin(g.yaw)]
    sw = w(-2, 6); cx = crossings(P[sw]); xs = [sw.start + i for i, _ in cx]
    xside = "".join(sorted({sd for _, sd in cx}))
    c = claims(r["parsed"]["ego_behavior_schema"]["effect_on_ego_behavior"])
    rows.append({"tag": r["tag"], "model": r["model"], "j": r["j"], "ep": r["ep"], "tau0": r["tau0"], "sec": r["section"],
                 "text": r["parsed"]["ego_behavior_schema"]["effect_on_ego_behavior"],
                 "gt_stop": bool(v[w(-2, 6)].min() < 0.3), "gt_stop04": bool(v[w(0, 4)].min() < 0.3),
                 "gt_dv02": float(v[min(len(v) - 1, t0 + 20)] - v[t0]), "gt_dv04": float(v[min(len(v) - 1, t0 + 40)] - v[t0]),
                 "gt_lc": bool(xs), "gt_lc_side": xside, "gt_lc_t": [round((i - t0) / 10, 1) for i in xs],
                 "gt_dyaw": float(g.yaw[min(len(g) - 1, t0 + 60)] - g.yaw[max(0, t0 - 20)]), "v0": float(v[t0]),
                 **{"c_" + k: x for k, x in c.items()}})
D = pd.DataFrame(rows)
D.to_pickle(H / "eval.pkl")
if __name__ == "__main__":
    def rate(m, num): return f"{int(num[m].sum())}/{int(m.sum())}"
    for (tag, model), d in D.groupby(["tag", "model"]):
        c9 = d.sec.between(9, 12)
        print(f"== {tag} {model} n={len(d)} (C4/C5 {int(c9.sum())})")
        print("  lane-change/merge claimed when lane line crossed (precision):", rate(d.c_lc, d.gt_lc),
              "| crossed & claimed (recall):", rate(d.gt_lc, d.c_lc))
        print("  stop claimed & ego stopped 0-4s:", rate(d.c_stop, d.gt_stop04), "| ego stopped & stop/yield claimed:", rate(d.gt_stop04, d.c_stop | d.c_yield))
        print("  accel claimed & dv(0-4s)>0:", rate(d.c_accel, d.gt_dv04 > 0), "| decel claimed & dv(0-4s)<0:", rate(d.c_decel, d.gt_dv04 < 0))
        h = d[c9].c_halluc.explode().dropna().value_counts().to_dict()
        print("  C4/C5 mentions of things that do not exist there:", h, "of", int(c9.sum()))
