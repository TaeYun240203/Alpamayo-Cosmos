#!/usr/bin/env python3
"""Roundabout entry stops (10/28): R2 fine-tune delivery vs. the 10/27 delivery (WoWmayo) on the same scenes (seed 28, laps 0-3).
Per model step in route s 840-950 (CP8 858.5, CP9 930.6) from the eval3 drive dirs: actual speed, the model's plan (speed
at 1/3/6.4 s and distance in 6.4 s from its 64 accelerations and v0), the stack's command (a, hold, 'no fresh plan'),
nav text, and the nearest NPC vehicle from Object Info. hold = |v| < 0.3 and the plan's speed < 0.3 for the next 1 s, so a
hold is the model's own stop plan.        python3 ring_entry_analysis.py RUN_DIR:LABEL ... -> table + ring_entry_*.png
"""
import gzip, json, sys
from pathlib import Path

import numpy as np

S_LO, S_HI = 840.0, 950.0


def load(d):
    d = Path(d)
    rows = [json.loads(l.split(",", 1)[1]) for l in open(d / "dbg_w1.csv").read().splitlines()[1:] if "," in l]
    cl = [l.split(",") for l in open(d / "clk_w1.csv").read().splitlines()]
    i0, i1 = cl[0].index("field.data0"), cl[0].index("field.data1")
    clk = np.array([(float(r[i0]), float(r[i1])) for r in cl[1:] if len(r) > i1])          # stack: wall, tau
    laps = {x["lap"]: x for x in map(json.loads, open(d / "laps.jsonl"))}
    return rows, clk, laps


def lap_steps(d, rows, clk, k):
    c = np.array(json.load(open(Path(d) / f"lap{k:03d}.clock.json")))                       # manager: wall, tau
    t0, t1 = np.interp([c[0, 0], c[-1, 0]], clk[:, 0], clk[:, 1])
    ob = json.load(gzip.open(Path(d) / f"lap{k:03d}.objects.json.gz", "rt"))
    ot = np.array([o[0] for o in ob["objects"]]) if ob["objects"] else np.zeros(0)
    out = []
    for r in rows:
        if not (t0 <= r["t0"] <= t1 and S_LO <= r["s"] <= S_HI):
            continue
        a, v0 = np.asarray(r["plan_a"], float), float(r["v0"])
        sp = np.maximum(v0 + np.cumsum(a) * 0.1, 0.0)
        cmd = r.get("cmd") or {}
        wall = np.interp(r["t0"], clk[:, 1], clk[:, 0]); tm = np.interp(wall, c[:, 0], c[:, 1])
        near = None
        if len(ot):
            j = int(np.argmin(np.abs(ot - tm))); e = ob["ego"][min(j, len(ob["ego"]) - 1)]
            dv = [np.hypot(o[2] - e[1], o[3] - e[2]) for o in ob["objects"][j][1] if o[1] == 1]
            near = min(dv) if dv else 99.0
        out.append(dict(t=r["t0"], s=r["s"], v=r["v"], v0=v0, sp1=sp[9], sp3=sp[29], sp64=sp[-1], d64=float(sp.sum() * 0.1),
                        hold=bool(cmd.get("hold")), stale="reason" in cmd, a_cmd=cmd.get("a"), nav=r["nav"], late=r.get("late"), npc=near))
    return out


def summary(label, k, st, lap):
    if not st:
        return f"{label} lap {k}: no steps in s {S_LO}-{S_HI}"
    stop = [x for x in st if abs(x["v"]) < 0.3]
    hold = sum(x["hold"] for x in stop); go = sum(x["sp3"] > 1.0 for x in stop); stale = sum(x["stale"] for x in stop)
    rets = [r for r in lap["returns"] if S_LO - 30 <= r["s"] <= S_HI]
    navs = sorted({x["nav"][:60] for x in stop})[:3]
    npc = [x["npc"] for x in stop if x["npc"] is not None]
    npc_s = f"nearest NPC while stopped p50 {np.median(npc):.1f} m, share < 15 m {np.mean(np.array(npc) < 15):.2f}" if npc else "no stop"
    return (f"{label} lap {k}: {len(st)} steps in the ring window ({len(st) / 10:.0f} s), stopped {len(stop)} ({len(stop) / 10:.0f} s): "
            f"hold (model stop plan) {hold}, plan says go (speed@3s > 1 m/s) {go}, no fresh plan {stale}; returns here "
            f"{[(r['reason'], r['s']) for r in rets]}; {npc_s}\n    nav while stopped: {navs}")


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    runs = [a.split(":") for a in sys.argv[1:]]
    fig, ax = plt.subplots(4, len(runs), figsize=(7 * len(runs), 11), sharey="row", squeeze=False)
    for j, (d, label) in enumerate(runs):
        rows, clk, laps = load(d)
        for k in range(3):
            if k not in laps:
                continue
            st = lap_steps(d, rows, clk, k)
            print(summary(label, k, st, laps[k]))
            if not st:
                continue
            t = (np.array([x["t"] for x in st]) - st[0]["t"])
            ax[0, j].plot(t, [x["v"] for x in st], label=f"lap {k}"); ax[1, j].plot(t, [x["sp3"] for x in st])
            ax[2, j].plot(t, [x["d64"] for x in st]); ax[3, j].plot(t, [x["npc"] if x["npc"] is not None else np.nan for x in st])
            h = [i for i, x in enumerate(st) if x["hold"]]
            ax[0, j].scatter(t[h], [0.0] * len(h), s=4, color="k")
        ax[0, j].set_title(f"{label}: roundabout window s {S_LO:.0f}-{S_HI:.0f} (black dots = hold)")
        for i, y in enumerate(("speed m/s", "plan speed @3 s", "plan distance 6.4 s (m)", "nearest NPC vehicle (m)")):
            ax[i, j].set_ylabel(y); ax[i, j].grid(alpha=0.3)
        ax[3, j].set_xlabel("s since entering the window (each lap)"); ax[0, j].legend()
    fig.tight_layout(); out = Path(__file__).parent / "ring_entry_r2_vs_wowmayo.png"; fig.savefig(out, dpi=90); print("figure", out)


if __name__ == "__main__":
    main()
