"""Bird's-eye figure of one label_check decision: solid paint, stop lines, the route, objects at t0 (and where they are
at 6.4 s), the label with its paint contacts, and every reference (the best one per failed rule in bold).

    from label_check_plot import plot
    plot(chk, ego, traj, path_png, title="", signal=..., objects=..., blackout=...)
"""
import numpy as np


def plot(chk, ego, traj, out, title="", **kw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = chk.check(ego, traj, debug=True, **kw)
    km = chk.km
    xy = r.get("label_xy", np.c_[[ego["x"]], [ego["y"]]])
    lo, hi = xy.min(0) - 15, xy.max(0) + 15
    span = (hi - lo).max()
    lo, hi = (lo + hi) / 2 - span / 2, (lo + hi) / 2 + span / 2
    fig, ax = plt.subplots(figsize=(8, 8))
    box = lambda p: ((p >= lo) & (p <= hi)).all(-1)
    for a, b, hw in zip(km.seg_a, km.seg_b, km.seg_hw):
        if box(a) or box(b):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="goldenrod" if hw > 0.1 else "0.35", lw=1.2)
    for sid, line in km.stop_lines.items():
        if box(line).any():
            ax.plot(*line.T, color="red" if sid in chk.stops else "0.6", lw=2.5)
    p = km.path[box(km.path)]
    ax.plot(*p.T, ":", color="0.6", lw=1)
    for name, ref in r.get("refs", {}).items():
        bold = name in r["better"].values()
        ax.plot(*ref["xy"].T, lw=2 if bold else 0.8, alpha=1 if bold else 0.5,
                label=f"{name} c{ref['cost']:.0f} lane {ref['lane']:.1f}" if bold or name.startswith("route") else None)
    ax.plot(*xy.T, "k-", lw=2.2, label=f"label c{r['label']['cost']:.0f} lane {r['label']['lane']:.1f}")
    if "label_xy" in r:
        c = np.r_[False, _contact(chk, ego, traj, kw)]
        ax.plot(*xy[c].T, "o", ms=4, mfc="none", mec="m", label="label paint contact")
    for o in kw.get("objects", ()):
        for t, ls in ((0.0, "-"), (6.4, ":")):
            cx, cy = o[2] + o[5] * t, o[3] + o[6] * t
            u, w = np.array([np.cos(o[4]), np.sin(o[4])]) * o[7] / 2, np.array([-np.sin(o[4]), np.cos(o[4])]) * o[8] / 2
            q = np.array([[cx, cy]] * 5) + np.array([u + w, u - w, -u - w, -u + w, u + w])
            ax.plot(*q.T, ls, color="tab:red")
    ax.set_xlim(lo[0], hi[0]), ax.set_ylim(lo[1], hi[1]), ax.set_aspect("equal")
    ax.set_title(f"{title}\n{r['verdict']} {';'.join(r['reasons'])} {';'.join(r['tags'])}", fontsize=9)
    ax.legend(fontsize=7, loc="best")
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return r


def _contact(chk, ego, traj, kw):
    """the label's paint contact per 0.1 s step (as label_check judges it, blackout aside)."""
    from label_check import _near
    tr = np.asarray(traj, float)
    c, s = np.cos(tr[:, 2]), np.sin(tr[:, 2])
    km = chk.km
    reach = km.seg_hw + chk.cfg["vehicle"]["tire_half_width_m"]
    return np.any([_near(np.c_[tr[:, 0] + c * wx - s * wy, tr[:, 1] + s * wx + c * wy], km.seg_a, km.seg_b, reach, km.grid)
                   for wx, wy in chk.wheels], axis=0)
