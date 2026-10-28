"""Lane-shaped ego histories for the R2 fine-tune (user 10/28: "이력은 전처리에서 차로 중앙선 모양(실패하면 곧은 이력)으로
만들고, 복구 행은 실제·정지·합성 1/3씩, D 출발 1.5초는 합성 이력으로"). CPU, Mac, no model.
For a row at t0 with pose (x, y, yaw) and speed v (its _ego table): the car drove the last 1.5 s at v along the route lane
centerline (coc_labeler global path, from kcity_lanelet2.osm) shifted parallel to its current offset, and turned into its
current heading over the last T_BLEND s:
    h(t) = path heading at s0 + v t  +  (yaw - path heading at s0) * clip(1 + t / T_BLEND, 0, 1),   t = -1.5 .. 0
positions integrated backward from the current one with midpoint headings (as morai_vla.data.ego_trajectory). The last
point is the current position, the last segment runs along the current heading, the rest parallel to the centerline.
T_BLEND = 1 s: the R2 D yaw errors go up to 22 deg, i.e. <= 22 deg/s and ~4 m/s^2 lateral at 11 m/s, a plausible
correction; 0.5 s would need twice that. A row that does not map gets the straight history along its current heading:
|d| > D_MAX (grass, most of the opposite lane), |yaw error| > YAW_MAX, the projection more than S_TOL away from the
recorded progress (another part of the route), or s0 - 1.5 v before the route start.
Rows: the R2 D index rows (train), the old_recovery 2 Hz train rows (R1), and the D start rows i < 15 that the index leaves
out for their short real history, when their 6.4 s future is trainable by the build_index rules applied to the future only
(traj_ok_on_grid, i + 64 < n, no glitch step in [i, i + 64] by build_index.glitch_steps 'net', the R2 rule of 10/28);
those carry extra = True and the index columns.
Output parquet: episode, i, method (lane | straight), hist_x / hist_y / hist_yaw (16, ego frame at t0, oldest first), extra
[+ tau, cell, split, nav_text, fog, hour, scene]. --figs: overlays of real path, synthetic history and centerline.
Usage (cwd train/): python scripts/ft_synth_hist.py --r2-ego <dir> --r2-index <step0_r2>/_index/index.parquet \\
    --r1-ego ~/.cache/step0_ego --r1-index <step0>/_index/index.parquet --out assets/ft/synth_hist.parquet [--figs <dir> --n-figs 12]"""
import argparse, json, sys
from pathlib import Path

import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from morai_vla.build_index import FUT, data, glitch_steps, maneuvers, nav_text, traj_ok_on_grid   # noqa: E402

N, DT, T_BLEND, D_MAX, YAW_MAX, S_TOL = 16, 0.1, 1.0, 4.5, np.radians(30), 5.0


def project(P, S, x, y, s_hint, win=30.0):
    """(s, d left +, path heading) of (x, y) on the segments within s_hint +- win."""
    lo, hi = max(np.searchsorted(S, s_hint - win) - 1, 0), min(np.searchsorted(S, s_hint + win) + 1, len(P) - 1)
    a, b = P[lo:hi], P[lo + 1:hi + 1]; ab = b - a; L2 = np.maximum((ab ** 2).sum(1), 1e-12)
    u = np.clip(((np.array([x, y]) - a) * ab).sum(1) / L2, 0, 1); f = a + u[:, None] * ab
    j = int(np.argmin(np.hypot(x - f[:, 0], y - f[:, 1]))); h = np.arctan2(ab[j, 1], ab[j, 0])
    d = -np.sin(h) * (x - f[j, 0]) + np.cos(h) * (y - f[j, 1])
    return S[lo + j] + u[j] * np.sqrt(L2[j]), d, h


def history(P, S, H, x, y, yaw, v, s_hint):
    """-> (hist_x, hist_y, hist_yaw) ego frame at t0, oldest first, and the method."""
    t = -DT * np.arange(N - 1, -1, -1)                                              # -1.5 .. 0
    s0, d, h0 = project(P, S, x, y, s_hint)
    e = (yaw - h0 + np.pi) % (2 * np.pi) - np.pi
    lane = abs(d) <= D_MAX and abs(e) <= YAW_MAX and abs(s0 - s_hint) <= S_TOL and s0 + v * t[0] >= S[0]
    if lane:
        hp = np.interp(s0 + v * t, S[:-1], H)                                      # path heading (unwrapped) along the way back
        h = hp - hp[-1] + h0 + e * np.clip(1 + t / T_BLEND, 0, 1)                 # = yaw at t0
    else:
        h = np.full(N, yaw)
    hm = 0.5 * (h[1:] + h[:-1]); px, py = np.zeros(N), np.zeros(N); px[-1], py[-1] = x, y
    for k in range(N - 1, 0, -1):
        px[k - 1], py[k - 1] = px[k] - v * DT * np.cos(hm[k - 1]), py[k] - v * DT * np.sin(hm[k - 1])
    c, sn = np.cos(yaw), np.sin(yaw); dx, dy = px - x, py - y
    return c * dx + sn * dy, -sn * dx + c * dy, h - yaw, "lane" if lane else "straight"


def path_arrays():
    P, S = data.path_xy(); d = np.diff(P, axis=0)
    return P, S, np.unwrap(np.arctan2(d[:, 1], d[:, 0]))


def rows_for(ego_dir, episodes, sel, P, S, H, extra_cols=None):
    out = []
    for ep, iis in sel.items():
        f = Path(ego_dir) / f"{ep}.npz"
        if not f.exists(): continue
        z = np.load(f)
        for i in iis:
            hx, hy, hyaw, m = history(P, S, H, float(z["x"][i]), float(z["y"][i]), float(z["yaw"][i]), float(z["v"][i]), float(z["path_s"][i]))
            r = dict(episode=ep, i=int(i), method=m, hist_x=hx.astype(np.float32), hist_y=hy.astype(np.float32), hist_yaw=hyaw.astype(np.float32), extra=False)
            if extra_cols is not None: r |= extra_cols(ep, z, int(i))
            out.append(r)
    return out


def start_rows(ego_dir, eps):
    """D start rows i < 15 with a trainable 6.4 s future (build_index rules, future only)."""
    sel = {}
    for ep in eps:
        f = Path(ego_dir) / f"{ep}.npz"
        try: L = pd.read_parquet(data.labels_file(ep), columns=["bag_time_ns", "traj_ok"])
        except FileNotFoundError: continue
        z = np.load(f); G = {k: z[k] for k in ("tau", "bag_ns")}; n = len(z["tau"]); i = np.arange(n)
        jump = np.r_[0, np.cumsum(glitch_steps(z["v"], z["tau"], "net"))]                    # as the R2 index (--glitch net)
        ok = traj_ok_on_grid(G, L) & (i + FUT < n) & (jump[np.minimum(i + FUT, n - 1)] - jump[i] == 0) & (i < 15)
        if ok.any(): sel[ep] = np.flatnonzero(ok)
    return sel


def main():
    ap = argparse.ArgumentParser()
    for k in ("--r2-ego", "--r2-index", "--r1-ego", "--r1-index", "--out"): ap.add_argument(k, required=True)
    ap.add_argument("--pools", default="assets/ft"); ap.add_argument("--figs"); ap.add_argument("--n-figs", type=int, default=12)
    a = ap.parse_args(); P, S, H = path_arrays()
    r2 = pd.read_parquet(a.r2_index).astype({"episode": str}); D = set(pd.read_parquet(f"{a.pools}/r2_D.parquet").episode.astype(str))
    dtr = r2[r2.episode.isin(D) & (r2.split == "train")]
    rows = rows_for(a.r2_ego, D, dtr.groupby("episode").i.apply(list).to_dict(), P, S, H)
    first = r2[r2.episode.isin(D)].sort_values("i").groupby("episode").first()   # split / scene of the episode's first index row
    M, lap, exempt = maneuvers(), float(S[-1]), data.kmap().speed_exempt_s                  # as build_index.main
    def cols(ep, z, i):
        f = first.loc[ep]
        return dict(extra=True, tau=float(z["tau"][i]), cell=data.ROUTE["cells"].get(int(z["section"][i]), "C?"), split=str(f.split),
                    nav_text=nav_text(float(z["path_s"][i]), M, lap, exempt), fog=bool(z["fog"][i]), hour=int(z["hour"][i]), scene=str(f.scene))
    starts = start_rows(a.r2_ego, sorted(e for e in D & set(first.index) if first.loc[e].split == "train"))   # train episodes only
    rows += rows_for(a.r2_ego, D, starts, P, S, H, cols)
    r1 = pd.read_parquet(a.r1_index, columns=["episode", "i", "split"]).astype({"episode": str})
    rec = pd.read_parquet(f"{a.pools}/old_recovery.parquet").astype({"episode": str}).merge(r1, on=["episode", "i"])
    rec = rec[(rec.split == "train") & (rec.i % 5 == 0)]
    rows += rows_for(a.r1_ego, set(rec.episode), rec.groupby("episode").i.apply(list).to_dict(), P, S, H)
    df = pd.DataFrame(rows); Path(a.out).parent.mkdir(parents=True, exist_ok=True); df.to_parquet(a.out, index=False)
    src = np.where(df.episode.isin(D), np.where(df.extra, "r2_D_start", "r2_D"), "old_recovery")
    summ = {s: {"rows": int((src == s).sum()), "lane": int(((src == s) & (df.method == "lane")).sum())} for s in ("r2_D", "r2_D_start", "old_recovery")}
    summ["r2_D_start"] |= {"episodes": int(df[df.extra].episode.nunique()), "rows_2hz": int((df.extra & (df.i % 5 == 0)).sum())}
    json.dump(summ, open(Path(a.out).with_suffix(".json"), "w"), indent=1); print(json.dumps(summ))
    if a.figs: figures(df, src, a, P, S)


def figures(df, src, a, P, S):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rng = np.random.default_rng(0); Path(a.figs).mkdir(parents=True, exist_ok=True)
    picks = [k for s in ("r2_D", "r2_D_start", "old_recovery") for k in rng.choice(np.flatnonzero(src == s), min(4, int((src == s).sum())), replace=False)]
    for k in picks[:a.n_figs]:
        r = df.iloc[k]; ego = a.r2_ego if src[k] != "old_recovery" else a.r1_ego; z = np.load(Path(ego) / f"{r.episode}.npz"); i = int(r.i)
        x0, y0, c0 = z["x"][i], z["y"][i], z["yaw"][i]; c, s = np.cos(c0), np.sin(c0)
        hx, hy = x0 + c * r.hist_x - s * r.hist_y, y0 + s * r.hist_x + c * r.hist_y
        lo, hi = max(i - 15, 0), min(i + 64, len(z["x"]) - 1); m = (S > z["path_s"][i] - 40) & (S < z["path_s"][i] + 70)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(P[m, 0], P[m, 1], "k--", lw=0.8, label="route centerline")
        ax.plot(z["x"][lo:i + 1], z["y"][lo:i + 1], "b-", lw=1.5, label="real history (1.5 s)")
        ax.plot(z["x"][i:hi], z["y"][i:hi], "g-", lw=1, alpha=0.6, label="real future (6.4 s)")
        ax.plot(hx, hy, "r.-", ms=4, lw=1, label=f"synthetic history ({r.method})"); ax.plot([x0], [y0], "ko")
        ax.set_xlim(x0 - 30, x0 + 30); ax.set_ylim(y0 - 30, y0 + 30); ax.set_aspect("equal"); ax.legend(fontsize=7); ax.set_title(f"{src[k]} {r.episode} i={i} v={z['v'][i]:.1f} m/s", fontsize=9)
        fig.tight_layout(); fig.savefig(Path(a.figs) / f"{src[k]}_{r.episode}_{i}.png", dpi=90); plt.close(fig)


if __name__ == "__main__":
    main()
