"""Example figures of the replay exclusions (dyn_exclude.py), two per rule: ex_ring.png (R1 speed over s at the ring + one
lap), ex_late_lc.png (speed and route offset around two late lane changes), ex_prop_close.png (route frame: Ego body vs the
scaled / unscaled prop footprint), ex_prevision.png (an R2 A stop window: rows out, pool rows, first seen+conflict),
pool_static_avoid.png (two clean avoidances: route frame + speed, pool rows marked).
Usage (cwd = workspace): python train/scripts/dyn_exclude_figs.py R1_ROOT R2_ROOT DYN_OUT FIG_DIR
"""
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/coc_labeler"))
from coc_labeler import data  # noqa: E402

SEAMS = (885.1, 914.8)


def ego(root, ep):
    z = np.load(Path(root) / "_ego" / f"{ep}.npz"); return {k: z[k] for k in ("tau", "v", "path_s", "d")}


def rows(X, name, ep):
    d = pd.read_parquet(X / f"ex_{name}.parquet"); return d.i[d.episode == ep].to_numpy()


def ring(r1, X, D):
    idx = pd.read_parquet(Path(r1) / "_index/index.parquet", columns=["episode"]).episode.astype(str).unique()
    S, V = [], []
    for ep in idx[::3]:
        z = ego(r1, ep); m = (z["path_s"] >= 850) & (z["path_s"] <= 940); S.append(z["path_s"][m]); V.append(z["v"][m])
    S, V = np.concatenate(S), np.concatenate(V)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    h = ax[0].hist2d(S, V * 3.6, bins=[90, 60], range=[[850, 940], [0, 45]], cmap="Greys", norm=matplotlib.colors.LogNorm())
    for s in SEAMS: ax[0].axvline(s, color="C3", ls="--")
    ax[0].axhspan(0.5 * 3.6, 3.2 * 3.6, xmin=(880 - 850) / 90, xmax=(925 - 850) / 90, color="C1", alpha=0.25, label="기어감 판정 0.5–3.2 m/s, s 880–925")
    ax[0].set(xlabel="경로 s (m)", ylabel="km/h", title=f"R1 속도 분포 (에피소드 {len(idx[::3])}개, 3개 중 1개)"); ax[0].legend(fontsize=8)
    fig.colorbar(h[3], ax=ax[0])
    cr = pd.read_parquet(X / "ex_ring_crawl.parquet"); ep = cr.episode.value_counts().index[len(cr.episode.unique()) // 2]
    z = ego(r1, ep); m = (z["path_s"] >= 850) & (z["path_s"] <= 940); k = np.flatnonzero(m)
    ax[1].plot(z["tau"][k], z["v"][k] * 3.6, color="0.5", lw=1.5, label="원래 속도")
    for name, c in (("ring_crawl", "C3"), ("ring_crawl_future", "C1"), ("ring_stop", "C4")):
        j = np.intersect1d(rows(X, name, ep), k); ax[1].plot(z["tau"][j], z["v"][j] * 3.6, ".", color=c, ms=4, label=f"{name} ({len(j)}행)")
    ax2 = ax[1].twinx(); ax2.plot(z["tau"][k], z["path_s"][k], color="C0", lw=0.8); ax2.set_ylabel("s (m)", color="C0")
    for s in SEAMS:
        j = k[np.argmin(np.abs(z["path_s"][k] - s))]; ax[1].axvline(z["tau"][j], color="C3", ls="--", lw=0.8)
    ax[1].set(xlabel="τ (s)", ylabel="km/h", title=f"{ep}: 로터리 구간 (빨간 점선 = 이음매 s 885.1, 914.8)"); ax[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(D / "ex_ring.png", dpi=110); plt.close(fig)


def late_lc(r1, X, D):
    E = pd.read_parquet(X / "late_lc_events.parquet")
    ex = [E[(E.s > 1190) & (E.s < 1230)].iloc[5], E[(E.s > 1390) & (E.s < 1470)].iloc[5]]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for a, e in zip(ax, ex):
        z = ego(r1, e.episode); k = np.arange(max(e.i_start - 100, 0), min(e.i_end + 60, len(z["v"])))
        out = np.intersect1d(rows(X, "late_lc", e.episode), k)
        a.plot(z["tau"][k], z["v"][k] * 3.6, color="0.3", label="속도 km/h")
        a.plot(z["tau"][out], z["v"][out] * 3.6, ".", color="C3", ms=3, label=f"제외 행 ({len(out)})")
        b = a.twinx(); b.plot(z["tau"][k], z["d"][k], color="C0", label="경로 횡오프셋 d"); b.set_ylabel("d (m)", color="C0")
        a.axvline(z["tau"][e.i_start], color="C1", ls="--", label="감속 시작")
        a.set(xlabel="τ (s)", ylabel="km/h", title=f"{e.episode} s {e.s:.0f}: {e.kph_before:.0f} → {e.kph_min:.0f} km/h 차로 변경"); a.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(D / "ex_late_lc.png", dpi=110); plt.close(fig)


def prop_close(r1, r2, X, D):
    pc = pd.read_parquet(X / "ex_prop_close.parquet"); eps = pc.episode.value_counts().index[:12]
    shown = 0; fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for ep in eps:
        if shown == 2: break
        _, sc = data.scene_of(ep); root = r2 if ep.startswith("r2") else r1; z = ego(root, ep)
        props = [p for p in (sc or {}).get("actors", []) if p.get("zone") == "tunnel_static" and "scale" in p]
        for p in props:
            sx, sy = p["scale"][:2]; s, d, v, tau = z["path_s"], z["d"].astype(float), z["v"], z["tau"]
            ds = np.abs(np.diff(s)) / np.maximum(np.diff(tau), 1e-3)
            real = np.r_[True, np.abs(ds - v[1:]) < 5.0] & np.r_[np.abs(ds - v[:-1]) < 5.0, True]   # as dyn_exclude
            over = (s + 3.9 >= p["s"] - sx / 2) & (s - 0.8 <= p["s"] + sx / 2) & real
            d = np.where(real, d, np.nan)                                                  # no lines across GT jumps
            if not over.any() or (np.abs(d[over] - p["lat"]) - 0.95 - sy / 2).min() >= 0.3 or sy < 1.3: continue
            a = ax[shown]; m = (s > p["s"] - 30) & (s < p["s"] + 20)
            dm = np.where(m, d, np.nan)                                                    # gaps (other laps) break the line
            a.plot(s, dm, color="0.3", label="자차 경로 (뒷차축)"); a.fill_between(s, dm - 0.95, dm + 0.95, where=m, color="0.85", label="자차 폭 ±0.95 m")
            a.set_xlim(p["s"] - 30, p["s"] + 20)
            a.add_patch(plt.Rectangle((p["s"] - sx / 2, p["lat"] - sy / 2), sx, sy, color="C3", alpha=0.5, label=f"실제 크기 {sx:.1f}×{sy:.1f} m (scale)"))
            a.add_patch(plt.Rectangle((p["s"] - 0.5, p["lat"] - 0.5), 1, 1, fill=False, ec="k", ls="--", label="Object Info 크기 1×1 m"))
            c = (np.abs(d[over] - p["lat"]) - 0.95 - sy / 2).min()
            a.set(xlabel="경로 s (m)", ylabel="횡 (m, 왼쪽 +)", title=f"{ep}: 터널 소품 s {p['s']:.0f}, 최소 간격 {c:.2f} m"); a.legend(fontsize=7)
            shown += 1; break
    fig.tight_layout(); fig.savefig(D / "ex_prop_close.png", dpi=110); plt.close(fig)


def prevision(r2, dyn, X, D):
    L = pd.read_parquet(dyn / "dyn_relabel_index.parquet"); pv = pd.read_parquet(X / "ex_prevision.parquet")
    eps = pv.episode.value_counts().index[[3, 12]]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for a, ep in zip(ax, eps):
        z = ego(r2, ep); out = rows(X, "prevision", ep); pool = L[(L.episode == ep) & (L.kind != "pass")]
        k = np.arange(max(out.min() - 30, 0), min(max(out.max(), pool.i.max() if len(pool) else 0) + 40, len(z["v"])))
        a.plot(z["tau"][k], z["v"][k] * 3.6, color="0.4", label="원래 속도")
        a.plot(z["tau"][out], z["v"][out] * 3.6, ".", color="C3", ms=4, label=f"(b) 제외: 보이기 전 ({len(out)})")
        a.plot(z["tau"][pool.i], z["v"][pool.i] * 3.6, ".", color="C2", ms=4, label=f"합성 라벨로 대체 ({len(pool)})")
        if len(pool): a.axvline(z["tau"][int(pool.seen_i.min())], color="C0", ls=":", label="처음 보임")
        a.set(xlabel="τ (s)", ylabel="km/h", title=f"{ep}: AEB/양보 정지 창"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(D / "ex_prevision.png", dpi=110); plt.close(fig)


def static_avoid(r1, r2, dyn, D):
    sys.path.insert(0, str(Path(__file__).resolve().parent)); from dyn_exclude import footprint, STATIC   # noqa: E402
    A = pd.read_parquet(dyn / "static_avoid.parquet"); fig, ax = plt.subplots(2, 2, figsize=(14, 8)); shown = 0
    for ep in [A[A.episode.str.startswith("run")].episode.value_counts().index[40], A[A.episode.str.startswith("r2")].episode.value_counts().index[3]]:
        _, sc = data.scene_of(ep); z = ego(r2 if ep.startswith("r2") else r1, ep); s, d, v = z["path_s"], z["d"].astype(float), z["v"]
        rows = A.i[A.episode == ep].to_numpy(); props = [p for p in sc["actors"] if p.get("zone") in STATIC and "s" in p]
        p = min(props, key=lambda p: np.abs(s[rows] - p["s"]).min()); sx, sy = footprint(p)
        k = np.flatnonzero((s > p["s"] - 60) & (s < p["s"] + 20)); k = k[: np.argmax(np.diff(k) > 1) + 1] if (np.diff(k) > 1).any() else k
        a, b = ax[0, shown], ax[1, shown]; inn = np.intersect1d(rows, k)
        a.plot(s[k], d[k], color="0.3", label="자차 경로 (뒷차축)"); a.fill_between(s[k], d[k] - 0.95, d[k] + 0.95, color="0.85", label="자차 폭 ±0.95 m")
        a.plot(s[inn], d[inn], ".", color="C2", ms=3, label=f"풀 행 ({len(inn)})")
        a.add_patch(plt.Rectangle((p["s"] - sx / 2, p["lat"] - sy / 2), sx, sy, color="C3", alpha=0.5, label=f"{p['zone']} 실제 {sx:.1f}×{sy:.1f} m"))
        a.set(xlabel="경로 s (m)", ylabel="횡 (m)", title=f"{ep}: 정적 소품 s {p['s']:.0f} 회피"); a.legend(fontsize=7)
        b.plot(z["tau"][k], v[k] * 3.6, color="0.3", label="속도"); b.plot(z["tau"][inn], v[inn] * 3.6, ".", color="C2", ms=3, label="풀 행")
        b.set(xlabel="τ (s)", ylabel="km/h"); b.legend(fontsize=7); shown += 1
    fig.tight_layout(); fig.savefig(D / "pool_static_avoid.png", dpi=110); plt.close(fig)


def main():
    r1, r2, dyn, D = map(Path, sys.argv[1:5]); D.mkdir(parents=True, exist_ok=True); X = dyn / "exclude"
    plt.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
    ring(r1, X, D); late_lc(r1, X, D); prop_close(r1, r2, X, D); prevision(r2, dyn, X, D); static_avoid(r1, r2, dyn, D)


if __name__ == "__main__":
    main()
