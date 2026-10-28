"""Check figures of dyn_relabel.py: per kind (soft stop, hard stop, behind_parked pop-out, pass) two example rows.
Left: speed, original (history + future) and label, with the object's first seen and first conflict times and the
restart (object out of the corridor; the label waits 0.5 s, then gains 1.5 m/s2). Right: route frame (s, lat): the Ego path, the corridor, the object's track (dots every 0.5 s),
its position at t0 and the stop point. Also kinds.md (row counts per kind / class / split).
Usage (cwd = workspace): python train/scripts/dyn_relabel_figs.py R2_ROOT DYN_OUT FIG_DIR
"""
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dyn_relabel as dr  # noqa: E402

KINDS = {"soft": "부드러운 정지 (soft)", "hard": "급정거 (hard)", "popout": "behind_parked 튀어나옴", "pass": "반대 예시: 통과 (pass)"}


def pick(L, kind, k=2):
    if kind == "soft": c = L[(L.kind == "soft") & (L.v0 > 5) & (L.v_min < 0.5) & L.restart_s.between(2.0, 4.5)]
    elif kind == "hard": c = L[(L.kind == "hard") & (L.a_need > 3.5) & (L.v_min < 0.5)]
    elif kind == "popout": c = L[L.kind.str.startswith("popout") & (L.v_min < 1.0)]
    else: c = L[(L.kind == "pass") & (L.v0 > 5) & L.ptype.isin(["cross_lr", "cross_rl", "along"])]
    first = c.sort_values("i").drop_duplicates(["episode", "obj"])                   # first row of an event
    return first.sample(min(k, len(first)), random_state=1)


def draw(ax1, ax2, R, r, T, F, L):
    z = np.load(Path(R) / "_ego" / f"{r.episode}.npz"); tau, v, s_e, d_e = z["tau"], z["v"], z["path_s"], z["d"]
    i = int(r.i); f = F[(F.episode == r.episode) & (F.i == i)].iloc[0]; o = T[(T.episode == r.episode) & (T.n == r.obj)].iloc[0].to_dict()
    t = tau - tau[i]; h = slice(max(i - 30, 0), min(i + dr.N + 1, len(v)))
    ax1.plot(t[h], v[h] * 3.6, color="0.6", lw=2, label="원래 주행")
    tf = dr.DT * np.arange(1, dr.N + 1); vv = f.fut_v * 3.6
    ax1.plot(np.r_[0, tf], np.r_[v[i] * 3.6, vv], color="C3", lw=2, label="합성 라벨")
    if np.isfinite(r.restart_s): ax1.axvline(r.restart_s, color="C3", ls="--", lw=1, label="물체가 경로 폭을 벗어남 (+0.5 s 뒤 재출발)")
    ev = L[(L.episode == r.episode) & (L.obj == r.obj) & (L.kind != "pass")]
    ax1.axvline(tau[int(r.seen_i)] - tau[i], color="C0", ls=":", label="처음 보임")
    if len(ev): ax1.axvline(tau[int(ev.i.min())] - tau[i], color="C1", ls="-.", label="충돌 경로 시작")
    ax1.axvline(0, color="k", lw=0.8)
    ax1.set_xlabel("t − t0 (s)"); ax1.set_ylabel("km/h"); ax1.set_ylim(bottom=-1); ax1.legend(fontsize=7, loc="best")
    ax1.set_title(f"{r.episode} i={i} {o['cls']}/{o['ptype']}/{o['timing']}\nv0 {r.v0 * 3.6:.0f} km/h, a_need {r.a_need:.2f}, D {r.d_stop:.1f} m", fontsize=8)
    w = slice(max(i - 30, 0), min(i + dr.N + 1, len(v)))
    ax2.plot(s_e[w], d_e[w], color="0.6", lw=2, label="자차 경로")
    corr = dr.HALF_W + dr.PAD + dr.R_OBJ
    ax2.fill_between(s_e[w], d_e[w] - corr, d_e[w] + corr, color="0.85", label="경로 폭")
    tt = tau[max(i - 30, 0):min(i + dr.N + 1, len(v))]
    so, lo, _, _, started = dr.obj_state(o, tt)
    ax2.plot(so[started], lo[started], ".", color="C2", ms=3, label="물체 궤적")
    so0, lo0, *_ = dr.obj_state(o, np.array([tau[i]]))
    ax2.plot(so0, lo0, "o", color="C2", ms=8, mfc="none", label="물체 @ t0")
    ax2.plot(s_e[i], d_e[i], "s", color="k", ms=6, label="자차 @ t0 (뒷차축)")
    if np.isfinite(r.d_stop): ax2.axvline(s_e[i] + dr.FRONT + r.d_stop, color="C3", ls="--", label="정지점 (코)")
    ax2.set_xlabel("경로 s (m)"); ax2.set_ylabel("횡 lat (m, 왼쪽 +)"); ax2.legend(fontsize=7, loc="best"); ax2.set_aspect("auto")


def main():
    R, O, D = map(Path, sys.argv[1:4]); D.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
    L = pd.read_parquet(O / "dyn_relabel_index.parquet"); F = pd.read_parquet(O / "dyn_future.parquet"); T = pd.read_parquet(O / "tracks.parquet")
    dr.A_MAX = 7.0
    for kind, title in KINDS.items():
        ex = pick(L, kind)
        fig, ax = plt.subplots(len(ex), 2, figsize=(13, 4.2 * len(ex)), squeeze=False)
        for n, r in enumerate(ex.itertuples()): draw(ax[n, 0], ax[n, 1], R, r, T, F, L)
        fig.suptitle(title); fig.tight_layout(); fig.savefig(D / f"dyn_{kind}.png", dpi=110); plt.close(fig)
    t = L.groupby(["kind", "cls", "split"], observed=True).size().unstack("split", fill_value=0)
    t["with_restart"] = L[L.restart_s.notna()].groupby(["kind", "cls"]).size()
    t = t.fillna(0).astype(int)
    (D / "kinds.md").write_text("| kind | cls | " + " | ".join(t.columns) + " |\n|---|---|" + "---|" * len(t.columns) + "\n" +
                                "\n".join(f"| {k} | {c} | " + " | ".join(str(x) for x in row) + " |" for (k, c), row in zip(t.index, t.values)) + "\n")
    print((D / "kinds.md").read_text())


if __name__ == "__main__":
    main()
