"""Check figures. `req`: pick episodes, write pull_req.json for winpc. `plot`: per-episode before/after figures + contact sheets."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).parent; OUT = HERE / "start_relabel"; PULL = HERE / "pull_out"; FIG = OUT / "figs"
EGO = Path.home() / ".cache/step0_ego"
A = pd.read_parquet(OUT / "start_rows_all.parquet"); F = pd.read_parquet(OUT / "start_future.parquet").set_index("episode")
rng = np.random.default_rng(20261027)
SHEET_N = 200


def pick():
    ep_ok = A[A.status == "relabeled"].groupby("episode").size()
    cp = F.start_cp
    sel = []
    for c in sorted(cp[ep_ok.index].unique()):                       # one random relabeled episode per start checkpoint
        cand = [e for e in ep_ok.index if cp[e] == c]
        sel.append(("relabeled", str(rng.choice(cand))))
    extra = [e for e in ep_ok.index if e not in {s for _, s in sel}]
    while len(sel) < 14: sel.append(("relabeled", str(extra.pop(int(rng.integers(len(extra)))))))
    for k in ("signal_red", "signal_yellow", "signal_unknown", "signal_change", "obstacle_lidar", "moving_object"):
        eps = sorted(A[of_kind(A, k)].episode.unique())
        for e in (rng.choice(eps, min(2 if k != "obstacle_lidar" else 4, len(eps)), replace=False) if eps else []):
            if e not in {s for _, s in sel}: sel.append((k, str(e)))
    sheet = [str(e) for e in rng.choice(ep_ok.index, min(SHEET_N, len(ep_ok)), replace=False)]
    return sel, sheet


def of_kind(g, kind):
    if kind == "obstacle_lidar": return g.obstacle.to_numpy()          # LiDAR hits can sit under signal_*
    if kind == "moving_object": return g.moving_near.to_numpy()
    return (g.status == kind).to_numpy()


def rows_of(ep):
    return A[A.episode == ep].sort_values("i")


if sys.argv[1] == "req":
    sel, sheet = pick(); req = {}
    for kind, ep in sel:
        g = rows_of(ep); z = np.load(EGO / f"{ep}.npz"); tau = z["tau"]; d = int(F.loc[ep, "src_i"])
        i0 = int(g.i.iloc[0]); im = int(g.i.iloc[len(g) // 2])
        tag_row = dict(first=i0, mid=im, last=d - 1, go2s=min(d + 20, len(tau) - 1))
        if kind != "relabeled":                                         # show a frame of the excluded kind
            gk = g[of_kind(g, kind)]; tag_row["excl"] = int(gk.i.iloc[len(gk) // 2])
        req[ep] = dict(ep=ep, cams=[[t, float(tau[i])] for t, i in tag_row.items()], bevs=[["mid", float(tau[tag_row.get("excl", im)])]])
    for ep in sheet:
        g = rows_of(ep); g = g[g.status == "relabeled"]; z = np.load(EGO / f"{ep}.npz")    # the middle relabeled row
        r = req.setdefault(ep, dict(ep=ep, cams=[], bevs=[]))
        r["cams"].append(["sheet", float(z["tau"][int(g.i.iloc[len(g) // 2])])])
    json.dump(dict(sel=sel, sheet=sheet), open(HERE / "fig_sel.json", "w"))
    json.dump(list(req.values()), open(HERE / "pull_req.json", "w"))
    print(len(sel), "figure episodes,", len(sheet), "sheet episodes,", sum(len(r["cams"]) for r in req.values()), "frames")
    sys.exit()

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

FIG.mkdir(parents=True, exist_ok=True)
S = json.load(open(HERE / "fig_sel.json")); SIGN = {0: "unknown", 1: "red", 2: "yellow", 3: "green"}
DT, N_FUT = 0.1, 64


def fut_xy(v, yaw, i):
    j = np.arange(i, i + N_FUT + 1); h = yaw[j] - yaw[i]
    step = 0.5 * (v[j][1:] + v[j][:-1]) * DT; hm = 0.5 * (h[1:] + h[:-1])
    return np.cumsum(step * np.cos(hm)), np.cumsum(step * np.sin(hm))


for k, (kind, ep) in enumerate(S["sel"]):
    z = np.load(EGO / f"{ep}.npz"); v = z["v"].astype(float); yaw = z["yaw"]; tau = z["tau"]; sig = z["signal"]
    g = rows_of(ep); d = int(F.loc[ep, "src_i"]); i0 = int(g.i.iloc[0])
    gk = g[of_kind(g, kind)]; ishow = int(gk.i.iloc[len(gk) // 2]) if kind != "relabeled" else int(g.i.iloc[len(g) // 2])
    st = g.set_index("i").status; st_obs = g.set_index("i").s_obs.fillna(-1)
    fig = plt.figure(figsize=(16, 9.5)); gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15])
    tags = ["first", "mid" if kind == "relabeled" else "excl", "last", "go2s"]
    rowof = dict(first=i0, mid=int(g.i.iloc[len(g) // 2]), excl=ishow, last=d - 1, go2s=min(d + 20, len(tau) - 1))
    for c, t in enumerate(tags):
        ax = fig.add_subplot(gs[0, c]); f = PULL / f"{ep}_{t}.jpg"
        if f.exists(): ax.imshow(Image.open(f))
        i = rowof[t]
        ax.set_title(f"{t}: row {i}, tau {tau[i]:.1f}s, v {v[i]:.1f}\nsignal {SIGN[int(sig[i])]}, {st.get(i, 'after departure')}", fontsize=9)
        ax.axis("off")
    # trajectories: original label at the shown row vs relabeled (future of the departure row)
    ax = fig.add_subplot(gs[1, 0:2])
    bev = PULL / f"{ep}_mid.npy"
    if bev.exists():                                                    # ground returns grey, 0.45-3 m objects (the check's band) black
        h = np.load(bev); rr, cc = np.nonzero(h); hx, hy = 60 - (rr + 0.5) * 0.25, 20 - (cc + 0.5) * 0.25
        obj = (h[rr, cc] >= 22) & (h[rr, cc] <= 70)
        ax.scatter(hy[~obj], hx[~obj], s=0.3, c="0.8", marker="s"); ax.scatter(hy[obj], hx[obj], s=3, c="k", marker="s", label="LiDAR 0.45-3 m")
    xo, yo = fut_xy(v, yaw, ishow); xr, yr = fut_xy(v, yaw, d)
    so = st_obs.get(ishow, np.nan)
    if so >= 0:
        q = min(int(np.searchsorted(np.r_[0, np.cumsum(np.hypot(np.diff(np.r_[0, xr]), np.diff(np.r_[0, yr])))], so)), len(xr) - 1)
        ax.plot(yr[q], xr[q], "mX", ms=12, label=f"LiDAR object in corridor at {so:.1f} m")
    ax.plot(yo, xo, "r.-", ms=3, lw=1, label=f"original label at row {ishow} ({np.hypot(xo[-1], yo[-1]):.1f} m in 6.4 s)")
    ax.plot(yr, xr, "g.-", ms=3, lw=1, label=f"relabeled = departure row {d} ({np.hypot(xr[-1], yr[-1]):.1f} m)")
    px, py = np.r_[0, xr], np.r_[0, yr]; tx, ty = np.gradient(px), np.gradient(py); nn = np.hypot(tx, ty) + 1e-9
    for side in (-1, 1): ax.plot(py + side * tx / nn, px - side * ty / nn, "g:", lw=0.6)
    ax.plot([0], [0], "k^", ms=8); ax.set_xlim(20, -20); ax.set_ylim(-5, 60); ax.set_xlabel("y (m, left +)"); ax.set_ylabel("x (m, forward)")
    ax.set_title(f"BEV (LiDAR height, row {ishow}) + future xy; corridor +-1 m (dotted)", fontsize=9); ax.legend(fontsize=8, loc="lower left")
    ax = fig.add_subplot(gs[1, 2:4]); t = np.arange(1, N_FUT + 1) * DT
    ax.plot(t, v[ishow + 1:ishow + N_FUT + 1], "r-", label="original label speed")
    ax.plot(t, v[d + 1:d + N_FUT + 1], "g-", label="relabeled speed")
    ax.set_xlabel("future time (s)"); ax.set_ylabel("v (m/s)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    n_ok = int((g.status == "relabeled").sum())
    ax.set_title(f"pre-departure rows {len(g)}: relabeled {n_ok}, "
                 + ", ".join(f"{s} {c}" for s, c in g.status.value_counts().items() if s != "relabeled"), fontsize=9)
    fig.suptitle(f"{ep}  [{kind}]  start CP{int(F.loc[ep, 'start_cp'])}  departure tau {tau[d]:.1f}s  scene {z['scene']}", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / f"{k:02d}_{kind}_{ep}.png", dpi=70); plt.close(fig)

# contact sheets: front frame at the middle of the stop, 10 x 10 per sheet
W, H, C = 224, 128, 10
for s0 in range(0, len(S["sheet"]), C * C):
    eps = S["sheet"][s0:s0 + C * C]; sheet = Image.new("RGB", (C * W, ((len(eps) + C - 1) // C) * H), "white")
    for n, ep in enumerate(eps):
        f = PULL / f"{ep}_sheet.jpg"
        if not f.exists(): continue
        im = Image.open(f).resize((W, H)); dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, W, 11], fill="black"); dr.text((2, 0), f"{ep[3:]} CP{int(F.loc[ep, 'start_cp'])}", fill="yellow")
        sheet.paste(im, ((n % C) * W, (n // C) * H))
    sheet.save(FIG / f"sheet_{s0 // (C * C)}.jpg", quality=85)
print("figures in", FIG)
