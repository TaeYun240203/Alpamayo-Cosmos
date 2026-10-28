"""Sample figures for the user's check: front camera with original and relabeled futures projected, top view, speed profiles.
`req`: write pull_req_samples.json for winpc pull.py. `plot`: render into the workspace reports/start_relabel_samples/."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).parent; OUT = HERE / "start_relabel"; PULL = HERE / "pull_samples"
FIG = Path("/Users/taeyunkim/Library/Mobile Documents/com~apple~CloudDocs/Workspaces/Alpamayo-Cosmos/reports/start_relabel_samples")
EGO = Path.home() / ".cache/step0_ego"
A = pd.read_parquet(OUT / "start_rows_all.parquet"); F = pd.read_parquet(OUT / "start_future.parquet").set_index("episode")
rng = np.random.default_rng(1027)
DT, N_FUT = 0.1, 64

ok_eps = A[A.status == "relabeled"].groupby("episode").size().index
def pick_cp(cp, prefer=None):
    if prefer: return prefer
    cand = sorted(e for e in ok_eps if F.loc[e, "start_cp"] == cp); return str(rng.choice(cand))
# (kind, episode, row selector)
SAMPLES = [("relabeled", pick_cp(0, "run1021n-w1-014"), None), ("relabeled", pick_cp(1), None), ("relabeled", pick_cp(3, "run1021v-w2-321"), None),
           ("relabeled", pick_cp(5, "run1020i-w2-004"), None), ("relabeled", pick_cp(8, "run1020k-w2-005"), None),
           ("relabeled", pick_cp(11, "run1021f-w2-001"), None), ("relabeled", pick_cp(12, "run1020h-w2-006"), None),
           ("relabeled", pick_cp(14), None),
           ("signal_red", "run1024a-w1-027", None), ("obstacle", "run1023g-w1-010", None), ("signal_yellow", "run1021v-w1-154", None),
           ("signal_unknown", "run1020h-w1-004", None), ("signal_red", "run1023d-w1-001", "dropout")]


def sample_row(kind, ep, sel=None):
    g = A[A.episode == ep].sort_values("i")
    m = {"relabeled": g.status == "relabeled", "obstacle": g.obstacle}.get(kind, g.status == kind)
    if sel == "dropout":                                              # red at the row, GT missing for a moment inside its 6.4 s
        s = np.load(EGO / f"{ep}.npz")["signal"]
        m = m & np.array([s[i] == 1 and (s[i:i + 65] == 0).any() for i in g.i])
    gk = g[m.to_numpy()]
    return int(gk.i.iloc[len(gk) // 2]), g


if sys.argv[1] == "req":
    req = []
    for kind, ep, sel in SAMPLES:
        i, _ = sample_row(kind, ep, sel); tau = np.load(EGO / f"{ep}.npz")["tau"]
        req.append(dict(ep=ep, cams=[["s", float(tau[i])]], bevs=[["s", float(tau[i])]]))
    json.dump(req, open(HERE / "pull_req_samples.json", "w")); print(len(req)); sys.exit()

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams["font.family"] = "AppleGothic"; plt.rcParams["axes.unicode_minus"] = False
FIG.mkdir(parents=True, exist_ok=True)
SIGN = {0: "없음", 1: "적", 2: "황", 3: "녹"}
REASON = {"relabeled": "재라벨", "signal_red": "제외: 이 행이 적신호 (정답 구간 신호가 출발 시점과 다름, GT 끊김 포함)",
          "signal_yellow": "제외: 황신호 → 적신호로 바뀌는 구간 (정답의 정지 시점이 신호에 달림)",
          "obstacle": "제외: 정답 경로 ±1 m 안에 LiDAR 물체 (교차로를 지나는 차량)",
          "signal_unknown": "제외: 신호 교차로인데 6.4 s 구간에 GT 신호가 빔"}
SIGC = {0: "0.75", 1: "red", 2: "orange", 3: "green"}
# front camera: 1280x720, HFOV 90 deg (bridge config.yaml), Camera Transform x 1.90 / z 1.20 / pitch 2 deg down (MORAI GUI); step-0 448x256
CX_M, CZ_M, PITCH = 1.90, 1.20, np.radians(2.0)
W, H = 448, 256; FX, FY = 640 * W / 1280, 640 * H / 720


def fut_xy(v, yaw, i):
    j = np.arange(i, i + N_FUT + 1); h = yaw[j] - yaw[i]
    step = 0.5 * (v[j][1:] + v[j][:-1]) * DT; hm = 0.5 * (h[1:] + h[:-1])
    return np.r_[0, np.cumsum(step * np.cos(hm))], np.r_[0, np.cumsum(step * np.sin(hm))]


def project(x, y):
    X, Y, Z = x - CX_M, y, -CZ_M * np.ones_like(x)
    depth = X * np.cos(PITCH) - Z * np.sin(PITCH); up = X * np.sin(PITCH) + Z * np.cos(PITCH)
    ok = depth > 0.5
    return W / 2 - FX * Y / np.where(ok, depth, np.nan), H / 2 - FY * up / np.where(ok, depth, np.nan)


for n, (kind, ep, sel) in enumerate(SAMPLES):
    z = np.load(EGO / f"{ep}.npz"); v = z["v"].astype(float); yaw = z["yaw"]; tau = z["tau"]; sig = z["signal"]
    i, g = sample_row(kind, ep, sel); d = int(F.loc[ep, "src_i"]); r = g.set_index("i").loc[i]
    xo, yo = fut_xy(v, yaw, i); xr, yr = fut_xy(v, yaw, d)
    fig = plt.figure(figsize=(17, 8.4)); gs = fig.add_gridspec(3, 2, width_ratios=[1.7, 1], height_ratios=[1.25, 1, 0.42])
    ax = fig.add_subplot(gs[:, 0]); f = PULL / f"{ep}_s.jpg"
    if f.exists(): ax.imshow(Image.open(f))
    for (x, y, c, lab) in ((xo, yo, "red", "원래 정답"), (xr, yr, "lime", "재라벨 정답")):
        u, w = project(x, y); ax.plot(u, w, "-", c=c, lw=2, alpha=0.85, label=lab)
        k = np.arange(10, N_FUT + 1, 10); ax.plot(u[k], w[k], "o", c=c, ms=6, mec="k")        # 1 s marks
        for kk in (20, 40, 60):                                         # label only marks inside the image (outside ones break the layout)
            if np.isfinite(u[kk]) and 0 <= u[kk] <= W and 0 <= w[kk] <= H: ax.text(u[kk] + (6 if c == "red" else -6), w[kk], f"{kk // 10}s", color=c, fontsize=9, weight="bold",
                                           ha="left" if c == "red" else "right", va="center")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off"); ax.legend(loc="lower left", fontsize=10)
    ax.set_title(f"전방 카메라, 행 {i} (τ {tau[i]:.1f} s, v {v[i]:.1f} m/s), 점 = 1 s 간격", fontsize=11)
    ax = fig.add_subplot(gs[0, 1]); bev = PULL / f"{ep}_s.npy"
    if bev.exists():
        h = np.load(bev); rr, cc = np.nonzero(h); hx, hy = 60 - (rr + 0.5) * 0.25, 20 - (cc + 0.5) * 0.25
        obj = (h[rr, cc] >= 22) & (h[rr, cc] <= 70)
        ax.scatter(hy[~obj], hx[~obj], s=0.3, c="0.85", marker="s"); ax.scatter(hy[obj], hx[obj], s=3, c="k", marker="s", label="LiDAR 물체 0.45–3 m")
    ax.plot(yo, xo, "r.-", ms=3, lw=1, label=f"원래 ({np.hypot(xo[-1], yo[-1]):.1f} m / 6.4 s)")
    ax.plot(yr, xr, "g.-", ms=3, lw=1, label=f"재라벨 ({np.hypot(xr[-1], yr[-1]):.1f} m / 6.4 s)")
    if r.obstacle:
        q = min(int(np.searchsorted(np.r_[0, np.cumsum(np.hypot(np.diff(xr), np.diff(yr)))], r.s_obs)), N_FUT)
        ax.plot(yr[q], xr[q], "mX", ms=13, label=f"경로 위 물체 {r.s_obs:.1f} m")
    ax.plot([0], [0], "k^", ms=8); ax.set_xlim(20, -20); ax.set_ylim(-5, 60); ax.set_aspect("equal")
    ax.set_xlabel("y (m, 왼쪽 +)"); ax.set_ylabel("x (m, 앞)"); ax.legend(fontsize=8, loc="lower left"); ax.set_title("위에서 본 궤적 + LiDAR", fontsize=10)
    ax = fig.add_subplot(gs[1, 1]); t = np.arange(1, N_FUT + 1) * DT
    ax.plot(t, v[i + 1:i + N_FUT + 1], "r-", label="원래 정답 속도"); ax.plot(t, v[d + 1:d + N_FUT + 1], "g-", label="재라벨 정답 속도")
    ax.set_xlabel("미래 시간 (s)"); ax.set_ylabel("v (m/s)"); ax.grid(alpha=0.3); ax.legend(fontsize=9); ax.set_title("속도 프로필 전후", fontsize=10)
    ax = fig.add_subplot(gs[2, 1]); t1 = min(len(tau) - 1, d + N_FUT + 20); t0 = max(0, min(i, d) - 20)
    for k in range(t0, t1):
        ax.axvspan(tau[k], tau[k] + DT, ymin=0.35, ymax=0.95, color=SIGC[int(sig[k])], lw=0)
    for k0, c, lab in ((i, "red", "이 행"), (d, "green", "출발")):
        ax.add_patch(plt.Rectangle((tau[k0], 0.02), N_FUT * DT, 0.3, fill=False, ec=c, lw=1.5, transform=ax.get_xaxis_transform()))
        ax.text(tau[k0], 0.17, f" {lab}+6.4 s", color=c, fontsize=8, va="center", transform=ax.get_xaxis_transform())
    ax.set_xlim(tau[t0], tau[t1]); ax.set_yticks([]); ax.set_xlabel("τ (s)")
    ax.set_title("GT 신호 (회색 = 없음, 빨강·주황·초록 = 적·황·녹) / 상자 = 6.4 s 정답 구간", fontsize=9)
    line = f", 다음 판정선 {r.line_d:.0f} m 앞" if np.isfinite(r.line_d) else ""
    cp = int(F.loc[ep, "start_cp"]); start = "랩 출발" if cp == 0 else f"CP{cp} 출발"
    fig.suptitle(f"[{REASON.get(kind, kind)}]  {ep}  {start}\n출발(속도 0 마지막 행) {d}행 τ {tau[d]:.1f} s → 이 행은 {tau[d] - tau[i]:.1f} s 당김. "
                 f"신호: 이 행 {SIGN[int(sig[i])]}, 출발 시점 {SIGN[int(sig[d])]}{line}", fontsize=12)
    fig.tight_layout(); name = f"{n:02d}_{'relabel' if kind == 'relabeled' else 'excluded_' + kind}_{'lap' if cp == 0 else f'cp{cp}'}_{ep}.png"
    fig.savefig(FIG / name, dpi=72); plt.close(fig); print(name)
