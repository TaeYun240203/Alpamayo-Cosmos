#!/usr/bin/env python3
"""The static box rule (user 10/29): static objects stand anywhere in the Ego's own lane across the box, and a footprint
counts once it is wholly inside the lane or reaches the band the Ego's body sweeps down the lane's middle (a big box whose
centre is outside still blocks the Ego). Left: the rule on a straightened stretch with sample footprints, judged by the
generator's own in_ego_lane(); right: where the v2 generator (eval/sync working tree) puts them over many scenes.
    uv run --with matplotlib --with numpy python3 reports/static_lane_rule.py CLCOL_JSON [OUT_PNG]
"""
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

W = Path(__file__).resolve().parents[2]
SRC = W / "MORAI-eval-sync/src"
spec = importlib.util.spec_from_file_location("sv", SRC / "e2e_driver/tools/scenario_variants.py")
sv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sv)
plt.rcParams["font.family"] = ["AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def road(ax, lo, hi, sec_f=None, lane_f=None):
    s = np.arange(lo, hi + 0.25, 0.25)
    sec = np.array([(sec_f or sv.static_sec)(q) for q in s])
    lane = np.array([(lane_f or sv.ego_lane)(q) for q in s])
    c = lane.mean(1)
    ax.fill_between(s, sec[:, 0], sec[:, 2], color="0.93", zorder=0, label="도로 (양방향, 정적 물체 발판 허용 범위)")
    ax.fill_between(s, lane[:, 0], lane[:, 1], color="#cfe3f7", zorder=1, label="에고 주행 차로 (lanelet2)")
    ax.fill_between(s, c - sv.EGO_PASS / 2, c + sv.EGO_PASS / 2, facecolor="none", edgecolor="tab:blue", hatch="///",
                    lw=0, zorder=2, label=f"차로 가운데로 달리는 Ego 몸통 띠 ({sv.EGO_PASS:.2f} m)")
    ax.plot(s, lane[:, 1], color="goldenrod", lw=1.4, zorder=3, label="차로 왼쪽 = 중앙선")
    ax.plot(s, lane[:, 0], color="0.4", lw=1.2, zorder=3, label="차로 오른쪽 경계")
    ax.plot(s, sec[:, 0], color="0.2", lw=0.8, zorder=3)
    ax.plot(s, sec[:, 2], color="0.2", lw=0.8, zorder=3, label="도로 끝")


def foot_xy(i, s, lat, dyaw):
    ln, wd = sv.size_of(i)
    y = math.radians(dyaw)
    cs = np.array([[ln / 2, wd / 2], [ln / 2, -wd / 2], [-ln / 2, -wd / 2], [-ln / 2, wd / 2], [ln / 2, wd / 2]])
    return cs @ np.array([[math.cos(y), math.sin(y)], [-math.sin(y), math.cos(y)]]) + [s, lat]


def concept(ax):
    lo, hi = 352.0, 394.0
    road(ax, lo, hi)
    samples = [  # (label, asset, s, lat, dyaw)
        ("작은 콘: 차로 안 가장자리", 40100027, 356.0, -1.35, 0.0),
        ("승용차: 차로 가운데", 40100013, 363.0, 0.3, 8.0),
        ("큰 화물차: 중심은 반대 차로,\n몸체가 Ego 띠까지 들어옴", 40100009, 373.0, 2.9, 80.0),
        ("상자: 절반이 차로 밖,\nEgo 띠에 안 닿음", 40100002, 383.0, -1.9, 0.0),
        ("승용차: 반대 차로 안", 40100013, 390.0, 3.6, 0.0),
    ]
    for k, (name, i, s, lat, dy) in enumerate(samples):
        hl, hw = sv.foot(i, dy)
        ok = sv.in_ego_lane(s, (s - hl, s + hl, lat - hw, lat + hw))
        q = foot_xy(i, s, lat, dy)
        ax.fill(q[:, 0], q[:, 1], color="tab:green" if ok else "tab:red", alpha=0.65, zorder=5)
        ax.plot(lat * 0 + s, lat, "k+", ms=7, zorder=6)
        ax.annotate(("✓ 포함  " if ok else "✗ 제외  ") + name, (s, lat), xytext=(s - 3.0, [-5.3, 6.0, 7.8, -4.4, 4.6][k]),
                    fontsize=8.5, color="darkgreen" if ok else "darkred", zorder=7,
                    arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6))
    ax.set_xlim(lo, hi)
    ax.set_ylim(-6, 9)
    ax.set_aspect("equal")
    ax.set_xlabel("경로 s (m, 펴서 그림)")
    ax.set_ylabel("왼쪽 + (m)")
    ax.set_title("규칙: 물체가 에고 차로 안에 통째로 있거나, 차로 가운데로 달리는 Ego 몸통 띠까지 들어오면 정적 장애물\n"
                 "(그 밖: 도로 위 발판, 물체끼리 안 겹침, 도로 전체가 막히지 않음은 그대로; + = 물체 중심)", fontsize=10)
    ax.legend(loc="upper left", fontsize=7.5, ncol=2)


def distribution(ax, axh, metas, n, zone="static", box=sv.STATIC_BOX, sec_f=None, lane_f=None, name="정적 박스"):
    st = [a for m in metas for a in m["actors"] if a["zone"] == zone]
    lane_f = lane_f or sv.ego_lane
    lo, hi = box
    road(ax, lo - 4, hi + 4, sec_f, lane_f)
    ax.axvspan(lo, hi, color="limegreen", alpha=0.06, zorder=0)
    alpha = min(0.4, 80.0 / len(st))
    for a in st:
        q = foot_xy(a["id"], a["s"], a["lat"], a["yaw"])
        ax.fill(q[:, 0], q[:, 1], color="tab:blue", alpha=alpha, lw=0, zorder=4)
    ax.scatter([a["s"] for a in st], [a["lat"] for a in st], s=1.2, c="navy", alpha=0.35, lw=0, zorder=5)
    ax.set_xlim(lo - 4, hi + 4)
    ax.set_ylim(-6, 9)
    ax.set_aspect("equal")
    ax.set_xlabel("경로 s (m, 펴서 그림)")
    ax.set_ylabel("왼쪽 + (m)")
    big = sum(1 for a in st if not (lane_f(a["s"])[0] <= a["lat"] <= lane_f(a["s"])[1]))
    ax.set_title(f"v2 생성기 {name} {n}장면: 물체 {len(st)}개 (발판 실제 크기·방향, 점 = 중심) · "
                 f"중심이 차로 밖인 물체 {big}개 ({100 * big / len(st):.0f} %)", fontsize=10)
    # lateral: centres and footprint coverage relative to the lane's middle
    rel = np.array([a["lat"] - np.mean(lane_f(a["s"])) for a in st])
    axh.hist(rel, bins=np.arange(-5, 5.01, 0.25), color="navy", alpha=0.6, label="물체 중심")
    cover = np.zeros(400)
    grid = np.linspace(-5, 5, 400)
    for a in st:
        hl, hw = sv.foot(a["id"], a["yaw"])
        c = np.mean(lane_f(a["s"]))
        cover += (grid >= a["lat"] - hw - c) & (grid <= a["lat"] + hw - c)
    axh2 = axh.twinx()
    axh2.plot(grid, cover / len(st), color="tab:orange", label="발판이 덮는 비율")
    lane = np.array([lane_f(q) for q in np.arange(lo, hi, 1.0)])
    half = np.mean(lane[:, 1] - lane[:, 0]) / 2
    for x in (-half, half):
        axh.axvline(x, color="goldenrod" if x > 0 else "0.4", lw=1)
    for x in (-sv.EGO_PASS / 2, sv.EGO_PASS / 2):
        axh.axvline(x, color="tab:blue", lw=1, ls="--")
    axh.set_xlabel("차로 가운데로부터 가로 위치 (m, 왼쪽 +; 실선 = 평균 차로 경계, 점선 = Ego 띠)")
    axh.set_ylabel("물체 중심 수")
    axh2.set_ylabel("덮는 비율")
    axh.legend(loc="upper left", fontsize=8)
    axh2.legend(loc="upper right", fontsize=8)
    axh.set_title(f"{name} 가로 분포", fontsize=10)


def main():
    base = json.load(open(sys.argv[1]))
    route = sv.Route(str(SRC / "map/2026_molit_comp_global_path.txt"))
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "scenario_random/static_rule.png"
    n = 3000
    metas = [json.loads(json.dumps(sv.make(base, route, random.Random(k), 1000)[1])) for k in range(n)]
    fig = plt.figure(figsize=(16, 21))
    gs = fig.add_gridspec(5, 1, height_ratios=[1.0, 1.25, 0.8, 0.9, 0.8])
    concept(fig.add_subplot(gs[0]))
    distribution(fig.add_subplot(gs[1]), fig.add_subplot(gs[2]), metas, n)
    tun = sum(m.get("tunnel") == "both" for m in metas)
    distribution(fig.add_subplot(gs[3]), fig.add_subplot(gs[4]), metas, n, "tunnel_static", sv.TUNNEL_S,
                 lambda q: sv.TUNNEL_SEC, lambda q: sv.TUNNEL_LANE, f"터널 (정적 포함 {tun}장면)")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    print(out)


if __name__ == "__main__":
    main()
