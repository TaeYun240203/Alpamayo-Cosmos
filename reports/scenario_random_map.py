#!/usr/bin/env python3
"""Where scenario_variants.py puts obstacles and NPCs (user 10/29): the unified v2 generator (working tree of rl/ppo-20261028
and eval/sync-20261028, one file for lap videos, sync eval and the real-time actors) against the generators before it
(git HEAD of both branches: lap videos / sync with the default preset, the real-time actors' DAgger dyn/tunnel segments
with the brake preset), on the map with the rules' §3-1 boxes registered onto the route.
Every scene is random.Random(S*1000+k) of the generator alone, so the scenes are regenerated here; the scene_meta that
real runs recorded (~/morai-eval) is checked against the old generators first.
    uv run --with matplotlib --with numpy python3 reports/scenario_random_map.py CLCOL_JSON [OUT_DIR]
"""
import collections
import glob
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

W = Path(__file__).resolve().parents[2]
REL = "src/e2e_driver/tools/scenario_variants.py"
MAP = W / "MORAI-eval-ppo/src/map"
CPS_S = [0.0, 72.6, 186.3, 237.4, 346.7, 471.1, 580.9, 635.1, 858.5, 930.6, 1118.7, 1614.6, 1741.7, 1859.6, 1996.7]   # 10/7
EVAL_S = [s for s in range(100) if s not in (7, 28, 29)]
VIDEO_SEEDS = [s * 1000 + k for s in EVAL_S for k in range(8)]
CONFIGS = {"v2": ("v2", "", [s * 1000 + k for s in EVAL_S for k in range(20)])}   # figures: v2 only (user 10/29)
LABEL = {"old_video": "이전: 영상 제작기·동기 (기본)", "old_rt": "이전: 실시간 액터 dyn·tunnel (brake)",
         "v2": "v2 통합 (세 경로 공통)"}
COL = {"old_video": "tab:gray", "old_rt": "tab:orange", "v2": "tab:blue"}
CLS_C = {"person": "tab:green", "other": "tab:orange", "car": "tab:red"}
SHAPE_C = {"cross_lr": "tab:purple", "cross_rl": "tab:brown", "from_centre": "magenta", "behind_parked": "black"}
SHAPE_L = {"cross_lr": "출발: 반대편 갓길 → 오른쪽", "cross_rl": "출발: 오른쪽 갓길 → 왼쪽", "from_centre": "출발: 중앙선 바로 너머 → 오른쪽",
           "behind_parked": "출발: 주차차 뒤 → 왼쪽"}
BOXES = {"static": (280.0, 390.0), "dynamic": (421.0, 503.0)}   # §3-1 figure on the route (+-5 m)
V_NOM = {"ra": 8.5, "rb": 14.0}   # the generator's V_NOMINAL


def load(path, name, preset):
    os.environ["SCENE_PRESET"] = preset   # the old generators read it at import; v2 ignores it
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gens(tmp):
    out = {}
    for name, repo, preset in (("old_sync", "MORAI-eval-sync", ""), ("old_ppo", "MORAI-eval-ppo", "brake")):
        p = Path(tmp) / f"{name}.py"
        p.write_text(subprocess.run(["git", "-C", str(W / repo), "show", f"HEAD:{REL}"], check=True,
                                    capture_output=True, text=True).stdout)
        out[name] = load(p, name, preset)
    out["v2"] = load(W / "MORAI-eval-ppo" / REL, "v2", "")
    return out


def scene(m, base, route, seed):
    return json.loads(json.dumps(m.make(base, route, random.Random(seed), 1000)[1]))   # as recorded (JSON)


def verify(g, base, route):
    """recorded scene_meta (video laps, DAgger episodes) == regeneration with the old generators"""
    files = glob.glob(os.path.expanduser("~/morai-eval/pod/2026*/*/w1/eval_map/laps.jsonl")) + \
        glob.glob(os.path.expanduser("~/morai-eval/dagger/shadow*/spool/*/episodes.jsonl"))
    seen, out = set(), collections.Counter()
    cand = {"default": g["old_sync"], "brake": g["old_ppo"]}
    for f in files:
        for row in map(json.loads, filter(str.strip, open(f))):
            meta = row.get("scene_meta") or {}
            key = (row["seed"], json.dumps(meta.get("actors"), sort_keys=True))
            if "tunnel" not in meta or key in seen:   # no "tunnel": the 10/27 generator before R2 obstacles
                continue
            seen.add(key)
            hit = [k for k, m in cand.items() if scene(m, base, route, row["seed"])["actors"] == meta["actors"]]
            out[("video" if "/pod/" in f else "dagger"), "+".join(hit) or "none"] += 1
    return out


def lanelet_lines():
    root = ET.parse(MAP / "kcity_lanelet2.osm").getroot()
    xy = {}
    for n in root.iter("node"):
        t = {k.get("k"): k.get("v") for k in n.iter("tag")}
        xy[n.get("id")] = (float(t["local_x"]), float(t["local_y"]))
    lines = []
    for w in root.iter("way"):
        t = {k.get("k"): k.get("v") for k in w.iter("tag")}
        if t.get("type") == "virtual":
            continue
        pts = [xy[nd.get("ref")] for nd in w.iter("nd") if nd.get("ref") in xy]
        if len(pts) > 1:
            lines.append((np.array(pts), t.get("type", ""), t.get("subtype", ""), t.get("color", "")))
    return lines


def frame(route, s_lo, s_hi):
    """map xy -> zone frame: origin on the route at the zone middle, x along its heading (true geometry, rotated)"""
    x0, y0, _, h = route.at((s_lo + s_hi) / 2, 0.0)
    c, s = math.cos(math.radians(h)), math.sin(math.radians(h))
    return lambda p: np.stack([(np.asarray(p)[..., 0] - x0) * c + (np.asarray(p)[..., 1] - y0) * s,
                               -(np.asarray(p)[..., 0] - x0) * s + (np.asarray(p)[..., 1] - y0) * c], -1)


def draw_map(ax, lines, route_xy, f=None):
    for pts, typ, sub, col in lines:
        q = f(pts) if f else pts
        style = dict(color="0.35", lw=0.9) if typ == "stop_line" else \
            dict(color="goldenrod" if col == "yellow" else "0.55", lw=0.7, ls="--" if sub == "dashed" else "-") if f else \
            dict(color="0.78", lw=0.4)
        ax.plot(q[:, 0], q[:, 1], zorder=1, **style)
    q = f(route_xy) if f else route_xy
    ax.plot(q[:, 0], q[:, 1], color="tab:blue", lw=1.0, alpha=0.4, zorder=2)


def actors(metas, zones):
    return [a for m in metas for a in m["actors"] if a["zone"] in zones]


def rect(ax, f, a, color, alpha):
    """the static object's footprint (size, yaw) in the zone frame"""
    if "size" not in a:
        p = f(np.array([a["xy"]]))
        ax.scatter(p[:, 0], p[:, 1], s=8, marker="s", c=color, alpha=alpha, zorder=4)
        return
    (ln, wd), y = a["size"], math.radians(a.get("yaw", 0.0))
    cs = np.array([[ln / 2, wd / 2], [ln / 2, -wd / 2], [-ln / 2, -wd / 2], [-ln / 2, wd / 2], [ln / 2, wd / 2]])
    rot = np.array([[math.cos(y), -math.sin(y)], [math.sin(y), math.cos(y)]])
    q = cs @ rot.T + np.array([0.0, 0.0])
    p0 = f(np.array([a["xy"]]))[0]
    ax.fill(q[:, 0] + p0[0], q[:, 1] + p0[1], color=color, alpha=alpha, lw=0, zorder=4)


def zooms(path, route, lines, metas):
    rows = [("정적 박스 s 280–390", 272, 398, ("static",), (-7, 18), BOXES["static"]),
            ("동적 박스 s 421–503", 412, 512, ("ra_dynamic", "ra_parked"), (-10, 14), BOXES["dynamic"]),
            ("터널 (GPS 음영)", 1872, 2004, ("tunnel_static", "rb_dynamic"), (-4, 8), (1880.0, 1996.0))]
    rxy = np.array([p[:2] for p in route.p])
    fig, axs = plt.subplots(len(rows), len(CONFIGS), figsize=(15, 12.5), squeeze=False,
                            gridspec_kw=dict(height_ratios=[r[4][1] - r[4][0] for r in rows]))
    for r, (title, lo, hi, zones, ylim, box) in enumerate(rows):
        f = frame(route, lo, hi)
        mid = (lo + hi) / 2
        for c, name in enumerate(CONFIGS):
            ax = axs[r][c]
            draw_map(ax, lines, rxy, f)
            if box:
                ax.axvspan(box[0] - mid, box[1] - mid, color="limegreen", alpha=0.08, zorder=0)
            got = actors(metas[name], zones)
            st = [a for a in got if "dynamic" not in a["zone"]]
            dy = [a for a in got if "dynamic" in a["zone"]]
            alpha = min(0.5, 60.0 / max(len(st), 1))
            for a in st:
                rect(ax, f, a, "0.25" if "parked" in a["zone"] else "tab:blue", alpha)
            if st:   # centres too, so sparse places stay visible
                p = f(np.array([a["xy"] for a in st]))
                ax.scatter(p[:, 0], p[:, 1], s=2, c="navy", alpha=0.25, zorder=4, lw=0)
            for a in dy:
                p = f(np.array([a["xy"], a["xy_end"]]))
                ax.plot(p[:, 0], p[:, 1], color=CLS_C[a["cls"]], lw=0.5, alpha=min(0.25, 150.0 / max(len(dy), 1)), zorder=3)
            if dy:   # start points by shape
                p = f(np.array([a["xy"] for a in dy]))
                ax.scatter(p[:, 0], p[:, 1], s=6, c=[SHAPE_C[a["ptype"]] for a in dy], zorder=5, alpha=0.7, lw=0)
            k = collections.Counter(a["cls"] for a in dy)
            ax.set_title(f"{title} · {LABEL[name]} · {len(metas[name])}장면\n정적 {len(st)} · 동적 {len(dy)} "
                         f"(사람 {k['person']} / 기타 {k['other']} / 차량형 {k['car']})", fontsize=10)
            ax.set_xlim(lo - mid, hi - mid)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.set_xlabel(f"경로 진행 방향 (m, 0 = s {mid:.0f})", fontsize=9)
            if c == 0:
                ax.set_ylabel("왼쪽 + (m)")
    h = [plt.Line2D([], [], color=v, lw=1.5, label={"person": "이동 경로: 사람형", "other": "이동 경로: 동물·자전거·소품 등",
                                                       "car": "이동 경로: 차량형 소품"}[k]) for k, v in CLS_C.items()]
    h += [plt.Line2D([], [], color=c, marker="o", lw=0, label=SHAPE_L[k]) for k, c in SHAPE_C.items()]
    h += [plt.Rectangle((0, 0), 1, 1, color="tab:blue", alpha=0.5, label="정적 물체 (실제 크기·방향)"),
          plt.Rectangle((0, 0), 1, 1, color="0.25", alpha=0.5, label="가림 주차차"),
          plt.Line2D([], [], color="0.55", label="흰 실선 (넘으면 안 됨)"), plt.Line2D([], [], color="0.55", ls="--", label="흰 점선"),
          plt.Line2D([], [], color="goldenrod", label="중앙선"),
          plt.Rectangle((0, 0), 1, 1, color="limegreen", alpha=0.2, label="규정 §3-1 박스 (그림 정합, ±5 m)")]
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=9)
    fig.suptitle("구간 확대 (지도 회전: 경로가 왼→오른쪽). 선 = 동적 물체 이동 경로(시작점 ●)", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(path, dpi=95)
    plt.close(fig)


def bars(ax, cats, series, title, ylabel="비율"):
    w = 0.8 / len(series)
    for i, (name, vals) in enumerate(series.items()):
        tot = sum(vals.values()) or 1
        ax.bar(np.arange(len(cats)) + i * w, [vals.get(c, 0) / tot for c in cats], w, label=LABEL[name], color=COL[name])
    ax.set_xticks(np.arange(len(cats)) + 0.4 - w / 2, [str(c) for c in cats], rotation=25, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)


def stat_fig(path, metas, sv2):
    fig, axs = plt.subplots(3, 4, figsize=(27, 14))
    for ax, (zone, title) in zip(axs[0], (("static", "정적 박스: 장면당 개수"), ("ra_dynamic", "동적 박스: 장면당 개수"),
                                          ("rb_dynamic", "터널 동적: 장면당 개수"))):
        bars(ax, [0, 1, 2, 3, 4], {n: collections.Counter(sum(a["zone"] == zone for a in m["actors"]) for m in ms)
                                   for n, ms in metas.items()}, title)
    axs[0][0].legend(fontsize=8)
    for ax, (zones, pool, title) in zip(axs[1][:2], ((("static", "tunnel_static"), sv2.STATIC_POOL, "정적 에셋별 빈도 (38종)"),
                                                     (("ra_dynamic",), [i for i, _ in sv2.MOVERS], "동적 박스 에셋별 빈도 (70종)"))):
        for name, ms in metas.items():
            c = collections.Counter(a["id"] for a in actors(ms, zones))
            tot = sum(c.values()) or 1
            ax.plot(range(len(pool)), [c.get(i, 0) / tot for i in pool], ".-", ms=4, lw=0.8, color=COL[name], label=LABEL[name])
        ax.axhline(1 / len(pool), color="k", lw=0.8, ls=":", label="균일")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("에셋 (DataID 순)")
        ax.set_ylabel("비율")
    axs[1][0].legend(fontsize=8)
    dyn = {n: [a for a in actors(ms, ("ra_dynamic", "rb_dynamic"))] for n, ms in metas.items()}
    bars(axs[1][2], ["must_stop", "pass_front", "passed", "no_conflict"],
         {n: collections.Counter(a["timing"] for a in v) for n, v in dyn.items()}, "동적 타이밍")
    bars(axs[2][0], ["dynamic", "both", "static", "none"],
         {n: collections.Counter(m["tunnel"] for m in ms) for n, ms in metas.items()}, "터널 구성")
    ring = {n: collections.Counter(next((str(sp["vehicle"]) for sp in m["spawn"] if sp["roundabout"]), "-") for m in ms)
            for n, ms in metas.items()}
    bars(axs[2][1], sorted({k for v in ring.values() for k in v}), ring, "회전교차로 차종 (장면마다; -1 = MORAI가 차마다 고름)")
    ax = axs[2][2]
    for name, ms in metas.items():
        per = [random.uniform(*p) if isinstance(p, list) else p   # v2: each interval uniform in [min, max]
               for m in ms for sp in m["spawn"] if not sp["roundabout"] for p in [sp["period"]]]
        ax.hist(per, bins=np.arange(2.5, 16, 0.5), histtype="step", lw=1.5, color=COL[name], label=LABEL[name], density=True)
    ax.set_title("고주로 spawn 간격 (s)", fontsize=10)
    ax.legend(fontsize=8)
    for ax, (key, bins, title) in zip(axs[:, 3], (("s", np.arange(270, 400, 5), "정적 박스: s 분포 (초록 = 규정 박스)"),
                                                 ("lat", np.arange(-7, 18, 1), "정적 박스: 횡위치 분포 (m, 왼쪽 +)"))):
        for name, ms in metas.items():
            ax.hist([a[key] for a in actors(ms, ("static",))], bins=bins, histtype="step", lw=1.5, color=COL[name],
                    label=LABEL[name], density=True)
        if key == "s":
            ax.axvspan(*BOXES["static"], color="limegreen", alpha=0.1)
        ax.set_title(title, fontsize=10)
    ax = axs[2][3]   # set off early enough: the Ego (route, V_NOMINAL) gets to the crossing after the object's lane entry
    for name, ms in metas.items():
        for timing, ls in (("must_stop", "-"), ("pass_front", "--")):
            v = []
            for a in actors(ms, ("ra_dynamic", "rb_dynamic")):
                if a["timing"] != timing:
                    continue
                corr = 1.2 + (1.05 if a["cls"] == "car" else 0.35)
                due = a["t_in"] if timing == "must_stop" else a["t_in"] + 2 * corr / (a["kph"] / 3.6)
                v.append(math.sqrt(max(a["active"] ** 2 - a["lat"] ** 2, 0.0)) / V_NOM[a["zone"][:2]] - due)
            ax.hist(v, bins=np.arange(-2, 8, 0.25), histtype="step", lw=1.5, ls=ls, color=COL[name], density=True,
                    label=f"{timing}: Ego 도착 - 물체 {'진입' if timing == 'must_stop' else '통과'} (s), 음수 {sum(x < 0 for x in v)}건")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("발동 시점 확인 (0보다 작으면 차가 지나간 뒤 진입)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=95)
    plt.close(fig)


def overview(path, base, route, lines, metas):
    rxy = np.array([p[:2] for p in route.p])
    fig, ax = plt.subplots(figsize=(15, 14))
    draw_map(ax, lines, rxy)
    ms = metas["v2"]
    for name, (lo, hi), c in (("규정 정적 박스", BOXES["static"], "black"), ("규정 동적 박스", BOXES["dynamic"], "darkgreen")):
        q = np.array([route.at(v, 0)[:2] for v in np.linspace(lo, hi, 40)])
        ax.plot(q[:, 0], q[:, 1], color=c, lw=12, alpha=0.25, solid_capstyle="butt", label=f"{name} s {lo:.0f}–{hi:.0f}")
    st = actors(ms, ("static", "tunnel_static", "ra_parked"))
    dy = actors(ms, ("ra_dynamic", "rb_dynamic"))
    ax.scatter([a["xy"][0] for a in st], [a["xy"][1] for a in st], s=4, c="k", marker="s", zorder=5,
               label=f"v2 정적 물체 ({len(ms)}장면, {len(st)}개)")
    ax.scatter([a["xy"][0] for a in dy], [a["xy"][1] for a in dy], s=4, c=[CLS_C[a["cls"]] for a in dy], zorder=5,
               label=f"v2 동적 물체 시작점 ({len(dy)}개, 색=종류)")
    for i, sp in enumerate(base["spawnPointList"]):
        ring = math.hypot(sp["pos"]["x"] + 111.34, sp["pos"]["y"] - 358.26) < 30
        ax.scatter(sp["pos"]["x"], sp["pos"]["y"], s=180, marker="*", color="gold" if ring else "tab:cyan", edgecolor="k", zorder=7)
        if ring or i == 0:
            ax.annotate("회전교차로 spawn: 차마다 무작위 차종(모든 길이 등급), 간격 3–10 s 균일, 10–35 kph" if ring else
                        "고주로 spawn 3곳: 차마다 무작위 차종, 간격 5–15 s 균일, 50–100 kph",
                        (sp["pos"]["x"], sp["pos"]["y"]), xytext=(-10, 12) if ring else (8, 4), textcoords="offset points", fontsize=8)
    for i, v in enumerate(CPS_S):
        x, y, _, _ = route.at(v, 0)
        ax.plot(x, y, "o", ms=4, color="tab:blue", zorder=8)
        ax.annotate(f"CP{i}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8, color="tab:blue")
    ax.set_aspect("equal")
    ax.set_xlim(-180, 150)
    ax.set_ylim(-600, 420)
    ax.set_title("v2 통합 생성기: 장애물·NPC 무작위화 (K-City, clcol 기준; 샘플에 저장된 NPC 9대는 뺌)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)


def summary(metas):
    out = {}
    for name, ms in metas.items():
        frac = lambda c: {str(k): round(v / max(sum(c.values()), 1), 3) for k, v in sorted(c.items(), key=str)}
        dy = actors(ms, ("ra_dynamic", "rb_dynamic"))
        st = actors(ms, ("static",))
        out[name] = dict(
            scenes=len(ms),
            per_scene={z: frac(collections.Counter(sum(a["zone"] == z for a in m["actors"]) for m in ms))
                       for z in ("static", "ra_dynamic", "tunnel_static", "rb_dynamic")},
            static_s=[round(float(q), 1) for q in np.percentile([a["s"] for a in st], [0, 5, 50, 95, 100])],
            static_lat=[round(float(q), 2) for q in np.percentile([a["lat"] for a in st], [0, 5, 50, 95, 100])],
            dyn_A_s=[round(float(q), 1) for q in np.percentile([a["s"] for a in dy if a["zone"] == "ra_dynamic"], [0, 50, 100])],
            tunnel=frac(collections.Counter(m["tunnel"] for m in ms)),
            cls=frac(collections.Counter(a["cls"] for a in dy)), timing=frac(collections.Counter(a["timing"] for a in dy)),
            ptype=frac(collections.Counter(a["ptype"] for a in dy)),
            static_assets=len({a["id"] for a in actors(ms, ("static", "tunnel_static"))}),
            mover_assets=len({a["id"] for a in dy}),
            ring_type=frac(collections.Counter(next((sp["vehicle"] for sp in m["spawn"] if sp["roundabout"]), None) for m in ms)))
    return out


def main():
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    base = json.load(open(sys.argv[1], encoding="utf-8-sig"))
    out = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).parent / "scenario_random")
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        g = gens(tmp)
        route = g["v2"].Route(MAP / "2026_molit_comp_global_path.txt")
        print("recorded scene_meta == old generator (source, preset that matches: count):")
        for k, v in sorted(verify(g, base, route).items()):
            print(f"  {k[0]:7s} {k[1]:10s} {v}")
        metas = {name: [scene(g[gn], base, route, s) for s in seeds] for name, (gn, _, seeds) in CONFIGS.items()}
    st = summary(metas)
    json.dump(st, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(st["v2"], indent=1, ensure_ascii=False))
    lines = lanelet_lines()
    overview(out / "overview.png", base, route, lines, metas)
    zooms(out / "zones.png", route, lines, metas)
    stat_fig(out / "stats.png", metas, g["v2"])


if __name__ == "__main__":
    main()
