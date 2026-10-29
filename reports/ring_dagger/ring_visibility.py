#!/usr/bin/env python3
"""K-City roundabout entry: can WoWmayo see the ring cars it must yield to (from the LEFT, ring runs counter-clockwise)?

Geometry only (no images exist locally at this spot): ego poses on route.json at front-axle s 870 / 880 / 887.5 (wait line),
ring cars on the fitted traffic circle that reach the entry conflict point (route crosses the circle) in T = 1..6 s at
5 / 8 / 10 m/s. Each car is projected into the 4 model cameras at their model-input sizes and into the LiDAR BEV extent.

Run (from anywhere, ~5 s, read-only):  nice -n 10 /usr/bin/python3 reports/ring_dagger/ring_visibility.py
Writes into this folder: fig_a_topdown.png, fig_b_visibility.png, fig_c_projection.png, fig_d_ring_speeds.png (only if the
local Object Info logs exist), visibility.csv.
Sources (all read, none edited):
  cameras   ../MORAI-eval-dagger/spec/0918sensors.json (pos, yaw/pitch, native size, cameraFOV)
            cameraFOV is HORIZONTAL with square pixels, fx = fy = W / (2 tan(FOV/2)): the stack's camera_info convention
            (MORAI-UDP-Bridge-Node bridge_node.cpp, bridge config horizontal_fov_deg); yaw/pitch signs checked below against
            ../MORAI-eval-dagger/src/tf_publisher/.../sensor_mounts.yaml quaternions.
  sizes     train/morai_vla/processor.py CAMERAS (model input W x H per camera)
  BEV       train/morai_vla/lidar_bev.py (RES, XF, YW, H, W, MOUNT, VERT)
  ring      ../MORAI-eval-dagger/src/e2e_driver/tools/dagger_label.py RING_C, RING_R (8,532 Object Info positions)
  route     ../MORAI-eval-dagger/src/e2e_driver/assets/route.json; front axle = base_link + 3.0 m (tools/scorer/thresholds.yaml)
  speeds    ~/morai-eval/an_ring/*/lap*.objects.json.gz (Object Info, 6 eval laps of 10/28), optional
"""
import csv, glob, gzip, importlib.util, json, os, re, ast
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EVAL = REPO.parent / "MORAI-eval-dagger"
SENSORS = EVAL / "spec/0918sensors.json"
MOUNTS = EVAL / "src/tf_publisher/config/static_tf_publisher/parameters/sensor_mounts.yaml"
ROUTE = EVAL / "src/e2e_driver/assets/route.json"
LABEL = EVAL / "src/e2e_driver/tools/dagger_label.py"
OBJ_LOGS = sorted(glob.glob(os.path.expanduser("~/morai-eval/an_ring/*/lap*.objects.json.gz")))

FRONT_S = (870.0, 880.0, 887.5)          # decision points, front axle s (887.5 = wait line)
FRONT_AXLE = 3.0                          # tools/scorer/thresholds.yaml vehicle.front_axle_m
T_ARR = np.arange(1, 7)                   # s until the ring car's centre reaches the conflict point
SPEEDS = (5.0, 8.0, 10.0)                 # m/s (measured ring speeds: fig_d)
CAR = (4.6, 1.9, 1.5)                     # assumed ring car L, W, H (Object Info ring cars: 4.8 x 2.1 x 1.5)
ISLAND_R = 15.0                           # ASSUMED island edge radius (not measured); island height unknown
EGO_BODY = (-0.8, 3.9, 0.95)              # base_link x from, x to, half width (dagger_label EGO_BOX)
HALF = 0.5                                # "seen" = at least this share of the car inside the image / BEV
CAM_COL = ["#eb6834", "#1baf7a", "#6250d6", "#eda100"]                       # Front, Left, Right, Cam7 wedges
T_COL = ["#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b"]   # ordinal blue ramp, T = 1..6
SPEED_MK = {5.0: "o", 8.0: "s", 10.0: "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#c3c2b7"

# ---------------------------------------------------------------- inputs
_spec = importlib.util.spec_from_file_location("lidar_bev", REPO / "train/morai_vla/lidar_bev.py")
bev = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bev)
src = (REPO / "train/morai_vla/processor.py").read_text()
SIZES = ast.literal_eval(re.search(r"CAMERAS = (\(\(.*?\)\))", src, re.S).group(1))       # (name, W, H) x 4
m = re.search(r"RING_C, RING_R.*?= \(([-\d.]+), ([-\d.]+)\), ([\d.]+)", LABEL.read_text())
RING_C, RING_R = np.array([float(m.group(1)), float(m.group(2))]), float(m.group(3))

cams = []
for c, (name, w, h) in zip(json.load(open(SENSORS))["cameraList"], SIZES):
    cc = c["cc"]; Wf, Hf, fov = cc["cameraResWidth"], cc["cameraResHeight"], float(cc["cameraFOV"])
    yaw, pitch = np.radians(float(c["rot"]["yaw"])), np.radians((float(c["rot"]["pitch"]) + 180) % 360 - 180)  # pitch + = down
    f = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)])
    r = np.array([np.sin(yaw), -np.cos(yaw), 0.0]); d = np.cross(f, r)                   # optical x right, y down, z forward
    fx_full = Wf / (2 * np.tan(np.radians(fov) / 2))
    cams.append(dict(name=name, tag=name.split()[1][0].upper() if name != "Front camera" else "F", W=w, H=h, fov=fov,
                     yaw=yaw, pos=np.array([float(c["pos"][k]) for k in "xyz"]), r=r, d=d, f=f,
                     fx=fx_full * w / Wf, fy=fx_full * h / Hf, native=(Wf, Hf), frame=c["cc"]["rosConfig"]["frameID"]))
cams[3]["tag"] = "7"                                    # Front upward camera = Cam7
assert [c["frame"] for c in cams] == ["Camera-1", "Camera-2", "Camera-3", "Camera-7"], "camera order changed"

# self-check: yaw/pitch convention == the stack's optical-frame quaternions (x, y, z, w)
quats = re.findall(r"camera_\w+_optical_frame:\s*translation: \[([^\]]*)\]\s*rotation: \[([^\]]*)\]", MOUNTS.read_text())
for cam, (t, q) in zip(cams, quats):
    x, y, z, w = map(float, q.split(","))
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    assert np.allclose(R, np.c_[cam["r"], cam["d"], cam["f"]], atol=2e-3), cam["name"]
    assert np.allclose(list(map(float, t.split(","))), cam["pos"]), cam["name"]

route = json.load(open(ROUTE)); P, S = np.asarray(route["path"], float), np.asarray(route["s"], float)


def pose(s):
    """route pose (x, y, yaw) at s; yaw from the chord s-1 .. s+1."""
    a, b = [np.array([np.interp(q, S, P[:, 0]), np.interp(q, S, P[:, 1])]) for q in (s - 1.0, s + 1.0)]
    return (*(0.5 * (a + b)), np.arctan2(*(b - a)[::-1]))


ss = np.arange(860.0, 930.0, 0.05)
rr = np.hypot(np.interp(ss, S, P[:, 0]) - RING_C[0], np.interp(ss, S, P[:, 1]) - RING_C[1])
S_CONF = ss[np.argmax(rr <= RING_R)]                                                       # route meets the ring circle
XC, YC, _ = pose(S_CONF); TH_C = np.arctan2(YC - RING_C[1], XC - RING_C[0])


def to_base(x, y, ego):
    dx, dy = np.asarray(x) - ego[0], np.asarray(y) - ego[1]; c, s = np.cos(ego[2]), np.sin(ego[2])
    return c * dx + s * dy, -s * dx + c * dy


def car_world(T, v, n=(13, 6, 5)):
    """ring car centre angle + sample grid (world x, y, z) of its box; CCW travel, so upstream = smaller angle."""
    th = TH_C - v * T / RING_R; h = th + np.pi / 2
    A, B, Z = np.meshgrid(*(np.linspace(-q / 2, q / 2, k) for q, k in zip(CAR[:2], n[:2])), np.linspace(0, CAR[2], n[2]),
                          indexing="ij")
    cx, cy = RING_C + RING_R * np.array([np.cos(th), np.sin(th)])
    return th, cx + A * np.cos(h) - B * np.sin(h), cy + A * np.sin(h) + B * np.cos(h), Z


def project(Pb, cam):
    """base_link points (N, 3) -> u, v at the model input (NaN behind the camera)."""
    q = Pb - cam["pos"]; Z = q @ cam["f"]; ok = Z > 0.1; Z = np.where(ok, Z, np.nan)
    return cam["W"] / 2 + cam["fx"] * (q @ cam["r"]) / Z, cam["H"] / 2 + cam["fy"] * (q @ cam["d"]) / Z


# sanity: a point on the optical axis lands mid-image, one at the FOV edge lands on the image edge
for cam in cams:
    u, v = project(np.array([cam["pos"] + 50 * cam["f"]]), cam); assert abs(u[0] - cam["W"] / 2) < 1e-6 and abs(v[0] - cam["H"] / 2) < 1e-6
    e = np.cos(np.radians(cam["fov"] / 2)) * cam["f"] + np.sin(np.radians(cam["fov"] / 2)) * cam["r"]
    u, _ = project(np.array([cam["pos"] + 50 * e]), cam); assert abs(u[0] - cam["W"]) < 1e-6

# ---------------------------------------------------------------- per (decision point, speed, T)
rows = []
for fs in FRONT_S:
    ego = pose(fs - FRONT_AXLE)
    for v in SPEEDS:
        for T in T_ARR:
            th, X, Y, Z = car_world(T, v)
            xb, yb = to_base(X.ravel(), Y.ravel(), ego); Pb = np.c_[xb, yb, Z.ravel()]
            row = dict(front_s=fs, speed=v, T=int(T), arc_m=v * T, theta_deg=np.degrees(th),
                       dist_m=float(np.hypot(*to_base(*(RING_C + RING_R * np.array([np.cos(th), np.sin(th)])), ego))),
                       bearing_deg=float(np.degrees(np.arctan2(yb.mean(), xb.mean()))))
            for cam in cams:
                u, w = project(Pb, cam); inn = (u >= 0) & (u < cam["W"]) & (w >= 0) & (w < cam["H"])
                row[cam["tag"] + "_frac"] = inn.mean()
                row[cam["tag"] + "_w"] = float(np.ptp(u[inn])) if inn.any() else 0.0
                row[cam["tag"] + "_h"] = float(np.ptp(w[inn])) if inn.any() else 0.0
            inb = (xb > bev.XF - bev.H * bev.RES) & (xb < bev.XF) & (np.abs(yb) < bev.YW)
            row["BEV_frac"] = inb.mean()
            dl = np.hypot(xb - bev.MOUNT[0], yb - bev.MOUNT[1]).min()                        # LiDAR range to nearest face
            zh = bev.MOUNT[2] + dl * np.tan(bev.VERT)
            row["lidar_rings"] = int(((zh > 0.1) & (zh < CAR[2])).sum())                    # VLP16 rings that hit the side
            az = np.degrees(np.arctan2(yb - bev.MOUNT[1], xb - bev.MOUNT[0]))
            row["lidar_pts"] = int(row["lidar_rings"] * np.ptp(az) / 0.2 * row["BEV_frac"])   # 0.2 deg azimuth step (10 Hz spec)
            # sightline Front camera -> car centre over the (assumed) island disc?
            c0 = np.array(ego[:2]) + cams[0]["pos"][0] * np.array([np.cos(ego[2]), np.sin(ego[2])])
            seg = c0 + np.linspace(0, 1, 200)[:, None] * (np.array([X.mean(), Y.mean()]) - c0)
            row["over_island"] = bool((np.hypot(*(seg - RING_C).T) < ISLAND_R).any())
            rows.append(row)

with open(HERE / "visibility.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader()
    wr.writerows({k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()} for r in rows)


H_BINS, H_COL = (6, 11, 21, 41), ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]   # px tall: <=5, 6-10, 11-20, 21-40, >40
H_LAB = ["≤5 px", "6–10 px", "11–20 px", "21–40 px", ">40 px"]


def cell(r):
    """colour bin of the tallest fully seen (>= HALF in image) camera view, its camera, and the cell text."""
    seen = [c for c in cams if r[c["tag"] + "_frac"] >= HALF]
    best = max(seen, key=lambda c: r[c["tag"] + "_h"]) if seen else None
    txt = [f"{c['tag']} {r[c['tag'] + '_w']:.0f}×{r[c['tag'] + '_h']:.0f}" + ("" if r[c["tag"] + "_frac"] >= HALF else "p")
           for c in cams if r[c["tag"] + "_frac"] > 0]
    txt.append(f"BEV {r['lidar_rings']}r" if r["BEV_frac"] >= HALF else ("BEV p" if r["BEV_frac"] > 0 else "no BEV"))
    return (int(np.digitize(round(r[best["tag"] + "_h"]), H_BINS)) if best else -1), "\n".join(txt)


# ---------------------------------------------------------------- fig a: top-down, ego frame of each decision point
fig, axs = plt.subplots(1, 3, figsize=(18, 7.4), constrained_layout=True)
for ax, fs in zip(axs, FRONT_S):
    ego = pose(fs - FRONT_AXLE); P2 = lambda x, y: (-np.asarray(y), np.asarray(x))       # plot: right = +, up = forward
    a = np.linspace(0, 2 * np.pi, 361)
    for rad, ls, lab in ((RING_R, "-", "ring traffic circle r 18.4 m (fit)"), (ISLAND_R, ":", "island edge (assumed r 15 m)")):
        ax.plot(*P2(*to_base(RING_C[0] + rad * np.cos(a), RING_C[1] + rad * np.sin(a), ego)), ls, c=INK2, lw=1.2, label=lab)
    k = (S > fs - 60) & (S < fs + 90)
    ax.plot(*P2(*to_base(P[k, 0], P[k, 1], ego)), "--", c=INK, lw=1, label="route centreline")
    ax.plot(*P2(*to_base(XC, YC, ego)), "x", c="#e34948", ms=11, mew=2.5, label=f"conflict point (s {S_CONF:.1f})")
    for cam, col, ls in zip(cams, CAM_COL, ("-", "-", "-", ":")):
        a0 = cam["yaw"] + np.radians(cam["fov"] / 2) * np.linspace(-1, 1, 60)
        pts = np.r_[[cam["pos"][:2]], cam["pos"][:2] + 80 * np.c_[np.cos(a0), np.sin(a0)]]
        ax.add_patch(Polygon(np.c_[-pts[:, 1], pts[:, 0]], closed=True, fc=col, alpha=0.06 if ls == "-" else 0, lw=0))
        ax.add_patch(Polygon(np.c_[-pts[:, 1], pts[:, 0]], closed=True, fill=False, ec=col, ls=ls, lw=1.6,
                             label=f"{cam['name']} FOV {cam['fov']:.0f}° (model {cam['W']}×{cam['H']})"))
    ax.add_patch(Rectangle((-bev.YW, bev.XF - bev.H * bev.RES), 2 * bev.YW, bev.H * bev.RES, fill=False, ec=INK, ls="--", lw=1.4,
                           label="LiDAR BEV extent (60 fwd, 20 back, ±20)"))
    ax.add_patch(Rectangle((-EGO_BODY[2], EGO_BODY[0]), 2 * EGO_BODY[2], EGO_BODY[1] - EGO_BODY[0], fc=INK, ec=INK))
    for v in SPEEDS:
        for T in T_ARR:
            th = TH_C - v * T / RING_R; rad = RING_R + (v - 8.0) * 0.6                     # offset radially for legibility
            ax.plot(*P2(*to_base(RING_C[0] + rad * np.cos(th), RING_C[1] + rad * np.sin(th), ego)), SPEED_MK[v], ms=8,
                    mfc=T_COL[T - 1], mec="white", mew=1.2)
    ax.set_xlim(-45, 45); ax.set_ylim(-22, 72); ax.set_aspect("equal"); ax.grid(c=GRID, lw=0.4)
    ax.set_title(f"front axle s {fs} " + ("(wait line, standing)" if fs == 887.5 else f"({887.5 - fs:.1f} m before the wait line)"), fontsize=11)
    ax.set_xlabel("m, ego right +"); ax.set_ylabel("m, ego forward")
h1, l1 = axs[0].get_legend_handles_labels()
h2 = [Line2D([], [], ls="", marker="s", ms=9, mfc=c, mec="white", label=f"T = {i + 1} s") for i, c in enumerate(T_COL)]
h3 = [Line2D([], [], ls="", marker=mk, ms=8, mfc="0.6", mec="white", label=f"{v:.0f} m/s") for v, mk in SPEED_MK.items()]
fig.legend(h1 + h2 + h3, l1 + [h.get_label() for h in h2 + h3], loc="center left", ncol=1, fontsize=9, frameon=False,
           bbox_to_anchor=(1.0, 0.5))
fig.suptitle("Ring cars that reach the entry conflict point in T s (ring runs counter-clockwise: they come from the ego's left). "
             "Ego frame; markers offset radially by speed", fontsize=12)
fig.savefig(HERE / "fig_a_topdown.png", dpi=110, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- fig b: visibility table
fig, axs = plt.subplots(1, 3, figsize=(19, 6.6), constrained_layout=True)
for ax, fs in zip(axs, FRONT_S):
    grid = np.zeros((len(SPEEDS), len(T_ARR)), int)
    for r in rows:
        if r["front_s"] != fs:
            continue
        i, j = SPEEDS.index(r["speed"]), r["T"] - 1; grid[i, j], txt = cell(r)
        ax.text(j, i, txt + f"\n{r['dist_m']:.0f} m @ {r['bearing_deg']:+.0f}°", ha="center", va="center", fontsize=7.4,
                color="white" if grid[i, j] >= 2 else INK, linespacing=1.15)
    assert grid.min() >= 0, "a car seen by no camera: add a colour for it"
    ax.imshow(grid, cmap=matplotlib.colors.ListedColormap(H_COL), vmin=-0.5, vmax=4.5, aspect="auto")
    ax.set_xticks(range(len(T_ARR)), [f"T {t} s" for t in T_ARR]); ax.set_yticks(range(len(SPEEDS)), [f"{v:.0f} m/s" for v in SPEEDS])
    ax.set_xticks(np.arange(-0.5, len(T_ARR)), minor=True); ax.set_yticks(np.arange(-0.5, len(SPEEDS)), minor=True)
    ax.grid(which="minor", c="white", lw=2); ax.tick_params(which="minor", length=0)
    ax.set_title(f"front axle s {fs}" + (" (wait line, standing)" if fs == 887.5 else f" ({887.5 - fs:.1f} m before the wait line)"), fontsize=11)
fig.legend([Rectangle((0, 0), 1, 1, fc=c) for c in H_COL], [f"tallest camera view {l}" for l in H_LAB],
           loc="lower center", ncol=5, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.1))
fig.suptitle("Where a left ring car arriving in T s is in the model input. Cell lines: camera W×H px at model input size "
             "(F = Front 448×256, L/R = side 352×256, 7 = Cam7 256×192; p = <50 % of the car in the image), "
             "BEV: VLP16 rings hitting it (r) or p / no BEV; range @ bearing from the rear axle (left +). One ViT token = 32×32 px",
             fontsize=10.5, wrap=True)
fig.savefig(HERE / "fig_b_visibility.png", dpi=110, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- fig c: blank canvases at model resolution, standing ego
ego = pose(FRONT_S[-1] - FRONT_AXLE)
EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
fig, axs = plt.subplots(len(SPEEDS), 5, figsize=(21, 11), gridspec_kw=dict(width_ratios=[448, 352, 352, 256, 120]),
                        constrained_layout=True)
for i, v in enumerate(SPEEDS):
    for cam, ax in zip(cams, axs[i, :4]):
        ax.imshow(np.full((cam["H"], cam["W"], 3), 0.97), extent=(0, cam["W"], cam["H"], 0))
        a0 = cam["yaw"] + np.radians(np.linspace(-89, 89, 200))                                 # horizon: far points at camera height
        ax.plot(*project(cam["pos"] + 1e4 * np.c_[np.cos(a0), np.sin(a0), np.zeros_like(a0)], cam), c=GRID, lw=1)
        a = np.linspace(0, 2 * np.pi, 1441)
        for rad, ls in ((RING_R, "-"), (ISLAND_R, ":")):
            xb, yb = to_base(RING_C[0] + rad * np.cos(a), RING_C[1] + rad * np.sin(a), ego)
            ax.plot(*project(np.c_[xb, yb, np.zeros_like(xb)], cam), ls, c=INK2, lw=1)
        k = (S > FRONT_S[-1] - 10) & (S < FRONT_S[-1] + 60); xb, yb = to_base(P[k, 0], P[k, 1], ego)
        ax.plot(*project(np.c_[xb, yb, np.zeros_like(xb)], cam), "--", c=INK, lw=0.8)
        for T in T_ARR:
            _, X, Y, Z = car_world(T, v, n=(2, 2, 2))
            xb, yb = to_base(X.ravel(), Y.ravel(), ego); u, w = project(np.c_[xb, yb, Z.ravel()], cam)
            for p, q in EDGES:
                ax.plot([u[p], u[q]], [w[p], w[q]], c=T_COL[T - 1], lw=1.8)
            if np.isfinite(u).all() and (0 <= u.mean() < cam["W"]) and (0 <= np.nanmin(w) < cam["H"]):
                ax.text(u.mean(), np.nanmin(w) - 3, f"{T}", ha="center", va="bottom", fontsize=9, color=INK, weight="bold")
        ax.set_xlim(0, cam["W"]); ax.set_ylim(cam["H"], 0); ax.set_xticks([0, cam["W"]]); ax.set_yticks([0, cam["H"]])
        ax.set_title(f"{cam['name']} {cam['W']}×{cam['H']}" + (f" — ring cars at {v:.0f} m/s" if cam is cams[0] else ""), fontsize=10)
    ax = axs[i, 4]                                                                               # BEV canvas, pixel grid
    ax.imshow(np.full((bev.H, bev.W, 3), 0.97), extent=(0, bev.W, bev.H, 0))
    BEVPX = lambda xb, yb: ((bev.YW - np.asarray(yb)) / bev.RES, (bev.XF - np.asarray(xb)) / bev.RES)
    for rad, ls in ((RING_R, "-"), (ISLAND_R, ":")):
        ax.plot(*BEVPX(*to_base(RING_C[0] + rad * np.cos(a), RING_C[1] + rad * np.sin(a), ego)), ls, c=INK2, lw=1)
    ax.add_patch(Polygon(np.c_[BEVPX([EGO_BODY[0], EGO_BODY[1], EGO_BODY[1], EGO_BODY[0]], [EGO_BODY[2], EGO_BODY[2], -EGO_BODY[2], -EGO_BODY[2]])],
                         fc=INK))
    for T in T_ARR:
        _, X, Y, _ = car_world(T, v, n=(2, 2, 2))
        xb, yb = to_base(X[:, :, 0].ravel()[[0, 1, 3, 2]], Y[:, :, 0].ravel()[[0, 1, 3, 2]], ego)
        ax.add_patch(Polygon(np.c_[BEVPX(xb, yb)], fc=T_COL[T - 1], ec=INK, lw=0.5))
    ax.set_xlim(0, bev.W); ax.set_ylim(bev.H, 0); ax.set_xticks([0, bev.W]); ax.set_yticks([0, bev.H])
    ax.set_title(f"LiDAR BEV {bev.W}×{bev.H}", fontsize=10)
fig.suptitle(f"Predicted ring-car boxes on blank canvases at model input size, ego standing at the wait line (front s {FRONT_S[-1]}). "
             "Number = T s to the conflict point; solid = ring circle, dotted = assumed island edge, dashed = route, grey = horizon",
             fontsize=11)
fig.savefig(HERE / "fig_c_projection.png", dpi=100, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- fig d: measured ring speeds upstream of the entry
if OBJ_LOGS:
    tracks = {}
    for f in OBJ_LOGS:
        for t, objs in json.load(gzip.open(f, "rt"))["objects"]:
            for o in objs:
                if o[1] == 1 and abs(np.hypot(o[2] - RING_C[0], o[3] - RING_C[1]) - RING_R) < 1.0:
                    tracks.setdefault((f, o[0]), []).append((t, o[2], o[3]))
    sp, ccw = [], []
    for tr in tracks.values():
        a = np.array(tr); dt = np.diff(a[:, 0]); th = np.arctan2(a[:, 2] - RING_C[1], a[:, 1] - RING_C[0])
        v = np.hypot(np.diff(a[:, 1]), np.diff(a[:, 2])) / np.where(dt > 0, dt, np.nan)
        up = (th[1:] > -np.pi) & (th[1:] < TH_C)                                                # arc upstream of the entry
        k = up & (dt > 0.03) & (dt < 0.6) & (v < 20)
        sp += list(v[k]); ccw += list((np.angle(np.exp(1j * np.diff(th))) > 0)[k & (v > 0.5)])            # + = counter-clockwise
    sp = np.array(sp); mv = sp[sp > 0.5]
    q = np.percentile(mv, [10, 50, 90])
    fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    ax.hist(mv, bins=np.arange(0, 12.5, 0.5), color="#2a78d6", ec="white", lw=1)
    for v in SPEEDS:
        ax.axvline(v, c=INK2, ls="--", lw=1); ax.text(v, ax.get_ylim()[1] * 0.95, f" {v:.0f}", color=INK2, fontsize=9, va="top")
    ax.set_xlabel("ring car speed, m/s"); ax.set_ylabel("Object Info samples")
    ax.set_title(f"Measured ring speeds upstream of the entry (Object Info, {len(OBJ_LOGS)} eval laps 10/28, {len(tracks)} tracks)\n"
                 f"moving p10/p50/p90 = {q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f} m/s, stopped {np.mean(sp <= 0.5):.0%}, "
                 f"counter-clockwise {np.mean(ccw):.1%} of moving samples", fontsize=9)
    ax.grid(axis="y", c=GRID, lw=0.4)
    fig.savefig(HERE / "fig_d_ring_speeds.png", dpi=110); plt.close(fig)
    print(f"ring speeds: n {len(sp)}, moving p10/50/90 {q.round(2)}, max {mv.max():.1f}, stopped {np.mean(sp <= 0.5):.2f}, ccw {np.mean(ccw):.4f}")

print(f"ring C {RING_C} R {RING_R}; conflict s {S_CONF:.2f} theta {np.degrees(TH_C):.1f} deg; "
      f"cameras {[(c['tag'], c['W'], c['H'], round(c['fx'], 1), round(c['fy'], 1)) for c in cams]}; BEV x {bev.XF - bev.H * bev.RES}..{bev.XF} |y|<{bev.YW}")
for fs in FRONT_S:
    print(f"front s {fs}: ego {np.round(pose(fs - FRONT_AXLE), 2)}")
    for r in rows:
        if r["front_s"] == fs:
            print(f"  v {r['speed']:4.0f} T {r['T']} arc {r['arc_m']:4.0f} m  {r['dist_m']:5.1f} m @ {r['bearing_deg']:+5.0f}  "
                  + "  ".join(f"{c['tag']} {r[c['tag'] + '_frac']:.2f} {r[c['tag'] + '_w']:3.0f}x{r[c['tag'] + '_h']:2.0f}" for c in cams)
                  + f"  BEV {r['BEV_frac']:.2f} rings {r['lidar_rings']} ~pts {r['lidar_pts']}  island {r['over_island']}")
