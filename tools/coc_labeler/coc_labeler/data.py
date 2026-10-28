"""Episode loading on the restored sim clock (tau), route geometry, scene actors and environment."""
import bisect, json, os, sys, xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np, pandas as pd, yaml

PKG = Path(__file__).resolve().parent.parent            # tools/coc_labeler
TOOLS = PKG.parent
sys.path.insert(0, str(TOOLS / "scorer"))
from scorer import KMap, load_config                    # noqa: E402  (shared map + judged stop lines)

MAC = Path(os.path.expanduser("~/.claude-morai-mac"))
LABELS = MAC / "labels-cache"
MANIFESTS = MAC / "manifests"
SCENE_LOGS = [MAC / "episode-scenes.log", MAC / "episode-scenes.wsl.log"]
WORKSPACES = TOOLS.parent.parent
OSM = WORKSPACES / "MORAI-Autonomous-Driving-System/src/map/kcity_lanelet2.osm"
MGEO = WORKSPACES / "lanelet_editor/outputs/lanelet-map-repair/map/source"
ROUTE = yaml.safe_load(open(PKG / "route.yaml"))
ROUTE_BY_ID = {f["id"]: f for f in ROUTE["features"]}
CAM_FWD = 2.0          # m, base_link (rear axle) -> front camera
BUMPER = 3.9           # m, base_link -> front bumper
VEHICLE_PROPS = {40100008, 40100009, 40100010, 40100011}
PED_NAMES = {50100017: "person walking a bicycle"}


@lru_cache(None)
def kmap():
    return KMap.from_files(load_config())


@lru_cache(None)
def path_xy():
    P, s = kmap().path, kmap().s
    keep = np.r_[True, np.hypot(*np.diff(P, axis=0).T) > 1e-6]
    return P[keep], s[keep]


def route_point(s, lat=0.0):
    """(x, y, heading) on the global path at progress s, offset lat (left +)."""
    P, S = path_xy(); i = int(np.clip(np.searchsorted(S, s) - 1, 0, len(P) - 2))
    u = (s - S[i]) / max(S[i + 1] - S[i], 1e-6); d = P[i + 1] - P[i]; h = np.arctan2(d[1], d[0])
    p = P[i] + u * d + lat * np.array([-np.sin(h), np.cos(h)])
    return p[0], p[1], h


def route_heading(s):
    P, S = path_xy(); i = int(np.clip(np.searchsorted(S, s) - 1, 0, len(P) - 2)); j = min(i + 3, len(P) - 1)
    h0 = np.arctan2(*(P[min(i + 1, len(P) - 1)] - P[i])[::-1]); h1 = np.arctan2(*(P[j] - P[max(j - 1, 0)])[::-1])
    return float(np.unwrap([h0, h1])[0])


@lru_cache(None)
def lane_lines():
    """every painted lane line segment (dashed included) from the lanelet osm."""
    root = ET.parse(OSM).getroot()
    tg = lambda e: {t.get("k"): t.get("v") for t in e.findall("tag")}
    nodes = {n.get("id"): (float(tg(n)["local_x"]), float(tg(n)["local_y"])) for n in root.iter("node")}
    A, B = [], []
    for w in root.iter("way"):
        if tg(w).get("type") in ("line_thin", "line_thick"):
            p = np.array([nodes[r.get("ref")] for r in w.findall("nd")])
            A += list(p[:-1]); B += list(p[1:])
    return np.array(A), np.array(B)


def crossings(P):
    """[(i, side)] where P[i]->P[i+1] crosses a lane line; side = way the ego moved across the line (taper-safe)."""
    A, B = lane_lines()
    lo, hi = P.min(0) - 5, P.max(0) + 5
    m = ((np.minimum(A, B) <= hi) & (np.maximum(A, B) >= lo)).all(1)
    a, b = A[m], B[m]; out = []
    cr = lambda o, p, q: (p[..., 0] - o[..., 0]) * (q[..., 1] - o[..., 1]) - (p[..., 1] - o[..., 1]) * (q[..., 0] - o[..., 0])
    for i in range(len(P) - 1):
        p, q = P[i], P[i + 1]
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))) or np.hypot(*(q - p)) < 1e-3: continue
        hit = (cr(a, b, p[None]) * cr(a, b, q[None]) < 0) & (cr(p[None], q[None], a) * cr(p[None], q[None], b) < 0)
        if hit.any():
            k = int(np.flatnonzero(hit)[0]); la, lb = a[k], b[k]
            if np.dot(lb - la, q - p) < 0: la, lb = lb, la
            out.append((i, "left" if cr(la[None], lb[None], q[None])[0] > 0 else "right"))
    return out


def feature_at(s, types=None):
    for f in ROUTE["features"]:
        if f["s"][0] <= s <= f["s"][1] and (types is None or f["type"] in types): return f
    return None


@lru_cache(None)
def traffic_lights():
    return {t["idx"]: np.asarray(t["point"][:2], float) for t in json.load(open(MGEO / "traffic_light_set.json"))}


# ---------------------------------------------------------------- environment / scenes
@lru_cache(None)
def env_rows(w):
    rows = []
    for f in (MAC / f"env-{w}.log", LABELS / f"env-{w}.wsl.log"):
        if f.exists():
            for line in open(f):
                t, wx, hr = line.split(); rows.append((float(t), f"{wx[0]}{hr}"))
    rows.sort()
    return [r[0] for r in rows], [r[1] for r in rows]


def env_at(w, t_sec):
    ts, vals = env_rows(w); i = bisect.bisect_right(ts, t_sec) - 1
    return vals[i] if i >= 0 else "?"


@lru_cache(None)
def scene_index():
    out = {}
    for f in SCENE_LOGS:
        if f.exists():
            for line in open(f):
                if line.strip(): ep, sc = line.split()[:2]; out[ep] = sc.split("/")[0]
    return out


@lru_cache(None)
def manifest(batch):
    f = MANIFESTS / f"{batch}-manifest.json"
    return json.load(open(f)) if f.exists() else {}


def scene_of(ep):
    sc = scene_index().get(ep)
    return (sc, manifest(sc.split("_")[0]).get(sc)) if sc else (None, None)


# ---------------------------------------------------------------- episode on the sim clock
@dataclass
class Episode:
    name: str
    worker: str
    G: pd.DataFrame                      # 10 Hz tau grid
    scene_id: str = None
    scene: dict = None
    labels_path: Path = None
    extra: dict = field(default_factory=dict)

    def env(self, i):
        return env_at(self.worker, self.G.bag_ns[i] / 1e9)

    def fog(self, i):
        return self.env(i).startswith("F")


def labels_file(ep):
    for d in ("mac-win-1", "mac-win-2", "wsl-win-1", "mac-bc-1"):   # mac-bc-1: R2 bc-250 instance (10/27)
        f = LABELS / d / f"{ep}.labels.parquet"
        if f.exists(): return f
    raise FileNotFoundError(ep)


def load_episode(ep) -> Episode:
    f = labels_file(ep); d = pd.read_parquet(f)
    p = d[["x", "y"]].to_numpy(float); v = d.speed_kph.to_numpy(float) / 3.6; t = d.t.to_numpy(float)
    ok = np.isfinite(p).all(1)
    # sim-time rate rho = |dpos| / (v dt) over 5 s windows (bags carry wall-clock stamps only; RTF 0.7-0.95)
    step = np.r_[np.hypot(*np.diff(p, axis=0).T), 0]; step[~np.isfinite(step)] = 0
    vdt = np.nan_to_num(v) * np.r_[np.diff(t), 0.1]
    k = np.ones(51); num, den = np.convolve(step, k, "same"), np.convolve(vdt, k, "same")
    rho = np.clip(pd.Series(np.where(den > 2.0, num / np.maximum(den, 1e-9), np.nan)).interpolate(limit_direction="both").fillna(0.8).to_numpy(), 0.5, 1.05)
    tau = np.r_[0, np.cumsum(rho[:-1] * np.diff(t))]
    g = np.arange(0, tau[-1], 0.1)
    interp = lambda col: np.interp(g, tau[ok], col[ok])
    fin = np.isfinite(v) & ok   # the labels hold 0 kph where the GT stream had a gap (R2: 3 % of frames): bridge it like x, y
    G = pd.DataFrame({"tau": g, "t": np.interp(g, tau, t), "x": interp(p[:, 0]), "y": interp(p[:, 1]),
                      "yaw": interp(np.unwrap(np.nan_to_num(d.yaw.to_numpy(float)))),
                      "v": np.interp(g, tau[fin], v[fin]) if fin.any() else np.zeros(len(g)),
                      "bag_ns": np.interp(g, tau, d.bag_time_ns.to_numpy(float)).astype(np.int64),
                      "path_s": np.interp(g, tau[np.isfinite(d.path_s)], d.path_s.to_numpy(float)[np.isfinite(d.path_s)])})
    near = np.clip(np.searchsorted(tau, g), 0, len(d) - 1)
    for col in ("route_section", "signal", "gps_blackout", "clean"):
        G[col] = d[col].to_numpy()[near]
    G["a"] = np.convolve(np.gradient(G.v, 0.1), np.ones(5) / 5, "same")
    G["yaw_rate"] = np.convolve(np.gradient(G.yaw, 0.1), np.ones(5) / 5, "same")
    dx, dy = np.gradient(G.x, 0.1), np.gradient(G.y, 0.1)
    G["v_lat"] = -(-np.sin(G.yaw) * dx + np.cos(G.yaw) * dy)            # + = rightward
    cx, cy = G.x + 1.4 * np.cos(G.yaw), G.y + 1.4 * np.sin(G.yaw)
    G["cx"], G["cy"] = cx, cy
    G["d"] = _lateral_offset(cx.to_numpy(), cy.to_numpy(), G.path_s.to_numpy())
    G["cam_s"] = G.path_s + CAM_FWD
    sc_id, sc = scene_of(ep)
    return Episode(ep, ep.split("-")[1], G, sc_id, sc, f)


def _lateral_offset(x, y, s_hint):
    """signed distance of the ego centre to the global path (left +), searched near the progress hint."""
    P, S = path_xy(); out = np.full(len(x), np.nan)
    for k, (px, py, sh) in enumerate(zip(x, y, s_hint)):
        lo, hi = np.searchsorted(S, sh - 30), np.searchsorted(S, sh + 30)
        lo, hi = max(lo - 1, 0), min(max(hi, lo + 2), len(P) - 1)
        a, b = P[lo:hi], P[lo + 1:hi + 1]; dd = b - a; pt = np.array([px, py])
        u = np.clip(((pt - a) * dd).sum(1) / (dd ** 2).sum(1), 0, 1); f = a + u[:, None] * dd
        j = int(np.argmin(np.hypot(*(pt - f).T)))
        out[k] = np.sign(dd[j, 0] * (pt - f[j])[1] - dd[j, 1] * (pt - f[j])[0]) * np.hypot(*(pt - f[j]))
    return out


def cell_of(section):
    return ROUTE["cells"].get(int(section), "C?")
