"""Offline competition scorer v0 (PLAN.md §4.2, rules v1.2 §3-2).

One episode bag (or its split chunks) -> episode score (time + penalties) and 10 Hz frame labels.
Filtering is labelling: nothing is deleted, `clean` and `excluded_by` say which frames to skip.

    uv run --python 3.12 --with-requirements tools/scorer/requirements.txt \
        python tools/scorer/scorer.py EPISODE.bag [EPISODE_1.bag ...] [-o labels.parquet]
"""
import argparse
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
CELL_M = 5.0  # lane-line grid cell


def load_config(path=HERE / "thresholds.yaml"):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    cfg["_path"], cfg["_hash"] = str(path), hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return cfg


def _point_seg_dist(p, a, b):
    """Distance from point p to each segment a[i]-b[i]."""
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    return np.hypot(*(a + t[:, None] * ab - p).T)


def _cross_params(p, q, a, b):
    """Fraction along each p[i]->q[i] where it crosses segment a-b, NaN where it does not."""
    d1, d2, w = q - p, b - a, a - p
    den = d1[:, 0] * d2[1] - d1[:, 1] * d2[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (w[:, 0] * d2[1] - w[:, 1] * d2[0]) / den
        v = (w[:, 0] * d1[:, 1] - w[:, 1] * d1[:, 0]) / den
    return np.where((u >= 0) & (u <= 1) & (v >= 0) & (v <= 1), u, np.nan)


def _signal_state(status, movement):
    """MORAI trafficLightStatus bits: red 1, yellow 4, green 16, green-left 32; -1 = default/unknown."""
    if status < 0:
        return "unknown"
    return "green" if status & (32 if movement == "left" else 16) else "yellow" if status & 4 else "red"


def _signal_ahead(t, front_s, signals, stops, max_age):
    """Per frame, the GT state for the next judged stop line (until 5 m past it) from its freshest mapped light.
    stops: [(path s, {movement, lights})]; signals: t, light, status."""
    out, start = np.full(len(t), "unknown", dtype=object), -np.inf
    for s_stop, spec in sorted(stops, key=lambda item: item[0]):
        sub = signals[signals.light.isin(spec["lights"])].sort_values("t")
        zone = (front_s > start) & (front_s <= s_stop + 5.0)
        start = s_stop + 5.0
        if not len(sub) or not zone.any():
            continue
        ts, status = sub.t.to_numpy(float), sub.status.to_numpy(int)
        k = np.maximum(np.searchsorted(ts, t[zone], side="right") - 1, 0)
        fresh = (ts[k] <= t[zone]) & (t[zone] - ts[k] <= max_age)
        out[zone] = [_signal_state(v, spec["movement"]) if ok else "unknown" for v, ok in zip(status[k], fresh)]
    return out


def _runs(mask):
    """Inclusive (start, end) index pairs of True runs."""
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def _grid(boxes, pad):
    """Cell -> indices of the (lo, hi) boxes overlapping it (expanded by pad)."""
    grid = defaultdict(list)
    for i, (lo, hi) in enumerate(boxes):
        (x0, y0), (x1, y1) = (np.asarray(lo) - pad) // CELL_M, (np.asarray(hi) + pad) // CELL_M
        for gx in range(int(x0), int(x1) + 1):
            for gy in range(int(y0), int(y1) + 1):
                grid[gx, gy].append(i)
    return {k: np.array(v) for k, v in grid.items()}


def _near(pts, a, b, reach, grid):
    """True where a point is within reach[i] of some segment a[i]-b[i] listed in its grid cell."""
    hit = np.zeros(len(pts), bool)
    for k, p in enumerate(pts):
        if np.isfinite(p).all():
            idx = grid.get((int(p[0] // CELL_M), int(p[1] // CELL_M)))
            if idx is not None:
                hit[k] = (_point_seg_dist(p, a[idx], b[idx]) <= reach[idx]).any()
    return hit


def _inside(p, poly):
    """Even-odd point-in-polygon."""
    a, b = poly, np.roll(poly, -1, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = a[:, 0] + (p[1] - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
    return bool(np.count_nonzero(((a[:, 1] > p[1]) != (b[:, 1] > p[1])) & (p[0] < xs)) % 2)


def _window(t, spans, before, after):
    mask = np.zeros(len(t), bool)
    for t0, t1 in spans:
        mask |= (t >= t0 - before - 1e-9) & (t <= t1 + after + 1e-9)
    return mask


class KMap:
    """Route, checkpoints, judged lane lines and stop lines in map coordinates."""

    def __init__(self, path, checkpoints, lane_lines, stop_lines, speed_exempt_s=(np.inf, np.inf), lanelets=()):
        self.path, self.checkpoints = np.asarray(path, float), np.asarray(checkpoints, float)
        self.s = np.r_[0, np.cumsum(np.hypot(*np.diff(self.path, axis=0).T))]
        self.heading = np.arctan2(*np.diff(self.path, axis=0)[:, ::-1].T)
        self.speed_exempt_s = speed_exempt_s
        self.cp_s = np.array([self.project(c) for c in self.checkpoints])
        segs = [(a, b, hw) for line, hw in lane_lines for a, b in zip(line[:-1], line[1:])]
        self.seg_a = np.array([a for a, _, _ in segs]).reshape(-1, 2)
        self.seg_b = np.array([b for _, b, _ in segs]).reshape(-1, 2)
        self.seg_hw = np.array([hw for _, _, hw in segs])
        self.grid = _grid(zip(np.minimum(self.seg_a, self.seg_b), np.maximum(self.seg_a, self.seg_b)), 1.0)
        self.path_grid = _grid(zip(np.minimum(self.path[:-1], self.path[1:]), np.maximum(self.path[:-1], self.path[1:])), 5.0)
        self.lanelets = [np.asarray(poly, float) for poly in lanelets]  # drivable area polygons
        self.lanelet_grid = _grid(((poly.min(0), poly.max(0)) for poly in self.lanelets), 0.0)
        self.stop_lines = {k: np.asarray(v, float) for k, v in stop_lines.items()}
        self.route_stops = {}  # stop line id -> path s where the route crosses it
        for sid, line in self.stop_lines.items():
            for a, b in zip(line[:-1], line[1:]):
                u = _cross_params(self.path[:-1], self.path[1:], a, b)
                hit = np.flatnonzero(~np.isnan(u))
                if len(hit):
                    self.route_stops[sid] = float(self.s[hit[0]] + u[hit[0]] * (self.s[hit[0] + 1] - self.s[hit[0]]))
                    break

    @classmethod
    def from_files(cls, cfg):
        m = cfg["map"]
        d = (Path(cfg["_path"]).parent / m["dir"]).resolve()
        root = ET.parse(d / m["osm"]).getroot()

        def tags(e):
            return {t.get("k"): t.get("v") for t in e.findall("tag")}

        nodes = {n.get("id"): (float(tags(n)["local_x"]), float(tags(n)["local_y"])) for n in root.iter("node")}
        ways = {w.get("id"): (tags(w), np.array([nodes[r.get("ref")] for r in w.findall("nd")])) for w in root.iter("way")}
        lane = cfg["lane"]
        lines, stops = [], {}
        for wid, (t, line) in ways.items():
            if t.get("type") == "stop_line":
                stops[wid] = line
            elif t.get("type") in lane["line_widths_m"] and t.get("subtype") in lane["subtypes"]:
                lines.append((line, lane["line_widths_m"][t["type"]] / 2))
        lanelets = []
        for rel in root.iter("relation"):
            if tags(rel).get("type") == "lanelet":
                bound = {mb.get("role"): ways[mb.get("ref")][1] for mb in rel.findall("member")
                         if mb.get("role") in ("left", "right")}
                left, right = bound["left"], bound["right"]
                if np.hypot(*(left[0] - right[0])) > np.hypot(*(left[0] - right[-1])):
                    right = right[::-1]
                lanelets.append(np.vstack([left, right[::-1]]))
        path = np.loadtxt(d / m["path"])[:, :2]
        cps = np.loadtxt(d / m["checkpoints"], delimiter=",", skiprows=1, usecols=(0, 1))
        kmap = cls(path, cps, lines, stops, lanelets=lanelets)
        first, last = cfg["speed"]["exempt_links"]
        ends = {first: [], last: []}
        for rel in root.iter("relation"):
            link = tags(rel).get("mgeo:link_id")
            if link in ends:
                bounds = [ways[mb.get("ref")][1] for mb in rel.findall("member") if mb.get("role") in ("left", "right")]
                ends[link] += [kmap.project((bounds[0][i] + bounds[1][i]) / 2) for i in (0, -1)]
        if not ends[first] or not ends[last]:
            raise ValueError(f"speed exemption links {first}/{last} not found in {m['osm']}")
        kmap.speed_exempt_s = (min(ends[first]), max(ends[last]))
        return kmap

    def project(self, xy):
        return float(self.s[np.argmin(np.hypot(*(self.path - xy).T))])

    def section_at(self, s):
        """section_annotator numbering: checkpoint row index + 1 of the last checkpoint passed."""
        return np.where(np.isnan(s), 0, np.searchsorted(self.cp_s, np.nan_to_num(s), side="right"))

    def progress(self, xy, yaw):
        """Path s per frame, tracked in a window so the closed loop's start/end overlap stays unambiguous."""
        out, i = np.full(len(xy), np.nan), None
        for k, (p, h) in enumerate(zip(xy, yaw)):
            if not np.isfinite(p).all():
                continue
            if i is not None:
                lo = max(i - 40, 0)
                win = np.hypot(*(self.path[lo:i + 400] - p).T)
                i = lo + int(win.argmin()) if win.min() < 5.0 else None
            if i is None:
                d = np.hypot(*(self.path - p).T)
                if np.isfinite(h):
                    d[:-1][np.cos(self.heading - h) < 0] = np.inf
                i = int(np.flatnonzero(d <= d.min() + 0.5)[0])  # lowest s among near ties (loop start)
            out[k] = self.s[i]
        return out

    def on_road(self, pts):
        """True where a point lies in some lanelet (always True without lanelets: nothing to judge)."""
        hit = np.ones(len(pts), bool) if not self.lanelets else np.zeros(len(pts), bool)
        for k, p in enumerate(pts):
            if self.lanelets and np.isfinite(p).all():
                idx = self.lanelet_grid.get((int(p[0] // CELL_M), int(p[1] // CELL_M)), ())
                hit[k] = any(_inside(p, self.lanelets[i]) for i in idx)
        return hit

    def near_line(self, pts, margin):
        """True where a point is within (painted half width + margin) of a judged lane line."""
        return _near(pts, self.seg_a, self.seg_b, self.seg_hw + margin, self.grid)

    def near_path(self, pts, radius):
        """True where a point is within radius (<= 5 m) of the global path."""
        return _near(pts, self.path[:-1], self.path[1:], np.full(len(self.path) - 1, radius), self.path_grid)


COMPETITION_EVENTS = ("collision", "signal_violation", "speed", "lane", "drive_inoperable")
TARGET_BREAKERS = COMPETITION_EVENTS + ("signal_unverified",)  # unknown light at a judged stop line may be a violation


def score(ep, kmap, cfg):
    """ep (see bag.read_episode): frames (10 Hz Ego Vehicle Status: t, x, y, yaw, speed_mps, section, signal,
    gps_blackout and optional noise, student), contacts (t, key, msg = CollisionData message index), signals (t, light,
    status), controls (t, steer, accel, brake), unavailable, sources. Returns (labels DataFrame, summary dict)."""
    f = ep["frames"].reset_index(drop=True)
    contacts, signals, controls = (ep.get(k) for k in ("contacts", "signals", "controls"))
    n, dt, veh = len(f), 1.0 / cfg["frame_hz"], cfg["vehicle"]
    t = f.t.to_numpy(float)
    x, y, yaw = (f[c].to_numpy(float) for c in ("x", "y", "yaw"))
    v = np.nan_to_num(f.speed_mps.to_numpy(float))
    signal = f.signal.astype(str).to_numpy()
    blackout = f.gps_blackout.to_numpy(bool) if "gps_blackout" in f else np.zeros(n, bool)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(yaw)
    c, s = np.cos(yaw), np.sin(yaw)

    def body(lx, ly):
        return np.c_[x + c * lx - s * ly, y + s * lx + c * ly]

    path_s = kmap.progress(np.c_[x, y], yaw)
    route_section = kmap.section_at(path_s)
    front = body(veh["front_axle_m"], 0.0)
    sig = cfg["signal"]
    judged = {sid: kmap.route_stops[sid] for sid in sig["stop_lines"] if sid in kmap.route_stops}
    if signals is not None and len(signals):
        signal = _signal_ahead(t, path_s + veh["front_axle_m"], signals,
                               [(st, sig["stop_lines"][sid]) for sid, st in judged.items()], sig["max_age_s"])
    events, active, excluded = [], defaultdict(lambda: np.zeros(n, bool)), defaultdict(lambda: np.zeros(n, bool))
    competition_failed = []

    def add(kind, t0, t1, penalty=0.0, **detail):
        events.append(dict(type=kind, t_start=round(float(t0), 3), t_end=round(float(t1), 3), penalty_s=float(penalty), **detail))

    # Time: from the first GT movement to the global path's last waypoint, where the judge stops automatically.
    tm = cfg["time"]
    first = np.flatnonzero(valid)
    moved = first[np.hypot(x[first] - x[first[0]], y[first] - y[first[0]]) > tm["move_m"]] if len(first) else first
    t_start = float(t[moved[0]]) if len(moved) else None
    done = np.flatnonzero((path_s >= kmap.s[-1] - tm["finish_m"]) & (t >= (t_start if t_start is not None else np.inf)))
    finished = bool(len(done))
    t_end = float(t[done[0]] if finished else t[-1]) if n else 0.0
    duration = t_end - t_start if t_start is not None else 0.0
    if duration > tm["limit_s"]:
        competition_failed.append("time_limit")

    # Collision: an object's first contact, and again whenever it was missing from at least one CollisionData message.
    col = cfg["collision"]
    if len(contacts) and "x" in contacts:  # CollisionData also lists collisions between other objects (NPC pile-ups)
        k = np.clip(np.searchsorted(t, contacts.t.to_numpy(float)), 0, n - 1)
        near = np.hypot(contacts.x.to_numpy(float) - x[k], contacts.y.to_numpy(float) - y[k]) <= col["max_dist_m"]
        contacts = contacts[near]
    for key, g in contacts.sort_values("msg").groupby("key"):
        msg, ts = g.msg.to_numpy(int), g.t.to_numpy(float)
        new = np.r_[True, np.diff(msg) > 1]
        for t0, t1 in zip(ts[new], ts[np.r_[new[1:], True]]):
            add("collision", t0, t1, col["penalty_s"], object=str(key))
            active["collision"] |= _window(t, [(t0, t1)], dt / 2, dt / 2)
            excluded["collision"] |= _window(t, [(t0, t1)], *col["exclude_s"])

    # Signal: GT state at the frame where the front axle crosses a judged stop line; once per stop line.
    seg_ok = valid[:-1] & valid[1:]
    for sid, line in ((k, kmap.stop_lines[k]) for k in judged):
        u = np.full(max(n - 1, 0), np.nan)
        for a, b in zip(line[:-1], line[1:]):
            u = np.fmin(u, _cross_params(front[:-1], front[1:], a, b))
        hit = np.flatnonzero(~np.isnan(u) & seg_ok)
        # Every crossing is judged so later laps keep their target masks honest; only the first one is scored
        # (the judge counts each intersection once, and later laps fall after the finish anyway).
        for j, h in enumerate(hit[np.r_[True, np.diff(hit) > 1]] if len(hit) else hit):
            i = h + int(u[h] >= 0.5)
            state = signal[i]
            if state in sig["violating_states"] or state == "unknown":
                kind = "signal_unverified" if state == "unknown" else "signal_violation"
                penalty = sig["penalty_s"] if kind == "signal_violation" and j == 0 else 0.0
                add(kind, t[i], t[i], penalty, stop_line=sid, state=state, movement=sig["stop_lines"][sid]["movement"],
                    repeat=j > 0)
                active[kind][i] = True
                excluded[kind] |= _window(t, [(t[i], t[i])], *sig["exclude_s"])

    # Speed (Ego Vehicle Status): > limit outside the exempt route span; +15 s at once and per further 3 s over it.
    spd = cfg["speed"]
    lo, hi = kmap.speed_exempt_s
    # Frames without a GT pose (dropped Ego status packets) keep the last known route position: on the highway they
    # were judged outside the exemption (run1020k-w2-001, 87-91 km/h at path_s NaN).
    s_known = pd.Series(path_s).ffill().to_numpy()
    over = (np.abs(v) * 3.6 > spd["limit_kph"]) & np.isfinite(s_known) & ~((s_known >= lo) & (s_known <= hi))
    period = round(spd["repeat_s"] / dt)
    for i0, i1 in _runs(over):
        add("speed", t[i0], t[i1], (1 + (i1 - i0 + 1) // period) * spd["penalty_s"],
            max_kph=round(float(np.abs(v[i0:i1 + 1]).max() * 3.6), 1))
        excluded["speed"] |= _window(t, [(t[i0], t[i1])], spd["exclude_before_s"], 0.0)
    active["speed"] |= over

    # Wheels: fixed base_link offsets on the Ego Vehicle Status pose (the judge reads Ego Detail Status).
    wheels = [body(*xy) for xy in veh["wheels_base_link_m"].values()]

    # Lane: a wheel touching white/yellow solid paint outside the GPS blackout; +5 s per 3 s continuous contact.
    lane = cfg["lane"]
    touch = np.any([kmap.near_line(w, veh["tire_half_width_m"]) for w in wheels], axis=0) & valid & ~blackout
    period = round(lane["period_s"] / dt)
    for i0, i1 in _runs(touch):
        penalty = (i1 - i0 + 1) // period * lane["penalty_s"]
        add("lane", t[i0], t[i1], penalty)
        excluded["lane"] |= _window(t, [(t[i0], t[i1])], lane["exclude_before_s"], 0.0)
        # Only penalized contact breaks targets: the judge lets a touch under one period go, and quick centre-line
        # crossings around static obstacles are the avoidance data C2 is short of (run1021a: 25% of frames lost).
        active["lane"][i0:i1 + 1] = penalty > 0

    # Drive-inoperable candidate: all four wheels off the drivable area (lanelets, plus a corridor around the route
    # where the lanelet map has gaps) outside the GPS blackout; inside it rejoining the route is allowed.
    ino = cfg["inoperable"]
    off = valid & ~blackout
    for w in wheels:
        off[off] = ~kmap.on_road(w[off])
        off[off] = ~kmap.near_path(w[off], ino["route_corridor_m"])
    for i0, i1 in _runs(off):
        if (i1 - i0 + 1) * dt >= ino["min_s"]:
            add("drive_inoperable", t[i0], t[i1])
            active["drive_inoperable"][i0:i1 + 1] = True

    # Checkpoints: radius entries in order from the first one reached; a later one first means a skip.
    radius = cfg["checkpoints"]["radius_m"]
    expected, reached = None, []
    for i in np.flatnonzero(valid):
        for k in np.flatnonzero(np.hypot(*(kmap.checkpoints - (x[i], y[i])).T) <= radius):
            expected = k if expected is None else expected
            if k == expected:
                reached.append(int(k))
                add("checkpoint", t[i], t[i], index=int(k))
                expected += 1
            elif k > expected:
                add("checkpoint_skip", t[i], t[i], index=int(k), missed=int(expected))
                expected = k + 1

    # Start: a run spawned at the route start must reach route_fraction of the route within within_s.
    sc, start_ok = cfg["start"], None
    first_s = np.flatnonzero(np.isfinite(path_s))
    if len(first_s) and path_s[first_s[0]] < sc["applies_if_start_s_below_m"]:
        early = path_s[(t <= t[first_s[0]] + sc["within_s"]) & np.isfinite(path_s)]
        start_ok = bool(early.max() >= sc["route_fraction"] * kmap.s[-1])
        if not start_ok:
            add("start_fail", t[first_s[0]], t[first_s[0]] + sc["within_s"])
            competition_failed.append("start_fail")

    for e in events:  # the judge stops at the last waypoint: later events neither cost time nor fail the run
        e["after_finish"] = finished and e["t_start"] > t_end + 1e-9
    scored = [e for e in events if not e["after_finish"] and not e.get("repeat")]
    competition_failed += [k for k in ("drive_inoperable", "checkpoint_skip") if any(e["type"] == k for e in scored)]
    if judged and sum(e["type"] == "signal_violation" for e in scored) == len(judged):
        competition_failed.append("all_signals_violated")
    competition_failed = sorted(set(competition_failed))
    excluded["no_gt"] |= ~valid
    for reason in competition_failed:
        excluded["episode:" + reason] |= True

    # Targets: future GT ego in the current GT frame (frame_hz steps) and the applied control chunk.
    tg = cfg["targets"]
    horizon = np.arange(n)[:, None] + np.arange(1, tg["traj_points"] + 1)
    later = np.minimum(horizon, n - 1)
    dx, dy = x[later] - x[:, None], y[later] - y[:, None]
    traj = np.stack([c[:, None] * dx + s[:, None] * dy, -s[:, None] * dx + c[:, None] * dy,
                     np.angle(np.exp(1j * (yaw[later] - yaw[:, None])))])
    traj[:, horizon >= n] = np.nan
    tq = t[:, None] + np.arange(tg["ctrl_points"]) * tg["ctrl_dt_s"]
    ctrl = np.full((2,) + tq.shape, np.nan)
    if controls is not None and len(controls):
        cs = controls.sort_values("t")
        ct = cs.t.to_numpy(float)
        k = np.maximum(np.searchsorted(ct, tq, side="right") - 1, 0)
        fresh = (ct[k] <= tq) & (tq - ct[k] <= tg["ctrl_max_age_s"])
        ctrl[0] = np.where(fresh, cs.steer.to_numpy(float)[k], np.nan)
        ctrl[1] = np.where(fresh, (cs.accel - cs.brake).to_numpy(float)[k], np.nan)
    bad = ~valid
    for kind in TARGET_BREAKERS:
        bad |= active[kind]
    for flag in ("noise", "student"):  # student: TODO, no topic records the driver yet (thresholds.yaml topics.driver)
        if flag in f:
            bad |= f[flag].to_numpy(bool)
    cum = np.r_[0, np.cumsum(bad)]

    def clear(frames_ahead):
        """Frames i..i+frames_ahead are recorded and hold no breaker. Driving past the finish stays valid training
        data (section episodes run into the next lap): events there are still judged and still break targets."""
        end = np.minimum(np.arange(n) + frames_ahead, n - 1)
        return (np.arange(n) + frames_ahead < n) & (cum[end + 1] == cum[np.arange(n)])

    traj_ok = clear(tg["traj_points"]) & np.isfinite(traj).all((0, 2))
    ctrl_ok = clear(round(tg["ctrl_points"] * tg["ctrl_dt_s"] / dt)) & np.isfinite(ctrl).all((0, 2))

    def joined(masks):
        return [";".join(k for k in masks if masks[k][i]) for i in range(n)]

    labels = pd.DataFrame(dict(
        t=t, bag_time_ns=f["bag_time_ns"] if "bag_time_ns" in f else pd.NA, x=x, y=y, yaw=yaw,
        speed_kph=v * 3.6, path_s=path_s, section=f.section.to_numpy(), route_section=route_section,
        signal=signal, gps_blackout=blackout, event_type=joined(active), excluded_by=joined(excluded),
        traj_x=list(traj[0]), traj_y=list(traj[1]), traj_yaw=list(traj[2]),
        ctrl_steer=list(ctrl[0]), ctrl_u=list(ctrl[1]), traj_ok=traj_ok, ctrl_ok=ctrl_ok))
    labels["clean"] = labels.excluded_by == ""
    penalty = float(sum(e["penalty_s"] for e in scored))
    counts = defaultdict(int)
    for e in scored:
        counts[e["type"]] += 1
    summary = dict(
        scorer_version=cfg["version"], config_hash=cfg["_hash"],
        time_start_s=t_start, time_end_s=round(t_end, 2), finished=finished, duration_s=round(duration, 2),
        penalty_s=penalty, score_s=round(duration + penalty, 2), counts=dict(counts),
        competition_failed=competition_failed,
        checkpoints_reached=reached, start_ok=start_ok,
        frames=n, clean_frames=int(labels.clean.sum()), clean_ratio=round(float(labels.clean.mean()), 3) if n else 0.0,
        traj_ok_frames=int(traj_ok.sum()), ctrl_ok_frames=int(ctrl_ok.sum()),
        sources=ep.get("sources", {}), unavailable=sorted(ep.get("unavailable", ())),
        judged_stop_lines={k: round(v_, 1) for k, v_ in judged.items()},
        speed_exempt_s=[round(float(a), 1) for a in kmap.speed_exempt_s], events=events,
        note="events are judgments: exclusion windows can be recomputed from them, but raw judge inputs are not "
             "stored, so the judgments themselves (e.g. the re-collision rule) cannot be redone")
    return labels, summary


def write_labels(labels, summary, out):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(labels, preserve_index=False)
    meta = dict(table.schema.metadata or {}, scorer_summary=json.dumps(summary, default=str, ensure_ascii=False))
    pq.write_table(table.replace_schema_metadata(meta), out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bags", nargs="+", help="one episode: a bag or all of its split chunks")
    ap.add_argument("-c", "--config", default=HERE / "thresholds.yaml")
    ap.add_argument("-o", "--out", help="labels parquet (default: <first bag>.labels.parquet)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    kmap = KMap.from_files(cfg)
    from bag import read_episode
    labels, summary = score(read_episode(args.bags, cfg), kmap, cfg)
    out = args.out or str(Path(args.bags[0]).with_suffix(".labels.parquet"))
    write_labels(labels, summary, out)
    brief = copy.deepcopy(summary)
    brief["events"] = [e for e in summary["events"] if e["type"] != "checkpoint"]
    print(json.dumps(brief, indent=1, ensure_ascii=False, default=str))
    print("labels ->", out)


if __name__ == "__main__":
    main()
