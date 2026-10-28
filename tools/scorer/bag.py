"""Read one episode (a bag or its split chunks) into the scorer's 10 Hz tables, on the bag receive clock.

Header stamps are not used; every receiver stamps on receipt or with its own source clock.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


EXTRACT = {
    "ego_gt": lambda m: (m.pose.pose.position.x, m.pose.pose.position.y, _yaw(m.pose.pose.orientation)),
    "speed": lambda m: (m.signed_velocity_mps, m.steering_angle_rad, m.accel, m.brake),
    "ego_status": lambda m: (m.pose.pose.position.x, m.pose.pose.position.y, _yaw(m.pose.pose.orientation),
                             m.twist.twist.linear.x),
    "gps": lambda m: bool(m.status.status < 0 or not (math.isfinite(m.latitude) and math.isfinite(m.longitude))
                          or (m.latitude == 0 and m.longitude == 0)),
    "section": lambda m: m.state,
    # positions are map frame like the Ego GT: CollisionData also reports collisions between other objects
    "collision": lambda m: [(f"{o.type}:{o.unique_id}:{o.name}", o.position.x, o.position.y)
                            for o in [*m.collision_object, *m.collision_objecta]],
    "signal": lambda m: (m.trafficLightIndex.strip("\x00 "), m.trafficLightStatus),
    "noise": lambda m: bool(m.data),
}


def _nearest(src_ns, grid_ns, tol_ns):
    """Index of the nearest source sample per grid time, -1 if farther than tol_ns."""
    if not len(src_ns):
        return np.full(len(grid_ns), -1)
    j = np.searchsorted(src_ns, grid_ns)
    lo, hi = np.clip(j - 1, 0, len(src_ns) - 1), np.clip(j, 0, len(src_ns) - 1)
    j = np.where(np.abs(src_ns[lo] - grid_ns) <= np.abs(src_ns[hi] - grid_ns), lo, hi)
    return np.where(np.abs(src_ns[j] - grid_ns) <= tol_ns, j, -1)


def read_episode(paths, cfg):
    """-> episode dict for scorer.score: frames, contacts, signals, controls, unavailable, sources."""
    topics = {k: v for k, v in cfg["topics"].items() if v}
    rows = {k: [] for k in topics}
    with AnyReader([Path(p) for p in paths], default_typestore=get_typestore(Stores.ROS1_NOETIC)) as reader:
        conns = [c for c in reader.connections if c.topic in topics.values()]
        key_of = {v: k for k, v in topics.items()}
        for conn, t_ns, raw in reader.messages(connections=conns):
            key = key_of[conn.topic]
            rows[key].append((t_ns, EXTRACT[key](reader.deserialize(raw, conn.msgtype))))
    return episode_from_rows(rows, cfg, paths)


def episode_from_rows(rows, cfg, origin="live stream"):
    """rows: {topic key: [(receive ns, EXTRACT value)]} from a bag or the live stream -> episode dict."""
    rows = {k: list(r) for k, r in rows.items()}
    missing = sorted(set(cfg["topics"]) - {k for k, r in rows.items() if r})
    pose_source = next((k for k in ("ego_status", "ego_gt") if rows.get(k)), None)
    if pose_source is None:
        raise ValueError(f"no GT ego pose on {cfg['topics']['ego_status']} or {cfg['topics']['ego_gt']} in {origin}")
    for r in rows.values():
        r.sort(key=lambda item: item[0])

    def times(key):
        return np.array([ts for ts, _ in rows.get(key, [])], dtype=np.int64)

    gt_ns = times(pose_source)
    step = round(1e9 / cfg["frame_hz"])
    grid = np.arange(gt_ns[0], gt_ns[-1] + 1, step, dtype=np.int64)
    pose = np.array([p[:3] for _, p in rows[pose_source]])
    j = _nearest(gt_ns, grid, cfg["max_gt_gap_s"] * 1e9)
    xyz = np.where(j[:, None] >= 0, pose[np.maximum(j, 0)], np.nan)
    frames = pd.DataFrame(dict(t=(grid - grid[0]) / 1e9, bag_time_ns=grid, x=xyz[:, 0], y=xyz[:, 1], yaw=xyz[:, 2]))

    def at_frames(key, values, tol_ns):
        j = _nearest(times(key), grid, tol_ns)
        values = np.asarray(values, float)
        return np.where((j >= 0).reshape((-1,) + (1,) * (values.ndim - 1)), values[np.maximum(j, 0)], np.nan)

    # The judge's speed is Ego Vehicle Status (rules §3-2); competition_status carries the same field name.
    speed_source = next((k for k in ("ego_status", "speed") if rows.get(k)), "gt_pose_difference")
    if speed_source == "gt_pose_difference":
        frames["speed_mps"] = np.r_[np.nan, np.hypot(np.diff(frames.x), np.diff(frames.y)) * cfg["frame_hz"]]
    else:
        frames["speed_mps"] = at_frames(speed_source, [v[3] if speed_source == "ego_status" else v[0]
                                                       for _, v in rows[speed_source]], 2 * step)
    if rows.get("gps"):  # latest fix at or before each frame; frames before the first fix count as covered
        k = np.searchsorted(times("gps"), grid, side="right") - 1
        frames["gps_blackout"] = np.where(k >= 0, np.array([b for _, b in rows["gps"]])[np.maximum(k, 0)], False)
    else:
        frames["gps_blackout"] = False
    if rows.get("section"):  # latest annotation at or before each frame (it is sticky by design)
        k = np.searchsorted(times("section"), grid, side="right") - 1
        frames["section"] = np.where(k >= 0, np.array([v for _, v in rows["section"]])[np.maximum(k, 0)], 0)
    else:
        frames["section"] = 0
    if rows.get("noise"):  # latest flag at or before each frame (the collection noise injector publishes it at 20 Hz)
        k = np.searchsorted(times("noise"), grid, side="right") - 1
        frames["noise"] = np.where(k >= 0, np.array([b for _, b in rows["noise"]])[np.maximum(k, 0)], False)
    frames["signal"] = "unknown"

    collision = rows.get("collision", [])
    contacts = pd.DataFrame([((ts - grid[0]) / 1e9, key, i, x, y) for i, (ts, recs) in enumerate(collision)
                             for key, x, y in recs], columns=["t", "key", "msg", "x", "y"])
    signals = pd.DataFrame([((ts - grid[0]) / 1e9, light, status) for ts, (light, status) in rows.get("signal", [])],
                           columns=["t", "light", "status"])
    controls = pd.DataFrame([((ts - grid[0]) / 1e9, steer, accel, brake) for ts, (_, steer, accel, brake)
                             in rows.get("speed", [])], columns=["t", "steer", "accel", "brake"])
    period = float(np.median(np.diff(times("collision")))) / 1e9 if len(collision) > 1 else None
    return dict(frames=frames, contacts=contacts, signals=signals, controls=controls, unavailable=missing,
                sources=dict(pose=cfg["topics"][pose_source], speed=cfg["topics"].get(speed_source, speed_source),
                             collision_period_s=period))
