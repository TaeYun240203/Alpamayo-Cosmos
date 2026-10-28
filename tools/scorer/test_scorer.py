"""Synthetic-trajectory checks, one or more per judged item.

    uv run --python 3.12 --with-requirements tools/scorer/requirements.txt python -m pytest tools/scorer -q
"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scorer import KMap, load_config, score

CFG = load_config()
DT = 0.1
# Straight 1 km route along +x; lane lines at y = +-1.75; drivable lanelet y in [-5.25, 5.25]; checkpoints every
# 100 m; judged stop lines at x = 350 (straight) and 450 (left); speed exemption 600-700 m.
LINES = [(np.array([[0.0, 1.75], [1000.0, 1.75]]), 0.075), (np.array([[0.0, -1.75], [1000.0, -1.75]]), 0.075)]
STOPS = {"S350": np.array([[350.0, -1.75], [350.0, 1.75]]), "S450": np.array([[450.0, -1.75], [450.0, 1.75]])}
ROAD = [np.array([[0.0, -5.25], [1000.0, -5.25], [1000.0, 5.25], [0.0, 5.25]])]
KM = KMap(np.c_[np.arange(0, 1000.5, 0.5), np.zeros(2001)], [[100.0 * i, 0.0] for i in range(10)], LINES, STOPS,
          (600.0, 700.0), lanelets=ROAD)


def cfg(**overrides):
    c = copy.deepcopy(CFG)
    c["signal"]["stop_lines"] = {"S350": {"movement": "straight", "lights": ["L1", "L1b"]},
                                 "S450": {"movement": "left", "lights": ["L2"]}}
    for key, value in overrides.items():
        section, name = key.split("__")
        c[section][name] = value
    return c


def drive(v_mps, x0=0.0, y=0.0, signal="green"):
    v = np.asarray(v_mps, float)
    x = x0 + np.r_[0.0, np.cumsum(v[:-1] * DT)]
    return pd.DataFrame(dict(t=np.arange(len(v)) * DT, x=x, y=y, yaw=0.0, speed_mps=v, section=0, signal=signal))


NO_CONTACTS = pd.DataFrame(columns=["t", "key", "msg"])


def run(frames, contacts=NO_CONTACTS, c=None, signals=None, controls=None):
    ep = dict(frames=frames, contacts=contacts, signals=signals, controls=controls)
    return score(ep, KM, c or cfg())


def touching(key, t0, t1, period=0.02):
    """CollisionData sightings of one object in every message from t0 to t1 (50 Hz)."""
    return [(t, key, round(t / period)) for t in np.arange(t0, t1 - 1e-9, period)]


def kinds(summary, kind):
    return [e for e in summary["events"] if e["type"] == kind]


def test_collision_recontact_after_missing_a_single_message():
    rows = [r for r in touching("1:7:cone", 5.0, 5.2) if r[2] != 255]   # absent from exactly one message
    _, summary = run(drive(np.full(100, 5.0), x0=110), contacts=pd.DataFrame(rows, columns=NO_CONTACTS.columns))
    assert len(kinds(summary, "collision")) == 2


def test_collision_first_contact_and_recontact_per_object():
    frames = drive(np.full(200, 5.0), x0=110)
    contacts = pd.DataFrame(
        touching("1:7:cone", 5.0, 5.5)                             # one continuous contact
        + touching("1:7:cone", 8.0, 8.2)                           # re-contact after a gap -> again
        + touching("2:9:barrier", 5.2, 5.22), columns=NO_CONTACTS.columns)   # another object -> its own event
    labels, summary = run(frames, contacts=contacts)
    assert len(kinds(summary, "collision")) == 3
    assert summary["penalty_s"] == 45
    excluded = labels.set_index(labels.t.round(1)).excluded_by.str.contains("collision")
    assert excluded[2.0] and excluded[9.1] and not excluded[1.9] and not excluded[9.3]   # [t0-3, t_end+1] unions
    assert labels.loc[labels.t.round(1) == 5.2, "event_type"].item() == "collision"


@pytest.mark.parametrize("state,kind,penalty", [("red", "signal_violation", 30), ("green", None, 0),
                                                ("unknown", "signal_unverified", 0)])
def test_signal_state_when_front_axle_crosses_stop_line(state, kind, penalty):
    frames = drive(np.full(100, 10.0), x0=300)       # front axle (+3 m) crosses x=350 at t=4.7 s
    frames["signal"] = np.where(frames.t.round(1) == 4.7, state, "green")
    labels, summary = run(frames)
    events = [e for e in summary["events"] if e["type"].startswith("signal")]
    assert [e["type"] for e in events] == ([kind] if kind else [])
    assert summary["penalty_s"] == penalty
    if kind:
        assert events[0]["t_start"] == 4.7 and events[0]["stop_line"] == "S350"   # S450 is crossed only at t=14.7
        win = labels[labels.excluded_by.str.contains(kind)].t
        assert np.isclose(win.min(), 0.7) and np.isclose(win.max(), 5.7)


def lights(*rows):
    return pd.DataFrame([(t, light, status) for t0, t1, light, status in rows for t in np.arange(t0, t1, 0.5)],
                        columns=["t", "light", "status"])


@pytest.mark.parametrize("l1,l2,expected", [
    (1, 48, ["signal_violation"]),          # straight on red
    (16, 48, []),                            # straight on green, left on green + left arrow
    (4, 36, []),                             # yellow / yellow + left arrow: no penalty
    (16, 16, ["signal_violation"]),          # left turn on a plain green (no arrow)
    (None, 32, ["signal_unverified"]),       # no message from the mapped light
])
def test_signal_messages_mapped_to_stop_line_and_movement(l1, l2, expected):
    frames = drive(np.full(200, 10.0), x0=300)     # crosses S350 (straight) at 4.7 s and S450 (left) at 14.7 s
    rows = [(0, 20, "L2", l2), (0, 20, "OTHER", 1)] + ([(0, 20, "L1b", l1)] if l1 is not None else [])
    labels, summary = run(frames.assign(signal="green"), signals=lights(*rows))
    assert [e["type"] for e in summary["events"] if e["type"].startswith("signal")] == expected
    assert labels.loc[labels.t.round(1) == 14.0, "signal"].item() == {48: "green", 36: "green", 16: "red", 32: "green"}[l2]


def test_signal_stale_message_is_unknown():
    frames = drive(np.full(100, 10.0), x0=300)
    _, summary = run(frames, signals=lights((0, 1, "L1", 16)))          # last update 4.2 s before the crossing
    assert summary["counts"].get("signal_unverified") == 1


def test_signal_red_seen_only_before_crossing_is_not_a_violation():
    frames = drive(np.full(100, 10.0), x0=300)
    frames["signal"] = np.where(frames.t < 4.65, "red", "green")
    assert run(frames)[1]["penalty_s"] == 0


def test_speed_immediate_then_every_3s_and_exemption():
    v = np.r_[np.full(50, 15.0), np.full(35, 70 / 3.6), np.full(50, 15.0)]   # 3.5 s over 60 kph
    labels, summary = run(drive(v, x0=110))
    assert [e["penalty_s"] for e in kinds(summary, "speed")] == [30]      # +15 at once, +15 after 3 s
    over = labels[labels.event_type.str.contains("speed")].t
    assert np.isclose(over.min(), 5.0) and np.isclose(over.max(), 8.4)
    assert np.isclose(labels[labels.excluded_by.str.contains("speed")].t.min(), 3.0)   # 2 s lead-in
    assert run(drive(v, x0=550))[1]["counts"].get("speed") is None          # over-speed at 625-693 m, exempt
    assert run(drive(np.full(100, 59.9 / 3.6), x0=110))[1]["penalty_s"] == 0


def test_speed_frames_without_pose_keep_the_last_route_position():
    v = np.r_[np.full(50, 15.0), np.full(35, 70 / 3.6), np.full(50, 15.0)]   # over-speed inside the exemption
    ep = drive(v, x0=550)
    ep.loc[60:75, ["x", "y", "yaw"]] = np.nan                               # dropped Ego status poses
    assert run(ep)[1]["counts"].get("speed") is None


@pytest.mark.parametrize("frames_over,penalty", [(29, 15), (30, 30), (59, 30), (60, 45)])
def test_speed_second_penalty_exactly_at_3s(frames_over, penalty):
    v = np.r_[np.full(20, 15.0), np.full(frames_over, 61 / 3.6), np.full(20, 15.0)]   # 10 Hz: 30 frames = 3.0 s
    assert run(drive(v, x0=110))[1]["penalty_s"] == penalty


def test_lane_contact_every_3s_resets_and_gps_blackout_exemption():
    labels, summary = run(drive(np.full(70, 5.0), x0=110, y=0.9))   # left wheels 0.035 m from the line
    assert [e["penalty_s"] for e in kinds(summary, "lane")] == [10]  # 7 s continuous -> 2 x 5 s
    assert not labels.clean.any()
    _, short = run(drive(np.full(25, 5.0), x0=110, y=0.9))
    assert short["penalty_s"] == 0 and len(kinds(short, "lane")) == 1   # labelled, below 3 s
    brief = np.where((np.arange(100) >= 40) & (np.arange(100) < 60), 0.9, 0.5)   # a 2 s touch: no penalty
    assert run(drive(np.full(100, 5.0), x0=110, y=brief))[0].traj_ok.iloc[0]    # so its targets are kept
    long_ = np.where((np.arange(100) >= 40) & (np.arange(100) < 75), 0.9, 0.5)   # 3.5 s: penalized
    assert not run(drive(np.full(100, 5.0), x0=110, y=long_))[0].traj_ok.iloc[0]
    y = np.where((np.arange(60) >= 25) & (np.arange(60) < 30), 0.0, 0.9)   # 2.5 s, release, 3.0 s
    assert [e["penalty_s"] for e in kinds(run(drive(np.full(60, 5.0), x0=110, y=y))[1], "lane")] == [0, 5]
    tunnel = drive(np.full(70, 5.0), x0=110, y=0.9).assign(gps_blackout=True)
    assert kinds(run(tunnel)[1], "lane") == []
    assert run(drive(np.full(70, 5.0), x0=110, y=0.5))[1]["counts"] == {}


def test_drive_inoperable_off_road_outside_gps_blackout_only():
    y = np.where((np.arange(100) >= 40) & (np.arange(100) < 60), 8.0, 0.0)   # 2 s with all wheels off the road
    _, summary = run(drive(np.full(100, 5.0), x0=110, y=y))
    assert summary["competition_failed"] == ["drive_inoperable"]
    assert run(drive(np.full(100, 5.0), x0=110, y=y).assign(gps_blackout=True))[1]["competition_failed"] == []
    brief = np.where((np.arange(100) >= 40) & (np.arange(100) < 45), 8.0, 0.0)
    assert run(drive(np.full(100, 5.0), x0=110, y=brief))[1]["competition_failed"] == []   # < min_s


def test_checkpoints_in_order_and_skip():
    _, summary = run(drive(np.full(500, 10.0), x0=50))
    assert summary["checkpoints_reached"] == [1, 2, 3, 4, 5] and summary["competition_failed"] == []
    frames = drive(np.full(500, 10.0), x0=50)
    frames.loc[(frames.x > 190) & (frames.x < 210), "y"] = 5.0     # swerves 5 m around checkpoint 2
    _, skipped = run(frames)
    assert skipped["competition_failed"] == ["checkpoint_skip"]


def test_start_reaches_5_percent_within_60s():
    assert run(drive(np.full(700, 1.0)))[1]["start_ok"] is True          # 60 m >= 5 % of 1 km in 60 s
    slow = run(drive(np.full(700, 0.7)))[1]                               # 42 m
    assert slow["start_ok"] is False and "start_fail" in slow["competition_failed"]
    assert run(drive(np.full(100, 1.0), x0=300))[1]["start_ok"] is None  # segment episode: not judged


def test_episode_score_is_time_plus_competition_penalty_only():
    v = np.r_[np.zeros(20), np.full(30, 15.0), np.full(10, 70 / 3.6), np.full(40, 15.0)]
    _, summary = run(drive(v, x0=110))
    # moves at t=2.1 (first frame > 5 cm), not finished -> ends at the last frame 9.9
    assert summary["time_start_s"] == 2.1 and summary["duration_s"] == 7.8 and summary["score_s"] == 22.8


def test_time_ends_at_last_waypoint_and_15_minute_limit():
    frames = drive(np.r_[np.full(220, 2.5), np.zeros(100)], x0=950)   # reaches 999 m (last waypoint - 1 m) at 19.6 s
    frames.loc[frames.t > 21.0, "y"] = 0.9                          # line contact after the finish is not scored
    _, summary = run(frames)
    assert summary["finished"] and summary["time_end_s"] == 19.6 and summary["duration_s"] == 19.5
    assert summary["counts"].get("lane") is None and summary["competition_failed"] == []
    slow = run(drive(np.full(9100, 0.1), x0=110))[1]               # 910 s, not finished
    assert slow["competition_failed"] == ["time_limit"]


def test_targets_future_gt_and_masks():
    frames = drive(np.full(200, 10.0), x0=110)
    contacts = pd.DataFrame(touching("1:7:cone", 12.0, 12.1), columns=NO_CONTACTS.columns)
    ctrl = pd.DataFrame([(t, 0.01 * t, 0.5, 0.1) for t in np.arange(0, 20.0, 0.02)], columns=["t", "steer", "accel", "brake"])
    labels, summary = run(frames, contacts=contacts, controls=ctrl)
    first = labels.iloc[0]
    assert np.allclose(first.traj_x[:3], [1.0, 2.0, 3.0]) and np.allclose(first.traj_y, 0) and len(first.traj_x) == 64
    assert np.allclose(first.ctrl_u, 0.4) and np.allclose(first.ctrl_steer[[0, 19]], [0.0, 0.0095], atol=2e-4)
    ok = labels.set_index(labels.t.round(1))
    assert ok.traj_ok[5.5] and not ok.traj_ok[5.6] and not ok.traj_ok[12.1] and ok.traj_ok[12.2]   # 6.4 s ahead
    assert ok.ctrl_ok[10.9] and not ok.ctrl_ok[11.0] and ok.ctrl_ok[12.2]                         # 1.0 s ahead
    assert not ok.traj_ok[13.6] and ok.ctrl_ok[18.9] and not ok.ctrl_ok[19.0]                    # episode end
    noisy = frames.assign(noise=(frames.t > 3) & (frames.t < 4))
    assert not run(noisy, controls=ctrl)[0].set_index(labels.t.round(1)).ctrl_ok[2.5]


@pytest.mark.skipif(not (Path(CFG["_path"]).parent / CFG["map"]["dir"]).exists(), reason="map repo not checked out")
def test_real_map_judged_stop_lines_and_speed_exemption():
    kmap = KMap.from_files(CFG)
    judged = {sid: kmap.route_stops.get(sid) for sid in CFG["signal"]["stop_lines"]}
    assert sorted(round(s, 1) for s in judged.values()) == [208.5, 315.9, 436.8, 609.9]
    assert all(kmap.section_at(np.array(judged[sid])) == spec["section"] for sid, spec in CFG["signal"]["stop_lines"].items())
    assert [spec["movement"] for spec in CFG["signal"]["stop_lines"].values()].count("left") == 1   # s7
    assert np.allclose(kmap.speed_exempt_s, (1118.7, 1741.7), atol=0.5)    # CP10 .. CP13 on the route
    assert round(kmap.s[-1]) == 2185
    assert len(kmap.lanelets) > 1500 and (kmap.on_road(kmap.path) | kmap.near_path(kmap.path, 2.0)).all()


def test_targets_stay_valid_after_finish_and_repeat_red_crossings_break_them():
    # finish: stop at the last waypoint and stay; the frames after the finish keep valid targets
    v = np.r_[np.full(390, 1.0), np.zeros(200)]                     # 960 m -> 999 m, then 20 s at rest
    labels, summary = run(drive(v, x0=960.0))
    assert summary["finished"]
    after = labels.t > summary["time_end_s"] + 1.0
    assert labels.traj_ok[after & (labels.t < labels.t.max() - 7)].all()
    # a second crossing of the same stop line on red: no extra penalty, but its targets are broken
    x = np.r_[np.linspace(340, 360, 100), np.nan, np.linspace(340, 360, 100)]   # no GT frame across the jump back
    sig_state = np.where(np.arange(201) < 100, "green", "red")
    frames = pd.DataFrame(dict(t=np.arange(201) * DT, x=x, y=0.0, yaw=0.0, speed_mps=2.0, section=0, signal=sig_state))
    labels, summary = run(frames)
    reds = kinds(summary, "signal_violation")
    assert len(reds) == 1 and reds[0]["repeat"] and summary["penalty_s"] == 0
    assert not labels.traj_ok[(labels.t > 11) & (labels.t < 15)].any()


def test_collision_records_far_from_the_ego_are_not_ego_collisions():
    frames = drive(np.full(300, 5.0), x0=110)                     # Ego at x 110..260, y 0
    near = [(t, "1:7:cone", m, 110 + 5 * t + 3.0, 0.5) for t, _, m in touching("1:7:cone", 2.0, 2.4)]
    far = [(t, "1:16:", m, -111.3, 358.3) for t, _, m in touching("1:16:", 1.0, 9.0)]   # NPC pile-up elsewhere
    contacts = pd.DataFrame(near + far, columns=["t", "key", "msg", "x", "y"])
    labels, summary = run(frames, contacts=contacts)
    assert [e["object"] for e in kinds(summary, "collision")] == ["1:7:cone"]
    assert labels.traj_ok[(labels.t > 4.5) & (labels.t < 20)].all()   # the far pile-up masks nothing
