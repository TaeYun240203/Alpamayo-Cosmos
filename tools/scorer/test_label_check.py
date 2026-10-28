"""Hand-built cases for label_check: each unavoidable-violation case (lane, signal, collision, speed, drivable area,
route, rule conflicts) with the verdict a correct checker must give, plus the plumbing.

    uv run --python 3.12 --with-requirements tools/scorer/requirements.txt python -m pytest tools/scorer -q
"""
import copy
import math

import numpy as np
import pytest
import yaml

from label_check import DT, HERE, N, LabelChecker, _near
from scorer import KMap, load_config

# Straight 1 km route along +x (route lane y in [-1.75, 1.75]). Solid paint: right edge y = -1.75 and the yellow centre
# line y = +1.75, both open at x 380-420 (intersection); the opposite lane's edge y = +5.25. Drivable: y -1.75..5.25.
# Judged stop lines x = 350 (straight) and 450 (left); S500 is unsignalised. Speed exemption 600-800; GPS blackout
# 850-950; checkpoints every 100 m.
LINES = [(np.array(p, float), 0.075) for p in ([[0, 1.75], [380, 1.75]], [[420, 1.75], [1000, 1.75]],
                                                ([0, -1.75], [380, -1.75]), ([420, -1.75], [1000, -1.75]),
                                                ([0, 5.25], [1000, 5.25]))]
STOPS = {k: np.array([[x, -1.75], [x, 1.75]]) for k, x in (("S350", 350.0), ("S450", 450.0), ("S500", 500.0))}
KM = KMap(np.c_[np.arange(0, 1000.5, 0.5), np.zeros(2001)], [[100.0 * i, 0.0] for i in range(10)], LINES, STOPS,
          (600.0, 800.0), lanelets=[np.array([[0.0, -1.75], [1000.0, -1.75], [1000.0, 5.25], [0.0, 5.25]])])
CFG = copy.deepcopy(load_config())
CFG["signal"]["stop_lines"] = {"S350": {"movement": "straight", "lights": ["L1"]}, "S450": {"movement": "left", "lights": ["L2"]}}
LC = yaml.safe_load((HERE / "label_check.yaml").read_text())
LC["blackout_s"] = [[850.0, 950.0]]
CHK = LabelChecker(CFG, KM, LC)
T = np.arange(N + 1) * DT
ON_LINE = 1.75 - 0.815   # ego y with its left wheels on the centre line


def drive(x0, y0, v, y_of=None, yaw0=0.0):
    """-> (ego, label (64, 4)): speed profile v (N + 1, or constant), y(t) = y_of(t) (default straight at yaw0)."""
    v = np.broadcast_to(np.asarray(v, float), T.shape).copy()
    s = np.r_[0, np.cumsum(0.5 * (v[1:] + v[:-1]) * DT)]
    if y_of is None:
        x, y, yaw = x0 + s * math.cos(yaw0), y0 + s * math.sin(yaw0), np.full(N + 1, yaw0)
    else:
        y = np.asarray(y_of(T), float)
        x = x0 + np.r_[0, np.cumsum(np.sqrt(np.maximum(np.diff(s) ** 2 - np.diff(y) ** 2, 0)))]
        yaw = np.arctan2(np.gradient(y), np.maximum(np.gradient(x), 1e-6))
    return dict(x=x0, y=y0, yaw=float(yaw[0]), v=float(v[0])), np.c_[x, y, yaw, v][1:]


def brake(v0, a, t0=0.0):
    return np.maximum(v0 - a * np.maximum(T - t0, 0), 0)


def ease(y0, y1, t0, t1):
    """y(t) easing from y0 to y1 between t0 and t1 (cosine)."""
    return lambda t: y0 + (y1 - y0) * 0.5 * (1 - np.cos(np.pi * np.clip((t - t0) / (t1 - t0), 0, 1)))


def check(ego, traj, **kw):
    return CHK.check(ego, traj, **kw)


# ------------------------------------------------------------------ plumbing
def test_fast_near_matches_scorer():
    rng = np.random.default_rng(0)
    pts = np.c_[rng.uniform(-5, 1005, 4000), rng.uniform(-3, 7, 4000)]
    reach = KM.seg_hw + 0.2
    assert (_near(pts, KM.seg_a, KM.seg_b, reach, KM.grid) == KM.near_line(pts, 0.2)).all()


def test_clean_drive_passes_and_ak_input_agrees():
    ego, tr = drive(100.0, 0.0, 10.0)
    r = check(ego, tr)
    assert r["verdict"] == "pass" and r["valid"] and not r["reasons"], r
    ak = np.zeros((N, 2))
    assert CHK.check(ego, ak=ak)["verdict"] == "pass"


def test_bad_input_is_unjudgeable_lost_starts_unrecoverable():
    ego, tr = drive(100.0, 0.0, 10.0)
    tr[10, 0] = np.nan
    assert check(ego, tr)["reasons"] == ["input:nonfinite"]
    ego, tr = drive(100.0, 0.0, 5.0, yaw0=math.pi)
    r = check(ego, tr)
    assert r["verdict"] == "unrecoverable" and not r["valid"] and r["reasons"] == ["start:wrong_way"]
    ego, tr = drive(100.0, 20.0, 5.0)
    assert check(ego, tr)["reasons"] == ["start:off_route"]


# ------------------------------------------------------------------ lane
def test_on_the_line_at_t0_escape_passes_riding_it_fails():
    ego, tr = drive(100.0, ON_LINE, 10.0, ease(ON_LINE, 0.0, 0.0, 2.0))
    r = check(ego, tr)
    assert r["verdict"] == "pass_unavoidable" and r["label"]["lane"] > 0 and r["oracle"]["lane"] > 0, r
    ego, tr = drive(100.0, ON_LINE, 10.0)
    r = check(ego, tr)
    assert r["verdict"] == "fail" and not r["valid"] and "lane:avoidable" in r["reasons"], r


def test_needless_line_touch_fails_brief_touch_is_tolerated():
    ego, tr = drive(100.0, 0.0, 10.0, ease(0.0, ON_LINE, 0.0, 2.0))   # drifts onto the centre line and stays
    assert check(ego, tr)["verdict"] == "fail"
    dip = lambda t: ease(0.0, ON_LINE, 1.0, 1.6)(t) - ease(0.0, ON_LINE, 1.7, 2.3)(t)   # touches ~0.6 s, comes back
    r = check(*drive(100.0, 0.0, 10.0, dip))
    assert r["verdict"] == "pass_unavoidable" and "lane:tolerated" in r["reasons"], r


def test_opposite_lane_crossing_back_passes_running_along_the_line_fails():
    ego, tr = drive(100.0, 3.5, 8.0, ease(3.5, 0.0, 0.0, 3.0))
    r = check(ego, tr)
    assert r["verdict"] == "pass_unavoidable" and r["valid"], r
    ego, tr = drive(100.0, 3.5, 8.0, ease(3.5, 1.75 + 0.815, 0.0, 1.5))   # right wheels onto the centre line, stays
    r = check(ego, tr)
    assert r["verdict"] == "fail" and "lane:avoidable" in r["reasons"], r
    r = check(*drive(100.0, 3.5, 8.0))                                      # stays in the opposite lane: no paint
    assert r["verdict"] == "pass" and "ends_off_route_lane" in r["tags"], r


def test_off_road_return_over_the_edge_line_passes():
    ego, tr = drive(100.0, -4.0, 5.0, ease(-4.0, 0.0, 0.0, 4.0))
    r = check(ego, tr)
    assert r["verdict"] == "pass_unavoidable" and r["label"]["inoperable"] > 0 and r["label"]["lane"] > 0, r


def test_heading_into_the_line_at_speed_is_unavoidable():
    yaw = math.radians(25)
    ego, tr = drive(100.0, 0.5, 12.0, lambda t: 0.5 + np.minimum(12 * t * math.sin(yaw), 1.3) - ease(0, 1.3, 0.6, 2.4)(t))
    r = check(ego, tr)
    assert r["verdict"] == "pass_unavoidable" and r["oracle"]["lane"] > 0, r


def test_curve_too_fast_to_hold_the_lane_is_unavoidable():
    # 100 m straight into a left arc of R 30 m; solid lines 1.75 m either side. 15 m/s 3 m before it needs 7.5 m/s^2 of lateral
    # acceleration: braking at 3 m/s^2 and turning, the label runs wide onto the outer line until it has slowed.
    R, s = 30.0, np.arange(0, 220, 0.5)
    th = np.clip(s - 100, 0, None) / R

    def at(s_, e):
        th_ = np.clip(s_ - 100, 0, None) / R
        bx, by = np.where(s_ < 100, s_, 100 + R * np.sin(th_)), np.where(s_ < 100, 0, R - R * np.cos(th_))
        return bx - e * np.sin(th_), by + e * np.cos(th_), th_
    path = np.c_[at(s, 0.0)[:2]]
    lines = [(np.c_[at(s, e)[:2]], 0.075) for e in (-1.75, 1.75)]
    km = KMap(path, [[0.0, 0.0]], lines, {}, (np.inf, np.inf), lanelets=[np.vstack([np.c_[at(s, -1.75)[:2]], np.c_[at(s, 1.75)[:2]][::-1]])])
    chk = LabelChecker(CFG, km, LC)
    v = np.maximum(15.0 - 3.0 * T, 6.0)
    sl = 97.0 + np.r_[0, np.cumsum(0.5 * (v[1:] + v[:-1]) * DT)]
    e = -0.95 * (np.clip((sl - 100) / 10, 0, 1) - np.clip((sl - 125) / 15, 0, 1))   # wide onto the outer line, back
    x, y, h = at(sl, e)
    r = chk.check(dict(x=x[0], y=y[0], yaw=h[0], v=v[0]), np.c_[x, y, h, v][1:])
    assert r["valid"] and r["label"]["lane"] > 0 and r["oracle"]["lane"] > 0, r


def test_waiting_at_red_with_a_wheel_on_the_line_is_unavoidable():
    r = check(*drive(330.0, ON_LINE, 0.0), signal="red")   # stopped 17 m before S350 on the centre line (R1 CP2 case)
    assert r["verdict"] == "pass_unavoidable" and r["reasons"] in (["lane:unavoidable"], ["lane:conflict"]), r   # pulling
    # away clears the paint but runs the red: the cheaper side


def test_waiting_with_a_wheel_on_paint_is_not_failed_by_pulling_away():
    car = (3, "vehicle", 130.0, -20.0, 1.5708, 0.0, 6.0, 4.6, 1.9)   # crosses the lane ahead at ~3 s (past the 2 s horizon)
    r = check(*drive(100.0, ON_LINE, 0.0), objects=[car])
    assert r["valid"] and r["verdict"] == "pass_unavoidable", r


def test_paint_free_zones_are_not_judged():
    assert check(*drive(380.0, ON_LINE, 4.0))["label"]["lane"] < 0.6   # intersection: the line stops at x 380
    assert check(*drive(860.0, ON_LINE, 10.0))["verdict"] == "pass"   # GPS blackout span
    r = check(*drive(100.0, ON_LINE, 10.0), blackout=True)            # blackout where the map has none: not judged
    assert r["verdict"] == "pass" and "blackout_unmapped" in r["tags"], r


# ------------------------------------------------------------------ signal
@pytest.mark.parametrize("state", ["green", "red"])
def test_crossing_far_line_on_green_passes_on_red_fails(state):
    ego, tr = drive(290.0, 0.0, 10.0)   # front axle 57 m before S350, stoppable
    r = check(ego, tr, signal=state)
    assert r["verdict"] == ("pass" if state == "green" else "fail"), r
    assert r["label"]["t_cross"] == pytest.approx(5.7, abs=0.15)


def test_red_stop_before_the_line_passes():
    ego, tr = drive(290.0, 0.0, brake(10.0, 2.0))   # 25 m to stop, line 57 m ahead
    assert check(ego, tr, signal="red")["verdict"] == "pass"


def test_red_too_close_to_stop_go_through_stopping_in_the_junction_fails():
    # front axle 10 m before the line at 15 m/s: 8 m/s^2 after 0.2 s needs 17 m (user 10/28: go through)
    r = check(*drive(337.0, 0.0, 15.0), signal="red")
    assert r["verdict"] == "pass_unavoidable" and "signal:unavoidable" in r["reasons"] and "red_unstoppable_go" in r["tags"], r
    r = check(*drive(337.0, 0.0, brake(15.0, 8.0)), signal="red")
    assert r["verdict"] == "fail" and "signal:red_stop_in_junction" in r["reasons"] and "red_unstoppable_stop" in r["tags"], r
    car = (3, "vehicle", 366.0, 0.0, 0.0, 0.0, 0.0, 4.6, 1.9)   # a car standing in the junction: stopping is right
    assert "signal:unavoidable" in check(*drive(337.0, 0.0, brake(15.0, 8.0)), signal="red", objects=[car])["reasons"]


def test_red_overshoot_by_a_few_metres_fails():
    v = brake(10.0, 1.8)   # stops 27.8 m on: front axle 2.8 m past the line
    r = check(*drive(322.0, 0.0, v), signal="red")
    assert r["verdict"] == "fail" and r["better"]["signal"] == "max_brake", r


def test_front_axle_past_the_line_on_red_goes_once_clear():
    r = check(*drive(349.0, 0.0, 5.0), signal="red")   # front axle 2 m past S350: the light is already judged
    assert r["verdict"] == "pass" and "red_past_line_go" in r["tags"], r
    r = check(*drive(349.0, 0.0, 0.0), signal="red")   # way clear, stays
    assert r["verdict"] == "fail" and "signal:stay_after_cross" in r["reasons"], r
    car = (3, "vehicle", 362.0, -12.0, 1.5708, 0.0, 4.0, 4.6, 1.9)   # crossing traffic ahead: waiting is right
    r = check(*drive(349.0, 0.0, 0.0), signal="red", objects=[car])
    assert r["verdict"] == "pass" and "red_past_line_wait" in r["tags"], r
    r = check(*drive(346.0, 0.0, 0.0), signal="red")   # body over the line, front axle 1 m short: stay
    assert r["verdict"] == "pass", r


def test_yellow():
    r = check(*drive(330.0, 0.0, 15.0), signal="yellow")   # 17 m at 15 m/s: z rule needs 37.5 m -> go
    assert r["verdict"] == "pass" and "yellow_dilemma_go" in r["tags"], r
    r = check(*drive(300.0, 0.0, 15.0), signal="yellow")   # 47 m: could stop at 3 m/s^2, crosses at 3.1 s (still yellow)
    assert r["verdict"] == "pass" and "yellow_stoppable_go" in r["tags"], r
    r = check(*drive(300.0, 0.0, 8.0), signal="yellow")    # crosses at 5.9 s: red or (yellow before green) green by then
    assert r["verdict"] == "unjudgeable" and r["valid"] and "signal:unknown" in r["reasons"], r


def test_unknown_light_at_a_crossing_is_unjudgeable_elsewhere_irrelevant():
    r = check(*drive(300.0, 0.0, 10.0), signal="unknown")
    assert r["verdict"] == "unjudgeable" and r["valid"] and "signal:unknown" in r["reasons"], r
    assert check(*drive(100.0, 0.0, 10.0), signal="unknown")["verdict"] == "pass"


def test_red_line_passed_beside_its_paint_is_tagged_not_judged():
    r = check(*drive(320.0, 0.0, 8.0, ease(0.0, -2.6, 0.0, 2.0)), signal="red")   # S350 ends at y -1.75
    assert r["label"]["signal"] == 0 and "red_line_bypassed" in r["tags"], r   # (fails for the edge line it crossed)


def test_unsignalised_line_is_not_judged():
    assert check(*drive(460.0, 0.0, 10.0), signal="red")["verdict"] == "pass"   # S500: not a judged stop line


def test_signal_at_reads_movement_bits():
    ego = dict(x=440.0, y=0.0, yaw=0.0, v=5.0)   # S450 is a left turn: plain green (16) is red for it
    assert CHK.signal_at([("tlgt", 9.9, "L2", 16)], 10.0, ego) == "red"
    assert CHK.signal_at([(9.9, "L2", 32)], 10.0, ego) == "green"
    assert CHK.signal_at([(5.0, "L2", 32)], 10.0, ego) == "unknown"   # older than max_age_s
    assert CHK.signal_at([], 10.0, ego) == "unknown"


# ------------------------------------------------------------------ collision
PED = lambda x, y=0.0, vx=0.0, vy=0.0: (7, "pedestrian", x, y, 0.0, vx, vy, 0.6, 0.6)


def test_hitting_a_pedestrian_one_could_stop_for_fails():
    r = check(*drive(100.0, 0.0, 10.0), objects=[PED(150.0)])
    assert r["verdict"] == "fail" and "collision:avoidable" in r["reasons"], r
    r = check(*drive(100.0, 0.0, brake(10.0, 3.0)), objects=[PED(150.0)])
    assert r["verdict"] == "pass", r


def test_pedestrian_inside_stopping_distance_full_brake_passes():
    r = check(*drive(100.0, 0.0, brake(12.0, 8.0)), objects=[PED(110.0)])
    assert r["verdict"] == "pass_unavoidable" and "collision:unavoidable" in r["reasons"], r


def test_swerving_over_the_centre_line_for_a_pedestrian_is_the_cheaper_violation():
    # squeezes past a pedestrian 12 m on (too close to stop for) with its left wheels on the centre line ~2 s
    ego, tr = drive(100.0, 0.0, 12.0, lambda t: ease(0.0, 1.12, 0.0, 0.8)(t) - ease(0.0, 1.12, 2.0, 2.8)(t))
    r = check(ego, tr, objects=[PED(112.0)])
    assert r["verdict"] == "pass_unavoidable" and r["label"]["collision"] == 0 and "lane:conflict" in r["reasons"], r


def test_struck_from_behind_while_stopped_at_red_passes():
    car = (3, "vehicle", 325.0, 0.0, 0.0, 10.0, 0.0, 4.6, 1.9)
    r = check(*drive(340.0, 0.0, 0.0), signal="red", objects=[car])
    assert r["verdict"] == "pass" and r["label"]["struck"] == 1, r
    r = check(*drive(320.0, 0.0, brake(10.0, 6.0)), signal="red", objects=[(3, "vehicle", 306.0, 0.0, 0.0, 10.0, 0.0, 4.6, 1.9)])
    assert r["verdict"] == "pass" and r["label"]["struck"] == 1, r   # hard stop for red, follower runs in


def test_moving_cars_count_only_within_the_trust_horizon():
    cutin = (4, "vehicle", 115.0, -3.5, 0.6, 2.0, 2.4, 4.6, 1.9)   # cuts into the lane right ahead within 1.5 s
    r = check(*drive(100.0, 0.0, 10.0), objects=[cutin])
    assert r["verdict"] == "fail" and "collision:avoidable" in r["reasons"], r
    far = (4, "vehicle", 141.5, 9.0, -1.5708, 0.0, -2.25, 4.6, 1.9)   # constant velocity meets the ego only at 4 s
    assert check(*drive(100.0, 0.0, 10.0), objects=[far])["verdict"] == "pass"   # (fails with vehicle_trust_s 9)


def test_object_size_error_is_tolerated():
    # boxes 0.1 m into each other at the closest pass (Object Info sizes can be off): not a collision for the label
    side = (6, "obstacle", 140.0, -0.95 - 0.5 + 0.1, 0.0, 0.0, 0.0, 1.0, 1.0)
    assert check(*drive(100.0, 0.0, 8.0), objects=[side])["label"]["collision"] == 0


def test_rear_car_in_the_lane_being_changed_into_is_the_labels():
    rear = (8, "vehicle", 85.0, 3.5, 0.0, 16.0, 0.0, 4.6, 1.9)   # in the next lane, 6 m/s faster, ~12 m behind
    merge = ease(0.0, 3.5, 0.5, 2.5)
    r = check(*drive(100.0, 0.0, 10.0, merge), objects=[rear])
    assert r["verdict"] == "fail" and "collision:lane_change_rear" in r["reasons"], r
    assert check(*drive(100.0, 0.0, 10.0), objects=[rear])["verdict"] == "pass"   # staying in lane: it passes
    # the lane is blocked too close to stop (box 13 m on at 12 m/s): every reference hits something -> unavoidable
    box, rear = (9, "obstacle", 113.0, 0.0, 0.0, 0.0, 0.0, 1.2, 1.2), (8, "vehicle", 88.0, 3.5, 0.0, 18.0, 0.0, 4.6, 1.9)
    r = check(*drive(100.0, 0.0, 12.0, ease(0.0, 3.5, 0.0, 1.5)), objects=[box, rear])
    assert r["verdict"] == "pass_unavoidable" and "collision:unavoidable" in r["reasons"], r


def test_car_following_in_the_egos_lane_is_not_a_lane_change_whatever_the_route_does():
    # shadow run g, s ~1614: the ego pulls away along its lane while the route offset grows 1.6 -> 3.0 m (route lanes
    # bend apart); the car 16 m behind was 0.4 m off the ego at t0, so it is a follower running into it
    car = (8, "vehicle", 84.0, 2.0, 0.0, 8.0, 0.0, 4.6, 1.9)
    pull = np.minimum(1.5 * T, 6.0)
    r = check(*drive(100.0, 1.6, pull, lambda t: 1.6 + 1.4 * np.clip(t / 1.5, 0, 1)), objects=[car])
    assert r["label"]["collision"] == 0 and not any("lane_change_rear" in x for x in r["reasons"]), r


def test_rear_cars_are_judged_whatever_the_model_saw():
    far = (8, "vehicle", 75.0, 3.5, 0.0, 22.0, 0.0, 4.6, 1.9)   # 25 m behind: past the shadow's 20 m rear range
    r = check(*drive(100.0, 0.0, 10.0, ease(0.0, 3.5, 0.5, 2.5)), objects=[far])
    assert r["verdict"] == "pass_unavoidable" and r["valid"] and r["debug_fail"], r
    assert r["reasons"] == ["lane:tolerated", "collision:lane_change_rear_hidden"], r
    lc = copy.deepcopy(LC)
    lc["merge_s"] = [[200.0, 300.0]]
    merge = LabelChecker(CFG, KM, lc)
    car = (8, "vehicle", 195.0, 0.0, 0.0, 18.0, 0.0, 4.6, 1.9)   # same lane, 15 m behind, 8 m/s faster
    r = merge.check(*drive(210.0, 0.0, 10.0), objects=[car])     # inside the merge: the label's
    assert r["label"]["collision"] == 1 and r["verdict"] != "pass", r
    r = merge.check(*drive(110.0, 0.0, 10.0), objects=[(8, "vehicle", 95.0, 0.0, 0.0, 18.0, 0.0, 4.6, 1.9)])
    assert r["verdict"] == "pass" and r["label"]["collision"] == 0, r   # elsewhere: not the label's


def test_hitting_what_was_hidden_at_t0_passes():
    # a pedestrian behind a parked van steps into the lane: hit at ~2.6 s, braking from t0 would have stopped in time
    walker = (7, "pedestrian", 130.0, -3.5, 0.0, 0.0, 1.5, 0.6, 0.6)
    van = (9, "obstacle", 125.0, -3.0, 0.0, 0.0, 0.0, 5.0, 2.0)
    assert list(CHK.check(*drive(100.0, 0.0, 10.0), objects=[walker])["reasons"]) == ["collision:avoidable"]
    r = check(*drive(100.0, 0.0, 10.0), objects=[walker, van])
    assert r["verdict"] == "pass_unavoidable" and r["reasons"] == ["collision:hidden_at_t0"] and r["label"]["hidden_hits"] == 1, r
    from label_check import visible
    assert list(visible(100.0, 0.0, 0.0, [walker, van], LC["visibility"])) == [False, True]


def test_contact_at_t0_is_ignored():
    r = check(*drive(100.0, 0.0, 0.0), objects=[(5, "vehicle", 106.0, 0.0, 0.0, 0.0, 0.0, 4.6, 1.9)])
    assert r["verdict"] == "pass", r


def test_static_obstacle_avoidance_over_the_centre_line_passes_hitting_it_fails():
    box = (9, "obstacle", 150.0, -0.3, 0.0, 0.0, 0.0, 1.2, 1.2)
    ego, tr = drive(110.0, 0.0, 8.0, lambda t: ease(0.0, 2.6, 2.0, 3.5)(t) - ease(0.0, 2.6, 5.0, 6.4)(t))
    r = check(ego, tr, objects=[box])
    assert r["valid"] and r["label"]["collision"] == 0 and r["label"]["lane"] > 0, r
    r = check(*drive(110.0, 0.0, 8.0), objects=[box])
    assert r["verdict"] == "fail" and "collision:avoidable" in r["reasons"], r


# ------------------------------------------------------------------ speed
def test_over_the_limit_at_t0():
    v0 = 70 / 3.6
    r = check(*drive(100.0, 0.0, np.maximum(brake(v0, 4.0), 59 / 3.6)))
    assert r["verdict"] == "pass_unavoidable" and "speed:unavoidable" in r["reasons"], r
    r = check(*drive(100.0, 0.0, v0))
    assert r["verdict"] == "fail" and "speed:avoidable" in r["reasons"], r


def test_speed_noise_and_exemption():
    assert check(*drive(100.0, 0.0, 60.5 / 3.6))["verdict"] == "pass"
    assert check(*drive(610.0, 0.0, 80 / 3.6))["verdict"] == "pass"


# ------------------------------------------------------------------ drivable area, route
def test_standing_on_the_grass_when_the_road_is_reachable_fails():
    r = check(*drive(100.0, -6.0, 0.0))   # route_go pulls back onto the road
    assert r["verdict"] == "fail" and "inoperable:avoidable" in r["reasons"], r


def test_checkpoint_missed_by_all_is_unrecoverable_staying_away_when_reachable_fails():
    r = check(*drive(95.0, -4.6, 6.0, ease(-4.6, 0.0, 0.0, 4.0)))   # grass 4.6 m off checkpoint 100 m: no way within 3 m
    assert r["verdict"] == "unrecoverable" and not r["valid"] and "checkpoint:missed" in r["reasons"], r
    r = check(*drive(60.0, 3.8, 8.0))    # opposite lane 40 m before it: the route pp gets back in time
    assert r["verdict"] == "fail" and "checkpoint:avoidable" in r["reasons"], r


# ------------------------------------------------------------------ rule conflicts
def test_recovery_into_a_red_light_stop_passes():
    ego, tr = drive(305.0, 3.5, brake(8.0, 1.6), ease(3.5, 0.0, 0.0, 3.0))
    r = check(ego, tr, signal="red")
    assert r["verdict"] == "pass_unavoidable" and r["label"]["signal"] == 0, r
