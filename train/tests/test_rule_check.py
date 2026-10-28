"""scripts/rule_check.py on a synthetic map: each rule fails exactly the trajectory that breaks it."""
import sys
from pathlib import Path

import numpy as np, pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rule_check import Rules


def make_rules(f):
    # map frame: road along +x; solid line at y = +3 (half width 0.075), stop line x = 50 across, obstacle at (30, -3.5)
    np.savez(f, seg_a=np.array([[-100.0, 3.0]]), seg_b=np.array([[300.0, 3.0]]), seg_hw=np.array([0.075]),
             path=np.array([[-100.0, 0], [300, 0]]), path_s=np.array([0.0, 400]), exempt_s=np.array([500.0, 600]),
             stop_ids=np.array(["s"]), stop_pts=np.array([[50.0, -5], [50, 5]]), stop_off=np.array([0, 2]),
             wheels=np.array([[3.0, 0.815], [3.0, -0.815], [0, 0.815], [0, -0.815]]), tire_hw=0.1275, front_axle=3.0,
             body=np.array([-1.0, 3.9, 0.95]), speed_limit_kph=60, lane_period_s=3.0, violating=np.array(["red"]),
             episodes=np.array(["ep"]), obs=np.array([[0, 30.0, -3.5, 0.0, 0.5, 0.5]]), thresholds_hash="x")
    return Rules(f)


@pytest.fixture(scope="module")
def rules(tmp_path_factory):
    return make_rules(tmp_path_factory.mktemp("ctx") / "ctx.npz")


def ego(signal=3, n=200):
    return {"episode": "ep", "x": np.zeros(n), "y": np.zeros(n), "yaw": np.zeros(n), "path_s": np.arange(n) * 1.0,
            "signal": np.full(n, signal, np.int8), "gps_blackout": np.zeros(n, bool)}


def traj(v=10.0, y=0.0):                                          # 64 points, 0.1 s apart, along +x at lateral y
    t = np.arange(1, 65) * 0.1
    return np.stack([v * t, np.full(64, y), np.zeros(64)], -1)[None, None]


def fails(rules, e, tr):
    return dict(zip(("speed", "lane", "signal", "obstacle"), rules.check(e, np.array([0]), tr)[0, 0]))


def test_clean_trajectory_passes(rules):
    assert not any(fails(rules, ego(), traj(v=10)).values())        # crosses the stop line at 5 s on green


def test_each_rule(rules):
    assert fails(rules, ego(), traj(v=17.5))["speed"]               # 63 kph
    assert fails(rules, ego(), traj(v=5, y=2.1))["lane"]             # left wheels on the solid line for 6.4 s
    assert not fails(rules, ego(), traj(v=5, y=1.5))["lane"]
    assert fails(rules, ego(signal=1), traj(v=10))["signal"]         # red at the crossing row
    assert fails(rules, ego(signal=0), traj(v=10))["signal"]         # unknown light at a judged stop line
    assert not fails(rules, ego(signal=1), traj(v=5))["signal"]      # stops short of x = 50 in 6.4 s (32 m)
    assert fails(rules, ego(), traj(v=8, y=-3.0))["obstacle"]        # drives through the box at x = 30
    assert not fails(rules, ego(), traj(v=8, y=-1.0))["obstacle"]


def test_blackout_exempts_lane(rules):
    e = ego(); e["gps_blackout"][:] = True
    assert not fails(rules, e, traj(v=5, y=2.1))["lane"]
