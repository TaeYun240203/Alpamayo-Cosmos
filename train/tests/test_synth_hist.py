"""scripts/ft_synth_hist.history (user 10/28): lane-shaped 1.5 s history, ego frame; straight fallback. Pure numpy."""
import sys
from pathlib import Path

import numpy as np, pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ft = pytest.importorskip("scripts.ft_synth_hist")


def arc(R=50.0, n=400, ds=0.5):                                                    # left turn of radius R, heading 0 at s = 0
    s = np.arange(n) * ds; th = s / R
    P = np.stack([R * np.sin(th), R * (1 - np.cos(th))], 1); d = np.diff(P, axis=0)
    return P, s, np.unwrap(np.arctan2(d[:, 1], d[:, 0]))


def test_straight_road_parallel_offset_and_heading_blend():
    s = np.arange(200) * 0.5; P = np.stack([s, np.zeros_like(s)], 1); H = np.zeros(len(P) - 1)
    hx, hy, hyaw, m = ft.history(P, s, H, x=50.0, y=1.5, yaw=0.0, v=10.0, s_hint=50.0)   # 1.5 m left, aligned
    assert m == "lane" and np.allclose(hx, -10.0 * 0.1 * np.arange(15, -1, -1)) and np.allclose(hy, 0) and np.allclose(hyaw, 0)
    hx, hy, hyaw, m = ft.history(P, s, H, x=50.0, y=1.5, yaw=np.radians(10), v=10.0, s_hint=50.0)   # turned 10 deg left
    assert m == "lane" and np.allclose([hx[-1], hy[-1], hyaw[-1]], 0)
    assert np.allclose(hyaw[:5], -np.radians(10)) and abs(hy[0]) > 1.0             # first 0.5 s: parallel to the road


def test_curve_follows_centerline_shape():
    P, s, H = arc()
    th = 60.0 / 50.0; x, y = 50 * np.sin(th), 50 * (1 - np.cos(th))
    hx, hy, hyaw, m = ft.history(P, s, H, x=x, y=y, yaw=th, v=10.0, s_hint=60.0)
    assert m == "lane" and np.isclose(hyaw[0], -15.0 / 50.0, atol=0.02)           # 15 m back along R = 50 m
    chord = 2 * 50 * np.sin(15.0 / 100)                                            # 15 m of arc on R = 50 m
    assert np.hypot(hx[0], hy[0]) == pytest.approx(chord, abs=0.05) and hy[0] > 0    # on a left turn the past lies left (centre side)


def test_fallback_straight():
    s = np.arange(200) * 0.5; P = np.stack([s, np.zeros_like(s)], 1); H = np.zeros(len(P) - 1)
    for kw in (dict(y=6.0, yaw=0.0), dict(y=0.0, yaw=np.radians(40))):              # off the road / against the lane
        hx, hy, hyaw, m = ft.history(P, s, H, x=50.0, v=8.0, s_hint=50.0, **kw)
        assert m == "straight" and np.allclose(hy, 0) and np.allclose(hyaw, 0) and np.isclose(hx[0], -12.0)
