"""Discrete-token decoding of the Stage 1 monitor and the tokenizer clamp count, on the release tokenizer settings."""
import numpy as np, torch
from alpamayo_r1.action_space.discrete_action_space import DiscreteTrajectoryTokenizer

from morai_vla.data import ego_trajectory
from morai_vla.stage1_eval import OnlyTrajBins, tokens_to_xyz
from scripts.tokenizer_clip import clip_stats

ACTION = {"_target_": "alpamayo_r1.action_space.UnicycleAccelCurvatureActionSpace", "a_lambda": 1e-4, "a_ridge": 1e-4,
          "accel_bounds": [-9.8, 9.8], "accel_mean": 0.02902694707164455, "accel_std": 0.6810426736454882,
          "curvature_bounds": [-0.33, 0.33], "curvature_mean": 0.0002692167976330542, "curvature_std": 0.026148280660833106,
          "dt": 0.1, "kappa_lambda": 1e-4, "kappa_ridge": 1e-4, "n_waypoints": 64, "theta_lambda": 1e-6, "theta_ridge": 1e-8,
          "v_lambda": 1e-6, "v_ridge": 1e-4}
TOK = DiscreteTrajectoryTokenizer(ACTION, dims_min=[-10, -10], dims_max=[10, 10], num_bins=3000)
START = 151669


def traj(v0=8.0, accel=0.5, yaw_rate=0.1):
    t = np.arange(200) * 0.1
    return ego_trajectory(np.maximum(v0 + accel * (t - 10), 0), yaw_rate * t, 100)


def test_generated_ids_decode_back_to_the_trajectory():
    hx, hr, fx, fr = traj()
    ids = TOK.encode(hx, hr, fx, fr) + START                                       # what greedy generation emits at best
    xyz = tokens_to_xyz(TOK, ids, START, hx, hr)
    assert xyz.shape == fx.shape and float((xyz - fx)[..., :2].norm(dim=-1).mean()) < 0.1


def test_only_traj_bins():
    s = OnlyTrajBins(5, 3)(None, torch.arange(10.)[None])
    assert s.argmax().item() == 7 and torch.isinf(s[0, :5]).all() and torch.isinf(s[0, 8:]).all()


def test_clip_stats_counts_out_of_range_tokens():
    a = torch.zeros(4, 64, 2); a[0, :8, 0] = 12.0; a[1, 0, 1] = -10.004
    st = clip_stats(TOK, a)
    assert st["accel_tokens_clamped"] == 8 / (4 * 64) and st["curv_tokens_clamped"] == 1 / (4 * 64) and st["trajectories_with_a_clamp"] == 0.5


def test_lateral_longitudinal_split():
    from scripts.lateral_error import split_error
    gt = np.zeros((1, 2, 2)); yaw = np.array([[0.0, np.pi / 2]])                  # heading +x, then +y
    lat, lon = split_error(gt + np.array([[[0.0, 0.5], [1.0, 0.0]]]), gt, yaw)
    assert np.allclose(lat, [[0.5, -1.0]]) and np.allclose(lon, [[0.0, 0.0]], atol=1e-12)


def test_cam7_allocate_fills_from_small_strata():
    from scripts.cam7_eval import allocate
    assert allocate([1, 1, 90], 150) == [1, 1, 90]                   # fewer candidates than n: all of them
    assert allocate([79, 63, 40, 30], 150) == [40, 40, 40, 30]       # a small stratum's share goes to the others
    assert allocate([79, 63, 60, 50], 150) == [38, 38, 38, 38]       # all large: ceil(n / strata) each, as before


def test_crossing_ignores_standing_jitter():
    from scripts.signal_eval import crosses
    z, ego = {"front_axle": 3.0}, {"x": [0.0], "y": [0.0], "yaw": [0.0]}
    jitter = np.random.default_rng(0).normal(0, 0.03, (64, 2))                  # a standing prediction (7294, 10/25)
    past = np.array([[2.0, -2.0], [2.0, 2.0]])                                  # front axle (x = 3) already 1 m past it
    ahead = np.array([[5.0, -2.0], [5.0, 2.0]])
    drive = np.stack([np.linspace(0.3, 19.2, 64), np.zeros(64)], 1)
    assert not crosses(z, ego, 0, jitter, past) and not crosses(z, ego, 0, jitter, ahead) and crosses(z, ego, 0, drive, ahead)
