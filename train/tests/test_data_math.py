"""tau selection, BEV packing/warp, ego trajectory frames, (a, kappa) label normalization round trip. No data files needed."""
import numpy as np, pytest, torch

from morai_vla.data import ego_trajectory, frame_targets, lidar_dropped, nearest, pack_bev, select_distinct
from morai_vla.lidar_bev import DENSITY_GAIN, RES as BEV_RES, XF as BEV_XF, YW as BEV_YW

# Alpamayo-1.5-10B release config.json (action_space_cfg / traj_tokenizer_cfg)
AS = dict(accel_mean=0.02902694707164455, accel_std=0.6810426736454882, curvature_mean=0.0002692167976330542,
          curvature_std=0.026148280660833106, a_lambda=1e-4, a_ridge=1e-4, kappa_lambda=1e-4, kappa_ridge=1e-4,
          theta_lambda=1e-6, theta_ridge=1e-8, v_lambda=1e-6, v_ridge=1e-4, dt=0.1, n_waypoints=64)


def test_frame_targets_nominal_and_jitter():
    np.testing.assert_allclose(frame_targets(10.0), [9.7, 9.8, 9.9, 10.0])
    rng = np.random.default_rng(0)
    for _ in range(200):
        t = frame_targets(5.0, rng)
        gaps = np.diff(t)
        assert t[-1] == 5.0 and (gaps >= 0.08 - 1e-9).all() and (gaps <= 0.12 + 1e-9).all()


def test_nearest():
    tau = np.array([0.0, 0.11, 0.19, 0.32, 0.4])
    assert list(nearest(tau, np.array([-1.0, 0.05, 0.06, 0.2, 0.36, 9.0]))) == [0, 0, 1, 2, 3, 4]


def test_select_distinct_scans_and_dropped_frames():
    scan_tau = np.arange(0, 3, 1 / 8.5)                     # 8.5 Hz scans vs the 0.1 s grid
    for t0 in np.arange(0.5, 2.5, 0.013):
        k = select_distinct(scan_tau, frame_targets(t0))
        assert (np.diff(k) > 0).all(), (t0, k)
        assert abs(scan_tau[k[-1]] - t0) <= 0.5 / 8.5 + 1e-9  # newest scan is the nearest one
    cam = np.array([0.0, 0.1, 0.2, 0.3, 0.6, 0.7])          # frames at 0.4, 0.5 dropped (seen on 10 % of rows)
    k = select_distinct(cam, frame_targets(0.64))           # targets .34 .44 .54 .64
    assert list(k) == [1, 2, 3, 4] and list(nearest(cam, frame_targets(0.64))) == [3, 3, 4, 4]


def test_lidar_dropout_is_a_function_of_the_key():
    keys = [("run1021a-w1-001", i) for i in range(4000)]
    a = [lidar_dropped(e, i, 0.15) for e, i in keys]
    assert a == [lidar_dropped(e, i, 0.15) for e, i in keys]
    assert 0.12 < np.mean(a) < 0.18
    assert not any(lidar_dropped(e, i, 0.0) for e, i in keys)


def _cell(x, y):
    return int(np.floor((BEV_XF - x) / BEV_RES)), int(np.floor((BEV_YW - y) / BEV_RES))


def test_bev_warp_translation_and_rotation():
    h_old = np.zeros((320, 160), np.uint8); h_old[200, 80] = 77
    x, y = BEV_XF - 200.5 * BEV_RES, BEV_YW - 80.5 * BEV_RES                    # cell centre (9.875, -0.125) in the old frame
    h_new = np.zeros_like(h_old); h_new[5, 5] = 9
    d_new = np.zeros_like(h_old); d_new[5, 5] = 3; d_new[6, 6] = 200
    old = np.stack([h_old, np.zeros_like(h_old)]); new = np.stack([h_new, d_new])
    # newer pose 2 m further along +x: the point is 2 m closer
    b = pack_bev(new, old, (2.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert b.shape == (3, 320, 160) and b.dtype == np.uint8
    assert b[1][_cell(x - 2.0, y)] == 77 and (b[1] > 0).sum() == 1
    assert b[0][5, 5] == 9 and b[2][5, 5] == 3 * DENSITY_GAIN and b[2][6, 6] == 255
    # newer pose turned +90 deg (left) in place: a point ahead ends up on the right
    b = pack_bev(new, old, (0.0, 0.0, np.pi / 2), (0.0, 0.0, 0.0))
    assert b[1][_cell(y, -x)] == 77
    # both frames moved identically: nothing moves
    b = pack_bev(new, old, (5.0, -3.0, 0.7), (5.0, -3.0, 0.7))
    assert b[1][200, 80] == 77


def _arc(n=90, v0=8.0, a=0.4, kappa=0.02, yaw0=0.3):
    """exact constant-curvature arc: speed v0 + a t, yaw = yaw0 + kappa s; positions from the closed form."""
    t = 0.1 * np.arange(n); v = v0 + a * t; s = v0 * t + 0.5 * a * t ** 2
    yaw = yaw0 + kappa * s
    x = 100 + (np.sin(yaw) - np.sin(yaw0)) / kappa; y = -50 - (np.cos(yaw) - np.cos(yaw0)) / kappa
    return x, y, yaw, v


def test_ego_trajectory_frame():
    x, y, yaw, v = _arc()
    hx, hr, fx, fr = ego_trajectory(v, yaw, 20)
    assert hx.shape == (1, 16, 3) and hr.shape == (1, 16, 3, 3) and fx.shape == (1, 64, 3) and fr.shape == (1, 64, 3, 3)
    assert torch.allclose(hx[0, -1], torch.zeros(3)) and torch.allclose(hr[0, -1], torch.eye(3))
    assert fx[0, 0, 0] > 0.7 and abs(fx[0, 0, 1]) < 0.05       # first future point ~0.8 m straight ahead
    assert fx[0, -1, 1] > 5                                     # kappa > 0 bends left (+y)
    j = np.arange(21, 85); c, s_ = np.cos(yaw[20]), np.sin(yaw[20])               # integrated == exact arc in the t0 frame
    ex = np.stack([c * (x[j] - x[20]) + s_ * (y[j] - y[20]), -s_ * (x[j] - x[20]) + c * (y[j] - y[20])], -1)
    assert np.abs(fx[0, :, :2].numpy() - ex).max() < 0.05


def test_label_normalization_round_trip():
    from alpamayo_r1.action_space import UnicycleAccelCurvatureActionSpace
    from alpamayo_r1.action_space.discrete_action_space import DiscreteTrajectoryTokenizer
    space = UnicycleAccelCurvatureActionSpace(**AS)
    x, y, yaw, v = _arc()
    hx, hr, fx, fr = (t.double() for t in ego_trajectory(v, yaw, 20))
    act = space.traj_to_action(hx, hr, fx, fr)                 # (1, 64, 2) normalised (a, kappa)
    a = act[0, 5:-5, 0] * AS["accel_std"] + AS["accel_mean"]; k = act[0, 5:-5, 1] * AS["curvature_std"] + AS["curvature_mean"]
    assert abs(a.mean() - 0.4) < 0.05 and abs(k.mean() - 0.02) < 0.002
    rx, _ = space.action_to_traj(act, hx, hr)
    err = (rx - fx)[..., :2].norm(dim=-1)
    assert err.mean() < 0.05 and err.max() < 0.2, (err.mean(), err.max())
    tok = DiscreteTrajectoryTokenizer(action_space_cfg={"_target_": "alpamayo_r1.action_space.UnicycleAccelCurvatureActionSpace", **AS},
                                      dims_min=[-10, -10], dims_max=[10, 10], num_bins=3000)
    ids = tok.encode(hx, hr, fx, fr)                          # (n_traj=1, T, ...) as the recipe flattens it
    assert ids.shape == (1, 128) and ids.min() >= 0 and ids.max() < 3000
    dx, _, _ = tok.decode(hx, hr, ids)
    assert (dx - fx)[..., :2].norm(dim=-1).max() < 0.3


def test_episode_rejects_partial_copy(tmp_path):
    from morai_vla.data import Episode
    d = tmp_path / "ep"; d.mkdir()
    np.savez(d / "cams_index.npz", cam=np.zeros(2, np.uint8), tau=np.array([0.0, 0.1]), offset=np.array([0, 100]), length=np.array([100, 100]))
    np.savez(d / "lidar_index.npz", tau=np.array([0.0]), offset=np.array([0]), length=np.array([50]), pose=np.zeros((1, 3)))
    np.savez(tmp_path / "ego.npz", v=np.zeros(3), yaw=np.zeros(3), tau=np.arange(3) * 0.1)
    (d / "lidar.bin").write_bytes(b"\0" * 50)
    (d / "cams.bin").write_bytes(b"\0" * 150)                  # index needs 200 bytes: still copying
    with pytest.raises(ValueError):
        Episode(d, tmp_path / "ego.npz")
    (d / "cams.bin").write_bytes(b"\0" * 200)
    Episode(d, tmp_path / "ego.npz")
