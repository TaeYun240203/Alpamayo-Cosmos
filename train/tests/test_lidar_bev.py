"""morai_vla.lidar_bev on synthetic input: packet decoding, and motion compensation + the 2-scan pair on a straight line and a
circle. Plain asserts; runs under pytest, or as `python3 test_lidar_bev.py` in the Noetic chroot (numpy 1.17, no pytest)."""
import os, sys

import numpy as np

try:
    from morai_vla import lidar_bev as L
except ImportError:                                                               # chroot: no package install
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "morai_vla")); import lidar_bev as L

PK_DT, PK_PER_SCAN = 1 / 754.0, 76                                                # VLP16: 754 packets/s, ~10 Hz scans


def encode(az_deg, dist_m):
    """inverse of decode: azimuth per block (n, 12) and range per channel (n, 12, 32) -> n x 1206 packet bytes."""
    n = az_deg.shape[0]
    blk = np.zeros((n, 12, 100), np.uint8); blk[:, :, 0], blk[:, :, 1] = 0xFF, 0xEE
    a = np.round(az_deg * 100).astype(np.uint16); blk[:, :, 2], blk[:, :, 3] = a & 0xFF, a >> 8
    d = np.round(dist_m / 0.002).astype(np.uint16); ch = np.zeros((n, 12, 32, 3), np.uint8)
    ch[..., 0], ch[..., 1] = d & 0xFF, d >> 8; blk[:, :, 4:] = ch.reshape(n, 12, 96)
    return np.concatenate([blk.reshape(n, 1200), np.zeros((n, 6), np.uint8)], 1).tobytes()


def true_pose(t, v, r):
    """exact pose at time t of constant (v, r) motion from the origin."""
    if r == 0: return v * t, 0.0 * t, 0.0 * t
    return v / r * np.sin(r * t), v / r * (1 - np.cos(r * t)), r * t


def test_decode_roundtrip():
    rng = np.random.RandomState(0)
    az = np.round(rng.uniform(0, 359.9, (5, 12)), 2); dist = np.round(rng.uniform(1, 90, (5, 12, 32)) / 0.002) * 0.002
    q, pk = L.decode(encode(az, dist))
    a = np.radians(az)[..., None] + np.where(np.arange(32) >= 16, np.radians(0.1), 0.0); w = np.tile(L.VERT, 2)
    want = np.stack([dist * np.cos(w) * np.cos(a), -dist * np.cos(w) * np.sin(a), dist * np.sin(w)], -1).reshape(-1, 3)
    assert q.shape == (5 * 12 * 32, 3) and (pk == np.repeat(np.arange(5), 384)).all()
    assert np.abs(q - want).max() < 1e-9


def test_compensation_line_and_circle():
    rng = np.random.RandomState(1)
    for v, r in ((15.0, 0.0), (10.0, 0.6)):
        t_all = np.arange(2 * PK_PER_SCAN) * PK_DT
        xr, yr, hr = true_pose(t_all[-1], v, r)                                   # static world points around the 2nd scan
        loc = np.stack([rng.uniform(2, 50, 400), rng.uniform(-15, 15, 400), rng.uniform(-0.5, 2, 400)], 1)
        W = np.stack([xr + np.cos(hr) * loc[:, 0] - np.sin(hr) * loc[:, 1], yr + np.sin(hr) * loc[:, 0] + np.cos(hr) * loc[:, 1], loc[:, 2]], 1)
        pk = np.arange(len(W)) % PK_PER_SCAN                                      # each point seen once per scan, at packet pk
        odom, scans, err_static = L.Odometry(), [], 0.0
        for s in range(2):
            t = t_all[s * PK_PER_SCAN:(s + 1) * PK_PER_SCAN]
            px, py, ph = true_pose(t[pk], v, r)                                   # base_link points at each packet time
            q = np.stack([*L.to_frame(W[:, 0], W[:, 1], 0.0, (px, py, ph))[:2], W[:, 2]], 1)
            x, y, h = odom.advance(t, np.full(len(t), v), np.full(len(t), r))
            pose = (x[-1], y[-1], h[-1])
            assert np.allclose(pose, true_pose(t[-1], v, r), atol=1e-9)
            P = L.compensate(q, pk, *L.to_frame(x, y, h, pose))
            want = np.stack([*L.to_frame(W[:, 0], W[:, 1], 0.0, true_pose(t[-1], v, r))[:2], W[:, 2]], 1)
            assert np.abs(P - want).max() < 1e-6, (v, r, np.abs(P - want).max())
            z = np.zeros(PK_PER_SCAN)
            err_static = max(err_static, np.abs(L.compensate(q, pk, z, z, z) - want)[:, :2].max())
            hgt, den, n = L.rasterize(P); scans.append((np.stack([hgt, den]), pose))
        assert err_static > 1.0                                                   # without compensation points smear by ~v*0.1 s
        new, old = scans[1], scans[0]

        def near(pair):
            occ, war = pair[0] > 0, np.pad(pair[1] > 0, 1)                        # new cell has a warped-old cell within 1 px
            hit = np.zeros_like(occ)
            for dr in (0, 1, 2):
                for dc in (0, 1, 2): hit |= war[dr:dr + L.H, dc:dc + L.W]
            return (occ & hit).sum() / occ.sum()
        assert near(L.pack_bev(new[0], old[0], new[1], old[1])) > 0.99
        assert near(L.pack_bev(new[0], old[0], new[1], new[1])) < 0.5            # ignoring the motion breaks the alignment


if __name__ == "__main__":
    test_decode_roundtrip(); test_compensation_line_and_circle(); print("ok")
