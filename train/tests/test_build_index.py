"""build_index.traj_ok_on_grid: a breaker (the scorer's bad label rows) masks every grid row whose 6.4 s tau window holds it, at any sim rate."""
import numpy as np, pandas as pd

from morai_vla.build_index import traj_ok_on_grid


def test_bad_label_masks_its_window_at_any_rate():
    for rho in (0.8, 1.0, 1.01, 1.03, 1.05):                # R1 ran at 0.7-0.95x, R2 at ~1x (rho clipped to 1.05)
        bag_ns = np.arange(0, 40e9, 1e8)                    # 10 Hz labels over 40 s wall
        G = {"bag_ns": bag_ns, "tau": bag_ns / 1e9 * rho}
        t = bag_ns / 1e9
        ok = ~((t <= 20) & (t + 6.4 >= 20))                 # the scorer's labels for a breaker at wall 20 s
        L = pd.DataFrame({"bag_time_ns": bag_ns.astype(np.int64), "traj_ok": ok})
        got = traj_ok_on_grid(G, L)
        holds = (G["tau"] <= 20 * rho) & (G["tau"] + 6.4 >= 20 * rho)
        assert not got[holds].any(), rho


def test_glitch_steps_net_keeps_real_braking():
    """--glitch net (R2, user 10/28): a stalled-then-caught-up hard brake stays, a spike, the start transient and an
    impossible change stay glitches; --glitch step (R1) flags every |dv| > 1.5 m/s per 0.1 s."""
    from morai_vla.build_index import glitch_steps
    tau = np.arange(100) * 0.1
    brake = np.r_[np.full(50, 10.0), 10, 10, 6.5, 6.5, 3.0, 3.0, 1.0, np.zeros(43)]   # ~8 m/s^2 in 2-row steps
    assert glitch_steps(brake, tau, "step").any() and not glitch_steps(brake, tau, "net").any()
    spike = np.full(100, 8.0); spike[60] = 5.0                                        # one bad sample
    assert glitch_steps(spike, tau, "net").sum() == 2
    start = np.r_[np.zeros(15), np.full(85, 11.1)]                                    # 40 kph applied at 1.5 s
    assert glitch_steps(start, tau, "net").any()
    wall = np.r_[np.full(50, 14.0), 7.0, 1.0, np.zeros(48)]                           # 14 m/s to 0 in 0.2 s
    assert glitch_steps(wall, tau, "net").any()
