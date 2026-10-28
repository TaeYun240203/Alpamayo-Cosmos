"""Lane-shaped 1.5 s histories that KEEP the real speed profile, for the stop / slow-down rows of the dynamic-stop fine-tune
(dyn_mix.py sources with history real 50 / synth 50: dyn_stop virtual episodes, red approach / wait). CPU, Mac.
scripts/ft_synth_hist.py drives the whole 1.5 s at the current speed v (its back-projection s0 + v t and its steps v DT):
that erases a deceleration, the cue a stop needs (main session 10/28). Here the same lane geometry (path heading along the
way back, turned into the current heading over the last T_BLEND s; straight history when the row does not map) is walked
with the real per-step distances 0.5 (v_k + v_k+1) DT of _ego v[i-15 .. i], as morai_vla.data.ego_trajectory: only the
shape changes, the speed (step lengths) stays the real one (self-check asserts it).
Out: OUT parquet (episode, i, method, hist_x / hist_y / hist_yaw (16, ego frame at t0, oldest first), extra=False); dyn_stop
rows carry the virtual episode name (history = the original episode's).
Usage (cwd = workspace): python train/scripts/dyn_synth_hist.py R1_ROOT R2_ROOT MIX_DIR DYN_OUT OUT
"""
import sys
from pathlib import Path

import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_synth_hist import D_MAX, DT, N, S_TOL, T_BLEND, YAW_MAX, path_arrays, project   # noqa: E402


def history(P, S, H, x, y, yaw, vh, s_hint):
    """vh: real speeds at t0-1.5 .. t0 (16) -> (hist_x, hist_y, hist_yaw) ego frame at t0, oldest first, and the method."""
    t = -DT * np.arange(N - 1, -1, -1)
    step = 0.5 * (vh[1:] + vh[:-1]) * DT                                            # real distance of each 0.1 s step
    back = np.r_[np.cumsum(step[::-1])[::-1], 0.0]                                  # distance from each point to t0
    s0, d, h0 = project(P, S, x, y, s_hint)
    e = (yaw - h0 + np.pi) % (2 * np.pi) - np.pi
    lane = abs(d) <= D_MAX and abs(e) <= YAW_MAX and abs(s0 - s_hint) <= S_TOL and s0 - back[0] >= S[0]
    if lane:
        hp = np.interp(s0 - back, S[:-1], H)
        h = hp - hp[-1] + h0 + e * np.clip(1 + t / T_BLEND, 0, 1)
    else:
        h = np.full(N, yaw)
    hm = 0.5 * (h[1:] + h[:-1]); px, py = np.zeros(N), np.zeros(N); px[-1], py[-1] = x, y
    for k in range(N - 1, 0, -1):
        px[k - 1], py[k - 1] = px[k] - step[k - 1] * np.cos(hm[k - 1]), py[k] - step[k - 1] * np.sin(hm[k - 1])
    c, sn = np.cos(yaw), np.sin(yaw); dx, dy = px - x, py - y
    hx, hy = c * dx + sn * dy, -sn * dx + c * dy
    assert np.allclose(np.hypot(np.diff(hx), np.diff(hy)), np.abs(step), atol=1e-6), "synthetic history changed the speed"
    return hx, hy, h - yaw, "lane" if lane else "straight"


def rows(root, sel, P, S, H, name=None):
    out = []
    for ep, iis in sel.items():
        z = np.load(Path(root) / "_ego" / f"{ep}.npz"); v = z["v"].astype(float)
        for i in iis:
            hx, hy, hyaw, m = history(P, S, H, float(z["x"][i]), float(z["y"][i]), float(z["yaw"][i]), v[i - N + 1:i + 1], float(z["path_s"][i]))
            out.append(dict(episode=name(ep, i) if name else ep, i=int(i), method=m, hist_x=hx.astype(np.float32), hist_y=hy.astype(np.float32),
                            hist_yaw=hyaw.astype(np.float32), extra=False))
    return out


def main():
    r1, r2, mix, dyn, out = map(Path, sys.argv[1:6]); P, S, H = path_arrays()
    vi = pd.read_parquet(dyn / "dyn_index.parquet"); vi = vi[vi.split == "train"]
    res = rows(r2, vi.groupby("src_episode").i.apply(list).to_dict(), P, S, H, name=lambda ep, i: f"{ep}_dyn{i:05d}")
    red = pd.concat([pd.read_parquet(f) for f in sorted(mix.glob("red_*.parquet"))]).drop_duplicates()
    red = red[red.i % 5 == 0]                                                       # old source: its 2 Hz rows are drawn
    res += rows(r1, red.groupby("episode").i.apply(list).to_dict(), P, S, H)
    df = pd.DataFrame(res); df.to_parquet(out, index=False)
    print(f"{len(df)} histories ({int((df.method == 'lane').sum())} lane-shaped): dyn_stop {len(vi)}, red 2 Hz {len(red)}")


if __name__ == "__main__":
    main()
