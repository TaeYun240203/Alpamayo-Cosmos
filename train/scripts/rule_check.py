"""Rule check of T*'s soft-target trajectories (MODEL_PLAN 4.5: velocity KD only on frames where T* drives legally)
-> rule_ok parquet (episode, i, rule_ok) for MoraiDataset(rule_ok=). A row passes when all K of its trajectories pass
the competition rules its labels were judged with (tools/scorer thresholds.yaml, packed by morai_vla.rule_context):
- speed: no 0.1 s step above the limit outside the exempt route span;
- lane: no wheel on a judged solid line (paint + tyre half width) for a whole lane period (3 s), outside GPS blackout;
- signal: the front axle crosses no judged stop line while that future row's GT light state is red or unknown;
- obstacles: the body box overlaps no static scenario obstacle box.
Trajectories are in the t0 ego frame; t0 pose, route progress, GT signal and blackout are the episode's _ego rows.
Not checked: pedestrians and NPC vehicles (no per-row trajectories recorded), drive-inoperable.
Usage (cwd train/): python scripts/rule_check.py --parts <soft>/parts --root <step-0 root> --out rule_ok.parquet
       python scripts/rule_check.py --gt 3000 --root <step-0 root> --out gt.parquet   (GT futures of train rows: calibration)
Writes <out> and <out>.json (pass rates per rule and cell)."""
import argparse, glob, json, sys
from functools import lru_cache
from pathlib import Path

import numpy as np, pandas as pd
from scipy.spatial import cKDTree

CTX = Path(__file__).resolve().parents[1] / "assets/rule_context.npz"
RULES = ("speed", "lane", "signal", "obstacle")
RED, UNKNOWN = 1, 0                                                    # tools/step0 ego_tables SIG: red 1, yellow 2, green 3


class Rules:
    def __init__(self, path=CTX):
        z = np.load(path); self.z = z
        a, b = z["seg_a"], z["seg_b"]                                  # judged solid lines, densified to 5 cm for a KD tree
        n = np.maximum(np.ceil(np.hypot(*(b - a).T) / 0.05).astype(int), 1)
        k = np.repeat(np.arange(len(a)), n + 1); u = np.concatenate([np.arange(m + 1) / m for m in n])
        self.tree, self.reach = cKDTree(a[k] + u[:, None] * (b - a)[k]), np.r_[z["seg_hw"][k] + float(z["tire_hw"]), -1.0]
        P, off = z["stop_pts"], z["stop_off"]
        self.stop = np.concatenate([np.stack([P[o:e - 1], P[o + 1:e]], 1) for o, e in zip(off[:-1], off[1:])])   # (S, 2, 2)
        self.ep_index = {e: j for j, e in enumerate(z["episodes"])}
        self.obs = z["obs"]; self.period = int(round(float(z["lane_period_s"]) / 0.1))

    def check(self, ego, rows, traj):
        """ego: _ego arrays of one episode; rows (N,); traj (N, K, 64, 3) x, y, yaw in each row's t0 frame
        -> fails (N, K, 4) bool in RULES order."""
        z = self.z; N, K, T, _ = traj.shape
        x0, y0, h0, s0 = (ego[k][rows][:, None, None] for k in ("x", "y", "yaw", "path_s"))
        tx, ty = np.concatenate([np.zeros((N, K, 1, 2)), traj[..., :2]], 2).transpose(3, 0, 1, 2)   # k = 0 is t0
        X, Y = x0 + np.cos(h0) * tx - np.sin(h0) * ty, y0 + np.sin(h0) * tx + np.cos(h0) * ty
        H = h0 + np.concatenate([np.zeros((N, K, 1)), traj[..., 2]], 2)
        fut = rows[:, None] + np.arange(1, T + 1)[None]                  # _ego rows of the 64 future points
        # speed
        step = np.hypot(np.diff(X, axis=2), np.diff(Y, axis=2))
        s = s0 + np.cumsum(step, 2); lo, hi = z["exempt_s"]
        speed = ((step / 0.1 * 3.6 > float(z["speed_limit_kph"])) & ~((s >= lo) & (s <= hi))).any(2)
        # lane: any wheel within reach of a judged solid line, a whole period long, outside the GPS blackout
        wx, wy = z["wheels"].T
        c, sn = np.cos(H[..., 1:, None]), np.sin(H[..., 1:, None])
        WX, WY = X[..., 1:, None] + c * wx - sn * wy, Y[..., 1:, None] + sn * wx + c * wy
        d, j = self.tree.query(np.stack([WX.ravel(), WY.ravel()], 1), distance_upper_bound=1.0, workers=-1)
        touch = (d < self.reach[np.minimum(j, len(self.reach) - 1)]).reshape(WX.shape).any(-1) & ~ego["gps_blackout"][fut][:, None]
        run = np.zeros(touch.shape[:2], int); best = np.zeros_like(run)
        for t in range(T):
            run = np.where(touch[..., t], run + 1, 0); best = np.maximum(best, run)
        lane = best >= self.period
        # signal: front-axle path vs judged stop lines; the light state of the row where it crosses (scorer convention)
        FX, FY = X + float(z["front_axle"]) * np.cos(H), Y + float(z["front_axle"]) * np.sin(H)
        p, q = np.stack([FX[..., :-1], FY[..., :-1]], -1)[..., None, :], np.stack([FX[..., 1:], FY[..., 1:]], -1)[..., None, :]
        a, b = self.stop[:, 0], self.stop[:, 1]
        cross = lambda o, u, v: (u[..., 0] - o[..., 0]) * (v[..., 1] - o[..., 1]) - (u[..., 1] - o[..., 1]) * (v[..., 0] - o[..., 0])
        d1, d2 = cross(a, b, p), cross(a, b, q)
        hit = (np.sign(d1) != np.sign(d2)) & (d1 != 0) & (cross(p, q, a) * cross(p, q, b) < 0)   # (N, K, T, S); a point exactly on
                                                                                                  # the line counts once
        u = np.where(hit, d1 / np.where(hit, d1 - d2, 1), 0).max(-1)
        row = np.minimum(rows[:, None, None] + np.arange(T)[None, None] + (u >= 0.5), len(ego["signal"]) - 1)
        state = ego["signal"][row]
        signal = (hit.any(-1) & ((state == RED) | (state == UNKNOWN))).any(2)
        return np.stack([speed, lane, signal, self.obstacle(ego, X, Y, H)], -1)

    def obstacle(self, ego, X, Y, H):
        """body box vs the episode's static obstacle boxes (separating axis test), any pose."""
        j = self.ep_index.get(ego["episode"]); o = self.obs[self.obs[:, 0] == j] if j is not None else self.obs[:0]
        if not len(o): return np.zeros(X.shape[:2], bool)
        rear, front, hw = self.z["body"]
        cx, cy = X + (rear + front) / 2 * np.cos(H), Y + (rear + front) / 2 * np.sin(H)
        e1 = np.array([(front - rear) / 2, hw]); hit = np.zeros(X.shape, bool)
        for ox, oy, oh, ol, ow in o[:, 1:]:
            ax = [np.stack([np.cos(H), np.sin(H)], -1), np.stack([-np.sin(H), np.cos(H)], -1),
                  np.broadcast_to([np.cos(oh), np.sin(oh)], X.shape + (2,)), np.broadcast_to([-np.sin(oh), np.cos(oh)], X.shape + (2,))]
            dc = np.stack([ox - cx, oy - cy], -1); sep = np.zeros(X.shape, bool)
            for n in ax:
                r1 = e1[0] * np.abs((ax[0] * n).sum(-1)) + e1[1] * np.abs((ax[1] * n).sum(-1))
                r2 = ol * np.abs((ax[2] * n).sum(-1)) + ow * np.abs((ax[3] * n).sum(-1))
                sep |= np.abs((dc * n).sum(-1)) > r1 + r2
            hit |= ~sep
        return hit.any(2)


@lru_cache(maxsize=512)
def ego_of(root, ep):
    z = dict(np.load(Path(root) / "_ego" / f"{ep}.npz")); z["episode"] = ep
    return z


def run(rules, root, keys, traj, cells):
    """keys 'ep/i', traj (N, K, 64, 3) -> DataFrame episode, i, cell, rule_ok, fail_<rule>."""
    df = pd.DataFrame({"key": keys, "cell": cells}); df["episode"] = df.key.str.rsplit("/", n=1).str[0]; df["i"] = df.key.str.rsplit("/", n=1).str[1].astype(int)
    fails = np.zeros((len(df), traj.shape[1], len(RULES)), bool)
    for ep, g in df.groupby("episode"):
        fails[g.index] = rules.check(ego_of(root, ep), g.i.to_numpy(), traj[g.index])
    for r, name in enumerate(RULES): df[f"fail_{name}"] = fails[..., r].any(1)
    df["rule_ok"] = ~fails.any((1, 2))
    return df.drop(columns="key")


def gt_traj(root, n, seed=0):
    """GT futures of n random usable train rows (index), as (keys, (n, 1, 64, 3), cells): calibration of the rules."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from morai_vla.data import ego_trajectory
    idx = pd.read_parquet(Path(root) / "_index/index.parquet", columns=["episode", "i", "cell", "split"])
    idx = idx[idx.split == "train"]
    have = {e for e in idx.episode.unique() if (Path(root) / "_ego" / f"{e}.npz").exists()}
    idx = idx[idx.episode.isin(have)].sample(n, random_state=seed)
    out = []
    for ep, i in zip(idx.episode, idx.i):
        z = ego_of(root, ep); _, _, fx, fr = ego_trajectory(z["v"].astype(np.float64), z["yaw"], int(i))
        out.append(np.concatenate([fx[0, :, :2].numpy(), np.arctan2(fr[0, :, 1, 0], fr[0, :, 0, 0]).numpy()[:, None]], 1))
    return [f"{e}/{i}" for e, i in zip(idx.episode, idx.i)], np.stack(out)[:, None], idx.cell.astype(str).to_numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--parts"); ap.add_argument("--gt", type=int, default=0); ap.add_argument("--context", default=str(CTX))
    a = ap.parse_args(); rules = Rules(a.context)
    if a.gt:
        df = run(rules, a.root, *gt_traj(a.root, a.gt))
    else:
        dfs = []
        for f in sorted(glob.glob(f"{a.parts}/part-*.npz")):
            z = np.load(f); dfs.append(run(rules, a.root, z["key"], z["xyyaw"].astype(np.float64), z["cell"]))
        df = pd.concat(dfs, ignore_index=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df[["episode", "i", "rule_ok"]].to_parquet(a.out)
    rate = lambda d: {"n": len(d), "rule_ok": round(float(d.rule_ok.mean()), 4)} | {f"fail_{r}": round(float(d[f"fail_{r}"].mean()), 4) for r in RULES}
    summary = rate(df) | {"by_cell": {c: rate(g) for c, g in df.groupby("cell")}}
    json.dump(summary, open(f"{a.out}.json", "w"), indent=1); print(json.dumps(summary))
    sys.exit(len(df) == 0)


if __name__ == "__main__":
    main()
