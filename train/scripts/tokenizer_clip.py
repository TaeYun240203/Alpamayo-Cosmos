"""How often the discrete trajectory tokenizer clamps MORAI futures: normalized (a, kappa) are quantized on [-10, 10] with
3000 bins (release config), and MORAI's normalized accel spread is about twice PAI's. Train rows on the 2 Hz grid
(frame_stride 5 as in training, no cell downsampling), trajectories as the dataset builds them (ego_trajectory).
Usage (cwd train/): python scripts/tokenizer_clip.py <release dir> <step-0 root> [--stride 5] [--out logs/tokenizer_clip.json]"""
import argparse, json

import hydra.utils as hyu, numpy as np, pandas as pd, torch

from morai_vla.data import ego_trajectory
from morai_vla.model import load_config


def clip_stats(tokenizer, action):
    """action (N, T, 2) normalized as the tokenizer sees it -> clamped-token fractions (encode() rounds, then clamps)."""
    lo, hi = torch.tensor(tokenizer.dims_min), torch.tensor(tokenizer.dims_max)
    raw = ((action - lo) / (hi - lo) * (tokenizer.num_bins - 1)).round()
    clamped = (raw < 0) | (raw > tokenizer.num_bins - 1)
    return {"tokens_clamped": float(clamped.float().mean()), "accel_tokens_clamped": float(clamped[..., 0].float().mean()),
            "curv_tokens_clamped": float(clamped[..., 1].float().mean()), "trajectories_with_a_clamp": float(clamped.flatten(1).any(1).float().mean()),
            "accel_std": float(action[..., 0].std()), "curv_std": float(action[..., 1].std()),
            "accel_abs_p99_9": float(np.percentile(action[..., 0].abs().numpy(), 99.9))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("release"); ap.add_argument("root")
    ap.add_argument("--stride", type=int, default=5); ap.add_argument("--out", default="logs/tokenizer_clip.json")
    a = ap.parse_args()
    tok = hyu.instantiate(load_config(a.release, "Qwen/Qwen3-VL-8B-Instruct").traj_tokenizer_cfg)
    df = pd.read_parquet(f"{a.root}/_index/index.parquet", columns=["episode", "i", "split"])
    df = df[(df.split == "train") & (df.i % a.stride == 0)]
    acts = []
    for ep, g in df.groupby("episode", observed=True):
        try: z = np.load(f"{a.root}/_ego/{ep}.npz")
        except FileNotFoundError: continue
        v, yaw = z["v"].astype(np.float64), z["yaw"]
        tr = [ego_trajectory(v, yaw, int(i)) for i in g.i]
        hx, hr, fx, fr = (torch.cat([t[n] for t in tr]) for n in range(4))
        with torch.no_grad(): acts.append(tok.action_space.traj_to_action(hx, hr, fx, fr).float())
    action = torch.cat(acts)
    res = {"rows": len(action), "bins": tok.num_bins, "range": [list(tok.dims_min), list(tok.dims_max)], **clip_stats(tok, action)}
    json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res, indent=1))
