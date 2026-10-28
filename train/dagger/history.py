"""History dependence of each published LoRA version (user 10/28: "러너가 LoRA를 공개할 때마다 고정 val에서 이력 의존도(이력
0·바꿔치기 때 예측 이동량)를 기록하게 해"), as scripts/ft_measure.py history. Fixed rows (configs/dagger.yaml
dagger.history_sets: MoraiDataset first n rows, seed 1, augment off, LiDAR on, generation prompts); per row the same
initial noise x0 (seed 1000 k) and deterministic Euler (dagger.history_steps, the engine's 10) under three history-token
inputs, the (a, kappa) always integrated from the row's real history (as the fine-tune's dropout and the deployment):
  base  the real history       zero  16 points at the origin, identity rotations (standing still)
  swap  the history of row (k + n/2) % n of the same set
Per condition vs base: move = mean distance between the two 6.4 s trajectories, dlat_2s / dlat_3s = |lateral| difference
at 2 / 3 s in the GT path frame (scripts/lateral_error.split_error). Recorded only, never blocks a publish.
"""
import numpy as np, torch
from types import SimpleNamespace

from morai_vla.data import MoraiDataset, ego_trajectory
from morai_vla.processor import collate_fn, get_preprocess_fn
from rl.episodes import DROP
from rl.policy import Policy
from scripts.lateral_error import split_error

AT = {"2s": 19, "3s": 29}


def sets(spec):
    """{name: [(MoraiDataset, row)]} of the configured parts ({n, MoraiDataset kwargs}); empty sets are left out."""
    out = {}
    for name, parts in spec.items():
        rows = []
        for p in parts:
            p = dict(p); n = p.pop("n")
            ds = MoraiDataset(**p, max_samples=n, augment=False, seed=1); ds.rows = ds.rows.iloc[:n].reset_index(drop=True)
            rows += [(ds, k) for k in range(len(ds))]
        if rows: out[name] = rows
    return out


def history_of(ds, k):
    r = ds.rows.iloc[k]; ep = ds._episode(r.episode)
    return ego_trajectory(ep.v, ep.yaw, int(r.i))[:2]                           # (1, 16, 3), (1, 16, 3, 3)


@torch.no_grad()
def plans(m, lora, sample, hists, x0, steps, dev):
    """xy (64, 2) per history-token input in hists; one prompt, the row's real history for the integration."""
    d = dict(sample); d["cot"] = None; d["tokenized_data"] = get_preprocess_fn(m.config, generation_mode=True)(data=d)
    b = collate_fn([{k: v for k, v in d.items() if k not in DROP}], model_config=m.config)
    ns = SimpleNamespace(model=m, device=dev, lora=lora)
    real = (b["ego_history_xyz"][:, 0].to(dev).float(), b["ego_history_rot"][:, 0].to(dev).float())
    t = torch.linspace(0, 1, steps + 1, device=dev); out = []
    for hx, hr in hists:
        with torch.autocast("cuda", torch.bfloat16, enabled=dev.type == "cuda"):
            pre = Policy.prefill(ns, dict(b, ego_history_xyz=hx[None], ego_history_rot=hr[None]), 1)
            x = x0.to(dev)
            for k in range(steps):
                x = x + (t[k + 1] - t[k]) * Policy.velocity(ns, pre, x, t[k].expand(1))
        out.append(m.action_space.action_to_traj(x.float(), *real)[0][0, :, :2].cpu().numpy())
    return out


def measure(m, lora, hist_sets, steps, dev, rank=0, world=1):
    """this rank's rows (k::world over every set) -> [{set, key, zero_move, ...}]."""
    m.eval(); out = []
    todo = [(name, j) for name, rows in hist_sets.items() for j in range(len(rows))][rank::world]
    for name, j in todo:
        rows = hist_sets[name]; ds, k = rows[j]; ods, ok = rows[(j + len(rows) // 2) % len(rows)]
        s = ds.sample(k)
        own = (s["ego_history_xyz"], s["ego_history_rot"])
        zero = (torch.zeros_like(own[0]), torch.eye(3).expand_as(own[1]).contiguous())
        x0 = torch.randn(1, 64, 2, generator=torch.Generator().manual_seed(1000 * j))
        base, pz, ps = plans(m, lora, s, [own, zero, history_of(ods, ok)], x0, steps, dev)
        r = s["ego_future_rot"][0].numpy(); yaw = np.arctan2(r[:, 1, 0], r[:, 0, 0])
        row = {"set": name, "key": s["key"], "swap_from": f"{ods.rows.episode.iloc[ok]}/{int(ods.rows.i.iloc[ok])}"}
        for c, p in (("zero", pz), ("swap", ps)):
            lat, _ = split_error(p[None], base[None], yaw[None])
            row.update({f"{c}_move": float(np.linalg.norm(p - base, axis=-1).mean()),
                        **{f"{c}_dlat_{h}": float(abs(lat[0, i])) for h, i in AT.items()}})
        out.append(row)
    return out


def summary(rows):
    """per set: n and mean / p90 of every measure."""
    keys = [k for k in rows[0] if k not in ("set", "key", "swap_from")] if rows else []
    by = {}
    for r in rows: by.setdefault(r["set"], []).append(r)
    return {s: {"n": len(rs), **{k: round(float(np.mean([r[k] for r in rs])), 4) for k in keys},
                **{f"{k}_p90": round(float(np.percentile([r[k] for r in rs], 90)), 4) for k in keys if k.endswith("move")}}
            for s, rs in sorted(by.items())}
