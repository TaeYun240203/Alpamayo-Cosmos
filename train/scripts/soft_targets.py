"""MODEL_PLAN 4 step 4: T* soft targets. BF16, no CoC (the prefix ends in <traj_future_start>), one prefill per sample for:
- K trajectories, Euler 10 (input of the rule check that gates the student's velocity KD); torch seed per batch is stored;
- the velocity field at K fixed points (x_t, t) on the GT interpolant x_t = t x + (1 - t) noise (x = the GT (a, kappa)
  action), t drawn like training (0.999 (1 - Beta(1.5, 1))), noise and t from a per-sample seed (stored).
Rows, tau frames (nominal spacing: augment off) and LiDAR mask are those of the student's dataset config.
Usage (cwd train/): [torchrun --nproc_per_node 8] python scripts/soft_targets.py --config-name student1 paths.teacher=<T*> \
    data.root=<share> +out=<dir> [+split=train] [+batch=8] [+k=6] [+part=2048] [data.train_dataset.max_samples=N]
Writes <out>/part-<rank>-<n>.npz (arrays per sample: key, cell, lidar_dropped, traj_seed, seed, t, v, xyyaw); finished
parts are skipped on restart. One H100: ~0.35 s/sample at batch 4-8 (README)."""
import os, time, zlib
from pathlib import Path

import hydra, hydra.utils as hyu, numpy as np, torch
from omegaconf import OmegaConf

from morai_vla.model import TeacherVLA
from morai_vla.processor import collate_fn


def fixed_points(key, x, k):
    """(t (k,), noise (k, 64, 2), seed) for one sample: the stored seed reproduces both."""
    seed = zlib.crc32(f"soft/{key}".encode())
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(k, *x.shape, generator=g)
    t = 0.999 * (1 - torch.rand(k, generator=g) ** (2 / 3))                      # inverse CDF of Beta(1.5, 1) (FlowMatching)
    return t, noise, seed


@hydra.main(version_base=None, config_path="../configs", config_name="student1")
def main(cfg):
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(dev)
    out = Path(cfg.out); out.mkdir(parents=True, exist_ok=True)
    B, K, P = cfg.get("batch", 8), cfg.get("k", 6), cfg.get("part", 2048)
    T = TeacherVLA.build(cfg.paths.teacher, vlm_name_or_path=cfg.model.teacher_vlm, attn_implementation="sdpa", device_map={"": dev.index}).eval()
    dcfg = OmegaConf.to_container(cfg.data.train_dataset, resolve=True)
    dcfg.update(augment=False, split=cfg.get("split", dcfg["split"]),
                vla_preprocess_args={"_target_": "morai_vla.processor.get_preprocess_fn", "generation_mode": True})
    ds = hyu.instantiate(dcfg, _convert_="partial", model_config=T.config)
    mine = list(range(rank, len(ds), world))
    for j in range(0, len(mine), P):
        f = out / f"part-{rank:02d}-{j // P:05d}.npz"
        if f.exists(): continue
        rows, t0 = [], time.time()
        for b in range(j, min(j + P, len(mine)), B):
            items = [ds[i] for i in mine[b:b + B]]
            batch = collate_fn(items, model_config=T.config)
            gpu = lambda x: {k: gpu(v) if isinstance(v, dict) else v.to(dev) if torch.is_tensor(v) else v for k, v in x.items()}
            batch = gpu(batch)
            x = T.action_space.traj_to_action(traj_history_xyz=batch["ego_history_xyz"], traj_history_rot=batch["ego_history_rot"],
                                              traj_future_xyz=batch["ego_future_xyz"], traj_future_rot=batch["ego_future_rot"]).reshape(len(items), -1, 2)
            pts = [fixed_points(s["key"], x[0], K) for s in items]
            tt = torch.stack([p[0] for p in pts]).to(dev); noise = torch.stack([p[1] for p in pts]).to(dev)
            x_t = tt[..., None, None] * x[:, None] + (1 - tt[..., None, None]) * noise                    # (B, K, 64, 2)
            traj_seed = zlib.crc32(items[0]["key"].encode()); torch.manual_seed(traj_seed)
            xyz, rot, v = T.sample_trajectories(batch["tokenized_data"], batch["ego_history_xyz"], batch["ego_history_rot"], num_samples=K,
                                                num_steps=10, velocity_at=(x_t.flatten(0, 1), tt.flatten()))
            yaw = torch.atan2(rot[..., 1, 0], rot[..., 0, 0])
            for n, s in enumerate(items):
                rows.append(dict(key=s["key"], cell=str(ds.rows.cell.iloc[mine[b + n]]), lidar_dropped=s["lidar_dropped"], traj_seed=traj_seed,
                                 seed=pts[n][2], t=pts[n][0].numpy(), v=v[n].cpu().numpy(),
                                 xyyaw=torch.cat([xyz[n, ..., :2], yaw[n, ..., None]], -1).float().cpu().numpy()))
        np.savez(f.with_suffix(".tmp.npz"), **{k: np.stack([r[k] for r in rows]) for k in rows[0]})
        os.replace(f.with_suffix(".tmp.npz"), f)
        print(f"rank {rank}: {f.name} {len(rows)} samples, {(time.time() - t0) / len(rows) * 1000:.0f} ms/sample", flush=True)


if __name__ == "__main__":
    main()
