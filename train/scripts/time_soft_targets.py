"""GPU time of soft-target generation with the teacher (BF16, no CoC prompt): K trajectory samples by Euler flow matching,
and velocity-field evaluations at fixed (x_t, t) = prefix prefill + one batched expert call (timed as a 1-step Euler run).
per expert call = (t[10 steps] - t[1 step]) / 9; prefill = t[1 step] - per expert call. Run alone on the GPU.
Usage (cwd train/): python scripts/time_soft_targets.py <checkpoint dir> <step-0 root> [--k 6] [--batches 1,4,8] [--n 10]"""
import argparse, json, time

import torch

from morai_vla.data import MoraiDataset
from morai_vla.model import TeacherVLA
from morai_vla.processor import collate_fn

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("root")
ap.add_argument("--k", type=int, default=6); ap.add_argument("--batches", default="1,4,8"); ap.add_argument("--n", type=int, default=10)
a = ap.parse_args(); batches = [int(b) for b in a.batches.split(",")]
m = TeacherVLA.build(a.ckpt, attn_implementation="sdpa", device_map={"": 0}).eval()
ds = MoraiDataset(a.root, "train", model_config=m.config, frame_stride=10, max_samples=(a.n + 1) * max(batches), augment=False, seed=2,
                  vla_preprocess_args={"_target_": "morai_vla.processor.get_preprocess_fn", "generation_mode": True})
items = [ds[k] for k in range(len(ds))]
dev = lambda x: {k: dev(v) if isinstance(v, dict) else v.cuda() if torch.is_tensor(v) else v for k, v in x.items()}
res = {"k": a.k}
for B in batches:
    for steps in (1, 10):
        ts = []
        for j in range(a.n + 1):                                                      # first call = warm-up
            b = dev(collate_fn(items[j * B:(j + 1) * B], model_config=m.config))
            torch.cuda.synchronize(); t = time.time()
            m.sample_trajectories(b["tokenized_data"], b["ego_history_xyz"], b["ego_history_rot"], num_samples=a.k, num_steps=steps)
            torch.cuda.synchronize(); ts.append(time.time() - t)
        res[f"b{B}_steps{steps}_s"] = sorted(ts[1:])[a.n // 2]
    e = (res[f"b{B}_steps10_s"] - res[f"b{B}_steps1_s"]) / 9
    res[f"b{B}"] = {"euler10_ms_per_sample": round(res[f"b{B}_steps10_s"] / B * 1000), "prefill_ms_per_sample": round((res[f"b{B}_steps1_s"] - e) / B * 1000),
                    "expert_call_ms_per_sample": round(e / B * 1000, 1), "velocity_eval_ms_per_sample": round(res[f"b{B}_steps1_s"] / B * 1000)}
res["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
print(json.dumps(res, indent=1))
