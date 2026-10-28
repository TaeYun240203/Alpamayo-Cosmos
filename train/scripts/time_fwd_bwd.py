"""Stage-1 forward+backward time per sample and peak GPU memory, without an optimizer (for stage-C time estimates; the
optimizer step on 8 GPUs with ZeRO-2 is separate). Run alone on the GPU.
Usage (cwd train/): python scripts/time_fwd_bwd.py <release dir> <step-0 root> [coc] [--batch 1] [--n 12]"""
import argparse, json, time

import torch

from morai_vla.data import MoraiDataset
from morai_vla.model import TeacherVLA
from morai_vla.processor import collate_fn

ap = argparse.ArgumentParser(); ap.add_argument("release"); ap.add_argument("root"); ap.add_argument("coc", nargs="?", default="")
ap.add_argument("--batch", type=int, default=1); ap.add_argument("--n", type=int, default=12); ap.add_argument("--coc-parquet", default=None)
a = ap.parse_args()
m = TeacherVLA.build(a.release, attn_implementation="flash_attention_2", train_parts=["camera_vit", "lidar", "llm", "lm_head"],
                     loss_weights={"traj_ce": 1.0, "lm": 1.0, "fm": 0.0}, device_map={"": 0})
m.gradient_checkpointing_enable({"use_reentrant": False}); m.train()
ds = MoraiDataset(a.root, "train", model_config=m.config, vla_preprocess_args={"_target_": "morai_vla.processor.get_preprocess_fn"},
                  coc=a.coc == "coc", coc_parquet=a.coc_parquet or f"{a.root}/_coc/coc_dataset.parquet", frame_stride=5,
                  max_samples=(a.n + 2) * a.batch, augment=False, lidar_dropout=0.15)
dev = lambda x: {k: dev(v) if isinstance(v, dict) else v.cuda() if torch.is_tensor(v) else v for k, v in x.items()}
times, toks = [], []
for k in range(a.n + 2):                                                            # 2 warm-up iterations
    b = dev(collate_fn([ds[k * a.batch + j] for j in range(a.batch)], model_config=m.config))
    toks.append(int(b["tokenized_data"]["attention_mask"].sum()))
    torch.cuda.synchronize(); t = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(**b).loss
    loss.backward(); m.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); times.append(time.time() - t)
t = sorted(times[2:])
print(json.dumps({"coc": a.coc == "coc", "batch": a.batch, "s_per_step_median": round(t[len(t) // 2], 3),
                  "s_per_sample": round(t[len(t) // 2] / a.batch, 3), "tokens_per_sample": round(sum(toks[2:]) / len(toks[2:]) / a.batch),
                  "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "losses": {k: round(s / n, 4) for k, (s, n) in m.loss_sums.items()}}))
