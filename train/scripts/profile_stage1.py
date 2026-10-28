"""Where Stage 1 time goes on one GPU. Full-VLM Stage 1 forward+backward (real weights, real batches preloaded, GC on):
plain s/step, then torch.profiler: GPU kernel time by class (GEMM / attention / elementwise-norm-rotary / other), the idle
share of the wall time, achieved model TFLOPS (6 x params x tokens, ViTs included; GC recompute not counted) against
--peak; plus a fused AdamW step over 1/n_gpus of the trainable parameters (one rank's ZeRO-2 share). Per-step loss and grad
norm on the fixed batches (no updates) check that a speed variant computes the same thing. est_8gpu_mem_gb = peak - full
bf16 grads + one rank's ZeRO-2 share (bf16 + fp32 grad partition, fp32 master + 2 Adam moments = 18 B/param / n_gpus) + 2 GB buckets.
Usage (cwd train/): python scripts/profile_stage1.py <release> <step-0 root> [--batch 4] [--n 8] [--attn flash_attention_2|sdpa]
                    [--peak 2250 (B200 dense BF16 TFLOPS; H100 SXM 989, NVL 835)] [--gpus 8] [--no-gc]
                    [--sdpa-backend cudnn|flash|efficient (with --attn sdpa)] [--out logs/profile.json]"""
import argparse, json, re, time
from contextlib import nullcontext

import torch
from torch.autograd import DeviceType
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

from morai_vla.data import MoraiDataset
from morai_vla.model import TeacherVLA
from morai_vla.processor import collate_fn

CLASSES = [("memcpy_memset", r"memcpy|memset"), ("gemm", r"gemm|cutlass|cublas|nvjet|xmma|wgmma|matmul|splitk"), ("attention", r"flash|fmha|attn|attention|sdpa|cudnn"),
           ("elementwise_norm_rotary", r"elementwise|vectorized|reduce|norm|rotary|softmax|silu|gelu|copy|cat|index|fill|where|unrolled|mul|add")]


def kernel_class(name):
    return next((c for c, rx in CLASSES if re.search(rx, name, re.I)), "other")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("release"); ap.add_argument("root")
    ap.add_argument("--batch", type=int, default=4); ap.add_argument("--n", type=int, default=8); ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--peak", type=float, default=2250.0); ap.add_argument("--gpus", type=int, default=8); ap.add_argument("--out", default=None)
    ap.add_argument("--no-gc", action="store_true"); ap.add_argument("--sdpa-backend", default=None, choices=["cudnn", "flash", "efficient"])
    a = ap.parse_args()
    m = TeacherVLA.build(a.release, attn_implementation=a.attn, train_parts=["camera_vit", "lidar", "llm", "lm_head"],
                         loss_weights={"traj_ce": 1.0, "lm": 1.0, "fm": 0.0}, device_map={"": 0})
    if not a.no_gc: m.gradient_checkpointing_enable({"use_reentrant": False})
    m.train()
    sdpa = {"cudnn": SDPBackend.CUDNN_ATTENTION, "flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION}.get(a.sdpa_backend)
    ds = MoraiDataset(a.root, "train", model_config=m.config, vla_preprocess_args={"_target_": "morai_vla.processor.get_preprocess_fn"},
                      frame_stride=5, max_samples=(a.n + 2) * a.batch, augment=False, lidar_dropout=0.15)
    batches = [collate_fn([ds[k * a.batch + j] for j in range(a.batch)], model_config=m.config) for k in range(a.n + 2)]
    tokens = sum(int(b["tokenized_data"]["attention_mask"].sum()) for b in batches[2:])
    patches = sum(int(b["tokenized_data"]["image_grid_thw"].prod(-1).sum()) for b in batches[2:])
    dev = lambda x: {k: dev(v) if isinstance(v, dict) else v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in x.items()}

    def step(b, check=False):
        with torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel([sdpa]) if sdpa else nullcontext():
            loss = m(**dev(b)).loss
        loss.backward()
        out = (float(loss), float(sum(p.grad.float().pow(2).sum() for p in m.parameters() if p.grad is not None) ** 0.5)) if check else None
        m.zero_grad(set_to_none=True); return out

    for b in batches[:2]: step(b)                                                   # warm-up
    torch.cuda.synchronize(); t = time.time()
    for b in batches[2:]: step(b)
    torch.cuda.synchronize(); wall = time.time() - t
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t = time.time()
        for b in batches[2:]: step(b)
        torch.cuda.synchronize(); wall_prof = time.time() - t
    by, top = {}, []
    for e in prof.key_averages():
        us = e.self_device_time_total
        if us <= 0 or e.device_type != DeviceType.CUDA: continue
        by[kernel_class(e.key)] = by.get(kernel_class(e.key), 0) + us; top.append((us, e.key[:90]))
    busy = sum(by.values()) / 1e6
    n_llm = sum(p.numel() for n, p in m.vlm.model.language_model.named_parameters() if "embed_tokens" not in n)
    n_vit = sum(p.numel() for p in m.vlm.model.visual.parameters())
    flops = 6 * (n_llm * tokens + n_vit * patches)                                  # both ViTs share n_vit; patches cover both
    peak_mem = torch.cuda.max_memory_allocated() / 2**30
    checks = [step(b, check=True) for b in batches[2:]]                             # same batches, no updates: loss, grad norm

    # one rank's optimizer share on --gpus GPUs: fused AdamW over trainable/gpus fp32 params (DeepSpeed FusedAdam is similar)
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad); del m, batches; torch.cuda.empty_cache()
    est8 = peak_mem + (18 / a.gpus - 2) * n_tr / 2**30 + 2
    ps = [torch.zeros(n_tr // a.gpus // 64, device="cuda", requires_grad=True) for _ in range(64)]
    for p in ps: p.grad = torch.randn_like(p)
    opt = torch.optim.AdamW(ps, lr=1e-5, fused=True); opt.step(); torch.cuda.synchronize(); t = time.time()
    for _ in range(5): opt.step()
    torch.cuda.synchronize(); adam = (time.time() - t) / 5

    res = {"gpu": torch.cuda.get_device_name(0), "attn": a.attn, "sdpa_backend": a.sdpa_backend, "gc": not a.no_gc, "batch": a.batch, "steps": a.n,
           "s_per_step": round(wall / a.n, 3), "s_per_sample": round(wall / a.n / a.batch, 3), "tokens_per_sample": round(tokens / a.n / a.batch),
           "peak_mem_gb": round(peak_mem, 1), f"est_{a.gpus}gpu_mem_gb": round(est8, 1), "profiled_s_per_step": round(wall_prof / a.n, 3),
           "loss_per_step": [round(l, 5) for l, _ in checks], "grad_norm_per_step": [round(g, 4) for _, g in checks],
           "gpu_busy_share": round(busy / wall_prof, 3), "kernel_share": {k: round(v / 1e6 / busy, 3) for k, v in sorted(by.items())},
           "model_tflops": round(flops / wall / 1e12, 1), "mfu_vs_peak": round(flops / wall / 1e12 / a.peak, 3),
           "adamw_step_s_one_rank_of": [round(adam, 3), a.gpus],
           "top_kernels_ms": [(round(us / 1e3 / a.n, 2), k) for us, k in sorted(top, reverse=True)[:12]]}
    print(json.dumps(res, indent=1))
    if a.out: json.dump(res, open(a.out, "w"), indent=1)
