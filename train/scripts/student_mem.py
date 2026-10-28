"""Memory and time of one student training step with T* in the loop, at the real layer widths (Qwen3-VL-4B student,
8B-width T*, 2048-wide expert) but reduced depth, on real MORAI samples (real token layout, 18 images). Random weights,
no optimizer (its states are added analytically in the README memory plan). Per (L LLM/expert layers, D ViT layers):
the memory above the resident weights at the end of the forward (activations, T* transient: what ZeRO-2 adds to
gradient buckets) and at the backward peak (every grad kept), and the times of the T* forward (no grad) and the student
forward+backward, for stage 1 (expert frozen) and stage 2 (expert trained, + vkd). Fit linear in L and D, extrapolate
to L = 36, D = 24 (student ViT; T* ViT depth only changes its no-grad forward).
Usage (cwd train/): python scripts/student_mem.py --release <1.5 dir (config only)> --root <share> --out <json> [--grid 2x3,5x3,2x9]"""
import argparse, gc, json, logging, tempfile, time

import torch

from morai_vla.data import MoraiDataset
from morai_vla.model import TeacherVLA, load_config
from morai_vla.processor import collate_fn, get_preprocess_fn
from morai_vla.student import StudentVLA

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
KEYS = ("tokenized_data", "ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot", "labels_mask", "vkd_ok")


def overrides(L, D):
    ds = [round(D * f) - 1 for f in (0.25, 0.5, 0.75)] if D >= 6 else list(range(3))
    return dict(text_config=dict(num_hidden_layers=L), vision_config=dict(depth=D, deepstack_visual_indexes=ds))


def gpu(x):
    return {k: gpu(v) if isinstance(v, dict) else v.cuda() if torch.is_tensor(v) else v for k, v in x.items()}


def measure(release, items, L, D, reps=3):
    with tempfile.TemporaryDirectory() as d:
        prev = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)        # random init straight in bf16 (RAM)
        try:
            T = TeacherVLA.build("random:" + release, attn_implementation="sdpa", vlm_overrides=overrides(L, D)); T.save_pretrained(d + "/t"); del T
            S = StudentVLA.build(teacher=d + "/t", init="random", attn_implementation="sdpa", vlm_overrides=overrides(L, D), distill=False)
            S.save_pretrained(d + "/s"); del S; gc.collect()
        finally:
            torch.set_default_dtype(prev)
        S = StudentVLA.build(teacher=d + "/t", init=d + "/s", attn_implementation="sdpa", device_map={"": 0})   # as from_pretrained: fp32 inv_freq
        T = S.teacher.cuda(); S.train()
        res = {}
        for stage, parts, vkd in (("stage1", ["camera_vit", "lidar", "llm", "lm_head", "kd_proj"], 0.0), ("stage2", ["camera_vit", "lidar", "llm", "lm_head", "kd_proj", "expert"], 1.0)):
            S.set_stage(parts, dict(kv=1.0, vit=1.0, fm=1.0, traj_ce=1.0, lm=1.0, vkd=vkd, coc_kl=0.0))
            n_train = sum(p.numel() for p in S.parameters() if p.requires_grad)
            torch.cuda.synchronize()
            resident = torch.cuda.memory_allocated()
            tt, ts, fwd, bwd = [], [], [], []
            for r in range(reps + 1):
                b = gpu(collate_fn([items[r % len(items)]], model_config=S.config)); b["vkd_ok"] = [True]
                torch.cuda.synchronize(); t0 = time.time()
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):              # the T* part alone (timing split)
                    td = dict(b["tokenized_data"]); T.vlm.model._image_is_lidar = td.pop("image_is_lidar")
                    T.vlm(input_ids=T.fuse_traj_tokens(td.pop("input_ids"), {k: b[k] for k in KEYS[1:3]}), use_cache=True, logits_to_keep=1, **td)
                torch.cuda.synchronize(); t1 = time.time(); torch.cuda.reset_peak_memory_stats()
                loss = S(**{k: v for k, v in b.items() if k in KEYS}).loss
                torch.cuda.synchronize(); fwd.append(torch.cuda.max_memory_allocated() - resident)
                loss.backward(); del loss
                torch.cuda.synchronize(); t2 = time.time(); bwd.append(torch.cuda.max_memory_allocated() - resident)
                if r: tt.append(t1 - t0); ts.append(t2 - t1 - (t1 - t0))               # the step re-runs T*: subtract it
                S.zero_grad(set_to_none=True)
            res[stage] = dict(n_train_b=round(n_train / 1e9, 3), resident_gb=round(resident / 2**30, 2),
                              fwd_act_gb=round(max(fwd) / 2**30, 2),                 # end of forward: activations + T* transient
                              bwd_peak_gb=round(max(bwd) / 2**30, 2),                # backward with every grad kept (no ZeRO)
                              teacher_fwd_s=round(sorted(tt)[len(tt) // 2], 3), student_fwd_bwd_s=round(sorted(ts)[len(ts) // 2], 3))
        res["student_params_b"] = round(sum(p.numel() for p in S.parameters()) / 1e9, 3); res["teacher_params_b"] = round(sum(p.numel() for p in T.parameters()) / 1e9, 3)
        res["tokens"] = int(b["tokenized_data"]["input_ids"].shape[1])
        del S, T; gc.collect(); torch.cuda.empty_cache()
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--grid", default="2x3,5x3,2x9")
    a = ap.parse_args()
    ds = MoraiDataset(a.root, ["val"], frame_stride=10, max_samples=4, augment=False, seed=1)
    mc = load_config(a.release, "Qwen/Qwen3-VL-8B-Instruct")
    items = []
    for i in range(len(ds)):
        s = ds.sample(i); s["cot"] = None; s["tokenized_data"] = get_preprocess_fn(mc)(data=s)
        items.append({k: v for k, v in s.items() if k not in ("camera_frames", "bev_frames", "camera_times", "bev_times")})
    out = {}
    for g in a.grid.split(","):
        L, D = map(int, g.split("x"))
        out[g] = measure(a.release, items, L, D); logging.info("%s %s", g, json.dumps(out[g]))
        json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
