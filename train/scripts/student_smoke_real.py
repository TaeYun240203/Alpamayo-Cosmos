"""Real-weight student smoke on a 16 GB card (no room for T* BF16 + student together):
 A. T* = the release (or --teacher) in NF4 (LLM/expert linears; ViT, buffers BF16 as check_release): for n val samples
    store the prefix KV of all 36 layers, the ViT patch features, 2 trajectories and the velocity at K fixed (x_t, t).
 B. the student built on CPU from Qwen3-VL-4B + the teacher checkpoint (distill off): strict transplant, every
    transplanted tensor equal to the file, expert config vs the 8B-built one, config-derived buffers vs T*'s.
 C. the student in BF16 on the GPU, inference only, through the training loss functions: per-layer KV MSE against the
    stored T* KV, linear-probe fit of the student ViT features to T*'s (what the 1024 -> 1152 KD map can reach at init),
    GT flow matching and velocity KD at the stored points, answer-token CE (trajectory tokens + end), and K trajectories from T*'s initial noise.
Usage (cwd train/): python scripts/student_smoke_real.py --teacher <release dir> --root <share> --out <dir> [--n 4]"""
import argparse, copy, gc, json, logging, sys, time
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from transformers import BitsAndBytesConfig, Qwen3VLConfig

from morai_vla.data import MoraiDataset
from morai_vla.model import TeacherVLA
from morai_vla.processor import collate_fn, get_preprocess_fn
from morai_vla.student import TRANSPLANT, StudentVLA, hook_vit_features, kv_mse

sys.path.insert(0, str(Path(__file__).parent))
from soft_targets import fixed_points

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
K, K_TRAJ = 6, 2                                    # fixed (x_t, t) points; trajectories (x6 KV copies do not fit 16 GB next to the student)


def batches(ds, i, cfg):
    """the same raw sample as a training batch (labels) and as an inference prompt (ends in <traj_future_start>)."""
    s = ds.sample(i); s["cot"] = None
    out = []
    for gen in (False, True):
        d = dict(s); d["tokenized_data"] = get_preprocess_fn(cfg, generation_mode=gen)(data=d)
        b = collate_fn([{k: v for k, v in d.items() if k not in ("camera_frames", "bev_frames")}], model_config=cfg)
        out.append({k: ({kk: vv.cuda() for kk, vv in v.items()} if isinstance(v, dict) else v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()})
    return s, out


def prefix(model, b):
    """VLM forward of a training batch; returns (out, input_ids, attention mask, labels, start)."""
    td = dict(b["tokenized_data"]); model.vlm.model._image_is_lidar = td.pop("image_is_lidar")
    traj = {k: b[k] for k in ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot")}
    ids = model.fuse_traj_tokens(td.pop("input_ids"), traj)
    start = int((ids == model.config.traj_token_ids["future_start"]).nonzero()[-1, 1]) + 1
    keep = ids.shape[1] - start + 1                                                     # logits of the answer tokens only (16 GB)
    out = model.vlm(input_ids=ids, use_cache=True, logits_to_keep=keep, **td)
    labels = torch.where(b["labels_mask"], ids, -100)[:, -keep:]
    return out, ids, td["attention_mask"], labels, start, traj


@torch.no_grad()
@torch.autocast("cuda", dtype=torch.bfloat16)
def velocity(model, out, ids, attn, x_t, t):
    vs = []
    for j in range(len(t)):                                                            # batch-1 cache: one point at a time
        vs.append(model._velocity(out, ids, attn, x_t[j:j + 1], t[j].view(1, 1, 1)).float())
    return torch.cat(vs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--student_vlm", default="Qwen/Qwen3-VL-4B-Instruct")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = {}

    # ---- A. teacher (NF4) features
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                           llm_int8_skip_modules=["visual", "lm_head", "action_in_proj", "action_out_proj"])
    t0 = time.time()
    T = TeacherVLA.build(a.teacher, attn_implementation="sdpa", quantization_config=q, device_map={"": 0}).eval()
    T.vit_feats = hook_vit_features(T)
    rep["teacher_load_s"] = round(time.time() - t0); rep["teacher_gpu_gb"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    ds = MoraiDataset(a.root, ["val", "val_asset"], frame_stride=10, max_samples=a.n, augment=False, seed=1)
    buffers = {f"{n}.{k}": b.cpu() for n in TRANSPLANT for k, b in T.get_submodule(n).named_buffers()}
    for i in range(len(ds)):
        s, (b, g) = batches(ds, i, T.config)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o, ids, attn, labels, start, traj = prefix(T, b)
            x = T.action_space.traj_to_action(traj_history_xyz=traj["ego_history_xyz"], traj_history_rot=traj["ego_history_rot"],
                                              traj_future_xyz=traj["ego_future_xyz"], traj_future_rot=traj["ego_future_rot"]).reshape(-1, 64, 2)
            t, noise, seed = fixed_points(s["key"], x[0], K)
            t, noise = t.cuda(), noise.cuda()
            x_t = t[:, None, None] * x + (1 - t[:, None, None]) * noise
            kv = [(l.keys[:, :, :start].cpu(), l.values[:, :, :start].cpu()) for l in o.past_key_values.layers]
            feats = {k: f.cpu() for k, f in T.vit_feats.items()}
            ce = T._compute_next_token_loss(SimpleNamespace(logits=o.logits), labels, labels != -100).item()
            v = velocity(T, o, ids, attn, x_t, t)
            torch.manual_seed(i)
            xyz, _ = T.sample_trajectories(g["tokenized_data"], g["ego_history_xyz"], g["ego_history_rot"], num_samples=K_TRAJ)
        torch.save(dict(key=s["key"], kv=kv, feats=feats, x=x.cpu(), t=t.cpu(), noise=noise.cpu(), x_t=x_t.cpu(), v=v.cpu(), ce=ce,
                        xy=xyz[0, ..., :2].float().cpu(), gt=s["ego_future_xyz"][0, :, :2], start=start), out / f"teacher_{i}.pt")
        logging.info("teacher sample %d %s: prefix %d tokens, answer CE %.3f", i, s["key"], start, ce)
    del T, o; gc.collect(); torch.cuda.empty_cache()

    # ---- B. student build on CPU (strict transplant)
    t0 = time.time()
    S = StudentVLA.build(teacher=a.teacher, vlm_name_or_path=a.student_vlm, distill=False, attn_implementation="sdpa").eval()
    rep["student_build_s"] = round(time.time() - t0)
    rep["params_b"] = {p: round(sum(x.numel() for path in S.PARTS[p] for x in S.get_submodule(path).parameters()) / 1e9, 3) for p in S.PARTS}
    rep["params_b"]["total"] = round(sum(x.numel() for x in S.parameters()) / 1e9, 3)
    mism, n = [], 0
    for f in sorted(Path(a.teacher).glob("*.safetensors")):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                if k.split(".")[0] in TRANSPLANT:
                    n += 1
                    if not torch.equal(h.get_tensor(k), S.get_parameter(k).data): mism.append(k)
    rep["transplanted_tensors"], rep["transplant_mismatch"] = n, mism
    tc = copy.deepcopy(Qwen3VLConfig.from_pretrained("Qwen/Qwen3-VL-8B-Instruct").text_config)
    for k, v in S.config.expert_cfg.items(): setattr(tc, k, v)
    sd, td = S.expert.config.to_dict(), tc.to_dict()
    rep["expert_config_diff_vs_8b_built"] = {k: (sd.get(k), td.get(k)) for k in set(sd) | set(td) if sd.get(k) != td.get(k)}
    rep["buffer_mismatch_vs_teacher"] = [k for k, b in buffers.items() if not torch.equal(S.get_buffer(k).float(), b.float())]
    rep["tied_embedding"] = S.vlm.lm_head.weight is S.vlm.model.language_model.embed_tokens.weight
    rep["vocab"] = [S.original_vocab_size, S.config.vocab_size]
    logging.info(json.dumps(rep))

    # ---- C. student on the GPU (BF16, inference) through the loss functions
    S.cuda(); rep["student_gpu_gb"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    rows = []
    for i in range(len(ds)):
        tt = torch.load(out / f"teacher_{i}.pt")
        s, (b, g) = batches(ds, i, S.config)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o, ids, attn, labels, start, traj = prefix(S, b)
            assert start == tt["start"]
            kvt = SimpleNamespace(layers=[SimpleNamespace(keys=k.cuda(), values=v.cuda()) for k, v in tt["kv"]])
            tot, per = kv_mse(o.past_key_values, kvt, attn, start)
            per_k = [float(((l.keys[:, :, :start].float() - lt.keys.float()) ** 2).mean()) for l, lt in zip(o.past_key_values.layers, kvt.layers)]
            t_ms = [float(lt.keys.float().pow(2).mean() + lt.values.float().pow(2).mean()) for lt in kvt.layers]
            probe = {}
            for k, f in S.vit_feats.items():                                            # least-squares 1024 -> 1152 fit at init
                X = torch.cat([f.float().cpu(), torch.ones(len(f), 1)], 1); Y = tt["feats"][k].float()
                W = torch.linalg.lstsq(X, Y).solution
                probe[k] = dict(n=len(f), teacher_ms=round(float(Y.pow(2).mean()), 4), fit_mse=round(float((X @ W - Y).pow(2).mean()), 4),
                                r2=round(1 - float((X @ W - Y).pow(2).sum() / (Y - Y.mean(0)).pow(2).sum()), 3))
            ce = S._compute_next_token_loss(SimpleNamespace(logits=o.logits), labels, labels != -100).item()
            v = velocity(S, o, ids, attn, tt["x_t"].cuda(), tt["t"].cuda())
            tgt = (tt["x"] - tt["noise"]).cuda()
            torch.manual_seed(i)
            xyz, _ = S.sample_trajectories(g["tokenized_data"], g["ego_history_xyz"], g["ego_history_rot"], num_samples=K_TRAJ)
        xy = xyz[0, ..., :2].float().cpu()
        r = dict(key=tt["key"], prefix=start, kv_layer_mean=round(float(tot), 3), kv_per_layer=[round(float(p), 3) for p in per],
                 k_mse_per_layer=[round(p, 3) for p in per_k], teacher_kv_meansq_per_layer=[round(p, 3) for p in t_ms], vit_probe=probe,
                 answer_ce_student=round(ce, 3), answer_ce_teacher=round(tt["ce"], 3),
                 fm_student=round(float((v - tgt).pow(2).mean()), 4), fm_teacher=round(float((tt["v"].cuda() - tgt).pow(2).mean()), 4),
                 vkd=round(float((v - tt["v"].cuda()).pow(2).mean()), 4),
                 ade_student=round(float((xy - tt["gt"]).norm(dim=-1).mean()), 3), ade_teacher=round(float((tt["xy"] - tt["gt"]).norm(dim=-1).mean()), 3),
                 st_ade=round(float((xy - tt["xy"]).norm(dim=-1).mean()), 3))
        rows.append(r); logging.info(json.dumps({k: v for k, v in r.items() if "per_layer" not in k}))
    rep["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2); rep["samples"] = rows
    json.dump(rep, open(out / "smoke_real.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "samples"}, indent=1))


if __name__ == "__main__":
    main()
