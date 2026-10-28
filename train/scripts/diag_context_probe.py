"""(local diagnostic, 2026-10-24) Does the release model use its context? curve samples: original / gray images / mirrored history / official CoC rollout.
Usage (cwd train/): ALPAMAYO15=<release dir> MORAI_ROOT=<share> python scripts/diag_context_probe.py [--bf16-expert]"""
import json, logging, sys, numpy as np, torch
from transformers import BitsAndBytesConfig
from morai_vla.data import MoraiDataset
from morai_vla.model import TeacherVLA
from morai_vla.processor import collate_fn, get_preprocess_fn
logging.basicConfig(level=logging.INFO)
import os
REL, ROOT = os.environ["ALPAMAYO15"], os.environ["MORAI_ROOT"]
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=["visual", "lm_head", "action_in_proj", "action_out_proj"] + (["expert"] if "--bf16-expert" in sys.argv else []))
m = TeacherVLA.build(REL, attn_implementation="sdpa", quantization_config=q, device_map={"": 0}).eval()
cfg = m.config
ds = MoraiDataset(ROOT, ["val", "val_asset"], frame_stride=10, max_samples=10, augment=False, seed=1)
f = get_preprocess_fn(cfg, generation_mode=True); fc = get_preprocess_fn(cfg, generation_mode=True, generate_cot=True)
def dev(x): return {k: dev(v) if isinstance(v, dict) else v.cuda() if torch.is_tensor(v) else v for k, v in x.items()}
def run(s, cot=False):
    d = dict(s); d["cot"] = None; d["tokenized_data"] = (fc if cot else f)(data=d)
    b = dev(collate_fn([{a: v for a, v in d.items() if a not in ("camera_frames", "bev_frames")}], model_config=cfg))
    torch.manual_seed(0)
    if not cot:
        xyz, _ = m.sample_trajectories(b["tokenized_data"], b["ego_history_xyz"], b["ego_history_rot"], num_samples=4); return xyz[0].float().cpu().numpy(), None
    td = b["tokenized_data"]; m.vlm.model._image_is_lidar = td.pop("image_is_lidar")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        xyz, _, ex = m.sample_trajectories_from_data_with_vlm_rollout(dict(tokenized_data=td, ego_history_xyz=b["ego_history_xyz"], ego_history_rot=b["ego_history_rot"]),
                                                                     num_traj_samples=1, max_generation_length=64, return_extra=True)
    return xyz[0, 0].float().cpu().numpy(), str(np.asarray(ex["cot"]).reshape(-1)[0])
M = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
for k in [4, 8, 2]:                                   # the three mid-curve samples of diag_release (same seed/order)
    s = ds.sample(k); gt = s["ego_future_xyz"][0].numpy()
    out = {"key": s["key"], "hist_end_y": round(float(s["ego_history_xyz"][0, 0, 1]), 2), "gt_end": np.round(gt[-1, :2], 1).tolist()}
    p, _ = run(s); out["orig"] = np.round(p[:, -1, :2].mean(0), 1).tolist()
    g = dict(s); g["camera_frames"] = [torch.full_like(c, 128) for c in s["camera_frames"]]; g["bev_frames"] = torch.zeros_like(s["bev_frames"]) if "bev_frames" in s else None
    p, _ = run(g); out["gray_images"] = np.round(p[:, -1, :2].mean(0), 1).tolist()
    h = dict(s); h["ego_history_xyz"] = s["ego_history_xyz"] @ M; h["ego_history_rot"] = M @ s["ego_history_rot"] @ M
    p, _ = run(h); out["mirrored_history"] = np.round(p[:, -1, :2].mean(0), 1).tolist()
    p, coc = run(s, cot=True); out["coc_rollout"] = np.round(p[0, -1, :2], 1).tolist(); out["coc"] = coc
    print(json.dumps(out), flush=True)
