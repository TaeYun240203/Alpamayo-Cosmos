"""(local diagnostic, 2026-10-24) Our MORAI prompt vs the native 1.5 prompt (frame k, 3 front cameras at the 1.5 pixel budget) on the same samples.
Usage (cwd train/): ALPAMAYO15=<release dir> MORAI_ROOT=<share> python scripts/diag_native_prompt.py 10"""
import json, logging, sys, time
import numpy as np, torch
from transformers import BitsAndBytesConfig
from morai_vla.data import MoraiDataset
from morai_vla.model import TeacherVLA
from morai_vla.processor import collate_fn, get_preprocess_fn, get_processor
from alpamayo.chat_template import components as C
logging.basicConfig(level=logging.INFO)
import os
REL, ROOT = os.environ["ALPAMAYO15"], os.environ["MORAI_ROOT"]
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                       llm_int8_skip_modules=["visual", "lm_head", "action_in_proj", "action_out_proj"])
m = TeacherVLA.build(REL, attn_implementation="sdpa", quantization_config=q, device_map={"": 0}).eval()
cfg = m.config; P = get_processor(cfg); proc = P.processor
ds = MoraiDataset(ROOT, ["val", "val_asset"], frame_stride=10, max_samples=int(sys.argv[1]), augment=False, seed=1)
ours = get_preprocess_fn(cfg, generation_mode=True)


def native(s, cot=False):
    """alpamayo1_5 helper.create_message layout: 'Front left camera: ' 'frame k ' images (cam idx 0,1,2), 1.5 min/max pixels."""
    txt = lambda t: {"type": "text", "text": t}
    user, imgs = [], []
    for name, cam in (("Front left camera", 1), ("Front camera", 0), ("Front right camera", 2)):
        user.append(txt(f"{name}: "))
        for k in range(4): user += [txt(f"frame {k} "), {"type": "image"}]
        imgs.append(s["camera_frames"][cam])
    user += C.construct_traj_history(48) + C.construct_route(s)
    comps = (["cot"] if cot else []) + ["traj_future"]
    user += C.construct_user_prompt(comps, comps, False)
    asst = C.construct_cot(s, ask_for_component=True) if cot else C.construct_traj_future(128, ask_for_component=True)
    text = proc.apply_chat_template([{"role": "system", "content": C.construct_system_prompt()}, {"role": "user", "content": user},
                                     {"role": "assistant", "content": asst}], tokenize=False, add_generation_prompt=False, continue_final_message=True)
    pv, thw = [], []
    for f in imgs:
        o = proc.image_processor(images=f.float() / 255, do_rescale=False, min_pixels=163840, max_pixels=196608); pv.append(o["pixel_values"]); thw.append(o["image_grid_thw"])
    thw = torch.cat(thw); parts = text.split(proc.image_token)
    text = parts[0] + "".join(proc.image_token * int(n // 4) + p for n, p in zip(thw.prod(-1), parts[1:]))
    s["label_components"], s["generation_mode"] = ["cot", "traj_future"], True
    return {"text": text, "pixel_values": torch.cat(pv), "image_grid_thw": thw, "image_is_lidar": torch.zeros(len(thw), dtype=torch.bool)}


def dev(x): return {k: dev(v) if isinstance(v, dict) else v.cuda() if torch.is_tensor(v) else v for k, v in x.items()}
def speed(xyz): return (np.diff(np.concatenate([np.zeros((1, 2)), xyz[:, :2]]), axis=0) ** 2).sum(1) ** 0.5 / 0.1

res = []
for k in range(len(ds)):
    s = ds.sample(k); gt = s["ego_future_xyz"][0, :, :2].numpy()
    hv = speed(s["ego_history_xyz"][0].numpy() - s["ego_history_xyz"][0, 0].numpy())[1:]
    r = dict(key=s["key"], cell=str(ds.rows.cell.iloc[k]), nav=s["nav_text"], v_hist=round(float(hv[-1]), 1),
             gt_end=np.round(gt[-1], 1).tolist(), gt_v=[round(float(x), 1) for x in speed(gt)[[9, 29, 63]]])
    for name, fn in (("ours", lambda d: ours(data=d)), ("native", native)):
        d = dict(s); d["cot"] = None; d["tokenized_data"] = fn(d)
        b = dev(collate_fn([{a: v for a, v in d.items() if a not in ("camera_frames", "bev_frames")}], model_config=cfg))
        torch.manual_seed(0)
        xyz, _ = m.sample_trajectories(b["tokenized_data"], b["ego_history_xyz"], b["ego_history_rot"], num_samples=4)
        p = xyz[0, :, :, :2].float().cpu().numpy()
        r[name + "_ade"] = round(float(np.linalg.norm(p - gt, axis=-1).mean()), 2)
        r[name + "_end"] = np.round(p[:, -1].mean(0), 1).tolist(); r[name + "_v"] = [round(float(x), 1) for x in speed(p.mean(0))[[9, 29, 63]]]
    res.append(r); print(json.dumps(r), flush=True)
print("mean ours", np.mean([r["ours_ade"] for r in res]), "native", np.mean([r["native_ade"] for r in res]))
