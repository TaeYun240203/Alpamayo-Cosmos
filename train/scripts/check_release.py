"""Real Alpamayo-1.5-10B through our processor on MORAI samples, inference only. On a 16 GB card the LLM and expert linears
load in NF4 (bitsandbytes); ViTs (camera + LiDAR copy), embeddings, LM head and action projections stay BF16.
Variants per sample: traj-only prompt (deployment format) with/without LiDAR, and the 1.5-style CoC rollout (1 sample).
cv_ade = constant-velocity baseline from the last 0.5 s of history.
Usage (cwd train/): python scripts/check_release.py --release <Alpamayo-1.5-10B dir> --root <step-0 share> --n 24 --out logs/release
  [--ckpt <trained teacher dir>] [--variants traj,traj_nolidar]   (summary.json also breaks every metric down by cell)
"""
import argparse, json, logging, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np, torch
from transformers import BitsAndBytesConfig

from morai_vla.data import MoraiDataset
from morai_vla.model import TeacherVLA
from morai_vla.processor import collate_fn, get_preprocess_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def to_dev(b, dev):
    return {k: (to_dev(v, dev) if isinstance(v, dict) else v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


def const_vel(hx, n=64):
    v = (hx[0, -1, :2] - hx[0, -6, :2]) / 0.5                                  # last 0.5 s of history, ego frame
    return torch.arange(1, n + 1)[:, None] * 0.1 * v[None]


def summarize(rows, copy):
    keys = [k for k in rows[0] if k.endswith(("_ade", "_fde", "_minade", "_ms"))]
    mean = lambda rs: {k: round(float(np.mean([r[k] for r in rs])), 3) for k in keys} | {"n": len(rs)}
    return mean(rows) | {"lidar_vit_is_camera_copy": copy, "by_cell": {c: mean([r for r in rows if r["cell"] == c]) for c in sorted({r["cell"] for r in rows})}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True); ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=24); ap.add_argument("--samples", type=int, default=6); ap.add_argument("--out", default="logs/release")
    ap.add_argument("--no-quant", action="store_true"); ap.add_argument("--ckpt", default=None, help="trained teacher instead of the release")
    ap.add_argument("--variants", default="traj,traj_nolidar,cot")
    ap.add_argument("--shard", default="0,1", help="k,n: rows k::n (one process per GPU; scripts/pipeline_gate.py merge)")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    q = None if a.no_quant else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                                                   llm_int8_skip_modules=["visual", "lm_head", "action_in_proj", "action_out_proj"])
    t = time.time()
    model = TeacherVLA.build(a.ckpt or a.release, attn_implementation="sdpa", quantization_config=q, device_map={"": 0}).eval()
    logging.info("loaded in %.0f s, GPU %.1f GB", time.time() - t, torch.cuda.memory_allocated() / 2**30)
    lv, cv = model.vlm.model.lidar_visual, model.vlm.model.visual
    copy = all(torch.equal(p, r) for p, r in zip(lv.state_dict().values(), cv.state_dict().values()))
    assert copy or a.ckpt, "LiDAR ViT is not a camera-ViT copy"
    cfg = model.config
    ds = MoraiDataset(a.root, ["val", "val_asset"], frame_stride=10, max_samples=a.n, augment=False, seed=1)
    k0, nsh = map(int, a.shard.split(",")); ds.rows = ds.rows.iloc[k0::nsh].reset_index(drop=True)
    fns = {"traj": get_preprocess_fn(cfg, generation_mode=True), "cot": get_preprocess_fn(cfg, generation_mode=True, generate_cot=True)}
    rows, plots = [], []
    for k in range(len(ds)):
        s = ds.sample(k)
        gt = s["ego_future_xyz"][0, :, :2]
        r = dict(key=s["key"], cell=str(ds.rows.cell.iloc[k]), nav=s["nav_text"],
                 v0=float((s["ego_history_xyz"][0, -1, :2] - s["ego_history_xyz"][0, -2, :2]).norm() / 0.1),
                 cv_ade=float((const_vel(s["ego_history_xyz"]) - gt).norm(dim=-1).mean()), cv_fde=float((const_vel(s["ego_history_xyz"]) - gt)[-1].norm()))
        for name, lidar, fn in (("traj", True, "traj"), ("traj_nolidar", False, "traj"), ("cot", True, "cot")):
            if name not in a.variants.split(","): continue
            d = dict(s); d["cot"] = None
            if not lidar: d.pop("bev_frames", None)
            d["tokenized_data"] = fns[fn](data=d)
            b = to_dev(collate_fn([{k2: v for k2, v in d.items() if k2 not in ("camera_frames", "bev_frames")}], model_config=cfg), "cuda")
            torch.manual_seed(0); t = time.time()
            if fn == "traj":
                xyz, _ = model.sample_trajectories(b["tokenized_data"], b["ego_history_xyz"], b["ego_history_rot"], num_samples=a.samples)
                xyz = xyz[0]
            else:
                td = b["tokenized_data"]; model.vlm.model._image_is_lidar = td.pop("image_is_lidar")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    xyz, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                        dict(tokenized_data=td, ego_history_xyz=b["ego_history_xyz"], ego_history_rot=b["ego_history_rot"]),
                        num_traj_samples=1, max_generation_length=64, return_extra=True)
                xyz = xyz[0, 0]; r["pred_coc"] = str(np.asarray(extra.get("cot", [[[""]]])).reshape(-1)[0])
            err = (xyz[..., :2].float().cpu() - gt).norm(dim=-1)                 # (samples, 64)
            r[f"{name}_ade"], r[f"{name}_fde"] = float(err.mean(1).mean()), float(err[:, -1].mean())
            r[f"{name}_minade"] = float(err.mean(1).min()); r[f"{name}_ms"] = round((time.time() - t) * 1000)
            if name == "traj": plots.append((gt.numpy(), xyz[..., :2].float().cpu().numpy(), r))
        rows.append(r); logging.info(json.dumps(r))
        if len(rows) % 100 == 0: json.dump(rows, open(out / "release_check.json", "w"), indent=1)
    json.dump(rows, open(out / "release_check.json", "w"), indent=1)
    summary = summarize(rows, copy)
    json.dump(summary, open(out / "summary.json", "w"), indent=1); print(json.dumps(summary, indent=1))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(4, 6, figsize=(24, 16))
        for axk, (gt, pr, r) in zip(ax.flat, plots):
            for p in pr: axk.plot(-p[:, 1], p[:, 0], "C1", alpha=0.6)
            axk.plot(-gt[:, 1], gt[:, 0], "k", lw=2)
            L = max(10.0, float(abs(np.concatenate([gt, pr.reshape(-1, 2)])).max()) * 1.1)
            axk.set_xlim(-L / 2, L / 2); axk.set_ylim(-0.1 * L, 0.9 * L); axk.set_aspect("equal")    # x: right +, y: forward
            axk.set_title(f"{r['key']} {r['cell']} v0={r['v0']:.1f}\n{r['nav'][:40]}  ADE {r['traj_ade']:.2f}", fontsize=8)
        fig.savefig(out / "trajectories.png", dpi=70); logging.info("wrote %s", out / "trajectories.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
