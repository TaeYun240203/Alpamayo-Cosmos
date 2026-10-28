"""Expert-free Stage 1 metrics (morai_vla/stage1_eval.py) for one checkpoint: CE with LiDAR on/off and discrete-token
ADE/FDE vs constant velocity, per cell. --nf4 for a 16 GB card (LLM linears NF4; ViTs, embeddings, LM head BF16).
Usage (cwd train/): python scripts/eval_stage1.py <release or checkpoint dir> <step-0 root> [--n 200] [--nf4] --out logs/x.json"""
import argparse, json, logging, time

import torch
from transformers import BitsAndBytesConfig

from morai_vla.model import TeacherVLA
from morai_vla.stage1_eval import EvalSamples, evaluate, summarize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("root")
ap.add_argument("--n", type=int, default=200); ap.add_argument("--nf4", action="store_true"); ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--out", required=True)
ap.add_argument("--shard", default="0,1", help="k,n: rows k::n (one process per GPU; pipeline_gate.py select merges)")
a = ap.parse_args()
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                       llm_int8_skip_modules=["visual", "lm_head", "action_in_proj", "action_out_proj"]) if a.nf4 else None
m = TeacherVLA.build(a.ckpt, attn_implementation="sdpa", loss_weights={"traj_ce": 1.0, "lm": 0.0, "fm": 0.0},
                     quantization_config=q, device_map={"": 0})
k, nsh = map(int, a.shard.split(","))
t = time.time(); rows = evaluate(m, EvalSamples(a.root, m.config, a.n, keep=slice(k, None, nsh)), a.workers)
res = summarize(rows) | {"ckpt": a.ckpt, "nf4": a.nf4, "s": round(time.time() - t)}
json.dump({"summary": res, "rows": rows}, open(a.out, "w"), indent=1); print(json.dumps(res, indent=1))
