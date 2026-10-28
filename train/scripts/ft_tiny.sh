#!/usr/bin/env bash
# CPU check of the R2 fine-tune (README "R2 미세조정"), the real entry point end to end: synthetic step-0 root
# (tests/ft_synth.py) -> tiny random T* and s2 student -> morai_vla.train student_ft +tiny=student_ft (mixture, per-row
# losses, history dropout, ft_select every 4 steps, final/) -> ft_measure history / static on the tiny final (BF16 path).
# Usage (cwd train/): bash scripts/ft_tiny.sh <dir with a 1.5 release config.json> <out dir>
# HF_HUB_CACHE must hold Qwen/Qwen3-VL-4B-Instruct (processor) and an 8B entry (upstream save_pretrained builds a default
# config that loads the 8B processor; on the Mac a scratch cache aliases it to the 4B files, the tokenizers are equal).
set -euo pipefail
R=$1; O=$2; mkdir -p "$O"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=$PWD
[ -f "$O/root/_index/index.parquet" ] || python tests/ft_synth.py "$O/root"
[ -f "$O/s2/final/config.json" ] || python - "$R" "$O" <<'PY'
import sys
from morai_vla.model import TeacherVLA
from morai_vla.student import StudentVLA
R, O = sys.argv[1:]; V = "Qwen/Qwen3-VL-4B-Instruct"
T = dict(text_config=dict(hidden_size=256, intermediate_size=512, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                          rope_scaling=dict(rope_type="default", mrope_section=[12, 10, 10], mrope_interleaved=True)),
         vision_config=dict(depth=4, hidden_size=128, intermediate_size=256, num_heads=4, out_hidden_size=256, deepstack_visual_indexes=[1, 2, 3]))
TeacherVLA.build("random:" + R, vlm_name_or_path=V, attn_implementation="sdpa", dtype="float32", vlm_overrides=T,
                 expert_overrides=dict(hidden_size=128, intermediate_size=256, num_attention_heads=4, head_dim=64)).save_pretrained(f"{O}/teacher")
# narrower than +tiny=student (text 64, ViT 64) to keep the Mac disk writes small: ft_select snapshots + final/ are
# dominated by the 155,697-row embedding; the KV layout (4 layers x 2 heads x 64) still matches T* and its expert
S = dict(text_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2, head_dim=64,
                          rope_scaling=dict(rope_type="default", mrope_section=[12, 10, 10], mrope_interleaved=True)),
         vision_config=dict(depth=4, hidden_size=64, intermediate_size=128, num_heads=2, out_hidden_size=64, deepstack_visual_indexes=[1, 2, 3]))
StudentVLA.build(teacher=f"{O}/teacher", init="random", vlm_name_or_path=V, teacher_vlm=V, attn_implementation="sdpa", dtype="float32",
                 vlm_overrides=S, distill=False).save_pretrained(f"{O}/s2/final")
PY
F=$O/root/_ft
python -m morai_vla.train --config-name student_ft +tiny=student_ft paths.s2=$O/s2/final paths.r2_root=$O/root \
  data.root=$O/root paths.ft_pools=$F paths.start_relabel=$F/start_relabel.parquet paths.output_dir=$O/ft   # no T* (user 10/28)
python scripts/ft_measure.py history --root $O/root --ckpt $O/ft/final --device cpu --n 4 --out $O/measure/history
python scripts/ft_measure.py static --root $O/root --ckpt $O/ft/final --device cpu --n 4 --keep $F/old_static.parquet --out $O/measure/static
python - $O <<'PY'
import json, sys
o = sys.argv[1]; c = json.load(open(f"{o}/ft/ft_choice.json")); rows = [json.loads(l) for l in open(f"{o}/ft/loss_components.jsonl")]
print("ft_choice", json.dumps({k: c[k] for k in ("chosen_step", "chosen_tag", "chosen_ade", "ref_start", "limit", "last_step")}))
print("measured steps", [h["step"] for h in c["history"]], "| loss keys", sorted({k for r in rows for k in r}))
PY
