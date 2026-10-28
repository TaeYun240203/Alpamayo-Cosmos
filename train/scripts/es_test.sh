#!/bin/bash
# winpc check of the trajectory-ADE early stop on the tiny Stage 2 model (configs/tiny_stage2.yaml), under torchrun so the
# rank-sharded scoring, gather and barrier run. A: the user's rule numbers; B: rel_improve 0.5 forces the stop + decay path.
# Then final/ must hold the chosen measurement's expert weights. Usage (winpc): bash scripts/es_test.sh
cd ~/morai-teacher/train && export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false && source ~/claude-bench/venv-teacher/bin/activate
REL=$(ls -d ~/.cache/huggingface/hub/models--nvidia--Alpamayo-1.5-10B/snapshots/*); ROOT=/mnt/d/share/2026_MORAI_step0; O=~/morai-teacher/runs/es_test
C="paths.alpamayo15=$REL data.root=$ROOT model.vlm_name_or_path=Qwen/Qwen3-VL-8B-Instruct"
for run in A B; do
  extra=""; [ $run = B ] && extra="+callbacks.traj_early_stop.rel_improve=0.5"
  torchrun --nproc_per_node 1 -m morai_vla.train --config-name tiny_stage2 $C paths.output_dir=$O/$run $extra > $O.$run.log 2>&1 || { echo "run $run failed"; continue; }
  python - $O/$run <<'PY'
import json, sys
from pathlib import Path
from safetensors.torch import load_file
d = Path(sys.argv[1]); sel = json.loads((d / "ade_select.json").read_text()); st = json.loads((d / "ade_early_stop.json").read_text())
snap = load_file(str(d / [h for h in sel["history"] if h["step"] == sel["chosen_step"]][0]["snapshot"]))
final = {}
for f in (d / "final").glob("*.safetensors"): final |= load_file(str(f))
same = all((final[k].float() - v.float()).abs().max() == 0 for k, v in snap.items())
print(json.dumps({"run": d.name, "min_step": st["min_step"], "steps_per_epoch": st["steps_per_epoch"], "stop_step": st["stop_step"],
                  "decay_end": st["decay_end"], "history": [(h["step"], h["ade"], h["tag"], h["seconds"]) for h in st["history"]],
                  "chosen": (sel["chosen_step"], sel["chosen_ade"], sel["chosen_tag"]), "final_equals_chosen_snapshot": same,
                  "snapshot_tensors": len(snap)}))
PY
done
