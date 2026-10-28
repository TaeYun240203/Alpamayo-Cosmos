#!/usr/bin/env bash
# Acceptance speed test while R1 step0 still downloads (user 10/27: health check during the download, never idle GPUs),
# restored from the 10/27 Texas box (B2 runs/r2ft/pro6000tx_57966802/speed.sh). The real ft code path (mixture, GT/replay
# rows, history dropout, T* distillation, ZeRO-2, the main run's attention) on the R2 root alone: the R1 sources are
# dropped and replay reads R2 rows (GT losses only, user 10/28: no T*, as the main run). ft_select runs
# at step 0 and at the end on 16-row sets (the measurement + snapshot code path the main run first hits at step 500).
# Writes /workspace/out/speed/{loss_components.jsonl, sys.log, SPEED.json}; r2ft_chain.sh kills it when R1 is in.
# Usage: STEPS=50 ATTN=sdpa bash /workspace/tools/vast/r2ft_speed.sh > /workspace/out/speed.log 2>&1
#   compare SPEED.json s_per_step with 2.75 (s2, 8x H200 Italy, fast CPU); 6.15 (PRO 6000 Pennsylvania) and 8.0 (H200
#   France, Xeon locked at 2.3 GHz) were CPU-dispatch bound: one python core per rank feeds the GPU.
W=/workspace; OUT=$W/out/stageR2; T=$W/out/speed; STEPS=${STEPS:-50}
source $W/alpamayo-recipes/recipes/alpamayo1_5_sft/a1_5_sft/bin/activate
cd $W/train; export PYTHONPATH=$PWD HF_HOME=$W/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
ulimit -n $(ulimit -Hn)
ATTN=${ATTN:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | grep -q '^12' && echo sdpa || echo flash_attention_2)}
mkdir -p $T; rm -f $T/loss_components.jsonl $T/ft_select.json $T/ft_val_rows.jsonl
( while :; do                                          # GPU util/power/clock, CPU MHz, the busiest python processes
    echo "== $(date +%T)"; nvidia-smi --query-gpu=index,utilization.gpu,power.draw,clocks.sm --format=csv,noheader
    grep MHz /proc/cpuinfo | awk '{print $4}' | sort -n | awk '{a[NR]=$1} END {print "cpu MHz min/med/max", a[1], a[int(NR/2)+1], a[NR]}'
    top -b -n 1 -o %CPU | grep -E 'python|torchrun' | head -12 | awk '{print $9 "% " $12}'
    sleep 15; done ) > $T/sys.log 2>&1 & SYS=$!
# NP + CUDA_VISIBLE_DEVICES: GPUs 1.. beside the QAD mini test on GPU 0 (r2ft_chain.sh)
torchrun --nproc_per_node ${NP:-$(nvidia-smi -L | wc -l)} -m morai_vla.train \
  --config-name student_ft data.root=/data/step0_r2 \
  paths.s2=$OUT/s2/final paths.r2_root=/data/step0_r2 paths.output_dir=$T model.attn_implementation=$ATTN \
  '~data.train_dataset.sources.old_recovery' '~data.train_dataset.sources.old_curve' '~data.train_dataset.sources.start_relabel' \
  data.train_dataset.sources.replay.dataset.exclude=null '~callbacks.ft_select.sets.static' '~callbacks.ft_select.sets.curve' \
  callbacks.ft_select.sets.general.0.n=16 callbacks.ft_select.sets.dynamic.0.n=16 callbacks.ft_select.every_steps=100000 \
  callbacks.ft_select.stop_ref=general callbacks.ft_select.stop_exclude=null \
  trainer.max_steps=$STEPS trainer.logging_steps=5 trainer.eval_strategy=no trainer.save_strategy=no save_final=false ${SPEED_EXTRA:-}
rc=$?; kill $SYS 2>/dev/null
python - $T $rc <<'PY'
import json, statistics, sys
t, rc = sys.argv[1], int(sys.argv[2])
try: rows = [json.loads(l) for l in open(f"{t}/loss_components.jsonl")]
except FileNotFoundError: rows = []
s = [r["s_per_step"] for r in rows if "s_per_step" in r][1:]                   # the first line includes warm-up
out = {"rc": rc, "steps": rows[-1]["step"] if rows else 0, "s_per_step_median": round(statistics.median(s), 3) if s else None,
       "peak_mem_gb": max((r.get("peak_mem_gb", 0) for r in rows), default=None), "reference_s2_h200_italy": 2.75}
json.dump(out, open(f"{t}/SPEED.json", "w"), indent=1); print("SPEED", json.dumps(out))
PY
