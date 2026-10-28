#!/usr/bin/env bash
# Stage C, unattended and resumable, on one N-GPU box (README "단계 C 파이프라인"):
#   t1t -> t1c -> t2 -> tgate -> soft -> s1 -> sgate -> s2 -> qad -> deliver (final export, right after QAD) -> qgate
#   (user 10/27, 10/28: the QAD ADE gate after the early delivery; its result lands in deliver/gate.json)
# Per stage: skip if done (marker in $OUT/done or on B2) -> run -> automatic gate -> push $OUT/<stage> to B2 without
# checkpoint-* -> verify (file count and bytes) -> done marker, local and on B2 -> delete its checkpoint-* (100-160 GB each;
# all stages' would not fit the 1.5 TB volume). A failure or tripped gate writes
# $OUT/HALT, pushes the logs and exits non-zero; the idle watchdog then stops the box. Re-running resumes after the last
# completed stage, on this box (in-stage checkpoints: resume=true) or on a new one (inputs pulled from B2).
# The student stages (soft .. deliver) are functions in scripts/pipeline_student.sh, owned by the student code.
# Needs: b2 CLI on PATH, authorized via B2_APPLICATION_KEY_ID / B2_APPLICATION_KEY (sourced from $W/b2.env).
# Usage: setsid nohup bash scripts/pipeline.sh > /workspace/out/pipeline.log 2>&1 < /dev/null &
#        STAGES="t1t t1c" bash scripts/pipeline.sh     (a subset)
#        dry run: EXTRA="trainer.max_steps=10 ..." TGATE_N=16 OUT=/workspace/out/dry B2=b2://<bucket>/runs/dry bash scripts/pipeline.sh
set -uo pipefail
ulimit -n $(ulimit -Hn)                                # QAD dataloader workers die at the 1,024 default (10/27); some boxes cap the hard limit below 1M
W=${W:-/workspace}                                     # box layout (VAST_SETUP paths); every path below can be overridden
source ${VENV:-$W/alpamayo-recipes/recipes/alpamayo1_5_sft/a1_5_sft}/bin/activate
[ -f $W/b2.env ] && set -a && source $W/b2.env && set +a
[ -n "${B2_APPLICATION_KEY_ID:-}" ] || eval "$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | grep -E '^B2_APPLICATION_KEY(_ID)?=' \
  | sed -E "s/^([A-Z0-9_]+)=(.*)$/export \1='\2'/")"            # RunPod secrets live in PID 1's env, not in ssh sessions
cd ${TRAIN:-$W/train}
export PYTHONPATH=$PWD HF_HOME=${HF_HOME:-$W/hf} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}   # GC-off Stage 1 peaks at 121 of 140 GB on H200: no fragmentation slack
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}          # 8x H200 10/25: a ZeRO-2 all-reduce hung 10 min with NVLS on (NCCL 2.27.3); off: 350 GB/s, plenty
OUT=${OUT:-$W/out/stageC}; B2=${B2:-b2://morai2026-alpamayo/runs/stageC}
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}   # e.g. GPUS=0,1,2,4,5,6,7 to leave out a sick GPU
export CUDA_VISIBLE_DEVICES=$GPUS; NG=$(echo $GPUS | tr , '\n' | wc -l)
REL=${REL:-$W/ckpt/Alpamayo-1.5-10B}; STEP2=${STEP2:-$W/ckpt/step2_final}; ROOT=${ROOT:-/data/step0}; COCP=${COCP:-/data/coc/coc_v1_1/coc_dataset.parquet}
C="paths.alpamayo15=$REL data.root=$ROOT data.coc_parquet=$COCP"
TR="torchrun --nproc_per_node $NG -m morai_vla.train"
# epochs (README budget); override from the environment
T1T_STOP=${T1T_STOP:-16000}; T1T_ANNEAL=${T1T_ANNEAL:-1500}   # user 10/25: t1t main run stops at T1T_STOP, then a short anneal
T1T_EPOCHS=${T1T_EPOCHS:-3}; T1C_EPOCHS=${T1C_EPOCHS:-3}; T2_EPOCHS=${T2_EPOCHS:-3}; TGATE_N=${TGATE_N:-1000}
EXTRA=${EXTRA:-}                                       # Hydra overrides appended to every teacher training run (dry runs)
# Stage 1 speed knobs (README "B200 최적화"): per-device batch (global = batch x GPUs, recipe 4/GPU), checkpointing, attention
T1_BATCH=${T1_BATCH:-4}; T1_GC=${T1_GC:-false}; T1_ATTN=${T1_ATTN:-flash_attention_2}   # B200 sweep: GC off 2.1x, est. 116 GB/GPU
T1="trainer.per_device_train_batch_size=$T1_BATCH trainer.gradient_accumulation_steps=$((4 / T1_BATCH > 0 ? 4 / T1_BATCH : 1)) trainer.gradient_checkpointing=$T1_GC model.attn_implementation=$T1_ATTN"
RELEASE_ADE=${RELEASE_ADE:-5.29}                       # 1.5 release, LiDAR copy, same 1,000 val samples (README 단계 B 결과); dry runs only raise it
ALL="t1t t1c t2 tgate soft s1 sgate s2 qad deliver qgate"
mkdir -p $OUT/done $OUT/logs

log() { echo "=== $(date '+%F %T') $*"; }
push() {                                               # $1 = stage dir name; verify count + bytes of what was sent
  local d=$OUT/$1
  local i; for i in 1 2 3; do                          # B2 answers 429 TooManyRequests to many parallel large-file parts: retry, sync skips done files
    b2 sync --threads 8 --exclude-dir-regex '(.*/)?(checkpoint-[0-9]+|ade_snapshots|ft_snapshots)' $d $B2/$1 > $OUT/push_$1.log 2>&1 && break   # log outside logs/ (pushed too)
    [ $i = 3 ] && return 1; sleep 60
  done
  local want got
  want=$(cd $d && find . -type f -not -path '*/checkpoint-*' -not -path '*/ade_snapshots/*' -not -path '*/ft_snapshots/*' -printf '%s\n' | awk '{n++; s+=$1} END {print n+0, s+0}')
  got=$(b2 ls --recursive --long $B2/$1/ | grep -v -e '/checkpoint-[0-9]*/' -e '/ade_snapshots/' -e '/ft_snapshots/' | awk '{n++; s+=$5} END {print n+0, s+0}')
  [ "$want" = "$got" ] || { echo "push $1: local $want != B2 $got"; return 1; }
}
halt() {
  log "HALT: $*"; echo "$(date '+%F %T') $*" > $OUT/HALT
  wait_pushes; cp $W/out/pipeline.log $OUT/logs/ 2>/dev/null; push logs; exit 1
}
PUSHES=()                                              # PUSH_ASYNC=1 (10/28): a stage's push runs while the next stage trains
wait_pushes() { local p; for p in "${PUSHES[@]}"; do wait $p; done; [ ! -f $OUT/PUSH_FAILED ]; }
finish() {                                             # $1 = stage: push + verify, done marker (local and B2), drop its checkpoints
  push $1 || return 1
  date '+%F %T' > $OUT/done/$1; push done || return 1
  python -c "import glob, shutil, sys; [shutil.rmtree(d) for d in glob.glob(sys.argv[1] + '/checkpoint-*') + glob.glob(sys.argv[1] + '/ade_snapshots')]" \
    $OUT/$1                                            # resume-only / per-measurement weights, final/ is on B2 (user 10/25: t1t's stop point not kept either)
}
is_done() { [ -f $OUT/done/$1 ]; }
need() {                                               # an input stage's final/ dir: local, else from B2
  [ -f $OUT/$1/final/config.json ] && return 0
  log "pull $1 from B2"; b2 sync --threads 16 $B2/$1/final $OUT/$1/final > $OUT/logs/pull_$1.log 2>&1 || halt "cannot pull $1/final"
}
resume_flag() { ls -d $1/checkpoint-* > /dev/null 2>&1 && echo true || echo null; }
train_gate() { python scripts/pipeline_gate.py train $OUT/$1 ${2:-} || halt "$1 gate: $(python scripts/pipeline_gate.py train $OUT/$1 ${2:-})"; }

# ------------------------------------------------------------------ teacher
smoke8() {                                             # first minutes on a new box (STAGES=smoke8): NCCL/ZeRO-2 on all GPUs, Stage 1
  $TR --config-name stage1t $C model.init=$STEP2 paths.output_dir=$OUT/smoke8 $T1 trainer.max_steps=10 trainer.logging_steps=2 \
    trainer.eval_steps=5 data.val_dataset.max_samples=16 callbacks.stage1_eval.n=16 trainer.save_strategy=no save_final=false \
    trainer.load_best_model_at_end=false
}                                                      # memory at the real settings, the Stage 1 monitor under torchrun, B2 push
gate_smoke8() { train_gate smoke8 --no-divergence; }
data_complete() {                                      # datasets enumerate episodes once at start: a partial copy would drop the rest for good
  python - $ROOT <<'PY'
import sys, pandas as pd
from pathlib import Path
r = Path(sys.argv[1]); eps = pd.read_parquet(r / "_index/index.parquet", columns=["episode"]).episode.unique()
miss = [e for e in eps if not all((r / e / f).exists() for f in ("cams.bin", "cams_index.npz", "lidar.bin", "lidar_index.npz"))]
print(f"{len(eps) - len(miss)} of {len(eps)} index episodes present"); sys.exit(len(miss) > 6)   # 6 have no step-0 data at all (README)
PY
}
t1t() {                                                # Stage 1-T (user 10/25): main run stopped at checkpoint-$T1T_STOP, short anneal, pick by ADE
  data_complete || { echo "data copy incomplete"; return 1; }
  local ck=$OUT/t1t/checkpoint-$T1T_STOP
  if [ ! -f $ck/trainer_state.json ]; then             # the 3-epoch recipe run; it is stopped by hand right after this save
    $TR --config-name stage1t $C model.init=$STEP2 paths.output_dir=$OUT/t1t trainer.num_train_epochs=$T1T_EPOCHS $T1 trainer.eval_steps=2000 \
      +trainer.eval_on_start=true trainer.save_steps=2000 trainer.save_total_limit=1 resume=$(resume_flag $OUT/t1t) $EXTRA || return 1
    [ -f $ck/trainer_state.json ] || { echo "main run ended without $ck"; return 1; }
  fi
  [ -f $OUT/t1t/anneal/final/config.json ] || t1t_anneal $ck || return 1
  t1t_select
}
t1t_anneal() {                                         # resume $1 (optimizer, data order), LR linear from its scheduled value to 0 in T1T_ANNEAL steps
  local r w n=$((T1T_STOP + T1T_ANNEAL))               # HF "linear": lr = base * (N - step) / (N - W) after W warmup; W so that step $T1T_STOP gives r
  r=$(python -c "import json, sys; h = [e['learning_rate'] for e in json.load(open(sys.argv[1]))['log_history'] if 'learning_rate' in e]; print(h[-1] / max(h))" $1/trainer_state.json)
  w=$(python -c "print(round($T1T_STOP + $T1T_ANNEAL - $T1T_ANNEAL / $r))"); echo "anneal from $1: lr ratio $r, warmup_steps $w"
  $TR --config-name stage1t $C model.init=$STEP2 paths.output_dir=$OUT/t1t/anneal $T1 resume=$1 trainer.max_steps=$n \
    trainer.lr_scheduler_type=linear ~trainer.lr_scheduler_kwargs.min_lr trainer.warmup_steps=$w trainer.eval_strategy=no trainer.save_strategy=no \
    trainer.load_best_model_at_end=false save_final=true $EXTRA
}
t1t_select() {                                         # discrete-token ADE on the 1,000 fixed val samples, GPUs in parallel, one candidate at a time
  local c k pids
  local -A W=([step12000]=$OUT/t1t_snapshots/step-12000 [ckpt$T1T_STOP]=$OUT/t1t/checkpoint-$T1T_STOP [anneal]=$OUT/t1t/anneal/final)
  mkdir -p $OUT/t1t/select
  for c in "${!W[@]}"; do
    pids=()
    for k in $(seq 0 $((NG - 1))); do
      CUDA_VISIBLE_DEVICES=$(echo $GPUS | cut -d, -f$((k + 1))) python scripts/eval_stage1.py ${W[$c]} $ROOT --n 1000 --shard $k,$NG \
        --out $OUT/t1t/select/$c.shard$k.json > $OUT/logs/t1t_select_${c}_$k.log 2>&1 & pids+=($!)
    done
    for p in "${pids[@]}"; do wait $p || return 1; done
  done
  python scripts/pipeline_gate.py select $OUT/t1t $(for c in "${!W[@]}"; do echo "$c=${W[$c]}"; done)
}
gate_t1t() {                                           # no CE divergence halt (val CE rises while ADE falls); NaN, or ADE not below constant velocity
  train_gate t1t --no-divergence; train_gate t1t/anneal --no-divergence
  python scripts/pipeline_gate.py max $OUT/t1t/best.json ade_minus_cv 0 || halt "t1t: chosen discrete-token ADE not below constant velocity"
}
t1c() {                                                # Stage 1-C: CoC keyframes; final = best eval checkpoint (memorises fast)
  need t1t
  $TR --config-name stage1c $C model.init=$OUT/t1t/final paths.output_dir=$OUT/t1c trainer.num_train_epochs=$T1C_EPOCHS $T1 trainer.save_total_limit=1 \
    resume=$(resume_flag $OUT/t1c) $EXTRA
}
gate_t1c() { train_gate t1c --no-divergence; }
t2() {                                                 # Stage 2: expert flow matching, VLM frozen, no CoC (recipe: batch 1/GPU, no DeepSpeed);
                                                       # trajectory-ADE early stop + decay inside the run, final = lowest ADE (configs/stage2.yaml)
  need t1c
  $TR --config-name stage2 $C model.init=$OUT/t1c/final paths.output_dir=$OUT/t2 trainer.num_train_epochs=$T2_EPOCHS \
    trainer.eval_steps=2000 trainer.save_steps=2000 trainer.save_total_limit=1 resume=$(resume_flag $OUT/t2) $EXTRA
}
gate_t2() { train_gate t2; }
tgate() {                                              # T* on the 1,000 val samples: expert ADE LiDAR on/off + expert-free metrics
  need t2; local k pids=()
  for k in $(seq 0 $((NG - 1))); do
    CUDA_VISIBLE_DEVICES=$(echo $GPUS | cut -d, -f$((k + 1))) python scripts/check_release.py --release $REL --ckpt $OUT/t2/final --root $ROOT --no-quant --n $TGATE_N \
      --variants traj,traj_nolidar --shard $k,$NG --out $OUT/tgate/shard$k > $OUT/logs/tgate_$k.log 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || return 1; done
  python scripts/pipeline_gate.py merge $OUT/tgate $OUT/tgate/shard* || return 1
  CUDA_VISIBLE_DEVICES=${GPUS%%,*} python scripts/eval_stage1.py $OUT/t2/final $ROOT --n $((TGATE_N < 200 ? TGATE_N : 200)) --out $OUT/tgate/stage1_eval.json
}
gate_tgate() {
  python scripts/pipeline_gate.py max $OUT/tgate/summary.json traj_ade $RELEASE_ADE \
    || halt "T* expert ADE (LiDAR on) not better than the release: $(python scripts/pipeline_gate.py max $OUT/tgate/summary.json traj_ade $RELEASE_ADE)"
}

# ------------------------------------------------------------------ student (soft s1 sgate s2 ft qad deliver qgate, gate_<stage>)
source scripts/pipeline_student.sh

# ------------------------------------------------------------------ queue
log "pipeline start: $NG GPUs ($GPUS), OUT=$OUT, B2=$B2"
b2 account get > /dev/null 2>&1 || halt "b2 CLI not authorized"
b2 sync $B2/done $OUT/done > /dev/null 2>&1                                   # markers of a previous box
( while kill -0 $$ 2> /dev/null; do touch $W/keepalive; sleep 120; done ) &   # gates/pushes are not GPU work
for s in ${STAGES:-$ALL}; do
  if is_done $s; then log "$s done before, skipped"; continue; fi
  log "$s start"; t=$(date +%s); mkdir -p $OUT/$s
  $s > $OUT/logs/$s.log 2>&1 || halt "$s exited non-zero (logs/$s.log)"
  ! declare -F gate_$s > /dev/null || gate_$s
  echo "{\"stage\": \"$s\", \"hours\": $(python -c "print(round(($(date +%s) - $t) / 3600, 2))"), \"gpus\": $NG}" > $OUT/$s/stage_time.json
  cp $OUT/logs/$s.log $OUT/$s/ 2>/dev/null
  if [ "${PUSH_ASYNC:-0}" = 1 ]; then                 # one push at a time (flock); the next stage reads this one's local final/
    ( flock 9; finish $s || echo "$(date '+%F %T') B2 push of $s (or its done marker) failed" > $OUT/PUSH_FAILED ) 9> $OUT/.push.lock & PUSHES+=($!)
  else
    finish $s || halt "B2 push of $s (or its done marker) failed"
  fi
  log "$s done in $(python -c "print(round(($(date +%s) - $t) / 3600, 2))") h"
  [ "$(cat $OUT/stop_after 2>/dev/null)" = "$s" ] && { wait_pushes; log "stop_after $s: queue stops here"; exit 0; }   # echo t1c > $OUT/stop_after
done
wait_pushes || halt "$(cat $OUT/PUSH_FAILED)"
cp $W/out/pipeline.log $OUT/logs/ 2>/dev/null; push logs
log "pipeline finished"; touch $OUT/PIPELINE_DONE
