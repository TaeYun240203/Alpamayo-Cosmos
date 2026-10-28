#!/usr/bin/env bash
# Dry run of the student queue (scripts/pipeline_student.sh, the real stage functions) on one GPU, a few steps per stage.
#   TINY=1 (default): tiny models; $OUT/t2/final = a tiny teacher (README student local validation).
#   TINY=0: full size (sm90/sm100 smoke on a 1x pod); $OUT/t2/final = T* or its stand-in (the 1.5 release dir).
# Usage (cwd train/): STAGES="soft s1 sgate s2 qad deliver" [TINY=0] bash scripts/student_dryrun.sh <out dir> <step-0 root> [steps]
set -uo pipefail
OUT=$1; ROOT=$2; COCP=${COCP:-$ROOT/_coc/coc_dataset.parquet}; NG=1; B2=${B2:-none}
TR="torchrun --master_port ${PORT:-29531} --nproc_per_node 1 -m morai_vla.train"
need() { [ -f $OUT/$1/final/config.json ] || { echo "missing $OUT/$1/final"; exit 1; }; }
halt() { echo "HALT: $*"; exit 1; }
resume_flag() { ls -d $1/checkpoint-* > /dev/null 2>&1 && echo true || echo null; }
mkdir -p $OUT/logs
STEPS=${3:-200}; SMALL="trainer.max_steps=$STEPS trainer.eval_steps=$((STEPS / 2)) trainer.save_steps=$((STEPS / 2)) trainer.logging_steps=10"
QEXTRA="quant.calib_samples=${CALIB:-16} quant.num_calib_steps=8 quant.num_score_steps=4 ${QEXTRA:-}"
S1EXTRA="callbacks.traj_early_stop.every_steps=$((STEPS / 4)) callbacks.traj_early_stop.decay_steps=$((STEPS / 8)) +callbacks.traj_early_stop.n=${ES_N:-16} ${S1EXTRA:-}"
if [ "${TINY:-1}" = 1 ]; then SEXTRA="+tiny=student $SMALL ${SEXTRA:-}"; SGATE_N=${SGATE_N:-8}; QAD_GATE_N=${QAD_GATE_N:-8}
else SEXTRA="$SMALL data.train_dataset.max_samples=${ROWS:-64} data.val_dataset.max_samples=16 ${ONE_GPU_OPT:-trainer.deepspeed=null +trainer.optim=sgd} ${SEXTRA:-}"   # one GPU: no ZeRO sharding of AdamW
     SGATE_N=${SGATE_N:-32}; QAD_GATE_N=${QAD_GATE_N:-16}; fi
source scripts/pipeline_student.sh
for s in ${STAGES:-soft s1 sgate s2}; do
  t=$(date +%s); $s > $OUT/logs/$s.log 2>&1 || halt "$s exited non-zero ($OUT/logs/$s.log)"
  ! declare -F gate_$s > /dev/null || gate_$s
  echo "== $s ok in $(( $(date +%s) - t )) s: $(tr -d '\n ' < $OUT/$s/metrics.json | cut -c1-400)"
done
