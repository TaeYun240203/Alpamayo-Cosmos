#!/usr/bin/env bash
# The DAgger run on the training box (README "실행"): spool down from B2 every SPOOL_EVERY s, metrics up
# (tools/vast/metrics_sync.sh, never lost), the runner (each LoRA version up to B2: file, then latest.json). Same command
# = resume. The starting values are the user's (README proposals): LR RANK ALPHA BATCH PUBLISH_MIN MIN_ROWS must be set.
# Usage (cwd /workspace/train): RUN=<name> BASE=<deliver>/final LR=.. RANK=.. ALPHA=.. BATCH=.. PUBLISH_MIN=.. \
#   MIN_ROWS=.. setsid nohup bash dagger/run.sh > /workspace/dagger_run.log 2>&1 < /dev/null &
#   EXTRA: more runner flags (e.g. "--alert_cmd '...'"); stop cleanly: touch $OUT/STOP
set -uo pipefail
: "${RUN:?}" "${BASE:?}" "${LR:?}"; LORA=${LORA:-expert}   # user 10/28: expert only (llm+expert on hold)
: "${RANK:?}" "${ALPHA:?}" "${BATCH:?}" "${PUBLISH_MIN:?}" "${MIN_ROWS:?}"
W=/workspace; B=b2://morai2026-alpamayo/runs/dagger/$RUN; OUT=${OUT:-$W/out/dagger/$RUN}; SP=${SPOOL:-$W/spool/$RUN}
VENV=${VENV:-$W/alpamayo-recipes/recipes/alpamayo1_5_sft/a1_5_sft}; NG=${NG:-$(nvidia-smi -L | wc -l)}
ATTN=${ATTN:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | grep -q '^12' && echo sdpa || echo flash_attention_2)}
[ -z "${B2_APPLICATION_KEY:-}" ] && eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^B2_APPLICATION_KEY(_ID)?=' | sed 's/^/export /')"
export PYTHONPATH=$PWD HF_HOME=${HF_HOME:-$W/hf} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false      # as scripts/pipeline.sh
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
mkdir -p $OUT/metrics $SP
( while :; do b2 sync --threads 16 --no-progress $B/spool/ $SP/ >> $OUT/metrics/spool_sync.log 2>&1 || echo "$(date '+%F %T') SPOOL SYNC FAILED" >> $OUT/metrics/spool_sync.log
    sleep ${SPOOL_EVERY:-30}; done ) & SYNC=$!
METRICS_SRC=$OUT/metrics METRICS_DST=$B/metrics METRICS_LOG=$OUT/metrics_sync.log bash $W/tools/vast/metrics_sync.sh --loop ${METRICS_EVERY:-120} & MET=$!
eval $VENV/bin/torchrun --nproc_per_node $NG -m dagger.runner --base $BASE --spool $SP --out $OUT --data_root /data/step0 --r2_root /data/step0_r2 \
  --e2e_server $W/e2e/server --lora $LORA --attn $ATTN --lr $LR --rank $RANK --alpha $ALPHA --batch $BATCH --publish_min $PUBLISH_MIN \
  --min_rows $MIN_ROWS --b2 $B ${EXTRA:-}
rc=$?
kill $SYNC $MET 2> /dev/null
METRICS_SRC=$OUT/metrics METRICS_DST=$B/metrics METRICS_LOG=$OUT/metrics_sync.log bash $W/tools/vast/metrics_sync.sh   # last push
echo "$(date '+%F %T') runner exit $rc"; exit $rc
