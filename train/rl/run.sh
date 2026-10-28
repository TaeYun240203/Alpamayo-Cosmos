#!/usr/bin/env bash
# The PPO learner on the training box (PPO_PLAN 2): the actors' spool down from B2 every SPOOL_EVERY s, metrics up
# (tools/vast/metrics_sync.sh; LoRA files and state.pt are skipped there), the learner on every GPU (torchrun) publishing
# each LoRA version to B2 runs/ppo/$RUN/lora/ (file, then latest.json). Same command = resume (state.pt). It stops by
# itself after STALE_MIN without new episodes (the idle watchdog then stops the box).
# Usage (cwd /workspace/train):
#   RUN=<name> BASE=<deliver>/final [WARM_B2=b2://.../warm-start spool] [KL_TARGET=14.4] [EXTRA="--lr 1e-5 ..."] \
#     setsid nohup bash rl/run.sh > /workspace/ppo_run.log 2>&1 < /dev/null &
#   KL_TARGET: nats per decision; user 10/28: a mean shift of 0.1-0.2 sigma per step = 6.4-25.6 (14.4 = 0.15 sigma)
set -uo pipefail
: "${RUN:?}" "${BASE:?}"
W=/workspace; B=b2://morai2026-alpamayo/runs/ppo/$RUN; OUT=${OUT:-$W/out/ppo/$RUN}; SP=${SPOOL:-$W/spool/ppo/$RUN}
VENV=${VENV:-$W/alpamayo-recipes/recipes/alpamayo1_5_sft/a1_5_sft}; NG=${NG:-$(nvidia-smi -L | wc -l)}
[ -z "${B2_APPLICATION_KEY:-}" ] && eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^B2_APPLICATION_KEY(_ID)?=' | sed 's/^/export /')"
export PYTHONPATH=$PWD HF_HOME=${HF_HOME:-$W/hf} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
mkdir -p $OUT $SP
WARM=()
if [ -n "${WARM_B2:-}" ] && [ ! -f $OUT/state.pt ]; then
  b2 sync --threads 16 --no-progress "$WARM_B2" $W/spool/ppo_warm/$RUN/ && WARM=(--warmstart $W/spool/ppo_warm/$RUN)
fi
# Spool down through a staging mirror: one b2 sync may bring an episode's done.json before its dumps, but the actors put
# done.json up only after the rest (ppo_relay.sh), so after a sync's exit 0 an episode with done.json is whole: only then
# it is hard-linked into the learner's spool (inflight.json / episodes.jsonl copied with their mtimes, the gate reads them).
ST=$W/spool/ppo_stage/$RUN; mkdir -p $ST
( while :; do
    if b2 sync --threads 16 --no-progress $B/spool/ $ST/ >> $OUT/spool_sync.log 2>&1; then
      for d in $ST/*/ep*/done.json; do
        [ -e "$d" ] || continue; e=$(dirname "$d"); a=$SP/$(basename "$(dirname "$e")")
        [ -e "$a/$(basename "$e")" ] || { mkdir -p "$a" && cp -al "$e" "$a/"; }
      done
      for f in $ST/*/inflight.json $ST/*/episodes.jsonl; do
        [ -e "$f" ] && mkdir -p "$SP/$(basename "$(dirname "$f")")" && cp -p "$f" "$SP/$(basename "$(dirname "$f")")/"
      done
    else echo "$(date '+%F %T') SPOOL SYNC FAILED" >> $OUT/spool_sync.log; fi
    sleep ${SPOOL_EVERY:-30}; done ) & SYNC=$!
METRICS_SRC=$OUT METRICS_DST=$B/learner METRICS_LOG=$OUT/metrics_sync.log bash $W/tools/vast/metrics_sync.sh --loop ${METRICS_EVERY:-120} & MET=$!
$VENV/bin/torchrun --nproc_per_node $NG -m rl.learner --base $BASE --spool $SP --out $OUT --e2e_server $W/e2e/server --b2 $B \
  --kl_target ${KL_TARGET:-14.4} --stale_stop_min ${STALE_MIN:-30} "${WARM[@]}" ${EXTRA:-} 2>&1 | tee -a $OUT/learner.log
rc=${PIPESTATUS[0]}
kill $SYNC $MET 2> /dev/null
METRICS_SRC=$OUT METRICS_DST=$B/learner METRICS_LOG=$OUT/metrics_sync.log bash $W/tools/vast/metrics_sync.sh   # last push
echo "$(date '+%F %T') learner exit $rc"; exit $rc
