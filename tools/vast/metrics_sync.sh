#!/usr/bin/env bash
# Copies every run's metrics, logs and eval outputs to B2 so that stopping or destroying a GPU box never loses them
# (user 10/27: "인스턴스 지운다고 메트릭들 잃지 않게 해줘"). Weights and optimizer state are skipped; they go up through
# each stage's own upload. Run once before any stop/destroy (idle_watchdog.sh and safe_destroy.sh call it), and in the
# background during training: METRICS_SRC=/workspace/runs METRICS_DST=b2://morai2026-alpamayo/runs/r2ft bash metrics_sync.sh --loop 1800
# The b2 CLI reads B2_APPLICATION_KEY_ID / B2_APPLICATION_KEY from the instance environment (vast account env vars).
# ssh sessions do not carry the container env (10/27: b2 failed with "Missing account data"): take the keys from PID 1
[ -z "${B2_APPLICATION_KEY:-}" ] && eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^B2_APPLICATION_KEY(_ID)?=' | sed 's/^/export /')"
SRC=${METRICS_SRC:?set METRICS_SRC to the run directory on this box}
DST=${METRICS_DST:?set METRICS_DST to the b2:// run prefix}
B2=${B2:-b2}
# .b2.sync.tmp: files another b2 sync is still downloading into SRC (10/27: s2/t2 shards mid-download failed every sync)
SKIP='.*\.(safetensors|bin|pt|npz|tar)$|.*\.b2\.sync\.tmp$|.*/(global_step|checkpoint-)[^/]*/.*|.*optim.*|.*/zero_[^/]*$|.*rng_state.*'
LOG=${METRICS_LOG:-/workspace/metrics_sync.log}
once() {
  [ -n "${METRICS_HOOK:-}" ] && { bash -c "$METRICS_HOOK" >> "$LOG" 2>&1 || echo "$(date '+%F %T') hook failed" >> "$LOG"; }   # e.g. ft_metrics.py CSVs + figures
  "$B2" sync --threads 8 --no-progress --exclude-regex "$SKIP" "$SRC" "$DST" >> "$LOG" 2>&1 \
    && echo "$(date '+%F %T') synced $SRC -> $DST" >> "$LOG" \
    || { echo "$(date '+%F %T') SYNC FAILED $SRC -> $DST" >> "$LOG"; return 1; }
}
if [ "${1:-}" = --loop ]; then
  while :; do once; sleep "${2:-1800}"; done
else
  once
fi
