#!/usr/bin/env bash
# start N step-0 workers on winpc (each in its own chroot namespace); safe to rerun: finished episodes are skipped
D=$(cd "$(dirname "$0")" && pwd); N=${1:-16}; L=/mnt/d/share/${STEP0_DIR:-2026_MORAI_step0}/_logs   # R2: STEP0_DIR=2026_MORAI_step0_R2
mkdir -p "$L"
for k in $(seq 0 $((N-1))); do nohup "$D/run.sh" step0_worker.py "$k" "$N" > "$L/stdout_$k.txt" 2>&1 & done
echo "started $N workers"
