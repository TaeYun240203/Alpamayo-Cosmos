#!/usr/bin/env bash
# Destroys a vast instance only after its metrics and logs are on B2 (user 10/27). Run from the Mac:
#   bash tools/vast/safe_destroy.sh <INSTANCE_ID> <run dir on the box> <b2:// run prefix>
# It runs metrics_sync.sh on the box over ssh, lists the B2 prefix, and destroys only when the sync succeeded.
# FORCE=1 skips the sync (only for a box that never ran anything, e.g. one with no network).
set -euo pipefail
ID=${1:?instance id}; SRC=${2:?run dir on the box}; DST=${3:?b2:// prefix}
if [ "${FORCE:-0}" != 1 ]; then
  url=$(vastai ssh-url "$ID")                      # ssh://root@host:port
  hp=${url#ssh://}; host=${hp%:*}; port=${hp##*:}
  ssh -o StrictHostKeyChecking=no -p "$port" "$host" \
    "METRICS_SRC='$SRC' METRICS_DST='$DST' bash /workspace/tools/vast/metrics_sync.sh && tail -1 /workspace/metrics_sync.log"
  echo "B2 now holds:"; ssh -o StrictHostKeyChecking=no -p "$port" "$host" "eval \"\$(tr '\\0' '\\n' < /proc/1/environ | grep -E '^B2_APPLICATION_KEY(_ID)?=' | sed 's/^/export /')\"; b2 ls -r '$DST' | grep -v -E '\.(safetensors|bin|pt)$' | wc -l"
fi
echo y | vastai destroy instance "$ID"
