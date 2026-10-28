#!/usr/bin/env bash
# Stops this vast.ai instance (from inside, with its per-instance key) when nobody is using it or when it reaches its budget,
# so GPU hours never run while a question waits for an answer. stop keeps the disk; storage is still billed.
#   busy = GPU util > 10 %, or a work process runs, or /workspace/keepalive was touched in the last IDLE_MIN minutes.
#   idle for IDLE_MIN -> stop. Work processes but GPU <= 10 % for HUNG_MIN (stuck job, long build) -> stop.
#   estimated spend (hours since start_date x dph_total + EXTRA_USD for bandwidth) >= the smaller of BUDGET_USD and
#   the account cap minus PRIOR_USD (spent before this instance) minus MARGIN_USD -> WARN line only, never a stop
#   (user 10/27: credit running short is the user's call; the owner session DMs them on Discord).
#   Account cap (vast support, 10/24): grows per second of account age, $52 at 1 day, $340 at 7, $1,444 at 30
#   = 4 + 48 x age_days; CREATED = account creation (vastai show user: created_at / 1000).
# Usage on the instance: BUDGET_USD=40 EXTRA_USD=8 PRIOR_USD=0 nohup bash idle_watchdog.sh > /dev/null 2>&1 &
# RunPod pod: DPH_USD=63.92 BUDGET_USD=3000 nohup bash idle_watchdog.sh > /dev/null 2>&1 &   (stop keeps a network volume's /workspace)
IDLE_MIN=${IDLE_MIN:-15}; HUNG_MIN=${HUNG_MIN:-60}; BUDGET_USD=${BUDGET_USD:-40}; EXTRA_USD=${EXTRA_USD:-0}
PRIOR_USD=${PRIOR_USD:-0}; MARGIN_USD=${MARGIN_USD:-3}; CREATED=${CREATED:-1792829102}
LOG=/workspace/watchdog.log
PAT='torchrun|morai_vla|pytest|uv (sync|pip|venv)|hf download|nvcc|cicc|ninja|convert_checkpoint|scripts/[A-Za-z0-9_]+\.py|git clone|apt-get|b2 sync'
# ssh sessions may not carry the container env: take the per-instance key and id from PID 1
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(CONTAINER_(API_KEY|ID)|RUNPOD_(API_KEY|POD_ID))=[A-Za-z0-9_-]+$' | sed 's/^/export /')"
if [ -n "${RUNPOD_POD_ID:-}" ]; then   # RunPod: pod-scoped key, rate from DPH_USD (pod $/h), no account cap (RunPod's is $80/h)
  START=$(date +%s); DPH=${DPH_USD:?set DPH_USD to the pod price per hour}; CREATED=0; PRIOR_USD=-1000000
  stop() { curl -s -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" -H "Authorization: Bearer $RUNPOD_API_KEY"; }
  bal() { curl -s https://api.runpod.io/graphql -H 'content-type: application/json' -H "Authorization: Bearer $RUNPOD_API_KEY" \
          -d '{"query":"query { myself { clientBalance } }"}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["myself"]["clientBalance"])' 2>/dev/null; }
else
  bal() { :; }
  V="vastai --api-key $CONTAINER_API_KEY"
  read -r START DPH < <($V show instance "$CONTAINER_ID" --raw | python3 -c 'import json,sys; d=json.load(sys.stdin); print(int(d["start_date"]), d["dph_total"])')
  stop() { $V stop instance "$CONTAINER_ID"; }
fi
idle_since=$(date +%s); gpu_since=$idle_since
echo "$(date '+%F %T') watchdog up: dph_total=$DPH budget=\$$BUDGET_USD extra=\$$EXTRA_USD idle=${IDLE_MIN}m hung=${HUNG_MIN}m" >> $LOG
while sleep ${TICK:-60}; do
  now=$(date +%s)
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -n | tail -1); util=${util:-0}
  work=0; pgrep -f -- "$PAT" > /dev/null && work=1
  keep=0; [ -e /workspace/keepalive ] && [ $((now - $(stat -c %Y /workspace/keepalive))) -lt $((IDLE_MIN * 60)) ] && keep=1
  [ "$util" -gt 10 ] && gpu_since=$now
  { [ "$util" -gt 10 ] || [ $work = 1 ] || [ $keep = 1 ]; } && idle_since=$now
  spent=$(python3 -c "print(round(($now - $START) / 3600 * $DPH + $EXTRA_USD, 2))")
  limit=$(python3 -c "print(round(min($BUDGET_USD, 4 + 48 * ($now - $CREATED) / 86400 - $PRIOR_USD - $MARGIN_USD), 2))")
  why=""
  [ $((now - idle_since)) -ge $((IDLE_MIN * 60)) ] && why="idle ${IDLE_MIN} min"
  [ $work = 1 ] && [ $keep = 0 ] && [ $((now - gpu_since)) -ge $((HUNG_MIN * 60)) ] && why="work processes but GPU idle ${HUNG_MIN} min"
  warn=""; python3 -c "import sys; sys.exit(0 if $spent >= $limit else 1)" && warn="estimated spend \$$spent >= limit \$$limit"
  echo "$(date '+%F %T') util=$util work=$work keep=$keep spent=\$$spent limit=\$$limit bal=$(bal) ${warn:+WARN: $warn} ${why:+STOP: $why}" >> $LOG
  # metrics and logs reach B2 before the box stops (METRICS_SRC/METRICS_DST as for metrics_sync.sh; 10 min cap)
  if [ -n "$why" ]; then
    [ -n "${METRICS_SRC:-}" ] && timeout 600 bash "$(dirname "$0")/metrics_sync.sh" >> $LOG 2>&1
    sync; stop >> $LOG 2>&1; sleep 600
  fi
done
