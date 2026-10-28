#!/usr/bin/env bash
# Mac-side watcher of an R2 fine-tune box (paid-box rule 6, 10/27: alert on FAILURE states, not only on start), adapted
# from the 10/27 main session's watch_train.sh. Polls over ssh every 2 min and exits with one ALERT/STAGE line on the first
# event, so the owner session (Monitor / background task) is woken:
#   ALERT  box unreachable 3 polls | no speed-test or training step 30 min after RENTED (stop-loss: destroy) | HALT |
#          ft stop switch | pipeline process gone | Traceback/NCCL/OOM/NaN in the stage log | s/step > 1.2 x EXPECT
#          (median of the last 3 lines) | max GPU util < 20 % for 5 polls (10 min) while the pipeline runs
#   STAGE  "pipeline start" in chain.log, a new "===" line in pipeline.log (stage start/done), PIPELINE_DONE, QAD mini PASS
#   ALERT  also: QAD mini test FAIL / INCOMPLETE (user 10/28: tell the user; ft goes on, QAD comes ~2-3 h later)
# Usage: HP="root@<host> -p <port>" RENTED=<epoch of the rental> EXPECT=2.75 bash tools/vast/r2ft_watch.sh
HP=${HP:?ssh target, e.g. "root@1.2.3.4 -p 22"}; RENTED=${RENTED:-$(date +%s)}; EXPECT=${EXPECT:-3.3}
probe() {
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=10 $HP '
    O=/workspace/out/stageR2; L=$O/ft/loss_components.jsonl
    pgrep -f "scripts/[p]ipeline.sh" > /dev/null && echo pipe=1 || echo pipe=0
    grep -q "pipeline start" /workspace/chain.log 2>/dev/null && echo started=1 || echo started=0
    [ -f $O/HALT ] && echo "halt=$(cat $O/HALT | tr " " _)" || echo halt=0
    [ -f $O/ft/ft_stop.json ] && echo stop=1 || echo stop=0
    [ -f $O/PIPELINE_DONE ] && echo done=1 || echo done=0
    echo "qadmini=$(head -c 80 /workspace/out/qadmini/RESULT 2>/dev/null | tr " " _ || true)"
    echo "stage=$(grep "^===" /workspace/out/pipeline.log 2>/dev/null | tail -1 | cut -c25- | tr " " _)"
    echo "speedsteps=$(grep -c s_per_step /workspace/out/speed/loss_components.jsonl 2>/dev/null || echo 0)"
    echo "trainsteps=$(grep -c s_per_step $L 2>/dev/null || echo 0)"
    echo "sps=$(grep -o "\"s_per_step\": [0-9.]*" $L 2>/dev/null | tail -3 | awk "{print \$2}" | sort -n | sed -n 2p)"
    G=$(ls -t $O/logs/*.log 2>/dev/null | head -1)
    echo "err=$(tail -300 "$G" 2>/dev/null | grep -c -E "Traceback|NonFiniteStop|CUDA error|NCCL error|OutOfMemory|Watchdog caught")"
    echo "gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -n | tail -1)"
  ' 2>/dev/null | grep -E "^[a-z]+="
}
last=""; idle=0; fails=0; started0=""; qm0="-"   # started0 / qm0: as seen at the first poll (a restarted watcher does not re-alert)
echo "watch $(date +%T): $HP, rented $(date -r $RENTED +%T), expect $EXPECT s/step"
while :; do
  p=$(probe); v() { echo "$p" | sed -n "s/^$1=//p"; }
  if [ -z "$p" ]; then fails=$((fails + 1)); [ $fails -ge 3 ] && { echo "ALERT $(date +%T): box unreachable 3 polls"; exit 0; }; sleep 120; continue; fi
  fails=0; age=$(( $(date +%s) - RENTED )); [ -z "$started0" ] && started0=$(v started); [ "$qm0" = - ] && qm0=$(v qadmini)
  qm=$(v qadmini); [ -n "$qm" ] && [ "$qm" != "$qm0" ] && { echo "$(case $qm in PASS*) echo STAGE;; *) echo ALERT;; esac) $(date +%T): QAD mini test $qm (b2 runs/r2ft/qadmini/)"; echo "$p"; exit 0; }
  [ "$(v halt)" != 0 ] && { echo "ALERT $(date +%T): HALT $(v halt)"; echo "$p"; exit 0; }
  [ "$(v stop)" = 1 ] && { echo "ALERT $(date +%T): ft stop switch (general_wo_relabel ADE > s2 +25 % twice)"; echo "$p"; exit 0; }
  [ "$(v done)" = 1 ] && { echo "STAGE $(date +%T): PIPELINE_DONE"; echo "$p"; exit 0; }
  [ "$(v err)" != 0 ] && { echo "ALERT $(date +%T): error lines in the stage log"; echo "$p"; exit 0; }
  [ $age -gt 1800 ] && [ "$(v speedsteps)" = 0 ] && [ "$(v trainsteps)" = 0 ] && { echo "ALERT $(date +%T): no step 30 min after rental (stop-loss)"; echo "$p"; exit 0; }
  if [ "$(v started)" = 1 ]; then
    [ $started0 = 0 ] && { echo "STAGE $(date +%T): pipeline start"; echo "$p"; exit 0; }
    [ "$(v pipe)" = 0 ] && { echo "ALERT $(date +%T): pipeline process gone"; echo "$p"; exit 0; }
    s=$(v sps); [ -n "$s" ] && python3 -c "import sys; sys.exit(0 if $s > 1.2 * $EXPECT else 1)" && { echo "ALERT $(date +%T): $s s/step > 1.2 x $EXPECT"; echo "$p"; exit 0; }
    if [ "$(v gpu)" -lt 20 ]; then idle=$((idle + 1)); else idle=0; fi
    [ $idle -ge 5 ] && { echo "ALERT $(date +%T): max GPU util < 20 % for 10 min"; echo "$p"; exit 0; }
    st=$(v stage); [ -n "$last" ] && [ "$st" != "$last" ] && { echo "STAGE $(date +%T): $st"; echo "$p"; exit 0; }; last=$st
  fi
  sleep 120
done
