#!/usr/bin/env bash
# Zero-delay chain for the R2 fine-tune box (user 10/27: "다운로드 완료 즉시 본학습 돌입", no agent or human in the loop),
# restored from the 10/27 Texas box (B2 runs/r2ft/pro6000tx_57966802/chain.sh). Reads /workspace/stage.log (r2ft_stage.sh):
#  A) once venv, student pkgs, checkpoints, step0_r2 and HF are in while R1 still downloads: the QAD R2 mini test on GPU 0
#     (r2ft_qadmini.sh, user 10/28 "이게 좋겠어") and the speed test (r2ft_speed.sh) on GPUs 1.. at the same time: one after
#     the other they would run into the ft start on a fast line (R1 in ~27-35 min); per-GPU work, so s/step compares
#  B) the second R1 step0 lands (rc=0): stop the speed test, check every other staging part ended rc=0 (waits for any still
#     running), start scripts/pipeline.sh STAGES="ft qad deliver qgate" (delivery right after QAD, then the gate). ft_select measures step 0 itself (~3-7 min, ~1-2 with
#     FAST): the 10/27 H200 values have no general_wo_relabel, which the stop switch reads since 10/28.
# Usage on the box (after r2ft_stage.sh started): setsid nohup bash /workspace/tools/vast/r2ft_chain.sh > /workspace/chain.log 2>&1 < /dev/null &
#   ATTN: attention for every student command (default by GPU: sm_12x -> sdpa, else flash_attention_2)
#   FAST=1 (only after scripts/ft_speed_probe.py passed on a GPU, README "R2 속도 최적화"): morai_vla/speed.py on, ft_select
#     measures 4 rows per prefill (step 0 too, so the reference uses the same method), the QAD gates on every GPU, stage
#     pushes in the background
L=/workspace/stage.log; W=/workspace; OUT=$W/out/stageR2
log() { echo "$(date '+%F %T') $*"; }
ok() { grep -qE "^[0-9:]+ $1 rc=0" $L; }
ATTN=${ATTN:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | grep -q '^12' && echo sdpa || echo flash_attention_2)}
FAST=${FAST:-0}; NGPU=$(nvidia-smi -L | wc -l)
[ "$FAST" = 1 ] && FASTX="speed.vit_sdpa_grouped=${FAST_VIT:-true} speed.teacher_vit_only=${FAST_TEACHER:-true} speed.lazy_loss_log=${FAST_LAZY:-true}"   # one off: FAST_TEACHER=false
log "chain up: attention $ATTN, FAST=$FAST"
ready() { [ "$(grep -cE '^[0-9:]+ (venv|student pkgs|ckpt|step0_r2|hf ckpt) rc=0' $L)" -ge 5 ]; }
( until ready; do sleep 2; done                    # QADMINI=0 (10/28 stop round: the QAD path ran on the R2 box) skips it
  [ "${QADMINI:-1}" = 0 ] || grep -qE '^[0-9:]+ step0 rc=' $L || { log "QAD mini test start (GPU 0)"; ATTN=$ATTN QADMINI_GPU=0 bash $W/tools/vast/r2ft_qadmini.sh > $W/out/qadmini.log 2>&1
    log "QAD mini test: $(cat $W/out/qadmini/RESULT 2>/dev/null)"; } ) &
( until ready; do sleep 2; done
  grep -qE '^[0-9:]+ step0 rc=' $L || { log "speed test start (GPUs 1..$((NGPU - 1)))"; CUDA_VISIBLE_DEVICES=$(seq -s, 1 $((NGPU - 1))) NP=$((NGPU - 1)) \
    ATTN=$ATTN SPEED_EXTRA="${FASTX:-}" bash $W/tools/vast/r2ft_speed.sh > $W/out/speed.log 2>&1; log "speed test end: $(cat $W/out/speed/SPEED.json 2>/dev/null | tr -d '\n ')"; } ) &
until grep -qE '^[0-9:]+ step0 rc=' $L; do sleep 1; done
ok step0 || { log "R1 step0 sync FAILED: $(grep -E '^[0-9:]+ step0 rc=' $L)"; exit 1; }
log "R1 step0 sync done"
pkill -f "workspace/out/[s]peed"; pkill -f "tools/vast/[r]2ft_speed.sh"
if [ -d $W/out/qadmini ] && [ ! -f $W/out/qadmini/RESULT ]; then             # never blocks ft: stop it, say so (watcher alerts)
  pkill -f "tools/vast/[r]2ft_qadmini.sh"; pkill -f "out/[q]admini"; echo "INCOMPLETE: R1 arrived first ($(ls $W/out/qadmini/*.log 2>/dev/null | xargs -n1 basename | tr '\n' ' '))" > $W/out/qadmini/RESULT
  log "QAD mini test stopped: $(cat $W/out/qadmini/RESULT)"; ( b2 sync --no-progress --exclude-regex '.*\.(safetensors|bin|pt|pth|npz)$' $W/out/qadmini/ b2://morai2026-alpamayo/runs/r2ft/qadmini/$(cat $W/RUN)/ > /dev/null 2>&1 & )
fi
while pgrep -f "workspace/out/[s]peed|out/[q]admini" > /dev/null; do sleep 1; done
for p in venv "student pkgs" ckpt step0_r2 "hf ckpt"; do                      # normally long done; a failed part halts here, not mid-run
  until grep -qE "^[0-9:]+ $p rc=" $L; do sleep 2; done
  ok "$p" || { log "staging part '$p' FAILED: $(grep -E "^[0-9:]+ $p rc=" $L)"; exit 1; }
done
[ "${WAIT_GO:-0}" = 1 ] && { log "waiting for $W/GO"; until [ -f $W/GO ]; do sleep 2; done
  set -a; . $W/GO; set +a; log "GO: $(tr '\n' ' ' < $W/GO)"; }   # 10/28 dyn round: data / init land after staging; GO = env lines (FTEXTRA=...)
log "pipeline start"
cd $W/train
OUT=$OUT B2=b2://morai2026-alpamayo/${RUNB2:-runs/r2ft} STAGES="ft qad deliver qgate" QAD_FROM=ft R2ROOT=/data/step0_r2 SEXTRA="model.attn_implementation=$ATTN ${FASTX:-} ${SEXTRA:-}" \
  FTEXTRA="$([ "$FAST" = 1 ] && echo callbacks.ft_select.batch=${FT_EVAL_BATCH:-4}) ${FTEXTRA:-}" GATE_SHARDS=$([ "$FAST" = 1 ] && echo $NGPU || echo 1) PUSH_ASYNC=$FAST \
  bash scripts/pipeline.sh > $W/out/pipeline.log 2>&1
log "pipeline exit $?"
