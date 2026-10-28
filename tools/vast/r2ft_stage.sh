#!/usr/bin/env bash
# Stage an 8-GPU box (vast/RunPod/Brev) for the WoWmayo R2 fine-tune (train/README "와우마요 R2 미세조정"), restored from the
# 10/27 Texas box (B2 runs/r2ft/pro6000tx_57966802/stage_r2.sh) with its bugs fixed. Code first (Mac: tar of train/ and
# tools/vast/ into /workspace), then this, then tools/vast/r2ft_chain.sh. In order:
#   tools, ONE idle watchdog (+ metrics sync loop with the ft_metrics hook), then in parallel: B2 step0 (R1, ~190 GB, the
#   long pole) | step0_r2 | s2/final (user 10/28: no T*, so no t2 / rule_ok) | venv tar -> student pkgs -> gpu_check | HF Qwen3-VL 4B + 8B
#   config. Each part logs "<HH:MM:SS> <part> rc=<n>" to stdout (= /workspace/stage.log), which r2ft_chain.sh reads.
# Not downloaded (not read by ft/qad/gate/deliver): coc/ and the Alpamayo-1.5 release (on 10/27 they cost 15+ min).
# Usage on the box: RUN=<gpu>_<id> setsid nohup bash /workspace/tools/vast/r2ft_stage.sh > /workspace/stage.log 2>&1 < /dev/null &
#   VENV_TAR (default by GPU): a1_5_sft_venv.tar on H100/H200 (stage B, flash-attn for sm_90; ran 8x H200 10/27), else
#   a1_5_sft_venv_sm100.tar (B200; RTX PRO 6000/5090 run it with sdpa, r2ft_chain.sh picks that). RUN: name of this box under b2 runs/r2ft/metrics/ (e.g. pro6000_<instance id>).
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
# ssh sessions don't carry the container env (vast env vars, RunPod secrets): take it from PID 1
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(B2_APPLICATION_KEY_ID|B2_APPLICATION_KEY|HF_TOKEN|CONTAINER_API_KEY|CONTAINER_ID|DPH_USD|RUNPOD_API_KEY|RUNPOD_POD_ID)=' | sed -E "s/^([A-Z0-9_]+)=(.*)$/export \1='\2'/")"
[ -f /workspace/b2.env ] && set -a && source /workspace/b2.env && set +a     # Brev etc.: no container env, keys put in b2.env by hand
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)     # 9.0 H100/H200: the sm_90 flash-attn build
VENV_TAR=${VENV_TAR:-$([ "${CC%%.*}" = 9 ] && echo a1_5_sft_venv.tar || echo a1_5_sft_venv_sm100.tar)}; B=b2://morai2026-alpamayo; X='(.*/)?\._.*'; W=/workspace; OUT=$W/out/stageR2
VENV=$W/alpamayo-recipes/recipes/alpamayo1_5_sft/a1_5_sft
# RUNB2: this run's prefix under runs/ (10/28 stop round: runs/r2stop); INIT_B2: the checkpoint staged at $OUT/s2/final (the
# start of ft; stop round: the R2 ft final)
RUNB2=${RUNB2:-runs/r2ft}; INIT_B2=${INIT_B2:-runs/stageC/s2/final}
RUN=${RUN:-box_${CONTAINER_ID:-${RUNPOD_POD_ID:-$(hostname)}}}
mkdir -p $W && cd $W && mkdir -p data out ckpt hf $OUT/s2/final $OUT/logs && ln -sfn $W/data /data && touch keepalive
apt-get update -qq && apt-get install -y -qq --no-install-recommends git python3 python3-venv python3.12-dev build-essential ninja-build tmux rsync > /dev/null
[ -x /opt/uvtool/bin/b2 ] || { python3 -m venv /opt/uvtool && /opt/uvtool/bin/pip install -q uv b2 vastai; }
ln -sf /opt/uvtool/bin/uv /opt/uvtool/bin/b2 /opt/uvtool/bin/vastai /usr/local/bin/
t0=$(date +%s); log() { echo "$(date +%T) $*"; }
log "run $RUN"; echo "$RUN" > $W/RUN
# one watchdog (10/27 Texas ran two, from two paths); budget never stops a box (user 10/27): BUDGET_USD only logs WARN
pkill -f "[i]dle_watchdog.sh"; pkill -f "[m]etrics_sync.sh --loop"
export METRICS_SRC=$OUT METRICS_DST=$B/$RUNB2/metrics/$RUN \
  METRICS_HOOK="cd $W/train && $VENV/bin/python scripts/ft_metrics.py $OUT/ft"
if [ -n "${CONTAINER_API_KEY:-}${RUNPOD_POD_ID:-}" ]; then   # vast / RunPod can stop themselves; elsewhere (Brev) the owner stops the box
  BUDGET_USD=1000000 PRIOR_USD=-1000000 IDLE_MIN=${IDLE_MIN:-20} setsid nohup bash $W/tools/vast/idle_watchdog.sh > /dev/null 2>&1 < /dev/null &
else log "no idle watchdog: no vast/RunPod key in the container env (stop this box by hand)"; fi
setsid nohup bash $W/tools/vast/metrics_sync.sh --loop ${METRICS_EVERY:-600} > /dev/null 2>&1 < /dev/null &
( while [ ! -e $W/STAGE_DONE ]; do touch $W/keepalive; sleep 120; done ) & KA=$!     # staging is network, the GPUs idle
if [ -n "${R1_EPISODES:-}" ]; then                    # probe box (r2ft_probe.sh): index, start relabel and the listed R1 episodes only
  ( r=0; mkdir -p /data/step0/_ego
    for d in _index _start_relabel; do b2 sync --threads 16 --no-progress $B/step0/$d/ /data/step0/$d/ > /dev/null 2>&1 || r=1; done
    export B; one() { b2 sync --threads 4 --no-progress --exclude-regex '.*/lidar\.npz$' $B/step0/$1/ /data/step0/$1/ && b2 file download --no-progress $B/step0/_ego/$1.npz /data/step0/_ego/$1.npz; }
    export -f one   # 10/28: 16 episodes at a time (one by one, 107 episodes took 30 min on the 5090 probe box)
    xargs -P 16 -I{} bash -c 'one {} > /dev/null 2>&1 || echo {}' < $R1_EPISODES > $W/step0_failed.txt; [ -s $W/step0_failed.txt ] && r=1
    log "step0 rc=$r ($(wc -l < $R1_EPISODES) episodes, failed $(wc -l < $W/step0_failed.txt))" ) &
else
  ( b2 sync --threads 64 --no-progress --exclude-regex "$X" --exclude-regex '.*/lidar\.npz$' --exclude-dir-regex '_coc' $B/step0/ /data/step0/ > $W/sync_step0.log 2>&1; log "step0 rc=$?" ) &
fi
( b2 sync --threads 32 --no-progress --exclude-regex "$X" $B/step0_r2/ /data/step0_r2/ > $W/sync_step0_r2.log 2>&1; log "step0_r2 rc=$?" ) &
( r=0; b2 sync --threads 16 --no-progress $B/$INIT_B2/ $OUT/s2/final/ > $W/sync_s2.log 2>&1 || r=1
  [ -f $OUT/s2/final/config.json ] && ls $OUT/s2/final/*.safetensors > /dev/null 2>&1 || r=1; log "ckpt rc=$r" ) &
( [ -d alpamayo-recipes ] || git clone -q https://github.com/NVlabs/alpamayo-recipes.git; cd alpamayo-recipes && git checkout -q 2e907da
  b2 file download --no-progress --threads 16 $B/env/$VENV_TAR $W/$VENV_TAR > /dev/null && tar xf $W/$VENV_TAR -C recipes/alpamayo1_5_sft && rm -f $W/$VENV_TAR
  source $VENV/bin/activate && python -c "import torch, flash_attn; print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(), torch.cuda.get_arch_list(), flash_attn.__version__)"; log "venv rc=$?"
  # student QAD packages (not in the 10/24 venv tar) + matplotlib for scripts/ft_metrics.py figures
  uv pip install -q --python $(which python) --no-deps nvidia-modelopt==0.47.0 "pulp<4" nvidia-ml-py && uv pip install -q --python $(which python) matplotlib \
    && python -c "import modelopt, pulp, scipy, pandas, matplotlib; print('modelopt', modelopt.__version__)"; log "student pkgs rc=$?"
  python $W/tools/vast/gpu_check.py ${GPU_CHECK_S:-300} $W/out/gpu_check > $W/gpu_check.log 2>&1 && r=PASS || r=FAIL; touch $W/GPU_CHECK_$r; log "gpu check $r"
  b2 sync --no-progress $W/out/gpu_check/ $B/$RUNB2/gpu_check/$RUN/ > /dev/null 2>&1; log "gpu check push rc=$?" ) &
( SKIP_RELEASE=1 bash $W/tools/vast/get_ckpt.sh > $W/get_ckpt.log 2>&1; log "hf ckpt rc=$?" ) &
wait $(jobs -p | grep -vx $KA)
log "staging done in $(( $(date +%s) - t0 )) s"; du -sh /data/step0 /data/step0_r2 $OUT/*/final $W/hf 2>/dev/null
touch $W/STAGE_DONE
