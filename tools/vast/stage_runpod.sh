#!/usr/bin/env bash
# Stage the 8-GPU RunPod pod for the whole remaining pipeline (RUNPOD_SETUP.md): tools + watchdog first, then in parallel
#   B2 -> /workspace/data/step0 (+ coc_v1_1), the venv tar for this GPU arch, an input checkpoint (CKPT), HF checkpoints;
#   once the venv is in, the GPU burn-in (gpu_check.py) -> /workspace/GPU_CHECK_PASS or _FAIL, results also on B2.
# Paths match the runbooks via /data -> /workspace/data. Needs B2_APPLICATION_KEY_ID/B2_APPLICATION_KEY (RunPod secrets) and DPH_USD.
# Usage on the pod: VENV_TAR=a1_5_sft_venv_sm100.tar nohup bash /workspace/stage_runpod.sh > /workspace/stage.log 2>&1 &
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
# ssh sessions don't carry the pod env (RunPod secrets, DPH_USD, pod-scoped key): take it from PID 1
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(B2_APPLICATION_KEY_ID|B2_APPLICATION_KEY|DPH_USD|RUNPOD_API_KEY|RUNPOD_POD_ID|CONTAINER_API_KEY|CONTAINER_ID)=' | sed -E "s/^([A-Z0-9_]+)=(.*)$/export \1='\2'/")"
VENV_TAR=${VENV_TAR:-a1_5_sft_venv_sm100.tar}; B=b2://morai2026-alpamayo; X='(.*/)?\._.*'
mkdir -p /workspace && cd /workspace && mkdir -p data/coc out ckpt hf && ln -sfn /workspace/data /data && touch keepalive
apt-get update -qq && apt-get install -y -qq --no-install-recommends git python3 python3-venv python3.12-dev build-essential ninja-build tmux rsync > /dev/null
[ -x /opt/uvtool/bin/b2 ] || { python3 -m venv /opt/uvtool && /opt/uvtool/bin/pip install -q uv b2 vastai; }   # vastai: the watchdog's stop on vast
ln -sf /opt/uvtool/bin/uv /opt/uvtool/bin/b2 /opt/uvtool/bin/vastai /usr/local/bin/
# on vast pass PRIOR_USD=-1000000: support's last word (10/24) is an hourly cap, so the watchdog's cumulative cap must not fire
pgrep -f idle_watchdog.sh > /dev/null || { BUDGET_USD=${BUDGET_USD:-4000} IDLE_MIN=${IDLE_MIN:-20} nohup bash /workspace/idle_watchdog.sh > /dev/null 2>&1 & }
( while [ ! -e /workspace/STAGE_DONE ]; do touch /workspace/keepalive; sleep 120; done ) & KA=$!   # staging is mostly network, GPU idle
t0=$(date +%s); log() { echo "$(date +%T) $*"; }
( b2 sync --threads 64 --no-progress --exclude-regex "$X" $B/step0/ /workspace/data/step0/ > /workspace/sync_step0.log 2>&1; log "step0 rc=$?" ) &
( b2 sync --threads 16 --no-progress --exclude-regex "$X" $B/coc/coc_v1_1/ /workspace/data/coc/coc_v1_1/ > /workspace/sync_coc.log 2>&1; log "coc rc=$?" ) &
# input checkpoint "B2 path:local dir"; t2 onwards needs t1c/final where pipeline.sh's need() looks (it pulls it itself if missing)
CKPT=${CKPT:-runs/stageC/t1c/final:/workspace/out/stageC/t1c/final}
( mkdir -p ${CKPT#*:}; b2 sync --threads 16 --no-progress $B/${CKPT%%:*}/ ${CKPT#*:}/ > /workspace/sync_ckpt.log 2>&1; log "ckpt ${CKPT%%:*} rc=$?" ) &
( [ -d alpamayo-recipes ] || git clone -q https://github.com/NVlabs/alpamayo-recipes.git; cd alpamayo-recipes && git checkout -q 2e907da
  b2 file download --threads 16 $B/env/$VENV_TAR /workspace/$VENV_TAR > /dev/null && tar xf /workspace/$VENV_TAR -C recipes/alpamayo1_5_sft && rm -f /workspace/$VENV_TAR
  source recipes/alpamayo1_5_sft/a1_5_sft/bin/activate && python -c "import torch, flash_attn; print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(), torch.cuda.get_arch_list(), flash_attn.__version__)"; log "venv rc=$?"
  # student QAD packages, not in the 10/24 venv tar (PyPI; same pins as the 10/25 pod)
  uv pip install -q --python $(which python) --no-deps nvidia-modelopt==0.47.0 "pulp<4" nvidia-ml-py && python -c "import modelopt, pulp, scipy, pandas; print('modelopt', modelopt.__version__)"; log "student pkgs rc=$?"
  python /workspace/gpu_check.py 300 /workspace/out/gpu_check > /workspace/gpu_check.log 2>&1 && r=PASS || r=FAIL; touch /workspace/GPU_CHECK_$r; log "gpu check $r"
  b2 sync --no-progress /workspace/out/gpu_check/ $B/runs/stageC/gpu_check/${RUNPOD_POD_ID:-pod}/ > /dev/null 2>&1; log "gpu check push rc=$?" ) &
( bash /workspace/get_ckpt.sh > /workspace/get_ckpt.log 2>&1; log "hf ckpt rc=$?" ) &
wait $(jobs -p | grep -vx $KA)   # not the keepalive loop: it waits for STAGE_DONE, written below
log "staging done in $(( $(date +%s) - t0 )) s"; du -sh /workspace/data/step0 /workspace/ckpt/* 2>/dev/null
find /workspace/data/step0 -type f | wc -l
touch /workspace/STAGE_DONE
