#!/usr/bin/env bash
# Warm-up of the DAgger learner box (Brev hyperstack_H100x8_NVLINK; user 10/28 "Brev H100"), as root, after
# dagger/brev_up.sh copied the code (/tmp/code.tgz) and the b2 account file (/root/.config/b2/account_info: no key on any
# command line or log). Modelled on tools/vast/r2ft_stage.sh; every part logs "<HH:MM:SS> <part> rc=<n>" to stdout
# (= $W/stage.log). In order:
#   /workspace on the largest disk, tools (uv, b2), b2 auth, acceptance (paid-box rule 2: CPU single-thread python loop
#   + MHz, B2 MB/s on the venv tar, gpu_check 8 GPUs) -> in parallel: venv tar -> student pkgs -> gpu_check | step0_r2 |
#   deliveries (R2 = runs/r2ft/deliver/final, + runs/r2stop/deliver/final when it exists) | HF Qwen3-VL-4B | R1 step0
#   (~190 GB, the long pole) -> as soon as venv, pkgs, step0_r2, the R2 delivery and HF are in: dagger/dryrun.sh (8 GPUs,
#   expert LoRA, R2 replay) while R1 still downloads. Metrics (stage log, acceptance, dry run) -> B2 runs/dagger/learner/$RUN.
# No training starts here; the box then waits (hyperstack is not stoppable on Brev).
# Usage (on the box): sudo RUN=<name> setsid nohup bash /tmp/brev_stage.sh > /tmp/stage_boot.log 2>&1 < /dev/null &
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
RUN=${RUN:-brev_$(hostname)}; B=b2://morai2026-alpamayo
big=$(df -B1 --output=avail,target -x tmpfs -x devtmpfs -x overlay 2>/dev/null | tail -n +2 | sort -n | tail -1 | awk '{print $2}')
mkdir -p "$big/workspace" && { [ -e /workspace ] || ln -sfn "$big/workspace" /workspace; }
W=/workspace; VENV=$W/alpamayo-recipes/recipes/alpamayo1_5_sft/a1_5_sft; OUT=$W/out; META=$OUT/learner_meta
mkdir -p $W/data $OUT $META $W/ckpt $W/hf && ln -sfn $W/data /data && touch $W/keepalive
exec >> $W/stage.log 2>&1                                                    # every part's line in one log
t0=$(date +%s); log() { echo "$(date +%T) $*"; }
log "run $RUN, /workspace -> $big ($(df -h $big | tail -1 | awk '{print $4}') free)"; echo "$RUN" > $W/RUN
tar xzf /tmp/code.tgz -C $W && log "code rc=$?"                               # train/, tools/, e2e/
apt-get update -qq && apt-get install -y -qq --no-install-recommends git python3 python3-venv build-essential tmux rsync > /dev/null
[ -x /opt/uvtool/bin/b2 ] || { python3 -m venv /opt/uvtool && /opt/uvtool/bin/pip install -q uv b2; }
ln -sf /opt/uvtool/bin/uv /opt/uvtool/bin/b2 /usr/local/bin/; log "tools rc=$?"
# the venv tar points at /usr/bin/python3.12 (vast images); a 22.04 VM has none: apt, else uv's build linked there
[ -x /usr/bin/python3.12 ] || apt-get install -y -qq python3.12 python3.12-dev > /dev/null 2>&1 \
  || { uv python install 3.12 > /dev/null && ln -sf "$(uv python find 3.12)" /usr/bin/python3.12; }
/usr/bin/python3.12 -V; log "python3.12 rc=$?"
b2 account get > /dev/null 2>&1; log "b2 auth rc=$?"

# acceptance 1: CPU (rule 2: a slow CPU wrecks the run, 10/27)
python3 - $META/cpu.json <<'PY'
import json, os, subprocess, sys, time
t = time.time(); x = 0
for i in range(10_000_000): x += i
loop = time.time() - t
mhz = [float(l.split(":")[1]) for l in open("/proc/cpuinfo") if l.startswith("cpu MHz")]
model = next((l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
mmax = next((l.split(":")[1].strip() for l in lscpu.splitlines() if l.startswith("CPU max MHz")), None)
json.dump({"python_1e7_loop_s": round(loop, 3), "cpu_model": model, "nproc": os.cpu_count(), "mhz_now_max": max(mhz) if mhz else None,
           "mhz_now_mean": round(sum(mhz) / len(mhz)) if mhz else None, "lscpu_max_mhz": mmax}, open(sys.argv[1], "w"), indent=1)
print(json.dumps(json.load(open(sys.argv[1]))))
PY
log "cpu check rc=$?"

# metrics up from the start (never lose them); the owner (Mac session) watches over ssh: Brev cannot stop this box itself
( while :; do cp $W/stage.log $META/ 2>/dev/null; METRICS_SRC=$OUT METRICS_DST=$B/runs/dagger/learner/$RUN METRICS_LOG=$W/metrics_sync.log bash $W/tools/vast/metrics_sync.sh
    sleep ${METRICS_EVERY:-120}; done ) > /dev/null 2>&1 &
( while :; do echo "$(date +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ')" >> $META/gpu_util.log; sleep 60; done ) &

( cd $W && { [ -d alpamayo-recipes ] || git clone -q https://github.com/NVlabs/alpamayo-recipes.git; } && cd alpamayo-recipes && git checkout -q 2e907da
  s=$(date +%s.%N); b2 file download --no-progress --threads 16 $B/env/a1_5_sft_venv.tar $W/venv.tar > /dev/null; r=$?
  python3 -c "import json, os, sys; s = float(sys.argv[1]); n = os.path.getsize(sys.argv[2]); import time; d = time.time() - s
json.dump({'file': 'env/a1_5_sft_venv.tar', 'bytes': n, 'seconds': round(d, 1), 'MB_per_s': round(n / d / 1e6, 1)}, open(sys.argv[3], 'w'))" $s $W/venv.tar $META/b2_speed.json
  log "b2 speed $(cat $META/b2_speed.json)"
  [ $r = 0 ] && tar xf $W/venv.tar -C recipes/alpamayo1_5_sft && rm -f $W/venv.tar
  source $VENV/bin/activate && python -c "import torch, flash_attn; print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name())"; log "venv rc=$?"
  uv pip install -q --python $(which python) --no-deps nvidia-modelopt==0.47.0 "pulp<4" nvidia-ml-py \
    && python -c "import modelopt, pulp; print('modelopt', modelopt.__version__)"; log "student pkgs rc=$?"
  python $W/tools/vast/gpu_check.py ${GPU_CHECK_S:-300} $OUT/learner_meta/gpu_check > $W/gpu_check.log 2>&1 && r=PASS || r=FAIL; log "gpu check $r" ) &
( b2 sync --threads 32 --no-progress --exclude-regex '(.*/)?\._.*' $B/step0_r2/ /data/step0_r2/ > $W/sync_step0_r2.log 2>&1; log "step0_r2 rc=$?" ) &
( r=0; b2 sync --threads 16 --no-progress $B/runs/r2ft/deliver/final/ $W/ckpt/deliver_r2/final/ > $W/sync_deliver.log 2>&1 || r=1
  [ -f $W/ckpt/deliver_r2/final/modelopt_state.pth ] || r=1; log "deliver_r2 rc=$r"
  if b2 ls $B/runs/r2stop/deliver/final/ 2>/dev/null | grep -q safetensors; then
    b2 sync --threads 16 --no-progress $B/runs/r2stop/deliver/final/ $W/ckpt/deliver_r2stop/final/ >> $W/sync_deliver.log 2>&1; log "deliver_r2stop rc=$?"
  else log "deliver_r2stop absent (runs/r2stop/deliver/final)"; fi ) &
( SKIP_RELEASE=1 bash $W/tools/vast/get_ckpt.sh > $W/get_ckpt.log 2>&1; log "hf ckpt rc=$?" ) &
( b2 sync --threads 64 --no-progress --exclude-regex '(.*/)?\._.*' --exclude-regex '.*/lidar\.npz$' --exclude-dir-regex '_coc' $B/step0/ /data/step0/ > $W/sync_step0.log 2>&1
  log "step0 rc=$?" ) &

# dry run as soon as its parts are in (health check during the download, user 10/27)
L=$W/stage.log
ok() { grep -qE "^[0-9:]+ $1 rc=0" $L; }
NEED="venv|student pkgs|step0_r2|deliver_r2|hf ckpt"
until { ok venv && ok "student pkgs" && ok step0_r2 && ok deliver_r2 && ok "hf ckpt" && grep -qE "^[0-9:]+ gpu check" $L; } \
      || grep -qE "^[0-9:]+ ($NEED) rc=[1-9]" $L; do sleep 5; done                 # gpu_check first: no overlap of measurements
if grep -qE "^[0-9:]+ ($NEED) rc=[1-9]" $L; then log "dry run not started: a part failed"; else
  cd $W/train && log "dry run start (8 GPUs, expert)"
  BASE=$W/ckpt/deliver_r2/final W=$W bash dagger/dryrun.sh expert > $W/dryrun.log 2>&1; log "dry run rc=$?"
  cp $W/dryrun.log $META/ 2>/dev/null
fi
wait
log "staging done in $(( $(date +%s) - t0 )) s"; du -sh /data/step0 /data/step0_r2 $W/ckpt/* 2>/dev/null
cp $L $META/; METRICS_SRC=$OUT METRICS_DST=$B/runs/dagger/learner/$RUN METRICS_LOG=$W/metrics_sync.log bash $W/tools/vast/metrics_sync.sh
touch $W/STAGE_DONE
