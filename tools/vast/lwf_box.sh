#!/usr/bin/env bash
# LwF dynamic-stop box steps (reports/lwf_plan_20261029.md 0.6, main 10/29), next to r2ft_stage.sh (R1, R2, R2 ft final,
# venv, HF; unchanged) and r2ft_chain.sh WAIT_GO=1 (FT_CFG=student_lwf QAD_CFG=student_qad_lwf). In order:
#   1) pop-out small files first (_ego, _popout/*/objects.json.gz, _popout_index, _quarantine.json), then its sensors
#   2) pre-processing: B2 _dyn_relabel_v2/DONE -> download it; else (user "전처리는 인스턴스 다운로드 동안 같이 해라") run
#      train/scripts/popout_rebuild.sh <root> <out> -j N here (once the venv is in; scp it in if the code tar predates it) and
#      push the result to _dyn_relabel_v2_box/<RUN>/ (record); neither within 40 min (2-min polls) -> stop and report
#   3) once v2 and step0_r2 are in: lwf_mix (rewrites configs/lwf_dynstop.yaml in this box's code copy) -> staging of the
#      virtual episodes -> dyn_load_check (R2, pop-out v2)
#   4) GPUs 1-7 after the chain's own speed test: 50 LwF steps with the teacher on the R2 + pop-out sources (s/step, peak
#      memory, step 0 vkd ~ 0); GPU 0: the dynbal nav_text probe (best effort; INCOMPLETE when R1 comes first)
#   5) R1 in (stage.log "step0 rc=0"): lwf_load_check on the whole config -> rc=0 -> touch /workspace/GO (the chain starts ft)
# Logs "<HH:MM:SS> lwf <step> rc=<n>" to stdout (/workspace/lwf_box.log); a failed step stops before GO (the watcher alerts).
# Usage on the box, after r2ft_stage.sh and r2ft_chain.sh (WAIT_GO=1) started:
#   RUN=<gpu>_<id> setsid nohup bash /workspace/tools/vast/lwf_box.sh > /workspace/lwf_box.log 2>&1 < /dev/null &
set -uo pipefail
# ssh sessions don't carry the container env (B2 keys): take it from PID 1, as r2ft_stage.sh does
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(B2_APPLICATION_KEY_ID|B2_APPLICATION_KEY|HF_TOKEN|CONTAINER_API_KEY|CONTAINER_ID)=' | sed -E "s/^([A-Z0-9_]+)=(.*)$/export \1='\2'/")"
[ -f /workspace/b2.env ] && set -a && source /workspace/b2.env && set +a
W=/workspace; L=$W/stage.log; OUT=$W/out/stageR2; B=b2://morai2026-alpamayo; RUN=${RUN:-$(cat $W/RUN 2>/dev/null)}
P=/data/step0_popout; R2=/data/step0_r2; V2=${POPOUT_DYN:-_dyn_relabel_v2};   # = student_lwf paths.popout_dyn
 T=$W/train; VENV=$W/alpamayo-recipes/recipes/alpamayo1_5_sft/a1_5_sft
R1_SOURCES=(replay lane_old_recovery static_general_r1_03_06 static_general_r1_10p static_tunnel_r1_03_06 static_tunnel_r1_06_10
            static_tunnel_r1_10p red_approach_7294 red_approach_6254 red_approach_4188 red_approach_4356 red_wait_7294 red_wait_6254
            red_wait_4188 red_wait_4356)                                       # read data.root (R1): left out of the probe
log() { echo "$(date +%T) lwf $*"; }
ok() { grep -qE "^[0-9:]+ $1 rc=0" $L; }
wait_for() { until grep -qE "^[0-9:]+ $1 rc=" $L; do sleep 5; done; ok "$1"; }
fail() { log "$1 FAILED (no GO)"; exit 1; }
export PATH=/opt/uvtool/bin:$PATH
until command -v b2 > /dev/null; do sleep 5; done                                  # r2ft_stage.sh installs b2 first

# 1) pop-out: small files, then the sensors (no lidar.npz: the loader reads lidar.bin)
mkdir -p $P
b2 sync --threads 32 --no-progress --exclude-regex '.*' --include-regex '^(_ego/.*|_popout/.*/(episode\.json|labels\.parquet|objects\.json\.gz)|_popout_index/.*|_quarantine\.json|[^_][^/]*/(cams_index|lidar_index)\.npz)$' \
  $B/step0_popout/ $P/ > $W/sync_popout_small.log 2>&1; r=$?; log "popout small rc=$r"; [ $r = 0 ] || fail "popout small"
( b2 sync --threads 32 --no-progress --exclude-regex '.*/lidar\.npz$' --exclude-dir-regex '^_.*' $B/step0_popout/ $P/ > $W/sync_popout.log 2>&1
  echo $? > $W/popout_sensors.rc ) & SENS=$!

# 2) pre-processing: v2 from B2 (DONE = summary.json md5), else here with popout_rebuild.sh; neither within 40 min -> stop
for k in $(seq 20); do
  b2 ls $B/step0_popout/$V2/ 2>/dev/null | grep -q "/DONE$" && { SRC=b2; break; }
  [ -f $T/scripts/popout_rebuild.sh ] && { SRC=box; break; }
  log "waiting for $V2/DONE or scripts/popout_rebuild.sh ($k/20)"; sleep 120
done
case ${SRC:-} in
  b2) b2 sync --threads 32 --no-progress $B/step0_popout/$V2/ $P/$V2/ > $W/sync_v2.log 2>&1; r=$?; log "v2 download rc=$r" ;;
  box) wait_for venv || fail "venv"; log "v2 not on B2: popout_rebuild.sh here"   # inputs: the small files + <ep>/{cams,lidar}_index.npz
       ( source $VENV/bin/activate && cd $T && PY=$VENV/bin/python bash scripts/popout_rebuild.sh $P $P/$V2 -j 96 ) > $W/out/popout_rebuild.log 2>&1; r=$?
       n=$(grep -oE "popout_rebuild: [0-9]+ episodes" $W/out/popout_rebuild.log | grep -oE "[0-9]+"); log "v2 rebuild over ${n:-?} episodes"
       [ "${n:-0}" -ge 700 ] || r=9                                           # main 10/29: 732 recorded; fewer = a partial v2
       log "v2 rebuild rc=$r"; [ $r = 0 ] && ( b2 sync --no-progress $P/$V2/ $B/step0_popout/${V2}_box/$RUN/ > /dev/null 2>&1 & ) ;;
  *) fail "v2 (no $V2/DONE on B2 and no scripts/popout_rebuild.sh after 40 min)" ;;
esac
[ $r = 0 ] && [ -f $P/$V2/dyn_index.parquet ] || fail "v2"
wait $SENS; r=$(cat $W/popout_sensors.rc); log "popout sensors rc=$r"; [ "$r" = 0 ] || fail "popout sensors"

# 3) mixture lists + staging + virtual-episode checks (CPU)
wait_for venv || fail "venv"; wait_for step0_r2 || fail "step0_r2"
source $VENV/bin/activate; cd $T; export PYTHONPATH=$PWD TOKENIZERS_PARALLELISM=false
python scripts/lwf_mix.py --popout $P/$V2 --popout-ego $P/_ego --popout-dir-rel $V2 --r2 $R2/_dyn_relabel --r2-mix $R2/_dyn_relabel/mix_ring \
  --r2-ego $R2/_ego --out-popout $P/$V2/mix_lwf --out-r2 $R2/_dyn_relabel/mix_lwf --drop-rh --near-passes 8 \
  --write-config configs/lwf_dynstop.yaml > $W/out/lwf_mix.log 2>&1; r=$?; log "lwf_mix rc=$r"; [ $r = 0 ] || fail "lwf_mix"
python scripts/lwf_mix.py --popout $P/$V2 --popout-ego $P/_ego --popout-dir-rel $V2 --r2 $R2/_dyn_relabel --r2-mix $R2/_dyn_relabel/mix_ring \
  --r2-ego $R2/_ego --out-popout /tmp/lwf_boost/p --out-r2 /tmp/lwf_boost/r --drop-rh --near-passes 8 --popout-share 0.35 --weight 8 \
  --write-config configs/lwf_dynstop_boost.yaml > $W/out/lwf_mix_boost.log 2>&1; log "lwf_mix boost rc=$?"   # student_lwf_boost (emergency variant)
( b2 sync --no-progress $P/$V2/mix_lwf/ $B/runs/lwf/$RUN/mix_lwf_popout/ > /dev/null 2>&1; b2 file upload --no-progress morai2026-alpamayo \
    configs/lwf_dynstop.yaml runs/lwf/$RUN/lwf_dynstop.yaml > /dev/null 2>&1
    b2 file upload --no-progress morai2026-alpamayo configs/lwf_dynstop_boost.yaml runs/lwf/$RUN/lwf_dynstop_boost.yaml > /dev/null 2>&1 ) &
{ python scripts/dyn_stage.py $R2 && python scripts/dyn_stage.py $P --relabel-dir $V2; } > $W/out/lwf_stage.log 2>&1   # v2 carries its ego/ tables
r=$?; log "stage rc=$r"; [ $r = 0 ] || fail "stage"
{ python scripts/dyn_load_check.py $R2 40 && python scripts/dyn_load_check.py $P 40 --relabel-dir $V2; } > $W/out/lwf_dyncheck.log 2>&1
r=$?; log "dyn_load_check rc=$r"; [ $r = 0 ] || fail "dyn_load_check"

# 4) GPU work while R1 downloads; never delays GO (stopped once R1 and the loader check are done)
( until grep -q "speed test end" $W/chain.log 2>/dev/null || ok step0; do sleep 5; done
  if ! ok step0; then
    CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 timeout 1200 torchrun --nproc_per_node 7 --master_port 29621 -m morai_vla.train --config-name student_lwf \
      data.root=$R2 paths.r2_root=$R2 paths.popout_root=$P paths.s2=$OUT/s2/final paths.output_dir=$W/out/lwf_probe \
      model.attn_implementation=flash_attention_2 callbacks.ft_select=null $(printf "~data.train_dataset.sources.%s " "${R1_SOURCES[@]}") \
      trainer.max_steps=50 trainer.logging_steps=5 trainer.eval_strategy=no trainer.save_strategy=no save_final=false > $W/out/lwf_probe.log 2>&1
    log "lwf probe rc=$? ($(grep -oE '"peak_mem_gb": [0-9.]+' $W/out/lwf_probe.log | tail -1))"
  fi ) &
( mkdir -p $W/ck/dynbal $W/out/navprobe && b2 sync --threads 16 --no-progress $B/runs/dynbal/ft/final/ $W/ck/dynbal/ > /dev/null 2>&1 \
    && CUDA_VISIBLE_DEVICES=0 python scripts/navprobe.py --ckpt $W/ck/dynbal --root $P --index $V2/dyn_index.parquet \
       --keep $P/$V2/mix_lwf/popout_val_onset.parquet --device cuda --out $W/out/navprobe > $W/out/navprobe.log 2>&1
  r=$?; [ -f $W/out/navprobe/summary.json ] || echo "INCOMPLETE rc=$r" > $W/out/navprobe/RESULT
  b2 sync --no-progress $W/out/navprobe/ $B/runs/lwf/$RUN/navprobe/ > /dev/null 2>&1; log "navprobe rc=$r" ) &

# 5) R1 in: the whole config through the loader, then GO
wait_for step0 || fail "R1 step0"
python scripts/lwf_load_check.py --config-name student_lwf data.root=/data/step0 paths.r2_root=$R2 paths.popout_root=$P \
  paths.s2=$OUT/s2/final paths.output_dir=/tmp/lwf_x > $W/out/lwf_load_check.log 2>&1; r=$?; log "lwf_load_check rc=$r"
[ $r = 0 ] && grep -q "rc=0" $W/out/lwf_load_check.log || fail "lwf_load_check"
for pat in "out/[l]wf_probe" "scripts/[n]avprobe.py"; do                       # PIDs first (no pkill -f on our own command line)
  for p in $(pgrep -f "$pat"); do kill $p 2>/dev/null; done
done
[ -f $W/out/navprobe/summary.json ] || echo "INCOMPLETE: R1 and the loader check came first" > $W/out/navprobe/RESULT
( b2 sync --no-progress $W/out/navprobe/ $B/runs/lwf/$RUN/navprobe/ > /dev/null 2>&1 & )
while pgrep -f "out/[l]wf_probe|scripts/[n]avprobe.py" > /dev/null; do sleep 1; done
touch $W/GO; log "GO"
