#!/usr/bin/env bash
# alpha pick for the WoWmayo weight interpolation theta = (1 - alpha) A + alpha B (reports/interp_plan_20261029.md;
# A = R2 ft final, B = dynbal ft final) on a rented 5090 box, 1 GPU or more. Rent only with the user's OK (CLAUDE.md gate).
# Mac first: the code tar (train/ + tools/vast/, as for r2ft_probe.sh) into /workspace, the 10/29 history-probe inputs
# (keep.parquet, synth.parquet, episodes.txt; rows.csv + hdep.py for its table) into /workspace/probe/. Then on the box:
#   RUN=5090_<id> setsid nohup bash /workspace/tools/vast/interp_eval.sh > /workspace/interp.log 2>&1 < /dev/null &
#  1. staging: r2ft_stage.sh in R1_EPISODES mode (only the R1 episodes the val sets and the probe rows read; step0_r2
#     whole; s2 final at $OUT/s2/final; venv, HF), beside it A, B and the pop-out val episodes; dyn_stage.py + loader checks
#  2. scripts/weight_interp.py for every alpha strictly inside (0, 1); alpha 0 / 1 are A / B as downloaded (lerp gives
#     them bit for bit, tests/test_weight_interp.py)
#  3. per alpha, scripts/gate_student.py records = FtSelect's measurement (early_stop.expert_ade: 6.4 s, K 6, LiDAR on,
#     seed 0 per row, one row per prefill) on student_ft's sets (general / curve as the R2 run's exact rows, assets/stop),
#     + start_relabel, + student_dyn's red_stop and dynamic_stop, + the pop-out val rows when B2 has them. R2-only sets
#     first (they run while R1 still downloads), then the R1 sets
#  4. scripts/ft_probe.py: s2 + every alpha on the 156 probe rows (real / zero / lane-shaped history, PROBE_K seeds)
#  5. summary.json / summary.md (+ hdep.md), push to b2 runs/interp/$RUN/ (file count checked); /workspace/INTERP_DONE or
#     INTERP_FAILED. The idle watchdog (r2ft_stage.sh) stops the box 20 min after this ends; destroy with safe_destroy.sh
#     after the user's OK.
# GPUs: the jobs (R2 records per alpha, the probe, R1 records per alpha) go round-robin over the GPUs, one at a time each.
# Env: A_B2 (runs/r2ft/ft/final), B_B2 (runs/dynbal/ft/final), ALPHAS ("0 0.3 0.5 0.7 1"; the ends written 0 and 1),
#      R2_SETS ("dynamic dynamic_stop popout"), R1_SETS ("general static curve start_relabel red_stop"), PROBE_K (2),
#      POPOUT_B2 (step0_popout), MIN_DISK_GB (220)
set -uo pipefail
W=/workspace; B=b2://morai2026-alpamayo; OUT=$W/out/stageR2; R=$OUT/interp; L=$W/stage.log; CK=$W/ck; PO=/data/step0_popout
A_B2=${A_B2:-runs/r2ft/ft/final}; B_B2=${B_B2:-runs/dynbal/ft/final}; ALPHAS=${ALPHAS:-0 0.3 0.5 0.7 1}; PROBE_K=${PROBE_K:-2}
R2_SETS=${R2_SETS:-dynamic dynamic_stop popout}; R1_SETS=${R1_SETS:-general static curve start_relabel red_stop}; POPOUT_B2=${POPOUT_B2:-step0_popout}
RUN=${RUN:-interp}; export RUN RUNB2=runs/interp                       # r2ft_stage.sh: metrics -> runs/interp/metrics/$RUN
log() { echo "$(date '+%F %T') $*"; }
save() { mkdir -p $R/inputs; cp $W/interp.log $L $W/r1_episodes.txt $R/ 2>/dev/null; cp $W/probe/* $R/inputs/ 2>/dev/null
         b2 sync --no-progress $R/ $B/runs/interp/$RUN/ > /dev/null 2>&1; }
fail() { log "FAILED: $*"; touch $W/INTERP_FAILED; save; exit 1; }
part() { until grep -qE "^[0-9:]+ $1 rc=" $L; do sleep 5; done; grep -qE "^[0-9:]+ $1 rc=0" $L; }   # a staging part of r2ft_stage.sh
ck_of() { case $1 in 0) echo $CK/A;; 1) echo $CK/B;; *) echo $CK/a$1;; esac; }
ck_ok() { [ -f $1/config.json ] && [ $(ls $1/*.safetensors 2>/dev/null | wc -l) -eq $(grep -o 'model-[0-9]*-of-[0-9]*\.safetensors' $1/model.safetensors.index.json | sort -u | wc -l) ]; }

eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^B2_APPLICATION_KEY(_ID)?=' | sed -E "s/^([A-Z0-9_]+)=(.*)$/export \1='\2'/")"
[ -f $W/b2.env ] && set -a && source $W/b2.env && set +a
[ -x /opt/uvtool/bin/b2 ] || { python3 -m venv /opt/uvtool && /opt/uvtool/bin/pip install -q uv b2 vastai; }
ln -sf /opt/uvtool/bin/uv /opt/uvtool/bin/b2 /opt/uvtool/bin/vastai /usr/local/bin/
mkdir -p $W/data/step0/_index $CK/A $CK/B $R/logs && ln -sfn $W/data /data
( while kill -0 $$ 2> /dev/null; do touch $W/keepalive; sleep 120; done ) &   # CPU / network phases count as busy
for f in keep.parquet synth.parquet episodes.txt; do [ -s $W/probe/$f ] || fail "probe input $W/probe/$f missing"; done
[ $(df -BG --output=avail $W | tail -1 | tr -dc 0-9) -ge ${MIN_DISK_GB:-220} ] || fail "disk: $(df -h $W | tail -1)"
for p in $A_B2 $B_B2; do [ $(b2 ls $B/$p/ | grep -c '\.safetensors$') -ge 1 ] || fail "no checkpoint on B2 at $p"; done

# ------------------------------------------------------------------ staging
b2 sync --no-progress $B/step0/_index/ /data/step0/_index/ > /dev/null && b2 sync --no-progress $B/step0/_start_relabel/ /data/step0/_start_relabel/ > /dev/null \
  || fail "R1 index download"
cd $W/train && uv run -q --no-project --with pandas,pyarrow python - > $W/r1_episodes.txt <<'PY' || fail "R1 episode pick"
import pandas as pd                                    # every R1 episode an R1 val set or the probe can open (FtSelect's rows)
val = pd.read_parquet("/data/step0/_index/index.parquet", columns=["episode", "i", "split"]).astype({"episode": str})
val = val[val.split.isin(["val", "val_asset"])][["episode", "i"]]
eps = set(open("/workspace/probe/episodes.txt").read().split())
for f in ("assets/stop/general_val.parquet", "assets/stop/curve_val.parquet", "assets/stop/red_val.parquet", "assets/ft/old_static.parquet",
          "/data/step0/_start_relabel/start_relabel_index.parquet"):
    k = pd.read_parquet(f, columns=["episode", "i"]).astype({"episode": str, "i": int})
    eps |= set(k.merge(val.astype({"i": int}), on=["episode", "i"]).episode)
print("\n".join(sorted(eps)))
PY
log "R1 episodes: $(wc -l < $W/r1_episodes.txt)"
R1_EPISODES=$W/r1_episodes.txt bash $W/tools/vast/r2ft_stage.sh > $L 2>&1 & STAGE=$!          # s2 final (INIT_B2 default) for the probe
( b2 sync --threads 16 --no-progress $B/$A_B2/ $CK/A/ > /dev/null 2>&1 & b2 sync --threads 16 --no-progress $B/$B_B2/ $CK/B/ > /dev/null 2>&1; wait
  ck_ok $CK/A && ck_ok $CK/B ) & AB=$!
( b2 ls $B/$POPOUT_B2/_dyn_relabel/mix/ 2> /dev/null | grep -q popout_val.parquet || { echo "no pop-out val on B2" > $W/POPOUT_SKIP; exit 0; }
  mkdir -p $PO/_dyn_relabel $PO/_popout_relabel $PO/_ego
  b2 sync --threads 16 --no-progress $B/$POPOUT_B2/_dyn_relabel/ $PO/_dyn_relabel/ > /dev/null 2>&1 \
    && b2 sync --threads 16 --no-progress $B/$POPOUT_B2/_popout_relabel/ $PO/_popout_relabel/ > /dev/null 2>&1 || exit 1
  cd $W/train && uv run -q --no-project --with pandas,pyarrow python -c "import pandas as pd; v = pd.read_parquet('$PO/_dyn_relabel/dyn_index.parquet')
print('\n'.join(sorted(set(v[v.split.isin(['val', 'val_asset'])].src_episode.astype(str)))))" > $W/popout_val_eps.txt || exit 1
  export B PO POPOUT_B2; one() { b2 sync --threads 4 --no-progress --exclude-regex '.*/lidar\.npz$' $B/$POPOUT_B2/$1/ $PO/$1/ && b2 file download --no-progress $B/$POPOUT_B2/_ego/$1.npz $PO/_ego/$1.npz; }
  export -f one; xargs -P 16 -I{} bash -c 'one {} > /dev/null 2>&1 || echo {}' < $W/popout_val_eps.txt > $W/popout_failed.txt; [ ! -s $W/popout_failed.txt ] ) & POP=$!

part venv && part "student pkgs" || fail "staging venv / student pkgs"
source $W/alpamayo-recipes/recipes/alpamayo1_5_sft/a1_5_sft/bin/activate
export PYTHONPATH=$W/train HF_HOME=$W/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
wait $AB || fail "A / B download ($A_B2, $B_B2)"; log "A, B in: $(du -sh $CK/A $CK/B | tr '\n\t' '  ')"
part step0_r2 && part "hf ckpt" || fail "staging step0_r2 / hf ckpt"
python scripts/dyn_stage.py /data/step0_r2 > $R/logs/dyn_stage_r2.log 2>&1 && python scripts/dyn_load_check.py /data/step0_r2 > $R/logs/load_check_r2.log 2>&1 \
  || fail "R2 dyn_relabel staging / loader check (logs/dyn_stage_r2.log, logs/load_check_r2.log)"
if wait $POP && [ ! -f $W/POPOUT_SKIP ] && python scripts/popout_mix.py stage $PO > $R/logs/dyn_stage_popout.log 2>&1 \
   && python scripts/dyn_stage.py $PO >> $R/logs/dyn_stage_popout.log 2>&1 && python scripts/dyn_load_check.py $PO > $R/logs/load_check_popout.log 2>&1; then
  log "pop-out val staged: $(wc -l < $W/popout_val_eps.txt) recorded episodes"
else                                                   # the pop-out set is optional (B2 may not have its val rows yet)
  log "pop-out val set DROPPED: $(cat $W/POPOUT_SKIP 2>/dev/null || echo 'download / staging / loader check failed, logs/*popout*')"; R2_SETS=${R2_SETS//popout/}
fi
until ls $W/GPU_CHECK_* > /dev/null 2>&1; do sleep 5; done; [ -f $W/GPU_CHECK_PASS ] || fail "gpu check (gpu_check.log)"

# ------------------------------------------------------------------ GPU jobs (round-robin) beside the interpolation (CPU)
# general / curve = the R2 run's exact rows (student_dyn / student_stop lists); the rest as student_ft / student_dyn define them
G=(--config-name student_ft data.root=/data/step0 data.coc_parquet=/data/coc/coc_v1_1/coc_dataset.parquet paths.s2=$OUT/s2/final paths.r2_root=/data/step0_r2
   "callbacks.ft_select.sets.general=[{root:'/data/step0',split:[val,val_asset],keep:'assets/stop/general_val.parquet',n:1000}]"
   "callbacks.ft_select.sets.curve=[{root:'/data/step0',split:[val,val_asset],keep:'assets/stop/curve_val.parquet',n:300},{root:'/data/step0_r2',split:[val,val_asset],keep:'assets/stop/curve_val_r2.parquet',n:300}]"
   "+callbacks.ft_select.sets.red_stop=[{root:'/data/step0',split:[val,val_asset],keep:'assets/stop/red_val.parquet',n:300}]"
   "+callbacks.ft_select.sets.dynamic_stop=[{root:'/data/step0_r2',split:[val,val_asset],index:'_dyn_relabel/dyn_index.parquet',keep:'/data/step0_r2/_dyn_relabel/dyn_val.parquet',n:187}]"
   "+callbacks.ft_select.sets.popout=[{root:'$PO',split:[val,val_asset],index:'_dyn_relabel/dyn_index.parquet',keep:'$PO/_dyn_relabel/mix/popout_val.parquet',n:300}]")
csv() { echo $* | tr ' ' ,; }
rec() {                                                # $1 r1|r2, $2 alpha -> $R/a<alpha>/<r1|r2>/sets.json
  local ck=$(ck_of $2) sets=$([ $1 = r1 ] && echo $R1_SETS || echo $R2_SETS) o=$R/a$2/$1
  [ -s $o/sets.json ] && return 0; [ -n "${sets// /}" ] || return 0
  until ck_ok $ck; do [ -f $W/INTERP_CK_FAILED ] && return 1; sleep 10; done
  if [ $1 = r1 ]; then until [ -f $W/R1_READY ]; do [ -f $W/R1_FAILED ] && return 1; sleep 10; done; fi
  python scripts/gate_student.py "${G[@]}" +student=$ck +out=$o "+sets=[$(csv $sets)]" +k=6 > $R/logs/rec_$1_a$2.log 2>&1
}
probe() {
  [ -f $R/probe/DONE ] && return 0
  local a more=(); for a in $ALPHAS; do more+=("a$a=$(ck_of $a)"); done
  until [ -f $W/R1_READY ]; do [ -f $W/R1_FAILED ] && return 1; sleep 10; done
  for a in $ALPHAS; do until ck_ok $(ck_of $a); do [ -f $W/INTERP_CK_FAILED ] && return 1; sleep 10; done; done
  python scripts/ft_probe.py --root /data/step0 --keep $W/probe/keep.parquet --synth $W/probe/synth.parquet --s2 $OUT/s2/final \
    --more "${more[@]}" --out $R/probe --k $PROBE_K > $R/logs/probe.log 2>&1
}
JOBS=(); for a in $ALPHAS; do JOBS+=(rec:r2:$a); done; JOBS+=(probe); for a in $ALPHAS; do JOBS+=(rec:r1:$a); done
NG=$(nvidia-smi -L | wc -l); P=(); rm -f $R/FAILED_JOBS
for g in $(seq 0 $((NG - 1))); do
  ( for ((j = g; j < ${#JOBS[@]}; j += NG)); do
      job=${JOBS[$j]}; log "GPU $g: $job start"; IFS=: read -r f x y <<< "$job"
      CUDA_VISIBLE_DEVICES=$g $f $x $y && log "GPU $g: $job done" || { log "GPU $g: $job FAILED (logs/)"; echo $job >> $R/FAILED_JOBS; }
    done ) & P+=($!)
done
log "GPU jobs on $NG GPU(s): ${JOBS[*]}"
( for a in $ALPHAS; do                                 # CPU, one at a time (disk-bound: reads 2 x 15 GB, writes 15 GB)
    ck_ok $(ck_of $a) && continue
    python scripts/weight_interp.py --a $CK/A --b $CK/B --alpha $a --out $CK/a$a > $R/logs/interp_a$a.log 2>&1 \
      && log "interp alpha $a: $(tail -1 $R/logs/interp_a$a.log)" || { touch $W/INTERP_CK_FAILED; log "weight_interp alpha $a FAILED (logs/interp_a$a.log)"; break; }
  done ) & P+=($!)
wait $STAGE; for p in step0 ckpt; do part $p || { touch $W/R1_FAILED; log "staging $p FAILED: $(grep -E "^[0-9:]+ $p rc=" $L)"; }; done
[ -f $W/R1_FAILED ] || { touch $W/R1_READY; log "R1 staged ($(grep -E '^[0-9:]+ step0 rc=' $L))"; }
for p in "${P[@]}"; do wait $p; done

# ------------------------------------------------------------------ summary, push
python - $R "$ALPHAS" > $R/summary.md <<'PY'
import json, sys
from pathlib import Path
R, alphas = Path(sys.argv[1]), sys.argv[2].split()
res = {a: {k: v for p in ("r2", "r1") if (R / f"a{a}/{p}/sets.json").exists() for k, v in json.load(open(R / f"a{a}/{p}/sets.json")).items()} for a in alphas}
json.dump(res, open(R / "summary.json", "w"), indent=1)
sets = list(dict.fromkeys(s for a in alphas for s in res[a]))
print("ADE / |lateral| (m), K 6, FtSelect's measurement; n rows per set: " + ", ".join(f"{s} {next((res[a][s]['n'] for a in alphas if s in res[a]), '-')}" for s in sets))
print("\n| alpha | " + " | ".join(sets) + " |\n" + "|---" * (len(sets) + 1) + "|")
for a in alphas: print(f"| {a} | " + " | ".join(f"{res[a][s]['ade']:.3f} / {res[a][s]['lat']:.3f}" if s in res[a] else "-" for s in sets) + " |")
PY
[ -f $R/probe/probe_rows.jsonl ] && [ -f $W/probe/hdep.py ] && python $W/probe/hdep.py $R/probe/probe_rows.jsonl $W/probe/rows.csv > $R/hdep.md 2>&1
log "summary:"; cat $R/summary.md $R/hdep.md 2>/dev/null
save; want=$(find $R -type f | wc -l); got=$(b2 ls --recursive $B/runs/interp/$RUN/ | wc -l)
[ "$want" = "$got" ] || fail "push: $want local files, $got on B2 runs/interp/$RUN"
[ ! -s $R/FAILED_JOBS ] && [ ! -f $W/INTERP_CK_FAILED ] && [ ! -f $W/R1_FAILED ] && [ -f $R/probe/DONE ] || fail "jobs: $(cat $R/FAILED_JOBS 2>/dev/null | tr '\n' ' ')"
touch $W/INTERP_DONE; log "INTERP_DONE: b2 runs/interp/$RUN ($got files)"
