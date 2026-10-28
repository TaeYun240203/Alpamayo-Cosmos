# Student half of scripts/pipeline.sh (sourced there; owned by the student code; README "학생" runbook): soft s1 sgate s2
# ft qad deliver qgate, one function per stage + gate_<stage>. Contract (README "단계 C 파이프라인"): everything under $OUT/<stage>,
# training stages end with <stage>/final, metrics.json per stage, non-zero exit on failure; pipeline.sh gates, pushes to
# B2 and marks the stage done.
# From pipeline.sh: $OUT $ROOT $COCP $NG $TR $B2, need <stage> (final/ dir, pulled from B2 if missing), halt, resume_flag.
# Dry runs: SEXTRA = Hydra overrides appended to every student command (e.g. "+tiny=student trainer.max_steps=20"),
# QEXTRA = to the QAD search only (quant.*), S1EXTRA = to s1 only (callbacks.traj_early_stop.*), SGATE_N / QAD_GATE_N =
# gate sample counts.
# R2 fine-tune (README "R2 미세조정"): STAGES="ft qad deliver qgate" QAD_FROM=ft R2ROOT=<R2 step-0 root> [RELABEL=<start relabel
# parquet>, default $ROOT/_start_relabel/start_relabel_index.parquet] with $OUT/s2/final in place; FTEXTRA = Hydra
# overrides of ft only. User 10/28: hard targets, no T* (no t2, no rule_ok): ft = student_ft, QAD = student_qad_r2 (the ft
# mixture), gates = the student alone on student_ft's val rows + records (dynamic, curve, start relabel; other GPUs).
# GATE_SHARDS=<n>: the QAD gates on n processes over the GPUs (10/28, math unchanged; unused: FAST off).
SEXTRA=${SEXTRA:-}; QEXTRA=${QEXTRA:-}; S1EXTRA=${S1EXTRA:-}; SGATE_N=${SGATE_N:-1000}; QAD_GATE_N=${QAD_GATE_N:-500}
QAD_FROM=${QAD_FROM:-s2}; FTEXTRA=${FTEXTRA:-}; FT_CFG=${FT_CFG:-student_ft}; QAD_CFG=${QAD_CFG:-student_qad_r2}   # 10/28 stop round: student_stop / student_qad_stop
 RELABEL=${RELABEL:-$ROOT/_start_relabel/start_relabel_index.parquet}
S="data.root=$ROOT data.coc_parquet=$COCP paths.teacher=$OUT/t2/final"
R2="data.root=$ROOT data.coc_parquet=$COCP paths.s2=$OUT/s2/final paths.r2_root=${R2ROOT:-} paths.start_relabel=$RELABEL"   # R2 round, no T*
GCFG="--config-name student1 $S"                       # the QAD gate's config (qad sets the R2 one)

need_dir() { [ -d $OUT/$1 ] || b2 sync --threads 16 $B2/$1 $OUT/$1 > $OUT/logs/pull_${1//\//_}.log 2>&1 || halt "cannot pull $1"; }
loss_metrics() {                                       # $1 = run dir: last train/eval loss lines, the kept checkpoint -> metrics.json
  python - $1 <<'PY'
import glob, json, sys
d = sys.argv[1]
rows = [json.loads(l) for l in open(f"{d}/loss_components.jsonl")]
tr = [r for r in rows if not any(k.startswith("eval_") for k in r)]; ev = [r for r in rows if any(k.startswith("eval_") for k in r)]
best = {k: min(r[k] for r in ev if k in r) for k in {k for r in ev for k in r if k.startswith("eval_")}}
ck = sorted(glob.glob(f"{d}/checkpoint-*/trainer_state.json"), key=lambda f: int(f.split("checkpoint-")[1].split("/")[0]))
st = json.load(open(ck[-1])) if ck else {}
final = {k: st.get(k) for k in ("best_global_step", "best_metric", "best_model_checkpoint")}   # final/ = this checkpoint (eval_fm)
try:                                                   # s1: final/ = the lowest trajectory-ADE measurement (traj_early_stop)
    sel = json.load(open(f"{d}/ade_select.json")); es = json.load(open(f"{d}/ade_early_stop.json"))
    final = {k: sel[k] for k in ("chosen_step", "chosen_ade", "chosen_tag")} | {k: es.get(k) for k in ("stop_step", "decay_end", "min_step")}
except FileNotFoundError:
    pass
try:                                                   # ft: final/ = the R2 selection rule (ft_select)
    sel = json.load(open(f"{d}/ft_choice.json"))
    final = {k: sel[k] for k in ("chosen_step", "chosen_tag", "chosen_ade", "ref_start", "limit", "last_step")}
except FileNotFoundError:
    pass
json.dump({"final": final, "last_train": tr[-1] if tr else None, "last_eval": ev[-1] if ev else None, "best_eval": best},
          open(f"{d}/metrics.json", "w"), indent=1)
PY
}
student_gate() {                                       # non-finite anywhere; divergence only on the stable eval losses (eval_lm of the
  python scripts/pipeline_gate.py train $OUT/$1 --no-divergence || halt "$1 gate: $(python scripts/pipeline_gate.py train $OUT/$1 --no-divergence)"
  python - $OUT/$1 <<'PY' || halt "$1 diverged: $(cat $OUT/$1/DIVERGED)"
import json, sys                                       # end token is ~1e-4 and FM is noise-sampled: 1.2x of those trips on noise)
rows = [json.loads(l) for l in open(f"{sys.argv[1]}/loss_components.jsonl")]
for k in ("eval_kv", "eval_traj_ce"):
    s = [r[k] for r in rows if k in r]
    if s and s[-1] > 1.2 * min(s) and min(s) > 1e-3:
        open(f"{sys.argv[1]}/DIVERGED", "w").write(f"{k} final {s[-1]:.4f} > 1.2 x best {min(s):.4f}"); sys.exit(1)
PY
}

soft() {                                               # T* soft targets on the student's train rows (rule check: start of s2)
  need t2
  torchrun --nproc_per_node $NG scripts/soft_targets.py --config-name student1 $S +out=$OUT/soft/parts +batch=8 +k=6 $SEXTRA || return 1
  python - $OUT/soft <<'PY'
import glob, json, re, sys, numpy as np
parts = sorted(glob.glob(f"{sys.argv[1]}/parts/part-*.npz")); n = sum(len(np.load(f)["key"]) for f in parts)
ms = [float(m) for f in glob.glob(f"{sys.argv[1]}/../logs/soft.log") for m in re.findall(r"(\d+) ms/sample", open(f).read())]
json.dump({"samples": n, "parts": len(parts), "ms_per_sample_per_rank": round(float(np.median(ms)), 1) if ms else None}, open(f"{sys.argv[1]}/metrics.json", "w"), indent=1)
sys.exit(n == 0)
PY
}
s1() {                                                 # student stage 1: 4B + LiDAR copy + T*'s expert (frozen); T* frozen in the step
  need t2
  $TR --config-name student1 $S paths.output_dir=$OUT/s1 trainer.save_total_limit=1 resume=$(resume_flag $OUT/s1) $SEXTRA $S1EXTRA \
    && loss_metrics $OUT/s1
}
gate_s1() { student_gate s1; }
sgate() {                                              # frozen-expert student vs T*: tgate's 1,000 val samples (seed 1), same noise
  need t2; need s1
  python scripts/gate_student.py --config-name student1 $S +student=$OUT/s1/final +out=$OUT/sgate +n=$SGATE_N +k=6 data.val_dataset.seed=1 $SEXTRA \
    && cp $OUT/sgate/summary.json $OUT/sgate/metrics.json
}
gate_sgate() {   # halt: `ratio` (student ADE / T* ADE, same samples and noise) not below 1.5. The plan's target ratio <= 1.1
                 # (README student runbook 6) is only reported: missing it is a decision (PAI pre-distillation), not a halt
  python scripts/pipeline_gate.py max $OUT/sgate/summary.json ratio 1.5 \
    || halt "student gate: $(python scripts/pipeline_gate.py max $OUT/sgate/summary.json ratio 1.5)"
}
s2() {                                                 # student stage 2: rule check of the soft targets, then expert at 0.1x lr + vkd
  need t2; need s1; need_dir soft/parts
  [ -f $OUT/s2/rule_ok.parquet ] || python scripts/rule_check.py --parts $OUT/soft/parts --root $ROOT --out $OUT/s2/rule_ok.parquet || return 1
  $TR --config-name student2 $S model.init=$OUT/s1/final data.train_dataset.rule_ok=$OUT/s2/rule_ok.parquet paths.output_dir=$OUT/s2 \
    trainer.save_total_limit=1 resume=$(resume_flag $OUT/s2) $SEXTRA && loss_metrics $OUT/s2
}
gate_s2() { student_gate s2; }
ft() {                                                 # R2 fine-tune from s2/final (configs/student_ft.yaml); final/ = last measurement
  need s2
  $TR --config-name $FT_CFG $R2 paths.output_dir=$OUT/ft resume=$(resume_flag $OUT/ft) $SEXTRA $FTEXTRA && loss_metrics $OUT/ft \
    && python scripts/ft_metrics.py $OUT/ft
}
gate_ft() {                                            # NaN / inf; the stop switch (ft_stop.json) halts before QAD
  python scripts/pipeline_gate.py train $OUT/ft --no-divergence || halt "ft gate: $(python scripts/pipeline_gate.py train $OUT/ft --no-divergence)"
  [ ! -f $OUT/ft/ft_stop.json ] || halt "ft stop switch at step $(python -c "import json; print(json.load(open('$OUT/ft/ft_stop.json'))['step'])")"
}
qad() {                                                # AutoQuantize on one GPU -> short QAD training -> manifest; then deliver, then
  need $QAD_FROM                                       # qgate (user 10/27, 10/28: early delivery right after QAD, the gate after it)
  local Q                                              # input: s2 (stage C) or ft (R2 round)
  if [ "$QAD_FROM" = ft ]; then Q="--config-name $QAD_CFG $R2"   # user 10/28: no T*
  else need t2; Q="--config-name student_qad $S data.train_dataset.rule_ok=$OUT/s2/rule_ok.parquet"; fi
  [ -f $OUT/qad/search/modelopt_state.pth ] || python scripts/qad_search.py $Q model.init=$OUT/$QAD_FROM/final \
    +out=$OUT/qad/search paths.output_dir=$OUT/qad $SEXTRA $QEXTRA || return 1
  [ -f $OUT/qad/final/modelopt_state.pth ] || $TR $Q model.init=$OUT/qad/search paths.output_dir=$OUT/qad \
    trainer.save_total_limit=1 resume=$(resume_flag $OUT/qad) $SEXTRA && loss_metrics $OUT/qad || return 1
  python scripts/qad_export.py manifest $OUT/qad || return 1
  # user 10/28 "QAD되자마자 납품": export + push the delivery here, before qad's own ~30 GB push (12 MB/s boxes: ~40 min earlier);
  # the deliver stage then rewrites the same files (copytree keeps the weights' mtimes, so its push skips them)
  python scripts/qad_export.py deliver $OUT/qad $OUT/deliver && push deliver
}
gate_qad() { student_gate qad; }                       # NaN / divergence of the QAD run itself; the ADE gate is qgate, after the delivery
deliver() { need qad; python scripts/qad_export.py deliver $OUT/qad $OUT/deliver; }   # what the 5080 engine conversion reads
qgate() {                                              # input vs QAD student on the same val rows (+ R2 records) -> qgate/summary.json,
  need $QAD_FROM; need qad                             # copied to deliver/gate.json; never removes the delivery
  local k=1 p x1 st x=() gp=(${GPUS//,/ }) G=$OUT/qgate
  if [ "$QAD_FROM" = ft ]; then
    GCFG="--config-name $FT_CFG $R2"                # R2: the student alone (no T*), student_ft's val rows
    for st in $OUT/ft/final:gate_s2 $OUT/qad/final:gate_qad; do   # records (user 10/28) on GPUs 1.. beside the gates
      for x1 in ${QGATE_SETS:-dynamic curve start_relabel}; do   # stop round: red_stop dynamic_stop curve start_relabel
        CUDA_VISIBLE_DEVICES=${gp[$((k % NG))]} python scripts/gate_student.py $GCFG +student=${st%%:*} +out=$G/${st#*:}/sets_$x1 \
          +sets=[$x1] +k=6 $SEXTRA > $OUT/logs/qgate_sets_${st#*:}_$x1.log 2>&1 & x+=($!); k=$((k + 1))
      done
    done
  fi
  CUDA_VISIBLE_DEVICES=${gp[0]} gate $OUT/$QAD_FROM/final $G/gate_s2 && CUDA_VISIBLE_DEVICES=${gp[0]} gate $OUT/qad/final $G/gate_qad || return 1
  for p in "${x[@]}"; do wait $p || log "QAD record job $p failed (logs/qgate_sets_*.log): records only, the gate stands"; done
  python scripts/qad_export.py summary $G || return 1
  if [ -d $OUT/deliver/final ]; then cp $G/summary.json $OUT/deliver/gate.json && push deliver || return 1; fi
}
gate_qgate() {                                         # halt (the delivery stays; deliver/gate.json says pass false): QAD ADE not below
  python scripts/pipeline_gate.py max $OUT/qgate/summary.json ratio_vs_s2 1.25 \
    || halt "QAD gate: $(python scripts/pipeline_gate.py max $OUT/qgate/summary.json ratio_vs_s2 1.25)"   # 1.25 x its input's
}
gate() {                                               # $1 student, $2 out: QAD gate rows on GATE_SHARDS processes, one per GPU
  local G="python scripts/gate_student.py $GCFG +n=$QAD_GATE_N +k=6 data.val_dataset.seed=1 $SEXTRA"
  [ "${GATE_SHARDS:-1}" -gt 1 ] || { $G +student=$1 +out=$2; return; }   # (10/26: 1 GPU, ~22 min for both gates, 7 idle)
  local k p pids=() gp=(${GPUS//,/ })
  for k in $(seq 0 $((GATE_SHARDS - 1))); do           # rows k::n keep their own seeds: the merged gate equals the 1-process one
    CUDA_VISIBLE_DEVICES=${gp[$((k % NG))]} $G +student=$1 +out=$2/shard$k +shard=$k/$GATE_SHARDS > $OUT/logs/$(basename $2)_$k.log 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || return 1; done
  python -c "import sys; from scripts.gate_student import merge; merge(sys.argv[1], sys.argv[2:])" $2 $2/shard*
}
