#!/usr/bin/env bash
# One-GPU acceptance of the R2 speed-ups and of the R2 pipeline wiring, ~1 h on a rented single-GPU box (README "R2 속도
# 최적화"; rent only with the user's OK). Code first, as for the 8-GPU box (README runbook step 1), then on the box:
#   RUN=<gpu>_<id> setsid nohup bash /workspace/tools/vast/r2ft_probe.sh > /workspace/probe.log 2>&1 < /dev/null &
#  1. staging (r2ft_stage.sh) with R1 cut to ~24 episodes that give every R1 source and validation set rows (~3 GB)
#  2. scripts/ft_speed_probe.py: identity (speed off / off again / on), timing (+ batch 2 under sdpa, kernel launches),
#     ft_select measurement one row per prefill vs 4
#  3. a short real pipeline with every FAST option: ft 10 steps (ft_select every 5 on 16-row sets, 4 rows per prefill),
#     QAD search (8 batches) + 10 steps, both gates on 2 processes, deliver, pushes in the background to
#     b2 runs/r2ft_dry/$RUN. Only kd_proj trains there (one GPU cannot hold the Adam states of 6.5B parameters): it checks
#     the wiring; the math is step 2's job.
# Ends with /workspace/PROBE_DONE (summary in probe.log) or PROBE_FAILED; the idle watchdog stops the box 20 min later.
set -uo pipefail
W=/workspace; B=b2://morai2026-alpamayo; OUT=$W/out/stageR2; DRY=$W/out/dry; RUN=${RUN:-probe}; export RUN
log() { echo "$(date '+%F %T') $*"; }
save() { mkdir -p $W/out/probe; cp $W/probe.log $W/stage.log $W/r1_episodes.txt $W/out/probe.log $W/out/dry_pipeline.log $W/out/probe/ 2>/dev/null
         b2 sync --no-progress $W/out/probe/ $B/runs/r2ft_dry/$RUN/probe/ > /dev/null 2>&1; }      # probe.json + logs (dry stages push themselves)
fail() { log "FAILED: $*"; touch $W/PROBE_FAILED; save; exit 1; }
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^B2_APPLICATION_KEY(_ID)?=' | sed -E "s/^([A-Z0-9_]+)=(.*)$/export \1='\2'/")"
[ -f $W/b2.env ] && set -a && source $W/b2.env && set +a
[ -x /opt/uvtool/bin/b2 ] || { python3 -m venv /opt/uvtool && /opt/uvtool/bin/pip install -q uv b2 vastai; }
ln -sf /opt/uvtool/bin/uv /opt/uvtool/bin/b2 /opt/uvtool/bin/vastai /usr/local/bin/
mkdir -p $W/data/step0/_index && ln -sfn $W/data /data
b2 sync --no-progress $B/step0/_index/ /data/step0/_index/ > /dev/null && b2 sync --no-progress $B/step0/_start_relabel/ /data/step0/_start_relabel/ > /dev/null \
  || fail "R1 index download"
cd $W/train && uv run -q --no-project --with pandas,pyarrow python - > $W/r1_episodes.txt <<'PY' || fail "R1 episode pick"
import pandas as pd                                    # 4 episodes per R1 source / validation pool, most rows first
ix = pd.read_parquet("/data/step0/_index/index.parquet", columns=["episode", "i", "split"]).astype({"episode": str})
top = lambda d, k=4: list(d.episode.astype(str).value_counts().index[:k])
rel = pd.read_parquet("/data/step0/_start_relabel/start_relabel_index.parquet").astype({"episode": str})
eps = top(rel[rel.split == "train"]) + top(pd.read_parquet("assets/ft/old_recovery.parquet")) + top(pd.read_parquet("assets/ft/old_curve.parquet"))
val = ix[ix.split.isin(["val", "val_asset"])]
eps += top(val[val.episode.isin(set(pd.read_parquet("assets/ft/old_static.parquet").episode.astype(str)))]) + top(val, 6)
eps += list(ix[ix.split == "train"].episode.drop_duplicates().sample(4, random_state=0))
print("\n".join(dict.fromkeys(eps)))
PY
log "R1 episodes: $(wc -l < $W/r1_episodes.txt)"
R1_EPISODES=$W/r1_episodes.txt bash $W/tools/vast/r2ft_stage.sh > $W/stage.log 2>&1
for p in step0 step0_r2 ckpt venv "student pkgs" "hf ckpt"; do grep -qE "^[0-9:]+ $p rc=0" $W/stage.log || fail "staging $p: $(grep -E "^[0-9:]+ $p rc=" $W/stage.log)"; done
log "staged: $(grep -E 'gpu check|staging done' $W/stage.log | tr '\n' ' ')"
source $W/alpamayo-recipes/recipes/alpamayo1_5_sft/a1_5_sft/bin/activate
export PYTHONPATH=$W/train HF_HOME=$W/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ATTN=${ATTN:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | grep -q '^12' && echo sdpa || echo flash_attention_2)}
log "probe start ($ATTN)"
python scripts/ft_speed_probe.py --t2 $OUT/t2/final --s2 $OUT/s2/final --r2 /data/step0_r2 --out $W/out/probe --attn $ATTN > $W/out/probe.log 2>&1 \
  || fail "ft_speed_probe.py (out/probe.log)"
log "probe done: $(python -c "import json; r = json.load(open('$W/out/probe/probe.json')); print(json.dumps({k: r[k] for k in ('timing', 'measure')} | {'identity': {k: v for k, v in r['identity'].items() if k != 'rows'}}))")"
# CPU checks after the timed probe, beside the dry run (10/28 code on the default path too): full pytest + the tiny ft run
( uv pip install -q --python $(which python) pytest > /dev/null 2>&1
  /opt/hfenv/bin/hf download nvidia/Alpamayo-1.5-10B config.json --revision 7aba8293c09993f2e125c6819df05d7fa3e873ea > /dev/null 2>&1
  R=$(dirname $(ls $W/hf/hub/models--nvidia--Alpamayo-1.5-10B/snapshots/*/config.json | head -1))
  CUDA_VISIBLE_DEVICES= python -m pytest tests -q -p no:cacheprovider > $W/out/pytest.log 2>&1; echo "pytest rc=$?" >> $W/out/cpu_checks.log
  CUDA_VISIBLE_DEVICES= bash scripts/ft_tiny.sh $R $W/out/ft_tiny > $W/out/ft_tiny.log 2>&1; echo "ft_tiny rc=$?" >> $W/out/cpu_checks.log ) & CPU=$!
mkdir -p $DRY && ln -sfn $OUT/t2 $DRY/t2 && ln -sfn $OUT/s2 $DRY/s2
log "dry pipeline start"
OUT=$DRY B2=$B/runs/r2ft_dry/$RUN STAGES="ft qad deliver" QAD_FROM=ft R2ROOT=/data/step0_r2 GATE_SHARDS=2 PUSH_ASYNC=1 QAD_GATE_N=16 \
  SEXTRA="model.attn_implementation=$ATTN speed.vit_sdpa_grouped=true speed.teacher_vit_only=true speed.lazy_loss_log=true model.train_parts=[kd_proj] trainer.max_steps=10 trainer.eval_steps=5 trainer.save_steps=5 trainer.warmup_steps=2 data.val_dataset.max_samples=16" \
  FTEXTRA="callbacks.ft_select.every_steps=5 callbacks.ft_select.batch=4 callbacks.ft_select.sets.general.0.n=16 callbacks.ft_select.sets.dynamic.0.n=16 callbacks.ft_select.sets.static.0.n=16 callbacks.ft_select.sets.curve.0.n=8 callbacks.ft_select.sets.curve.1.n=8" \
  QEXTRA="quant.calib_samples=8 quant.num_calib_steps=4 quant.num_score_steps=4" bash scripts/pipeline.sh > $W/out/dry_pipeline.log 2>&1
[ -f $DRY/PIPELINE_DONE ] || fail "dry pipeline: $(cat $DRY/HALT 2>/dev/null) (out/dry_pipeline.log)"
log "dry pipeline done: $(grep -h '"stage"' $DRY/*/stage_time.json | tr '\n' ' ')"
wait $CPU; cp $W/out/pytest.log $W/out/ft_tiny.log $W/out/cpu_checks.log $W/out/probe/ 2>/dev/null
log "cpu checks: $(tr '\n' ' ' < $W/out/cpu_checks.log) $(tail -1 $W/out/pytest.log)"
grep -q "pytest rc=0" $W/out/cpu_checks.log && grep -q "ft_tiny rc=0" $W/out/cpu_checks.log || fail "CPU checks (out/pytest.log, out/ft_tiny.log)"
log "ft_select: $(python -c "import json; print([(h['step'], h.get('metrics', {}).get('general'), h['seconds']) for h in json.load(open('$DRY/ft/ft_select.json'))['history']])")"
log "qad gate: $(cat $DRY/qad/summary.json | tr -d '\n ')"
touch $W/PROBE_DONE; log "PROBE_DONE"; save
