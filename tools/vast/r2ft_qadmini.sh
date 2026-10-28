#!/usr/bin/env bash
# QAD R2 mini test while R1 still downloads (user 10/28 "이게 좋겠어"): the new QAD path end to end on the real s2, R2 data
# only, one GPU, ~10-15 min, beside the speed test (r2ft_chain.sh: GPU 0 here, GPUs 1.. the speed test):
#   qad_search (student_qad_r2, 8 calibration batches) -> QAD training 5 steps (only the LiDAR ViT trains: one GPU cannot
#   hold the Adam states of 6.5B parameters; the fake-quant forward/backward, save and selection are the real ones)
#   -> manifest -> deliver (export) -> gates (student alone, 8 rows) + records (dynamic, 8 rows) -> summary -> gate.json.
# Never blocks ft. Result: /workspace/out/qadmini/RESULT = "PASS <s>" | "FAIL <step>" (| "INCOMPLETE ..." written by the
# chain when R1 lands first); logs and JSON (no weights) to b2 runs/r2ft/qadmini/<RUN>/. tools/vast/r2ft_watch.sh alerts.
# the chain runs from an ssh session, which has no container env: B2 keys from PID 1, as pipeline.sh (10/28 box: without them the
# start-relabel download failed and gate_s2 stopped on the missing file)
[ -n "${B2_APPLICATION_KEY_ID:-}" ] || eval "$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | grep -E '^B2_APPLICATION_KEY(_ID)?=' | sed 's/^/export /')"
W=/workspace; OUT=$W/out/stageR2; T=$W/out/qadmini; B=b2://morai2026-alpamayo; RUN=$(cat $W/RUN 2>/dev/null || echo box)
source $W/alpamayo-recipes/recipes/alpamayo1_5_sft/a1_5_sft/bin/activate
cd $W/train; export PYTHONPATH=$PWD HF_HOME=$W/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${QADMINI_GPU:-0}; ulimit -n $(ulimit -Hn)
ATTN=${ATTN:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | grep -q '^12' && echo sdpa || echo flash_attention_2)}
mkdir -p $T; rm -f $T/RESULT; t0=$(date +%s)
finish() { b2 sync --no-progress --exclude-regex '.*\.(safetensors|bin|pt|pth|npz)$' $T/ $B/runs/r2ft/qadmini/$RUN/ > /dev/null 2>&1; }
step() { local n=$1; shift; echo "$(date +%T) $n"; "$@" > $T/$n.log 2>&1 || { echo "FAIL $n" > $T/RESULT; echo "$(date +%T) FAIL $n"; finish; exit 1; }; }
b2 file download --no-progress $B/step0/_start_relabel/start_relabel_index.parquet $T/start_relabel.parquet > /dev/null 2>&1   # 75 KB
# the R2 root alone (R1 is still downloading): R1 sources out, replay reads R2 rows, as r2ft_speed.sh
R2="data.root=/data/step0_r2 paths.s2=$OUT/s2/final paths.r2_root=/data/step0_r2 paths.start_relabel=$T/start_relabel.parquet
    model.attn_implementation=$ATTN ~data.train_dataset.sources.old_recovery ~data.train_dataset.sources.old_curve
    ~data.train_dataset.sources.start_relabel data.train_dataset.sources.replay.dataset.exclude=null"
G="python scripts/gate_student.py --config-name student_ft $R2 +k=2 data.val_dataset.seed=1"
step search python scripts/qad_search.py --config-name student_qad_r2 $R2 model.init=$OUT/s2/final +out=$T/qad/search \
  paths.output_dir=$T/qad quant.calib_samples=8 quant.num_calib_steps=4 quant.num_score_steps=4
step train torchrun --nproc_per_node 1 --master_port 29611 -m morai_vla.train --config-name student_qad_r2 $R2 model.init=$T/qad/search \
  paths.output_dir=$T/qad model.train_parts=[lidar] trainer.max_steps=5 trainer.eval_steps=5 trainer.save_steps=5 trainer.warmup_steps=1 \
  trainer.logging_steps=1 trainer.save_total_limit=1 data.val_dataset.max_samples=8
step manifest python scripts/qad_export.py manifest $T/qad
step deliver python scripts/qad_export.py deliver $T/qad $T/deliver
step gate_s2 $G +n=8 +student=$OUT/s2/final +out=$T/qgate/gate_s2
step gate_qad $G +n=8 +student=$T/qad/final +out=$T/qgate/gate_qad
step sets_s2 $G +sets=[dynamic] callbacks.ft_select.sets.dynamic.0.n=8 +student=$OUT/s2/final +out=$T/qgate/gate_s2/sets_dynamic
step sets_qad $G +sets=[dynamic] callbacks.ft_select.sets.dynamic.0.n=8 +student=$T/qad/final +out=$T/qgate/gate_qad/sets_dynamic
step summary python scripts/qad_export.py summary $T/qgate
cp $T/qgate/summary.json $T/deliver/gate.json
echo "PASS $(( $(date +%s) - t0 ))" > $T/RESULT; echo "$(date +%T) $(cat $T/RESULT)"; finish
