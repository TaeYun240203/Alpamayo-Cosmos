#!/usr/bin/env bash
# Stage-B runbook (README "단계 B 런북" 1-7) unattended on the vast box. Usage: nohup bash scripts/stage_b.sh > /workspace/out/stage_b.log 2>&1 &
# Every step logs to $OUT/logs/; one failing step does not stop the next (the summary at the end lists exit codes).
source /workspace/alpamayo-recipes/recipes/alpamayo1_5_sft/a1_5_sft/bin/activate
cd /workspace/train
export PYTHONPATH=/workspace/train HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 OUT=/workspace/out TOKENIZERS_PARALLELISM=false
REL=/workspace/ckpt/Alpamayo-1.5-10B
COCP=/data/coc/coc_v1_1/coc_dataset.parquet; [ -f $COCP ] || COCP=/data/step0/_coc/coc_dataset.parquet
C="paths.alpamayo15=$REL data.root=/data/step0 data.coc_parquet=$COCP"
T="torchrun --nproc_per_node 1 -m morai_vla.train"
mkdir -p $OUT/logs
declare -A RC
run() { local name=$1; shift; echo "=== $(date +%T) $name"; "$@" > $OUT/logs/$name.log 2>&1; RC[$name]=$?; echo "    exit ${RC[$name]}"; }

run 1_tests python -m pytest -q tests
run 2_release python scripts/check_release.py --release $REL --root /data/step0 --no-quant --n 48 --out $OUT/release_bf16

O="data.train_dataset.max_samples=20 data.train_dataset.augment=false trainer.max_steps=300 trainer.gradient_accumulation_steps=1 trainer.warmup_steps=10 trainer.eval_strategy=no trainer.save_strategy=no save_final=false"
run 3_ovf_step2  $T --config-name step2  $C paths.output_dir=$OUT/ovf_step2 $O
run 3_ovf_stage2 $T --config-name stage2 $C model.init=$REL paths.output_dir=$OUT/ovf_stage2 $O

P="trainer.max_steps=30 trainer.logging_steps=10 trainer.gradient_accumulation_steps=1 trainer.eval_strategy=no trainer.save_strategy=no save_final=false"
for B in 1 2 4; do run 4_prof_step2_b$B $T --config-name step2 $C paths.output_dir=$OUT/prof_step2_b$B trainer.per_device_train_batch_size=$B $P; done
run 4_prof_stage2_b1 $T --config-name stage2 $C model.init=$REL paths.output_dir=$OUT/prof_stage2_b1 trainer.per_device_train_batch_size=1 $P
run 4_prof_stage2_b2 $T --config-name stage2 $C model.init=$REL paths.output_dir=$OUT/prof_stage2_b2 trainer.per_device_train_batch_size=2 model.attn_implementation=sdpa $P

run 5_io python scripts/bench_io.py --config-name step2 $C trainer.dataloader_num_workers=8 +n=400

# resume: stop cleanly at a checkpoint (step 20), then continue to 40 from it
R="data.train_dataset.max_samples=20 data.train_dataset.augment=false trainer.save_steps=20 trainer.warmup_steps=5 trainer.eval_strategy=no save_final=false"
run 6_resume_a $T --config-name step2 $C paths.output_dir=$OUT/resume trainer.max_steps=20 $R
run 6_resume_b $T --config-name step2 $C paths.output_dir=$OUT/resume trainer.max_steps=40 $R resume=true

# full step 2: episodes are enumerated at start, so run it on the complete copy. Batch 4 x accumulation 1 = the recipe's
# effective 4, ~25 % faster than 1 x 4 (measured). No intermediate checkpoints: one is ~71 GB (DeepSpeed model states +
# HF weights) and the 300 GB disk also holds the 200 GB data; output_dir/final (~23 GB) is saved at the end.
run 7_step2 $T --config-name step2 $C paths.output_dir=$OUT/step2 trainer.per_device_train_batch_size=4 trainer.gradient_accumulation_steps=1 trainer.save_strategy=no

echo "=== $(date +%T) done"; for k in "${!RC[@]}"; do echo "$k ${RC[$k]}"; done | sort
