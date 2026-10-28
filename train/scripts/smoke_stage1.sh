#!/usr/bin/env bash
# Stage 1-T and 1-C at real size on one GPU (ZeRO-2 + CPU-offloaded AdamW): ~10 optimizer steps each, save the model,
# strict reload + per-part weight change vs the release, then delete the checkpoint (~23 GB). Correctness only: its
# step time includes the CPU optimizer (use scripts/time_fwd_bwd.py for timing). Runs beside another job on port 29501.
# Usage: bash scripts/smoke_stage1.sh   (same shell setup as stage_b.sh)
source /workspace/alpamayo-recipes/recipes/alpamayo1_5_sft/a1_5_sft/bin/activate
cd /workspace/train
export PYTHONPATH=/workspace/train HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 OUT=/workspace/out TOKENIZERS_PARALLELISM=false
REL=/workspace/ckpt/Alpamayo-1.5-10B
COCP=/data/coc/coc_v1_1/coc_dataset.parquet; [ -f $COCP ] || COCP=/data/step0/_coc/coc_dataset.parquet
C="paths.alpamayo15=$REL data.root=/data/step0 data.coc_parquet=$COCP model.init=$REL"
S="trainer.deepspeed=configs/deepspeed/zero2_offload.json trainer.max_steps=10 trainer.gradient_accumulation_steps=1 trainer.warmup_steps=2 trainer.logging_steps=2 trainer.eval_strategy=no trainer.save_strategy=no data.train_dataset.max_samples=40 trainer.dataloader_num_workers=4 trainer.load_best_model_at_end=false"
mkdir -p $OUT/logs
for st in stage1t stage1c; do
  echo "=== $(date +%T) smoke_$st"
  torchrun --nproc_per_node 1 --master_port 29501 -m morai_vla.train --config-name $st $C paths.output_dir=$OUT/smoke_$st $S > $OUT/logs/8_smoke_$st.log 2>&1
  echo "    train exit $?; final $(du -sh $OUT/smoke_$st/final 2>/dev/null | cut -f1)"
  python scripts/reload_check.py $OUT/smoke_$st/final $REL > $OUT/logs/8_reload_$st.log 2>&1
  echo "    reload exit $?"; tail -30 $OUT/logs/8_reload_$st.log | grep -E "max_abs|camera|lidar|llm|lm_head|expert|other" | tr -s " " | head -12
  python -c "import shutil, sys; shutil.rmtree(sys.argv[1], ignore_errors=True)" $OUT/smoke_$st/final
done
echo "=== $(date +%T) smoke done"
