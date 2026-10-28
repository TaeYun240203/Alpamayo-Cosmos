#!/usr/bin/env bash
# Stage-B follow-up queue on the vast box, started while 7_step2 and 9_baseline_release run. Two lanes share the GPU once
# step 2 has exited (each lane ~25-35 GB); the timing runs then go alone. Writes $OUT/logs/queue.done at the end.
# Usage: setsid nohup bash scripts/stage_b_queue.sh > /workspace/out/logs/queue.log 2>&1 < /dev/null &
source /workspace/alpamayo-recipes/recipes/alpamayo1_5_sft/a1_5_sft/bin/activate
cd /workspace/train
export PYTHONPATH=/workspace/train HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 OUT=/workspace/out TOKENIZERS_PARALLELISM=false
REL=/workspace/ckpt/Alpamayo-1.5-10B; CK=$OUT/step2/final; L=$OUT/logs
COCP=/data/coc/coc_v1_1/coc_dataset.parquet
C="paths.alpamayo15=$REL data.root=/data/step0 data.coc_parquet=$COCP"
T="torchrun --nproc_per_node 1 --master_port 29502 -m morai_vla.train"   # 29501: smoke
run() { local name=$1; shift; echo "=== $(date +%T) $name"; "$@" > $L/$name.log 2>&1; echo "    $name exit $?"; }
wait_exit() { until grep -q "^exit" $L/$1.log 2>/dev/null; do sleep 30; done; echo "=== $(date +%T) $1 $(grep '^exit' $L/$1.log)"; }
rmtree() { python -c "import shutil, sys; shutil.rmtree(sys.argv[1], ignore_errors=True)" "$1"; }

wait_exit 7_step2
lane_a() {                                     # smoke (CPU AdamW, train + strict reload) then resume check
  bash scripts/smoke_stage1.sh
  R="data.train_dataset.max_samples=20 data.train_dataset.augment=false trainer.save_steps=20 trainer.warmup_steps=5 trainer.eval_strategy=no save_final=false"
  run 12_resume_a $T --config-name step2 $C paths.output_dir=$OUT/resume trainer.max_steps=20 $R
  run 12_resume_b $T --config-name step2 $C paths.output_dir=$OUT/resume trainer.max_steps=40 $R resume=true trainer.save_strategy=no
  rmtree $OUT/resume                           # ~71 GB checkpoint-20
}
lane_b() {                                     # step-2 checkpoint: 1,000-sample ADE with LiDAR on/off, then teacher-forced CE on/off
  wait_exit 9_baseline_release
  run 10_ckpt_baseline python scripts/check_release.py --release $REL --ckpt $CK --root /data/step0 --no-quant --n 1000 --variants traj,traj_nolidar --out $OUT/baseline_step2
  run 10_lidar_onoff python scripts/lidar_onoff_ce.py $CK /data/step0 --n 1000 --out $OUT/lidar_onoff.json
}
lane_a & lane_b & wait
I=$CK; [ -f $CK/config.json ] || I=$REL      # step 2 failed: time on the release
echo "=== $(date +%T) timing alone on $I"; nvidia-smi --query-gpu=memory.used --format=csv,noheader
run 11_time_fwd_bwd_T python scripts/time_fwd_bwd.py $I /data/step0 --n 12
run 11_time_fwd_bwd_C python scripts/time_fwd_bwd.py $I /data/step0 coc --coc-parquet $COCP --n 12
run 11_time_soft_targets python scripts/time_soft_targets.py $I /data/step0 --k 6 --batches 1,4,8 --n 10
du -sh $OUT; touch $L/queue.done; echo "=== $(date +%T) queue done"
