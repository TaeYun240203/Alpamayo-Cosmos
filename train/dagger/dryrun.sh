#!/usr/bin/env bash
# Dry run of the DAgger runner on the training box while R1 step-0 downloads (README "8×H100 dry run"): the real delivery
# and sizes on every GPU, replay = R2 sources only (R1 not in yet; same code path; R1 val sets come up empty), a fake spool of real-size requests
# (dagger/fake.py) -> load -> 20 steps -> samples/s, memory, actors at balance (metrics/summary.json) -> one checked
# publish. Starting values here are the README proposals, for the measurement only.
# Needs: venv + student pkgs, /data/step0_r2, HF Qwen3-VL-4B, the delivery (BASE, or BASE_B2 to fetch it), the engine
# code at $E2E (tar of the eval repo's src/e2e_driver/server + src/e2e_driver/src, README).
# Usage on the box (cwd /workspace/train): BASE_B2=b2://morai2026-alpamayo/runs/<...>/deliver/final bash dagger/dryrun.sh
#   default expert (user 10/28: the DAgger LoRA; its summary's actors_at_balance sizes the actors); llm+expert: on hold
set -euo pipefail
LORA=${1:-expert}; W=${W:-/workspace}; O=$W/out/dagger_dry/${LORA/+/_}; NG=${NG:-$(nvidia-smi -L | wc -l)}
VENV=${VENV:-$W/alpamayo-recipes/recipes/alpamayo1_5_sft/a1_5_sft}; PY=$VENV/bin/python
E2E=${E2E:-$W/e2e/server}; R2=${R2:-/data/step0_r2}; BASE=${BASE:-$W/ckpt/deliver/final}
ATTN=${ATTN:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | grep -q '^12' && echo sdpa || echo flash_attention_2)}
[ -z "${B2_APPLICATION_KEY:-}" ] && eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^B2_APPLICATION_KEY(_ID)?=' | sed 's/^/export /')"
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH} HF_HOME=${HF_HOME:-$W/hf} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false      # as scripts/pipeline.sh
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
[ -f $BASE/config.json ] || b2 sync --threads 16 --no-progress ${BASE_B2:?no delivery at $BASE: set BASE_B2} $BASE/
[ -f $E2E/assemble.py ] || { echo "no engine code at $E2E"; exit 1; }
mkdir -p $O; [ -d $O/spool ] || $PY -m dagger.fake $O/spool --e2e_server $E2E --actors $NG --episodes 2 --decisions 60
R2ONLY=(--override dagger.replay.sources.r1_general.share=0 --override dagger.replay.sources.signal_approach.share=0   # R1 not in
        --override dagger.replay.sources.signal_hold.share=0                                                       # yet:
        --override dagger.replay.sources.new_a.share=0.25 --override dagger.replay.sources.new_b.share=0.125     # the replay half
        --override dagger.replay.sources.new_d.share=0.125)                                                       # from R2 only
$VENV/bin/torchrun --nproc_per_node $NG -m dagger.runner --base $BASE --spool $O/spool --out $O/run --data_root $R2 --r2_root $R2 \
  --e2e_server $E2E "${R2ONLY[@]}" --lora $LORA --attn $ATTN ${DRY_EXTRA:-} \
  --lr 5e-5 --rank 16 --alpha 32 --batch ${BATCH:-$((4 * NG))} --publish_min 1000 --min_rows 64 \
  --max_steps ${STEPS:-20} --publish_at_end --val_n 32 --val_dagger 32 --stale_stop_min 0 2>&1 | tee $O/dryrun.log
cat $O/run/metrics/summary.json; ls -la $O/run/lora/
