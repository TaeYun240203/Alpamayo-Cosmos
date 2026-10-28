#!/usr/bin/env bash
# The PPO learner's speed on this box while the data and the delivery come in (PPO_PLAN 8; user 10/28: size the actors
# from the learner's measured speed): a fake spool at the real sizes (rl/fake.py) -> one update on every GPU (torchrun)
# -> policy decisions per second -> actors at balance (metrics in $O/summary.json).
# Needs: venv + student pkgs, HF Qwen3-VL-4B, the delivery (BASE, or BASE_B2 to fetch it), the engine code at $E2E (tar of
# the eval repo's src/e2e_driver/server + src/e2e_driver/src, as train/dagger/README.md).
# Usage on the box (cwd /workspace/train): BASE_B2=b2://morai2026-alpamayo/<deliver>/final bash rl/dryrun.sh
#   ACTOR_RATE: policy decisions per second of one actor = 10 Hz x RTF 0.84 (vast 5090 real time, measured) x 0.77 (share
#   of decisions inside the PPO expert zone s 800-1841.7 over round + hwy episodes half and half, estimate) = 6.5
set -euo pipefail
W=${W:-/workspace}; O=$W/out/ppo_dry; NG=${NG:-$(nvidia-smi -L | wc -l)}
VENV=${VENV:-$W/alpamayo-recipes/recipes/alpamayo1_5_sft/a1_5_sft}; PY=$VENV/bin/python
E2E=${E2E:-$W/e2e/server}; BASE=${BASE:-$W/ckpt/deliver/final}; ACTOR_RATE=${ACTOR_RATE:-6.5}
[ -z "${B2_APPLICATION_KEY:-}" ] && eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^B2_APPLICATION_KEY(_ID)?=' | sed 's/^/export /')"
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH} HF_HOME=${HF_HOME:-$W/hf} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
[ -f $BASE/config.json ] || b2 sync --threads 16 --no-progress ${BASE_B2:?no delivery at $BASE: set BASE_B2} $BASE/
[ -f $E2E/assemble.py ] || { echo "no engine code at $E2E"; exit 1; }
rm -rf $O/run; mkdir -p $O
[ -d $O/spool ] || $PY -m rl.fake $O/spool --e2e_server $E2E --actors 1 --episodes $((3 * NG)) --decisions 100
$VENV/bin/torchrun --nproc_per_node $NG -m rl.learner --base $BASE --spool $O/spool --out $O/run --e2e_server $E2E \
  --min_decisions 1 --max_updates 1 --stale_stop_min 0 --kl_target 14.4 ${DRY_EXTRA:-} 2>&1 | tee $O/dryrun.log
$PY - "$O" "$NG" "$ACTOR_RATE" <<'EOF'
import json, math, sys
o, ng, rate = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
x = json.loads(open(f"{o}/run/log.jsonl").readline())
pps = 1 / x["s_per_policy_decision"]                 # includes the proximal pass, the epoch and the critic fit
s = dict(gpus=ng, policy_decisions=x["policy_decisions"], policy_decisions_per_s=round(pps, 2), per_gpu=round(pps / ng, 2),
         peak_mem_gb=x["peak_mem_gb"], actor_rate=rate, actors_at_balance=math.floor(pps / rate), update=x)
json.dump(s, open(f"{o}/summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in s.items() if k != "update"}))
EOF
