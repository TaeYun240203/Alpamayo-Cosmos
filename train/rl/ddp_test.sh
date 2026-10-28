#!/usr/bin/env bash
# rl/ddp_test.py on 2 CPU ranks (gloo). Usage (cwd train/): E2E_SERVER=... bash rl/ddp_test.sh <tiny deliver> <work> [python]
set -euo pipefail
DELIVER=$1; WORK=$2; PY=${3:-python}
rm -rf "$WORK/spool" "$WORK/out"; mkdir -p "$WORK"
export HF_HUB_OFFLINE=1 PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
$PY -m rl.ddp_test prep0 "$DELIVER" "$WORK"
$PY -m torch.distributed.run --nproc_per_node 2 -m rl.ddp_test run "$DELIVER" "$WORK"
$PY -m rl.ddp_test prep1 "$DELIVER" "$WORK"
$PY -m torch.distributed.run --nproc_per_node 2 -m rl.ddp_test run "$DELIVER" "$WORK"   # resumes from state.pt
$PY -m rl.ddp_test check "$DELIVER" "$WORK"
