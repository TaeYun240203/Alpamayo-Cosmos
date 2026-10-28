#!/usr/bin/env bash
# Rent and warm up the DAgger learner on Brev (user 10/28 "Brev H100"): create -> code + b2 account -> dagger/brev_stage.sh
# (acceptance, downloads, 8-GPU expert dry run). The b2 key goes from winpc's authorised b2 account file straight into the
# box's root config through pipes: never on a command line, in a file on the Mac, or in a log.
# Usage (Mac): NAME=dagger-learner bash train/dagger/brev_up.sh      then: ssh dagger-learner tail -f /workspace/stage.log
set -euo pipefail
NAME=${NAME:-dagger-learner}; TYPE=${TYPE:-hyperstack_H100x8_NVLINK}; RUN=${RUN:-brev_h100x8_$(date +%m%d%H%M)}
WS=$(cd "$(dirname "$0")/../../.." && pwd); REPO=$WS/Alpamayo-Cosmos; EVAL=$WS/MORAI-eval-sync/src/e2e_driver
brev create "$NAME" --type "$TYPE"
brev refresh > /dev/null 2>&1 || true
until ssh -T -o ConnectTimeout=10 "$NAME" true 2> /dev/null; do sleep 10; done
T=$(mktemp -d); mkdir -p $T/c/e2e/server $T/c/e2e/src $T/c/tools
# the repo lives in iCloud: a file read can time out (10/28), so copy with rsync (retried) and tar the local copy
for k in 1 2 3; do
  rsync -a --exclude '*.bak*' --exclude __pycache__ --exclude 'logs/' --exclude 'outputs/' $REPO/train $T/c/ \
    && rsync -a --exclude __pycache__ $REPO/tools/vast $REPO/tools/scorer $T/c/tools/ \
    && rsync -a $EVAL/server/assemble.py $T/c/e2e/server/ && rsync -a --exclude __pycache__ $EVAL/src/e2e_driver $T/c/e2e/src/ && break
  [ $k = 3 ] && { echo "code copy failed"; exit 1; }; sleep 5
done
tar czf $T/code.tgz -C $T/c train tools e2e
scp -q $T/code.tgz $REPO/train/dagger/brev_stage.sh "$NAME":/tmp/
ssh winpc-wsl 'cat ~/.config/b2/account_info' \
  | ssh -T "$NAME" 'sudo mkdir -p /root/.config/b2 && sudo tee /root/.config/b2/account_info > /dev/null && sudo chmod 600 /root/.config/b2/account_info'
ssh -T "$NAME" "sudo RUN=$RUN setsid nohup bash /tmp/brev_stage.sh > /tmp/stage_boot.log 2>&1 < /dev/null &"
echo "$NAME up at $(date +%T), RUN=$RUN"
