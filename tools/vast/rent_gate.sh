#!/bin/bash
# Pre-rent gate (CLAUDE.md §1). Writes ~/.alpamayo-state/rent_ok (valid 30 min) only when every check passes;
# the user-level PreToolUse hook (~/.claude/hooks/rent_guard.sh) blocks vastai create / runpod / brev create without it.
# Usage: bash tools/vast/rent_gate.sh --purpose "dynbal ft+QAD" --target "ft 1000 + QAD 800 step" --end "deliver on B2 runs/dynbal" \
#          --b2 b2://morai2026-alpamayo/step0_popout/_popout_relabel/ --expect 5231 --loader-log /path/with/rc=0 --gpu "num_gpus=8 gpu_name in [H100_SXM,H200]"
set -u; S=~/.alpamayo-state; mkdir -p $S
PURPOSE=; TARGET=; END=; B2=; EXPECT=; LOADER=; GPU=
while [ $# -gt 0 ]; do case $1 in --purpose) PURPOSE=$2;; --target) TARGET=$2;; --end) END=$2;; --b2) B2=$2;; --expect) EXPECT=$2;; --loader-log) LOADER=$2;; --gpu) GPU=$2;; esac; shift 2; done
fail() { echo "GATE FAIL: $*"; exit 1; }
[ -e $S/no_rent ] && fail "no_rent flag set ($(cat $S/no_rent)) — the user forbade renting; remove only on the user's word"
[ -n "$PURPOSE" ] && [ -n "$TARGET" ] && [ -n "$END" ] || fail "--purpose, --target (numeric goal) and --end (stop condition) are required"
W="ssh -o ConnectTimeout=8 -o HostKeyAlias=[100.98.142.90]:2222 -p 2222 -i $HOME/.ssh/id_ed25519 -o IdentitiesOnly=yes taeyun_kim@192.168.0.1"
if [ -n "$B2" ]; then
  n=$($W "~/b2tool/bin/b2 ls -r $B2 | wc -l" 2>/dev/null | tr -d ' '); echo "B2 $B2: $n files (expect $EXPECT)"
  [ -n "$n" ] && [ "${n:-0}" -ge "${EXPECT:-0}" ] || fail "B2 has $n files, expected >= $EXPECT — data not fully uploaded"
fi
if [ -n "$LOADER" ]; then grep -q 'rc=0' "$LOADER" 2>/dev/null || fail "loader check log $LOADER has no rc=0"; echo "loader check OK ($LOADER)"; fi
echo "== offers (total job cost = \$/h x hours; show this table to the user) =="
[ -n "$GPU" ] && vastai search offers "$GPU rentable=true verified=true" -o 'dph+' --raw 2>/dev/null | python3 -c "
import json,sys
for x in json.load(sys.stdin)[:6]: print('vast', x['id'], x['gpu_name'], x['num_gpus'], round(x['dph_total'],2), x.get('geolocation'), 'cpu_ghz', x.get('cpu_ghz'), 'down', int(x.get('inet_down',0)), 'rel', round(x.get('reliability2',0),3))"
[ -x tools/vast/runpod_stock.sh ] && bash tools/vast/runpod_stock.sh 8 "${GPU##*in }" 2>/dev/null | head -5
command -v brev >/dev/null && brev search gpu --gpu-name "$(echo "$GPU" | grep -oE 'H200|H100|B200|5090|6000' | head -1)" 2>/dev/null | head -8
echo "== three clouds listed above; pick on TOTAL cost, then rent within 30 min =="
printf 'ts=%s\npurpose=%s\ntarget=%s\nend=%s\n' "$(date +%s)" "$PURPOSE" "$TARGET" "$END" > $S/rent_ok; echo "GATE OK -> $S/rent_ok (30 min)"
