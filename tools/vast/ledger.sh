#!/bin/bash
# Cost ledger (CLAUDE.md §3). add on rent, close on destroy. reports/cost_ledger.md
# bash tools/vast/ledger.sh add <id> <$/h> "<purpose>" "<target>" "<end condition>"
# bash tools/vast/ledger.sh close <id> <$ spent> "<waste $ and reason, or ->"
L="$(dirname "$0")/../../reports/cost_ledger.md"; [ -f "$L" ] || printf '# Cost ledger (every paid box)\n\n| id | rented | $/h | purpose | target | end condition | destroyed | $ spent | waste ($, reason) |\n|---|---|---|---|---|---|---|---|---|\n' > "$L"
case $1 in
  add) printf '| %s | %s | %s | %s | %s | %s | | | |\n' "$2" "$(date +%m/%d\ %H:%M)" "$3" "$4" "$5" "$6" >> "$L";;
  close) python3 - "$L" "$2" "$(date +%m/%d\ %H:%M)" "$3" "$4" <<'PY'
import sys; p,i,t,s,w=sys.argv[1:]; L=open(p).read().split('\n')
for k,l in enumerate(L):
    if l.startswith(f'| {i} |') and l.endswith('| | | |'): L[k]=l[:-len('| | | |')]+f'| {t} | {s} | {w} |'
open(p,'w').write('\n'.join(L))
PY
  ;;
esac; tail -1 "$L"
