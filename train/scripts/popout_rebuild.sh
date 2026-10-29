#!/usr/bin/env bash
# Near pop-out synthetic labels from scratch (main 10/29: Mac or the training box while the sensors download), one entry point:
# popout_relabel.py (default --onset lidar, no restart rows) over every episode in -j chunks, then popout_mix.py merge into OUT
# with ego/, popout_index, cruise and val lists; OUT/DONE = md5 of summary.json. Skips when OUT/DONE exists. Deterministic:
# sorted episodes, merge sorts its rows (any -j gives the same files).
# Inputs (no cams.bin / lidar.bin): S0/_ego/<ep>.npz, S0/_popout/<ep>/{episode.json,labels.parquet,objects.json.gz},
#   S0/_popout_index/*/index.parquet, S0/<ep>/{cams_index.npz,lidar_index.npz}; python with numpy, pandas, torch (merge).
# Usage (any cwd): bash train/scripts/popout_rebuild.sh S0 OUT [-j N] [extra popout_relabel.py args]
set -euo pipefail
S0=$(cd "$1" && pwd); OUT=$2; shift 2
J=$(${PY:-python} -c 'import os; print(os.cpu_count())')
if [ "${1:-}" = "-j" ]; then J=$2; shift 2; fi
[ -f "$OUT/DONE" ] && { echo "popout_rebuild: $OUT/DONE exists, skipping"; exit 0; }
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd); cd "$(dirname "$0")/.."       # train/
PY=${PY:-python}; T0=$(date +%s)
( cd "$S0" && for f in _ego/*.npz; do e=$(basename "$f" .npz); [ -f "$e/cams_index.npz" ] && echo "$e"; done ) | sort > "$OUT/_episodes.txt"
mkdir -p "$OUT/_idx"
$PY -c 'import glob, sys, pandas as pd; fs = sorted(glob.glob(sys.argv[1] + "/_popout_index/*/index.parquet"))
pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).to_parquet(sys.argv[2] + "/index.parquet", index=False)' "$S0" "$OUT/_idx"
N=$(wc -l < "$OUT/_episodes.txt"); [ "$J" -gt "$N" ] && J=$N
echo "popout_rebuild: $N episodes, $J jobs"
awk -v J="$J" -v D="$OUT/_chunks" 'BEGIN { system("mkdir -p " D) } { print > (D "/" (NR % J)) }' "$OUT/_episodes.txt"
pids=()
for f in "$OUT"/_chunks/*; do
  k=$(basename "$f"); $PY scripts/popout_relabel.py "$S0" "$OUT/_idx" "$OUT/_runs/c$k" $(cat "$f") "$@" > "$OUT/_runs.c$k.log" 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || { echo "popout_rebuild: a relabel chunk failed (logs $OUT/_runs.c*.log)"; exit 1; }; done
$PY scripts/popout_mix.py merge "$S0" "$OUT" "$OUT/_runs" > "$OUT/merge.log" 2>&1 || { tail -20 "$OUT/merge.log"; exit 1; }
$PY - "$OUT" <<'PY'
import shutil, sys
from pathlib import Path
o = Path(sys.argv[1])
for p in ["_runs", "_chunks", "_idx", "_episodes.txt"] + [q.name for q in o.glob("_runs.c*.log")]:   # intermediates (box disk)
    shutil.rmtree(o / p) if (o / p).is_dir() else (o / p).unlink()
PY
$PY -c 'import hashlib, sys; print(hashlib.md5(open(sys.argv[1] + "/summary.json", "rb").read()).hexdigest())' "$OUT" > "$OUT/DONE"
echo "popout_rebuild: done in $(( $(date +%s) - T0 )) s, DONE $(cat "$OUT/DONE")"
