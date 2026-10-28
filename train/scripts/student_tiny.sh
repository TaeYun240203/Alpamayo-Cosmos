#!/usr/bin/env bash
# Local student check (README, student local validation) on 20 fixed MORAI samples with tiny random models of the real
# architecture: tiny T* (the teacher's tiny_overfit, CoC off) -> student stage 1 -> soft targets -> placeholder rule_ok
# -> stage 2 -> gate (frozen-expert ADE, start vs stage 1) and which parts moved in each stage.
# Usage (cwd train/): bash scripts/student_tiny.sh <1.5 release dir (config only)> <step-0 share> <out dir>
set -euo pipefail
R=$1; D=$2; O=$3; mkdir -p "$O"
TR="python -m morai_vla.train"; TINY="+tiny=student paths.teacher=$O/teacher/final data.root=$D"
[ -d "$O/teacher/final" ] || $TR --config-name tiny_overfit paths.alpamayo15="$R" data.root="$D" data.train_dataset.coc=false \
    trainer.max_steps=1000 trainer.save_steps=1000 trainer.eval_steps=1000 paths.output_dir="$O/teacher"
[ -d "$O/s1/final" ] || $TR --config-name student1 $TINY paths.output_dir="$O/s1"
python scripts/soft_targets.py --config-name student1 $TINY +out="$O/soft" +batch=4
python - "$O" <<'PY'
import glob, sys, zlib, numpy as np, pandas as pd     # placeholder for the rule check (tests only): 70 % of rows pass, by hash
keys = np.concatenate([np.load(f)["key"] for f in sorted(glob.glob(f"{sys.argv[1]}/soft/part-*.npz"))])
pd.DataFrame({"episode": [k.rsplit("/", 1)[0] for k in keys], "i": [int(k.rsplit("/", 1)[1]) for k in keys],
              "rule_ok": [zlib.crc32(k.encode()) % 10 < 7 for k in keys]}).to_parquet(f"{sys.argv[1]}/rule_ok.parquet")
PY
[ -d "$O/s2/final" ] || $TR --config-name student2 $TINY model.init="$O/s1/final" data.train_dataset.rule_ok="$O/rule_ok.parquet" paths.output_dir="$O/s2"
G="python scripts/gate_student.py --config-name student1 $TINY +dataset=train_dataset"
$G +student=start +out="$O/gate_start"; $G +student="$O/s1/final" +out="$O/gate_s1"; $G +student="$O/s2/final" +out="$O/gate_s2"
python scripts/parts_moved.py "$O/s1" > "$O/parts_s1.json"; python scripts/parts_moved.py "$O/s2" > "$O/parts_s2.json"
cat "$O"/parts_s*.json
