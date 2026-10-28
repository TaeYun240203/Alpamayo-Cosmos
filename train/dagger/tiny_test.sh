#!/usr/bin/env bash
# CPU end-to-end check of the DAgger runner (README "검증"): synthetic step-0 root (tests/ft_synth.py) as R1 and R2, the
# tiny delivery of scripts/student_dryrun.sh as the base (quantizers off on CPU), a fake spool (dagger/fake.py):
#   1. one process, empty spool: waits for rows; fake episodes arrive -> first update -> publishes -> stops at max_steps
#   2. the same command again: resumes (step, version, replay position, rows)
#   3. torchrun 2 processes (gloo), llm+expert LoRA with LLM checkpointing: same LoRA on both ranks at every save
#   4. checks: versions, events, metrics, the published LoRA merges into the delivery (rl.policy.merge)
# Usage (cwd train/): bash dagger/tiny_test.sh <tiny deliver dir> <work dir> <eval repo src/e2e_driver/server> [python]
set -euo pipefail
D=$1; O=$2; S=$3; PY=${4:-python}
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH} OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
mkdir -p $O; [ -f $O/root/_index/index.parquet ] || $PY tests/ft_synth.py $O/root > /dev/null
F=$O/root/_ft; P=$O/pools; mkdir -p $P                         # dagger/pools.py stand-ins on the synthetic root
cp $F/old_curve.parquet $P/signal_approach.parquet; cp $F/old_curve.parquet $P/signal_hold.parquet; cp $F/old_curve.parquet $P/highway.parquet; cp $F/r2_A.parquet $P/r2_A_dyn.parquet; cp $F/r2_B.parquet $P/r2_B_dyn.parquet; cp $F/start_relabel.parquet $P/r2_unseen.parquet; cp $F/start_relabel.parquet $P/r1_replay_exclude.parquet; cp $F/start_relabel.parquet $P/future_green.parquet
C=(--base $D/final --data_root $O/root --r2_root $O/root --e2e_server $S --device cpu --dtype float32 --workers 2
   --override +tiny=student_ft --override paths.ft_pools=$F --override paths.start_relabel=$F/start_relabel.parquet --override paths.dagger_pools=$P
   --rank 4 --alpha 8 --min_rows 12 --val_n 4 --val_dagger 4 --scan_s 1 --save_every 2 --publish_at_end)

rm -rf $O/spool $O/run1 $O/run2
( until grep -qs waiting_rows $O/run1/metrics/events.jsonl; do sleep 1; done; $PY -m dagger.fake $O/spool --e2e_server $S --actors 2 --episodes 1 --decisions 40 ) &
$PY -m dagger.runner "${C[@]}" --spool $O/spool --out $O/run1 --lora expert --lr 2e-3 --batch 4 --publish_min 0.3 --max_steps 6
wait
$PY -m dagger.fake $O/spool --e2e_server $S --actors 2 --episodes 1 --decisions 40 --start 1 --seed 1
$PY -m dagger.runner "${C[@]}" --spool $O/spool --out $O/run1 --lora expert --lr 2e-3 --batch 4 --publish_min 0.3 --max_steps 3
$PY -m torch.distributed.run --nproc_per_node 2 -m dagger.runner "${C[@]}" --spool $O/spool --out $O/run2 --lora llm+expert --grad_ckpt \
  --lr 2e-3 --batch 4 --publish_min 100 --max_steps 4

$PY - $O $D <<'EOF'
import json, sys
from pathlib import Path
from rl.policy import merge
o, d = Path(sys.argv[1]), Path(sys.argv[2])
for run, steps, world in (("run1", 9, 1), ("run2", 4, 2)):
    m = o / run / "metrics"
    rows = [json.loads(x) for x in open(m / "steps.jsonl")]
    ev = [json.loads(x)["event"] for x in open(m / "events.jsonl")]
    latest = json.load(open(o / run / "lora/latest.json")); s = json.load(open(m / "summary.json"))
    assert [r["step"] for r in rows] == list(range(1, steps + 1)), [r["step"] for r in rows]
    assert all(k in rows[-1] for k in ("replay/fm", "replay/traj_ce", "dagger/fm", "dagger/traj_ce", "samples_per_s", "backlog", "consumed_rows")), rows[-1]
    assert any("replay/signal_approach/fm" in r or "replay/signal_hold/fm" in r for r in rows) and any("replay/r1_general/fm" in r for r in rows), rows[-1]
    assert sum(r["tag_signal_rows"] + r["tag_lane_rows"] for r in rows) > 0 and any("dagger_lane/fm" in r for r in rows)
    val = [json.loads(x) for x in open(m / "val.jsonl")]
    hist = [json.loads(x) for x in open(m / "history.jsonl")]
    assert [h["version"] for h in hist] == sorted({0, *[h["version"] for h in hist]}) and all("zero_move" in x and "swap_dlat_3s" in x for h in hist for x in h["sets"].values()), hist
    print(run, "history", [(h["version"], h["seconds"], {k: (x["n"], x["zero_move"], x["swap_move"]) for k, x in h["sets"].items()}) for h in hist])
    assert all(any(k.startswith(f"val_{v}/") for k in val[-1]) for v in ("signal_approach", "signal_hold", "lane")), val[-1]
    assert ev[:3] == (["start", "waiting_rows", "first_update"] if run == "run1" else ["start", "first_update", "publish"]), ev
    assert latest["version"] >= 1 and "publish" in ev and s["world"] == world, (latest, ev, s)
    assert rows[-1]["uses_mean"] >= 1 and rows[-1]["uses_max"] >= 1 and "rows_used_2plus" in rows[-1], rows[-1]
    if run == "run1":
        assert "resume" in ev and rows[6]["replay_pos"] == rows[5]["replay_pos"] + 2 and rows[-1]["consumed_rows"] > rows[5]["consumed_rows"]
        assert all(r["uses_max"] == 1 for r in rows if r["reused_rows"] == 0), [(r["step"], r["uses_max"]) for r in rows]   # resume counts once
    n = merge(d, o / run / "lora" / latest["path"], o / run / "merged")
    print(run, "steps", len(rows), "version", latest["version"], "merged layers", n, "events", ev,
          "last", {k: rows[-1][k] for k in ("replay/fm", "dagger/fm", "dagger/traj_ce", "backlog", "consumed_rows", "samples_per_s")})
print("PASS")
EOF
