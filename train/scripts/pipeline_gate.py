"""Automatic halt checks of scripts/pipeline.sh (thresholds are fixed in README "단계 C 파이프라인" before the run).
Exit 0 = pass, 1 = halt (reason on stdout).
  train <run dir> [--no-divergence]   non-finite value in loss_components.jsonl; final eval_* > 1.2 x the run's best
  max <json> <key> <limit>            halt unless json[key] < limit (dotted key into nested dicts)
  merge <out dir> <shard dirs...>     check_release shards -> <out dir>/release_check.json + summary.json
  select <stage dir> name=path ...    t1t: eval_stage1 shards <stage>/select/<name>.shard*.json -> lowest discrete-token ADE;
                                      its weights -> <stage>/final, all candidates -> <stage>/best.json (ade_minus_cv for the gate)
"""
import json, math, sys
from pathlib import Path

DIVERGENCE = 1.2


def train_check(run, divergence=True):
    rows = [json.loads(l) for l in open(Path(run) / "loss_components.jsonl")]
    bad = [(r["step"], k) for r in rows for k, v in r.items() if isinstance(v, float) and not math.isfinite(v)]
    if bad: return f"non-finite {bad[0][1]} at step {bad[0][0]}"
    if (Path(run) / "NONFINITE").exists(): return (Path(run) / "NONFINITE").read_text().strip()
    if divergence:
        for k in sorted({k for r in rows for k in r if k.startswith("eval_")} - {"eval_peak_mem_gb"}):
            ser = [r[k] for r in rows if k in r]
            if ser and min(ser) > 1e-3 and ser[-1] > DIVERGENCE * min(ser):          # ignore ~0 series (e.g. eval_lm 1e-5)
                return f"{k} diverged: final {ser[-1]:.4f} > {DIVERGENCE} x best {min(ser):.4f}"
    return None


def max_check(path, key, limit):
    v = json.load(open(path))
    for k in key.split("."): v = v[k]
    return None if float(v) < float(limit) else f"{key} = {float(v):.4f} not below {limit}"


def merge(out, shards):
    from scripts.check_release import summarize
    rows = [r for d in shards for r in json.load(open(Path(d) / "release_check.json"))]
    copy = all(json.load(open(Path(d) / "summary.json"))["lidar_vit_is_camera_copy"] for d in shards)
    Path(out).mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(Path(out) / "release_check.json", "w"), indent=1)
    json.dump(summarize(rows, copy), open(Path(out) / "summary.json", "w"), indent=1)


def select(stage, cands):
    """cands {name: weights dir}. Copies the chosen weights (config + safetensors) to <stage>/final."""
    import glob, shutil
    from morai_vla.stage1_eval import summarize
    res = {}
    for name in cands:
        rows = [r for f in sorted(glob.glob(f"{stage}/select/{name}.shard*.json")) for r in json.load(open(f))["rows"]]
        res[name] = {k: v for k, v in summarize(rows).items() if k != "by_cell"} | {"weights": cands[name]}
    assert len({r["n"] for r in res.values()}) == 1, "candidates must be scored on the same samples"
    best = min(res, key=lambda k: res[k]["ade"])
    out = Path(stage) / "final"
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in glob.glob(f"{cands[best]}/config.json") + glob.glob(f"{cands[best]}/model*.safetensors*"):
        shutil.copy(f, out)
    json.dump({"chosen": best, "metric": "discrete-token ADE (6.4 s, 1,000 fixed val+val_asset samples)", "candidates": res,
               "ade_minus_cv": res[best]["ade"] - res[best]["cv_ade"]}, open(Path(stage) / "best.json", "w"), indent=1)
    return best


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "merge": merge(a[0], a[1:]); sys.exit(0)
    if cmd == "select": print(select(a[0], dict(x.split("=", 1) for x in a[1:]))); sys.exit(0)
    why = train_check(a[0], "--no-divergence" not in a) if cmd == "train" else max_check(*a)
    print(why or "pass"); sys.exit(1 if why else 0)
