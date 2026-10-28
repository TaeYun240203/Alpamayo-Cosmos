"""Official CoC validation stages 1-3 (alpamayo-coc-autolabeler scripts/coc_validation_stage @4dda809, prompts verbatim)
on the QA sample, judged by a Codex model. Usage: python -m coc_labeler.official_val OUT [--judge gpt-6-astra] [--max 120]
The prompt modules are fetched from GitHub once into ~/.cache/coc_labeler/val/."""
import argparse, json, subprocess, sys, types, importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import frames
from .vlm import Codex

VAL = Path("~/.cache/coc_labeler/val").expanduser()
STAGES = {1: "eval_stage1_coc_evidence_vllm.py", 2: "eval_stage2_coc_behavior_vllm.py", 3: "eval_stage3_coc_causal_vllm.py"}
FR = {1: range(0, 5), 2: range(4, 17), 3: range(0, 17)}


def modules():
    VAL.mkdir(parents=True, exist_ok=True)
    stub = types.ModuleType("common")
    stub.NormalizedRow = lambda **k: types.SimpleNamespace(**k); stub.StageSpec = lambda **k: types.SimpleNamespace(**k); stub.run_stage = lambda *a, **k: 0
    sys.modules["common"] = stub; mods = {}
    for n, f in STAGES.items():
        p = VAL / f
        if not p.exists():
            body = subprocess.run(["gh", "api", f"repos/NVlabs/alpamayo-coc-autolabeler/contents/scripts/coc_validation_stage/{f}?ref=4dda809", "--jq", ".content"],
                                  check=True, capture_output=True, text=True).stdout
            import base64; p.write_bytes(base64.b64decode(body))
        spec = importlib.util.spec_from_file_location(f"st{n}", p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); mods[n] = m
    return mods


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--judge", default="gpt-6-astra"); ap.add_argument("--max", type=int, default=120)
    a = ap.parse_args(); out = Path(a.out); M = modules(); V = Codex(model=a.judge)
    recs = [json.loads(l) for f in sorted((out / "labels").glob("*.jsonl")) for l in open(f) if l.strip()]
    recs = [r for r in recs if r["qa"] and r["status"] == "ok" and len(list(frames.key_dir(r["episode"], r["key"]).glob("*.jpg"))) == 17]
    import random; random.Random(0).shuffle(recs); recs = recs[:a.max]          # random, not the first episodes

    def judge(job):
        r, n = job; m = M[n]
        row = types.SimpleNamespace(clip_id=r["episode"], event_timestamp=r["tau0"], label_class_identifier=None, coc_label=r["coc"])
        schema = {"type": "object", "additionalProperties": False, "required": list(m.EVAL_KEYS),
                  "properties": {k: {"type": "string"} if k == "explanation" else {"type": "number"} for k in m.EVAL_KEYS}}
        fs = sorted(frames.key_dir(r["episode"], r["key"]).glob("*.jpg"))
        res = V.ask("", m.build_prompt(row), [(f"frame {i}", fs[i].read_bytes()) for i in FR[n]], schema, f"stage{n}")
        return dict(episode=r["episode"], key=r["key"], kind=r["kind"], cell=r["cell"], coc=r["coc"], stage=n, **res)

    with ThreadPoolExecutor(6) as ex:
        rows = list(ex.map(judge, [(r, n) for r in recs for n in (1, 2, 3)]))
    with open(out / "official_val.jsonl", "w") as f:
        for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    import pandas as pd
    D = pd.DataFrame(rows); W = D.pivot_table(index=["episode", "key", "cell"], columns="stage", aggfunc="first")
    T = pd.DataFrame({"s1": (W[("visual_evidence_score", 1)] >= 0.65) & (W[("scene_factor_hallucination_score", 1)] < 0.5),
                      "s2": (W[("behavior_match_score", 2)] >= 0.65) & (W[("behavior_hallucination_score", 2)] < 0.5),
                      "s3": (W[("causal_consistency_score", 3)] >= 0.55) & (W[("causal_hallucination_score", 3)] < 0.5)})
    T["all"] = T.s1 & T.s2 & T.s3
    s = {"n": len(T), "pass": T.mean().round(3).to_dict(), "by_cell": T.groupby(level=2).mean().round(2).to_dict("index"), "judge": a.judge}
    json.dump(s, open(out / "official_val_summary.json", "w"), indent=1); print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
