"""Official CoC validation stages 1-3 (autolabeler @4dda809 prompts, verbatim) with a Codex judge.
Frames: stage 1 = t -2..0 s (history), stage 2 = t 0..+6 s, stage 3 = all 17 (2 Hz), as the official segment windows."""
import base64, json, sys, types, importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from codex_call import call

H = Path(__file__).parent
stub = types.ModuleType("common")
stub.NormalizedRow = lambda **k: types.SimpleNamespace(**k)
stub.StageSpec = lambda **k: types.SimpleNamespace(**k)
stub.run_stage = lambda *a, **k: 0
sys.modules["common"] = stub
ST = {}
for n, f in ((1, "eval_stage1_coc_evidence_vllm.py"), (2, "eval_stage2_coc_behavior_vllm.py"), (3, "eval_stage3_coc_causal_vllm.py")):
    spec = importlib.util.spec_from_file_location(f"st{n}", H / "val" / f); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); ST[n] = m
FR = {1: range(0, 5), 2: range(4, 17), 3: range(0, 17)}
SCHEMA = {n: {"type": "json_schema", "name": f"stage{n}", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": list(ST[n].EVAL_KEYS),
    "properties": {k: {"type": "string"} if k == "explanation" else {"type": "number"} for k in ST[n].EVAL_KEYS}}} for n in ST}


def judge(item, stage, model):
    row = stub.NormalizedRow(clip_id=item["ep"], event_timestamp=item["tau0"], label_class_identifier=None, coc_label=item["label"])
    text = ST[stage].build_prompt(row)
    fs = sorted((H / "frames" / item["dir"]).glob("*.jpg"))
    parts = [{"type": "input_text", "text": text}] + [
        {"type": "input_image", "detail": "high", "image_url": "data:image/jpeg;base64," + base64.b64encode(fs[i].read_bytes()).decode()} for i in FR[stage]]
    r = call(model, "", parts, effort="medium", text_format=SCHEMA[stage])
    try: sc = json.loads(r["text"])
    except Exception: sc = {"error": r.get("error") or r.get("text")}
    rec = {**{k: item[k] for k in ("set", "labeler", "key", "ep", "tau0", "label")}, "stage": stage, "judge": model, **sc,
           "used": r.get("rate", {}).get("x-codex-primary-used-percent"), "in": (r.get("usage") or {}).get("input_tokens")}
    with open(H / "val_results.jsonl", "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


if __name__ == "__main__":
    items = json.load(open(H / "val_items.json"))
    model, workers = sys.argv[1], int(sys.argv[2])
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else len(items)
    start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    jobs = [(it, s) for it in items[start:lim] for s in (1, 2, 3)]
    with ThreadPoolExecutor(workers) as ex:
        for rec in ex.map(lambda a: judge(a[0], a[1], model), jobs):
            print(rec["set"], rec["labeler"], rec["key"], "st%d" % rec["stage"], "used%", rec["used"],
                  {k: v for k, v in rec.items() if k.endswith("score") or k.startswith("decision")}, rec.get("error", "")[:200] if "error" in rec else "", flush=True)
