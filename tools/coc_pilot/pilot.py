"""Run CoC pilot calls: autolabeler prompt @4dda809 verbatim, 17 native front frames, strict JSON schema."""
import base64, importlib.util, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from codex_call import call
H = Path(__file__).parent
spec = importlib.util.spec_from_file_location("coc", "/private/tmp/claude-501/-Users-taeyunkim-Library-Mobile-Documents-com-apple-CloudDocs-Workspaces-Alpamayo-Cosmos/e25ca47a-dbfb-433d-bd51-fa7077bd7dff/scratchpad/coc/src/coc_labeling/prompts/vlm/coc.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
SCHEMA = {"type": "json_schema", "name": "OutputSchema", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["ego_behavior_schema"],
    "properties": {"ego_behavior_schema": {"type": "object", "additionalProperties": False, "required": ["effect_on_ego_behavior"],
                                           "properties": {"effect_on_ego_behavior": {"type": "string"}}}}}}
K = json.load(open(H / "keyframes.json"))


def content(j, detail="high", variant="A"):
    k = K[j]
    imgs = [{"type": "input_image", "detail": detail, "image_url": "data:image/jpeg;base64," + base64.b64encode(f.read_bytes()).decode()}
            for f in sorted((H / f"frames/{j:02d}").glob("*.jpg"))]
    assert len(imgs) == 17
    T = lambda s: {"type": "input_text", "text": s}
    if variant == "B":  # fixed output prompt moved into instructions so the prefix is cacheable
        return [T(P.images_prompt), *imgs, T(P.meta_action_prompt), T(k["meta"]), T(P.ego_state_prompt), T(k["ego"]), T(P.question_prompt)]
    return [T(P.images_prompt), *imgs, T(P.meta_action_prompt), T(k["meta"]), T(P.ego_state_prompt), T(k["ego"]), T(P.output_prompts)]


def run(j, model, effort, schema=True, detail="high", tag="", variant="A"):
    ins = P.system + "\n" + P.output_prompts if variant == "B" else P.system
    r = call(model, ins, content(j, detail, variant), effort=effort, text_format=SCHEMA if schema else None,
             sid="coc-pilot-B" if variant == "B" else None)
    r.pop("events", None); r.get("rate", {}).pop("x-codex-turn-state", None)
    rec = {"j": j, "ep": K[j]["ep"], "tau0": K[j]["tau0"], "section": K[j]["section"], "change": K[j]["path_change"],
           "model": model, "effort": effort, "schema": schema, "detail": detail, "tag": tag, "variant": variant, "t": time.time(), **r}
    try: rec["parsed"] = json.loads(r["text"]); rec["json_ok"] = isinstance(rec["parsed"]["ego_behavior_schema"]["effect_on_ego_behavior"], str)
    except Exception: rec["json_ok"] = False
    with open(H / "results.jsonl", "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    u = r.get("usage") or {}
    print(j, rec["status"], r.get("latency_s"), "in", u.get("input_tokens"), "cached", (u.get("input_tokens_details") or {}).get("cached_tokens"),
          "out", u.get("output_tokens"), "reason", (u.get("output_tokens_details") or {}).get("reasoning_tokens"),
          "used%", r.get("rate", {}).get("x-codex-primary-used-percent"), "json", rec["json_ok"], "|", (r.get("text") or r.get("error", ""))[:300], flush=True)
    return rec


if __name__ == "__main__":
    model, effort, js, workers, tag = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")], int(sys.argv[4]), sys.argv[5]
    variant = sys.argv[6] if len(sys.argv) > 6 else "A"
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda j: run(j, model, effort, tag=tag, variant=variant), js))
