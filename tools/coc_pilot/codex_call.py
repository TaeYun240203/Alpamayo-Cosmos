"""Direct ChatGPT-OAuth call to the Codex Responses backend (no Codex CLI system prompt). Never prints the token."""
import json, os, time, uuid, urllib.request, urllib.error

URL = "https://chatgpt.com/backend-api/codex/responses"
_A = json.load(open(os.path.expanduser("~/.codex/auth.json")))["tokens"]


def _headers(sid):
    return {"Authorization": "Bearer " + _A["access_token"], "chatgpt-account-id": _A["account_id"],
            "OpenAI-Beta": "responses=experimental", "originator": "codex_cli_rs", "session_id": sid,
            "Content-Type": "application/json", "Accept": "text/event-stream"}


def usage_snapshot():
    """Plan usage as the CLI's /status sees it (wham/usage); None if the endpoint is unavailable."""
    req = urllib.request.Request("https://chatgpt.com/backend-api/wham/usage", headers=_headers(str(uuid.uuid4())))
    try:
        return json.load(urllib.request.urlopen(req, timeout=30))
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": e.read()[:300].decode("utf-8", "replace")}


def call(model, instructions, content, effort="low", text_format=None, sid=None, timeout=600):
    """content: list of Responses input parts. Returns dict(text, usage, rate headers, status, latency)."""
    sid = sid or str(uuid.uuid4())
    body = {"model": model, "instructions": instructions,
            "input": [{"type": "message", "role": "user", "content": content}],
            "tools": [], "tool_choice": "auto", "parallel_tool_calls": False,
            "reasoning": {"effort": effort, "summary": "auto"}, "store": False, "stream": True,
            "include": ["reasoning.encrypted_content"], "prompt_cache_key": sid}
    if text_format: body["text"] = {"format": text_format}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=_headers(sid), method="POST")
    t0 = time.time(); out = {"status": None, "text": "", "usage": None, "events": {}}
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        out.update(status=e.code, error=e.read()[:2000].decode("utf-8", "replace"),
                   rate={k: v for k, v in e.headers.items() if k.lower().startswith("x-codex")})
        return out
    out["status"] = r.status
    out["rate"] = {k: v for k, v in r.headers.items() if k.lower().startswith("x-codex")}
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"): continue
        try: ev = json.loads(line[5:])
        except ValueError: continue
        typ = ev.get("type", "?"); out["events"][typ] = out["events"].get(typ, 0) + 1
        if typ == "response.output_text.delta": out["text"] += ev.get("delta", "")
        elif typ in ("response.completed", "response.incomplete"):
            out["usage"] = ev["response"].get("usage"); out["final_status"] = ev["response"].get("status")
            out["incomplete"] = ev["response"].get("incomplete_details")
        elif typ in ("response.failed", "error"):
            out["error"] = json.dumps(ev)[:2000]
        elif "rate_limit" in typ:
            out["rate_event"] = ev
    out["latency_s"] = round(time.time() - t0, 1)
    return out
