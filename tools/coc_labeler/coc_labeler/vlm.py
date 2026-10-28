"""Codex (ChatGPT OAuth) Responses client: our own instructions (no CLI system prompt), strict JSON, disk cache,
retry with backoff, and a weekly-quota budget guard. The token is read from ~/.codex/auth.json and never logged."""
import base64, hashlib, json, os, threading, time, urllib.error, urllib.request, uuid
from pathlib import Path

URL = "https://chatgpt.com/backend-api/codex/responses"
CACHE = Path(os.path.expanduser("~/.cache/coc_labeler/vlm.jsonl"))
_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


class Codex:
    def __init__(self, model="gpt-6.1-sol", effort="medium", budget_pct=None, max_used_pct=None):
        self.model, self.effort = model, effort
        self.budget_pct = budget_pct          # stop once the weekly window passes start + budget
        self.max_used_pct = max_used_pct      # or an absolute ceiling shared by parallel shards
        self.start_used = None; self.used = None; self.calls = 0
        self.cache = {}
        if CACHE.exists():
            for line in open(CACHE):
                try: r = json.loads(line); self.cache[r["key"]] = r["out"]
                except ValueError: pass

    def _headers(self):
        a = json.load(open(os.path.expanduser("~/.codex/auth.json")))["tokens"]   # re-read: the CLI may refresh it
        return {"Authorization": "Bearer " + a["access_token"], "chatgpt-account-id": a["account_id"],
                "OpenAI-Beta": "responses=experimental", "originator": "codex_cli_rs", "session_id": str(uuid.uuid4()),
                "Content-Type": "application/json", "Accept": "text/event-stream"}

    def ask(self, instructions, text, images, schema, name="out"):
        """images: list of (label, jpeg bytes). Returns parsed JSON (dict)."""
        parts = []
        for label, jpg in images:
            parts += [{"type": "input_text", "text": label},
                      {"type": "input_image", "detail": "high", "image_url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}]
        parts.append({"type": "input_text", "text": text})
        key = hashlib.sha256(json.dumps([self.model, self.effort, instructions, text, [hashlib.sha1(j).hexdigest() for _, j in images], schema]).encode()).hexdigest()
        if key in self.cache: return self.cache[key]
        if self.max_used_pct is not None and self.used is not None and self.used >= self.max_used_pct:
            raise BudgetExceeded(f"weekly usage {self.used}% >= ceiling {self.max_used_pct}%")
        if self.budget_pct is not None and self.used is not None and self.start_used is not None and self.used >= self.start_used + self.budget_pct:
            raise BudgetExceeded(f"weekly usage {self.used}% >= start {self.start_used}% + budget {self.budget_pct}%")
        body = {"model": self.model, "instructions": instructions, "input": [{"type": "message", "role": "user", "content": parts}],
                "tools": [], "tool_choice": "auto", "parallel_tool_calls": False, "reasoning": {"effort": self.effort, "summary": "auto"},
                "store": False, "stream": True, "include": ["reasoning.encrypted_content"],
                "text": {"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}}}
        for attempt in range(6):
            try:
                txt, usage, used = self._post(body)
                out = json.loads(txt)
                break
            except urllib.error.HTTPError as e:
                if e.code in (401, 403): raise
                time.sleep(min(60, 5 * 2 ** attempt))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ConnectionError):
                time.sleep(min(60, 5 * 2 ** attempt))
        else:
            raise RuntimeError("Codex request failed 6 times")
        with _lock:
            self.calls += 1
            if used is not None:
                self.used = used; self.start_used = used if self.start_used is None else self.start_used
            self.cache[key] = out
            with open(CACHE, "a") as f: f.write(json.dumps({"key": key, "out": out, "usage": usage, "used": used, "t": time.time()}) + "\n")
        return out

    def _post(self, body):
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=self._headers(), method="POST")
        r = urllib.request.urlopen(req, timeout=600)
        used = r.headers.get("x-codex-primary-used-percent")
        txt, usage = "", None
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            try: ev = json.loads(line[5:])
            except ValueError: continue
            if ev.get("type") == "response.output_text.delta": txt += ev.get("delta", "")
            elif ev.get("type") in ("response.completed", "response.incomplete"): usage = ev["response"].get("usage")
            elif ev.get("type") in ("response.failed", "error"): raise ConnectionError(json.dumps(ev)[:500])
        return txt, usage, (int(used) if used and used.isdigit() else None)
