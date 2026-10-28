"""QA: (1) decision-vs-ego-GT checks for every label, (2) an HTML review sheet for the ~10% human audit (AR1 Sec. 4.3),
(3) optional official 3-stage VLM validation (autolabeler scripts/coc_validation_stage prompts) on the QA sample."""
import base64, html, io, json
from pathlib import Path

import numpy as np

from . import data, frames
from .events import LAT, LON


def gt_check(rec, ep):
    """True/False per channel: does the labeled decision match what the ego does right after the keyframe?"""
    G = ep.G; i = rec["t0"]; n = len(G); v = G.v.to_numpy(); dec = rec["decision"]; out = {}
    win = lambda a, b: slice(max(0, i + a), min(n, i + b))
    lon = dec["longitudinal"]
    if lon == LON["stop"]: out["lon"] = bool((v[win(0, 100)] < 0.3).any())
    elif lon in (LON["yield_"], LON["lead"], LON["gap"]): out["lon"] = bool(v[win(0, 50)].min() < v[i] - 1.0 or (v[win(0, 60)] < 0.3).any() or (lon == LON["gap"]))
    elif lon == LON["adapt"]: out["lon"] = bool(v[win(0, 50)].min() < v[i] - 0.5) or "maintain a safe speed" in (rec["coc"] or "")
    elif lon == LON["set_speed"]: out["lon"] = not bool((v[win(5, 30)] < 0.3).all())
    lat, d = dec["lateral"], dec.get("direction")
    if lat in (LAT["lc"], LAT["merge"]):
        cx = data.crossings(np.c_[G.cx, G.cy][win(-20, 60)]); out["lat"] = any(sd == d for _, sd in cx)
    elif lat in (LAT["out_nudge"], LAT["in_nudge"]):
        dd = G.d.to_numpy()[win(0, 60)]; m = dd.max() if d == "left" else -dd.min()
        out["lat"] = bool(m > (1.0 if lat == LAT["out_nudge"] else 0.3))
    elif lat == LAT["turn"]:                     # peak heading excursion: roundabout entries turn right, then left
        dy = G.yaw.to_numpy()[win(0, 80)] - G.yaw[i]; out["lat"] = bool(dy.max() > 0.5) if d == "left" else bool(dy.min() < -0.5)
    elif lat == LAT["keep"]:                     # lines crossed inside junctions / plazas / tapers are geometry, not a maneuver
        sl = win(0, 40); s = G.path_s.to_numpy()[sl]
        cx = [j for j, _ in data.crossings(np.c_[G.cx, G.cy][sl])
              if not data.feature_at(s[j], ("intersection", "roundabout", "toll_plaza", "construction", "merge"))]
        out["lat"] = not cx
    return out


def montage_b64(ep, key, cols=(0, 2, 4, 6, 8, 10, 12, 16)):
    from PIL import Image, ImageDraw
    fs = sorted(frames.key_dir(ep, key).glob("*.jpg"))
    if len(fs) < 17: return None
    M = Image.new("RGB", (320 * 4, 180 * 2))
    for n, i in enumerate(cols):
        im = Image.open(fs[i]).resize((320, 180)); ImageDraw.Draw(im).text((6, 6), f"t={0.5 * (i - 4):+.1f}s", fill=(255, 255, 0))
        M.paste(im, ((n % 4) * 320, (n // 4) * 180))
    b = io.BytesIO(); M.save(b, "JPEG", quality=75)
    return base64.b64encode(b.getvalue()).decode()


def review_sheet(recs, path):
    rows = []
    for r in recs:
        if not r["qa"]: continue
        img = montage_b64(r["episode"], r["key"])
        comps = "; ".join(f"{c.get('category')}: {c.get('appearance') or c.get('type')} {c.get('state') or ''}".strip() for c in r["critical_components"])
        rows.append(f"<section><h3>{html.escape(r['episode'])} {r['key']} · {r['cell']} · {r['env']} · {html.escape(r['kind'])}</h3>"
                    + (f"<img src='data:image/jpeg;base64,{img}'>" if img else "<p>(frames missing)</p>")
                    + f"<p class=coc>{html.escape(r['coc'] or '[' + r['status'] + ']')}</p>"
                    f"<p>decision: {html.escape(str(r['decision']))}<br>components: {html.escape(comps)}<br>gt: {html.escape(str(r.get('gt')))}</p>"
                    "<p>verdict: <label><input type=radio name=" + r['key'] + r['episode'] + "> ok</label> <label><input type=radio name="
                    + r['key'] + r['episode'] + "> decision wrong</label> <label><input type=radio name=" + r['key'] + r['episode']
                    + "> cause wrong / not visible</label></p></section>")
    Path(path).write_text("<!doctype html><meta charset=utf-8><title>CoC review</title><style>body{font:14px sans-serif;max-width:1300px;margin:auto;padding:16px}"
                          "img{width:100%}.coc{font-size:17px;font-weight:600}section{border-bottom:1px solid #ccc;padding:8px 0}</style>"
                          f"<h1>CoC review sample ({len(rows)})</h1>" + "".join(rows))
