"""Quota calibration: fresh 17-frame windows every 3 s (no repeated images), variant B, stop after 2 used% ticks or N calls."""
import json, sys, threading, numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rosbags.highlevel import AnyReader
import prep, pilot
H = Path(__file__).parent; S = H.parent
N, OFF, BASE, STOP_AT = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
pilot_t = {(k["ep"], k["tau0"]) for k in pilot.K}
K = []
for ep in prep.EPS:
    G, _ = prep.episode(ep)
    ts, data = [], []
    with AnyReader([S / f"data/{ep}_0.bag"]) as r:
        c = [c for c in r.connections if c.topic == "/image/front/compressed_10hz"]
        for con, t, raw in r.messages(connections=c):
            ts.append(t); data.append(bytes(r.deserialize(raw, con.msgtype).data))
    ts = np.asarray(ts)
    for t0 in range(23 + OFF, len(G) - 61, 30):
        if any(e == ep and abs(G.tau[t0] - t) < 1.0 for e, t in pilot_t): continue
        idx = [t0 + 5 * k for k in range(-4, 13)]
        if G.bag_ns[idx[-1]] > ts[-1]: break
        j = BASE + len(K); d = H / f"frames/{j:03d}"; d.mkdir(parents=True, exist_ok=True)
        for i, b in enumerate(G.bag_ns[idx]): (d / f"{i:02d}.jpg").write_bytes(data[int(np.argmin(np.abs(ts - b)))])
        rel = [0.5 * k for k in range(-4, 13)]
        K.append({"ep": ep, "tau0": round(float(G.tau[t0]), 2), "section": int(G.sec[t0]), "path_change": "", "dir": f"{j:03d}",
                  "meta": "\nMeta-actions by sampled timestamp (relative to event, unit: seconds):\n" + "\n".join(
                      f"(t={r_:+.1f}s) longitudinal: {G.lon[i]}, lateral: {G.lat[i]}, lane: {G.lane[i]}" for r_, i in zip(rel, idx)),
                  "ego": "Ego speed time series (relative to event time t=0.0s):\n- Longitudinal (m/s): [" + ", ".join(f"(t={r_:.1f}s, v_long={G.v[i]:.3f} m/s)" for r_, i in zip(rel, idx)) +
                         "]\n- Lateral (m/s): [" + ", ".join(f"(t={r_:.1f}s, v_lat={G.v_lat[i]:.3f} m/s)" for r_, i in zip(rel, idx)) + "]\nPositive lateral speed means rightward motion, negative means leftward motion."})
print(len(K), "calibration windows", flush=True)
# frames dir differs from pilot's 2-digit layout: point pilot at these windows
pilot.K = K
_orig = pilot.content
def content(j, detail="high", variant="A"):
    k = K[j]; frames = sorted((H / f"frames/{k['dir']}").glob("*.jpg"))
    import base64
    T = lambda s: {"type": "input_text", "text": s}
    imgs = [{"type": "input_image", "detail": detail, "image_url": "data:image/jpeg;base64," + base64.b64encode(f.read_bytes()).decode()} for f in frames]
    assert len(imgs) == 17
    return [T(pilot.P.images_prompt), *imgs, T(pilot.P.meta_action_prompt), T(k["meta"]), T(pilot.P.ego_state_prompt), T(k["ego"]), T(pilot.P.question_prompt)]
pilot.content = content
ticks, stop, lock = [], threading.Event(), threading.Lock()
def one(j):
    if stop.is_set(): return
    rec = pilot.run(j, "gpt-6.1-sol", "medium", tag="calib", variant="B")
    u = int(rec.get("rate", {}).get("x-codex-primary-used-percent", -1))
    with lock:
        ticks.append((rec["t"], u))
        if u >= STOP_AT: stop.set()
with ThreadPoolExecutor(8) as ex: list(ex.map(one, range(min(N, len(K)))))
ticks.sort(); print("calls", len(ticks), "used% sequence", [u for _, u in ticks])
