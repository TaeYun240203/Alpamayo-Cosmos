"""Pull the 17 front JPEGs (native, no re-encode) nearest each keyframe's bag times."""
import json, os, sys, numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
S = Path(sys.argv[1]); K = json.load(open(S / "coc/keyframes.json"))
for ep in sorted({k["ep"] for k in K}):
    ts, data = [], []
    with AnyReader([S / f"data/{ep}_0.bag"]) as r:
        c = [c for c in r.connections if c.topic == "/image/front/compressed_10hz"]
        for con, t, raw in r.messages(connections=c):
            ts.append(t); data.append(bytes(r.deserialize(raw, con.msgtype).data))
    ts = np.asarray(ts)
    for j, k in enumerate(K):
        if k["ep"] != ep: continue
        d = S / f"coc/frames/{j:02d}"; d.mkdir(parents=True, exist_ok=True)
        gap = []
        for i, b in enumerate(k["bag_ns"]):
            n = int(np.argmin(np.abs(ts - b))); gap.append(abs(ts[n] - b) / 1e6)
            (d / f"{i:02d}.jpg").write_bytes(data[n])
        k["frame_gap_ms_max"] = round(max(gap), 1)
        print(j, ep, "max gap ms", k["frame_gap_ms_max"])
json.dump(K, open(S / "coc/keyframes.json", "w"), indent=1)
