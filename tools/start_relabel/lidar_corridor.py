"""Inside the Noetic chroot (python 3.8, numpy 1.17): LiDAR objects in the relabeled label's corridor, per pre-departure scan.
In: /tmp/lidar_req.json [{ep, t_lo, t_hi, path: 64 x 2 future xy of the departure row}]. Out: /tmp/lidar_out.jsonl, one line per
scan {ep, tau, s_obs (base_link path length of the first object, -1 none), cells, lat_min}. BEV per step-0 README / lidar_bev.py."""
import json, sys, zlib
import numpy as np

RES, XF, YW = 0.25, 60.0, 20.0
H_LO, H_HI = 22, 70   # height codes 0.05 m from -1 m: ground returns are 8-14 (-0.6..-0.3 m), so >= ~0.45 m above it, <= ~3 m
HALF_W, BODY_X, MIN_CELLS, WIN = 1.0, 4.2, 3, 2.0
ROOT = "/mnt/2026_MORAI_step0/"
part, parts = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)

req = json.load(open("/tmp/lidar_req.json"))[part::parts]
out = open("/tmp/lidar_out.%d.jsonl" % part, "w")
for r in req:
    try:
        li = np.load(ROOT + r["ep"] + "/lidar_index.npz")
    except IOError:
        out.write(json.dumps({"ep": r["ep"], "missing": 1}) + "\n"); continue
    P = np.vstack([[0.0, 0.0], np.asarray(r["path"], float)])
    seg = np.hypot(*np.diff(P, axis=0).T); arc = np.r_[0, np.cumsum(seg)]
    s_q = np.arange(0, arc[-1] + 1e-6, 0.25)
    Q = np.stack([np.interp(s_q, arc, P[:, 0]), np.interp(s_q, arc, P[:, 1])], 1)
    k = np.flatnonzero((li["tau"] >= r["t_lo"]) & (li["tau"] <= r["t_hi"]))
    with open(ROOT + r["ep"] + "/lidar.bin", "rb") as fh:
        for j in k:
            fh.seek(int(li["offset"][j])); h = np.frombuffer(zlib.decompress(fh.read(int(li["length"][j]))), np.uint8).reshape(2, 320, 160)[0]
            rr, cc = np.nonzero((h >= H_LO) & (h <= H_HI))
            x, y = XF - (rr + 0.5) * RES, YW - (cc + 0.5) * RES
            m = x > BODY_X; x, y = x[m], y[m]
            res = {"ep": r["ep"], "tau": round(float(li["tau"][j]), 3), "s_obs": -1, "cells": 0, "lat_min": -1}
            if len(x):
                dd = np.hypot(x[:, None] - Q[None, :, 0], y[:, None] - Q[None, :, 1])
                q = dd.argmin(1); lat = dd[np.arange(len(x)), q]; s = s_q[q]
                inside = (lat <= HALF_W) & (s >= BODY_X)
                res["cells"] = int(inside.sum())
                if inside.any():
                    res["lat_min"] = round(float(lat[inside].min()), 2)
                    si = np.sort(s[inside])
                    cnt = np.searchsorted(si, si + WIN, side="right") - np.arange(len(si))
                    hit = np.flatnonzero(cnt >= MIN_CELLS)
                    if len(hit): res["s_obs"] = round(float(si[hit[0]]), 2)
            out.write(json.dumps(res) + "\n")
    out.flush()
out.close()
print("done", part, len(req))
