"""Inside the Noetic chroot (python 3.8, numpy 1.17): moving objects near the relabeled label's path, per pre-departure scan.
The ego stands still from row i to the departure row, so the static scene is the same in every scan of the window; objects in
a scan near the path that the departure-time scans do not have (and the reverse) are traffic that differs from what the label
saw. In: /tmp/lidar_req.json (as lidar_corridor.py). Out: /tmp/lidar_dyn.<part>.jsonl {ep, tau, new, new_s, new_lat, gone}."""
import json, sys, zlib
import numpy as np

RES, XF, YW = 0.25, 60.0, 20.0
H_LO, H_HI = 22, 70                      # as lidar_corridor.py
ZONE, BODY_X, MIN_CELLS, WIN = 8.0, 4.2, 4, 2.0
ROOT = "/mnt/2026_MORAI_step0/"
part, parts = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)


def occ(fh, li, j):
    fh.seek(int(li["offset"][j])); h = np.frombuffer(zlib.decompress(fh.read(int(li["length"][j]))), np.uint8).reshape(2, 320, 160)[0]
    return (h >= H_LO) & (h <= H_HI)


def dilate(m):
    p = np.pad(m, 1); out = np.zeros_like(m)
    for a in range(3):
        for b in range(3): out |= p[a:a + m.shape[0], b:b + m.shape[1]]
    return out


def first_cluster(s):
    si = np.sort(s); cnt = np.searchsorted(si, si + WIN, side="right") - np.arange(len(si))
    hit = np.flatnonzero(cnt >= MIN_CELLS)
    return float(si[hit[0]]) if len(hit) else -1.0


rr, cc = np.mgrid[0:320, 0:160]; X, Y = XF - (rr + 0.5) * RES, YW - (cc + 0.5) * RES
out = open("/tmp/lidar_dyn.%d.jsonl" % part, "w")
for r in json.load(open("/tmp/lidar_req.json"))[part::parts]:
    try:
        li = np.load(ROOT + r["ep"] + "/lidar_index.npz")
    except IOError:
        continue
    P = np.vstack([[0.0, 0.0], np.asarray(r["path"], float)])
    arc = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]; s_q = np.arange(0, arc[-1] + 1e-6, 0.5)
    Q = np.stack([np.interp(s_q, arc, P[:, 0]), np.interp(s_q, arc, P[:, 1])], 1)
    near = np.full((320, 160), np.inf); s_near = np.zeros((320, 160))
    for k in range(len(Q)):                                   # distance of every cell to the path (sampled every 0.5 m)
        dd = np.hypot(X - Q[k, 0], Y - Q[k, 1]); m = dd < near; near[m] = dd[m]; s_near[m] = s_q[k]
    zone = (near <= ZONE) & (X > BODY_X)
    k = np.flatnonzero((li["tau"] >= r["t_lo"]) & (li["tau"] <= r["t_hi"]))
    if len(k) < 3: continue
    with open(ROOT + r["ep"] + "/lidar.bin", "rb") as fh:
        ref = occ(fh, li, k[-1]) | occ(fh, li, k[-2])          # the two scans at departure
        ref_d = dilate(ref)
        for j in k[:-2]:
            o = occ(fh, li, j)
            new = o & ~ref_d & zone; gone = ref & ~dilate(o) & zone
            res = {"ep": r["ep"], "tau": round(float(li["tau"][j]), 3), "new": int(new.sum()), "gone": int(gone.sum()),
                   "new_s": round(first_cluster(s_near[new]), 2) if new.any() else -1,
                   "new_lat": round(float(near[new].min()), 2) if new.any() else -1}
            out.write(json.dumps(res) + "\n")
    out.flush()
out.close()
print("done", part)
