"""R2 labels: recompute traj_ok without the GT-stream gaps (10/27 20:25, main session: R2 index kept 29 % of the recording).

The scorer leaves a frame without pose when Ego status is > 0.1 s away (max_gt_gap_s), and every such frame breaks the
6.4 s target window of the 64 frames before it. R2 Ego status arrives at ~40 Hz but in bursts (UDP bridge drops under
Mac load): 3 % of frames had no pose, 91 % of the gaps <= 0.5 s, yet they masked ~30 % of all rows. Training targets come
from the _ego grid, which interpolates the pose (and now the speed) across gaps, so a gap of <= GAP_FILL frames between
two posed frames no longer breaks a window. Everything else breaks it as before: the scorer's breakers (event_type),
the noise flag (latest /collect_meta/noise_active message at or before the frame, from noise_dump.py JSON), longer gaps
and gaps at the start or end. Output keeps every column; traj_ok is replaced and the scorer's is kept as traj_ok_scorer.
Asserted: every scorer-ok row stays ok on the future window and every added row had a gap in it (the bag noise flag is
a frame off the live scorer's at some window edges: 3 rows it would drop stay ok, 42 it would add stay masked). Then rows
with a gap > 0.5 s in their 1.5 s history (the model's ego history comes from _ego positions) are dropped too.
Usage: r2_relabel_gaps.py OK_LIST NOISE_DIR OUT_DIR
"""
import json, os, sys
from pathlib import Path

import numpy as np, pyarrow as pa, pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coc_labeler"))
from coc_labeler import data   # noqa: E402

BREAKERS = {"collision", "signal_violation", "speed", "lane", "drive_inoperable", "signal_unverified"}   # scorer.TARGET_BREAKERS
GAP_FILL, HORIZON, HIST = 5, 64, 30   # frames: gap <= 0.5 s of unposed frames; scorer traj_points; 1.5 s history in wall frames down to rho 0.5 (its clip)


def relabel(ep, noise_dir):
    f = data.labels_file(ep); t = pq.read_table(f); L = t.to_pandas()
    m = len(L); ns = L.bag_time_ns.to_numpy(np.int64)
    posed = np.isfinite(L[["x", "y", "yaw"]].to_numpy(float)).all(1)
    gap = ~posed
    edges = np.diff(np.r_[0, gap.astype(int), 0]); s, e = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    long_gap = np.zeros(m, bool)
    for a, b in zip(s, e):
        if b - a > GAP_FILL or a == 0 or b == m: long_gap[a:b] = True
    brk = np.array([bool(BREAKERS & set(x.split(";"))) for x in L.event_type.fillna("")])
    noise = np.zeros(m, bool); nf = Path(noise_dir) / f"{ep}.json"
    if nf.exists():
        msgs = np.array(json.load(open(nf)), np.int64).reshape(-1, 2)
        if len(msgs):
            k = np.searchsorted(msgs[:, 0], ns, side="right") - 1
            noise = np.where(k >= 0, msgs[np.maximum(k, 0), 1] > 0, False)
    bad = brk | noise | long_gap
    cum = np.r_[0, np.cumsum(bad)]; i = np.arange(m); end = np.minimum(i + HORIZON, m - 1)
    ok = (i + HORIZON < m) & (cum[end + 1] == cum[i])
    old = (L.traj_ok_scorer if "traj_ok_scorer" in L else L.traj_ok).to_numpy(bool)   # rerun: start from the scorer's
    lost = int((old & ~ok).sum())                       # must be 0: only the gap reason was relaxed
    gcum = np.r_[0, np.cumsum(gap)]; gapped = gcum[end + 1] > gcum[i]
    unexplained = int((ok & ~old & ~gapped).sum())     # gained without a gap in the window: noise edges (live vs bag stamps)
    fut = (ok & (old | gapped)) | old                  # relax only rows the scorer masked with a gap in their window
    assert not (old & ~fut).any() and gapped[fut & ~old].all(), ep   # scorer rows kept; every added row had a gap
    lcum = np.r_[0, np.cumsum(long_gap)]; start = np.maximum(i - HIST, 0)
    hist_ok = lcum[i + 1] == lcum[start]               # main session 20:40: no gap > 0.5 s in the 1.5 s history either
    ok = fut & hist_ok
    t = t.set_column(t.schema.get_field_index("traj_ok"), "traj_ok", pa.array(ok))
    if "traj_ok_scorer" not in t.column_names:
        t = t.append_column("traj_ok_scorer", pa.array(old)).replace_schema_metadata(t.schema.metadata)
    return t, dict(ep=ep, rows=m, old=int(old.sum()), new=int(ok.sum()), lost=lost, unexplained=unexplained,
                   hist_dropped=int((fut & ~hist_ok).sum()),
                   gap_frames=int(gap.sum()), long_gap_frames=int(long_gap.sum()), noise_frames=int(noise.sum()))


if __name__ == "__main__":
    oklist, noise_dir, out = sys.argv[1:4]; Path(out).mkdir(parents=True, exist_ok=True)
    tot = {}
    for ep in open(oklist).read().split():
        t, st = relabel(ep, noise_dir)
        pq.write_table(t, Path(out) / f"{ep}.labels.parquet")
        for k, v in st.items():
            if k != "ep": tot[k] = tot.get(k, 0) + v
        if st["lost"] or st["unexplained"]: print("CHECK", json.dumps(st))
    print(json.dumps(tot))
