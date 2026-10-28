"""label_check on the collection expert's own driving (R1/R2 labels cache, Mac): the 6.4 s it really drove after each
sampled decision, on the restored sim clock (coc_labeler.data.load_episode), goes in as the label. Expert recoveries
should come out pass / pass_unavoidable; every fail is a checker bug, an expert mistake or a real violation.

    uv run --python 3.12 --with-requirements tools/scorer/requirements.txt python tools/scorer/label_check_expert.py \
        OUT_DIR [--stride 5] [--jobs 6] [--only r2]

Inputs per decision: GT pose and speed (t0), the GT light state for the stop line ahead (labels `signal`), the GPS
blackout flag, and the scene manifest's standing obstacles placed at their (s, lat) with class-typical footprints
(Object Info is not in the bags: walking pedestrians and NPC cars are missing, so collisions with them are not judged).
Skipped: futures with a GT jump, and R1 steering-noise stretches (traj_ok off without a scorer event).
Out: OUT_DIR/decisions.parquet (one row per decision; events = scorer events in the window).
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "coc_labeler"))
from coc_labeler import data  # noqa: E402
from label_check import N, RULES, LabelChecker  # noqa: E402

SMALL_M, CAR_M = (1.0, 1.0), (4.6, 1.9)        # scenario-variants.py SMALL props 0.6-1.35 m (barrier 1 x 3), VEHICLE 4.6-6.6
EVENT_RULE = {"lane": "lane", "signal_violation": "signal", "speed": "speed", "collision": "collision",
              "drive_inoperable": "inoperable", "checkpoint_skip": "checkpoint"}
CHK = None


def standing_objects(scene):
    out = []
    for k, a in enumerate((scene or {}).get("actors", [])):
        if a.get("zone") in ("static", "tunnel_static") and "s" in a:
            x, y, h = data.route_point(a["s"], a.get("lat", 0.0))
            size = CAR_M if a.get("id") in data.VEHICLE_PROPS else SMALL_M
            out.append((1000 + k, "obstacle", x, y, h, 0.0, 0.0, *size))
    return out


def one(args):
    ep, stride = args
    try:
        E = data.load_episode(ep)
    except Exception as e:  # noqa: BLE001
        return [dict(episode=ep, verdict="load_error", reasons=f"{type(e).__name__}: {e}")]
    G = E.G
    lab = pd.read_parquet(E.labels_path, columns=["t", "traj_ok", "event_type"])
    summ = json.loads(pq.read_schema(E.labels_path).metadata[b"scorer_summary"])
    events = [e for e in summ["events"] if e["type"] in EVENT_RULE]
    objs = standing_objects(E.scene)
    x, y, yaw, v, tw = (G[c].to_numpy(float) for c in ("x", "y", "yaw", "v", "t"))
    # the restored sim clock jitters (90th pct 4 m/s between step length and speed); a GT jump is metres
    jump = np.abs(np.hypot(np.diff(x), np.diff(y)) - 0.05 * (v[1:] + v[:-1])) > 3.0
    cj = np.r_[0, np.cumsum(jump)]
    lt, lok = lab.t.to_numpy(float), lab.traj_ok.to_numpy(bool)
    lev = lab.event_type.astype(str).to_numpy() != ""
    cev = np.r_[0, np.cumsum(lev)]
    r1 = not ep.startswith("r2")
    rows = []
    for i in range(0, len(G) - N - 1, stride):
        if cj[i + N] - cj[i]:
            continue
        a, b = np.searchsorted(lt, tw[i]), np.searchsorted(lt, tw[i + N])
        if r1 and a < len(lok) and not lok[a] and cev[min(b, len(cev) - 1)] == cev[a]:
            continue   # R1 steering noise (or a GT gap) in the window, no judged event to explain it
        f = slice(i + 1, i + N + 1)
        ego = dict(x=x[i], y=y[i], yaw=yaw[i], v=v[i], s=G.path_s[i])
        near = [o for o in objs if np.hypot(o[2] - x[i], o[3] - y[i]) < 120]
        r = CHK.check(ego, np.c_[x[f], y[f], yaw[f], v[f]], signal=str(G.signal[i]), objects=near,
                      blackout=bool(G.gps_blackout[i]))
        ev = sorted({EVENT_RULE[e["type"]] for e in events if e["t_start"] <= tw[i + N] and e["t_end"] >= tw[i]})
        tc = r["label"]["t_cross"]   # the light the expert really met at the line (future info the label may not use)
        at_cross = str(G.signal[i + min(int(round(tc / 0.1)), N)]) if tc is not None else ""
        rows.append(dict(episode=ep, i=i, tau=round(G.tau[i], 1), t=round(tw[i], 2), s=round(G.path_s[i], 1),
                         d=round(G.d[i], 2), v=round(v[i], 2), signal=str(G.signal[i]), signal_at_cross=at_cross,
                         t_cross=tc, paint_t0=r["label"]["paint_t0"], off_road_t0=r["label"]["off_road_t0"],
                         d_t0=r["label"]["d_t0"], lane_penalty_s=r["label"]["lane_penalty_s"], verdict=r["verdict"],
                         reasons=";".join(r["reasons"]), tags=";".join(r["tags"]), events=";".join(ev),
                         better=json.dumps(r["better"]), obstacles=len(near),
                         **{f"L_{k}": r["label"][k] for k in RULES}, **{f"O_{k}": r["oracle"][k] for k in RULES}))
    return rows


def init():
    global CHK
    os.nice(10)
    CHK = LabelChecker()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--stride", type=int, default=5, help="decisions every stride x 0.1 s")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--only", help="episode name prefix")
    a = ap.parse_args()
    eps = sorted({f.name.split(".")[0] for f in data.LABELS.glob("*/*.labels.parquet")})
    eps = [e for e in eps if data.scene_of(e)[0] and (not a.only or e.startswith(a.only))]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    rows = []
    with Pool(a.jobs, initializer=init) as pool:
        for k, r in enumerate(pool.imap_unordered(one, [(e, a.stride) for e in eps], chunksize=2)):
            rows += r
            if k % 100 == 0:
                print(k, len(eps), len(rows), flush=True)
    df = pd.DataFrame(rows)
    df["kind"] = [str(data.scene_of(e)[0])[:2] for e in df.episode]
    df.to_parquet(Path(a.out) / "decisions.parquet")
    print(df.groupby("kind").verdict.value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()
