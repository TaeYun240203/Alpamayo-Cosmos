"""R2 step 0, Mac side: one metadata row per QA-ok R2 episode (DATA_PLAN_R2.md) -> <out>/r2_episodes.parquet.

Sources: scene log + manifests (kind, start pose/speed, actors with type and timing class), QA lines (r2-qa.py: noise
seconds, collisions, AEB/yield counts, cameras, RTF), labels parquets (collision rows), r2.log (GT AEB brake and yield
on/off lines, unix s), _ego tables and _index/index.parquet (tau grid, trainable rows).
Intervals are [start, end] on the sim clock tau of the _ego grid (the index's time base), merged across gaps < 1 s.
Usage: python r2_meta.py OUT --qa qa_all.jsonl --ok ok.txt --ego EGO_DIR --index INDEX_PARQUET --r2logs DIR
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coc_labeler"))
from coc_labeler import data   # noqa: E402

KIND = {"ra": "A", "rb": "B", "rd": "D"}


def runs(ts, gap=1.0):
    """sorted times -> [[start, end], ...] split where consecutive times are more than gap apart."""
    out = []
    for t in ts:
        if out and t - out[-1][1] <= gap: out[-1][1] = t
        else: out.append([t, t])
    return [[round(a, 1), round(b, 1)] for a, b in out]


def r2_events(path):
    """r2.log -> (brake unix times, yield (on, off) unix pairs)."""
    brake, yields, on = [], [], None
    for line in open(path, errors="replace") if os.path.exists(path) else []:
        p = line.split()
        if len(p) < 3 or p[1] not in ("brake", "yield"): continue
        t = float(p[0])
        if p[1] == "brake": brake.append(t)
        elif p[2] == "on": on = t if on is None else on
        elif p[2] == "off" and on is not None: yields.append((on, t)); on = None
    if on is not None: yields.append((on, on))
    return brake, yields


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out")
    for k in ("qa", "ok", "ego", "index", "r2logs"): ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    qa = {r["name"]: r for r in map(json.loads, open(a.qa))}
    idx = pd.read_parquet(a.index, columns=["episode", "i"])
    rows10 = idx.groupby("episode", observed=True).size(); rows2 = idx[idx.i % 5 == 0].groupby("episode", observed=True).size()
    out = []
    for ep in sorted(open(a.ok).read().split()):
        q, (sc_id, sc) = qa[ep], data.scene_of(ep)
        sc = sc or {}; st = sc.get("start", {}); acts = sc.get("actors", [])
        z = np.load(Path(a.ego) / f"{ep}.npz"); bag_s, tau = z["bag_ns"] / 1e9, z["tau"]
        to_tau = lambda ts: [float(np.interp(t, bag_s, tau)) for t in ts]
        L = pd.read_parquet(data.labels_file(ep), columns=["bag_time_ns", "event_type"])
        coll = L.bag_time_ns[L.event_type.fillna("").str.contains("collision")].to_numpy() / 1e9
        brake, yields = r2_events(Path(a.r2logs) / f"{ep}.r2.log")
        dyn = [x for x in acts if x.get("timing")]
        out.append(dict(
            episode=ep, worker=ep.split("-")[1], kind=KIND.get(sc.get("kind"), "?"), scene=sc_id,
            tunnel=sc.get("tunnel", ""), start_checkpoint=st.get("checkpoint"), start_lat_m=st.get("lat"),
            start_dyaw_deg=st.get("dyaw"), start_kph=st.get("kph"), start_cls=st.get("cls", ""),   # D: offset class
            end_s=sc.get("end_s"), n_actors=len(acts),
            actor_ids=[int(x["id"]) for x in acts], actor_cls=[x.get("cls", x.get("zone", "")) for x in acts],
            actor_ptype=[x.get("ptype", "") for x in acts], actor_timing=[x.get("timing", "") for x in acts],
            actor_kph=[float(x.get("kph", 0.0)) for x in acts],
            timing=dyn[0]["timing"] if dyn else "",                    # A/B: the scene's timing class (first mover)
            objects_moved_max_m=max(q.get("moved", {}).values(), default=0.0),
            noise=bool(q.get("noise_s", 0) > 0), noise_s=q.get("noise_s", 0.0),
            collisions=q.get("collisions", 0), collision_tau=runs(to_tau(coll)),
            aeb=q.get("aeb", 0), aeb_fast=q.get("aeb_fast", 0), aeb_fast_alone=q.get("aeb_fast_alone", 0),
            aeb_tau=runs(to_tau(brake)), yields=q.get("yield_on", 0),
            yield_tau=[[round(a_, 1), round(b_, 1)] for a_, b_ in zip(to_tau([y[0] for y in yields]), to_tau([y[1] for y in yields]))],
            drove_m=q.get("drove_m"), rtf=q.get("rtf"), sim_s=round(float(tau[-1]), 1), front_cam_hz=q.get("cam_hz", [None])[0],
            first_move_s=q.get("first_move_s"), bag_gb=q.get("gb"), label_frames=q.get("frames"),
            index_rows_10hz=int(rows10.get(ep, 0)), index_rows_2hz=int(rows2.get(ep, 0))))
    df = pd.DataFrame(out); Path(a.out).mkdir(parents=True, exist_ok=True)
    df.to_parquet(Path(a.out) / "r2_episodes.parquet", index=False)
    print(json.dumps(dict(episodes=len(df), by_kind=df.kind.value_counts().to_dict(),
                          timing_A=df[df.kind == "A"].timing.value_counts().to_dict(),
                          start_kph=df.start_kph.value_counts().to_dict(), noise=int(df.noise.sum()),
                          collision_eps=int((df.collision_tau.str.len() > 0).sum()), aeb_eps=int((df.aeb > 0).sum()),
                          rows_10hz=int(df.index_rows_10hz.sum()), rows_2hz=int(df.index_rows_2hz.sum()),
                          sim_h=round(df.sim_s.sum() / 3600, 2)), indent=1))


if __name__ == "__main__":
    main()
