"""Row pools of the R2 fine-tune (configs/student_ft.yaml, DATA_PLAN_R2 7), as `keep` parquets for MoraiDataset. Mac, CPU.
  old   (from the old index + the step-0 _ego tables; every split, the dataset picks its split)
        the "기존 복구(R1·R2)·곡선 프레임" source (10 %) as two halves, because curve rows outnumber recovery rows ~6:1
        (train 2 Hz, index_v3: recovery 14.9k, curve 89.2k) and a draw over their union would leave recovery ~15 % of it:
        old_recovery.parquet  episode, i: |d| >= 0.5 m off the global path and the GT is 0.3 m closer 3 s later, v >= 2 m/s,
                    not C5 (highway lane changes). The step-0 tables carry no noise flag (/collect_meta/noise_active was not
                    extracted), so this takes the returns after R1 steering noise and R2 spawn offsets and after passing a
                    static obstacle (most rows are C6, C3, C2), i.e. any return to the path.
        old_curve.parquet     episode, i: GT heading change over the next 3 s >= 15 deg at v >= 3 m/s, or path_s in the
                    curves the dense drive left the lane on (DATA_PLAN_R2 4 D: s 83-116, 193, 439, 594, 682 +-15 m) at
                    v >= 2 m/s. Also the old half of the curve validation set.
        old_static.parquet    episode, i: cell C2 rows (static-obstacle section; static validation set)
  r2    (from episode-kinds.tsv of the R2 collection: name, kind ra/rb/rd, dynamic flag; QA files optional)
        r2_A / r2_B / r2_D / r2_AB.parquet   episode: the R2 episodes of each bundle that passed QA (qa.jsonl "ok"),
        without the pilot batches (r2p*) unless --with-pilot
  drop-props (user 10/28 "ㅇㅇ"; MORAI 10/6: no static obstacle moves in the final) r2_A / r2_B / r2_AB without the episodes
        whose moving objects (kph > 0 in step0_r2/_meta/r2_episodes.parquet) are all props (asset ids 4xxxxxxx: car-shaped
        props, unmanned bicycles, scooters, strollers on a waypoint path); any person, animal or ridden bicycle (5xxxxxxx)
        keeps the episode
Usage (cwd train/): python scripts/ft_pools.py old --index ~/coc_runs/index_v3/index.parquet --ego ~/.cache/step0_ego --out assets/ft
                    python scripts/ft_pools.py r2 --kinds ~/.claude-morai-mac/episode-kinds.tsv [--qa <dir>/qa.jsonl ...] --out assets/ft
                    python scripts/ft_pools.py drop-props --meta <step0_r2>/_meta/r2_episodes.parquet --out assets/ft
Each run writes <out>/pools.json (row and episode counts)."""
import argparse, json
from pathlib import Path

import numpy as np, pandas as pd

RECOVERY_D, RECOVERY_BACK, RECOVERY_V = 0.5, 0.3, 2.0
CURVE_DEG, CURVE_V, CURVE_S, CURVE_HALF = 15.0, 3.0, ((83.0, 116.0), (193.0, 193.0), (439.0, 439.0), (594.0, 594.0), (682.0, 682.0)), 15.0
AHEAD = 30                                                       # rows = 3 s


def old_pools(index, ego):
    idx = pd.read_parquet(index, columns=["episode", "i", "cell", "split"]).astype({"episode": str, "cell": str})
    out = []
    for ep, g in idx.groupby("episode", sort=False):
        z = np.load(Path(ego) / f"{ep}.npz"); n = len(z["tau"]); i = g.i.to_numpy(); j = np.minimum(i + AHEAD, n - 1)
        d, v, yaw, s = z["d"].astype(float), z["v"].astype(float), z["yaw"].astype(float), z["path_s"].astype(float)
        rec = (np.abs(d[i]) >= RECOVERY_D) & (np.abs(d[j]) <= np.abs(d[i]) - RECOVERY_BACK) & (v[i] >= RECOVERY_V) & (g.cell.to_numpy() != "C5")
        near = np.zeros(len(i), bool)
        for lo, hi in CURVE_S: near |= (s[i] >= lo - CURVE_HALF) & (s[i] <= hi + CURVE_HALF)
        cur = ((np.abs(yaw[j] - yaw[i]) >= np.radians(CURVE_DEG)) & (v[i] >= CURVE_V)) | (near & (v[i] >= RECOVERY_V))
        out.append(g.assign(recovery=rec, curve=cur))
    return pd.concat(out)


def r2_lists(kinds, qa, with_pilot):
    k = pd.read_csv(kinds, sep="\t", header=None, names=["episode", "kind", "dynamic"], dtype=str)
    if not with_pilot: k = k[~k.episode.str.startswith("r2p")]
    if qa:
        res = pd.concat([pd.read_json(f, lines=True)[["name", "ok"]] for f in qa]).drop_duplicates("name", keep="last")
        k = k.merge(res.rename(columns={"name": "episode"}), on="episode", how="left")
        k = k[k.ok.fillna(False).astype(bool)]
    return {b: k[k.kind.isin(ks)][["episode"]].drop_duplicates() for b, ks in (("A", ["ra"]), ("B", ["rb"]), ("D", ["rd"]), ("AB", ["ra", "rb"]))}


def moving_props_only(meta):
    """episodes whose moving objects are all props (ids 4xxxxxxx); episodes without movers are not affected."""
    out = set()
    for r in meta.itertuples():
        mv = [int(i) for i, k in zip(r.actor_ids, r.actor_kph) if k > 0]
        if mv and all(i // 10_000_000 == 4 for i in mv): out.add(str(r.episode))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["old", "r2", "drop-props"]); ap.add_argument("--out", required=True)
    ap.add_argument("--index"); ap.add_argument("--ego"); ap.add_argument("--kinds"); ap.add_argument("--qa", nargs="*", default=[])
    ap.add_argument("--with-pilot", action="store_true"); ap.add_argument("--meta")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f = out / "pools.json"; summary = json.loads(f.read_text()) if f.exists() else {}
    if a.what == "old":
        df = old_pools(a.index, a.ego)
        sel = {"old_recovery": df[df.recovery][["episode", "i"]], "old_curve": df[df.curve][["episode", "i"]],
               "old_static": df[df.cell == "C2"][["episode", "i"]]}
        for name, d in sel.items():
            d.reset_index(drop=True).to_parquet(out / f"{name}.parquet", index=False)
            m = df.merge(d[["episode", "i"]], on=["episode", "i"])
            summary[name] = {"rows": len(d), "train_rows_2hz": int(((m.split == "train") & (m.i % 5 == 0)).sum()), "episodes": int(d.episode.nunique())}
        tr = df[(df.split == "train") & (df.i % 5 == 0)]
        summary["old_recovery"]["train_2hz_by_cell"] = tr[tr.recovery].cell.value_counts().to_dict()
        summary["old_curve"]["train_2hz_by_cell"] = tr[tr.curve].cell.value_counts().to_dict()
        summary["old_source"] = {"index": str(a.index), "ego": str(a.ego)}
    elif a.what == "drop-props":
        drop = moving_props_only(pd.read_parquet(a.meta, columns=["episode", "actor_ids", "actor_kph"]))
        for b in ("A", "B", "AB"):
            d = pd.read_parquet(out / f"r2_{b}.parquet"); keep = d[~d.episode.astype(str).isin(drop)]
            keep.reset_index(drop=True).to_parquet(out / f"r2_{b}.parquet", index=False)
            summary[f"r2_{b}"] = {"episodes": len(keep), "dropped_moving_props": len(d) - len(keep)}
        summary["r2_drop_props"] = {"meta": str(a.meta), "episodes": sorted(drop)}
    else:
        for b, d in r2_lists(a.kinds, a.qa, a.with_pilot).items():
            d.to_parquet(out / f"r2_{b}.parquet", index=False); summary[f"r2_{b}"] = {"episodes": len(d)}
        summary["r2_source"] = {"kinds": str(a.kinds), "qa": a.qa, "with_pilot": a.with_pilot}
    f.write_text(json.dumps(summary, indent=1)); print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
