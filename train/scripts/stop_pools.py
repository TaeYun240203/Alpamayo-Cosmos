"""Row pools of the stop-reinforcement fine-tune (user 10/28 09:10, doc "와우마요 정지 보강 학습 계획"). CPU, no model.
R2 A (pools r2_A: moving-prop-only episodes already out; R2 index = the net glitch rule), per AEB run / yield interval of
_meta/r2_episodes.parquet: the stop = first row at or after its start with v < 0.5 m/s (within 8 s); rows from 6.4 s
before the stop to the stop -> r2_aeb, r2_yield (train) and r2_dynstop_val (val + val_asset, stride 10).
R1 (index_v3 rows, _ego per row; signal 1 = red, as reports/r2ft_regression_analysis.md):
  red_approach  signal 1, v >= 1 and v(+3 s) < v - 1            red_stop  signal 1, v < 0.5 (incl. waits after respawn)
  highway       path s 1118.7-1741.7 (CP10-CP12)                 ring_entry  s 860-885, v >= 1, no stop in the next 6.4 s
                and past s 886 by then (enters the ring; entry s 879-885, reports/ring_entry_20261028.md)
  ring_wait     s 870-886 inside a stop (v < 0.5) longer than 5 s -> out of the replay (replay_exclude = + start relabel)
  red_val       red_approach + red_stop rows of val + val_asset, stride 10
Train pools are cut to an episode set E (the box downloads E + the val episodes only): episodes in order of their scarce
rows (ring entry, red approach, highway), until each pool has --margin x its 2 Hz need at --samples.
Usage (cwd train/): python scripts/stop_pools.py --r1-index .. --r1-ego .. --r2-index .. --r2-ego .. --r2-meta ..
    --relabel <start_relabel_index.parquet> --out assets/stop [--samples 32000]"""
import argparse, json
from pathlib import Path

import numpy as np, pandas as pd

SHARE = dict(r2_aeb=.15, r2_yield=.10, red_approach=.15, red_stop=.10, highway=.10, new_d=.05, old_recovery=.05,
             ring_entry=.05, replay=.20, start_relabel=.05)
FUT, STOP_V, LOOK = 64, 0.5, 80


def stops(z, t0, t1):
    """grid row of the first v < STOP_V in [t0, t0 + 8 s] (None: no stop)."""
    tau, v = z["tau"], z["v"]; k = np.flatnonzero((tau >= t0 - 0.05) & (tau <= max(t1, t0) + 8) & (v < STOP_V))
    return int(k[0]) if len(k) else None


def r2_rows(idx, ego, meta, eps):
    out = {"aeb": [], "yield": []}
    for r in meta[meta.episode.isin(eps)].itertuples():
        f = Path(ego) / f"{r.episode}.npz"
        if not f.exists(): continue
        z = np.load(f)
        for kind, runs in (("aeb", r.aeb_tau), ("yield", r.yield_tau)):
            for t0, t1 in (runs if runs is not None else []):
                s = stops(z, float(t0), float(t1))
                if s is None: continue
                out[kind] += [(r.episode, i) for i in range(max(s - FUT, 0), s + 1)]
    return {k: pd.DataFrame(v, columns=["episode", "i"]).drop_duplicates().merge(idx, on=["episode", "i"]) for k, v in out.items()}


def r1_tags(idx, ego):
    """per index row: red_approach, red_stop, highway, ring_entry, ring_wait (bool columns)."""
    parts = []
    for ep, g in idx.groupby("episode", observed=True):
        f = Path(ego) / f"{ep}.npz"
        if not f.exists(): continue
        z = np.load(f); v, s, sig = z["v"], z["path_s"], z["signal"].astype(int); n = len(v); i = g.i.to_numpy()
        j3, j64 = np.minimum(i + 30, n - 1), np.minimum(i + FUT, n - 1)
        stopped = v < STOP_V; run = np.zeros(n)                                   # length (s) of the stop run each row is in
        k = 0
        while k < n:
            if stopped[k]:
                e = k
                while e + 1 < n and stopped[e + 1]: e += 1
                run[k:e + 1] = (e - k + 1) * 0.1; k = e + 1
            else: k += 1
        nostop = np.array([v[a:b + 1].min() >= STOP_V for a, b in zip(i, j64)])
        parts.append(g.assign(red_approach=(sig[i] == 1) & (v[i] >= 1) & (v[j3] < v[i] - 1), red_stop=(sig[i] == 1) & stopped[i],
                              highway=(s[i] >= 1118.7) & (s[i] <= 1741.7),
                              ring_entry=(s[i] >= 860) & (s[i] <= 885) & (v[i] >= 1) & nostop & (s[j64] > 886),
                              ring_wait=(s[i] >= 870) & (s[i] <= 886) & stopped[i] & (run[i] > 5)))
    return pd.concat(parts, ignore_index=True)


def pick_episodes(t, need, margin, first=()):
    """the val episodes (downloaded anyway) first, then episodes in order of their scarce rows, until every pool has
    margin x need 2 Hz rows."""
    t2 = t[t.i % 5 == 0]; cnt = t2.groupby("episode")[list(need)].sum()
    score = sum(cnt[c] / max(cnt[c].sum(), 1) for c in ("ring_entry", "red_approach", "highway")) + cnt.index.isin(list(first)) * 1e3
    have, E = dict.fromkeys(need, 0), []
    for ep in score.sort_values(ascending=False).index:
        if all(have[c] >= margin * need[c] for c in need) and ep not in first: break
        E.append(ep); have = {c: have[c] + int(cnt.loc[ep, c]) for c in need}
    return E, have


def main():
    ap = argparse.ArgumentParser()
    for k in ("--r1-index", "--r1-ego", "--r2-index", "--r2-ego", "--r2-meta", "--relabel", "--out"): ap.add_argument(k, required=True)
    ap.add_argument("--val-rows", required=True, help="ft_val_rows.jsonl of the R2 run: its general / curve keys are kept exactly")
    ap.add_argument("--pools", default="assets/ft"); ap.add_argument("--samples", type=int, default=32000); ap.add_argument("--margin", type=float, default=1.3)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    i1 = pd.read_parquet(a.r1_index, columns=["episode", "i", "split"]).astype({"episode": str})
    i2 = pd.read_parquet(a.r2_index, columns=["episode", "i", "split"]).astype({"episode": str})
    meta = pd.read_parquet(a.r2_meta).astype({"episode": str}); A = set(pd.read_parquet(f"{a.pools}/r2_A.parquet").episode.astype(str))
    r2 = r2_rows(i2, a.r2_ego, meta, A)
    for k, d in r2.items(): d[d.split == "train"][["episode", "i"]].to_parquet(out / f"r2_{k}.parquet", index=False)
    dyn = pd.concat(r2.values()).drop_duplicates(["episode", "i"]); dyn = dyn[dyn.split.isin(["val", "val_asset"]) & (dyn.i % 5 == 0)]
    dyn[["episode", "i"]].to_parquet(out / "r2_dynstop_val.parquet", index=False)
    vr = pd.DataFrame([json.loads(l) for l in open(a.val_rows)]); vr = vr[vr.step == 0]
    for name in ("general", "curve"):                                          # the R2 run's exact rows (s2 reference values)
        k = vr[vr.set == name].key.str.rsplit("/", n=1, expand=True).rename(columns={0: "episode", 1: "i"}).astype({"i": int})
        r2k = k.episode.str.startswith("r2")
        k[~r2k].to_parquet(out / f"{name}_val.parquet", index=False)
        if r2k.any(): k[r2k].to_parquet(out / f"{name}_val_r2.parquet", index=False)
    t = r1_tags(i1, a.r1_ego); tr = t[t.split == "train"]
    rv = t[t.split.isin(["val", "val_asset"]) & (t.red_approach | t.red_stop) & (t.i % 10 == 0)]
    rv = pd.concat([rv[rv.red_approach].sample(min(150, int(rv.red_approach.sum())), random_state=0),
                    rv[rv.red_stop & ~rv.red_approach].sample(150, random_state=0)])           # 300 rows: half approach, half stop
    rv[["episode", "i"]].to_parquet(out / "red_val.parquet", index=False)
    V = set(pd.read_parquet(out / "general_val.parquet").episode) | set(pd.read_parquet(out / "curve_val.parquet").episode) | set(rv.episode)
    need = {c: int(SHARE[c] * a.samples) for c in ("red_approach", "red_stop", "highway", "ring_entry")}
    E, have = pick_episodes(tr, need, a.margin, V)
    rec = pd.read_parquet(f"{a.pools}/old_recovery.parquet").astype({"episode": str})
    rel = pd.read_parquet(a.relabel).astype({"episode": str})
    for c in need: tr[tr[c] & tr.episode.isin(E)][["episode", "i"]].to_parquet(out / f"{c}.parquet", index=False)
    rec[rec.episode.isin(E)].to_parquet(out / "old_recovery.parquet", index=False)
    rel[rel.episode.isin(E)].to_parquet(out / "start_relabel.parquet", index=False)
    pd.DataFrame({"episode": sorted(E)}).to_parquet(out / "replay_episodes.parquet", index=False)       # replay: s2 rows of E
    ex = pd.concat([rel[["episode", "i"]], t[t.ring_wait][["episode", "i"]]]).drop_duplicates()
    ex.to_parquet(out / "replay_exclude.parquet", index=False)
    (out / "r1_episodes.txt").write_text("\n".join(sorted(set(E) | V)) + "\n")
    s = {"E": len(E), "val_episodes": len(V), "r1_download_episodes": len(set(E) | V), "need_2hz": need, "have_2hz": have,
         "r2_aeb_rows": len(r2["aeb"][r2["aeb"].split == "train"]), "r2_yield_rows": len(r2["yield"][r2["yield"].split == "train"]),
         "r2_dynstop_val": len(dyn), "red_val": len(rv), "ring_wait_rows": int(t.ring_wait.sum()), "replay_exclude": len(ex),
         "old_recovery_rows": int(rec.episode.isin(E).sum()), "start_relabel_rows": int(rel.episode.isin(E).sum())}
    json.dump(s, open(out / "pools.json", "w"), indent=1); print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
