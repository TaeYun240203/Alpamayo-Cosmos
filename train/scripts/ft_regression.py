"""Where did the R2 fine-tune's validation ADE grow? (user 10/28: "원인 분석 둘 다 해", CPU part.) CPU, no model.
Reads ft/ft_val_rows.jsonl (FtSelect's per-row ADE / FDE / lat / lon per measurement step) and tags every row from its
_ego table and index row: GT speed band at t0, future (+3 s) accelerating / braking / cruising, stopped and staying
stopped, cell, signal, turn ahead (nav_text), start (tau <= 15 s), episode source. Prints / writes per-group means at
each step and the share of the ADE / lon growth, and the rows that were stopped and stay stopped but now get a moving
prediction (ADE up).
Usage (cwd train/): python scripts/ft_regression.py --rows <ft_val_rows.jsonl> --r1-index <step0>/_index/index.parquet
    --r1-ego ~/.cache/step0_ego --r2-index <step0_r2>/_index/index.parquet --r2-ego <R2 _ego dir> --out <dir>"""
import argparse, json, re
from pathlib import Path

import numpy as np, pandas as pd


def tags(rows, idx, ego):
    """rows: DataFrame with key "episode/i" -> + v0, dv3, v64, band, motion, cell, signal, turn, start, source."""
    out = []
    cache = {}
    for k in rows.key.unique():
        ep, i = k.rsplit("/", 1); i = int(i)
        if ep not in cache:
            f = next((Path(d) / f"{ep}.npz" for d in ego if (Path(d) / f"{ep}.npz").exists()), None)
            cache[ep] = dict(np.load(f)) if f else None
        z = cache[ep]
        if z is None: continue
        n = len(z["v"]); v0 = float(z["v"][i]); v3 = float(z["v"][min(i + 30, n - 1)]); v64 = float(z["v"][min(i + 64, n - 1)])
        out.append(dict(key=k, episode=ep, i=i, v0=v0, dv3=v3 - v0, v64=v64, signal=int(z["signal"][i]) if "signal" in z else -1,
                        tau=float(z["tau"][i])))
    t = pd.DataFrame(out).merge(idx, on=["episode", "i"], how="left")
    t["band"] = pd.cut(t.v0, [-1, 0.5, 5, 10, 15, 99], labels=["stop<0.5", "0.5-5", "5-10", "10-15", ">15"])
    t["motion"] = np.select([(t.v0 < 0.5) & (t.v64 < 0.5), t.dv3 > 1.0, t.dv3 < -1.0], ["stays stopped", "accel", "brake"], "cruise")
    t["turn"] = t.nav_text.fillna("").map(lambda s: "turn<30m" if re.search(r"Turn \w+ in ([0-2]?\d)m", s) else "straight/far")
    t["start"] = np.where(t.tau <= 15, "start<=15s", "later")
    t["source"] = t.episode.str.replace(r"-w\d+-\d+$", "", regex=True)
    return t


def growth(df, by, a, b, m="ade"):
    """per group: rows, mean m at steps a and b, mean change, share of the set's total change."""
    p = df.pivot_table(index=["set", "key"], columns="step", values=m).reset_index()
    p = p.merge(df.drop_duplicates(["set", "key"]).drop(columns=["step", "ade", "fde", "lat", "lon"]), on=["set", "key"])
    p["d"] = p[b] - p[a]
    g = p.groupby(["set", by], observed=True).agg(n=("d", "size"), **{f"{m}_{a}": (a, "mean"), f"{m}_{b}": (b, "mean")}, d=("d", "mean"), dsum=("d", "sum"))
    g["share"] = g.dsum / g.groupby(level=0).dsum.transform("sum")
    return g.drop(columns="dsum").round(3)


def main():
    ap = argparse.ArgumentParser()
    for k in ("--rows", "--r1-index", "--r1-ego", "--r2-index", "--r2-ego", "--out"): ap.add_argument(k, required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame([json.loads(l) for l in open(a.rows)])
    idx = pd.concat([pd.read_parquet(p, columns=["episode", "i", "cell", "nav_text", "split"]) for p in (a.r1_index, a.r2_index)])
    t = tags(rows, idx.astype({"episode": str}), [a.r1_ego, a.r2_ego])
    df = rows.merge(t, on="key")
    steps = sorted(df.step.unique()); s0, s1 = steps[0], steps[-1]
    res = {}
    for by in ("band", "motion", "cell", "turn", "start", "signal", "source"):
        for m in ("ade", "lon", "lat"):
            for b in steps[1:]:
                res[f"{by}/{m}/{s0}->{b}"] = growth(df, by, s0, b, m)
    pd.concat(res, names=["table"]).to_csv(out / "growth.csv")
    p = df.pivot_table(index=["set", "key"], columns="step", values="ade").reset_index().merge(t, on="key")
    stay = p[(p.motion == "stays stopped") & (p[s1] - p[s0] > 0.5)].sort_values(s1, ascending=False)
    stay.to_csv(out / "stopped_rows_moving.csv", index=False)
    json.dump({"steps": [int(x) for x in steps], "rows": int(len(df)), "untagged": int(rows.key.nunique() - t.key.nunique())},
              open(out / "summary.json", "w"))
    figs(df, steps, out)
    print(json.dumps({"steps": [int(x) for x in steps], "rows": len(df), "stopped_moving": len(stay)}))


def figs(df, steps, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for by in ("band", "motion"):
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        for ax, m in zip(axs, ("ade", "lon")):
            g = df[df.set.isin(["general", "static", "curve"])].groupby([by, "step"], observed=True)[m].mean().unstack()
            g.plot.bar(ax=ax, rot=0); ax.set_title(f"{m} (m), general+static+curve, by {by}"); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(out / f"{by}.png", dpi=90); plt.close(fig)


if __name__ == "__main__":
    main()
