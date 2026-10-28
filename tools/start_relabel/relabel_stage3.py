"""Stage 3 (Mac): merge the LiDAR corridor and moving-object checks, decide each pre-departure row, write the relabel files and the summary."""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).parent; IN = HERE / "relabel"; OUT = HERE / "start_relabel"; OUT.mkdir(exist_ok=True)
SCAN_GAP = 0.15
R = pd.read_parquet(IN / "rows_stage1.parquet"); E = pd.read_parquet(IN / "episodes_stage1.parquet")
lid = pd.DataFrame([json.loads(l) for f in sorted(glob.glob(str(IN / "lidar_out.*.jsonl"))) for l in open(f)])
missing = sorted(lid[lid["missing"].fillna(0) == 1].ep.unique()) if "missing" in lid else []
lid = lid[lid.tau.notna()].rename(columns={"ep": "episode"}).sort_values(["episode", "tau"])

dyn = pd.DataFrame([json.loads(l) for f in sorted(glob.glob(str(IN / "lidar_dyn.*.jsonl"))) for l in open(f)])
dyn = dyn.rename(columns={"ep": "episode"}).sort_values(["episode", "tau"])


def nearest_scan(R, tab, cols):
    """per row: the columns of its episode's nearest scan (scans 8.4-8.9 Hz) and the gap to it; NaN where the episode has none."""
    parts = []
    for ep, g in R.groupby("episode", sort=False):
        s = tab[tab.episode == ep]; tq = g.tau.to_numpy()
        if not len(s):
            parts.append(g.assign(**{c: np.nan for c in cols}, gap=np.nan)); continue
        t = s.tau.to_numpy(); k = np.clip(np.searchsorted(t, tq), 0, len(t) - 1); k0 = np.maximum(k - 1, 0)
        k = np.where(np.abs(t[k0] - tq) <= np.abs(t[k] - tq), k0, k)
        parts.append(g.assign(**{c: s[c].to_numpy()[k] for c in cols}, gap=np.abs(t[k] - tq)))
    return pd.concat(parts, ignore_index=True)


R = nearest_scan(R, lid, ["s_obs", "cells"]).rename(columns={"gap": "scan_dt"})
R["obstacle"] = R.s_obs.fillna(-1) >= 0
# moving objects near the path at row i that the departure-time scans lack. The last two scans are that reference and are not
# diffed, so rows nearest to them (more than 0.15 s from a diffed scan) have none.
R = nearest_scan(R, dyn, ["new", "new_s", "gone"]).rename(columns={"gap": "dyn_dt"})
far = ~(R.dyn_dt <= 0.15)
R.loc[far, ["new", "gone"]] = 0; R.loc[far, "new_s"] = -1.0
R["moving_near"] = R.new_s >= 0

orig = pd.read_parquet(IN / "index_orig.parquet", columns=["episode", "i", "split", "nav_text", "cell"]).astype({"episode": str})
R = R.merge(orig.rename(columns={"split": "orig_split", "nav_text": "orig_nav", "cell": "orig_cell"}), on=["episode", "i"], how="left")
R["in_index"] = R.orig_split.notna()

def status(r):
    if r.episode in missing: return "no_step0_lidar"          # MoraiDataset skips episodes without lidar_index.npz
    if r.signal_reason: return r.signal_reason
    if r.obstacle: return "obstacle_lidar"
    if r.moving_near: return "moving_object"
    if not r.mask_ok: return "mask"
    if r.glitch: return "glitch"
    return "relabeled"
R["status"] = R.apply(status, axis=1)

# consistency with build_index for rows that are in both
both = R[R.in_index]
chk = dict(split_match=float((both.split == both.orig_split).mean()), nav_match=float((both.nav_text == both.orig_nav).mean()),
           cell_match=float((both.cell == both.orig_cell).mean()))

ok = R[R.status == "relabeled"]
cols = ["episode", "i", "tau", "cell", "split", "nav_text", "fog", "hour", "scene"]
idx = ok[cols + ["src_i", "shift_s"]].copy()
idx["src_tau"] = ok.tau + ok.shift_s
for c in ("episode", "cell", "split", "nav_text", "scene"): idx[c] = idx[c].astype("category")
idx = idx.sort_values(["episode", "i"]).reset_index(drop=True)
idx.to_parquet(OUT / "start_relabel_index.parquet", index=False)

allrows = R[["episode", "i", "tau", "src_i", "shift_s", "status", "signal_reason", "obstacle", "s_obs", "cells", "moving_near", "new_s",
             "new", "gone", "mask_ok", "glitch",
             "line_id", "line_d", "in_index", "split"]].sort_values(["episode", "i"]).reset_index(drop=True)
allrows.to_parquet(OUT / "start_rows_all.parquet", index=False)

fut = E[E.episode.isin(R.episode.unique())][["episode", "start_cp", "src_i", "tau_dep", "fut_len", "fut_vmax", "fut_xyz", "fut_yaw"]]
fut = fut.assign(fut_xyz=[np.asarray(x, np.float32) for x in fut.fut_xyz], fut_yaw=[np.asarray(x, np.float32) for x in fut.fut_yaw])
fut.to_parquet(OUT / "start_future.parquet", index=False)

# summary
ep_stat = R.groupby("episode").status.agg(lambda s: "relabeled" if (s == "relabeled").any() else "none")
E2 = E.set_index("episode")
cp_rows = (R.assign(cp=R.episode.map(E2.start_cp)).groupby("cp").status.value_counts().unstack(fill_value=0))
summary = dict(
    episodes_total=int(len(E)), episodes_with_stop_rows=int(R.episode.nunique()),
    episodes_no_stop=int((E.skip == "no_stop").sum()), episodes_short=int((E.skip == "short").sum()),
    episodes_relabeled=int((ep_stat == "relabeled").sum()),
    departure_tau_s={k: round(float(v), 2) for k, v in E.tau_dep.describe(percentiles=[.5, .9, .99]).items()},
    rows_pre_departure=int(len(R)), rows_in_original_index=int(R.in_index.sum()),
    rows_by_status={k: int(v) for k, v in R.status.value_counts().items()},
    relabeled_rows=int(len(idx)), relabeled_by_split={k: int(v) for k, v in idx.split.value_counts().items()},
    relabeled_hours=round(len(idx) / 36000, 3),
    relabeled_in_original_index=int(ok.in_index.sum()),
    excluded_reason_overlap=dict(signal_and_obstacle=int((R.signal_reason.astype(bool) & R.obstacle).sum()),
                                 obstacle_any=int(R.obstacle.sum()), moving_any=int(R.moving_near.sum()),
                                 signal_and_moving=int((R.signal_reason.astype(bool) & R.moving_near).sum()),
                                 relabeled_with_traffic_gone=int(((R.status == "relabeled") & (R.gone > 0)).sum())),
    episodes_excluded_by_reason={k: int(R[R.status == k].episode.nunique()) for k in R.status.unique() if k != "relabeled"},
    shift_s={k: round(float(v), 2) for k, v in idx.shift_s.describe(percentiles=[.5, .9]).items()},
    rows_by_start_cp={int(k): {c: int(v) for c, v in row.items() if v} for k, row in cp_rows.iterrows()},
    consistency_with_build_index=chk, lidar_missing_episodes=missing,
    rows_without_scan_within_0_15s=int((R.scan_dt.fillna(9) > SCAN_GAP).sum()))
json.dump(summary, open(OUT / "summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(summary, indent=1, ensure_ascii=False))
