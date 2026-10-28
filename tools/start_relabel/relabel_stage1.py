"""Stage 1 (Mac): pre-departure rows, their relabeled futures, mask and signal checks; LiDAR request for winpc."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

WS = Path("/Users/taeyunkim/Library/Mobile Documents/com~apple~CloudDocs/Workspaces/Alpamayo-Cosmos")
sys.path.insert(0, str(WS / "train"))
from morai_vla import build_index as bi   # noqa: E402
data = bi.data

OUT = Path(__file__).parent / "relabel"; OUT.mkdir(exist_ok=True)
HIST, FUT, DT = bi.HIST, bi.FUT, 0.1
MOVE_M = 0.05                 # scorer time.move_m: GT displacement that counts as movement
STILL_V = 0.05                # m/s: standstill (the speed ramps 0.3-0.5 m/s per row once the driver releases)
FRONT_AXLE = 3.0              # base_link -> front axle (scorer vehicle.front_axle_m)
SIG_MARGIN = 50.0             # m past the label's reach within which a judged light still matters: the collection driver
                              # stops at the first OSM stop line, 20-40 m before the judged one (DATA_PLAN_R2.md 1)
CFG = data.load_config()
STOPS = {sid: data.kmap().route_stops[sid] for sid in CFG["signal"]["stop_lines"]}


def future(v, yaw, d):
    """ego future (64 pts) at grid row d exactly as morai_vla.data.ego_trajectory: xyz (64, 3) and yaw rel. to row d."""
    j = np.arange(d - HIST + 1, d + FUT + 1)
    h = yaw[j] - yaw[d]
    step = 0.5 * (v[j][1:] + v[j][:-1]) * DT; hm = 0.5 * (h[1:] + h[:-1])
    x, y = np.r_[0, np.cumsum(step * np.cos(hm))], np.r_[0, np.cumsum(step * np.sin(hm))]
    x, y = x - x[HIST - 1], y - y[HIST - 1]
    xyz = np.stack([x, y, np.zeros(len(j))], -1).astype(np.float32)
    return xyz[HIST:], h[HIST:].astype(np.float32)


def main():
    held = json.load(open(OUT / "splits_orig.json"))["held_out_assets"]   # the built index's list: new R2 manifests change a recompute
    M, lap, exempt = bi.maneuvers(), float(data.path_xy()[1][-1]), data.kmap().speed_exempt_s
    rows, eps, req = [], [], []
    for f in sorted(bi.EGO.glob("*.npz")):
        ep = f.stem; z = np.load(f)
        G = {k: z[k] for k in ("tau", "bag_ns", "path_s", "section", "fog", "hour", "v", "yaw", "signal")}
        v = G["v"].astype(np.float64); yaw = G["yaw"]; n = len(v)
        dist = np.r_[0, np.cumsum(0.5 * (v[1:] + v[:-1]) * DT)]
        moved = np.flatnonzero(dist > MOVE_M)
        d = int(moved[0]) - 1 if len(moved) else n - 1          # last row before the start's first GT movement
        while d > 0 and v[d] > STILL_V: d -= 1                  # ... at standstill, so its future starts at v = 0 like row i's history
        sc_id, sc = data.scene_of(ep)
        cp = (sc or {}).get("start", {}).get("checkpoint", 0 if sc else -1)   # rv laps start at the lap start (0)
        e = dict(episode=ep, scene=sc_id or "", start_cp=int(cp), s0=float(G["path_s"][0]), src_i=d, tau_dep=float(G["tau"][d]),
                 n=n, stop_rows=max(d - (HIST - 1), 0))
        eps.append(e)
        if d < HIST or d + FUT >= n:                            # no pre-departure row with a full history / future
            e["skip"] = "no_stop" if d < HIST else "short"; continue
        L = pd.read_parquet(data.labels_file(ep), columns=["bag_time_ns", "traj_ok"])
        ok_win = bi.traj_ok_on_grid(G, L)
        jump = np.r_[0, np.cumsum(np.abs(np.diff(v)) > bi.MAX_DV)]
        fxyz, fyaw = future(v, yaw, d)
        arc = np.r_[0, np.cumsum(np.hypot(*np.diff(np.r_[[[0, 0]], fxyz[:, :2]], axis=0).T))][1:]   # path length at each future pt
        e.update(fut_len=float(arc[-1]), fut_vmax=float(v[d:d + FUT + 1].max()))
        # split exactly as build_index
        unit = sc_id or ep
        base_split = "val" if bi.unit_hash(0, unit) < 0.05 else "train"
        cells = data.ROUTE["cells"]
        sig = G["signal"]
        for i in range(HIST - 1, d):
            r = dict(episode=ep, i=i, tau=float(G["tau"][i]), src_i=d, shift_s=float(G["tau"][d] - G["tau"][i]))
            r["mask_ok"] = bool(ok_win[d])
            r["glitch"] = bool((jump[i] - jump[i - HIST + 1]) > 0 or (jump[d + FUT] - jump[d]) > 0)
            s = float(G["path_s"][i]); fs = s + FRONT_AXLE
            ahead = sorted((st - fs, sid) for sid, st in STOPS.items() if st + 5.0 - fs > 0)
            r["line_id"], r["line_d"] = (ahead[0][1], float(ahead[0][0])) if ahead else ("", np.nan)
            # The label from d obeyed the light over [d, d + 6.4 s]; at row i the light over [i, i + 6.4 s] must be the same
            # single state, or the shifted label is not what the driver would have done. Only where the line can matter.
            reason = ""
            if ahead and ahead[0][0] <= arc[-1] + SIG_MARGIN:
                si, sd = sig[i:i + FUT + 1], sig[d:d + FUT + 1]
                if (si == 0).any() or (sd == 0).any() or not ((si == si[0]).all() and (sd == si[0]).all()):
                    # named by the row's own light: red/yellow at the row (GT gaps in the windows included), else no GT at the
                    # row or in a window, else a change within the windows
                    reason = {1: "signal_red", 2: "signal_yellow"}.get(int(sig[i]), "")
                    reason = reason or ("signal_unknown" if (si == 0).any() or (sd == 0).any() else "signal_change")
            r["signal_reason"] = reason
            split = base_split
            for act in (sc or {}).get("actors", []):
                if act.get("id") in held and act["s"] + bi.ASSET_WINDOW[0] <= s <= act["s"] + bi.ASSET_WINDOW[1] and split == "train":
                    split = "val_asset"
            r.update(cell=cells.get(int(G["section"][i]), "C?"), split=split, nav_text=bi.nav_text(s, M, lap, exempt),
                     fog=bool(G["fog"][i]), hour=int(G["hour"][i]), scene=sc_id or "")
            rows.append(r)
        # manifest actors within the label's reach (information for the obstacle check)
        near = [dict(zone=a.get("zone"), id=a.get("id"), ds=round(a["s"] - e["s0"], 1)) for a in (sc or {}).get("actors", [])
                if -5 <= a["s"] - e["s0"] <= arc[-1] + 10]
        e["actors_near"] = json.dumps(near)
        e["fut_xyz"] = fxyz.reshape(-1).tolist(); e["fut_yaw"] = fyaw.tolist()
        req.append(dict(ep=ep, t_lo=float(G["tau"][HIST - 1]) - 0.2, t_hi=float(G["tau"][d]) + 0.05,
                        path=np.round(fxyz[:, :2], 3).tolist()))
    E = pd.DataFrame(eps); R = pd.DataFrame(rows)
    E.to_parquet(OUT / "episodes_stage1.parquet", index=False); R.to_parquet(OUT / "rows_stage1.parquet", index=False)
    json.dump(req, open(OUT / "lidar_req.json", "w"))
    print("episodes", len(E), "with rows", R.episode.nunique(), "rows", len(R))
    print(E.skip.value_counts(dropna=False).to_string())
    print(R.signal_reason.value_counts().to_string()); print("mask_ok", R.mask_ok.mean(), "glitch", R.glitch.sum())
    print("held", held)


if __name__ == "__main__":
    main()
