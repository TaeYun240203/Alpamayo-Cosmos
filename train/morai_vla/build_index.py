"""Mac side: one sample index for teacher/student training (MODEL_PLAN.md step 0 outputs -> index.parquet + splits.json).

Per 10 Hz tau-grid row of every episode: traj_ok on the sim clock, cell, nav_text, split. CoC is not baked in: the
dataset joins a CoC parquet given in its config (data.coc_parquet), so a relabel needs no rebuild.
Only rows that can be trained on are written: 1.5 s of history, 6.4 s of future, traj_ok, no speed glitch.
Usage: uv run --with pandas,pyarrow,pyyaml,numpy,scipy python -m morai_vla.build_index OUT_DIR   (cwd: train/)
R2 collection (README "R2 미세조정"): --ego <dir of the R2 _ego tables> --held-out 40100027,40100029,50100001 (index_v3's
held-out assets, so the old and the R2 splits hold out the same actors) --glitch net (10/28: real hard braking stays).
"""
import argparse, json, sys, zlib
from pathlib import Path

import numpy as np, pandas as pd

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS / "coc_labeler"))
from coc_labeler import data   # noqa: E402  (route.yaml, labels cache, manifests, scene log, map)

EGO = Path.home() / ".cache/step0_ego"
HIST, FUT, HORIZON_S = 16, 64, 6.4
NAV_RANGE = 150.0                     # m: farther maneuvers -> "Follow the road"
ASSET_WINDOW = (-120.0, 30.0)         # m of route s around a held-out actor whose rows leave train
MAX_DV = 1.5                          # m/s per 0.1 s (15 m/s^2): beyond any braking; GT speed glitches / impacts
START_S, NET_LAG, MAX_NET = 3.5, 0.8, 15.0   # --glitch net (R2, user 10/28): start transient, net-change span, m/s^2


def traj_ok_on_grid(G, L):
    """label traj_ok is judged over [t, t + 6.4 s] of WALL time; the sim ran at 0.7-0.95x, so re-judge on tau:
    row i is ok iff every label row from tau_i until one whose window reaches tau_i + 6.4 is ok."""
    tau_of = lambda ns: np.interp(ns, G["bag_ns"], G["tau"], right=np.inf)
    lt = tau_of(L.bag_time_ns.to_numpy(np.float64))
    end = tau_of(L.bag_time_ns.to_numpy(np.float64) + HORIZON_S * 1e9)          # window end of each label row (inf: past the bag)
    end = np.maximum.accumulate(np.where(np.isfinite(end), end, np.inf))
    bad = np.r_[0, np.cumsum(~L.traj_ok.to_numpy(bool))]
    j0 = np.searchsorted(lt, G["tau"]); j1 = np.searchsorted(end, G["tau"] + HORIZON_S)
    j1 = np.maximum(j1, j0)   # R2 runs at ~1x (rho up to 1.05): row j0's window already reaches tau_i + 6.4, j1 < j0 checked nothing
    return (j1 < len(L)) & (bad[np.minimum(j1 + 1, len(L))] - bad[j0] == 0)


def glitch_steps(v, tau, rule="step"):
    """(n - 1,) bool: the 0.1 s speed steps that make rows unusable. step (R1, the recipe): every |dv| > MAX_DV.
    net (R2, user 10/28 "동적 장애물에 의한 급제동 데이터를 남겨야지"): R2 Ego status samples stall and catch up, so a real
    hard brake shows 0.1 s steps over MAX_DV; such a step is kept when the net change around it (mean v 0.3-0.5 s after
    minus 0.2-0.4 s before, ~NET_LAG apart) has its sign, at least half its size and at most MAX_NET m/s^2. Still a glitch:
    a step that comes back (GT spike), any step in the first START_S s (the 20/40 kph start state applied at once) and a
    change faster than MAX_NET."""
    v = np.asarray(v, float); dv = np.diff(v); big = np.abs(dv) > MAX_DV
    if rule == "step": return big
    n, out = len(v), big.copy()
    for k in np.flatnonzero(big):
        net = v[min(k + 3, n - 1):min(k + 6, n)].mean() - v[max(k - 3, 0):max(k - 1, 1)].mean()
        real = np.sign(net) == np.sign(dv[k]) and abs(net) >= 0.5 * abs(dv[k]) and abs(net) / NET_LAG <= MAX_NET
        out[k] = tau[k] < START_S or not real
    return out


def maneuvers():
    """(route s of the maneuver point, text with {d}) from route.yaml."""
    out = []
    for f in data.ROUTE["features"]:
        if f["type"] == "intersection":
            out.append((f["stop_s"], f["s"][1], "Go straight through the intersection in {d}m" if f["movement"] == "straight"
                        else f"Turn {f['movement']} in {{d}}m"))
        elif f["type"] == "roundabout":
            out += [(f["entry_s"], f["entry_s"], "Enter the roundabout in {d}m"), (f["exit_s"], f["exit_s"], "Exit the roundabout in {d}m")]
        elif f.get("ramp"):
            out.append((f["s"][0], f["s"][0], "Take the ramp on the right in {d}m"))
        elif f["type"] == "merge":
            out.append((f["cross_s"], f["cross_s"], "Merge left in {d}m"))
        elif f["type"] == "toll_plaza":
            out.append((f["gate_s"], f["gate_s"], "Go through the toll gate in {d}m"))
    return sorted(out)


def nav_text(s, M, lap, exempt):
    """next maneuver within NAV_RANGE (distance clamped at 0 inside its junction), plus the speed rule."""
    text = "Follow the road"
    for k in (0, 1):                                           # this lap, then the next lap's first maneuvers
        hit = [(p + k * lap, t) for p, end, t in M if s <= end + k * lap and p + k * lap - s <= NAV_RANGE]
        if hit:
            p, t = hit[0]; text = t.format(d=int(round(max(p - s, 0.0)))); break
    return text + (". No speed limit" if exempt[0] <= s <= exempt[1] else ". Speed limit 60 km/h")


def held_out_assets(frac, seed):
    ids = set()
    for f in data.MANIFESTS.glob("*-manifest.json"):
        for v in json.load(open(f)).values():
            ids |= {a["id"] for a in v.get("actors", []) if a.get("id")}
    rng = np.random.default_rng(seed); out = []
    for group in (401, 501):                                   # obstacles 40100xxx, pedestrians 50100xxx
        g = sorted(i for i in ids if i // 100000 == group)
        out += sorted(int(x) for x in rng.choice(g, max(1, round(frac * len(g))), replace=False))
    return out


def unit_hash(*key):
    return zlib.crc32("/".join(map(str, key)).encode()) / 2**32


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--asset-frac", type=float, default=0.15); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ego", default=str(EGO)); ap.add_argument("--held-out", help="comma-separated actor ids (default: drawn)")
    ap.add_argument("--glitch", choices=["step", "net"], default="step", help="speed-step rule (glitch_steps); net = R2, 10/28")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    M, lap, exempt = maneuvers(), float(data.path_xy()[1][-1]), data.kmap().speed_exempt_s
    held = [int(x) for x in a.held_out.split(",")] if a.held_out else held_out_assets(a.asset_frac, a.seed)
    rows, stats = [], {"episodes": 0, "no_labels": [], "glitch_rows": 0}
    for f in sorted(Path(a.ego).expanduser().glob("*.npz")):
        ep = f.stem
        try:
            L = pd.read_parquet(data.labels_file(ep), columns=["bag_time_ns", "traj_ok"])
        except FileNotFoundError:
            stats["no_labels"].append(ep); continue
        z = np.load(f); G = {k: z[k] for k in ("tau", "bag_ns", "path_s", "section", "fog", "hour", "v")}
        n = len(G["tau"]); i = np.arange(n)
        ok = traj_ok_on_grid(G, L) & (i >= HIST - 1) & (i + FUT < n)
        jump = np.r_[0, np.cumsum(glitch_steps(G["v"], G["tau"], a.glitch))]                 # trajectories integrate GT speed
        glitch = (jump[np.minimum(i + FUT, n - 1)] - jump[np.maximum(i - HIST + 1, 0)]) > 0
        stats["glitch_rows"] += int((ok & glitch).sum()); ok &= ~glitch
        sc_id, sc = data.scene_of(ep)
        unit = sc_id or ep                                     # scenario seed = scene variant; old runs without one: the episode
        split = np.full(n, "val" if unit_hash(a.seed, unit) < a.val_frac else "train", object)
        for act in (sc or {}).get("actors", []):
            if act.get("id") in held:
                near = (G["path_s"] >= act["s"] + ASSET_WINDOW[0]) & (G["path_s"] <= act["s"] + ASSET_WINDOW[1])
                split[near & (split == "train")] = "val_asset"
        cells = data.ROUTE["cells"]
        for k in np.flatnonzero(ok):
            rows.append((ep, int(k), float(G["tau"][k]), cells.get(int(G["section"][k]), "C?"), split[k],
                         nav_text(float(G["path_s"][k]), M, lap, exempt), bool(G["fog"][k]), int(G["hour"][k]), sc_id or ""))
        stats["episodes"] += 1
    df = pd.DataFrame(rows, columns=["episode", "i", "tau", "cell", "split", "nav_text", "fog", "hour", "scene"])
    for c in ("episode", "cell", "split", "nav_text", "scene"): df[c] = df[c].astype("category")
    df.to_parquet(out / "index.parquet", index=False)
    summary = dict(stats, rows=len(df), held_out_assets=held, val_frac=a.val_frac, seed=a.seed, glitch=a.glitch,
                   by_split=df.split.value_counts().to_dict(),
                   hours_by_cell={c: round(n / 36000, 2) for c, n in df[df.split == "train"].cell.value_counts().items()})
    json.dump(summary, open(out / "splits.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in summary.items() if k != "no_labels"}, indent=1, default=str))


if __name__ == "__main__":
    main()
