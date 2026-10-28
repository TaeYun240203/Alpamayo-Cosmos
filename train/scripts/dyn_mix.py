"""Mixture of the dynamic-stop fine-tune (student_dyn.yaml) at the target shares of 10/28 (main session, user-facing proposal):
every source is split into sub-sources along its axes so the axis ratios hold whatever the row counts are (MixtureDataset
draws a source's share from its rows; rows repeat only when a sub-source is short). CPU, Mac.
  dyn_stop 17 %   section A 65 / B 35 x kind soft 50 / hard 30 / popout 12 / hold 8 (B has no pop-out: its share goes to
                  B's other kinds pro rata); object person / prop natural (reported); history real 50 / synth 50
  dyn_pass 5 %    section 65 / 35 x type along 40 / late_cross 30 / crossed 30; history default
  static 15 %     zone general (R1 static) / tunnel R1 / tunnel R2 = 50 / 10 / 40 (R1 60 / R2 40; ra_parked never intrudes on
                  the lane: 0 rows) x clearance 0.3-0.6 m 30 / 0.6-1.0 m 50 / >= 1.0 m 20; a side > 70 % is cut to 50 / 50
  red 10 %        approach 60 / wait 40 x judged line (7294 s 208.5, 6254 s 315.9, 4188 s 436.8, 4356 s 609.9) 25 % each;
                  history real 50 / synth 50
  lane 15 %       R2 D 10 / old recovery 5; history real / zero / synth thirds
  replay 38 %     the stop round's R1 replay rows minus the union, cell_keep as before
Out: OUT/mix/<source>.parquet (keep lists), OUT/mix/mix.json (per source: share, samples at 1,000 steps, rows, 2 Hz rows,
repeat factor, axis ratios inside and of the whole), OUT/mix/sources.yaml (the config block).
Usage (cwd = workspace): python train/scripts/dyn_mix.py R1_ROOT R2_ROOT DYN_OUT [--steps 1000]
"""
import argparse, json
from pathlib import Path

import numpy as np, pandas as pd

TOTAL_PER_STEP, STOP = 32, Path("train/assets/stop")
LINES = {"7294": 208.5, "6254": 315.9, "4188": 436.8, "4356": 609.9}
SECTIONS = [("intersections s<700", 0, 700), ("s 700-840", 700, 840), ("roundabout s 840-950", 840, 950),
            ("s 950-1118", 950, 1118.7), ("highway s 1118.7-1741.7", 1118.7, 1741.7), ("s 1741-1859", 1741.7, 1859), ("tunnel s 1859-1996", 1859, 1996), ("s >= 1996", 1996, 1e9)]


def line_of(s):
    """the judged line a red-light row approaches or waits at: the nearest line 5 m behind .. 120 m ahead."""
    best = None
    for k, ls in LINES.items():
        if -5 <= ls - s <= 120 and (best is None or ls - s < LINES[best] - s): best = k
    return best or "other"


def ego_col(root, df, col, f=None):
    out = np.empty(len(df), object); df = df.reset_index(drop=True)
    for ep, g in df.groupby("episode"):
        z = np.load(Path(root) / "_ego" / f"{ep}.npz"); out[g.index] = f(z, g.i.to_numpy()) if f else z[col][g.i.to_numpy()]
    return out


def turn(z, i):
    """future heading change over 6.4 s: left / right / straight (10 deg)."""
    j = np.minimum(i + 64, len(z["yaw"]) - 1); dh = z["yaw"][j] - z["yaw"][i]
    return np.where(dh > 0.17, "left", np.where(dh < -0.17, "right", "straight"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("r1"); ap.add_argument("r2"); ap.add_argument("dyn"); ap.add_argument("--steps", type=int, default=1000)
    a = ap.parse_args(); R1, R2, D = Path(a.r1), Path(a.r2), Path(a.dyn); M = D / "mix"; M.mkdir(exist_ok=True)
    N = a.steps * TOTAL_PER_STEP
    i1 = pd.read_parquet(R1 / "_index/index.parquet").astype({"episode": str}); i2 = pd.read_parquet(R2 / "_index/index.parquet").astype({"episode": str})
    U = pd.MultiIndex.from_frame(pd.read_parquet(D / "exclude/replay_exclude_dyn.parquet").astype({"episode": str}))
    srcs, info = [], {}

    cells = []

    def add(name, family, share, rows, root, new, hist, extra=None, axes=None, group=None):
        rows = rows.drop_duplicates(["episode", "i"]); n2 = int((rows.i % 5 == 0).sum())
        cells.append(dict(name=name, family=family, group=group or name, target=share, rows=rows, root=root, new=new, hist=hist,
                          extra=extra or {}, axes=axes or {}, avail=len(rows) if new else n2, n2=n2))

    def ratios(df, col):
        return {str(k): round(float(v), 3) for k, v in df[col].value_counts(normalize=True).items()} if len(df) else {}

    # dyn stop (virtual episodes)
    vi = pd.read_parquet(D / "dyn_index.parquet"); vi = vi[vi.split == "train"].copy()
    vi["k4"] = vi.kind.str.replace("popout_soft", "popout").str.replace("popout_hard", "popout")
    KIND = {"soft": .50, "hard": .30, "popout": .12, "hold": .08}
    for sec, w in (("A", .65), ("B", .35)):
        have = {k: r for k, r in KIND.items() if ((vi.section == sec) & (vi.k4 == k)).any()}; z = sum(have.values())
        for k, r in have.items():
            g = vi[(vi.section == sec) & (vi.k4 == k)]
            add(f"dyn_stop_{sec}_{k}", "dyn_stop", .17 * w * r / z, g, "r2", True, "real_synth", {"index": "_dyn_relabel/dyn_index.parquet"},
                dict(object=ratios(g, "cls"), timing=ratios(g, "timing")), group=f"dyn_stop_{sec}")
    # dyn pass (original episodes)
    L = pd.read_parquet(D / "dyn_relabel_index.parquet"); P = L[(L.kind == "pass") & (L.split == "train")]
    for sec, w in (("A", .65), ("B", .35)):
        for t, r in (("along", .40), ("late_cross", .30), ("crossed", .30)):
            g = P[(P.section == sec) & (P.pass_type == t)]
            add(f"dyn_pass_{sec}_{t}", "dyn_pass", .05 * w * r, g, "r2", True, "default", None, dict(object=ratios(g, "cls"), ptype=ratios(g, "ptype")), group=f"dyn_pass_{sec}")
    # static avoidance
    A = pd.read_parquet(D / "static_avoid.parquet").astype({"episode": str})
    A = A.merge(pd.concat([i1[["episode", "i", "split"]], i2[["episode", "i", "split"]]]), on=["episode", "i"]); A = A[A.split == "train"]
    A["data"] = np.where(A.episode.str.startswith("r2"), "r2", "r1"); A["gap"] = pd.cut(A.clearance, [0.3, 0.6, 1.0, 1e9], right=False, labels=["0.3-0.6", "0.6-1.0", "1.0+"]).astype(str)
    for zone, data, w in (("static", "r1", .50), ("tunnel_static", "r1", .10), ("tunnel_static", "r2", .40)):
        for gap, r in (("0.3-0.6", .30), ("0.6-1.0", .50), ("1.0+", .20)):
            g = A[(A.zone == zone) & (A.data == data) & (A.gap == gap)]
            sd = g.side.value_counts(normalize=True)
            if len(sd) > 1 and sd.max() > 0.7:                                       # cut the dominant side to the other's count (one side only: kept)
                n = int(g.side.value_counts().min()); g = pd.concat([x.sample(min(len(x), max(n, 1)), random_state=0) for _, x in g.groupby("side")])
            zn = "general" if zone == "static" else "tunnel"
            add(f"static_{zn}_{data}_{gap.replace('.', '').replace('-', '_').replace('+', 'p')}", "static_avoid", .15 * w * r, g, data, data == "r2", "default", None,
                dict(side=ratios(g, "side"), side_before=ratios(A[(A.zone == zone) & (A.data == data) & (A.gap == gap)], "side")), group=f"static_{zone}_{data}")
    # red light
    for phase, fname, w in (("approach", "red_approach", .60), ("wait", "red_stop", .40)):
        g = pd.read_parquet(STOP / f"{fname}.parquet").astype({"episode": str}).merge(i1, on=["episode", "i"])
        g = g[~pd.MultiIndex.from_frame(g[["episode", "i"]]).isin(U)]
        g["line"] = [line_of(s) for s in ego_col(R1, g, "path_s")]
        have = [k for k in LINES if (g.line == k).any()]
        for k in have:
            h = g[g.line == k]
            add(f"red_{phase}_{k}", "red", .10 * w / len(have), h, "r1", False, "real_synth", None, dict(fog=ratios(h, "fog"), hour=ratios(h, "hour")), group=f"red_{phase}")
    # lane following / recovery
    meta = pd.read_parquet(R2 / "_meta/r2_episodes.parquet")
    d = i2[i2.episode.isin(set(pd.read_parquet("train/assets/ft/r2_D.parquet").episode.astype(str))) & (i2.split == "train")]
    d = d[~pd.MultiIndex.from_frame(d[["episode", "i"]]).isin(U)].merge(meta[["episode", "start_cls"]], on="episode"); d["turn"] = ego_col(R2, d, None, turn)
    add("lane_r2_D", "lane", .10, d, "r2", True, "thirds", {"synth": "lane"}, dict(start_cls=ratios(d, "start_cls"), turn=ratios(d, "turn")))
    o = pd.read_parquet(STOP / "old_recovery.parquet").astype({"episode": str}).merge(i1, on=["episode", "i"])
    o = o[~pd.MultiIndex.from_frame(o[["episode", "i"]]).isin(U)]; o["turn"] = ego_col(R1, o, None, turn)
    add("lane_old_recovery", "lane", .05, o, "r1", False, "thirds", {"synth": "lane"}, dict(turn=ratios(o, "turn")))
    # replay
    E = set(pd.read_parquet(STOP / "replay_episodes.parquet").episode.astype(str))
    rp = i1[i1.episode.isin(E) & (i1.split == "train")]; rp = rp[~pd.MultiIndex.from_frame(rp[["episode", "i"]]).isin(U)]
    keep = {"C0": .4, "C3": .65, "C5": .55}; w = rp.cell.astype(str).map(lambda c: keep.get(c, 1.0))
    rp2 = rp[rp.i % 5 == 0]; s_ = ego_col(R1, rp2, "path_s"); sec = pd.Series(pd.cut(s_.astype(float), [x[1] for x in SECTIONS] + [1e9], right=False, labels=[x[0] for x in SECTIONS])).astype(str)
    ww = rp2.cell.astype(str).map(lambda c: keep.get(c, 1.0)).to_numpy()
    add("replay", "replay", .38, rp, "r1", False, "default", {"cell_keep": keep, "keep_file": "stop_round"},
        dict(cell_before=ratios(rp2, "cell"), cell_after={str(c): round(float(ww[(rp2.cell.astype(str) == c).to_numpy()].sum() / ww.sum()), 3) for c in sorted(rp2.cell.astype(str).unique())},
             section_after={k: round(float(ww[(sec == k).to_numpy()].sum() / ww.sum()), 3) for k in sec.unique()}))
    allocate(cells, N)
    for c in cells:
        if c["share"] <= 0: continue
        c["rows"][["episode", "i"]].to_parquet(M / f"{c['name']}.parquet", index=False); want = c["share"] * N
        srcs.append(dict(name=c["name"], share=c["share"], root=c["root"], new=c["new"], hist=c["hist"], extra=c["extra"]))
        info[c["name"]] = dict(family=c["family"], target_pct=round(100 * c["target"], 2), share_pct=round(100 * c["share"], 2), samples=int(round(want)),
                               rows_10hz=len(c["rows"]), rows_2hz=c["n2"], repeat=round(want / max(c["avail"], 1), 2), hist=c["hist"], axes=c["axes"])
    info["_empty_or_moved"] = {c["name"]: dict(target_pct=round(100 * c["target"], 2), share_pct=round(100 * c["share"], 2), rows_10hz=len(c["rows"])) for c in cells if abs(c["share"] - c["target"]) > 1e-4}
    info["_total"] = dict(samples=N, share_pct=round(100 * sum(s["share"] for s in srcs), 2), sources=len(srcs), max_repeat=MAX_REPEAT)
    json.dump(info, open(M / "mix.json", "w"), indent=1, ensure_ascii=False)
    yaml_block(srcs, M / "sources.yaml"); print(json.dumps({k: (v["share_pct"], v["samples"], v["rows_10hz"], v["repeat"]) for k, v in info.items() if not k.startswith("_")}))


MAX_REPEAT = 3.0


def allocate(cells, N):
    """target shares -> shares: a cell takes at most MAX_REPEAT x its rows (new: all rows, else 2 Hz rows); what it cannot
    take goes to the uncapped cells of its group pro rata to their targets, then to those of its family."""
    for c in cells: c["share"], c["cap"] = c["target"], MAX_REPEAT * c["avail"] / N
    for key in ("group", "family"):
        for _ in range(20):
            moved = False
            for g in {c[key] for c in cells}:
                cs = [c for c in cells if c[key] == g]; over = sum(max(c["share"] - c["cap"], 0) for c in cs)
                if over < 1e-9: continue
                for c in cs: c["share"] = min(c["share"], c["cap"])
                free = [c for c in cs if c["share"] < c["cap"] - 1e-12]; tw = sum(c["target"] for c in free)
                if not free or tw <= 0: continue
                for c in free: c["share"] += over * c["target"] / tw
                moved = True
            if not moved: break


def yaml_block(srcs, path):
    root = {"r1": "${data.root}", "r2": "${paths.r2_root}"}; lines = []
    for s in srcs:
        ds = [f'root: "{root[s["root"]]}"', "split: train"]
        if "index" in s["extra"]: ds.append(f'index: {s["extra"]["index"]}')
        ds.append(f'keep: "${{paths.mix}}/{s["name"]}.parquet"')
        if s["extra"].get("synth") == "lane": ds.append('synth: "${paths.synth_hist}"')
        elif s["hist"] == "real_synth": ds.append('synth: "${paths.synth_dyn}"')
        if "cell_keep" in s["extra"]: ds.append("cell_keep: {C0: 0.4, C3: 0.65, C5: 0.55}")
        opt = (", new: true" if s["new"] else "") + ("" if s["hist"] == "default" else f', hist: "${{paths.hist_{s["hist"]}}}"' if s["hist"] == "real_synth" else f", hist: {s['hist']}")
        loss = "distill" if s["name"] == "replay" else "gt"
        lines.append(f'      {s["name"]}:\n        {{share: {s["share"]:.5f}, loss: {loss}{opt}, dataset: {{{", ".join(ds)}}}}}')
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
