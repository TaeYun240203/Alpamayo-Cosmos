"""LwF dynamic-stop mixture (reports/lwf_dynstop_recipe_20261029.md 4-5, user 10/29 "오케이 이걸로 해보자"): keep lists and
shares for configs/student_lwf.yaml. CPU, no sensors.

Label braking start tau_b per event (source episode, obj): the earliest tau_i + 0.1 k_i over the event's relabelled rows,
k_i = the first future step whose label speed is below the original drive (_ego v at i+1+k) by more than --dev m/s
(hard: the first seen+conflict row; soft: where the envelope bites). An event whose label never leaves the original
drive (the driver braked at least as hard) falls back to the first seen+conflict row (seen_i). Decision rows: tau_b <=
tau < tau_b + --onset-s (with --visible: from max(tau_b, tau_vis)).
Cleanup (popout audit, reports/popout_audit_20261029.md; thresholds live in the audit's own tables):
  --exclude PARQUET  rows to drop: (episode, i) with virtual or source episode names, or (src_episode, i)
  --rowvis PARQUET   the audit's per-row Front visibility (a3b_rowvis.py: new_px, new_h, moving_s; approved rule 10/29):
                     tau_vis = the first stop row of an event whose newest Front frame shows the mover >= --vis-px px and
                     >= --vis-h px tall after it moved >= --vis-moving s; the event's stop rows before tau_vis are dropped
                     (an event never visible loses all its stop rows; restart rows stay) and its decision window starts at
                     max(tau_b, tau_vis)
Split: every popout `hard` cell and the R2 soft / hard / popout stop lists -> <name>_onset / <name>_rest; hold and restart
(rh) lists are not split. Shares: popout cells by popout_mix.balance on the cleaned train counts, summing --popout-share;
R2 stop lists keep their student_dyn proportions, summing --r2-share; a decision row weighs --weight x a row of the rest
of its cell (--norm cell, default: every cell keeps its share) or of its group (--norm group: restart / hold cells give way).
Popout hard events are short (the label brakes from the first seen row, rows end at the collision): most of their rows
are decision rows, so --norm cell barely moves them (lwf_mix.json `decision_rows_share` shows the effective weights).
Out: OUT_POPOUT/{popout_*.parquet, popout_val.parquet, popout_val_onset.parquet}, OUT_R2/dyn_stop_*_{onset,rest}.parquet,
OUT_POPOUT/lwf_mix.json (counts, tau_b - seen, shares) and the YAML sources block for the two groups on stdout.
User 10/29 base: --drop-rh (no pop-out restart cells) and --near-passes 8 (the near-shoulder cells get 8 x their rows /
--samples, the other pop-out cells share the rest as before; lwf_mix.json popout_passes). Switches: --popout-share / --replay-share
move share between the pop-outs and replay; with --mixin PATH the shares go to a Hydra mixin (configs/lwf_mix/<name>.yaml,
launch: SEXTRA="+lwf_mix=<name>", ft, QAD and gates alike) instead of the base block.
Usage (cwd train/): python scripts/lwf_mix.py --popout <step0_popout>/_dyn_relabel --popout-ego <step0_popout>/_ego \
  --r2 <step0_r2>/_dyn_relabel --r2-mix <step0_r2>/_dyn_relabel/mix_ring --r2-ego <step0_r2>/_ego \
  --out-popout <dir> --out-r2 <dir> --drop-rh --near-passes 8 [--exclude x.parquet] [--rowvis rowvis.parquet]
  v2 labels (10/29 rebuild, tools/vast/lwf_box.sh): --popout <step0_popout>/_dyn_relabel_v2 --popout-dir-rel _dyn_relabel_v2
  --merge-popout-index --write-config configs/lwf_dynstop.yaml. With a `kind` column in <popout>/mix/popout_val.parquet the val
  lists are its onset / pre_onset rows (FtSelect popout_onset / popout_pre); --merge-popout-index writes the original pop-out
  index (_popout_index/<run>/index.parquet, one per batch) as one file for popout_cruise.
"""
import argparse, glob, json, sys
from pathlib import Path

import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from popout_mix import ZONE, balance  # noqa: E402

DT = 0.1
R2_SHARES = {"dyn_stop_A_soft": 0.016575, "dyn_stop_A_hard": 0.009945, "dyn_stop_A_popout": 0.003978, "dyn_stop_A_hold": 0.002652,
             "dyn_stop_B_soft": 0.010143, "dyn_stop_B_hard": 0.006084, "dyn_stop_B_hold": 0.001623}   # student_dyn (dyn_mix.py x 0.3)
R2_SPLIT = ("soft", "hard", "popout")


def onset(lab, fut, ego_dir, dev=0.1):
    """lab: relabel rows (episode = source, i, obj, seen_i), fut: (episode, i, fut_v) -> lab + tau, tau_b, tau_seen per row."""
    f = fut.drop_duplicates(["episode", "i"]).set_index(["episode", "i"]).fut_v
    lab = lab.reset_index(drop=True); n = len(lab)
    tau, t_dev, seen = np.zeros(n), np.full(n, np.inf), np.zeros(n)
    for ep, idx in lab.groupby("episode").indices.items():
        z = np.load(Path(ego_dir) / f"{ep}.npz"); v, tz = z["v"].astype(float), z["tau"].astype(float)
        for j in idx:
            i = int(lab.i.iat[j]); tau[j], seen[j] = tz[i], tz[min(int(lab.seen_i.iat[j]), len(tz) - 1)]
            fv = f.get((ep, i))
            if fv is None: continue                                              # restart rows: no stop label
            d = (v[np.minimum(np.arange(i + 1, i + 65), len(v) - 1)] - np.asarray(fv, float)) > dev
            if d.any(): t_dev[j] = tz[i] + DT * int(np.argmax(d))
    lab = lab.assign(tau=tau, t_dev=t_dev, tau_seen=seen)
    first = lab.groupby(["episode", "obj"]).t_dev.transform("min")
    lab["tau_b"] = np.where(np.isfinite(first), first, lab.tau_seen)
    lab["fallback"] = ~np.isfinite(first)
    return lab


def mark(keys, lab, vis, onset_s):
    """keys: (src, i) frame -> (decision, keep) bool arrays: inside an event's decision window / not cut by --visible (a row
    of several events: decision if any, kept if all)."""
    l = lab[["episode", "i", "obj", "tau", "tau_b"]]
    if vis is not None:
        l = l.merge(vis[["src_episode", "obj", "tau_vis"]].rename(columns={"src_episode": "episode"}), on=["episode", "obj"], how="left")
        start, keep = np.fmax(l.tau_b, l.tau_vis), ~(l.tau < l.tau_vis)          # NaN tau_vis: no cut
    else:
        start, keep = l.tau_b, True
    l = l.assign(dec=(l.tau >= start - 1e-6) & (l.tau < start + onset_s - 1e-6), keep=keep)
    l = l.groupby(["episode", "i"]).agg(dec=("dec", "any"), keep=("keep", "all")).reset_index().rename(columns={"episode": "src"})
    m = keys.merge(l, on=["src", "i"], how="left")
    return m.dec.fillna(False).to_numpy(bool), m.keep.fillna(True).to_numpy(bool)


def visible_from(rv, lab, px, h, moving):
    """(src_episode, obj, tau_vis) per pop-out stop event from the audit's row table: the first stop row with the mover on
    Front >= px / h pixels after >= moving s of motion; inf when no row qualifies."""
    rv = rv[rv.kind != "restart_hold"].merge(lab[["episode", "i", "obj", "tau"]].rename(columns={"episode": "src"}), on=["src", "i"])
    rv = rv.assign(ok=(rv.new_px >= px) & (rv.new_h >= h) & (rv.moving_s >= moving))
    t = rv.groupby(["src", "obj"]).apply(lambda g: g.tau[g.ok].min() if g.ok.any() else np.inf, include_groups=False)
    return t.rename("tau_vis").reset_index().rename(columns={"src": "src_episode"})


def dropper(paths):
    """--exclude rows as a set of (name, i) keys matching virtual or source names."""
    out = set()
    for p in paths or []:
        x = pd.read_parquet(p)
        name = x.src_episode if "src_episode" in x else x.episode
        out |= set(zip(name.astype(str), x.i.astype(int)))
    return out


def weigh(nat, rows_on, weight, total, norm="cell"):
    """nat {name: share}, rows_on {name: (onset rows, rows)} -> {name or name_onset / name_rest: share} summing to total.
    A decision row weighs `weight` x a row of the rest of its cell (norm cell: every cell keeps its natural share) or of
    the rest of its group (norm group: the unsplit cells, e.g. restart rows, give way)."""
    w = {}
    for k, s in nat.items():
        if k in rows_on and rows_on[k][1]:
            a, n = rows_on[k]; c = n / (weight * a + n - a) if norm == "cell" else 1.0
            w[f"{k}_onset"] = c * weight * s * a / n; w[f"{k}_rest"] = c * s * (n - a) / n
        else:
            w[k] = s
    tot = sum(w.values())
    return {k: total * v / tot for k, v in w.items() if v > 0}


def main():
    ap = argparse.ArgumentParser()
    for a in ("--popout", "--popout-ego", "--r2", "--r2-mix", "--r2-ego", "--out-popout", "--out-r2"): ap.add_argument(a, required=True)
    ap.add_argument("--exclude", action="append"); ap.add_argument("--rowvis")
    ap.add_argument("--vis-px", type=int, default=30); ap.add_argument("--vis-h", type=int, default=8); ap.add_argument("--vis-moving", type=float, default=0.3)
    ap.add_argument("--drop-rh", action="store_true"); ap.add_argument("--replay-share", type=float); ap.add_argument("--mixin")
    ap.add_argument("--near-passes", type=float); ap.add_argument("--samples", type=int, default=64000)   # near-shoulder cells: passes over their rows
    ap.add_argument("--popout-index", default="dyn_index.parquet")   # dyn_index_nav.parquet: the 10/29 nav_text patch of the 06:36 merge
    ap.add_argument("--popout-dir-rel", default="_dyn_relabel")      # kept for old calls; the emitted config reads paths.popout_dyn (one knob)
    ap.add_argument("--write-config")                                # configs/lwf_dynstop.yaml: the dyn-stop sources (student_lwf's defaults)
    ap.add_argument("--merge-popout-index", action="store_true")     # OUT_POPOUT/popout_index_all.parquet (popout_cruise)
    ap.add_argument("--onset-s", type=float, default=1.0); ap.add_argument("--weight", type=float, default=4.0)
    ap.add_argument("--dev", type=float, default=0.1); ap.add_argument("--popout-share", type=float, default=0.20)
    ap.add_argument("--r2-share", type=float, default=0.05)
    ap.add_argument("--norm", choices=("cell", "group"), default="cell")   # weigh(); the recipe keeps sub-ratios: cell
    a = ap.parse_args()
    P, R, OP, OR = Path(a.popout), Path(a.r2), Path(a.out_popout), Path(a.out_r2)
    OP.mkdir(parents=True, exist_ok=True); OR.mkdir(parents=True, exist_ok=True)
    drop = dropper(a.exclude)
    rep = {"args": vars(a)}

    # ---- popout: cells (zone, start, kind2) of the cleaned train rows, hard cells split
    vi = pd.read_parquet(P / a.popout_index).astype({"episode": str, "src_episode": str})
    lab = onset(pd.read_parquet(P / "dyn_relabel_index.parquet").astype({"episode": str}), pd.read_parquet(P / "dyn_future.parquet").astype({"episode": str}), a.popout_ego, a.dev)
    vis = visible_from(pd.read_parquet(a.rowvis), lab, a.vis_px, a.vis_h, a.vis_moving) if a.rowvis else None
    dec, ok = mark(vi[["src_episode", "i"]].rename(columns={"src_episode": "src"}), lab, vis, a.onset_s)
    hard = (vi.kind2 == "hard").to_numpy(); vi["decision"] = dec & hard; ok = ok | ~hard             # the cut is for stop rows only
    ex = np.array([(e, i) in drop or (s, i) in drop for e, s, i in zip(vi.episode, vi.src_episode, vi.i)], bool)
    cut = ~ok & ~ex & (vi.split == "train").to_numpy()
    rep["popout_dropped"] = {"exclude": int(ex.sum()), "before_visible_train": int(cut.sum()), "before_visible_val": int((~ok & ~ex).sum() - cut.sum()),
                             "events_without_visible_row": int(np.isinf(vis.tau_vis).sum()) if vis is not None else 0,
                             "hard_train_rows_left": int(((vi.split == "train").to_numpy() & hard & ok & ~ex).sum())}
    vi = vi[~ex & ok]
    tr, va = vi[vi.split == "train"], vi[vi.split == "val"]
    va[["episode", "i"]].to_parquet(OP / "popout_val.parquet", index=False)
    va[va.decision][["episode", "i"]].to_parquet(OP / "popout_val_onset.parquet", index=False)
    pv = next((pd.read_parquet(x) for x in (P / "mix/popout_val.parquet", P / "popout_val.parquet") if x.exists()), None)
    if pv is not None and "kind" in pv:                                 # v2: the val events' onset / pre-onset rows
        for kind, f in (("onset", "popout_val_onset"), ("pre_onset", "popout_val_pre")): pv[pv.kind == kind][["episode", "i"]].to_parquet(OP / f"{f}.parquet", index=False)
        rep["popout_val_rows"] = pv.kind.value_counts().to_dict()
    if a.merge_popout_index:
        ix = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(P.parent / "_popout_index/*/index.parquet")))], ignore_index=True)
        ix.to_parquet(OP / "popout_index_all.parquet", index=False); rep["popout_index_rows"] = len(ix)
    cells, rows_on = {}, {}
    for (z, st, k), g in tr.groupby(["zone", "start", "kind2"]):
        name = f"popout_{ZONE[z]}_{st}_{k}"; cells[(z, st, k)] = len(g)
        if k == "hard":
            g[g.decision][["episode", "i"]].to_parquet(OP / f"{name}_onset.parquet", index=False)
            g[~g.decision][["episode", "i"]].to_parquet(OP / f"{name}_rest.parquet", index=False)
            rows_on[name] = (int(g.decision.sum()), len(g))
        else:
            g[["episode", "i"]].to_parquet(OP / f"{name}.parquet", index=False)
    pc = {c: n for c, n in cells.items() if not (a.drop_rh and c[2] == "rh")}
    near = {c: a.near_passes * n / a.samples for c, n in pc.items() if a.near_passes and c[1] == "near"}   # fixed shares, the rest balanced
    nat = {f"popout_{ZONE[z]}_{st}_{k}": s for (z, st, k), s in {**balance({c: n for c, n in pc.items() if c not in near}, a.popout_share - sum(near.values())), **near}.items()}
    sp = weigh(nat, rows_on, a.weight, a.popout_share, a.norm)
    h = lab[lab.kind != "restart_hold"].drop_duplicates(["episode", "obj"])
    cs = {c: sum(v for k, v in sp.items() if k == n or k.startswith(n + "_")) for c in pc for n in [f"popout_{ZONE[c[0]]}_{c[1]}_{c[2]}"]}
    per = lambda f: round(sum(cs[c] for c in pc if f(c)) * a.samples / max(sum(pc[c] for c in pc if f(c)), 1), 2)   # samples / rows
    rep["popout_passes"] = {"near": per(lambda c: c[1] == "near"), "other": per(lambda c: c[1] != "near"), "all": per(lambda c: True),
                            "rows_near": sum(n for c, n in pc.items() if c[1] == "near"), "rows_other": sum(n for c, n in pc.items() if c[1] != "near")}
    rep["popout"] = {"train_rows": len(tr), "val_rows": len(va), "val_decision_rows": int(va.decision.sum()), "rows_onset": rows_on,
                     "events": len(h), "fallback_events": int(h.fallback.sum()),
                     "tau_b_minus_seen_s": {q: round(float(np.nanpercentile(h.tau_b - h.tau_seen, q)), 2) for q in (5, 50, 95)}}

    # ---- R2: the student_dyn stop lists, soft / hard / popout split
    rvi = pd.read_parquet(R / "dyn_index.parquet", columns=["episode", "i", "src_episode"]).astype({"episode": str, "src_episode": str})
    rlab = onset(pd.read_parquet(R / "dyn_relabel_index.parquet", columns=["episode", "i", "obj", "kind", "seen_i"]).astype({"episode": str}),
                 pd.read_parquet(R / "dyn_future.parquet", columns=["episode", "i", "fut_v"]).astype({"episode": str}), a.r2_ego, a.dev)
    src = dict(zip(rvi.episode, rvi.src_episode)); r2_on = {}
    for name in R2_SHARES:
        k = pd.read_parquet(Path(a.r2_mix) / f"{name}.parquet").astype({"episode": str})
        k = k[[(e, i) not in drop and (src.get(e), i) not in drop for e, i in zip(k.episode, k.i)]]
        if not name.endswith(R2_SPLIT): continue
        d, ok = mark(pd.DataFrame({"src": k.episode.map(src), "i": k.i}), rlab, None, a.onset_s)
        k[d][["episode", "i"]].to_parquet(OR / f"{name}_onset.parquet", index=False)
        k[~d][["episode", "i"]].to_parquet(OR / f"{name}_rest.parquet", index=False)
        r2_on[name] = (int(d.sum()), len(k))
    tot = sum(R2_SHARES.values())
    sr = weigh({k: v / tot for k, v in R2_SHARES.items()}, r2_on, a.weight, a.r2_share, a.norm)
    hr = rlab[rlab.kind.isin(["soft", "hard", "popout_soft", "popout_hard"])].drop_duplicates(["episode", "obj"])
    rep["r2"] = {"rows_onset": r2_on, "events": len(hr), "fallback_events": int(hr.fallback.sum()),
                 "tau_b_minus_seen_s": {q: round(float(np.nanpercentile(hr.tau_b - hr.tau_seen, q)), 2) for q in (5, 50, 95)}}
    rep["shares"] = {"popout": {k: round(v, 6) for k, v in sp.items()}, "r2": {k: round(v, 6) for k, v in sr.items()}}
    frac = lambda sh, tot: round(sum(v for k, v in sh.items() if k.endswith("_onset")) / tot, 4)
    nat_frac = lambda nat, on: round(sum(s * on[k][0] / on[k][1] for k, s in nat.items() if k in on) / sum(nat.values()), 4)
    rep["decision_rows_share"] = {"popout": {"natural": nat_frac(nat, rows_on), "weighted": frac(sp, a.popout_share)},
                                  "r2": {"natural": nat_frac(R2_SHARES, r2_on), "weighted": frac(sr, a.r2_share)}}
    (OP / "lwf_mix.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k != "args"}, indent=1, default=str))

    if a.mixin:                                                        # share overrides only (sources stay student_lwf's)
        sh = {**{k: round(v, 6) for k, v in {**sp, **sr}.items()}, **({"replay": a.replay_share} if a.replay_share is not None else {})}   # same --drop-rh as the base
        Path(a.mixin).write_text(f"# @package _global_\n# scripts/lwf_mix.py --popout-share {a.popout_share} --r2-share {a.r2_share}"
                                 f"{' --drop-rh' if a.drop_rh else ''}{f' --replay-share {a.replay_share}' if a.replay_share is not None else ''}{f' --near-passes {a.near_passes}' if a.near_passes else ''}"
                                 f" (switch, launch SEXTRA=\"+lwf_mix={Path(a.mixin).stem}\")\ndata:\n  train_dataset:\n    sources:\n"
                                 + "".join(f"      {k}: {{share: {v}}}\n" for k, v in sh.items()))
        print("mixin", a.mixin); return

    # ---- sources block (student_lwf.yaml: paths.mix_lwf_popout, paths.mix_lwf_r2)
    pop = lambda n: f'{{root: "${{paths.popout_root}}", split: train, index: "${{paths.popout_dyn}}/{a.popout_index}", keep: "${{paths.mix_lwf_popout}}/{n}.parquet"'
    lines = []
    for n, s in sorted(sp.items()):
        hard = "_hard" in n
        lines.append(f"      {n}:\n        {{share: {s:.6f}, loss: gt, new: true,{' hist: \"${paths.hist_real_synth}\",' if hard else ''} dataset: {pop(n)}"
                     + (', synth: "${paths.popout_dyn}/synth_hist_popout.parquet"' if hard else "") + "}}")
    for n, s in sorted(sr.items()):
        keep = f"${{paths.mix_lwf_r2}}/{n}.parquet" if n.endswith(("_onset", "_rest")) else f"${{paths.mix}}/{n}.parquet"
        lines.append(f'      {n}:\n        {{share: {s:.6f}, loss: gt, new: true, hist: "${{paths.hist_real_synth}}", dataset: {{root: "${{paths.r2_root}}", '
                     f'split: train, index: _dyn_relabel/dyn_index.parquet, keep: "{keep}", synth: "${{paths.synth_dyn}}"}}}}')
    print("\n".join(lines))
    if a.write_config:
        Path(a.write_config).write_text("# @package _global_\n# dyn-stop sources of student_lwf (pop-out + R2): scripts/lwf_mix.py "
                                        + " ".join(sys.argv[1:]).replace(str(Path.home()), "~") + "\ndata:\n  train_dataset:\n    sources:\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
