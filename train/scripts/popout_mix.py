"""Near pop-out pool (step0_popout, reports/popout_collect_plan_20261029.md) for the dyn fine-tune: merge / stage.
merge S0 (Mac, after the boxes): every batch's _popout_relabel/<run>/ (popout_relabel.py on the box) -> _dyn_relabel/
  dyn_index.parquet (split: 5 % of the recorded episodes -> val, crc32 of the episode), dyn_future.parquet, keep lists
  mix/popout_<zone>_<start>_<kind>.parquet (train) and mix/popout_val.parquet (val rows), synth_hist_popout.parquet
  (speed-preserving lane histories of the stop rows, dyn_synth_hist.rows), summary.json + the student_dyn.yaml sources
  block (shares: natural row counts, zones >= 40 % and visible / hidden >= 35 % each, sum SHARE).
  S0 needs _popout_relabel/*/*.parquet and _ego/<recorded>.npz (the B2 copies; no sensors).
merge S0 OUT RUNS (10/29 v2, scripts/popout_rebuild.sh): the runs from RUNS/*/, out to OUT (B2 _dyn_relabel_v2/), rows sorted
  (the same bytes for any chunking), plus ego/<virtual>.npz (the tables themselves), popout_index.parquet (every episode's own
  rows: build_index columns, sensors and collision cut as clean, no traj_ok: the route tail is the future),
  mix/popout_cruise{,_val}.parquet (original rows <= first stop row - 0.5 s; an episode without a stop row only without a
  scorer collision; label = the original drive) and mix/popout_val.parquet (episode, i, kind: onset = val virtual rows <
  first stop row + 1 s, rest, pre_onset = original rows of popout_index from first stop row - 1.5 s to - 0.5 s).
stage S0 (training box, after the B2 download): _dyn_relabel/ego/<virtual>.npz -> links to _popout_relabel/<run>/ego/ (tables
  already in _dyn_relabel/ego/, the 10/29 rebuild, stay as they are), then dyn_stage.py links the virtual episodes' sensors
  and _ego tables (loader unchanged).
"""
import glob, json, shutil, sys, zlib
from pathlib import Path

import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
SHARE, VAL = 0.119, 0.05
ZONE = {"ra": "dyn", "rb": "tunnel"}


def balance(n, share=SHARE):
    """cell -> share: natural counts, raised so each zone >= 40 % and visible (near+far) / hidden >= 35 %."""
    w = {k: float(v) for k, v in n.items() if v > 0}
    for _ in range(50):
        moved = False
        for lim, groups in ((0.4, [lambda k: k[0] == "ra", lambda k: k[0] == "rb"]), (0.35, [lambda k: k[1] != "hidden", lambda k: k[1] == "hidden"])):
            tot = sum(w.values())
            for f in groups:
                part = sum(v for k, v in w.items() if f(k))
                if 0 < part < lim * tot - 1e-9:
                    rest = tot - part; scale = lim * rest / ((1 - lim) * part)
                    for k in w:
                        if f(k): w[k] *= scale
                    moved = True
        if not moved: break
    tot = sum(w.values())
    return {k: share * v / tot for k, v in w.items()}


def clean(s0, vi, keep, eps=()):
    """main 10/29 review: no row at or after the scorer's first collision (the RDDF Ego drove into the mover; hold rows were
    the stopped car after it), no episode whose mover never spawned, only rows MoraiDataset.usable passes (frame gaps).
    Needs _popout/<ep>/labels.parquet + episode.json and <ep>/cams_index.npz + lidar_index.npz. eps: more episodes to judge
    (the original rows of popout_index). Returns vi, keep, dropped and the row test ok(ep, i)."""
    import torch  # noqa: F401  (morai_vla.data)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from morai_vla.data import DT as FDT, MAX_FRAME_GAP, MAX_SCAN_GAP, N_FRAMES, nearest, select_distinct
    cut, bad = {}, set()
    no_sensors = 0
    for ep in sorted(set(vi.src_episode) | set(eps)):
        if not all((s0 / ep / f).exists() for f in ("cams_index.npz", "lidar_index.npz")) or not (s0 / "_popout" / ep / "labels.parquet").exists():
            bad.add(ep); no_sensors += 1; continue             # 10/29: boxes destroyed mid-upload left episodes without sensors on B2
        z = np.load(s0 / "_ego" / f"{ep}.npz"); L = pd.read_parquet(s0 / "_popout" / ep / "labels.parquet")
        rec = json.load(open(s0 / "_popout" / ep / "episode.json"))
        zone = "ra" if "ra" in ep.split("-")[0][-4:] else "rb"   # a scene holds both zones' movers; only this segment's counts
        mv = [a for a in rec["scene_meta"]["actors"] if a.get("ptype") == "popout"]
        if not all(f for a, f in zip(mv, rec.get("mover_in_objinfo", [True] * len(mv))) if a["zone"].startswith(zone)): bad.add(ep)
        if rec.get("t_sim", 99.0) < 2.0: bad.add(ep)                 # main 10/29: boxes that hit something 1 s after every spawn
        from popout_relabel import collision_ns
        ns = collision_ns(L, rec)
        cut[ep] = float(np.interp(ns, z["bag_ns"], z["tau"])) if ns is not None else np.inf
        ci, li = np.load(s0 / ep / "cams_index.npz"), np.load(s0 / ep / "lidar_index.npz")
        cut[ep, "cams"] = [np.sort(ci["tau"][ci["cam"] == c].astype(float)) for c in range(4)]
        cut[ep, "scan"] = np.sort(li["tau"].astype(float)); cut[ep, "tau"] = z["tau"]

    def ok(ep, i):
        if ep in bad or cut[ep, "tau"][i] >= cut[ep]: return False
        t0 = cut[ep, "tau"][[i]][:, None] - FDT * np.arange(N_FRAMES)[None, ::-1]
        for tau in cut[ep, "cams"]:
            k = select_distinct(tau, t0)
            if len(tau) < 8 or not ((k[:, 0] >= 0) & (np.abs(tau[k] - t0) <= MAX_FRAME_GAP).all(1))[0]: return False
        st = cut[ep, "scan"]; k = nearest(st, t0[:, -1])
        return len(st) >= 8 and bool((np.abs(st[k] - t0[:, -1]) <= MAX_SCAN_GAP)[0] and k[0] >= N_FRAMES - 1)
    m = np.array([ok(e, i) for e, i in zip(vi.src_episode, vi.i)])
    km = np.array([e in cut and ok(e, i) for e, i in zip(keep.episode, keep.i)]) if len(keep) else np.zeros(0, bool)
    dropped = dict(rows=int((~m).sum()), no_mover_episodes=len(bad),
                   after_collision=int(sum(cut[e, "tau"][i] >= cut[e] for e, i in zip(vi.src_episode, vi.i) if e not in bad)), no_sensors_episodes=no_sensors)
    print("clean:", dropped, flush=True)
    return vi[m].reset_index(drop=True), keep[km] if len(keep) else keep, dropped, ok


def merge(s0, out=None, runs=None):
    d = Path(out) if out else s0 / "_dyn_relabel"; (d / "mix").mkdir(parents=True, exist_ok=True)
    runs = sorted(Path(p).parent for p in glob.glob(str(Path(runs) if runs else s0 / "_popout_relabel") + "/*/dyn_index.parquet"))
    runs = [r for r in runs if (r / "ego").is_dir()]   # 10/29: a batch whose ego tables never reached B2 (box destroyed mid-upload) has no usable rows
    print("merge: runs with ego tables", len(runs), flush=True)
    cat = lambda f: pd.concat([pd.read_parquet(r / f) for r in runs if (r / f).exists()], ignore_index=True)
    vi, fut, lab = cat("dyn_index.parquet"), cat("dyn_future.parquet"), cat("dyn_relabel_index.parquet")
    keep = cat("keep.parquet") if any((r / "keep.parquet").exists() for r in runs) else pd.DataFrame(columns=["episode", "i"])
    first = {ep: e.get("first_stop_i", -1) for r in runs for ep, e in json.load(open(r / "summary.json"))["episodes"].items()}
    eps_all = sorted(e for e in first if (s0 / e / "cams_index.npz").exists())
    vi, keep, dropped, ok = clean(s0, vi, keep, eps_all)
    if "kind" in fut:                                                  # self-check: stop labels brake at most A_MAX (6.96) per step
        f = fut[fut.kind != "restart_hold"].merge(lab[["episode", "i", "kind", "v0"]], on=["episode", "i", "kind"])
        bad = {(r.episode, r.i) for r in f.itertuples() if np.min(np.diff(np.r_[r.v0, r.fut_v])) / 0.1 < -6.97}
        m = np.array([not (k != "restart_hold" and (e, i) in bad) for e, i, k in zip(vi.src_episode, vi.i, vi.kind)])
        dropped["decel_over_a_max"] = int((~m).sum()); vi = vi[m].reset_index(drop=True)
    vi = vi.sort_values(["src_episode", "i", "episode"]).reset_index(drop=True)   # 10/29: same bytes whatever the chunking (-j)
    lab = lab.sort_values(["episode", "i", "kind"]).reset_index(drop=True); keep = keep.sort_values(["episode", "i"]).reset_index(drop=True)
    eps = sorted(vi.src_episode.unique())
    is_val = lambda e: zlib.crc32(e.encode()) % 1000 < VAL * 1000
    val_eps = {e for e in eps if is_val(e)}
    vi["split"] = np.where(vi.src_episode.isin(val_eps), "val", "train")
    vi["zone"] = vi.src_episode.str.extract(r"(r[ab])\d\d-w1-")[0]
    vi["kind2"] = np.where(vi.kind == "restart_hold", "rh", "hard")
    vi.to_parquet(d / "dyn_index.parquet", index=False)
    fut = fut[fut.kind != "restart_hold"] if "kind" in fut else fut     # dyn_load_check keys (src, i): the stop rows' futures
    fut.drop_duplicates(["episode", "i"]).sort_values(["episode", "i"]).to_parquet(d / "dyn_future.parquet", index=False)
    lab.to_parquet(d / "dyn_relabel_index.parquet", index=False)
    keep[~keep.episode.isin(val_eps)].to_parquet(d / "mix/popout_replay_keep.parquet", index=False)
    (d / "ego").mkdir(exist_ok=True); tables = {Path(f).stem: f for r in runs for f in glob.glob(str(r / "ego/*.npz"))}
    for e in vi.episode: shutil.copy2(tables[e], d / "ego" / f"{e}.npz")   # 10/29 v2: the tables live here (no stage links)
    # original rows (main 10/29 "popout_cruise"): every episode's own rows (build_index columns from its _ego row, no traj_ok:
    # the route tail is the future past the recording / the collision), sensors as clean, no build_index speed step in the window
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from morai_vla.build_index import FUT, HIST, data, glitch_steps, maneuvers, nav_text   # noqa: E402
    M, lap, exempt, cells = maneuvers(), float(data.path_xy()[1][-1]), data.kmap().speed_exempt_s, data.ROUTE["cells"]
    rows = []
    for ep in eps_all:
        z = np.load(s0 / "_ego" / f"{ep}.npz"); n = len(z["tau"]); sc = json.load(open(s0 / "_popout" / ep / "episode.json"))["scene"]
        jump = np.r_[0, np.cumsum(glitch_steps(z["v"], z["tau"]))]
        for i in range(HIST - 1, int(z["ext"])):
            if i + FUT >= n or jump[i + FUT] - jump[i - HIST + 1] > 0 or not ok(ep, i): continue
            rows.append((ep, i, float(z["tau"][i]), cells.get(int(z["section"][i]), "C?"), "val" if is_val(ep) else "train",
                         nav_text(float(z["path_s"][i]), M, lap, exempt), bool(z["fog"][i]), int(z["hour"][i]), sc))
    oi = pd.DataFrame(rows, columns=["episode", "i", "tau", "cell", "split", "nav_text", "fog", "hour", "scene"])
    oi.to_parquet(d / "popout_index.parquet", index=False)
    # cruise (label = the original drive): rows up to 0.5 s before the episode's first stop row; an episode without one only
    # when the scorer saw no collision (the mover never came into the path); val episodes apart
    from popout_relabel import collision_ns                           # noqa: E402
    collided = {e for e in eps_all if collision_ns(pd.read_parquet(s0 / "_popout" / e / "labels.parquet"), json.load(open(s0 / "_popout" / e / "episode.json"))) is not None}
    lim = oi.episode.map(lambda e: first[e] - 5 if first[e] is not None and first[e] >= 0 else (-1 if e in collided or first[e] == -1 else 10 ** 9))
    cr = oi[oi.i <= lim]
    cr[cr.split == "train"][["episode", "i"]].to_parquet(d / "mix/popout_cruise.parquet", index=False)
    cr[cr.split == "val"][["episode", "i"]].to_parquet(d / "mix/popout_cruise_val.parquet", index=False)
    # val (3rd session's onset contrast): every val virtual row, kind onset (< first stop row + 1 s) / rest, plus the original
    # rows [first stop row - 1.5 s, - 0.5 s] as pre_onset (names of popout_index.parquet, not dyn_index)
    vv = vi[vi.split == "val"]; f0 = vv.src_episode.map(first)
    po = oi[(oi.split == "val") & oi.episode.map(lambda e: first[e] is not None and first[e] >= 0)]
    po = po[(po.i >= po.episode.map(first) - 15) & (po.i <= po.episode.map(first) - 5)]
    val = pd.concat([vv[["episode", "i"]].assign(kind=np.where(vv.i < f0 + 10, "onset", "rest")), po[["episode", "i"]].assign(kind="pre_onset")], ignore_index=True)
    val.to_parquet(d / "mix/popout_val.parquet", index=False)
    tr = vi[vi.split == "train"]
    cells = {}
    for (z, st, k), g in tr.groupby(["zone", "start", "kind2"]):
        g[["episode", "i"]].to_parquet(d / "mix" / f"popout_{z}_{st}_{k}.parquet", index=False); cells[(z, st, k)] = len(g)
    from dyn_synth_hist import rows                                  # noqa: E402
    from ft_synth_hist import path_arrays                            # noqa: E402
    P, S, H = path_arrays(); hard = tr[tr.kind2 == "hard"]
    res = []
    for ep, g in hard.groupby("src_episode"):
        names = dict(zip(g.i, g.episode)); res += rows(s0, {ep: list(g.i)}, P, S, H, name=lambda e, i, m=names: m[i])
    pd.DataFrame(res).to_parquet(d / "synth_hist_popout.parquet", index=False)
    sh = balance(cells)
    lines = ["      # near pop-outs (step0_popout, reports/popout_collect_plan_20261029.md): natural shares, zones >= 40 %, visible / hidden >= 35 %"]
    for (z, st, k), s in sorted(sh.items()):
        name = f"popout_{ZONE[z]}_{st}_{k}"
        hist = ' hist: "${paths.hist_real_synth}",' if k == "hard" else ""
        synth = f', synth: "${{paths.popout_root}}/{d.name}/synth_hist_popout.parquet"' if k == "hard" else ""
        lines.append(f"      {name}:\n        {{share: {s:.6f}, loss: gt, new: true,{hist} dataset: {{root: \"${{paths.popout_root}}\", split: train, "
                     f"index: {d.name}/dyn_index.parquet, keep: \"${{paths.popout_root}}/{d.name}/mix/popout_{z}_{st}_{k}.parquet\"{synth}}}}}")
    yaml = "\n".join(lines)
    (d / "sources_block.yaml").write_text(yaml + "\n")
    n_zone = tr.groupby("zone").size().to_dict(); n_vis = tr.assign(v=np.where(tr.start == "hidden", "hidden", "visible")).groupby("v").size().to_dict()
    dec = tr.i < tr.src_episode.map(first) + 10
    summ = dict(dropped=dropped, episodes=len(eps), val_episodes=len(val_eps), train_rows=len(tr), val_rows=int((vi.split == "val").sum()),
                cells={"_".join(k): v for k, v in sorted(cells.items())}, shares={"_".join(k): round(v, 6) for k, v in sorted(sh.items())},
                zone_rows=n_zone, visibility_rows=n_vis, kinds=lab.kind.value_counts().sort_index().to_dict(), synth_hist=len(res),
                onset_rows_train=int(dec.sum()), onset_cells={"_".join(k): int(n) for k, n in tr[dec].groupby(["zone", "start", "kind2"]).size().items()}, popout_index_rows=len(oi), popout_index_episodes=int(oi.episode.nunique()),
                cruise=dict(train_10hz=int((cr.split == "train").sum()), train_2hz=int(((cr.split == "train") & (cr.i % 5 == 0)).sum()),
                            val_10hz=int((cr.split == "val").sum()), val_2hz=int(((cr.split == "val") & (cr.i % 5 == 0)).sum()),
                            episodes=int(cr.episode.nunique()), no_event_episodes=int(cr[cr.episode.map(lambda e: first[e] is None)].episode.nunique())),
                val_kinds=val.kind.value_counts().sort_index().to_dict())
    json.dump(summ, open(d / "summary.json", "w"), indent=1, sort_keys=True)
    print(json.dumps(summ, indent=1)); print(yaml)


def stage(s0):
    d = s0 / "_dyn_relabel" / "ego"; d.mkdir(parents=True, exist_ok=True); n = 0
    for f in glob.glob(str(s0 / "_popout_relabel/*/ego/*.npz")):
        t = d / Path(f).name
        if not t.is_symlink() and not t.exists():               # 10/29 rebuild: B2 _dyn_relabel/ego/ holds the tables itself
            t.symlink_to(Path("../../_popout_relabel") / Path(f).parent.parent.name / "ego" / Path(f).name); n += 1
    print(f"popout_mix stage: {n} virtual ego tables linked")


if __name__ == "__main__":
    {"merge": merge, "stage": stage}[sys.argv[1]](*map(Path, sys.argv[2:]))
