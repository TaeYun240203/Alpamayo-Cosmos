"""Does a student still see the red light? Linear probe of its LLM features, red vs green (10/28, after the R2 fine-tune
stopped holding at red lights; user: "신호 인식 확인은 특징 프로브로 해"). Three steps:
  select  (Mac) R1 val + val_asset rows whose next judged stop line is 10-60 m ahead (signal_eval.candidates) with the GT
          light red (1) or green (3) at t0, in --n-eps episodes that have both; rows >= --gap grid rows (1 s) apart within an episode; red and
          green matched per (stop line, 10 m distance bin, speed bin), since red rows brake or stand and the history
          tokens carry the speed. --stationary instead: stopped rows only, red and green paired at the same stop event
          (stationary()), so speed, history and scene motion cannot tell them apart; --eps limits the episodes (e.g. those
          already downloaded). Writes rows.parquet (episode, i, stop, dist, light, fog, v0, a0, v_gt64, v_m05/10/15).
  run     (GPU) one model (ft_measure.BF16 load = the gate's: a QAD delivery comes up fake-quantized): the deployment
          prefix (no CoC) and Euler 10 from seed = row position; forward hooks on LLM layers 0, 4, ..., 32 and the last keep
          the prefill's hidden state at the last token (<traj_future_start>) and its mean over the prefix; the plan's speed
          at 6.4 s. Writes feats_<tag>.npz (key, layers, last / mean (N, L, d) fp16, v_end).
  probe   (any) per model x layer x pooling: L2 logistic regression red vs green, 5 folds by episode (no episode in both
          train and test), the L2 weight chosen by inner folds by episode; accuracy over all rows. Baseline: the same probe
          on what the prefix tells about the motion: speeds at t0, -0.5, -1, -1.5 s and the distance. Stop rate (plan speed at 6.4 s <= 3 m/s) per model x light, and the GT's.
Usage (cwd train/):
  python scripts/signal_probe.py select --root <R1 dir with _index/ and _ego/> --out rows.parquet [--n 200 --n-eps 40]
  python scripts/signal_probe.py run --root <R1 step0> --rows rows.parquet --ckpt <student dir> --tag s2 --out <dir>
  python scripts/signal_probe.py probe --rows rows.parquet --feats <dir>/feats_s2.npz <dir>/feats_deliver.npz --out <dir>"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
LIGHT = {1: "red", 3: "green"}
DBINS, VBINS = [10, 20, 30, 40, 50, 60], [-1, 2, 6, 10, 99]
STOP_V, LAMS, FOLDS = 3.0, (1e-2, 1e-1, 1.0, 10.0), 5          # 1e-3 takes ~1000x longer on separable d >> n data


def select(a):
    from scripts.signal_eval import CTX, candidates
    c = candidates(a.root, np.load(CTX), stride=1)
    c = c[(c.dist >= 10) & (c.dist <= 60) & c.light.isin(list(LIGHT))].reset_index(drop=True)
    ego = {e: np.load(Path(a.root) / "_ego" / f"{e}.npz") for e in c.episode.unique()}
    c["v0"] = [float(ego[e]["v"][i]) for e, i in zip(c.episode, c.i)]; c["a0"] = [float(ego[e]["a"][i]) for e, i in zip(c.episode, c.i)]
    c["v_gt64"] = [float(ego[e]["v"][min(i + 64, len(ego[e]["v"]) - 1)]) for e, i in zip(c.episode, c.i)]
    # history speeds for the probe baseline (a0 is smoothed over +-0.2 s: the model never sees it)
    for d in (5, 10, 15): c[f"v_m{d:02d}"] = [float(ego[e]["v"][max(i - d, 0)]) for e, i in zip(c.episode, c.i)]
    if a.eps: c = c[c.episode.isin(open(a.eps).read().split())]
    if a.stationary: return stationary(a, c, ego)
    both = c.groupby("episode").light.nunique().pipe(lambda s: s[s == 2].index.to_numpy())
    rng = np.random.default_rng(0); eps = set(rng.choice(sorted(both), min(a.n_eps, len(both)), replace=False))
    c = c[c.episode.isin(eps)].sample(frac=1, random_state=0)
    keep, last = [], {}                                                          # rows >= --gap grid rows apart within an episode
    for k, (e, i) in enumerate(zip(c.episode, c.i)):
        if all(abs(i - j) >= a.gap for j in last.get(e, [])): keep.append(k); last.setdefault(e, []).append(i)
    c = c.iloc[keep].assign(cell=lambda d: list(zip(d.stop, pd.cut(d.dist, DBINS, include_lowest=True, labels=False), pd.cut(d.v0, VBINS, labels=False))))
    pairs = [(g[g.light == 1], g[g.light == 3]) for _, g in c.groupby("cell")]
    m = np.array([min(len(r), len(g)) for r, g in pairs]); take = np.floor(m * min(1.0, a.n / 2 / max(m.sum(), 1))).astype(int)
    sel = pd.concat([pd.concat([r.iloc[:t], g.iloc[:t]]) for (r, g), t in zip(pairs, take) if t])
    write_sel(a, sel.drop(columns="cell"), len(c))


def stationary(a, c, ego):
    """stopped rows only (speed < 0.3 m/s over the whole 1.5 s history: static history tokens and camera frames), paired per
    green onset at a stop line: up to 2 green rows in the first second after it (the car still waiting) and as many red rows
    1-3 s before it, same place and distance; only the light differs."""
    c = c[[float(ego[e]["v"][max(i - 15, 0):i + 1].max()) < 0.3 for e, i in zip(c.episode, c.i)]]
    keep = []
    for _, g in c.groupby(["episode", "stop"]):
        gi, ri = np.sort(g.i[g.light == 3].to_numpy()), np.sort(g.i[g.light == 1].to_numpy())
        for on in gi[np.r_[True, np.diff(gi) > 10]] if len(gi) else []:
            gr, rr = gi[(gi >= on) & (gi < on + 10)][::3][:2], ri[(ri <= on - 10) & (ri >= on - 30)][::10][:2]
            k = min(len(gr), len(rr)); keep += [(g.episode.iloc[0], int(i)) for i in [*gr[:k], *rr[:k]]]
    sel = c.merge(pd.DataFrame(keep, columns=["episode", "i"]), on=["episode", "i"]).drop_duplicates(["episode", "i"])
    write_sel(a, sel, len(c))


def write_sel(a, sel, n_cand):
    sel = sel.sort_values(["episode", "i"]).reset_index(drop=True); sel.to_parquet(a.out, index=False)
    print(json.dumps({"candidates_in_eps": n_cand, "episodes": int(sel.episode.nunique()), "rows": sel.light.map(LIGHT).value_counts().to_dict(),
                      "dist_mean": sel.groupby("light").dist.mean().round(1).to_dict(), "v0_mean": sel.groupby("light").v0.mean().round(2).to_dict(),
                      "gt_stop_rate": sel.assign(s=sel.v_gt64 <= STOP_V).groupby("light").s.mean().round(3).to_dict(),
                      "fog": sel.groupby("light").fog.mean().round(2).to_dict()}, indent=1))


def run(a):
    import torch
    from morai_vla.data import MoraiDataset
    from scripts.ft_measure import BF16
    rows = pd.read_parquet(a.rows).astype({"episode": str})
    ds = MoraiDataset(a.root, ["val", "val_asset"], frame_stride=1, augment=False, seed=1, keep=a.rows)
    ds.rows = ds.rows.astype({"episode": str}).sort_values(["episode", "i"]).reset_index(drop=True)
    model = BF16(a.ckpt, a.device); layers = model.m.vlm.model.language_model.layers
    L = sorted({*range(0, len(layers), 4), len(layers) - 1}); cap = {}
    def hook(j):
        def f(_, __, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1: cap[j] = (h[0, -1].float().cpu().numpy(), h[0].float().mean(0).cpu().numpy())   # prefill only
        return f
    for j in L: layers[j].register_forward_hook(hook(j))
    last, mean, v_end, keys = [], [], [], []; t0 = time.time()
    for k in range(len(ds.rows)):
        s = ds.sample(k); cap.clear()
        xy = model(s, (s["ego_history_xyz"], s["ego_history_rot"]), 1000 * k)
        last.append(np.stack([cap[j][0] for j in L])); mean.append(np.stack([cap[j][1] for j in L]))
        v_end.append(float(np.linalg.norm(xy[-1] - xy[-2]) / 0.1)); keys.append(s["key"])
        if k % 20 == 0: print(f"{k + 1}/{len(ds.rows)} {s['key']} {(time.time() - t0) / (k + 1):.2f} s/row", flush=True)
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    np.savez(o / f"feats_{a.tag}.npz", key=np.array(keys), layers=np.array(L), last=np.stack(last).astype(np.float16),
             mean=np.stack(mean).astype(np.float16), v_end=np.array(v_end, np.float32))
    print(json.dumps({"tag": a.tag, "ckpt": a.ckpt, "rows": len(keys), "of": len(rows), "s_per_row": round((time.time() - t0) / max(len(keys), 1), 3)}))


def fit(X, y, lam):
    """standardized L2 logistic regression (full-batch LBFGS) -> predict function."""
    import torch
    mu, sd = X.mean(0), X.std(0) + 1e-6; Z = torch.tensor((X - mu) / sd, dtype=torch.float32); t = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(Z.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(Z @ w + b, t) + lam * (w ** 2).sum(); loss.backward(); return loss
    opt.step(closure); w, b = w.detach().numpy(), float(b.detach())
    return lambda Xq: ((Xq - mu) / sd) @ w + b > 0


def folds_of(groups, n, seed):
    u = np.random.default_rng(seed).permutation(sorted(set(groups)))
    f = {g: k % n for k, g in enumerate(u)}; return np.array([f[g] for g in groups])


def cv_acc(X, y, groups):
    """accuracy over all rows, 5 folds by episode; the L2 weight from inner folds by episode on each training part."""
    fo, right = folds_of(groups, FOLDS, 0), np.zeros(len(y), bool)
    for f in range(FOLDS):
        tr, te = fo != f, fo == f
        fi = folds_of(groups[tr], FOLDS - 1, 1)
        inner = lambda lam: np.mean([(fit(X[tr][fi != g], y[tr][fi != g], lam)(X[tr][fi == g]) == y[tr][fi == g]).mean() for g in range(FOLDS - 1)])
        right[te] = fit(X[tr], y[tr], max(LAMS, key=inner))(X[te]) == y[te]
    return round(float(right.mean()), 3)


def probe(a):
    rows = pd.read_parquet(a.rows).astype({"episode": str}); rows["key"] = rows.episode + "/" + rows.i.astype(str)
    out = {"baseline_hist_speed_dist": None, "models": {}}; base = ["v0", "v_m05", "v_m10", "v_m15", "dist"]
    for f in a.feats:
        z = np.load(f); tag = Path(f).stem.removeprefix("feats_"); r = rows.set_index("key").loc[list(z["key"])]
        y, g = (r.light.to_numpy() == 1).astype(int), r.episode.to_numpy()
        if out["baseline_hist_speed_dist"] is None: out["baseline_hist_speed_dist"] = cv_acc(r[base].to_numpy(float), y, g)
        acc = {p: {int(l): cv_acc(z[p][:, k].astype(np.float32), y, g) for k, l in enumerate(z["layers"])} for p in ("last", "mean")}
        stop = pd.Series(z["v_end"] <= STOP_V).groupby(r.light.map(LIGHT).to_numpy()).mean().round(3).to_dict()
        out["models"][tag] = {"n": len(y), "red": int(y.sum()), "episodes": int(len(set(g))), "acc": acc, "stop_rate": stop}
        print(tag, json.dumps(out["models"][tag]), flush=True)
    out["gt_stop_rate"] = rows.assign(s=rows.v_gt64 <= STOP_V).groupby(rows.light.map(LIGHT)).s.mean().round(3).to_dict()
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True); (o / "probe.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for tag, m in out["models"].items():
        for p, ls in (("last", "-"), ("mean", "--")):
            ax.plot(list(m["acc"][p]), list(m["acc"][p].values()), ls, marker="o", ms=3, label=f"{tag} {p}")
    ax.axhline(out["baseline_hist_speed_dist"], color="gray", ls=":", label="baseline: history speeds, dist"); ax.axhline(0.5, color="k", lw=0.5)
    ax.set_xlabel("LLM layer"); ax.set_ylabel("red vs green accuracy (CV by episode)"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(o / "probe_acc.png", dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="what", required=True)
    p = sp.add_parser("select"); p.add_argument("--root", required=True); p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=200); p.add_argument("--n-eps", type=int, default=40); p.add_argument("--gap", type=int, default=10)
    p.add_argument("--stationary", action="store_true"); p.add_argument("--eps", help="file of episode names to draw from")
    p = sp.add_parser("run"); [p.add_argument(k, required=True) for k in ("--root", "--rows", "--ckpt", "--tag", "--out")]; p.add_argument("--device", default="cuda:0")
    p = sp.add_parser("probe"); p.add_argument("--rows", required=True); p.add_argument("--feats", nargs="+", required=True); p.add_argument("--out", required=True)
    a = ap.parse_args(); {"select": select, "run": run, "probe": probe}[a.what](a)


if __name__ == "__main__":
    main()
