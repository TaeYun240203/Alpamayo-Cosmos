"""R2 fine-tune metrics as CSVs and figures (user 10/27: every 10 steps the per-source / per-loss values and the grad norm,
every 500 steps the 4 validation sets x ADE, FDE, lateral, longitudinal error; synced to B2 runs/r2ft/metrics/).
Reads a run dir (loss_components.jsonl, ft_select.json, ft_val_rows.jsonl) and writes <run>/metrics/:
  train.csv   one row per log line (every logging_steps): loss, grad_norm, learning_rate, s_per_step, each loss component
              (all ranks) and "<source>/<loss>" per mixture source
  eval.csv    HF eval lines (old val 512, eval_<loss>)
  val.csv     one row per (measurement, set): ade, fde, lat, lon, n, ade_vs_step0; the sets include FtSelect's stop_ref
              (general_wo_relabel: general without the start-relabelled rows, user 10/28)
  *.png       train_losses, train_sources, grad_lr, val (matplotlib; skipped with a note when it is missing)
Usage (cwd train/): python scripts/ft_metrics.py <run dir>
Run by the metrics sync loop on the box (METRICS_HOOK) and at the end of the ft stage. Never fails the caller on a plot."""
import argparse, csv, json, sys
from pathlib import Path

SETS = ("general", "dynamic", "static", "curve")


def write_csv(path, rows):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)


def val_rows(run):
    st = json.loads((run / "ft_select.json").read_text()) if (run / "ft_select.json").exists() else {}
    hist = st.get("history", [])
    out = []
    for h in sorted(hist, key=lambda h: h["step"]):
        for s, e in (h.get("metrics") or {k: {"ade": v} for k, v in h["ade"].items()}).items():   # seeded step 0: ADE only
            out.append({"step": h["step"], "tag": h.get("tag"), "set": s, "n": h.get("n", {}).get(s), **e})
    base = {r["set"]: r["ade"] for r in out if r["step"] == 0}
    for r in out:
        if base.get(r["set"]): r["ade_vs_step0"] = round(r["ade"] / base[r["set"]] - 1, 5)
    return out, (st.get("stop_ref", "general"), st.get("stop_rel_worse"), st.get("stop_metric", "ade"))   # rel None: off; user 10/28: lat


def plots(d, train, val, stop=("general", 0.10, "ade")):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        print("ft_metrics: no matplotlib, CSVs only"); return
    def series(rows, k): return [r["step"] for r in rows if r.get(k) not in (None, "")], [float(r[k]) for r in rows if r.get(k) not in (None, "")]
    losses = [k for k in ("fm", "traj_ce", "vit", "kv", "vkd", "lm") if any(k in r for r in train)]
    if train:
        fig, ax = plt.subplots(len(losses) + 1, 1, figsize=(10, 2.2 * (len(losses) + 1)), sharex=True)
        for a, k in zip(ax, ["loss"] + losses):
            a.plot(*series(train, k), lw=0.8); a.set_ylabel(k); a.grid(alpha=0.3)
        ax[-1].set_xlabel("step"); fig.tight_layout(); fig.savefig(d / "train_losses.png", dpi=110); plt.close(fig)
        srcs = sorted({k.split("/")[0] for r in train for k in r if "/" in k})
        fig, ax = plt.subplots(len(losses), 1, figsize=(10, 2.2 * len(losses)), sharex=True, squeeze=False)
        for a, k in zip(ax[:, 0], losses):
            for s in srcs:
                x, y = series(train, f"{s}/{k}")
                if x: a.plot(x, y, lw=0.8, label=s)
            a.set_ylabel(k); a.grid(alpha=0.3)
        ax[0, 0].legend(fontsize=7, ncol=4); ax[-1, 0].set_xlabel("step"); fig.tight_layout(); fig.savefig(d / "train_sources.png", dpi=110); plt.close(fig)
        fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        for a, k in zip(ax, ("grad_norm", "learning_rate", "s_per_step")):
            a.plot(*series(train, k), lw=0.8); a.set_ylabel(k); a.grid(alpha=0.3)
        ax[-1].set_xlabel("step"); fig.tight_layout(); fig.savefig(d / "grad_lr.png", dpi=110); plt.close(fig)
    if val:
        fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        for a, m in zip(ax.flat, ("ade", "fde", "lat", "lon")):
            for s in sorted({r["set"] for r in val}):
                x, y = series([r for r in val if r["set"] == s], m)
                if x: a.plot(x, y, "o-", ms=3, lw=1, label=s)
            if m == (stop[2] if len(stop) > 2 else "ade"):
                g = [r[m] for r in val if r["set"] == stop[0] and r["step"] == 0 and r.get(m) is not None]
                if g and stop[1] is not None: a.axhline((1 + stop[1]) * g[0], color="r", ls="--", lw=0.8, label=f"stop: {stop[0]} +{100 * stop[1]:.0f} %")
            a.set_title(m + " (m)"); a.grid(alpha=0.3)
        ax[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(d / "val.png", dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); a = ap.parse_args()
    run = Path(a.run); d = run / "metrics"; d.mkdir(exist_ok=True)
    lines = [json.loads(l) for l in open(run / "loss_components.jsonl")] if (run / "loss_components.jsonl").exists() else []
    train = [r for r in lines if not any(k.startswith("eval_") for k in r)]; ev = [r for r in lines if r not in train]
    val, stop = val_rows(run)
    for name, rows in (("train", train), ("eval", ev), ("val", val)):
        if rows: write_csv(d / f"{name}.csv", rows)
    try:
        plots(d, train, val, stop)
    except Exception as e:                                                        # a figure never stops the sync or the stage
        print(f"ft_metrics: plots failed: {e!r}", file=sys.stderr)
    print(f"ft_metrics: {len(train)} train lines, {len(ev)} eval lines, {len(val)} val rows -> {d}")


if __name__ == "__main__":
    main()
