"""Ring-entry labels per dagger_label.py setting (reports/ring_dagger_plan_20261029.md 3): relabelled 10/28 runs, rows at
route s 860-890 with the car's front outside the ring's band at t0.
  go          the label's rollout enters the band within 6.4 s (else it yields outside it)
  t_arr       the nearest upstream moving ring car's time to the band entry angle (-75 deg), every GT car (seen or not)
  hit / hit_v the go label meets a ring car (any GT car, boxes at the same time, 0.5 m pad) / and the label is valid
  cut         the go label has a ring car reach its box within 1 s while it merges (gap 1 s)
  t_go        first time (s) the label's speed passes 1 m/s (standing rows: does it wait for the car first?)
Usage: python ring_relabel_compare.py SWEEP REC OUT   (SWEEP/<setting>/<run>/spool/<actor>/epK, REC/<run>/rec_w1; needs pandas)
       python ring_relabel_compare.py --plot OUT          (OUT/relabel_summary.json -> fig_e_relabel.png; needs matplotlib)
"""
import glob, json, math, sys
from pathlib import Path

import numpy as np

E2E = Path(__file__).resolve().parents[3] / "MORAI-eval-ring-gap/src/e2e_driver/tools"
sys.path.insert(0, str(E2E)); import dagger_label as L

TH_E = math.radians(-75.0)                                      # base_link where the car's front reaches the band (run f)


def t_arr(objs):
    cx, cy = L.RING_C; best = 99.0
    for o in L.ring_cars(objs):
        r = math.hypot(o[2] - cx, o[3] - cy); w = ((o[2] - cx) * o[6] - (o[3] - cy) * o[5]) / (r * r)
        if w <= 0: continue
        d = ((TH_E - math.atan2(o[3] - cy, o[2] - cx)) % (2 * math.pi)) * r
        best = min(best, 0.0 if d > 2 * math.pi * r - 6 else d / (w * r))   # just past the entry: there now
    return best


def rows(sweep, rec_root):
    import pandas as pd
    recs, out = {}, []
    for lab in sorted(glob.glob(f"{sweep}/*/*/spool/*/ep*/dagger_labels.npz")):
        ep = Path(lab).parent; cfg, run = ep.parts[-5], ep.parts[-4]
        if run not in recs: recs[run] = L.Rec(L.load_rec(f"{rec_root}/{run}/rec_w1", 0, 4e9))
        Z, D = np.load(lab), np.load(ep / "dagger_debug.npz", allow_pickle=True)
        for j, st in enumerate(json.loads(l) for l in open(ep / "steps.jsonl")):
            ro = D["rollout"][j]
            if not (860 <= st["s"] <= 890) or not D["rolled"][j] or not L.outside_band(ro[:1]): continue
            objs = recs[run].objects(D["t_wall"][j]); go = not L.outside_band(ro); v = ro[:, 3]
            hit = go and L.ring_meet(ro, objs) is not None
            out.append(dict(cfg=cfg, who="expert" if "drive" in ep.parent.name else "policy", s=st["s"], v0=st["v"], v3=v[29],
                            vmax=v.max(), go=go, valid=bool(Z["valid"][j].all()), mode=str(D["mode"][j]), t_arr=t_arr(objs),
                            hit=hit, hit_v=hit and bool(Z["valid"][j].all()), cut=go and L.ring_meet(ro, objs, gap_s=1.0) is not None,
                            t_go=0.1 * (int(np.argmax(v > 1.0)) + 1) if (v > 1.0).any() else np.inf))
    return pd.DataFrame(out)


def plot(out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    j = json.loads((out / "relabel_summary.json").read_text()); cfgs = list(j["summary"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for c in cfgs: ax[0].plot(j["bins"], [j["standing_go"][c].get(b, np.nan) for b in j["bins"]], marker="o", label=c)
    ax[0].set(xlabel="nearest upstream ring car: time to the entry (s, every GT car)", ylabel="share of go labels",
              title="standing at the entry line (v0 < 0.3 m/s)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    x = np.arange(len(cfgs))
    ax[1].bar(x - .2, [j["summary"][c]["hit"] for c in cfgs], .4, label="go label meets a ring car (any)")
    ax[1].bar(x + .2, [j["summary"][c]["hit_valid"] for c in cfgs], .4, label="... and the label is valid (trained)")
    ax[1].set_xticks(x, cfgs, rotation=20, fontsize=8); ax[1].set(title="go labels into ring cars (rows)"); ax[1].legend(fontsize=8)
    fig.suptitle("Ring-entry shadow labels per setting: 10/28 runs d, f relabelled (route s 860-890, outside the band at t0)")
    fig.tight_layout(); fig.savefig(out / "fig_e_relabel.png", dpi=110)


def main():
    if sys.argv[1] == "--plot":
        return plot(Path(sys.argv[2]))
    import pandas as pd
    sweep, rec_root, out = sys.argv[1:4]; out = Path(out)
    df = rows(sweep, rec_root); df.to_csv(out / "relabel_rows.csv", index=False)
    bins = ["<2", "2-3", "3-4", "4-5", "5-6", "6-8", ">8"]
    df["tb"] = pd.cut(df.t_arr, [-1, 2, 3, 4, 5, 6, 8, 99], labels=bins)
    st = df[df.v0 < 0.3]
    summ = df.groupby("cfg").agg(rows=("s", "size"), go=("go", "mean"), valid=("valid", "mean"), go_n=("go", "sum"), hit=("hit", "sum"),
                                 hit_valid=("hit_v", "sum"), cut_1s=("cut", "sum"))
    summ["go_v3_ms"] = df[df.go].groupby("cfg").v3.median()
    piv = st.pivot_table(index="tb", columns="cfg", values="go", aggfunc="mean", observed=True)
    n = st[st.cfg == st.cfg.iloc[0]].groupby("tb", observed=True).size()
    tgo = st[st.go].pivot_table(index="tb", columns="cfg", values="t_go", aggfunc="median", observed=True)
    print(summ.round(3).to_string()); print("\nstanding rows (v0 < 0.3): go share by t_arr"); print(piv.assign(rows=n).round(2).to_string())
    print("\nstanding go labels: median start time (s, speed > 1 m/s) by t_arr"); print(tgo.round(1).to_string())
    (out / "relabel_summary.json").write_text(json.dumps(dict(
        bins=bins, summary={c: {k: float(v) for k, v in r.items()} for c, r in summ.iterrows()},
        standing_go={c: {str(b): float(v) for b, v in piv[c].items()} for c in piv.columns}, standing_rows={str(b): int(v) for b, v in n.items()},
        standing_t_go={c: {str(b): float(v) for b, v in tgo[c].items()} for c in tgo.columns}), indent=1))


if __name__ == "__main__":
    main()
