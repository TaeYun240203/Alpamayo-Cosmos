"""python -m coc_labeler.cli --out DIR [--episodes a,b | --all] [--frames winpc|DIR] [--budget-pct 25] [--workers 8]
   python -m coc_labeler.cli summarize DIR | recompose SRC_DIR OUT_DIR"""
import argparse, glob, json, os, time
from collections import Counter
from pathlib import Path

from . import data, frames, pipeline, qa
from .vlm import BudgetExceeded, Codex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", help="comma-separated episode names")
    ap.add_argument("--all", action="store_true", help="every episode in the labels cache")
    ap.add_argument("--frames", default="winpc", help="'winpc' (extract remotely) or a folder with <ep>_<i>.bag")
    ap.add_argument("--budget-pct", type=float, default=None, help="stop once the weekly Codex window rises this much")
    ap.add_argument("--max-used-pct", type=float, default=None, help="absolute weekly-usage ceiling (shared by shards)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--qa-rate", type=float, default=0.1)
    ap.add_argument("--model", default="gpt-6.1-sol")
    ap.add_argument("--shard", default="0/1", help="k/n: this process labels episodes with index % n == k")
    ap.add_argument("--no-summary", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    eps = sorted(os.path.basename(f).split(".")[0] for f in glob.glob(str(data.LABELS / "*" / "*.labels.parquet"))) if a.all else a.episodes.split(",")
    k, n = map(int, a.shard.split("/")); eps = [e for i, e in enumerate(eps) if i % n == k]
    provider = frames.Winpc() if a.frames == "winpc" else frames.LocalBags(a.frames)
    vlm = Codex(model=a.model, budget_pct=a.budget_pct, max_used_pct=a.max_used_pct)
    done = {p.stem for p in (out / "labels").glob("*.jsonl")}
    log = open(out / f"run_{k}of{n}.log", "a"); t0 = time.time()
    for j, ep in enumerate(eps):
        if ep in done: continue
        try:
            recs = pipeline.label_episode(ep, provider, vlm, out, workers=a.workers, qa_rate=a.qa_rate)
        except BudgetExceeded as e:
            print("budget reached:", e, file=log, flush=True); break
        except Exception as e:                                   # one bad episode must not stop the run
            print(f"{ep} FAILED {type(e).__name__}: {e}", file=log, flush=True); continue
        c = Counter(r["status"] for r in recs)
        print(f"[{j + 1}/{len(eps)}] {ep} {dict(c)} calls={vlm.calls} used={vlm.used}% {time.time() - t0:.0f}s", file=log, flush=True)
    if not a.no_summary: summarize(out)


def summarize(out):
    out = Path(out); recs = []
    for f in sorted((out / "labels").glob("*.jsonl")):
        recs += [json.loads(l) for l in open(f) if l.strip()]
    eps = {}
    for r in recs:                                               # decision-vs-GT checks need the episode grid
        if r["status"] == "ok":
            ep = eps.get(r["episode"]) or eps.setdefault(r["episode"], data.load_episode(r["episode"]))
            r["gt"] = qa.gt_check(r, ep)
    import pandas as pd
    D = pd.DataFrame([dict(episode=r["episode"], key=r["key"], cell=r["cell"], kind=r["kind"], status=r["status"], env=r["env"],
                           lon=r["decision"]["longitudinal"], lat=r["decision"]["lateral"], dir=r["decision"].get("direction"),
                           coc=r["coc"], tau0=r["tau0"], bag_ns=r["bag_ns"], qa=r["qa"], clean=r["clean"],
                           gt_lon=(r.get("gt") or {}).get("lon"), gt_lat=(r.get("gt") or {}).get("lat"),
                           components=json.dumps(r["critical_components"], ensure_ascii=False)) for r in recs])
    D["use"] = (D.status == "ok") & (D.gt_lon != False) & (D.gt_lat != False)   # training set: observed cause, decision matches GT
    D.to_parquet(out / "coc_dataset.parquet")
    qa.review_sheet(recs, out / "review.html")
    s = {"keyframes": len(D), "status": D.status.value_counts().to_dict(),
         "by_cell": D.groupby("cell").status.value_counts().unstack(fill_value=0).to_dict("index"),
         "gt_lon_agree": float(D.gt_lon.dropna().astype(bool).mean()) if D.gt_lon.notna().any() else None,
         "gt_lat_agree": float(D.gt_lat.dropna().astype(bool).mean()) if D.gt_lat.notna().any() else None,
         "gt_fail": D[(D.gt_lon == False) | (D.gt_lat == False)][["episode", "key", "kind", "coc"]].to_dict("records")[:50]}
    json.dump(s, open(out / "summary.json", "w"), indent=1, ensure_ascii=False, default=str)
    return s


def recompose(src, out):
    """an earlier run's labels -> labels from the current events/compose code with the stored VLM answers (no Codex calls).
    Statuses set after that run (invalid_no_signal) and has_step0 are carried over from its parquet."""
    import pandas as pd
    src, out = Path(src), Path(out)
    P = pd.read_parquet(src / "coc_dataset.parquet"); bad = P[P.status == "invalid_no_signal"]
    invalid = {}
    for e, k in zip(bad.episode, bad.key): invalid.setdefault(e, set()).add(k)
    stale = {}
    for f in sorted((src / "labels").glob("*.jsonl")):
        old = [json.loads(l) for l in open(f) if l.strip()]
        _, st = pipeline.recompose_episode(f.stem, old, out, invalid.get(f.stem, ()))
        if st: stale[f.stem] = st
    if stale: raise SystemExit(f"{sum(map(len, stale.values()))} keyframes in {len(stale)} episodes no longer match: {dict(list(stale.items())[:5])}")
    s = summarize(out)
    if "has_step0" in P:
        D = pd.read_parquet(out / "coc_dataset.parquet"); D["has_step0"] = D.episode.map(P.groupby("episode").has_step0.first())
        D["use"] &= D.has_step0                                  # no bags, no training sample
        D.to_parquet(out / "coc_dataset.parquet")
    return s


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["summarize"]: print(json.dumps({k: v for k, v in summarize(sys.argv[2]).items() if k != "gt_fail"}, indent=1, default=str))
    elif sys.argv[1:2] == ["recompose"]: print(json.dumps({k: v for k, v in recompose(*sys.argv[2:4]).items() if k != "gt_fail"}, indent=1, default=str))
    else: main()
