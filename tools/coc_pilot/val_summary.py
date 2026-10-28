"""Aggregate official-stage judge scores per set x labeler; a label 'passes' a stage on the official score anchors."""
import json, pandas as pd
R = pd.DataFrame([json.loads(l) for l in open("val_results.jsonl")])
R = R.sort_values("stage").drop_duplicates(["set", "labeler", "key", "stage"], keep="last")
w = R.pivot_table(index=["set", "labeler", "key"], columns="stage", aggfunc="first")
cols = {"visual_evidence_score": 1, "scene_factor_hallucination_score": 1, "behavior_match_score": 2, "behavior_hallucination_score": 2,
        "decision_supported_by_video": 3, "causal_evidence_score": 3, "causal_consistency_score": 3, "causal_hallucination_score": 3}
T = pd.DataFrame({k: w[(k, s)] for k, s in cols.items()}).astype(float)
# pass rules from the official anchors: evidence >=0.65 (mostly visible), behavior match >=0.65, causal consistency >=0.55, hallucination <0.5
T["pass_s1"] = (T.visual_evidence_score >= 0.65) & (T.scene_factor_hallucination_score < 0.5)
T["pass_s2"] = (T.behavior_match_score >= 0.65) & (T.behavior_hallucination_score < 0.5)
T["pass_s3"] = (T.causal_consistency_score >= 0.55) & (T.causal_hallucination_score < 0.5)
T["pass_all"] = T.pass_s1 & T.pass_s2 & T.pass_s3
g = T.groupby(level=[0, 1])
out = g.mean().round(2); out.insert(0, "n", g.size())
pd.set_option("display.width", 250)
print(out[["n", "visual_evidence_score", "scene_factor_hallucination_score", "behavior_match_score", "decision_supported_by_video",
           "causal_consistency_score", "causal_hallucination_score", "pass_s1", "pass_s2", "pass_s3", "pass_all"]].to_string())
T.to_pickle("val_table.pkl")
print("\nfailures:")
for (st, lb, key), r in T[~T.pass_all].iterrows():
    print(st, lb, key, "s1" if not r.pass_s1 else "", "s2" if not r.pass_s2 else "", "s3" if not r.pass_s3 else "")
