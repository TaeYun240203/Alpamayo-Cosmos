"""QAD outputs, in the user's order (10/27, again 10/28 "납품이랑 관문 순서가 반대야"): QAD -> early delivery -> gate.
  manifest <qad dir>             quant_manifest.json (formats and amax of the trained quantizers of final/), right after QAD
  deliver <qad dir> <deliver>    what the RTX 5080 NVFP4/FP8 engine conversion needs (final/, quant_manifest.json)
  summary <gate dir>             after the delivery: from gate_s2/, gate_qad/ (gate_student.py on the input and the QAD
                                 student, same samples and noise; R2: + sets_*/) write summary.json (ratio_vs_s2 = QAD
                                 student ADE / input ADE, pass = ratio < 1.25; R2 records). pipeline qgate copies it to
                                 <deliver>/gate.json; a failed gate never removes the delivery.
Usage (cwd train/): python scripts/qad_export.py manifest|summary <dir>   |   python scripts/qad_export.py deliver <qad dir> <deliver dir>"""
import json, shutil, sys
from pathlib import Path

CONTENTS = """QAD student for the RTX 5080 engine (MODEL_PLAN 5.1, train/README "학생").
final/                  HF StudentVLA dir: BF16 weights after QAD (incl. quantizer amax buffers), config.json,
                        modelopt_state.pth (loads back with quantizers: morai_vla.student.StudentVLA.build(init=...)).
quant_manifest.json     per quantized linear: format (nvfp4 | fp8), block size (16 for nvfp4), weight_amax, input_amax,
                        shape. NVFP4 per-tensor scale = amax / (6 * 448) with e4m3 block scales per 16 values;
                        FP8 scale = amax / 448. Layers not listed stay BF16 (LiDAR ViT, mergers, embeddings, action proj).
gate.json               added after the delivery (pipeline qgate): ADE of the QAD student vs its input (s2, or ft final in the
                        R2 round) and T* (stage C) on the same val samples, pass = ratio_vs_s2 < 1.25; R2: records on the
                        dynamic, curve and start-relabel sets.
Inference format: no CoC, prefix ending in <traj_future_start>, Euler 10 steps (FP32 state), K = 1.
"""


def manifest(d):
    import torch
    from morai_vla import qad
    from morai_vla.student import StudentVLA
    d = Path(d)
    m = StudentVLA.build(init=str(d / "final"), attn_implementation="sdpa", device_map={"": 0} if torch.cuda.is_available() else None)
    meta = qad.write_manifest(m, d)
    prev = json.load(open(d / "metrics.json")) if (d / "metrics.json").exists() else {}       # loss_metrics: the kept checkpoint
    json.dump({**prev, "quant": meta}, open(d / "metrics.json", "w"), indent=1); print(json.dumps(meta))


def summary(d):
    d = Path(d)
    g2, gq = (json.load(open(d / g / "summary.json")) for g in ("gate_s2", "gate_qad"))
    s = {"ratio_vs_s2": round(gq["student_ade"] / g2["student_ade"], 4), "qad_ade": gq["student_ade"], "s2_ade": g2["student_ade"],
         "teacher_ade": gq.get("teacher_ade"), "qad_vs_teacher": gq.get("ratio"), "n": gq["n"]}   # R2 (no T*): None
    s["pass"] = s["ratio_vs_s2"] < 1.25                                             # = gate_qgate's halt rule
    rec = {}                                                                          # R2 records (user 10/28), not the halt rule
    if "student_ade_wo_relabel" in gq:
        rec["general_wo_relabel"] = {"input": g2["student_ade_wo_relabel"], "qad": gq["student_ade_wo_relabel"], "n": gq["n_wo_relabel"]}
    for f in sorted(d.glob("gate_qad/sets_*/sets.json")):
        g = d / "gate_s2" / f.parent.name / "sets.json"
        if not g.exists(): continue
        q, i = json.load(open(f)), json.load(open(g))
        rec |= {k: {"input": i[k]["ade"], "qad": v["ade"], "ratio": round(v["ade"] / i[k]["ade"], 4), "n": v["n"]} for k, v in q.items()}
    if rec: s["records"] = rec
    json.dump(s, open(d / "summary.json", "w"), indent=1); json.dump(s, open(d / "metrics.json", "w"), indent=1)
    print(json.dumps(s))


def deliver(d, out):
    d, out = Path(d), Path(out); out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(d / "final", out / "final", dirs_exist_ok=True)
    shutil.copy(d / "quant_manifest.json", out / "quant_manifest.json")             # gate.json follows after the gate (qgate)
    (out / "CONTENTS.txt").write_text(CONTENTS)
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    json.dump({"files": files, "bytes": sum((out / f).stat().st_size for f in files)}, open(out / "metrics.json", "w"), indent=1)


if __name__ == "__main__":
    {"manifest": lambda: manifest(sys.argv[2]), "summary": lambda: summary(sys.argv[2]), "deliver": lambda: deliver(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
