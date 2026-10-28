"""scripts/qad_export.py, R2 order (user 10/28): deliver right after QAD (no gate result yet), then the gate summary without T*
(pass flag, records) that pipeline qgate copies to deliver/gate.json. No torch."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import qad_export  # noqa: E402


def test_deliver_then_gate_summary(tmp_path):
    q = tmp_path / "qad"; (q / "final").mkdir(parents=True)
    (q / "final" / "config.json").write_text("{}"); (q / "quant_manifest.json").write_text("{}")
    qad_export.deliver(q, tmp_path / "deliver")                                   # before any gate
    assert sorted(p.name for p in (tmp_path / "deliver").iterdir()) == ["CONTENTS.txt", "final", "metrics.json", "quant_manifest.json"]
    g = tmp_path / "qgate"
    for name, ade, wo in (("gate_s2", 1.50, 1.40), ("gate_qad", 1.65, 1.47)):
        (g / name).mkdir(parents=True)
        json.dump({"student_ade": ade, "student_ade_wo_relabel": wo, "n_wo_relabel": 496, "n": 500}, open(g / name / "summary.json", "w"))
        (g / name / "sets_dynamic").mkdir(); json.dump({"dynamic": {"ade": 4.0 if name == "gate_s2" else 4.2, "n": 300}}, open(g / name / "sets_dynamic" / "sets.json", "w"))
    qad_export.summary(g)
    s = json.load(open(g / "summary.json"))
    assert s["ratio_vs_s2"] == 1.1 and s["pass"] is True and s["teacher_ade"] is None
    assert s["records"]["general_wo_relabel"] == {"input": 1.40, "qad": 1.47, "n": 496} and s["records"]["dynamic"]["ratio"] == 1.05
