"""scripts/ft_metrics.py on a fake run dir (no torch): CSVs, an ADE-only step 0, FtSelect's stop_ref set (general_wo_relabel)
in val.csv, the change vs step 0."""
import csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import ft_metrics  # noqa: E402


def test_csvs(tmp_path, monkeypatch):
    run = tmp_path / "ft"; run.mkdir()
    (run / "loss_components.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"step": 10, "loss": 3.0, "grad_norm": 1.5, "fm": 1.0, "new_a/fm": 1.2, "replay/kv": 0.5},
        {"step": 20, "eval_fm": 0.9}, {"step": 20, "loss": 2.0, "grad_norm": 1.1, "fm": 0.8, "s_per_step": 2.7}]) + "\n")
    m = lambda a, f: {"ade": a, "fde": f, "lat": 0.5, "lon": 1.0}
    hist = [{"step": 0, "tag": "start", "ade": {"general": 2.0, "general_wo_relabel": 1.9}, "n": {"general": 3}},   # ADE only
            {"step": 500, "tag": "main", "ade": {"general": 2.2, "general_wo_relabel": 2.09}, "n": {"general": 3, "general_wo_relabel": 2},
             "metrics": {"general": m(2.2, 5.0), "general_wo_relabel": m(2.09, 4.0)}}]
    (run / "ft_select.json").write_text(json.dumps({"history": hist, "stop_ref": "general_wo_relabel"}))
    monkeypatch.setattr(sys, "argv", ["ft_metrics", str(run)])
    ft_metrics.main()
    read = lambda n: list(csv.DictReader(open(run / "metrics" / n)))
    assert [r["step"] for r in read("train.csv")] == ["10", "20"] and read("train.csv")[0]["new_a/fm"] == "1.2"
    assert read("eval.csv")[0]["eval_fm"] == "0.9"
    v = {(r["step"], r["set"]): r for r in read("val.csv")}
    assert v[("0", "general")]["ade"] == "2.0" and v[("500", "general")]["fde"] == "5.0"
    assert abs(float(v[("500", "general")]["ade_vs_step0"]) - 0.1) < 1e-9
    assert v[("500", "general_wo_relabel")]["n"] == "2" and abs(float(v[("500", "general_wo_relabel")]["ade_vs_step0"]) - 0.1) < 1e-9
