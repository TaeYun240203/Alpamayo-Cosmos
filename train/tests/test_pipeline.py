"""scripts/pipeline.sh pieces: the automatic halt checks, the non-finite stop, and the teacher stages' Hydra overrides."""
import json, re
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from alpamayo1_5_sft.trainer import TrainingArguments
from morai_vla.guards import NonFiniteStop
from scripts.pipeline_gate import max_check, train_check

TRAIN = Path(__file__).resolve().parents[1]
T1_DEFAULT = ("trainer.per_device_train_batch_size=4 trainer.gradient_accumulation_steps=1 trainer.gradient_checkpointing=false "
              "model.attn_implementation=flash_attention_2")                       # $T1 of pipeline.sh with its defaults


def write_log(d, rows):
    (d / "loss_components.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_train_check(tmp_path):
    write_log(tmp_path, [{"step": 10, "traj_ce": 1.2}, {"step": 500, "eval_traj_ce": 1.0}, {"step": 1000, "eval_traj_ce": 1.1}])
    assert train_check(tmp_path) is None                                            # 1.1 <= 1.2 x 1.0
    write_log(tmp_path, [{"step": 500, "eval_traj_ce": 1.0}, {"step": 1000, "eval_traj_ce": 1.3}])
    assert "diverged" in train_check(tmp_path) and train_check(tmp_path, divergence=False) is None
    write_log(tmp_path, [{"step": 10, "traj_ce": float("nan")}])
    assert "non-finite" in train_check(tmp_path)


def test_max_check(tmp_path):
    (tmp_path / "s.json").write_text(json.dumps({"traj_ade": 5.0, "by_cell": {"C0": {"traj_ade": 6.0}}}))
    assert max_check(tmp_path / "s.json", "traj_ade", 5.29) is None
    assert max_check(tmp_path / "s.json", "by_cell.C0.traj_ade", 5.29) is not None


def test_nonfinite_stop(tmp_path):
    control, state = SimpleNamespace(should_training_stop=False), SimpleNamespace(global_step=7, is_world_process_zero=True)
    NonFiniteStop().on_log(SimpleNamespace(output_dir=tmp_path), state, control, logs={"loss": 1.0, "grad_norm": float("inf")})
    assert control.should_training_stop and "grad_norm" in (tmp_path / "NONFINITE").read_text()
    write_log(tmp_path, [{"step": 7, "traj_ce": 1.0}])
    assert "non-finite" in train_check(tmp_path)


def stage_overrides(fn):
    """the Hydra overrides a teacher stage function of pipeline.sh passes, with its shell variables filled in."""
    body = re.search(rf"^{fn}\(\) {{(.*?)^}}", (TRAIN / "scripts/pipeline.sh").read_text(), re.S | re.M).group(1)
    words = re.sub(r"#.*", "", body).replace("\\\n", " ").replace("$T1 ", f"{T1_DEFAULT} ").split()
    sub = {"$(resume_flag": "null", "$T1T_EPOCHS": "2", "$T1C_EPOCHS": "3", "$T2_EPOCHS": "3", "$n": "17500", "$w": "14400", "$1": "/ckpt"}
    hydra = re.compile(r"^[+~]?(trainer|data|model|callbacks|save_final|resume)[.=]")
    out = []
    for w in words:
        if not hydra.match(w) or w.startswith(("model.init=", "paths.output_dir=")): continue
        for k, v in sub.items(): w = w.replace(k, v)
        out.append(w)
    return out


@pytest.mark.parametrize("fn,config", [("smoke8", "stage1t"), ("t1t", "stage1t"), ("t1t_anneal", "stage1t"), ("t1c", "stage1c"), ("t2", "stage2")])
def test_pipeline_teacher_overrides(fn, config, tmp_path):
    ov = [o if not o.startswith("resume=") else "resume=null" for o in stage_overrides(fn)]
    with initialize_config_dir(str(TRAIN / "configs"), version_base=None):
        cfg = compose(config, overrides=["paths.alpamayo15=/rel", "data.root=/data", "model.init=/ckpt", f"paths.output_dir={tmp_path}",
                                         "trainer.deepspeed=null", *ov])
    t = OmegaConf.to_container(cfg.trainer, resolve=True); TrainingArguments(**t)    # the stage overrides make valid TrainingArguments
    if t["lr_scheduler_type"] == "linear":                                         # HF's linear schedule takes no min_lr (10/25 anneal crash)
        import inspect as _i; from transformers.optimization import get_linear_schedule_with_warmup
        assert set(t.get("lr_scheduler_kwargs") or {}) <= set(_i.signature(get_linear_schedule_with_warmup).parameters)
    if t.get("load_best_model_at_end"):                                            # HF picks the best among saved checkpoints only
        assert t["save_steps"] == t["eval_steps"], (t["save_steps"], t["eval_steps"])


def test_select_picks_lowest_ade_and_copies_its_weights(tmp_path):
    from scripts.pipeline_gate import select
    rows = lambda ade: [{"key": f"k{i}", "cell": "C0", "ce_on": 1.0, "ce_off": 1.1, "ade": ade, "fde": 3 * ade, "cv_ade": 5.0, "cv_fde": 12.0} for i in range(4)]
    (tmp_path / "select").mkdir(); cands = {}
    for name, ade in (("a", 2.0), ("b", 1.5), ("c", 1.8)):
        for k in (0, 1):
            (tmp_path / "select" / f"{name}.shard{k}.json").write_text(json.dumps({"rows": rows(ade)[k::2]}))
        w = tmp_path / f"w_{name}"; w.mkdir(); (w / "config.json").write_text(name); (w / "model-00001-of-00001.safetensors").write_text(name)
        cands[name] = str(w)
    assert select(str(tmp_path), cands) == "b"
    best = json.loads((tmp_path / "best.json").read_text())
    assert (tmp_path / "final" / "config.json").read_text() == "b" and best["candidates"]["b"]["n"] == 4 and best["ade_minus_cv"] == -3.5
    assert max_check(tmp_path / "best.json", "ade_minus_cv", 0) is None
