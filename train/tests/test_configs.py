"""Every run config composes, its trainer section is valid TrainingArguments, and its parts/loss/dataset keys exist."""
import inspect
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from alpamayo1_5_sft.trainer import TrainingArguments
from morai_vla.data import MoraiDataset
from morai_vla.model import PARTS

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
VLM = ["camera_vit", "lidar", "llm", "lm_head"]
RUNS = {"step2": (["lidar"], False), "stage1t": (VLM, False), "stage1c": (VLM, True), "stage2": (["expert"], False)}


@pytest.mark.parametrize("name", [*RUNS, "tiny_overfit", "tiny_stage2"])
def test_config(name, tmp_path):
    with initialize_config_dir(str(CONFIGS), version_base=None):
        cfg = compose(name, overrides=["paths.alpamayo15=/rel", "data.root=/data", "model.init=/ckpt", f"paths.output_dir={tmp_path}",
                                       "trainer.deepspeed=null"])                        # DeepSpeed itself needs nvcc to import
    c = OmegaConf.to_container(cfg, resolve=True)
    TrainingArguments(**c["trainer"])
    assert set(c["model"]["train_parts"]) <= set(PARTS) and set(c["model"]["loss_weights"]) == {"traj_ce", "lm", "fm"}
    params = inspect.signature(MoraiDataset).parameters
    for ds in ("train_dataset", "val_dataset"):
        assert set(c["data"][ds]) - {"_target_", "_recursive_"} <= set(params), set(c["data"][ds]) - set(params)
    if name in RUNS:
        parts, coc = RUNS[name]
        assert c["model"]["train_parts"] == parts and c["data"]["train_dataset"]["coc"] is coc
        fm = c["model"]["loss_weights"]["fm"] > 0
        assert fm == (name == "stage2") and (not fm or not c["trainer"]["gradient_checkpointing"])   # FM needs the KV cache
        assert c["data"]["coc_parquet"] == "/data/_coc/coc_dataset.parquet"
        # user, 10/25: final = the best eval_loss checkpoint (full epochs, no early stop); Stage 1 keeps the expert-free monitor
        if name.startswith("stage1"):
            t = c["trainer"]
            assert t["load_best_model_at_end"] and t["metric_for_best_model"] == "eval_loss" and not t["greater_is_better"]
            assert t["save_steps"] % t["eval_steps"] == 0
        extra = {"stage1t": {"stage1_eval"}, "stage1c": {"stage1_eval"}, "stage2": {"traj_early_stop"}}.get(name, set())
        assert set(c["callbacks"]) == {"loss_components", "nonfinite_stop"} | extra
        if name == "stage2":                                   # user 10/25 (later): final/ = lowest trajectory ADE, not eval_loss
            from morai_vla.early_stop import TrajEarlyStop
            cb = dict(c["callbacks"]["traj_early_stop"]); cb.pop("_target_")
            assert not c["trainer"]["load_best_model_at_end"] and cb == {"root": "/data", "every_steps": 4000, "decay_steps": 3000}
            assert set(cb) <= set(inspect.signature(TrajEarlyStop).parameters)
        if name.startswith("stage1"):
            from morai_vla.stage1_eval import Stage1EvalCallback
            cb = c["callbacks"]["stage1_eval"]; cb.pop("_target_")
            assert cb["root"] == "/data" and set(cb) <= set(inspect.signature(Stage1EvalCallback).parameters)


# every override the README runbook and scripts/stage_b.sh pass must name an existing key (Hydra refuses new ones)
SMOKE = ["data.train_dataset.max_samples=20", "data.train_dataset.augment=false", "trainer.max_steps=30", "trainer.gradient_accumulation_steps=1",
         "trainer.warmup_steps=10", "trainer.eval_strategy=no", "trainer.save_strategy=no", "trainer.save_steps=20", "trainer.logging_steps=10",
         "trainer.per_device_train_batch_size=2", "trainer.dataloader_num_workers=8", "model.attn_implementation=sdpa", "resume=true",
         "data.coc_parquet=/coc.parquet", "trainer.deepspeed=null", "save_final=false"]


@pytest.mark.parametrize("name", ["step2", "stage1t", "stage2"])
def test_runbook_overrides(name, tmp_path):
    with initialize_config_dir(str(CONFIGS), version_base=None):
        cfg = compose(name, overrides=["paths.alpamayo15=/rel", "data.root=/data", "model.init=/ckpt", f"paths.output_dir={tmp_path}", *SMOKE])
    c = OmegaConf.to_container(cfg, resolve=True)
    TrainingArguments(**c["trainer"])
    assert c["data"]["train_dataset"]["max_samples"] == 20 and c["trainer"]["max_steps"] == 30
