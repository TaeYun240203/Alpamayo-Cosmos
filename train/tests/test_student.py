"""Student: the 4B processor gives the teacher's tokens, the expert transplant is strict and computes exactly what T*'s
does, every loss runs and reaches only the stage's trainable parts (stage 1: VLM + KD maps, the FM gradient goes into
the VLM through the frozen expert; stage 2: + expert), the teacher stays out of the student's weights, save -> strict
reload, and the configs compose. Tiny random models on CPU; needs the HF cache (Qwen3-VL-4B/8B processors, the 1.5
release config.json)."""
import inspect, os
from pathlib import Path

import numpy as np, pytest, torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from alpamayo1_5_sft.trainer import TrainingArguments
from morai_vla.data import MoraiDataset
from morai_vla.model import TeacherVLA
from morai_vla.processor import collate_fn, get_preprocess_fn
from morai_vla.student import StudentVLA, transplant
from test_layout import CFG, sample

from huggingface_hub.constants import HF_HUB_CACHE

RELEASE = os.environ.get("ALPAMAYO15") or next((str(p.parent) for p in Path(HF_HUB_CACHE).glob("models--nvidia--Alpamayo-1.5-10B/snapshots/*/config.json")), None)
need_release = pytest.mark.skipif(RELEASE is None, reason="Alpamayo-1.5-10B config.json not found (HF cache or $ALPAMAYO15)")
CONFIGS = Path(__file__).resolve().parents[1] / "configs"
TINY_T = dict(text_config=dict(hidden_size=256, intermediate_size=512, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                               rope_scaling=dict(rope_type="default", mrope_section=[12, 10, 10], mrope_interleaved=True)),
              vision_config=dict(depth=4, hidden_size=128, intermediate_size=256, num_heads=4, out_hidden_size=256, deepstack_visual_indexes=[1, 2, 3]))
TINY_S = OmegaConf.to_container(OmegaConf.load(CONFIGS / "tiny/student.yaml").model.vlm_overrides)
EXPERT = dict(hidden_size=128, intermediate_size=256, num_attention_heads=4, head_dim=64)
STAGE1 = ["camera_vit", "lidar", "llm", "lm_head", "kd_proj"]


def test_processor_4b_gives_teacher_tokens():
    s = sample()
    a = get_preprocess_fn(CFG)(data=dict(s))
    b = get_preprocess_fn(CFG.__class__(**{**vars(CFG), "vlm_name_or_path": "Qwen/Qwen3-VL-4B-Instruct"}))(data=dict(s))
    assert a["text"] == b["text"] and torch.equal(a["pixel_values"], b["pixel_values"]) and torch.equal(a["image_grid_thw"], b["image_grid_thw"])
    from morai_vla.processor import get_processor
    ta = get_processor(CFG).processor.tokenizer; tb = get_processor(CFG.__class__(**{**vars(CFG), "vlm_name_or_path": "Qwen/Qwen3-VL-4B-Instruct"})).processor.tokenizer
    assert ta(a["text"])["input_ids"] == tb(b["text"])["input_ids"] and len(ta) == len(tb)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("teacher")
    T = TeacherVLA.build("random:" + RELEASE, attn_implementation="sdpa", dtype="float32", vlm_overrides=TINY_T, expert_overrides=EXPERT)
    T.save_pretrained(d)
    S = StudentVLA.build(teacher=str(d), init="random", attn_implementation="sdpa", dtype="float32", vlm_overrides=TINY_S, train_parts=STAGE1,
                         loss_weights=dict(kv=1.0, vit=1.0, fm=1.0, traj_ce=1.0, lm=1.0, vkd=0.0, coc_kl=0.0))
    return S, d


def batch(model, cots=(None, None), lidar=(True, False), vkd_ok=None):
    rows = []
    for cot, li in zip(cots, lidar):
        s = sample(cot=cot, lidar=li)
        s["ego_history_xyz"][0, :, 0] = torch.linspace(-7.5, 0, 16); s["ego_future_xyz"][0, :, 0] = torch.linspace(0.5, 32, 64)   # 5 m/s
        s["tokenized_data"] = get_preprocess_fn(model.config)(data=s)
        for k in ("camera_frames", "bev_frames", "camera_times", "bev_times"): s.pop(k, None)
        rows.append(s)
    b = collate_fn(rows, model_config=model.config)
    if vkd_ok is not None: b["vkd_ok"] = list(vkd_ok)
    return b


def run(S, b):
    S.loss_sums.clear()
    loss = S(**{k: v for k, v in b.items() if k in ("tokenized_data", "ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot",
                                                   "labels_mask", "vkd_ok")}).loss
    return loss, {k: s / n for k, (s, n) in S.loss_sums.items()}


def grads(S):
    out = {}
    for p, paths in S.PARTS.items():
        g = [x.grad for path in paths for x in S.get_submodule(path).parameters() if x.grad is not None]
        out[p] = float(sum(x.abs().sum() for x in g)) if g else 0.0
    S.zero_grad(set_to_none=True)
    return out


@need_release
def test_transplant_is_strict_and_exact(models, tmp_path):
    S, d = models
    T = S.teacher
    for name in ("expert", "action_in_proj", "action_out_proj"):
        for (k, a), (_, b) in zip(T.get_submodule(name).state_dict().items(), S.get_submodule(name).state_dict().items()):
            assert torch.equal(a, b), f"{name}.{k}"
    assert S.expert.config.hidden_size == 128 and S.vlm.config.text_config.hidden_size == 192 != T.vlm.config.text_config.hidden_size
    b = batch(S)
    td = dict(b["tokenized_data"]); T.vlm.model._image_is_lidar = td.pop("image_is_lidar")
    ids = T.fuse_traj_tokens(td.pop("input_ids"), {k: b[k] for k in ("ego_history_xyz", "ego_history_rot")})
    x, t = torch.randn(2, 64, 2), torch.rand(2, 1, 1)
    with torch.no_grad():
        out = T.vlm(input_ids=ids, use_cache=True, logits_to_keep=1, **td)
        vt = T._velocity(out, ids, td["attention_mask"], x, t)
        vs = S._velocity(out, ids, td["attention_mask"], x, t)                         # student expert on the SAME (T*) KV
    assert torch.equal(vt, vs)
    bad = TeacherVLA.build("random:" + RELEASE, attn_implementation="sdpa", dtype="float32", vlm_overrides=TINY_T, expert_overrides={**EXPERT, "intermediate_size": 320})
    bad.save_pretrained(tmp_path)
    with pytest.raises(RuntimeError, match="size mismatch"):
        transplant(S, tmp_path)


@need_release
def test_stage1_losses_and_gradient_routing(models):
    S, _ = models
    S.set_stage(STAGE1, dict(kv=1.0, vit=1.0, fm=1.0, traj_ce=1.0, lm=1.0, vkd=0.0, coc_kl=0.0))
    loss, parts = run(S, batch(S))
    assert set(parts) == {"kv", "vit", "fm", "traj_ce", "lm"} and all(np.isfinite(v) and v > 0 for v in parts.values()), parts
    loss.backward()
    g = grads(S)
    assert g["expert"] == 0 and all(g[p] > 0 for p in ("camera_vit", "lidar", "llm", "kd_proj")), g
    assert not any(p.grad is not None for p in S.teacher.parameters())
    S.set_stage(STAGE1, dict(kv=0.0, vit=0.0, fm=1.0, traj_ce=0.0, lm=0.0))                # FM alone reaches the VLM (no stop-gradient)
    run(S, batch(S))[0].backward()
    g = grads(S)
    assert g["llm"] > 0 and g["camera_vit"] > 0 and g["expert"] == 0 and g["kd_proj"] == 0, g
    assert not any(k.startswith("teacher") for k in S.state_dict())
    S.set_stage(STAGE1, dict(kv=1.0, vit=1.0, fm=1.0, traj_ce=1.0, lm=1.0)); S.train()
    run(S, batch(S, lidar=(False, False)))[0].backward()               # LiDAR-free batch: zero-weighted LiDAR pass, same graph
    assert S.vit_kd["lidar"].weight.grad is not None and S.vlm.model.lidar_visual.merger.norm.weight.grad is not None
    S.zero_grad(set_to_none=True)


@need_release
def test_stage2_vkd_gate_and_coc(models):
    S, _ = models
    S.set_stage(STAGE1 + ["expert"], dict(kv=1.0, vit=1.0, fm=1.0, traj_ce=1.0, lm=1.0, vkd=1.0, coc_kl=0.0))
    S.train()
    with pytest.raises(AssertionError, match="rule check"):
        run(S, batch(S))
    loss, parts = run(S, batch(S, vkd_ok=(True, False)))
    assert parts["vkd"] > 0
    loss.backward(); assert grads(S)["expert"] > 0
    assert run(S, batch(S, vkd_ok=(False, False)))[1]["vkd"] == 0
    S.set_stage(STAGE1, dict(kv=1.0, vit=1.0, fm=1.0, traj_ce=1.0, lm=1.0, vkd=0.0, coc_kl=1.0))
    loss, parts = run(S, batch(S, cots=("Stop for the red light.", None)))           # FM only on the CoC-free row
    assert parts["coc_kl"] > 0 and np.isfinite(parts["fm"])
    loss, parts = run(S, batch(S, cots=("Stop for the red light.", "Yield to the pedestrian.")))
    assert "fm" not in parts and parts["coc_kl"] > 0


@need_release
def test_eval_fm_metric_is_deterministic(models):
    """evaluation FM uses noise and t fixed by the sample keys; EvalComponentMetric reports its mean as eval_fm."""
    from morai_vla.student import EvalComponentMetric
    S, _ = models
    S.set_stage(STAGE1, dict(kv=1.0, vit=1.0, fm=1.0, traj_ce=1.0, lm=1.0, vkd=0.0, coc_kl=0.0)); S.eval(); S.eval_sums.clear()
    b = batch(S.cpu())
    keys = ("tokenized_data", "ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot", "labels_mask", "key")
    fm = []
    for _ in range(2):
        with torch.no_grad(): S(**{k: v for k, v in b.items() if k in keys})
        fm.append(S.loss_sums.pop("fm")[0]); S.loss_sums.clear()
    assert fm[0] == fm[1]                                                 # same keys -> same noise and t
    metrics = {}
    EvalComponentMetric("fm").on_evaluate(None, None, None, metrics=metrics, model=S)
    assert metrics["eval_fm"] == pytest.approx(fm[0]) and S.eval_sums == {}
    S.train()


@need_release
def test_save_reload_strict(models, tmp_path):
    S, d = models
    S.save_pretrained(tmp_path)
    R = StudentVLA.build(teacher=str(d), init=str(tmp_path), attn_implementation="sdpa", dtype="float32")
    for (k, a), (k2, b) in zip(S.state_dict().items(), R.state_dict().items()):
        assert k == k2 and torch.equal(a, b), k
    assert R.vlm.lm_head.weight is R.vlm.model.language_model.embed_tokens.weight and R.teacher is not None


@pytest.mark.parametrize("name,tiny", [("student1", False), ("student2", False), ("student1", True), ("student2", True), ("student_qad", False)])
def test_student_configs(name, tiny, tmp_path):
    over = ["paths.teacher=/t", "data.root=/data", f"paths.output_dir={tmp_path}", "trainer.deepspeed=null"]
    over += ["model.init=/s1", "data.train_dataset.rule_ok=/r.parquet"] if name != "student1" else []
    with initialize_config_dir(str(CONFIGS), version_base=None):
        cfg = compose(name, overrides=over + (["+tiny=student"] if tiny else []))
    c = OmegaConf.to_container(cfg, resolve=True)
    TrainingArguments(**c["trainer"])
    m = c["model"]
    assert set(m["train_parts"]) <= set(StudentVLA.PARTS) and set(m["loss_weights"]) == {"kv", "vit", "fm", "traj_ce", "lm", "vkd", "coc_kl"}
    assert set(m) - {"_target_"} <= set(inspect.signature(StudentVLA.build).parameters)
    assert not c["trainer"]["gradient_checkpointing"]                                # the expert reads the student KV cache
    for ds in ("train_dataset", "val_dataset"):
        assert set(c["data"][ds]) - {"_target_", "_recursive_"} <= set(inspect.signature(MoraiDataset).parameters)
    s2 = name != "student1"
    assert ("expert" in m["train_parts"]) == s2 and (m["loss_weights"]["vkd"] > 0) == s2 and ("vkd_weight" in c["callbacks"]) == s2
    assert (m["init"] == ("random" if tiny and not s2 else "/s1" if s2 else None))
    if name == "student2": assert c["trainer"]["lr_multiplier"]["expert"] == 0.1 and c["data"]["train_dataset"]["rule_ok"] == "/r.parquet"
    if name == "student_qad": assert c["quant"]["effective_bits"] == 4.75 and c["callbacks"]["vkd_weight"]["start"] == 0.2 and c["trainer"]["max_steps"] > 0
    t, cb = c["trainer"], c["callbacks"]
    if name == "student1":                             # final/ = lowest trajectory-ADE measurement (user rule, early_stop.py)
        es = cb["traj_early_stop"]
        assert not t["load_best_model_at_end"] and es["_target_"] == "morai_vla.early_stop.TrajEarlyStop" and es["root"] == "/data"
        assert set(es) == {"_target_", "root", "every_steps", "decay_steps"} and 0 < es["decay_steps"] and 0 < es["every_steps"]
    else:                                              # s2, QAD: final/ = best eval_fm checkpoint, selectable only at saves
        assert t["load_best_model_at_end"] and t["metric_for_best_model"] == "eval_fm" and not t["greater_is_better"]
        assert t["save_steps"] % t["eval_steps"] == 0 and (tiny or t["save_steps"] == t["eval_steps"]) and "traj_early_stop" not in cb
    assert cb["eval_fm"]["_target_"] == "morai_vla.student.EvalComponentMetric" and "early_stop" not in cb


@need_release
def test_qad_search_manifest_reload(models, tmp_path):
    """AutoQuantize keeps every LLM / expert linear quantized (NVFP4 or FP8, no BF16) near the bit budget, the camera ViT
    in FP8 and the rest BF16; the quantized student saves and loads back with its quantizers."""
    pytest.importorskip("modelopt")
    if not torch.cuda.is_available(): pytest.skip("NVFP4 fake quant needs CUDA")
    from morai_vla import qad
    S, d = models
    S.cuda().set_stage(STAGE1 + ["expert"], dict(kv=1.0, vit=1.0, fm=1.0, traj_ce=1.0, lm=1.0, vkd=0.0, coc_kl=0.0)); S.train()
    keys = ("tokenized_data", "ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot", "labels_mask")
    gpu = lambda x: {k: gpu(v) if isinstance(v, dict) else v.cuda() if torch.is_tensor(v) else v for k, v in x.items()}
    batches = [gpu({k: v for k, v in batch(S).items() if k in keys}) for _ in range(2)]
    Q, _ = qad.search(S, batches, effective_bits=6.5, num_calib_steps=2, num_score_steps=2)
    man = qad.manifest(Q)
    llm = [n for n, m in Q.named_modules() if isinstance(m, torch.nn.Linear) and (".language_model.layers." in n or n.startswith("expert.layers."))]
    assert llm and all(n in man and man[n]["format"] in ("nvfp4", "fp8") for n in llm)
    assert {man[n]["format"] for n in man if ".model.visual.blocks." in n} == {"fp8"}
    assert not any(k in n for n in man for k in ("lidar_visual", "lm_head", "action_", "vit_kd", "merger"))
    assert 4.5 <= qad.effective_bits(man) <= 6.6 and all(v["weight_amax"] and v["input_amax"] for v in man.values())
    Q.save_pretrained(tmp_path)
    R = StudentVLA.build(init=str(tmp_path), attn_implementation="sdpa", dtype="float32")
    assert {k: v["format"] for k, v in qad.manifest(R).items()} == {k: v["format"] for k, v in man.items()}
