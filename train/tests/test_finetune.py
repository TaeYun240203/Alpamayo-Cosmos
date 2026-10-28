"""R2 fine-tune (configs/student_ft.yaml, morai_vla/finetune.py): the mixture's length, shares and per-pass lattices;
keep / start-relabel rows; per-item loss flags; per-row loss routing (kv and vkd on replay rows only, GT rows fm + CE +
ViT KD); history dropout touching only the history tokens (the action targets keep the real history); the final-selection
rule; the config. CPU; the model tests need a 1.5 release config.json ($ALPAMAYO15 or the HF cache) and the
Qwen3-VL-4B processor (the Mac has no 8B: teacher_vlm = 4B, same tokenizer)."""
import inspect, json, os
from pathlib import Path

import numpy as np, pandas as pd, pytest, torch
from hydra import compose, initialize_config_dir
from huggingface_hub.constants import HF_HUB_CACHE
from omegaconf import OmegaConf

from alpamayo1_5_sft.trainer import TrainingArguments
from ft_synth import STAND, make
from morai_vla.data import MoraiDataset, ego_trajectory
from morai_vla.finetune import MixtureDataset, choose, dense_rows, plan

RELEASE = os.environ.get("ALPAMAYO15") or next((str(p.parent) for p in Path(HF_HUB_CACHE).glob("models--nvidia--Alpamayo-1.5-10B/snapshots/*/config.json")), None)
need_release = pytest.mark.skipif(RELEASE is None, reason="Alpamayo-1.5-10B config.json not found (HF cache or $ALPAMAYO15)")
CONFIGS = Path(__file__).resolve().parents[1] / "configs"
V4 = "Qwen/Qwen3-VL-4B-Instruct"


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return make(tmp_path_factory.mktemp("synth"))


def test_plan_density_shares_no_repeats():
    ep = lambda n, name: np.array([f"{name}-{k // 100}" for k in range(n)])          # episodes of 100 grid rows
    grid = lambda n: np.tile(np.arange(100), n // 100)
    rows = {s: (ep(n, s), grid(n)) for s, n in (("a", 1000), ("b", 300), ("old", 50000), ("small", 200))}
    shares = {"a": 0.4, "b": 0.2, "old": 0.3, "small": 0.1}
    sched, s = plan(rows, shares, ["a", "b"], stride=5, extra=(2, 3), seed=3)
    dense = {k: dense_rows(*rows[k], 5, (2, 3), 3) for k in ("a", "b")}
    o = {k: rows[k][1] % 5 for k in ("a", "b")}
    assert (dense["a"] == ((o["a"] == 0) | np.isin(o["a"], (2, 3)) & dense["a"])).all() and dense["a"].sum() == 2 * (o["a"] == 0).sum()
    assert set(np.unique(o["a"][dense["a"] & (o["a"] != 0)])) == {2, 3}              # 0.2 s and 0.3 s, one per period
    n4 = int(dense["a"].sum() + dense["b"].sum()); assert s["new_rows_4hz"] == n4 == 2 * s["new_rows_2hz"]
    assert s["total"] == round(n4 / 0.6) == len(sched)
    for j, name in enumerate(rows):
        mine = sched[sched[:, 0] == j]
        assert abs(len(mine) - shares[name] * s["total"]) <= 1
        first = mine[mine[:, 2] == 0]
        assert len(np.unique(first[:, 1])) == len(first)                              # no row twice
        if name in dense: assert dense[name][first[:, 1]].all()
        else: assert (rows[name][1][first[:, 1]] % 5 == 0).all()                       # old sources: 2 Hz rows only
    a = s["sources"]["a"]; b = s["sources"]["b"]
    assert a["repeated"] == a["other_offsets"] == 0 and a["rows_used"] < 1             # a: more rows than its share
    assert b["other_offsets"] > 0 and b["repeated"] == 0                               # b: fewer ~4 Hz rows -> unused offsets
    bb = sched[sched[:, 0] == 1]; assert len(np.unique(bb[:, 1])) == len(bb)           # still no sample twice
    assert s["sources"]["small"]["repeated"] > 0                                       # an old source past its 2 Hz rows repeats
    assert (plan(rows, shares, ["a", "b"], 5, (2, 3), seed=3)[0] == sched).all()       # fixed by the seed (resume)


def test_keep_rows_and_start_relabel(root):
    ds = MoraiDataset(root, "train", keep=str(root / "_ft/r2_A.parquet"))
    assert set(ds.rows.episode) == {"r2-a-001", "r2-a-002"}
    rel = MoraiDataset(root, "train", keep=str(root / "_ft/start_relabel.parquet"), augment=False)
    assert len(rel.rows) and (rel.rows.i < STAND).all() and (rel.rows.src_i == STAND).all()
    s = rel.sample(0); ep = rel._episode(rel.rows.episode.iloc[0]); i = int(rel.rows.i.iloc[0])
    hx, _, fx, _ = ego_trajectory(ep.v, ep.yaw, i); _, _, fs, _ = ego_trajectory(ep.v, ep.yaw, STAND)
    assert torch.equal(s["ego_history_xyz"], hx) and torch.equal(s["ego_future_xyz"], fs) and not torch.equal(fs, fx)
    assert float(s["ego_future_xyz"][0, -1, 0]) > 10                                  # the relabelled future drives away


def test_exclude_rows(root):
    rel = pd.read_parquet(root / "_ft/start_relabel.parquet").astype({"episode": str})
    ds = MoraiDataset(root, "train", exclude=str(root / "_ft/start_relabel.parquet"))
    full = MoraiDataset(root, "train")
    k = lambda d: set(zip(d.episode.astype(str), d.i))
    assert k(ds.rows) == k(full.rows) - k(rel) and k(rel) & k(full.rows)


def test_mixture_items(root):
    src = lambda keep, **kw: {"root": str(root), "split": "train", **({"keep": str(root / "_ft" / keep)} if keep else {}), **kw}
    m = MixtureDataset({"new_a": {"share": 0.5, "loss": "gt", "new": True, "dataset": src("r2_A.parquet")},
                        "relabel": {"share": 0.2, "loss": "gt", "dataset": src("start_relabel.parquet")},
                        "replay": {"share": 0.3, "loss": "distill", "dataset": src(None, rule_ok=str(root / "rule_ok.parquet"))}},
                       frame_stride=5, extra_offsets=(2, 3), hist_dropout=0.5, common={"augment": False})
    assert len(m) == m.summary["total"] == round(m.summary["new_rows_4hz"] / 0.5)
    items = [m[k] for k in range(len(m))]
    by = pd.DataFrame([{k: it[k] for k in ("source", "gt_row", "hist_drop", "vkd_ok", "ft_tier", "ft_offset", "key")} for it in items])
    assert (by.gt_row == (by.source != "replay")).all() and not by[~by.gt_row].hist_drop.any() and not by[by.gt_row].vkd_ok.any()
    assert 0.2 < by[by.gt_row].hist_drop.mean() < 0.8
    ok = pd.read_parquet(root / "rule_ok.parquet").astype({"episode": str}); ok = dict(zip(ok.episode + "/" + ok.i.astype(str), ok.rule_ok))
    r = by[by.source == "replay"]; assert (r.vkd_ok == r.key.map(ok).fillna(False)).all() and r.vkd_ok.any()   # unlisted rows: False
    a = by[(by.source == "new_a") & (by.ft_tier == 0)]
    assert set(a.ft_offset) == {0, 2, 3} and a.key.is_unique                        # ~4 Hz rows, each once
    assert set(by[(by.source != "new_a") & (by.ft_tier == 0)].ft_offset) == {0}       # old sources: 2 Hz rows


def test_synth_histories_and_start_rows(root):
    """user 10/28: lane-shaped histories ride along (synth_xyz), D start rows i < 15 join with them as their history and the
    real future; the recovery sources draw real / zero / lane-shaped a third each, the other GT rows real / zero."""
    syn = str(root / "_ft/synth_hist.parquet")
    d = MoraiDataset(root, "train", keep=str(root / "_ft/r2_D.parquet"), synth=syn, augment=False)
    extra = d.rows[d.rows.synth_only]; assert len(extra) and (extra.i < 15).all() and (extra.i >= 3).all()
    k = int(extra.index[0]); i = int(d.rows.i.iloc[k]); s = d.sample(k); ep = d._episode("r2-d-001")
    _, _, fx, _ = ego_trajectory(np.r_[np.full(15, ep.v[0]), ep.v], np.r_[np.full(15, ep.yaw[0]), ep.yaw], i + 15)
    assert torch.equal(s["ego_future_xyz"], fx) and torch.allclose(s["ego_history_xyz"][0, :, 0], torch.tensor(-ep.v[i] * 0.1 * np.arange(15, -1, -1)).float())
    assert "synth_xyz" not in s
    n = int(d.rows.index[(~d.rows.synth_only) & (d.rows.i == 20)][0]); assert d.sample(n)["synth_xyz"].shape == (1, 16, 3)
    src = lambda keep, **kw: {"root": str(root), "split": "train", "keep": str(root / "_ft" / keep), **kw}
    m = MixtureDataset({"new_a": {"share": 0.4, "loss": "gt", "new": True, "dataset": src("r2_A.parquet")},
                        "new_d": {"share": 0.4, "loss": "gt", "new": True, "hist": "thirds", "dataset": src("r2_D.parquet", synth=syn)},
                        "replay": {"share": 0.2, "loss": "distill", "dataset": src("old_static.parquet")}},
                       frame_stride=5, extra_offsets=(2, 3), hist_dropout=0.5, common={"augment": False})
    by = pd.DataFrame([{x: it[x] for x in ("source", "hist_drop", "hist_synth", "key")} | {"so": it["key"].split("/")[1]} for it in (m[k] for k in range(len(m)))])
    dd, aa = by[by.source == "new_d"], by[by.source == "new_a"]
    assert not aa.hist_synth.any() and 0.2 < aa.hist_drop.mean() < 0.8 and not by[by.source == "replay"][["hist_drop", "hist_synth"]].any().any()
    assert 0.1 < dd.hist_synth.mean() < 0.55 and 0.1 < dd.hist_drop.mean() < 0.55 and not (dd.hist_drop & dd.hist_synth).any()
    assert not dd[dd.so.astype(int) < 15][["hist_drop", "hist_synth"]].any().any() and (dd.so.astype(int) < 15).any()   # start rows
    assert {"hist_xyz", "hist_rot"} <= set(m[0]) and m.summary["hist_synth_share"] > 0


def test_choose_rule():
    h = lambda s, a: {"step": s, "ade": {"general": a}}
    assert choose([h(0, 1.0), h(500, 1.01), h(1000, 1.019)])[0]["step"] == 1000        # last holds the 2 % rule -> last
    assert choose([h(0, 1.0), h(500, 1.01), h(1000, 1.03)])[0]["step"] == 500          # last fails -> last that holds
    assert choose([h(0, 1.0), h(500, 1.05), h(1000, 1.03)])[0]["step"] == 0            # none -> s2 final itself
    assert choose([h(0, 1.0), h(500, 0.9), h(1000, 1.02)])[0]["step"] == 1000


@pytest.mark.parametrize("tiny", [False, True])
def test_student_ft_config(tiny, tmp_path):
    over = ["paths.s2=/s2/final", "paths.r2_root=/r2", "paths.start_relabel=/rel.parquet", "data.root=/data",
            f"paths.output_dir={tmp_path}", "trainer.deepspeed=null"] + (["+tiny=student_ft"] if tiny else [])
    with initialize_config_dir(str(CONFIGS), version_base=None):
        c = OmegaConf.to_container(compose("student_ft", overrides=over), resolve=True)
    TrainingArguments(**c["trainer"])
    m, d, t, cb = c["model"], c["data"]["train_dataset"], c["trainer"], c["callbacks"]
    from morai_vla.student import StudentVLA
    assert set(m) - {"_target_"} <= set(inspect.signature(StudentVLA.build).parameters) and m["init"] == "/s2/final"
    # user 10/28: hard targets, no T* (kd_proj only fed the T* ViT KD)
    assert m["teacher"] is None and set(m["train_parts"]) == set(StudentVLA.PARTS) - {"kd_proj"}
    assert m["loss_weights"] == {"kv": 0.0, "vit": 0.0, "fm": 1.0, "traj_ce": 1.0, "lm": 1.0, "vkd": 0.0, "coc_kl": 0.0}
    s = d["sources"]
    assert {k: v["share"] for k, v in s.items()} == {"new_a": 0.30, "new_b": 0.15, "new_d": 0.15, "old_recovery": 0.05, "old_curve": 0.05,
                                                      "start_relabel": 0.05, "replay": 0.25}
    assert {k for k, v in s.items() if v.get("new")} == {"new_a", "new_b", "new_d"} and [k for k, v in s.items() if v["loss"] == "distill"] == ["replay"]
    assert "rule_ok" not in s["replay"]["dataset"] and s["start_relabel"]["dataset"]["keep"] == "/rel.parquet"
    for v in s.values():
        assert set({**d["common"], **v["dataset"]}) <= set(inspect.signature(MoraiDataset).parameters)
    assert set(d) - {"_target_", "_recursive_"} <= set(inspect.signature(MixtureDataset).parameters)   # no base.yaml leftovers
    assert d["frame_stride"] == 5 and d["extra_offsets"] == [2, 3] and d["hist_dropout"] == 0.5 and "passes" not in d
    assert s["replay"]["dataset"]["exclude"] == "/rel.parquet" and all("exclude" not in v["dataset"] for k, v in s.items() if k != "replay")
    assert t["learning_rate"] == 3e-6 and t["lr_scheduler_kwargs"]["min_lr"] == 1e-6 and t["lr_multiplier"]["vlm.model.visual"] == 0.1
    assert 0.3 <= t["lr_multiplier"]["expert"] <= 0.5 and t["lr_multiplier"]["action_in_proj"] == t["lr_multiplier"]["expert"]
    assert not t["load_best_model_at_end"] and not t["gradient_checkpointing"] and (tiny or (t["warmup_steps"] == 200 and t["max_steps"] == -1))
    f = cb["ft_select"]
    assert f["_target_"] == "morai_vla.finetune.FtSelect" and f["ref"] == "general" and f["rel_worse"] is None and f["stop_rel_worse"] == 1.0 and f["stop_metric"] == "lat" and set(f["sets"]) == {"general", "dynamic", "static", "curve"}
    assert tiny or (f["every_steps"] == 500 and f["sets"]["general"] == [{"root": "/data", "split": ["val", "val_asset"], "frame_stride": 10, "n": 1000}])



def test_student_stop_configs(tmp_path):
    """user 10/28 09:10 stop round: the doc's shares (+ ring entry 5 % from replay), 1,000 steps at 1e-6 -> 3e-7, the lateral
    switch against s2's 0.230; its QAD = student_qad_r2's quantization / schedule on the same mixture; every pool file exists."""
    over = ["paths.s2=/ft/final", "paths.r2_root=/r2", "data.root=/data", f"paths.output_dir={tmp_path}", "trainer.deepspeed=null"]
    with initialize_config_dir(str(CONFIGS), version_base=None):
        c = OmegaConf.to_container(compose("student_stop", overrides=over), resolve=True)
        q = OmegaConf.to_container(compose("student_qad_stop", overrides=over + ["model.init=/search"]), resolve=True)
        r = OmegaConf.to_container(compose("student_qad_r2", overrides=over + ["paths.start_relabel=/rel.parquet", "model.init=/search"]), resolve=True)
    TrainingArguments(**c["trainer"]); TrainingArguments(**q["trainer"])
    src = c["data"]["train_dataset"]["sources"]
    assert {k: v["share"] for k, v in src.items()} == {"r2_aeb": .15, "r2_yield": .10, "red_approach": .15, "red_stop": .10, "highway": .10,
                                                      "new_d": .05, "old_recovery": .05, "ring_entry": .05, "start_relabel": .05, "replay": .20}
    assert abs(sum(v["share"] for v in src.values()) - 1) < 1e-9 and src["replay"]["loss"] == "distill"
    assert all(v["loss"] == "gt" for k, v in src.items() if k != "replay") and src["new_d"]["hist"] == "thirds" == src["old_recovery"]["hist"]
    t = c["trainer"]; assert (t["max_steps"], t["learning_rate"], t["warmup_steps"], t["lr_scheduler_kwargs"]["min_lr"]) == (1000, 1e-6, 50, 3e-7)
    f = c["callbacks"]["ft_select"]
    assert (f["stop_metric"], f["stop_rel_worse"], f["stop_patience"], f["stop_ref"], f["stop_base"]) == ("lat", 1.0, 2, "general_wo_relabel", 0.230)
    assert set(f["sets"]) == {"general", "red_stop", "dynamic_stop", "curve"} and c["model"]["init"] == "/ft/final"
    assert q["data"]["train_dataset"]["sources"] == src and q["callbacks"]["ft_select"] is None
    assert q["quant"] == r["quant"] and all(q["trainer"][k] == r["trainer"][k] for k in ("max_steps", "learning_rate", "warmup_steps", "eval_steps",
                                                                                         "save_steps", "metric_for_best_model", "lr_scheduler_kwargs"))
    files = [v["dataset"][k] for v in src.values() for k in ("keep", "exclude", "synth") if k in v["dataset"]] + \
            [p["keep"] for parts in f["sets"].values() for p in parts]
    assert all((CONFIGS.parent / x).exists() for x in files), [x for x in files if not (CONFIGS.parent / x).exists()]


def test_student_qad_r2_config(tmp_path):
    """user 10/28: QAD on the ft mixture with GT losses, no T*; student_qad's quantization and schedule."""
    over = ["paths.s2=/s2/final", "paths.r2_root=/r2", "paths.start_relabel=/rel.parquet", "data.root=/data", "model.init=/search",
            f"paths.output_dir={tmp_path}", "trainer.deepspeed=null"]
    with initialize_config_dir(str(CONFIGS), version_base=None):
        c = OmegaConf.to_container(compose("student_qad_r2", overrides=over), resolve=True)
        q = OmegaConf.to_container(compose("student_qad", overrides=["paths.teacher=/t", "data.train_dataset.rule_ok=/r", "data.root=/data",
                                                                     "model.init=/search", f"paths.output_dir={tmp_path}", "trainer.deepspeed=null"]), resolve=True)
        f = OmegaConf.to_container(compose("student_ft", overrides=over), resolve=True)
    TrainingArguments(**c["trainer"])
    t, tq = c["trainer"], q["trainer"]
    assert c["quant"] == q["quant"] and all(t[k] == tq[k] for k in ("max_steps", "learning_rate", "lr_multiplier", "warmup_steps", "eval_steps",
                                                                  "save_steps", "load_best_model_at_end", "metric_for_best_model"))
    assert c["data"]["train_dataset"] == f["data"]["train_dataset"] and c["model"]["loss_weights"] == f["model"]["loss_weights"]
    assert c["model"]["teacher"] is None and c["model"]["init"] == "/search"
    assert c["callbacks"]["ft_select"] is None and "vkd_weight" not in c["callbacks"] and "eval_fm" in c["callbacks"]

# ------------------------------------------------------------------ model: per-row losses and history dropout
TINY_T = dict(text_config=dict(hidden_size=256, intermediate_size=512, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                               rope_scaling=dict(rope_type="default", mrope_section=[12, 10, 10], mrope_interleaved=True)),
              vision_config=dict(depth=4, hidden_size=128, intermediate_size=256, num_heads=4, out_hidden_size=256, deepstack_visual_indexes=[1, 2, 3]))
EXPERT = dict(hidden_size=128, intermediate_size=256, num_attention_heads=4, head_dim=64)
ALL = ["camera_vit", "lidar", "llm", "lm_head", "kd_proj", "expert"]
W = dict(kv=1.0, vit=10.0, fm=1.0, traj_ce=1.0, lm=1.0, vkd=0.2, coc_kl=0.0)


@pytest.fixture(scope="module")
def student(tmp_path_factory):
    from morai_vla.model import TeacherVLA
    from morai_vla.student import StudentVLA
    if RELEASE is None: pytest.skip("no release config")
    d = tmp_path_factory.mktemp("teacher")
    TeacherVLA.build("random:" + RELEASE, vlm_name_or_path=V4, attn_implementation="sdpa", dtype="float32", vlm_overrides=TINY_T,
                     expert_overrides=EXPERT).save_pretrained(d)
    ts = OmegaConf.to_container(OmegaConf.load(CONFIGS / "tiny/student.yaml").model.vlm_overrides)
    return StudentVLA.build(teacher=str(d), init="random", vlm_name_or_path=V4, teacher_vlm=V4, attn_implementation="sdpa", dtype="float32",
                            vlm_overrides=ts, train_parts=ALL, loss_weights=W)


def batch(S, gt, drop, vkd):
    from morai_vla.processor import collate_fn, get_preprocess_fn
    from test_layout import sample
    rows = []
    for k in range(len(gt)):
        s = sample(lidar=True)
        s["ego_history_xyz"][0, :, 0] = torch.linspace(-7.5 - k, 0, 16); s["ego_future_xyz"][0, :, 0] = torch.linspace(0.5, 32 + 4 * k, 64)
        s["tokenized_data"] = get_preprocess_fn(S.config)(data=s)
        for x in ("camera_frames", "bev_frames", "camera_times", "bev_times"): s.pop(x, None)
        rows.append(s | {"gt_row": gt[k], "hist_drop": drop[k], "vkd_ok": vkd[k]})
    b = collate_fn(rows, model_config=S.config)
    return {k: v for k, v in b.items() if k in ("tokenized_data", "ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot",
                                                 "labels_mask", "vkd_ok", "gt_row", "hist_drop")}


def run(S, b, seed=0):
    S.loss_sums.clear(); torch.manual_seed(seed)
    loss = S(**b).loss
    return loss, {k: s / n for k, (s, n) in S.loss_sums.items()}


@need_release
def test_per_row_losses(student):
    S = student; S.train()
    _, gt = run(S, batch(S, [True, True], [False, False], [False, False]))
    assert set(gt) == {"vit", "fm", "traj_ce"}, gt                                   # GT rows: no T* KV, no velocity KD, no lm
    _, rep = run(S, batch(S, [False, False], [False, False], [True, True]))
    assert set(rep) == {"kv", "vit", "fm", "traj_ce", "lm", "vkd"} and rep["vkd"] > 0 and rep["kv"] > 0
    _, mixed = run(S, batch(S, [True, False], [False, False], [False, True]))
    assert mixed["kv"] > 0 and mixed["vkd"] > 0                                      # computed on the replay row
    _, gated = run(S, batch(S, [True, False], [False, False], [False, False]))
    assert gated["vkd"] == 0                                                         # replay row without rule_ok: no vkd
    loss, _ = run(S, batch(S, [True, False], [True, False], [False, True])); loss.backward()
    g = {p: sum(float(x.grad.abs().sum()) for path in paths for x in S.get_submodule(path).parameters() if x.grad is not None) for p, paths in S.PARTS.items()}
    assert all(g[p] > 0 for p in ("camera_vit", "lidar", "llm", "kd_proj", "expert")), g
    S.zero_grad(set_to_none=True)


@need_release
def test_kv_only_on_replay_rows(student):
    """kv of a [GT, replay] batch = kv of the replay row alone (the GT row's prefix tokens are masked out)."""
    from morai_vla.student import kv_mse
    S = student; S.eval()
    b = batch(S, [True, False], [False, False], [False, True])
    td = dict(b["tokenized_data"]); lid = td.pop("image_is_lidar")
    ids = S.fuse_traj_tokens(td.pop("input_ids"), {k: b[k] for k in ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot")})
    with torch.no_grad():
        S.vlm.model._image_is_lidar = lid; o = S.vlm(input_ids=ids, use_cache=True, logits_to_keep=1, **td)
        S.teacher.vlm.model._image_is_lidar = lid; t = S.teacher.vlm(input_ids=ids, use_cache=True, logits_to_keep=1, **td)
    start = int((ids == S.config.traj_token_ids["future_start"]).nonzero()[-1, 1]) + 1
    both = kv_mse(o.past_key_values, t.past_key_values, td["attention_mask"], start)[0]
    rep = kv_mse(o.past_key_values, t.past_key_values, td["attention_mask"], start, torch.tensor([False, True]))[0]
    gtr = kv_mse(o.past_key_values, t.past_key_values, td["attention_mask"], start, torch.tensor([True, False]))[0]
    assert rep != both and gtr != both and min(rep, gtr) <= both <= max(rep, gtr)
    S.train()


@need_release
def test_history_dropout_tokens_only(student, monkeypatch):
    S = student; S.train()
    b = batch(S, [True, True], [True, False], [False, False])
    td = dict(b["tokenized_data"]); raw = td["input_ids"]
    traj = {k: b[k] for k in ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot")}
    full = S.fuse_traj_tokens(raw, traj)
    dropped = S.drop_history_tokens(raw, full, traj, torch.tensor([True, False]))
    hist = raw == S.config.traj_token_ids["history"]
    assert torch.equal(dropped[1], full[1]) and torch.equal(dropped[0][~hist[0]], full[0][~hist[0]])     # only row 0's history tokens
    assert not torch.equal(dropped[0][hist[0]], full[0][hist[0]])
    zero = S.fuse_traj_tokens(raw, {"ego_history_xyz": torch.zeros_like(traj["ego_history_xyz"]),
                                    "ego_history_rot": torch.eye(3).expand_as(traj["ego_history_rot"]).contiguous()})
    assert torch.equal(dropped[0][hist[0]], zero[0][hist[0]])                       # = a car standing still (deployment, no history)
    seen = {}
    orig = S._process_traj_future_training
    monkeypatch.setattr(S, "_process_traj_future_training", lambda t: seen.update(t) or orig(t))
    run(S, b)
    assert torch.equal(seen["ego_history_xyz"], traj["ego_history_xyz"])            # the (a, kappa) targets keep the real history


def test_static_clearance():
    """scripts/ft_measure.clearance: body box (rear -1, front 3.9, half width 0.95) driving +x past a 2 x 1 m box."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.ft_measure import clearance
    from scripts.rule_check import Rules
    r = object.__new__(Rules); r.ep_index = {"ep": 0}; r.z = {"body": np.array([-1.0, 3.9, 0.95])}
    ego = {"episode": "ep", "x": np.array([0.0]), "y": np.array([0.0]), "yaw": np.array([0.0])}
    path = np.stack([np.linspace(0.5, 32, 64), np.zeros(64)], -1)[None]
    for cy, want in ((3.0, 3.0 - 0.5 - 0.95), (-2.0, 2.0 - 0.5 - 0.95), (0.5, 0.0)):
        r.obs = np.array([[0, 15.0, cy, 0.0, 1.0, 0.5]])
        cl, hit = clearance(r, ego, np.array([0]), path)
        assert abs(cl[0] - want) < 1e-6 and bool(hit[0]) == (want == 0.0), (cy, cl, hit)
    r.ep_index = {}
    assert np.isnan(clearance(r, ego, np.array([0]), path)[0][0])                     # no static obstacle in the episode


def test_stop_switch_reads_general_without_relabel_rows(tmp_path, monkeypatch):
    """user 10/28: FtSelect records general_wo_relabel = general minus the start-relabelled (episode, i) and the stop switch
    reads it (two measurements in a row > 10 % above step 0); general itself stays the selection set. No model."""
    from types import SimpleNamespace
    from morai_vla import finetune
    pd.DataFrame({"episode": ["e"], "i": [9]}).to_parquet(tmp_path / "relabel.parquet")
    ades = iter([{"e/1": 1.0, "e/9": 1.0}, {"e/1": 1.0, "e/9": 9.0}, {"e/1": 1.2, "e/9": 1.0}, {"e/1": 1.2, "e/9": 1.0}])
    monkeypatch.setattr(finetune, "expert_ade", lambda *a, **k: [(key, {"ade": v, "fde": v, "lat": 0.0, "lon": 0.0}) for key, v in next(ades).items()])
    cb = finetune.FtSelect({"general": []}, every_steps=1, rel_worse=None, stop_rel_worse=0.10, stop_ref="general_wo_relabel",
                           stop_exclude=str(tmp_path / "relabel.parquet"))
    cb.samples = {"general": None}
    args, model = SimpleNamespace(output_dir=str(tmp_path)), torch.nn.Linear(1, 1)
    st = lambda s: SimpleNamespace(global_step=s, max_steps=10, is_world_process_zero=True)
    cb.on_train_begin(args, st(0), None, model=model)                             # step 0: both 1.0
    hits = []
    for s in (1, 2, 3):
        control = SimpleNamespace(should_training_stop=False, should_save=False)
        cb.on_step_end(args, st(s), control, model=model); hits.append(control.should_training_stop)
    h = {x["step"]: x for x in cb.st["history"]}
    assert h[1]["ade"] == {"general": 5.0, "general_wo_relabel": 1.0} and h[1]["n"]["general_wo_relabel"] == 1
    assert hits == [False, False, True]                                           # step 1: general +400 % but wo_relabel 0 %
    assert json.load(open(tmp_path / "ft_stop.json"))["stop_ref"] == "general_wo_relabel"


def test_stop_switch_lateral(tmp_path, monkeypatch):
    """user 10/28: the switch reads general_wo_relabel's lateral error (+100 % = 2x s2), not its ADE."""
    import torch
    from types import SimpleNamespace
    from morai_vla import finetune
    pd.DataFrame({"episode": ["e"], "i": [9]}).to_parquet(tmp_path / "relabel.parquet")
    lats = iter([0.2, 0.5, 0.3, 0.5, 0.5])                                         # ADE stays 1.0 (+0 %) throughout
    monkeypatch.setattr(finetune, "expert_ade", lambda *a, **k: [("e/1", {"ade": 1.0, "fde": 1.0, "lat": next(lats), "lon": 0.0})])
    cb = finetune.FtSelect({"general": []}, every_steps=1, rel_worse=None, stop_rel_worse=1.0, stop_ref="general_wo_relabel",
                           stop_exclude=str(tmp_path / "relabel.parquet"), stop_metric="lat")
    cb.samples = {"general": None}
    args, model = SimpleNamespace(output_dir=str(tmp_path)), torch.nn.Linear(1, 1)
    st = lambda s: SimpleNamespace(global_step=s, max_steps=10, is_world_process_zero=True)
    cb.on_train_begin(args, st(0), None, model=model)                             # step 0: lat 0.2 -> limit 0.4
    hits = []
    for s in (1, 2, 3, 4):
        control = SimpleNamespace(should_training_stop=False, should_save=False)
        cb.on_step_end(args, st(s), control, model=model); hits.append(control.should_training_stop)
    assert hits == [False, False, False, True]                                    # 0.5, 0.3 (reset), 0.5, 0.5
    j = json.load(open(tmp_path / "ft_stop.json")); assert j["stop_metric"] == "lat" and j["step"] == 4
    cb2 = finetune.FtSelect({"general": []}, every_steps=1, rel_worse=None, stop_rel_worse=1.0, stop_ref="general_wo_relabel",
                            stop_exclude=str(tmp_path / "relabel.parquet"), stop_metric="lat", stop_base=0.3)
    cb2.st = {"history": [{"step": 0, "ade": {}, "metrics": {"general_wo_relabel": {"lat": 0.2}}}] +
              [{"step": s, "ade": {}, "metrics": {"general_wo_relabel": {"lat": 0.5}}} for s in (1, 2)]}
    assert not cb2._stop_hit()                                                    # 0.5 < 2 x 0.3 (stop_base), though > 2 x 0.2 (step 0)
