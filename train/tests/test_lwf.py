"""LwF self-distillation (configs/student_lwf.yaml, reports/lwf_plan_20261029.md): the teacher is a frozen copy of a saved
student; kd rows (MixtureDataset `kd: true`, minus red-signal rows) take kv / vkd / traj_kl from it and their GT at kd_gt;
answer rows stay GT only and skip the teacher; at step 0 (student = teacher) the distillation losses are 0, also on rows
whose history tokens were dropped (the teacher sees the student's tokens); with kd_row = ~gt_row & rule_ok the losses equal
the replay path's. CPU, same environment as tests/test_finetune.py (1.5 release config.json + Qwen3-VL-4B processor)."""
import inspect, os
from pathlib import Path

import numpy as np, pandas as pd, pytest, torch
from hydra import compose, initialize_config_dir
from huggingface_hub.constants import HF_HUB_CACHE
from omegaconf import OmegaConf

from alpamayo1_5_sft.trainer import TrainingArguments
from ft_synth import make
from morai_vla.finetune import MixtureDataset, signal_rows

RELEASE = os.environ.get("ALPAMAYO15") or next((str(p.parent) for p in Path(HF_HUB_CACHE).glob("models--nvidia--Alpamayo-1.5-10B/snapshots/*/config.json")), None)
need_release = pytest.mark.skipif(RELEASE is None, reason="Alpamayo-1.5-10B config.json not found (HF cache or $ALPAMAYO15)")
CONFIGS = Path(__file__).resolve().parents[1] / "configs"
V4 = "Qwen/Qwen3-VL-4B-Instruct"
ROPE = dict(rope_type="default", mrope_section=[12, 10, 10], mrope_interleaved=True)
TINY_T = dict(text_config=dict(hidden_size=256, intermediate_size=512, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=64, rope_scaling=ROPE),
              vision_config=dict(depth=4, hidden_size=128, intermediate_size=256, num_heads=4, out_hidden_size=256, deepstack_visual_indexes=[1, 2, 3]))
NARROW = dict(text_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2, head_dim=64, rope_scaling=ROPE),
              vision_config=dict(depth=4, hidden_size=64, intermediate_size=128, num_heads=2, out_hidden_size=64, deepstack_visual_indexes=[1, 2, 3]))   # = scripts/ft_tiny.sh
LW = dict(kv=1.0, vit=0.0, fm=1.0, traj_ce=1.0, lm=1.0, vkd=1.0, coc_kl=0.0, traj_kl=1.0, kd_gt=1.0)
KEYS = ("tokenized_data", "ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot", "labels_mask", "vkd_ok", "gt_row", "hist_drop", "kd_row")


@pytest.fixture(scope="module")
def lwf(tmp_path_factory):
    """a tiny saved 'R2 ft final' (student from a tiny T*) and the LwF student built from it with itself as the teacher."""
    from morai_vla.model import TeacherVLA
    from morai_vla.student import StudentVLA
    if RELEASE is None: pytest.skip("no release config")
    d = tmp_path_factory.mktemp("lwf")
    TeacherVLA.build("random:" + RELEASE, vlm_name_or_path=V4, attn_implementation="sdpa", dtype="float32", vlm_overrides=TINY_T,
                     expert_overrides=dict(hidden_size=128, intermediate_size=256, num_attention_heads=4, head_dim=64)).save_pretrained(d / "tstar")
    StudentVLA.build(teacher=str(d / "tstar"), init="random", vlm_name_or_path=V4, teacher_vlm=V4, attn_implementation="sdpa", dtype="float32",
                     vlm_overrides=NARROW, distill=False).save_pretrained(d / "r2ft")
    return StudentVLA.build(teacher=str(d / "r2ft"), init=str(d / "r2ft"), self_teacher=True, vlm_name_or_path=V4, attn_implementation="sdpa",
                            dtype="float32", train_parts=["camera_vit", "lidar", "llm", "lm_head", "expert"], loss_weights=LW)


def batch(S, gt, kd=None, drop=None, vkd=None):
    from morai_vla.processor import collate_fn, get_preprocess_fn
    from test_layout import sample
    rows, n = [], len(gt)
    for k in range(n):
        s = sample(lidar=True)
        s["ego_history_xyz"][0, :, 0] = torch.linspace(-7.5 - k, 0, 16); s["ego_future_xyz"][0, :, 0] = torch.linspace(0.5, 32 + 4 * k, 64)
        s["tokenized_data"] = get_preprocess_fn(S.config)(data=s)
        for x in ("camera_frames", "bev_frames", "camera_times", "bev_times"): s.pop(x, None)
        rows.append(s | {"gt_row": gt[k], "hist_drop": (drop or [False] * n)[k], "vkd_ok": (vkd or [False] * n)[k]} | ({} if kd is None else {"kd_row": kd[k]}))
    return {k: v for k, v in collate_fn(rows, model_config=S.config).items() if k in KEYS}


def run(S, b, seed=0, **w):
    S.loss_weights.update(LW | w); S.loss_sums.clear(); torch.manual_seed(seed)
    loss = S(**b).loss
    return loss, {k: s / n for k, (s, n) in S.loss_sums.items()}


@need_release
def test_self_teacher_is_a_frozen_copy(lwf):
    S, T = lwf, lwf.teacher
    assert type(T) is type(S) and not T.training and not any(p.requires_grad for p in T.parameters())
    assert "teacher" not in dict(S.named_children())                                 # not saved, optimised or wrapped
    sd = S.state_dict(); assert all(torch.equal(sd[k], v) for k, v in T.state_dict().items())


@need_release
def test_zero_at_step0_with_the_students_history_tokens(lwf):
    S = lwf; S.train()
    b, seen = batch(S, [True, False], kd=[True, True], drop=[True, False]), {}     # row 0: zero-displacement history tokens
    hs = [m.register_forward_pre_hook(lambda mod, a, kw, n=n: seen.__setitem__(n, kw["input_ids"]), with_kwargs=True) for n, m in (("s", S.vlm), ("t", S.teacher.vlm))]
    try:
        _, l = run(S, b)
    finally:
        for h in hs: h.remove()
    traj = {k: b[k] for k in ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot")}; raw = b["tokenized_data"]["input_ids"]
    zero = S.drop_history_tokens(raw, S.fuse_traj_tokens(raw, traj), traj, torch.tensor([True, False]))
    assert torch.equal(seen["t"], seen["s"]) and torch.equal(seen["t"], zero)       # the teacher reads the zero history too
    assert {"kv", "vkd", "traj_kl", "fm", "traj_ce"} <= set(l), l
    assert l["kv"] < 1e-12 and l["vkd"] < 1e-12 and abs(l["traj_kl"]) < 1e-7 and l["fm"] > 0 and l["traj_ce"] > 0, l


@need_release
def test_answer_rows_gt_only_without_teacher_pass(lwf):
    S = lwf; S.train(); calls = []
    h = S.teacher.vlm.register_forward_hook(lambda *a: calls.append(1))
    try:
        _, l = run(S, batch(S, [True, True], kd=[False, False]))
    finally:
        h.remove()
    assert set(l) == {"fm", "traj_ce"} and not calls, (l, calls)


@need_release
def test_drift_replay_equivalence_and_gt_weight(lwf):
    S = lwf; S.train(); keep = {k: v.clone() for k, v in S.state_dict().items()}
    try:
        with torch.no_grad():                                                        # the student moves away from the teacher
            for p in S.vlm.model.language_model.layers[0].parameters(): p.add_(0.05 * torch.randn_like(p))
        b = batch(S, [True, False], kd=[False, True])
        _, l = run(S, b)
        assert l["kv"] > 0 and l["vkd"] > 0 and l["traj_kl"] > 0, l
        # kd_row = ~gt_row & rule_ok reproduces the replay path (kv on replay rows, vkd on rule_ok replay rows)
        _, legacy = run(S, {k: v for k, v in b.items() if k != "kd_row"} | {"vkd_ok": [False, True]}, traj_kl=0.0)
        _, new = run(S, b | {"vkd_ok": [False, False]}, traj_kl=0.0)
        assert legacy.keys() == new.keys() and all(legacy[k] == pytest.approx(new[k], rel=1e-6) for k in legacy), (legacy, new)
        # GT on kd rows x kd_gt (fm per row, traj_ce per token); 0 = teacher only
        k2 = batch(S, [True, True], kd=[True, True])
        full, half, zero = (run(S, k2, kd_gt=g)[1] for g in (1.0, 0.5, 0.0))
        for x in ("fm", "traj_ce"): assert half[x] == pytest.approx(0.5 * full[x], rel=1e-5) and zero[x] == 0, (x, full, half, zero)
        assert half["vkd"] == pytest.approx(full["vkd"], rel=1e-6)
    finally:
        S.load_state_dict(keep); S.loss_weights.update(LW)


def test_mixture_kd_rows_skip_red(tmp_path):
    root = make(tmp_path / "root")
    for ep in ("old-001", "old-002"):                                                # R1-style signal column: 1 = red at rows 40-79
        f = root / "_ego" / f"{ep}.npz"; z = dict(np.load(f)); s = np.zeros(len(z["tau"]), np.int8); s[40:80] = 1; s[80:] = 3
        np.savez(f, **z, signal=s)
    src = lambda keep=None: {"root": str(root), "split": "train", **({"keep": str(root / "_ft" / keep)} if keep else {})}
    sources = {"stop": {"share": 0.4, "loss": "gt", "new": True, "dataset": src("r2_A.parquet")},
               "lane": {"share": 0.3, "loss": "gt", "new": True, "kd": True, "dataset": src("r2_D.parquet")},
               "replay": {"share": 0.3, "loss": "distill", "kd": True, "dataset": src()}}
    m = MixtureDataset(sources, frame_stride=5, extra_offsets=(2, 3), hist_dropout=0.5, common={"augment": False})
    red = lambda ep, i: pd.Series(ep).astype(str).isin(["old-001", "old-002"]).to_numpy() & (np.asarray(i) >= 40) & (np.asarray(i) < 80)
    p = m.parts[m.names.index("replay")]; r = red(p.rows.episode, p.rows.i)
    assert r.any() and (m.kd[m.names.index("replay")] == ~r).all() and m.kd[m.names.index("stop")] is None
    assert m.summary["sources"]["replay"]["signal_rows"] == int(r.sum()) and m.summary["sources"]["lane"]["signal_rows"] == 0
    by = pd.DataFrame([{x: it[x] for x in ("source", "kd_row", "hist_drop", "key")} for it in (m[k] for k in range(len(m)))])
    ep, i = by.key.str.split("/").str[0], by.key.str.split("/").str[1].astype(int)
    assert (by.kd_row == (by.source.isin(["lane", "replay"]) & ~red(ep, i))).all()
    assert 0.2 < by[by.source == "lane"].hist_drop.mean() < 0.8                      # kd sources keep their history modes
    assert m.summary["kd_share"] == round(float(by.kd_row.mean()), 4)
    soon = red(p.rows.episode, p.rows.i + 10)                                        # kd_skip_horizon 10: red now or within 1 s
    assert (signal_rows(p, [1], 10) == (r | soon)).all() and (soon & ~r).any()
    legacy = MixtureDataset({k: {x: y for x, y in v.items() if x != "kd"} for k, v in sources.items()}, frame_stride=5, common={"augment": False})
    assert "kd_row" not in legacy[0] and "kd_share" not in legacy.summary            # configs without kd: items as before
    # replay red_split (user 10/29 "replay 전체"): kd rows real / zero; red rows thirds with a synthetic history, else real / zero.
    # The synth file has no index columns and no extra rows (as dyn_synth_hist's)
    syn = {("old-001", i) for i in range(40, 80, 5)}
    pd.DataFrame({"episode": [e for e, _ in syn], "i": [i for _, i in syn], "method": "straight", "hist_x": [np.linspace(-7.5, 0, 16)] * len(syn),
                  "hist_y": [np.zeros(16)] * len(syn), "hist_yaw": [np.zeros(16)] * len(syn), "extra": False}).to_parquet(tmp_path / "syn.parquet")
    rs = {**sources, "replay": {**sources["replay"], "hist": "red_split", "dataset": {**src(), "synth": str(tmp_path / "syn.parquet")}}}
    m2 = MixtureDataset(rs, frame_stride=5, extra_offsets=(2, 3), hist_dropout=0.5, common={"augment": False}, epochs=8)
    j = m2.names.index("replay"); rows = m2.parts[j].rows
    modes = pd.DataFrame([(rows.episode.iat[r], int(rows.i.iat[r]), *m2._hist_mode(k)) for k, (jj, r, _) in enumerate(m2.schedule) if jj == j],
                         columns=["ep", "i", "hdrop", "hsyn"])
    rd, hs = red(modes.ep, modes.i), np.array([(e, i) in syn for e, i in zip(modes.ep, modes.i)])
    assert not modes.hsyn[~(rd & hs)].any() and modes.hsyn[rd & hs].any() and modes.hdrop[rd & hs].any()
    assert 0.3 < modes.hdrop[~rd].mean() < 0.7 and len(modes[rd & ~hs])                       # kd rows: real / zero
    it = m2[int(next(k for k, (jj, r, _) in enumerate(m2.schedule) if jj == j and m2._hist_mode(k)[1]))]
    assert it["hist_synth"] and not it["kd_row"]                                             # a red row reads its synthetic history
    # kd_all (LwF QAD, the ft final as teacher): every row of every source, red rows too
    m3 = MixtureDataset(sources, frame_stride=5, extra_offsets=(2, 3), common={"augment": False}, kd_all=True)
    assert all(k is not None and k.all() for k in m3.kd) and m3[0]["kd_row"] and m3.summary["kd_share"] == 1.0


def test_lwf_configs(tmp_path):
    from morai_vla.student import StudentVLA
    over = ["paths.s2=/s2/final", "paths.r2_root=/r2", "data.root=/data", f"paths.output_dir={tmp_path}", "trainer.deepspeed=null"]
    with initialize_config_dir(str(CONFIGS), version_base=None):
        l, d, q, qd, qg = (OmegaConf.to_container(compose(n, overrides=over + x), resolve=True)
                           for n, x in (("student_lwf", []), ("student_dyn", []), ("student_qad_lwf", ["model.init=/search"]),
                                        ("student_qad_dyn", ["model.init=/search"]), ("student_qad_lwf_gt", ["model.init=/search"])))
        bo = OmegaConf.to_container(compose("student_lwf_boost", overrides=over + ["paths.ft_ckpt=/out/ft/checkpoint-1000"]), resolve=True)
    for c in (l, q): TrainingArguments(**c["trainer"])
    m, td = l["model"], l["data"]["train_dataset"]
    assert set(m) - {"_target_"} <= set(inspect.signature(StudentVLA.build).parameters)
    assert set(td) - {"_target_", "_recursive_"} <= set(inspect.signature(MixtureDataset).parameters)
    assert m["self_teacher"] and m["teacher"] == m["init"] == "/s2/final" and td["kd_skip_signal"] == [1] and td["kd_skip_horizon"] == 64
    assert l["paths"]["teacher"] is None and q["paths"]["teacher"] is None   # gate_student.py builds paths.teacher as a T*: gates stay student-only
    assert m["loss_weights"] == {"kv": 0.0, "vit": 0.0, "fm": 1.0, "traj_ce": 1.0, "lm": 1.0, "vkd": 1.0, "coc_kl": 0.0, "traj_kl": 0.0, "kd_gt": 1.0}
    s, sd = td["sources"], d["data"]["train_dataset"]["sources"]; kd = {k for k, v in s.items() if v.get("kd")}
    assert kd == {"lane_r2_D", "lane_old_recovery", "replay"} | {k for k in s if k.startswith("static_")}   # 제안: static teacher, dyn pass GT
    assert not kd & {k for k in s if k.startswith(("dyn_stop", "popout", "red_", "dyn_pass"))}
    # recipe (reports/lwf_dynstop_recipe_20261029.md 4): group totals, onset / rest splits, thirds histories, 2,000 steps
    group = lambda p: round(sum(v["share"] for k, v in s.items() if k.startswith(p)), 4)
    assert [group(p) for p in ("popout_dyn", "popout_tunnel", "popout_cruise", "dyn_stop", "dyn_pass", "static_", "red_", "lane_", "replay")][2:] \
        == [0.05, 0.05, 0.05, 0.15, 0.10, 0.15, 0.25] and round(group("popout_dyn") + group("popout_tunnel"), 4) == 0.20   # main 10/29: cruise 5, replay 25
    assert abs(sum(v["share"] for v in s.values()) - 1) < 1e-4
    cr = s["popout_cruise"]; assert cr["loss"] == "gt" and "kd" not in cr and "hist" not in cr and cr["dataset"]["index"].endswith("_dyn_relabel_v2/popout_index.parquet")
    hard = {k for k in s if k.startswith(("popout", "dyn_stop")) and not k.endswith(("_rh", "_hold")) and k != "popout_cruise"}
    assert all(k.endswith(("_onset", "_rest")) and "mix_lwf" in s[k]["dataset"]["keep"] for k in hard) and len(hard) >= 20
    assert all(s[k]["hist"] == "thirds" for k in s if k in hard or k.startswith(("red_", "dyn_stop", "lane_")))
    assert not any(k.endswith("_rh") for k in s)                                       # user 10/29: no pop-out restart rows
    same = lambda v: {x: y for x, y in v.items() if x not in ("kd", "share", "hist")}
    assert all(same(s[k]) == same(sd[k]) for k in s if k.startswith(("dyn_pass", "static_", "red_", "lane_")))
    assert {x: y for x, y in s["replay"]["dataset"].items() if x != "synth"} == sd["replay"]["dataset"]
    assert {k: v for k, v in td.items() if k not in ("sources", "kd_skip_signal", "kd_skip_horizon", "epochs")} == {k: v for k, v in d["data"]["train_dataset"].items() if k not in ("sources", "epochs")}
    assert td["epochs"] == 3 and l["trainer"]["max_steps"] == 2000 and {k: v for k, v in l["trainer"].items() if k != "max_steps"} == {k: v for k, v in d["trainer"].items() if k != "max_steps"}
    fs, fd = l["callbacks"]["ft_select"], d["callbacks"]["ft_select"]
    assert set(fs["sets"]) == set(fd["sets"]) | {"popout_onset", "popout_pre", "static"} and {k: v for k, v in fs.items() if k != "sets"} == {k: v for k, v in fd.items() if k != "sets"}
    assert m["train_parts"] == d["model"]["train_parts"]
    # QAD (user 10/29 "이걸로 하자"): teacher = the ft final next to the QAD output (pipeline layout), every row kd, LwF losses and
    # mixture, student_qad_dyn's quantization and schedule; memory fallback student_qad_lwf_gt = student_qad_dyn's losses
    assert q["model"]["teacher"] == f"{tmp_path}/../ft/final" and q["model"]["init"] == "/search" and q["model"]["self_teacher"]
    qt = q["data"]["train_dataset"]
    assert qt["kd_all"] and {k: v for k, v in qt.items() if k != "kd_all"} == l["data"]["train_dataset"] and q["model"]["loss_weights"] == m["loss_weights"]
    assert q["quant"] == qd["quant"] and q["trainer"] == qd["trainer"] and q["callbacks"] == qd["callbacks"]
    assert qg["model"]["teacher"] is None and not qg["data"]["train_dataset"]["kd_all"] and qg["quant"] == q["quant"] and qg["trainer"] == q["trainer"]
    assert {k: v for k, v in qg["model"]["loss_weights"].items() if k not in ("traj_kl", "kd_gt")} == qd["model"]["loss_weights"]
    rp = s["replay"]; assert rp["hist"] == "red_split" and rp["kd"] and rp["dataset"]["synth"].endswith("synth_hist_dyn.parquet")
    # emergency variant (main 10/29): pop-outs 35, replay 10 (sum 1), decision rows x8, 1,000 steps from checkpoint-1000, same teacher
    bs = bo["data"]["train_dataset"]["sources"]; bg = lambda p: round(sum(v["share"] for k, v in bs.items() if k.startswith(p)), 4)
    assert (round(bg("popout_dyn") + bg("popout_tunnel"), 4), bg("replay"), bo["trainer"]["max_steps"]) == (0.35, 0.10, 1000) and set(bs) == set(s)
    assert abs(sum(x["share"] for x in bs.values()) - 1) < 1e-3
    assert bo["model"]["init"] == "/out/ft/checkpoint-1000" and bo["model"]["teacher"] == "/s2/final" and bo["model"]["loss_weights"] == m["loss_weights"]
    # user 10/29 "켜": near-shoulder stop cells ~8 passes (lwf_mix --near-passes 8): far above their natural pop-out share
    near = sum(v["share"] for k, v in s.items() if "_near_hard" in k)
    assert 0.01 < near < 0.05, near
