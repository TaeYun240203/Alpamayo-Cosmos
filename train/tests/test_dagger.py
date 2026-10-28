"""DAgger runner parts (train/dagger): 4 Hz row choice, label checks, the DAgger-row loss (= R2 ft's GT-row fm and traj_ce
when the label is the GT future; valid mask), LLM checkpointing that keeps the KV (same gradients), llm+expert LoRA
names = delivered names and merge. CPU, quantizers off. The model tests need the tiny delivery of
scripts/student_dryrun.sh ($TINY_DELIVER, default ~/claude-bench/rl-tiny/deliver) and the eval repo's engine code
($E2E_SERVER) for the spool test."""
import os
from pathlib import Path

import numpy as np, pytest, torch

from dagger import spool
from dagger.loss import checkpoint_llm, dagger_forward, normalize

DELIVER = Path(os.environ.get("TINY_DELIVER", "~/claude-bench/rl-tiny/deliver")).expanduser()
SERVER = os.environ.get("E2E_SERVER", str(Path(__file__).resolve().parents[3] / "MORAI-eval-sync/src/e2e_driver/server"))
need_model = pytest.mark.skipif(not (DELIVER / "final/config.json").exists(), reason=f"no tiny delivery at {DELIVER}")
need_server = pytest.mark.skipif(not (Path(SERVER) / "assemble.py").exists(), reason=f"no engine code at {SERVER}")


def test_rows_4hz():
    t = 100 + np.arange(40) * 0.1 + np.random.default_rng(0).uniform(-0.01, 0.01, 40)
    sel = spool.rows_4hz(t)
    g = np.round(t[sel] / 0.25)
    assert len(np.unique(g)) == len(sel) and np.abs(t[sel] - g * 0.25)[:-1].max() <= 0.061   # one per grid point; 0.05 + jitter
    assert len(sel) == len(np.unique(np.round(t / 0.25)))
    late = np.array([0.0, 0.1, 0.2, 0.7, 0.8, 1.3])                                        # 0.25 and 0.5 have no decision near
    assert spool.rows_4hz(late).tolist() == [0, 2, 3, 5]      # 0.0->0, 0.2->0.25, 0.7 / 0.8 -> 0.75 (tie: first), 1.3->1.25; 0.5, 1.0 none


def test_read_labels(tmp_path):
    f = tmp_path / "l.npz"
    ak = np.zeros((3, 64, 2), np.float32); ak[1, 5, 0] = np.nan
    np.savez(f, step=np.array([0, 2, 4], np.int32), ak=ak, valid=np.ones((3, 64), bool), source=np.array([0, 1, 0], np.int8))
    lab = spool.read_labels(f, 5)
    assert sorted(lab) == [0, 2, 4] and not lab[2][1][5] and lab[2][1].sum() == 63 and np.isfinite(lab[2][0]).all()
    assert lab[0][3:] == (0, np.inf)                                                     # optional signal / stop_d absent
    np.savez(tmp_path / "s.npz", step=np.array([0], np.int32), ak=ak[:1], valid=np.ones((1, 64), bool), source=np.zeros(1, np.int8),
             signal=np.array([1], np.int8), stop_d=np.array([25.0], np.float32))
    assert spool.read_labels(tmp_path / "s.npz", 5)[0][3:] == (1, 25.0)
    with pytest.raises(AssertionError):
        spool.read_labels(f, 4)                                                           # step 4 past the episode
    np.savez(f, step=np.array([0], np.int32), ak=np.zeros((1, 32, 2), np.float32), valid=np.ones((1, 32), bool), source=np.zeros(1, np.int8))
    with pytest.raises(AssertionError):
        spool.read_labels(f, 4)


@need_server
def test_fake_episode_rows(tmp_path):
    from dagger.fake import episode
    from rl.episodes import e2e
    d = tmp_path / "a0" / "ep00000"; episode(d, SERVER, n=30, seed=1, version=3)
    assert spool.episodes(tmp_path) == [d]
    rows, st = spool.episode_rows(d, e2e(SERVER)[0])
    assert st["rows_4hz"] == st["rows"] + st["unlabeled"] + st["no_valid"] and st["unlabeled"] > 0 and st["no_valid"] == 1
    assert all(r["version"] == 3 and r["valid"].any() for r in rows) and st["planner"] + st["route"] == st["rows"]
    assert st["tag_lane"] > 0 and st["tag_signal"] > 0 and all(r["w"] == 1.0 for r in rows)          # weights default 1
    rows2, _ = spool.episode_rows(d, e2e(SERVER)[0], weights={"signal": 2.0, "lane": 2.0})
    assert all(r["w"] == (2.0 if r["signal"] or r["lane"] else 1.0) for r in rows2)
    (d / rows[0]["req"]).rename(d / "x.b2.sync.tmp")                                      # still syncing: not read yet
    assert spool.episode_rows(d, e2e(SERVER)[0]) is None


@pytest.fixture(scope="module")
def model():
    if not (DELIVER / "final/config.json").exists(): pytest.skip("no tiny delivery")
    from morai_vla.student import StudentVLA
    from rl.tiny_test import quant_off
    m = StudentVLA.build(init=str(DELIVER / "final"), attn_implementation="sdpa", dtype="float32")
    quant_off(m); m.requires_grad_(False)
    m.loss_weights.update(kv=0.0, vit=0.0, fm=1.0, traj_ce=1.0, lm=1.0, vkd=0.0, coc_kl=0.0)
    return m


def batch(m, n=2, generation_mode=False):
    from morai_vla.data import ego_trajectory
    from morai_vla.processor import collate_fn, get_preprocess_fn
    from test_layout import sample
    rows = []
    for k in range(n):
        s = sample(lidar=k == 0)
        v = np.r_[np.linspace(6 + k, 8, 16), np.linspace(8, 12 - 2 * k, 64)]; yaw = np.cumsum(np.full(80, 0.004 * (k + 1)))
        hx, hr, fx, fr = ego_trajectory(v, yaw, 15)
        s.update(ego_history_xyz=hx, ego_history_rot=hr, ego_future_xyz=fx, ego_future_rot=fr)
        s["tokenized_data"] = get_preprocess_fn(m.config, generation_mode=generation_mode)(data=s)
        for x in ("camera_frames", "bev_frames", "camera_times", "bev_times"): s.pop(x, None)
        rows.append(s | {"gt_row": True, "hist_drop": False, "vkd_ok": False})
    b = collate_fn(rows, model_config=m.config)
    b = {k: v for k, v in b.items() if k in ("tokenized_data", "ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot",
                                             "labels_mask", "vkd_ok", "gt_row", "hist_drop")}
    s = m.action_space
    x = s.traj_to_action(b["ego_history_xyz"][:, 0], b["ego_history_rot"][:, 0], b["ego_future_xyz"][:, 0], b["ego_future_rot"][:, 0]).view(n, 64, 2)
    b["ak"] = torch.stack([x[..., 0] * s.accel_std + s.accel_mean, x[..., 1] * s.curvature_std + s.curvature_mean], -1).float()
    b["valid"] = torch.ones(n, 64, dtype=torch.bool)
    return b


@need_model
def test_dagger_loss_equals_r2_gt_row(model):
    """label = the GT future's (a, kappa), all valid: the DAgger loss is R2 ft's GT-row fm + traj_ce (same draws)."""
    m = model; m.train(); b = batch(m)
    m.loss_sums.clear(); torch.manual_seed(5)
    ref = m(**{k: v for k, v in b.items() if k not in ("ak", "valid")}).loss
    r = {k: s / n for k, (s, n) in m.loss_sums.items()}; m.loss_sums.clear()
    assert set(r) == {"fm", "traj_ce"}
    torch.manual_seed(5); loss, per = dagger_forward(m, b)
    assert abs(float(loss) - float(ref)) < 1e-4 * float(ref), (float(loss), float(ref))
    assert abs(float(per["fm"].mean()) - r["fm"]) < 1e-4 and abs(float(per["ce_sum"].sum() / per["ce_n"].sum()) - r["traj_ce"]) < 1e-4
    assert per["ce_n"].tolist() == [130.0, 130.0]                                          # 128 future tokens + start + end


@need_model
def test_dagger_valid_mask(model):
    m = model; m.train(); b = batch(m)
    b["valid"][0, 40:] = False; b["valid"][1, :10] = False
    torch.manual_seed(1); l1, p1 = dagger_forward(m, b, seed=7)
    b2 = dict(b); b2["ak"] = b["ak"].clone(); b2["ak"][0, 40:] = 50.0; b2["ak"][1, :10] = -50.0  # garbage where invalid
    l2, p2 = dagger_forward(m, b2, seed=7)
    assert torch.equal(p1["fm"], p2["fm"]) and torch.equal(p1["ce_sum"], p2["ce_sum"]) and p1["ce_n"].tolist() == [2 * 40 + 2, 2 * 54 + 2]
    assert torch.isfinite(l1) and float(l1) == float(l2)                                  # invalid points: token of 0, no loss
    x = normalize(m, b["ak"]); assert x.shape == (2, 64, 2) and torch.isfinite(x).all()


@need_model
def test_checkpoint_llm_same_gradients(model):
    """llm+expert LoRA: the checkpointed LLM (KV kept for the expert) gives the gradients of the plain one."""
    import copy
    from rl.policy import LLM, TARGET, LoRA
    base = copy.deepcopy(model); base.train()
    grads = []
    for ck in (False, True):
        m = copy.deepcopy(base)
        if ck: checkpoint_llm(m)
        lora = LoRA(m, 4, 8, (LLM, TARGET))
        torch.manual_seed(0)
        with torch.no_grad():
            for k in lora.B: lora.B[k].normal_(0, 0.02)
            for k in lora.A: lora.A[k].normal_(0, 0.05)
        b = batch(m); torch.manual_seed(3)
        loss, _ = dagger_forward(m, b)
        b.pop("ak"); b.pop("valid"); torch.manual_seed(4)
        loss = loss + m(**b).loss                                                         # the replay path too
        loss.backward()
        grads.append({k: p.grad.clone() for k, p in lora.named_parameters()})
    assert grads[0].keys() == grads[1].keys() and any("language_model" in k for k in grads[0])
    for k in grads[0]:
        assert torch.allclose(grads[0][k], grads[1][k], rtol=1e-4, atol=1e-7), k
    assert sum(float(g.abs().sum()) for k, g in grads[1].items() if "language_model" in k) > 0


@need_model
def test_llm_lora_names_and_merge(model, tmp_path):
    from safetensors import safe_open
    from rl.policy import LLM, TARGET, LoRA, merge
    lora = LoRA(model, 4, 8, (LLM, TARGET))
    with safe_open(str(DELIVER / "final/model.safetensors"), "pt") as h: keys = set(h.keys())
    assert all(n + ".weight" in keys for n in lora.names) and any(n.startswith(LLM) for n in lora.names)
    with torch.no_grad():
        for k in lora.B: lora.B[k].normal_(0, 0.02)
    lora.export(tmp_path / "v00001.safetensors", version=1)
    merge(DELIVER, tmp_path / "v00001.safetensors", tmp_path / "merged")
    n = next(x for x in lora.names if x.startswith(LLM)); k = n.replace(".", "/")
    with safe_open(str(tmp_path / "merged/final/model.safetensors"), "pt") as h: w = h.get_tensor(n + ".weight").float()
    with safe_open(str(DELIVER / "final/model.safetensors"), "pt") as h: w0 = h.get_tensor(n + ".weight").float()
    d = (lora.B[k].bfloat16().float() @ lora.A[k].bfloat16().float()) * lora.scale
    assert torch.allclose(w, (w0 + d).to(w.dtype).float(), atol=1e-2)


def test_row_tags_and_weights():
    obs = [10.0] + [0.0] * 6 + [0.2] + [0.0] * 22
    assert spool.row_tags(0, 1, 40.0, obs) == (True, False)                                # red, line 40 m ahead
    assert spool.row_tags(0, 2, 80.0, obs) == (False, False)                               # yellow but 80 m away
    assert spool.row_tags(0, 1, np.inf, [0.1] + obs[1:]) == (True, False)                  # standing at red
    assert spool.row_tags(0, 3, 10.0, obs) == (False, False)                               # green
    assert spool.row_tags(1, 0, np.inf, obs) == (False, True)                              # route fallback label
    assert spool.row_tags(0, 0, np.inf, obs[:7] + [-0.6] + obs[8:]) == (False, True)      # 0.6 m right of the lane centre
    assert spool.row_tags(0, 0, np.inf, None) == (False, False)
    w = {"signal": 2.0, "lane": 3.0}
    assert (spool.row_weight(False, False, w), spool.row_weight(True, False, w), spool.row_weight(True, True, w)) == (1.0, 2.0, 3.0)
    assert spool.row_weight(True, False, {"signal": 0.5, "lane": 1.0}) == 0.5


def test_future_green_rule():
    from dagger.pools import future_green, motion
    n = 200; v = np.full(n, 5.0); s = np.arange(n) * 0.5                                  # 5 m/s, 0.5 m per row
    sig = np.full(n, 1, np.int8); sig[100:] = 3                                            # red, green from row 100
    stops = [60.0]                                                                         # line at s 60 (row ~114 for the axle)
    i = np.array([20, 40, 99, 120, 150])
    r = future_green(v, sig, s, i, stops)
    assert r[0] == "" and r[1] == "to_green" and r[2] == "to_green"                         # row 20: green after its 6.4 s window
    assert r[3] == "" and r[4] == ""                                                       # past the line (more than 5 m behind)
    sig0 = sig.copy(); sig0[60] = 0
    assert future_green(v, sig0, s, np.array([20]), stops)[0] == "unknown"                 # no GT inside the window
    assert future_green(v, sig0, s, np.array([20]), [500.0])[0] == ""                      # line out of reach + 50 m: kept
    vv = np.r_[np.full(50, 4.0), np.zeros(150)]
    assert motion(vv, np.array([30, 60])).tolist() == ["brake", "stays stopped"]
    from dagger.pools import next_line
    assert next_line(s, np.array([20, 112, 120, 130]), stops).tolist() == [47.0, 1.0, -3.0, np.inf]   # axle 3 m ahead; 5 m behind counts


@need_model
def test_dagger_row_weight(model):
    m = model; m.train(); b = batch(m)
    l1, _ = dagger_forward(m, b, seed=3)
    b["w"] = torch.ones(2); l1w, _ = dagger_forward(m, b, seed=3)
    b["w"] = torch.full((2,), 2.0); l2, _ = dagger_forward(m, b, seed=3)
    b["w"] = torch.tensor([2.0, 1.0]); lm_, _ = dagger_forward(m, b, seed=3)
    assert float(l1) == float(l1w) and abs(float(l2) - 2 * float(l1)) < 1e-5 * float(l1) and float(l1) < float(lm_) < float(l2)


def test_dagger_config(tmp_path):
    import inspect
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from morai_vla.data import MoraiDataset
    from morai_vla.finetune import MixtureDataset
    configs = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(str(configs), version_base=None):
        c = OmegaConf.to_container(compose("dagger", overrides=["data.root=/data", "paths.r2_root=/r2", f"paths.output_dir={tmp_path}"]), resolve=True)
    d = c["dagger"]; s = d["replay"]["sources"]
    assert abs(d["share"] + sum(v["share"] for v in s.values()) - 1) < 1e-9 and d["share"] == 0.5
    assert (s["signal_approach"]["share"], s["signal_hold"]["share"]) == (0.10, 0.05)                         # users 10/28: 15 % = 10 + 5
    assert s["signal_approach"]["loss"] == s["signal_hold"]["loss"] == "gt" and {"signal_approach", "signal_hold", "lane"} <= set(d["val_sets"])
    assert s["r1_general"]["loss"] == "distill" and s["r1_general"]["dataset"]["exclude"].endswith("r1_replay_exclude.parquet")
    assert all(v["dataset"].get("exclude", "").endswith("future_green.parquet") for k, v in s.items() if k in ("old_recovery", "old_curve"))
    assert d["weights"] == {"signal": 2.0, "lane": 2.0}                                                     # user 10/28: x2 on
    assert set(d["replay"]) - {"_target_", "_recursive_"} <= set(inspect.signature(MixtureDataset).parameters)
    for v in s.values():
        assert set({**d["replay"]["common"], **v["dataset"]}) <= set(inspect.signature(MoraiDataset).parameters)
    for parts in d["val_sets"].values():
        assert all(set(p) - {"n"} <= set(inspect.signature(MoraiDataset).parameters) for p in parts)
    pools = Path(__file__).resolve().parents[1] / "assets/dagger"
    assert all((pools / f).exists() for f in ("signal_approach.parquet", "signal_hold.parquet", "future_green.parquet", "r1_replay_exclude.parquet",
                                              "highway.parquet", "pools.json"))
    assert set(d["history_sets"]) == {"general", "curve", "highway"} and d["history_steps"] == 10
    assert all(s[k]["dataset"]["keep"].endswith(f"r2_{k[-1].upper()}_dyn.parquet") for k in ("new_a", "new_b"))   # user 10/28: braking rows
    assert all((pools / f"r2_{k}_dyn.parquet").exists() for k in "AB") and (pools / "r2_unseen.parquet").exists()
    assert s["new_a"]["dataset"]["exclude"].endswith("r2_unseen.parquet") and s["new_b"]["dataset"]["exclude"].endswith("r2_unseen.parquet")
    for parts in d["history_sets"].values():
        assert all(set(p) - {"n"} <= set(inspect.signature(MoraiDataset).parameters) for p in parts)
    assert c["model"]["loss_weights"] == {"kv": 0.0, "vit": 0.0, "fm": 1.0, "traj_ce": 1.0, "lm": 1.0, "vkd": 0.0, "coc_kl": 0.0}


@need_model
def test_history_measure(model, tmp_path):
    """dagger/history.py: same numbers twice (fixed noise, deterministic Euler), zero / swap move the plan, own history
    does not, partner = row (k + n/2) % n."""
    from ft_synth import make
    from dagger import history
    from rl.policy import LoRA, TARGET
    root = make(tmp_path / "synth")
    hs = history.sets({"general": [{"root": str(root), "split": ["val", "val_asset"], "frame_stride": 10, "n": 4}]})
    lora = LoRA(model, 4, 8, (TARGET,))
    a = history.measure(model, lora, hs, 3, torch.device("cpu"))
    b = history.measure(model, lora, hs, 3, torch.device("cpu"))
    assert a == b and len(a) == len(hs["general"]) and all(r["zero_move"] > 0 and r["swap_move"] >= 0 for r in a)
    n = len(a); assert a[0]["swap_from"] == a[n // 2]["key"]
    ds, k = hs["general"][0]; s = ds.sample(k); own = (s["ego_history_xyz"], s["ego_history_rot"])
    x0 = torch.randn(1, 64, 2, generator=torch.Generator().manual_seed(0))
    p1, p2 = history.plans(model, lora, s, [own, own], x0, 3, torch.device("cpu"))
    assert np.array_equal(p1, p2) and p1.shape == (64, 2)
    sm = history.summary(a)["general"]; assert sm["n"] == n and {"zero_move", "zero_dlat_2s", "swap_dlat_3s", "zero_move_p90"} <= set(sm)


def test_planner_reuse_counts(tmp_path):
    from dagger.runner import Planner
    pl = Planner.__new__(Planner)
    pl.rows, pl.uses, pl.fifo, pl.seed = [{"k": i} for i in range(5)], [0] * 5, 0, 0
    rows, new, idx = pl.take(4, 0); assert new == 4 and idx == [0, 1, 2, 3]
    rows, new, idx = pl.take(4, 1); assert new == 1 and idx[0] == 4 and all(i < 4 for i in idx[1:])   # 1 new + 3 trained rows again
    r = pl.reuse(idx); assert pl.fifo == 5 and sum(pl.uses) == 8 and r["uses_mean"] == 1.6 and r["rows_used_2plus"] >= 1


def test_r2_unseen_rows(tmp_path):
    """dagger/pools.py r2 --r2logs: a static object 100 m down a straight road; the ego at 10 m/s brakes for it at 8 s
    (20 m before it). Rows from 8 - 6.4 = 1.6 s are unseen until the object enters the 60 m BEV (ego at 40 m, 4.0 s)."""
    import sys, types, yaml
    from dagger import pools as P
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/scorer"))
    from label_check import visible
    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[2] / "tools/scorer/label_check.yaml"))["visibility"]
    tau = np.arange(0, 14, 0.1); v = np.where(tau < 8, 10.0, np.clip(10 - 5 * (tau - 8), 0, 10)); x = np.r_[0, np.cumsum(v[:-1] * 0.1)]
    ego = {"tau": tau, "bag_ns": (1000 + tau) * 1e9, "v": v, "x": x, "y": np.zeros_like(x), "yaw": np.zeros_like(x)}
    log = tmp_path / "e.r2.log"
    log.write_text("".join(f"{1000 + t:.2f} disp 1:obstacle:0.0\n" for t in range(0, 14, 2)) + "1008.00 brake object 1 v=10.0\n1009.00 brake object 1 v=5.0\n")
    disp, events = P.read_r2log(log)
    assert events == [(1008.0, 1009.0, 1)] and set(disp) == {1}
    acts = [{"id": 40100027, "s": 100.0, "lat": 0.0}]
    assert P.actor_map(acts, disp) == {1: 0} and P.actor_map(acts + [{"id": 50100001, "s": 5, "lat": 0}], disp) is None
    data = types.SimpleNamespace(route_point=lambda s, lat: (s, lat, 0.0), route_heading=lambda s: 0.0)
    rows = np.round(np.arange(0, 13, 0.5), 2)
    r = P.unseen("e", rows, acts, log, ego, data, cfg, visible)
    got = dict(zip(rows.tolist(), r.tolist()))
    assert got[1.0] == "no_event" and got[2.0] == "unseen" and got[3.5] == "unseen" and got[4.5] == "" and got[10.0] == "" and got[12.5] == "no_event"
