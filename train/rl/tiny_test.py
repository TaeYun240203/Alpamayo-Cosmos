"""Tiny check of the learner (README): the student_dryrun tiny WoWmayo delivery (QAD final/ + modelopt_state.pth +
quant_manifest.json) as the frozen base, synthetic episodes in the actor's record format, and a fake reward that pays the
action's mean normalised acceleration (rl-env reward fields; critic_obs[0] = decisions left, the rest noise). Checks:
  1. the learner's one-pass log-prob of a recorded chain = the actor's step-by-step log-prob (version 0, lag 0)
  2. PPO raises the deterministic (sigma 0) acceleration of the policy on held-out inputs (actor 1 version behind,
     sampling with the exported BF16 LoRA: the decoupled correction is exercised)
  3. merged delivery (rl.policy.merge) = base + the exported LoRA, compared by per-step log-probs of recorded chains
     (end trajectories of the tiny model are chaotic: a 1e-7 relative weight change moves the action by ~0.02)
  4. the learner loop: critic warm start (held-out explained variance), publish, staleness gate, state/resume, log,
     an abandoned episode (no done.json) still trained on
CPU: ModelOpt's NVFP4 fake quant needs CUDA, so the quantizers are switched off there (RL code path otherwise the same).
Usage (cwd train/): python -m rl.tiny_test <deliver dir> <work dir> --e2e_server <eval repo src/e2e_driver/server> [--device cuda]
"""
import argparse, io, json, os, shutil, sys, threading, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np, torch
from PIL import Image

from morai_vla.processor import BEV, CAMERAS, collate_fn, get_preprocess_fn
from rl import learner
from rl.episodes import DROP, Decisions, e2e, loader, rewards, write_json
from rl.policy import Policy, merge

D, K, SIGMA, OBS = 8, 10, 0.3, 30                                 # decisions per episode, Euler steps, exploration std, critic obs


def quant_off(model):
    from modelopt.torch.quantization.nn import TensorQuantizer
    for m in model.modules():
        if isinstance(m, TensorQuantizer): m.disable()


class CpuPolicy(Policy):
    """ModelOpt NVFP4 fake quantization is CUDA only: CPU runs with the quantizers off."""
    def __init__(self, *x, **k): super().__init__(*x, **k); quant_off(self.model)


def jpeg(rng, w, h):
    b = io.BytesIO(); Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(b, "JPEG", quality=90); return b.getvalue()


def write_episode(d, policy, seed, version, gen, server):
    """D decisions of a fake drive as the actor engine dumps them (wire req/out frames + rl-env fields in steps.jsonl):
    random camera JPEGs and LiDAR scans in sliding windows, 10 m/s straight. Decision 0 runs on the base expert (expert 0,
    deterministic: environment, not policy); the others on the trained set (expert 1), sampled all at once."""
    wire, build_sample = e2e(server)
    rng = np.random.default_rng(seed)
    cams = [[jpeg(rng, w, h) for _ in range(D + 3)] for _, w, h in CAMERAS]
    scans = [(rng.integers(0, 2, (BEV[2], BEV[1])) * rng.integers(0, 255, (BEV[2], BEV[1]))).astype(np.uint8) for _ in range(2 * (D + 3))]
    reqs = [dict(t0=0.1 * i, camera_jpegs=[cams[c][i:i + 4] for c in range(4)], camera_times=np.tile([-0.3, -0.2, -0.1, 0.0], (4, 1)),
                 nav_text="Follow the road. No speed limit", scan_height=np.stack(scans[2 * i:2 * i + 8:2]),
                 scan_density=np.stack(scans[2 * i + 1:2 * i + 8:2]), scan_pose=np.array([[j * 1.0, 0.0, 0.0] for j in range(i, i + 4)]),
                 bev_times=np.array([-0.1, 0.0]), ego_v=np.full(16, 10.0), ego_yaw=np.zeros(16), id=i) for i in range(D)]
    pre = get_preprocess_fn(policy.model.config, generation_mode=True)
    samples = []
    for r in reqs:
        x = build_sample(r); x["tokenized_data"] = pre(data=x)
        samples.append({k: v for k, v in x.items() if k not in DROP})
    chain, lp = policy.sample(collate_fn(samples, model_config=policy.model.config), torch.full((K,), SIGMA), gen)
    chain, lp = chain.cpu(), lp.cpu()
    acc = chain[:, -1, :, 0].mean(1)                               # action = x_K, dim 0 = normalised acceleration
    d.mkdir(parents=True)
    with open(d / "steps.jsonl", "w") as f:
        for i, r in enumerate(reqs):
            out = dict(expert=0, lora_version=None) if i == 0 else \
                dict(expert=1, lora_version=str(version), chain=chain[i].numpy(), sigma=np.full(K, SIGMA, np.float32), logp=lp[i].numpy())
            (d / f"req_{i:06d}.bin").write_bytes(wire.pack(r)); (d / f"out_{i:06d}.bin").write_bytes(wire.pack(out))
            bonus = float(acc[i]) if i else 0.0
            f.write(json.dumps(dict(req=f"req_{i:06d}.bin", out=f"out_{i:06d}.bin", r_time=-0.5, r_penalty=bonus, r_terminal=0.0,
                                    phi=0.0, phi_next=0.0, events=[{"type": "fake_accel", "penalty_s": -bonus}],
                                    critic_obs=[float(D - i)] + rng.standard_normal(OBS - 1).tolist())) + "\n")
    write_json(d / "done.json", {"terminated": True, "truncated": False, "end": "goal"})
    return float(acc[1:].mean())


def det_eval(policy, batch):
    """deterministic Euler (sigma 0) on held-out decisions: mean normalised accel and 6.4 s travel (m) of the action."""
    chain, _ = policy.sample(batch, torch.zeros(K), torch.Generator().manual_seed(123))   # same x_0 every call
    a = chain[:, -1]
    xyz, _ = policy.model.action_space.action_to_traj(a, batch["ego_history_xyz"][:, 0].to(a.device), batch["ego_history_rot"][:, 0].to(a.device))
    return float(a[..., 0].mean()), float(xyz[:, -1, 0].mean()), a


def main():
    p = argparse.ArgumentParser(); p.add_argument("deliver"); p.add_argument("work")
    p.add_argument("--e2e_server", default=os.environ.get("E2E_SERVER"), required=not os.environ.get("E2E_SERVER"))
    p.add_argument("--device", default="cpu"); p.add_argument("--updates", type=int, default=12); p.add_argument("--episodes", type=int, default=6)
    a = p.parse_args()
    deliver, work = Path(a.deliver), Path(a.work)
    shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
    cpu = a.device == "cpu"
    if cpu:                                                        # same patch for the loop test below
        learner.Policy = CpuPolicy
    P = learner.Policy(deliver / "final", rank=8, alpha=16, dtype="float32", device=a.device)
    log = {"device": a.device, "quantizers": "off (CPU)" if cpu else "on (NVFP4/FP8 fake quant as delivered)"}

    # 0. GAE / reward sanity
    adv, ret = learner.gae(torch.ones(3), torch.zeros(3), 1.0, 1.0, True)
    assert adv.tolist() == [3.0, 2.0, 1.0] and ret.tolist() == [3.0, 2.0, 1.0]
    r = rewards([{"r_time": -0.1, "r_penalty": -15.0, "r_terminal": -100.0, "phi": 2.0, "phi_next": 3.0}], 0.5)
    assert abs(float(r[0]) - (-0.1 - 15 - 100 + 0.5 * 3 - 2)) < 1e-5

    args = learner.parse(["--base", "-", "--spool", "-", "--out", "-", "--e2e_server", a.e2e_server, "--lr", "2e-3", "--critic_lr", "2e-3", "--epochs", "2",
                          "--minibatch", "16", "--micro", "8", "--workers", "0", "--max_lag", "1"])
    opt = torch.optim.AdamW(list(P.lora.parameters()), lr=args.lr)
    critic = learner.Critic(OBS).to(P.device); opt_v = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr)
    gen = torch.Generator().manual_seed(0)
    base = [work / "baseline" / "a0" / f"e{e}" for e in range(10)]     # baseline drives of the delivered model (version 0)
    for e, d in enumerate(base): write_episode(d, P, 3000 + e, 0, gen, a.e2e_server)
    log["warmstart"] = learner.warmstart(critic, opt_v, base, args)
    print(json.dumps(log["warmstart"]), flush=True)
    assert log["warmstart"]["ev_val"] > 0.5, log["warmstart"]
    held = work / "held"; write_episode(held, P, 999, 0, gen, a.e2e_server)
    hd = Decisions([held], P.model.config, a.e2e_server)
    eval_b = next(iter(loader(hd, P.model.config, D, hd.policy)))
    acc0, travel0, a0 = det_eval(P, eval_b)
    log["start"] = {"det_accel": round(acc0, 4), "det_travel_m": round(travel0, 3)}
    print(json.dumps(log), flush=True)

    out = work / "out"; (out / "lora").mkdir(parents=True)
    learner.publish(P, out, 0)
    hist = []
    for n in range(a.updates):
        actor_v = max(n - 1, 0)                                    # the actor drives one version behind (max_lag 1)
        mine = {k: v.clone() for k, v in P.lora.state_dict().items()}
        P.lora.load(out / "lora" / f"v{actor_v:05d}.safetensors")
        eps = [work / "spool" / f"u{n:03d}_e{e}" for e in range(a.episodes)]
        noisy = [write_episode(d, P, 1000 * n + e, actor_v, gen, a.e2e_server) for e, d in enumerate(eps)]
        P.lora.load_state_dict(mine)
        st = learner.update(P, critic, opt, opt_v, eps, n, args)
        learner.publish(P, out, n + 1)
        acc, travel, _ = det_eval(P, eval_b)
        st.update(noisy_accel=round(float(np.mean(noisy)), 4), det_accel=round(acc, 4), det_travel_m=round(travel, 3))
        st.pop("events"); st.pop("ends"); hist.append(st); print(json.dumps(st), flush=True)
        if n == 0:                                                 # check 1: same weights, same inputs -> same log-prob
            assert abs(st["behav_gap_mean"]) < 1e-2 and st["behav_gap_std"] < 1e-2, st
    log["updates"] = hist
    acc1, travel1, a1 = det_eval(P, eval_b)
    log["end"] = {"det_accel": round(acc1, 4), "det_travel_m": round(travel1, 3)}
    if a.updates >= 8:                                             # check 2 (a shorter run only checks the plumbing)
        assert acc1 > acc0 + 0.3 and travel1 > travel0, (acc0, acc1, travel0, travel1)

    # check 3: merged delivery drives like base + LoRA
    merged = work / "merged"
    merge(deliver, out / "lora" / f"v{a.updates:05d}.safetensors", merged)
    M = learner.Policy(merged / "final", rank=8, alpha=16, dtype="float32", device=a.device)
    P.lora.load(out / "lora" / f"v{a.updates:05d}.safetensors")    # base + the BF16 file the actor gets
    with torch.no_grad():                                          # per-step log-probs of recorded chains: the IS weight's view
        lp_l, lp_m = P.evaluate(eval_b)["logp"].sum(1), M.evaluate(eval_b)["logp"].sum(1)
    accm, travelm, _ = det_eval(M, eval_b)
    log["merged"] = {"det_accel": round(accm, 4), "det_travel_m": round(travelm, 3),
                     "logp_gap_max": round(float((lp_m - lp_l).abs().max()), 5), "logp_mean": round(float(lp_l.mean()), 2)}
    if cpu:                                                        # with quantizers on it is the requant gap: reported only
        assert log["merged"]["logp_gap_max"] < 1e-2, log["merged"]
    del M

    # check 4: the loop (publish, gate, state/resume, log) on a fresh spool
    spool, out2 = work / "loop_spool", work / "loop_out"
    for e in range(2): write_episode(spool / "a0" / f"e{e}", P, 5000 + e, 0, gen, a.e2e_server)
    write_json(spool / "a0" / "inflight.json", {"version": 0, "episode": "e2"})
    base_args = ["--base", str(deliver / "final"), "--spool", str(spool), "--out", str(out2), "--device", a.device, "--dtype", "float32",
                 "--e2e_server", a.e2e_server,
                 "--rank", "8", "--alpha", "16", "--min_decisions", str(2 * D), "--micro", "8", "--minibatch", "16", "--workers", "0",
                 "--max_updates", "1", "--poll", "0.2", "--warm_epochs", "50", "--kl_target", "5"]
    learner.main(base_args + ["--warmstart", str(work / "baseline")])
    assert json.load(open(out2 / "warmstart.json"))["held_out"] == 1
    for e in range(2, 5): write_episode(spool / "a0" / f"e{e}", P, 5000 + e, 1, gen, a.e2e_server)
    dead = spool / "a0" / "e4"; (dead / "done.json").unlink()      # actor died mid-episode: half a line, no done.json
    with open(dead / "steps.jsonl", "a") as f: f.write('{"version": 1, "nav')
    os.utime(dead / "steps.jsonl", (0, 0))
    learner.main(base_args)                                        # resumes from state.pt
    latest = json.load(open(out2 / "lora" / "latest.json")); lines = [json.loads(x) for x in open(out2 / "log.jsonl")]
    assert latest["version"] == 2 and [(x["episodes"], x["ends"].get("abandoned", 0)) for x in lines] == [(2, 0), (3, 1)], (latest, lines)
    # adaptive KL: measured each update, the coefficient x2 / x1 / /2 from 0.1 / target, carried over the resume
    k0, k1 = lines[0]["kl_coef_next"], lines[1]["kl_coef_next"]
    assert "kl_sft" in lines[1] and k0 / 0.02 in (0.5, 1.0, 2.0) and k1 / k0 in (0.5, 1.0, 2.0), lines
    g = SimpleNamespace(max_lag=1, actor_timeout=600, poll=0.1)
    write_json(spool / "a0" / "inflight.json", {"version": 3, "episode": "e9"})
    t = threading.Timer(1.0, lambda: write_json(spool / "a0" / "inflight.json", {"version": 5, "episode": "e10"})); t.start()
    t0 = time.time(); learner.gate(spool, 5, g); waited = time.time() - t0
    assert waited >= 0.9, waited                                   # blocked while the actor ran version 3 < 5 + 1 - 1
    (spool / "a1").mkdir(); write_json(spool / "a1" / "inflight.json", {"version": 0, "episode": "e0"})
    for f in spool.glob("*/inflight.json"): os.utime(f, (0, 0))
    learner.gate(spool, 9, g)                                      # actors silent for > actor_timeout are ignored
    log["loop"] = {"latest": latest, "log_versions": [x["version"] for x in lines], "gate_waited_s": round(waited, 2)}
    json.dump(log, open(work / "summary.json", "w"), indent=1)
    print("PASS", json.dumps({k: log[k] for k in ("device", "quantizers", "start", "end", "merged", "loop")}))


if __name__ == "__main__":
    sys.exit(main())
