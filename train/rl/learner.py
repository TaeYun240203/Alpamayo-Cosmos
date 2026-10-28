"""Decoupled PPO learner for the WoWmayo expert LoRA (RL_PLAN 3.2-3.5, 4.2). Loop: wait until every live actor runs a
recent enough version (staleness gate) -> take ALL finished, unconsumed episodes from the spool -> pass 0 (no grad):
proximal log-probs with the current weights; values from the privileged critic; GAE per episode -> `epochs` passes of
minibatch PPO on the LoRA, critic fit on the GAE returns (own optimizer) -> publish LoRA version n+1
(<out>/lora/vNNNNN.safetensors + latest.json) -> save state (resume) -> log.

Staleness: a record of version v trained by the update at version n has lag n - v. The actor reads latest.json at every
episode start and writes <spool>/<actor>/inflight.json; before update n -> n+1 the learner waits until every live actor's
in-flight version is >= n + 1 - max_lag, so later records have lag <= max_lag. Records are never dropped (a dead actor's
leftovers are trained on and counted in lag_over).
Usage (cwd train/): python -m rl.learner --base <deliver>/final --spool <dir> --out <dir> [--warmstart <baseline spool>]
Several GPUs: torchrun --nproc_per_node N -m rl.learner ... (one process per GPU: each trains its share of the policy
decisions, LoRA gradients are averaged, rank 0 waits for episodes, fits the critic and publishes).
"""
import argparse, datetime, itertools, json, logging, math, os, shlex, subprocess, time
from pathlib import Path

import torch, torch.distributed as dist, torch.nn as nn

from rl.episodes import Decisions, finished, loader, read_episode, rewards, write_json
from rl.policy import Policy

log = logging.getLogger("rl.learner")


def rank_world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def shared(obj):
    """rank 0's object on every rank."""
    if not dist.is_initialized(): return obj
    box = [obj]; dist.broadcast_object_list(box, 0); return box[0]


def sync(module):
    """rank 0's parameters and buffers on every rank (the critic is fitted on rank 0 only)."""
    if dist.is_initialized():
        for t in [*module.parameters(), *module.buffers()]: dist.broadcast(t.data, 0)


def reduce_sum(x, dev):
    if not dist.is_initialized(): return x
    x = x.to(dev); dist.all_reduce(x); return x.cpu()


class Critic(nn.Module):
    """privileged (asymmetric) value function, learner only (user decision 10/27, critic B): MLP 2 x 256 on the ~30 GT
    values per decision that rl-env records (`critic_obs`), inputs normalised by a running mean/std."""

    def __init__(self, dim, hidden=256):
        super().__init__()
        for k in ("n", "mean", "m2"): self.register_buffer(k, torch.zeros(() if k == "n" else dim, dtype=torch.float64))
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    @torch.no_grad()
    def observe(self, x):
        """running mean/std over every decision the critic trains on (parallel-variance update)."""
        x = x.to(self.mean); mb, n0 = x.mean(0), self.n.clone(); tot, d = n0 + len(x), mb - self.mean
        self.m2 += ((x - mb) ** 2).sum(0) + d ** 2 * n0 * len(x) / tot
        self.mean += d * len(x) / tot; self.n.copy_(tot)

    def forward(self, x):
        std = (self.m2 / self.n.clamp(min=1)).sqrt().clamp(min=1e-6)
        return self.net(((x.to(self.mean) - self.mean) / std).clamp(-10, 10).float()).squeeze(-1)


def gae(r, v, gamma, lam, terminated):
    """advantages and returns of one episode; a cut episode (time limit, actor died) bootstraps from its last value."""
    adv, g, nxt = torch.zeros_like(r), 0.0, 0.0 if terminated else float(v[-1])
    for i in reversed(range(len(r))):
        g = float(r[i] + gamma * nxt - v[i]) + gamma * lam * g
        adv[i], nxt = g, float(v[i])
    return adv, adv + v


def explained_variance(v, ret):
    return 1 - float((ret - v).var() / ret.var().clamp(min=1e-8)) if len(ret) > 1 else 0.0


def critic_episode(steps, meta, gamma):
    return torch.tensor([st["critic_obs"] for st in steps], dtype=torch.float32), rewards(steps, gamma), bool(meta["terminated"])


@torch.no_grad()
def values(critic, obs):
    return critic(obs.to(critic.mean.device)).float().cpu()


def fit(critic, opt, obs, target, epochs, minibatch):
    dev, last = critic.mean.device, 0.0
    for _ in range(epochs):
        for i in torch.randperm(len(obs)).split(minibatch):
            loss = (critic(obs[i].to(dev)) - target[i].to(dev)).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); last = loss.item()
    return last


def warmstart(critic, opt, dirs, a):
    """RL_PLAN TODO 9: before PPO, fit the critic to Monte-Carlo returns of baseline drives of the delivered model
    (lambda = 1 returns; cut episodes bootstrap from the current critic, recomputed every epoch). Every 10th episode (by
    name) is held out for explained variance."""
    eps = [critic_episode(*read_episode(d), a.gamma) for d in sorted(dirs, key=lambda d: Path(d).name)]
    val, train = (eps[::10], [e for i, e in enumerate(eps) if i % 10]) if len(eps) > 1 else ([], eps)
    critic.observe(torch.cat([o for o, _, _ in train]))
    ret = lambda es: [gae(r, values(critic, o), a.gamma, 1.0, t)[1] for o, r, t in es]
    for _ in range(a.warm_epochs):
        loss = fit(critic, opt, torch.cat([o for o, _, _ in train]), torch.cat(ret(train)), 1, a.critic_minibatch)
    ev = lambda es: explained_variance(torch.cat([values(critic, o) for o, _, _ in es]), torch.cat(ret(es))) if es else None
    return {"warmstart_episodes": len(eps), "held_out": len(val), "mse": round(loss, 4), "ev_train": ev(train), "ev_val": ev(val)}


def ppo_loss(logp, logp_prox, logp_behav, adv, clip, w_max):
    """decoupled PPO (Hilton et al. 2022, AReaL): the trust region clips pi / pi_prox, the off-policy correction is the
    truncated weight pi_prox / mu (mu = the actor's NVFP4 policy, up to max_lag versions old). Log-probs: sums over steps."""
    w = (logp_prox - logp_behav).clamp(max=math.log(w_max)).exp()
    r = (logp - logp_prox).clamp(-20, 20).exp()
    loss = -(w * torch.min(r * adv, r.clamp(1 - clip, 1 + clip) * adv)).mean()
    return loss, {"clip_frac": float(((r - 1).abs() > clip).float().mean()), "approx_kl": (logp_prox - logp).mean().item()}


def update(policy, critic, opt_pi, opt_v, episodes, version, a):
    """one PPO update on these finished episodes; returns stats. Several ranks: each trains its share of the policy
    decisions (the same split everywhere, drawn from the version) with the same number of optimizer steps, LoRA gradients
    averaged; the critic is fitted on rank 0 and copied."""
    t0, dev = time.time(), policy.device
    rank, world = rank_world()
    ds = Decisions(episodes, policy.model.config, a.e2e_server, a.train_expert)
    n, pol = len(ds), torch.tensor(ds.policy, dtype=torch.long)         # pol: the trained expert's decisions
    mine = pol if world == 1 else pol[torch.randperm(len(pol), generator=torch.Generator().manual_seed(version))][rank::world]
    lp_prox, lp_behav = torch.zeros(n), torch.zeros(n)
    with torch.no_grad():
        for b in (loader(ds, policy.model.config, a.micro, mine.tolist(), a.workers) if len(mine) else ()):
            i = b["idx"]; lp_prox[i], lp_behav[i] = policy.evaluate(b)["logp"].sum(1).float().cpu(), b["logp"].sum(1)
    obs, value, adv, ret = [], torch.zeros(n), torch.zeros(n), torch.zeros(n)
    types, ends, ep_time, ep_pen = {}, {}, [], []
    for d, s, e, steps, meta in ds.episodes:
        o, r, term = critic_episode(steps, meta, a.gamma)
        obs.append(o); value[s:e] = values(critic, o)
        adv[s:e], ret[s:e] = gae(r, value[s:e], a.gamma, a.lam, term)
        ep_time.append(-sum(st["r_time"] for st in steps)); ep_pen.append(-sum(st["r_penalty"] for st in steps))
        ends[meta.get("end")] = ends.get(meta.get("end"), 0) + 1
        for st in steps:
            for ev in st.get("events", []): types[ev["type"]] = types.get(ev["type"], 0) + 1
    obs = torch.cat(obs)
    lag = torch.tensor([version - ds.version(i) for i in ds.policy] or [0])
    g = (lp_prox - lp_behav)[mine].double()
    gs = reduce_sum(torch.stack([g.sum(), (g ** 2).sum(), (g > math.log(a.w_max)).sum().double(), torch.tensor(float(len(g)), dtype=torch.float64)]), dev)
    gap_mean = float(gs[0] / gs[3].clamp(min=1)); gap_std = math.sqrt(max(float(gs[1] / gs[3].clamp(min=1)) - gap_mean ** 2, 0.0))
    nadv = adv.clone()
    if len(pol) > 1: nadv[pol] = (adv[pol] - adv[pol].mean()) / (adv[pol].std() + 1e-8)
    use_kl = a.kl_coef > 0 or a.kl_target > 0
    keys = ["approx_kl", "clip_frac", "pg"] + (["kl_sft"] if use_kl else [])
    acc = max(1, math.ceil(a.minibatch / world / a.micro))             # micro-batches per optimizer step on each rank
    n_steps = math.ceil(math.ceil(len(pol) / world) / (acc * a.micro))  # the same on every rank
    params, sums, gnorm = list(policy.lora.parameters()), torch.zeros(len(keys) + 1, dtype=torch.float64), []
    for _ in range(a.epochs if len(pol) else 0):
        it = iter(loader(ds, policy.model.config, a.micro, mine[torch.randperm(len(mine))].tolist(), a.workers) if len(mine) else ())
        for _ in range(n_steps):
            opt_pi.zero_grad()
            for b in itertools.islice(it, acc):
                i = b["idx"]
                o = policy.evaluate(b, kl=use_kl)
                pg, st = ppo_loss(o["logp"].sum(1).float(), lp_prox[i].to(dev), lp_behav[i].to(dev), nadv[i].to(dev), a.clip, a.w_max)
                loss = pg + (a.kl_coef * o["kl"].mean() if use_kl else 0)
                (loss * len(i) / (a.micro * acc)).backward()
                st.update(pg=pg.item(), **({"kl_sft": o["kl"].mean().item()} if use_kl else {}))
                sums += torch.tensor([st[key] for key in keys] + [1.0], dtype=torch.float64)
            if world > 1:
                flat = torch.cat([(q.grad if q.grad is not None else torch.zeros_like(q)).flatten() for q in params])
                dist.all_reduce(flat); flat /= world; k = 0
                for q in params:
                    q.grad = flat[k:k + q.numel()].view_as(q).clone(); k += q.numel()
            gnorm.append(float(torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm)))
            opt_pi.step()
    sums = reduce_sum(sums, dev); k = max(float(sums[-1]), 1.0)
    vf = 0.0
    if rank == 0:
        critic.observe(obs); vf = fit(critic, opt_v, obs, ret, a.critic_epochs, a.critic_minibatch)
    sync(critic)
    return {"version": version, "episodes": len(ds.episodes), "decisions": n, "policy_decisions": len(pol), "ranks": world,
            **{key: round(float(sums[j]) / k, 5) for j, key in enumerate(keys)},
            "grad_norm": round(sum(gnorm) / max(len(gnorm), 1), 4), "opt_steps": len(gnorm), "vf": round(vf, 4),
            "value_ev": round(explained_variance(value, ret), 4),              # values from before this fit: new episodes = held out
            "ep_time_s": round(sum(ep_time) / len(ep_time), 3), "ep_penalty_s": round(sum(ep_pen) / len(ep_pen), 3),
            "ends": ends, "events": types, "adv_std": round(float(adv.std()), 4) if n > 1 else 0.0,
            "lag_max": int(lag.max()), "lag_over": int((lag > a.max_lag).sum()),
            "behav_gap_mean": round(gap_mean, 3), "behav_gap_std": round(gap_std, 3),
            "w_truncated": round(float(gs[2] / gs[3].clamp(min=1)), 4),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2) if dev.type == "cuda" else None,
            "s_per_decision": round((time.time() - t0) / n, 4), "s_per_policy_decision": round((time.time() - t0) / max(len(pol), 1), 4)}


def publish(policy, out, version):
    name = f"v{version:05d}.safetensors"
    policy.lora.export(out / "lora" / name, version=version)
    write_json(out / "lora" / "latest.json", {"version": version, "path": name, "scale": policy.lora.scale, "rank": policy.lora.rank})


def upload(out, version, b2, prev):
    """the version's file, then latest.json, to <b2>/lora/ (the actors fetch files before latest.json); in the background,
    after the previous upload."""
    if prev is not None and prev.wait(): log.warning("B2 upload of a LoRA version failed (rc %s), see b2_upload.log", prev.returncode)
    bucket, _, prefix = b2.removeprefix("b2://").partition("/")
    up = lambda f: f"b2 file upload --no-progress {shlex.quote(bucket)} {shlex.quote(str(out / 'lora' / f))} {shlex.quote(prefix.rstrip('/') + '/lora/' + f)}"
    return subprocess.Popen(["bash", "-c", f"{up(f'v{version:05d}.safetensors')} && {up('latest.json')}"],
                            stdout=subprocess.DEVNULL, stderr=open(out / "b2_upload.log", "a"))


def wait_new(a, consumed, version):
    """rank 0: the staleness gate, then every finished unconsumed episode once they hold min_decisions; None after
    stale_stop_min without enough (no paid GPU waits on dead actors; the state is saved, the same command resumes)."""
    t0 = time.time()
    while True:
        gate(a.spool, version, a)
        new = [d for d in finished(a.spool, a.actor_timeout) if str(d) not in consumed]
        if sum(sum(1 for _ in open(d / "steps.jsonl")) for d in new) >= a.min_decisions: return new
        if a.stale_stop_min and time.time() - t0 > 60 * a.stale_stop_min: return None
        time.sleep(a.poll)


def gate(spool, version, a):
    """block until every live actor (inflight.json younger than actor_timeout) runs a version >= version + 1 - max_lag."""
    while True:
        old = []
        for f in Path(spool).glob("*/inflight.json"):
            try:
                if time.time() - f.stat().st_mtime < a.actor_timeout and json.loads(f.read_text())["version"] < version + 1 - a.max_lag:
                    old.append(f.parent.name)
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                continue                                                # being rewritten
        if not old: return
        log.info("gate: waiting for %s to reach version %d", old, version + 1 - a.max_lag)
        time.sleep(a.poll)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="delivered QAD final/ dir (frozen, fake-quantized base)")
    p.add_argument("--spool", required=True); p.add_argument("--out", required=True)
    p.add_argument("--e2e_server", default=os.environ.get("E2E_SERVER"),
                   help="eval repo src/e2e_driver/server of the actor engine (wire format, assemble.build_sample)")
    p.add_argument("--train_expert", type=int, default=1, help="response `expert` whose decisions the LoRA drives (2nd set)")
    p.add_argument("--device", default="cuda"); p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--rank", type=int, default=16); p.add_argument("--alpha", type=float, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=1, help="PPO passes over each batch (each pass re-runs the VLM prefill)")
    p.add_argument("--minibatch", type=int, default=64, help="decisions per optimizer step")
    p.add_argument("--micro", type=int, default=2, help="decisions per forward (memory)")
    p.add_argument("--clip", type=float, default=0.2); p.add_argument("--w_max", type=float, default=2.0)
    p.add_argument("--kl_coef", type=float, default=0.0, help="> 0: KL to the delivered (SFT) policy")
    p.add_argument("--kl_target", type=float, default=0.0, help="> 0: adaptive KL coefficient (PPO paper, Schulman 2017): "
                   "x2 above 1.5 x target, /2 below target / 1.5, nats per decision; starts at 0.1 / target unless "
                   "--kl_coef. User 10/28: a mean shift of 0.1-0.2 sigma per step = 640 x^2 = 6.4-25.6 nats")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.995); p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--critic_lr", type=float, default=1e-3); p.add_argument("--critic_epochs", type=int, default=4)
    p.add_argument("--critic_minibatch", type=int, default=256)
    p.add_argument("--warmstart", help="spool of baseline drives of the delivered model: fit the critic first (fresh start only)")
    p.add_argument("--warm_epochs", type=int, default=200)
    p.add_argument("--min_decisions", type=int, default=256, help="wait for at least this many new decisions")
    p.add_argument("--max_lag", type=int, default=2); p.add_argument("--actor_timeout", type=float, default=600.0)
    p.add_argument("--poll", type=float, default=5.0); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max_updates", type=int, default=0, help="stop after this many updates (0 = run forever)")
    p.add_argument("--stale_stop_min", type=float, default=30, help="stop when no new episodes come for this long (0 = never)")
    p.add_argument("--b2", help="b2://bucket/prefix: each published version uploaded to <prefix>/lora/ (file, then latest.json)")
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:                      # torchrun; rank 0 may wait hours for the actors
        dist.init_process_group("nccl" if a.device == "cuda" else "gloo", timeout=datetime.timedelta(hours=12))
        if a.device == "cuda":
            a.device = f"cuda:{int(os.environ['LOCAL_RANK'])}"; torch.cuda.set_device(a.device)
    rank = rank_world()[0]
    out = Path(a.out); (out / "lora").mkdir(parents=True, exist_ok=True)
    policy = Policy(a.base, a.rank, a.alpha, a.dtype, a.device)
    sync(policy.lora)                                                   # each rank drew its own random LoRA A
    opt_pi = torch.optim.AdamW(list(policy.lora.parameters()), lr=a.lr, weight_decay=0.0)
    critic = opt_v = up = None

    def make_critic(dim):
        c = Critic(dim).to(policy.device); sync(c)
        return c, torch.optim.AdamW(c.parameters(), lr=a.critic_lr, weight_decay=0.0)

    def publish_all(version):
        nonlocal up
        publish(policy, out, version)
        if a.b2: up = upload(out, version, a.b2, up)

    version, consumed = 0, set()
    if a.kl_target > 0 and a.kl_coef <= 0:
        a.kl_coef = 0.1 / a.kl_target
    if (out / "state.pt").exists():
        s = torch.load(out / "state.pt", map_location=policy.device, weights_only=False)
        critic, opt_v = make_critic(s["critic_dim"])
        policy.lora.load_state_dict(s["lora"]); critic.load_state_dict(s["critic"])
        opt_pi.load_state_dict(s["opt_pi"]); opt_v.load_state_dict(s["opt_v"]); version, consumed = s["version"], set(s["consumed"])
        a.kl_coef = s.get("kl_coef", a.kl_coef)
        log.info("resumed at version %d (%d episodes consumed)", version, len(consumed))
    else:
        if a.warmstart:
            dirs = finished(a.warmstart, a.actor_timeout)
            critic, opt_v = make_critic(len(read_episode(dirs[0])[0][0]["critic_obs"]))
            if rank == 0:
                st = warmstart(critic, opt_v, dirs, a)
                write_json(out / "warmstart.json", st); log.info("critic warm start %s", json.dumps(st))
            sync(critic)
        if rank == 0: publish_all(0)                                     # version 0 = zero LoRA = the delivered expert
    done = 0
    while not a.max_updates or done < a.max_updates:
        new = shared(wait_new(a, consumed, version) if rank == 0 else None)
        if new is None:
            log.warning("no new episodes for %s min: stopping", a.stale_stop_min); break
        if critic is None: critic, opt_v = make_critic(len(read_episode(new[0])[0][0]["critic_obs"]))
        stats = update(policy, critic, opt_pi, opt_v, new, version, a)
        version += 1; consumed |= {str(d) for d in new}; done += 1
        if a.kl_target > 0 and "kl_sft" in stats:                       # the same on every rank (reduced stats)
            d = stats["kl_sft"]
            a.kl_coef *= 2.0 if d > 1.5 * a.kl_target else 0.5 if d < a.kl_target / 1.5 else 1.0
            stats["kl_coef_next"] = a.kl_coef
        if dist.is_initialized():                                          # every rank must hold the same LoRA
            cs = torch.stack([q.detach().double().sum() for q in policy.lora.parameters()]).sum().reshape(1)
            got = [torch.zeros_like(cs) for _ in range(rank_world()[1])]; dist.all_gather(got, cs)
            assert all(torch.equal(x, got[0]) for x in got), f"LoRA differs between ranks: {[float(x) for x in got]}"
        if rank: continue
        publish_all(version)
        tmp = out / "state.pt.tmp"
        torch.save({"lora": policy.lora.state_dict(), "critic": critic.state_dict(), "critic_dim": critic.mean.numel(),
                    "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "version": version, "consumed": sorted(consumed),
                    "kl_coef": a.kl_coef}, tmp)
        tmp.replace(out / "state.pt")
        with open(out / "log.jsonl", "a") as f: f.write(json.dumps(stats) + "\n")
        log.info(json.dumps(stats))
    if up is not None: up.wait()
    if dist.is_initialized(): dist.destroy_process_group()
    return policy


if __name__ == "__main__":
    main()
