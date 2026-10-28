"""Learner input (RL_PLAN 3.5): one directory per episode, <spool>/<actor>/<episode>/, assembled from the actor engine's
dumps (model_server --dump) and rl-env's reward rows:

  steps.jsonl   one JSON object per decision (10 Hz sim time), in order:
    req, out    file names (in the episode dir) of the engine's wire frames of this decision. The request is the full
                observation; the learner builds the model input with the engine's own assemble.build_sample, so both
                see the same prompt. The response gives expert, lora_version and, where some sigma_k > 0, chain
                (K+1, 64, 2), sigma (K,), logp (K,) (rl/policy.py docstring).
    r_time, r_penalty, r_terminal, phi, phi_next   the rl-env row of this decision (eval repo src/e2e_driver/tools/
                rl_reward.py), seconds of competition score: reward = r_time + r_penalty + r_terminal + gamma*phi_next - phi
    events      [{"type": str, "penalty_s": float, ...}] (counted per update, e.g. signal violations)
    critic_obs  the privileged GT vector of the critic (~30 floats, fixed order, no NaN)
  done.json     written LAST, atomically (rl_reward's last row): {"terminated": bool, "truncated": bool, "end": str, ...}
  <spool>/<actor>/inflight.json   {"version": int, "episode": str}, rewritten at every episode start (staleness gate)
The PPO loss uses the decisions of the trained expert set (response `expert` == train_expert: the 고주로 set of section
switching A). The other decisions (base expert before CP10 / after CP13) are part of the environment: their rewards and
values still enter GAE. A trained-expert decision needs sigma_k > 0 at every step. The critic warm start reads baseline
drives in the same layout and only needs the rl-env fields.
Nothing is dropped: every decision of every finished episode is trained on; an episode left without done.json (actor
died) is trained on as a time-limit cut once its steps.jsonl has been silent for actor_timeout.
"""
import json, os, sys, time
from functools import partial
from pathlib import Path

import torch

from morai_vla.processor import collate_fn, get_preprocess_fn

DROP = ("camera_frames", "bev_frames", "camera_times", "bev_times", "_image_keys", "ego_future_xyz", "ego_future_rot")


def e2e(server):
    """the actor engine's wire format and sample assembly: eval repo src/e2e_driver/server (+ ../src), the engine's code."""
    server = Path(server).expanduser()
    for p in (str(server), str(server.parent / "src")):
        if p not in sys.path: sys.path.append(p)
    from assemble import build_sample
    from e2e_driver import wire
    return wire, build_sample


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj)); os.replace(tmp, path)


def finished(spool, abandoned_after=600.0):
    """episode dirs with done.json, plus dirs without one whose steps.jsonl has not changed for abandoned_after s (the actor
    died: their decisions are still trained on, as a time-limit cut), oldest first."""
    now, out = time.time(), []
    for f in Path(spool).glob("*/*/steps.jsonl"):
        done = f.parent / "done.json"
        if done.exists() or now - f.stat().st_mtime > abandoned_after:
            out.append((done.stat().st_mtime if done.exists() else f.stat().st_mtime, f.parent))
    return [d for _, d in sorted(out)]


def read_episode(d):
    d = Path(d)
    steps = []
    for line in open(d / "steps.jsonl"):
        try:
            steps.append(json.loads(line))
        except json.JSONDecodeError:                              # an abandoned episode's half-written last line
            assert not (d / "done.json").exists(), f"{d}: corrupt steps.jsonl"
    assert steps, f"{d}: no decisions"
    if not (d / "done.json").exists():
        return steps, {"terminated": False, "truncated": True, "end": "abandoned"}
    return steps, json.load(open(d / "done.json"))


def rewards(steps, gamma):
    """per-decision reward (s of competition score) with the shaping re-done for gamma (potential-based: the optimum stays)."""
    return torch.tensor([st["r_time"] + st["r_penalty"] + st["r_terminal"] + gamma * st["phi_next"] - st["phi"] for st in steps])


class Decisions(torch.utils.data.Dataset):
    """the decisions of some episodes: model input (generation-mode prompt, as the engine's) + the recorded chain.
    `policy` = indices of the trained expert's decisions (the only ones the loaders are asked for)."""

    def __init__(self, episodes, model_config, server, train_expert=1):
        self.server = str(server)
        wire, _ = e2e(server)
        self.items, self.episodes = [], []                   # episodes: (dir, first item, end item, steps, meta)
        for d in episodes:
            d = Path(d); steps, meta = read_episode(d)
            self.episodes.append((d, len(self.items), len(self.items) + len(steps), steps, meta))
            for st in steps:
                o = wire.unpack((d / st["out"]).read_bytes())
                self.items.append((d, st, {k: o.get(k) for k in ("expert", "lora_version", "chain", "sigma", "logp")}))
        self.policy = [i for i, (_, _, o) in enumerate(self.items) if o["expert"] == train_expert]
        for i in self.policy:
            d, st, o = self.items[i]
            assert o["sigma"] is not None and (o["sigma"] > 0).all(), f"{d}/{st['out']}: trained-expert decision without noise"
        self.preprocess = get_preprocess_fn(model_config, generation_mode=True)

    def version(self, i):
        v = self.items[i][2]["lora_version"]
        return 0 if v is None else int(v)                    # None: no LoRA merged yet = version 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d, st, o = self.items[i]
        wire, build_sample = e2e(self.server)                # DataLoader workers import it again
        s = build_sample(wire.unpack((d / st["req"]).read_bytes()), key=f"{d.name}/{i}")
        s["tokenized_data"] = self.preprocess(data=s)
        for k in DROP: s.pop(k, None)
        s.update(chain=torch.from_numpy(o["chain"]).float(), sigma=torch.from_numpy(o["sigma"]).float(),
                 logp=torch.from_numpy(o["logp"]).float(), idx=torch.tensor(i))
        return s


def loader(ds, model_config, batch_size, indices, workers=0):
    return torch.utils.data.DataLoader(torch.utils.data.Subset(ds, indices), batch_size=batch_size, num_workers=workers,
                                       collate_fn=partial(collate_fn, model_config=model_config))
