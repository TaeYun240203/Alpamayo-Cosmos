"""Fake PPO spool at the real sizes for the learner's speed run (rl/dryrun.sh): dagger.fake's episodes (real camera JPEG
and LiDAR BEV sizes, real token counts) turned into PPO records. 4 of 5 decisions are on the trained expert set
(expert 1) with a random chain (K+1, 64, 2), sigma 0.1 and logp 0: for speed only, the log-prob gap means nothing here.
Rewards are -0.1 s per decision; critic_obs has the 50 values of the eval repo's rl_critic.FIELDS.
Usage (cwd train/): python -m rl.fake <spool> --e2e_server <eval repo src/e2e_driver/server> [--actors 1 --episodes 3 --decisions 100]
"""
import argparse, json, os
from pathlib import Path

import numpy as np

from dagger.fake import episode
from rl.episodes import e2e, write_json

K, OBS = 10, 50                                                 # Euler steps (engine), critic_obs length (rl_critic.FIELDS)


def ppo_episode(d, server, n, seed):
    wire, _ = e2e(server)
    episode(d, server, n=n, seed=seed)
    rng, steps = np.random.default_rng(seed), [json.loads(x) for x in open(d / "steps.jsonl")]
    with open(d / "steps.jsonl", "w") as f:
        for i, st in enumerate(steps):
            pol = i % 5 != 0
            out = dict(expert=int(pol), lora_version=None)
            if pol:
                out.update(chain=rng.standard_normal((K + 1, 64, 2)).astype(np.float32), sigma=np.full(K, 0.1, np.float32),
                           logp=np.zeros(K, np.float32))
            (d / st["out"]).write_bytes(wire.pack(out))
            last = i == len(steps) - 1
            st.update(r_time=-0.1, r_penalty=0.0, r_terminal=0.0, phi=0.0, phi_next=0.0, events=[], critic_obs=rng.normal(size=OBS).round(3).tolist(),
                      done=last, terminated=last, truncated=False)
            f.write(json.dumps(st) + "\n")
    (d / "dagger_labels.npz").unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("spool"); p.add_argument("--e2e_server", default=os.environ.get("E2E_SERVER"), required=not os.environ.get("E2E_SERVER"))
    p.add_argument("--actors", type=int, default=1); p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--decisions", type=int, default=100)
    a = p.parse_args()
    for j in range(a.actors):
        for k in range(a.episodes):
            ppo_episode(Path(a.spool) / f"a{j}" / f"ep{k:05d}", a.e2e_server, a.decisions, 100 * j + k)
        write_json(Path(a.spool) / f"a{j}" / "inflight.json", {"version": 0, "episode": f"ep{a.episodes:05d}"})
    print(f"{a.actors * a.episodes} fake PPO episodes x {a.decisions} decisions in {a.spool}")


if __name__ == "__main__":
    main()
