"""Several-GPU learner check on CPU (gloo, quantizers off, the tiny delivery of tiny_test): two learner ranks train on a
fake spool, then resume; the learner itself asserts that every rank holds the same LoRA after each update.
    E2E_SERVER=<eval repo src/e2e_driver/server> bash rl/ddp_test.sh <tiny deliver> <work dir>   (cwd train/)
Steps (ddp_test.sh): prep 0 -> torchrun 2 ranks, 1 update -> prep 1 (new episodes, actors on version 1) -> torchrun again
(resume, 1 update) -> check.
"""
import json, os, sys
from pathlib import Path

import torch

from rl import learner
from rl.episodes import write_json
from rl.tiny_test import D, CpuPolicy, write_episode


def run(deliver, work):
    learner.Policy = CpuPolicy
    learner.main(["--base", str(deliver / "final"), "--spool", str(work / "spool"), "--out", str(work / "out"), "--device", "cpu",
                  "--dtype", "float32", "--e2e_server", os.environ["E2E_SERVER"], "--rank", "8", "--alpha", "16", "--lr", "2e-3",
                  "--min_decisions", str(2 * D), "--micro", "4", "--minibatch", "16", "--workers", "0", "--max_updates", "1",
                  "--poll", "0.2", "--kl_target", "5", "--max_lag", "1"])


def prep(deliver, work, version):
    P = CpuPolicy(deliver / "final", rank=8, alpha=16, dtype="float32", device="cpu")
    gen = torch.Generator().manual_seed(version)
    for e in range(3):                                            # 3 episodes x D decisions, odd counts per rank
        write_episode(work / "spool" / "a0" / f"v{version}e{e}", P, 7000 + 10 * version + e, version, gen, os.environ["E2E_SERVER"])
    write_json(work / "spool" / "a0" / "inflight.json", {"version": version, "episode": f"v{version}e3"})


def check(work):
    lines = [json.loads(x) for x in open(work / "out" / "log.jsonl")]
    latest = json.load(open(work / "out" / "lora" / "latest.json"))
    assert latest["version"] == 2 and [x["ranks"] for x in lines] == [2, 2] and [x["episodes"] for x in lines] == [3, 3], (latest, lines)
    assert all(x["opt_steps"] > 0 and x["behav_gap_std"] == x["behav_gap_std"] and "kl_coef_next" in x for x in lines), lines
    print("PASS", json.dumps([{k: x[k] for k in ("version", "policy_decisions", "opt_steps", "pg", "clip_frac", "behav_gap_mean",
                                                   "kl_sft", "kl_coef_next", "s_per_policy_decision")} for x in lines]))


if __name__ == "__main__":
    mode, rest = sys.argv[1], [Path(x) for x in sys.argv[2:]]
    {"run": lambda: run(*rest), "prep0": lambda: prep(*rest, 0), "prep1": lambda: prep(*rest, 1), "check": lambda: check(rest[-1])}[mode]()
