"""Trajectory-ADE early stop (user 10/25 rule): the stop count, the LR decay, per-measurement snapshots, resume and the
final choice, with scripted ADE values on a toy model (no data, CPU)."""
import json
from types import SimpleNamespace

import torch

from morai_vla.early_stop import TrajEarlyStop, stop_now


def test_stop_rule():
    h = lambda *a: list(zip(range(0, 100 * len(a), 100), a))
    assert not stop_now(h(5.0, 4.0, 3.9, 3.85, 3.84), min_step=0)              # 2 no-improvements only
    assert stop_now(h(5.0, 4.0, 3.95, 3.93, 3.99), min_step=0)                  # 3 in a row not > 2 % below the best before them
    assert not stop_now(h(5.0, 4.0, 3.95, 3.93, 3.80), min_step=0)              # 3.80 < 0.98 * 3.93: the streak resets
    assert not stop_now(h(5.0, 4.0, 4.1, 4.1, 4.1), min_step=500)               # before 0.5 epoch nothing counts
    assert stop_now(h(5.0, 4.0, 4.1, 4.1, 4.1, 4.1), min_step=200)              # counted from step 200 on: 200, 300, 400


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.w = torch.nn.Parameter(torch.zeros(3)); self.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)


def run(tmp_path, ades, max_steps=40, every=4, decay=6, per_epoch=8, start=0, model=None):
    """drive the callback like HF does; ADE of the k-th measurement = ades[k]; the toy weight = the step it was measured at."""
    tmp_path.mkdir(exist_ok=True)                                                         # train.py creates output_dir
    model = model or Toy(); opt = torch.optim.SGD(model.parameters(), lr=1.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 - 0.5 * s / max_steps)
    cb = TrajEarlyStop(root=None, every_steps=every, decay_steps=decay)
    it = iter(ades); cb.score = lambda m, r, w: [("k", next(it))]
    args = SimpleNamespace(output_dir=str(tmp_path), gradient_accumulation_steps=1)
    st = SimpleNamespace(global_step=start, max_steps=max_steps, is_world_process_zero=True)
    for _ in range(start): sched.step()
    cb.on_train_begin(args, st, None, train_dataloader=range(per_epoch)); lrs = []
    while st.global_step < max_steps:
        cb.on_step_begin(args, st, None, lr_scheduler=sched)
        lrs.append(opt.param_groups[0]["lr"]); sched.step(); st.global_step += 1
        with torch.no_grad(): model.w.fill_(st.global_step)
        ctl = SimpleNamespace(should_training_stop=False)
        cb.on_step_end(args, st, ctl, model=model, lr_scheduler=sched)
        if ctl.should_training_stop: break
    return cb, model, lrs, st.global_step


def test_stop_decay_snapshots_and_choice(tmp_path):
    # measurements at 4, 8, 12, 16, 20; min_step 4 (0.5 x 8); 12, 16, 20 are not > 2 % below 3.0 -> stop at 20
    cb, model, lrs, last = run(tmp_path, [4.0, 3.0, 2.99, 3.1, 3.05, 2.5])
    st = json.loads((tmp_path / "ade_early_stop.json").read_text())
    assert st["stop_step"] == 20 and st["decay_end"] == 26 and last == 26                # 6 decay steps, then the run ends
    f = 1.0 - 0.5 * 20 / 40
    assert abs(lrs[20] - f) < 1e-9 and abs(lrs[25] - f / 6) < 1e-9                         # linear from the scheduled LR to 0
    assert [h["step"] for h in st["history"]] == [4, 8, 12, 16, 20, 26] and st["history"][-1]["tag"] == "after_decay"
    assert all((tmp_path / h["snapshot"]).exists() for h in st["history"])                # one snapshot per measurement
    sel = cb.finalize(model, tmp_path)
    assert sel["chosen_step"] == 26 and float(model.w[0]) == 26 and float(model.frozen[0]) == 1   # lowest ADE: after the decay
    assert json.loads((tmp_path / "ade_select.json").read_text())["chosen_ade"] == 2.5


def test_best_before_decay_wins_and_end_without_stop(tmp_path):
    cb, model, _, _ = run(tmp_path / "a", [4.0, 3.0, 2.99, 3.1, 3.05, 3.2]); cb.finalize(model, tmp_path / "a")
    assert float(model.w[0]) == 12                                                          # 2.99 at step 12 beats 3.2 after decay
    cb, model, _, last = run(tmp_path / "b", [5, 4, 3, 2.5, 2, 1.5, 1.2, 1.0, 0.9, 0.8], max_steps=38)
    st = json.loads((tmp_path / "b" / "ade_early_stop.json").read_text())
    assert st["stop_step"] is None and last == 38 and st["history"][-1] == {**st["history"][-1], "step": 38, "tag": "end"}


def test_resume_drops_later_measurements_and_keeps_count(tmp_path):
    run(tmp_path, [4.0, 3.0, 2.99, 3.1, 3.05, 2.5])                                      # stopped at 20, decayed to 26
    cb, _, lrs, last = run(tmp_path, [3.05, 2.4], start=16)                               # resumed from a step-16 checkpoint
    st = json.loads((tmp_path / "ade_early_stop.json").read_text())
    assert [h["step"] for h in st["history"]] == [4, 8, 12, 16, 20, 26] and st["history"][-1]["ade"] == 2.4 and last == 26
