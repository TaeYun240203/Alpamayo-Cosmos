"""DAgger-row loss and the LLM checkpointing the llm+expert LoRA needs on small GPUs (train/dagger/README.md).

DAgger rows (user 10/28): hard targets on the shadow label (a, kappa), both dims, the label's valid mask:
  fm       flow matching through the expert (gradient into the LLM KV, as R2 ft), on the label normalised with the
           model's own (a, kappa) stats; per row the mean over valid points, then the mean over rows
  traj_ce  the discrete trajectory tokens of the label (the tokenizer's 3000 bins over [-10, 10] normalised; token 2t + d
           = point t, dim d) teacher-forced, CE on valid points + <traj_future_start> + <traj_future_end>, as R2 ft's
           traj_ce. An invalid point's token is the bin of 0 (masked from the CE; the expert never reads past
           <traj_future_start>)
No lm (no CoC, no answer text loss), no history dropout (the actor's real history is the state DAgger visited).
Row weight w (batch key "w", configs/dagger.yaml dagger.weights): multiplies the row's fm and its tokens' CE; the means stay
over rows / tokens (a x2 row counts as two; dividing by the weight sum would cancel it at micro-batch 1).
Replay rows go through StudentVLA.forward unchanged (R2 ft's losses).
"""
import zlib

import torch
from alpamayo_r1.models.base_model import IGNORE_INDEX, replace_pad_token
from torch.utils.checkpoint import checkpoint


def normalize(m, ak):
    """physical (a, kappa) (..., 64, 2) -> the model's normalised action."""
    s = m.action_space
    return torch.stack([(ak[..., 0] - s.accel_mean.float()) / s.accel_std.float(),
                        (ak[..., 1] - s.curvature_mean.float()) / s.curvature_std.float()], -1)


def future_tokens(m, x):
    """normalised action (B, 64, 2) -> future token ids (B, 128), as DiscreteTrajectoryTokenizer.encode after traj_to_action."""
    tk = m.traj_tokenizer
    lo, hi = (torch.tensor(v, dtype=x.dtype, device=x.device) for v in (tk.dims_min, tk.dims_max))
    return ((x - lo) / (hi - lo) * (tk.num_bins - 1)).round().long().clamp(0, tk.num_bins - 1).reshape(len(x), -1) + m.future_token_start_idx


def dagger_forward(m, b, seed=None):
    """-> (loss, per-row {fm (B,), ce_sum (B,), ce_n (B,)}, unweighted). b: collated DAgger rows (training prompts) + ak
    (B, 64, 2), valid (B, 64), optional w (B,). seed: fixed noise and t (validation), else the global RNG (training)."""
    w = m.loss_weights
    td = dict(b["tokenized_data"]); is_lidar = td.pop("image_is_lidar", None); raw = td.pop("input_ids")
    dev = raw.device
    ids = m.fuse_traj_tokens(raw, dict(ego_history_xyz=b["ego_history_xyz"], ego_history_rot=b["ego_history_rot"]))
    valid = b["valid"].to(dev).bool()
    rw = torch.as_tensor(b["w"], dtype=torch.float32, device=dev).view(-1) if "w" in b else torch.ones(len(raw), device=dev)
    x = normalize(m, b["ak"].to(dev).float()).masked_fill(~valid[..., None], 0.0)
    fut = raw == m.config.traj_token_ids["future"]
    assert (fut.sum(1) == x.shape[1] * x.shape[2]).all(), "training prompts (generation_mode=False) with 128 future tokens"
    ids = replace_pad_token(ids, future_tokens(m, x), m.config.traj_token_ids["future"])
    keep_tok = torch.zeros_like(fut)
    keep_tok[fut] = valid.repeat_interleave(2, 1).flatten()
    traj = ((ids >= m.future_token_start_idx) & (ids < m.future_token_start_idx + m.config.traj_vocab_size) & keep_tok
            | (ids == m.special_token_ids["traj_future_start"]) | (ids == m.special_token_ids["traj_future_end"]))
    labels = torch.where(b["labels_mask"].to(dev) & traj, ids, IGNORE_INDEX)
    attn = td["attention_mask"]
    keep = ids.shape[1] - int(b["labels_mask"].float().argmax(1)[b["labels_mask"].any(1)].min()) + 1
    labels = labels[:, -keep:]
    with torch.autocast("cuda", torch.bfloat16, enabled=ids.is_cuda):
        m.vlm.model._image_is_lidar = is_lidar
        out = m.vlm(input_ids=ids, use_cache=True, logits_to_keep=keep, **td)
        lb = labels[:, 1:]; sel = lb != IGNORE_INDEX
        ce = torch.nn.functional.cross_entropy(out.logits[:, :-1][sel].float(), lb[sel], reduction="none")
        row = sel.nonzero()[:, 0]
        ce_sum = torch.zeros(len(ids), device=dev).index_add(0, row, ce.detach()); ce_n = sel.sum(1).float()
        with torch.random.fork_rng(devices=[dev] if ids.is_cuda else [], enabled=seed is not None):
            if seed is not None: torch.manual_seed(seed)
            t = m.diffusion.construct_training_data(x)
        v = m._velocity(out, ids, attn, t["noisy_x"], t["timesteps"], detach=False).float()
        err = ((v - (t["x"] - t["noise"]).float()) ** 2 * valid[..., None]).sum((1, 2)) / (2 * valid.sum(1)).clamp(min=1)
        has = valid.any(1)
        losses = {"fm": (err * rw)[has].mean(), "traj_ce": (ce * rw[row]).mean() if len(ce) else ce.sum()}
    return sum(w[k] * v for k, v in losses.items()), {"fm": err.detach(), "ce_sum": ce_sum, "ce_n": ce_n}


def row_seed(keys):
    return zlib.crc32("|".join(keys).encode())


class _Replay:
    """the cache a checkpointed LLM layer sees when torch recomputes it in backward: hands its K/V straight back (the
    prefill wrote them into the real cache the first time; DynamicLayer.update on an empty layer = cat([], k))."""

    def update(self, k, v, *a, **kw):
        return torch.cat([k.new_empty(0), k], -2), torch.cat([v.new_empty(0), v], -2)


def checkpoint_llm(m):
    """activation checkpointing on every LLM decoder layer that KEEPS the prefill KV (HF's gradient checkpointing drops the
    cache, which the expert reads): non-reentrant torch checkpoint per layer; the first call writes the real cache, the
    recompute in backward gets _Replay. Math unchanged (tests/test_dagger.py compares gradients)."""
    for layer in m.vlm.model.language_model.layers:
        def fwd(hidden_states, *args, _f=layer.forward, past_key_values=None, **kw):
            calls = [0]

            def run(h):
                calls[0] += 1
                return _f(h, *args, past_key_values=past_key_values if calls[0] == 1 or past_key_values is None else _Replay(), **kw)
            return checkpoint(run, hidden_states, use_reentrant=False) if torch.is_grad_enabled() else run(hidden_states)
        layer.forward = fwd
