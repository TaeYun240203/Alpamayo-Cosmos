"""PPO policy for the learner (RL_PLAN 3.2): the delivered WoWmayo (QAD final/ + modelopt_state.pth, NVFP4/FP8 fake
quantized as the actor's engine) loaded frozen and a BF16-trained LoRA on every linear of the expert's decoder layers.
The camera/LiDAR ViTs and the LLM only run forward (no grad). The critic is not here: it reads privileged GT values
(learner.Critic).

Policy = the noisy Euler chain the actor samples in RL episodes (DPPO / ReinFlow style, RL_PLAN 3.1), in the normalised
(a, kappa) action space (64 x 2), t_k = linspace(0, 1, K+1)[k], dt_k = t_{k+1} - t_k as alpamayo FlowMatching._euler:
    x_0 ~ N(0, I),   x_{k+1} = x_k + dt_k v(x_k, t_k) + sigma_k eps_k,   action = x_K
    log pi(chain) = sum_k log N(x_{k+1}; x_k + dt_k v(x_k, t_k), sigma_k^2 I)      (x_0's density does not depend on the policy)
The learner recomputes all K steps of a decision in ONE expert pass (K rows reading the same prefix KV).
"""
import json, math, shutil
from pathlib import Path
from types import SimpleNamespace

import torch, torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from morai_vla.student import StudentVLA

TARGET = "expert.layers."            # LoRA on q k v o gate up down of every expert decoder layer (not action_in/out_proj)
LLM = "vlm.model.language_model.layers."   # train/dagger --lora llm+expert: the same on every LLM decoder layer


class LoRA(nn.Module):
    """y = base(x) + scale * B A x on the frozen linears, through forward hooks: module names, the delivered state dict and
    the ModelOpt quantizers stay untouched. A x reads the layer's own (fake) input quantizer output, as the merged and
    re-quantized layer on the actor does. B = 0 at start: version 0 is exactly the delivered model.
    targets: module-name prefixes (default the expert; train/dagger adds LLM)."""

    def __init__(self, model, rank=16, alpha=32, targets=(TARGET,)):
        super().__init__()
        self.rank, self.scale, self.enabled = rank, alpha / rank, True
        self.names = [n for n, m in model.named_modules() if n.startswith(tuple(targets)) and isinstance(m, nn.Linear)]
        assert all(any(n.startswith(t) for n in self.names) for t in targets), f"no linear under one of {targets}"
        self.A, self.B = nn.ParameterDict(), nn.ParameterDict()
        for n in self.names:
            m, k = model.get_submodule(n), n.replace(".", "/")
            self.A[k] = nn.Parameter(torch.randn(rank, m.in_features) / math.sqrt(m.in_features))
            self.B[k] = nn.Parameter(torch.zeros(m.out_features, rank))
            m.register_forward_hook(self._hook(k))

    def _hook(self, k):
        def add(mod, args, out):
            if not self.enabled: return None
            x, q = args[0], getattr(mod, "input_quantizer", None)
            if q is not None and q.is_enabled: x = q(x)
            return out + ((x.to(self.A[k].dtype) @ self.A[k].T) @ self.B[k].T * self.scale).to(out.dtype)
        return add

    def export(self, path, **meta):
        """BF16 safetensors {<module>.lora_A (r, in), <module>.lora_B (out, r)}; metadata scale, rank, version, ..."""
        t = {}
        for n in self.names:
            k = n.replace(".", "/")
            t[f"{n}.lora_A"] = self.A[k].detach().to(torch.bfloat16).cpu().contiguous()
            t[f"{n}.lora_B"] = self.B[k].detach().to(torch.bfloat16).cpu().contiguous()
        tmp = Path(str(path) + ".tmp")
        save_file(t, tmp, metadata={"scale": str(self.scale), "rank": str(self.rank), **{k: str(v) for k, v in meta.items()}})
        tmp.replace(path)

    def load(self, path):
        t = load_file(path)
        with torch.no_grad():
            for n in self.names:
                k = n.replace(".", "/")
                self.A[k].copy_(t[f"{n}.lora_A"]); self.B[k].copy_(t[f"{n}.lora_B"])


def gauss_logp(x, mu, sigma):
    """log N(x; mu, sigma^2 I) over the last two dims (64 x 2); sigma broadcasts from (..., )."""
    s = sigma[..., None, None]
    return (-0.5 * ((x - mu) / s) ** 2 - s.log() - 0.5 * math.log(2 * math.pi)).sum((-1, -2))


class Policy:
    """not an nn.Module on purpose: only `lora` is trained, saved and optimised; the model is the frozen base."""

    def __init__(self, base, rank=16, alpha=32, dtype="bfloat16", device="cuda"):
        self.device = torch.device(device)
        self.model = StudentVLA.build(init=str(base), attn_implementation="sdpa", dtype=dtype,
                                      device_map={"": self.device.index or 0} if self.device.type == "cuda" else None)
        self.model.to(self.device).eval().requires_grad_(False)
        self.lora = LoRA(self.model, rank, alpha).to(self.device)

    def autocast(self):
        return torch.autocast("cuda", torch.bfloat16, enabled=self.device.type == "cuda")

    @torch.no_grad()
    def prefill(self, batch, R):
        """frozen VLM prefill of the prompts (generation mode: they end in <traj_future_start>), its KV repeated R times
        per sample: the expert rows of one decision all read the same prefix."""
        m = self.model
        td = {k: v.to(self.device) for k, v in batch["tokenized_data"].items()}
        m.vlm.model._image_is_lidar = td.pop("image_is_lidar", None)
        hist = dict(ego_history_xyz=batch["ego_history_xyz"].to(self.device), ego_history_rot=batch["ego_history_rot"].to(self.device))
        ids = m.fuse_traj_tokens(td.pop("input_ids"), hist)
        assert (ids[:, -1] == m.config.traj_token_ids["future_start"]).all(), "build the prompt with generation_mode=True"
        out = m.vlm(input_ids=ids, use_cache=True, logits_to_keep=1, **td)
        out.past_key_values.batch_repeat_interleave(R)
        rep = lambda t: t.repeat_interleave(R, 0)
        return SimpleNamespace(past_key_values=out.past_key_values, rope_deltas=rep(out.rope_deltas)), rep(ids), rep(td["attention_mask"])

    def velocity(self, pre, x, t, lora=True):
        """expert velocity at rows (x (N, 64, 2), t (N,)); N = samples x R."""
        self.lora.enabled = lora
        try:
            return self.model._velocity(*pre, x, t.view(-1, 1, 1)).float()
        finally:
            self.lora.enabled = True

    def evaluate(self, batch, kl=False):
        """the learner's pass over decisions: per-step log-probs of the recorded chains (B, K) and, with kl, the
        closed-form KL to the delivered policy (LoRA off) summed over steps (B,)."""
        chain, sigma = batch["chain"].to(self.device).float(), batch["sigma"].to(self.device).float()
        B, K = sigma.shape
        t = torch.linspace(0, 1, K + 1, device=self.device)
        with self.autocast():
            pre = self.prefill(batch, K)
            x, tt = chain[:, :K].flatten(0, 1), t[:K].repeat(B)
            v = self.velocity(pre, x, tt).view(B, K, *chain.shape[2:])
            mu = chain[:, :K] + (t[1:] - t[:-1]).view(1, K, 1, 1) * v
            out = {"logp": gauss_logp(chain[:, 1:], mu, sigma)}
            if kl:
                with torch.no_grad():
                    v0 = self.velocity(pre, x, tt, lora=False).view(B, K, *chain.shape[2:])
                mu0 = chain[:, :K] + (t[1:] - t[:-1]).view(1, K, 1, 1) * v0
                out["kl"] = ((mu - mu0) ** 2 / (2 * sigma[..., None, None] ** 2)).sum((-1, -2, -3))
        return out

    @torch.no_grad()
    def sample(self, batch, sigma, generator=None):
        """reference actor (tests; the spec of the actor engine's RL sampler): noisy Euler from x_0 ~ N(0, I), one expert
        call per step. sigma (K,) per-step std (0 = deterministic step, logp 0). Returns chain (B, K+1, 64, 2), logp (B, K)."""
        K = len(sigma)
        t = torch.linspace(0, 1, K + 1, device=self.device)
        with self.autocast():
            pre = self.prefill(batch, 1)
            B = pre[1].shape[0]
            x = torch.randn(B, *self.model.action_space.get_action_space_dims(), generator=generator).to(self.device)
            chain, lp = [x], []
            for k in range(K):
                mu = x + (t[k + 1] - t[k]) * self.velocity(pre, x, t[k].expand(B))
                if sigma[k] > 0:
                    x = mu + sigma[k] * torch.randn(mu.shape, generator=generator).to(self.device)
                    lp.append(gauss_logp(x, mu, torch.tensor(float(sigma[k]), device=self.device)))
                else:
                    x = mu; lp.append(torch.zeros(B, device=self.device))
                chain.append(x)
        return torch.stack(chain, 1), torch.stack(lp, 1)


def merge(deliver, lora_path, out):
    """a new delivery dir (same layout as qad_export.py deliver) with W + scale * B A in the expert linears, for the
    actor's engine conversion (RL_PLAN 3.1 / 5). Quantizer amax and quant_manifest.json stay as delivered: the merged
    weights are quantized on the same grid (the delivered amax are QAD's calibrated values, not max|W|)."""
    deliver, out = Path(deliver), Path(out)
    shutil.copytree(deliver, out, ignore=shutil.ignore_patterns("*.safetensors"))
    with safe_open(lora_path, "pt") as h:
        scale = float(h.metadata()["scale"])
        lora = {k: h.get_tensor(k) for k in h.keys()}
    names = {k.rsplit(".", 1)[0] for k in lora}
    done = set()
    for f in sorted((deliver / "final").glob("*.safetensors")):
        with safe_open(f, "pt") as h:
            meta, t = h.metadata(), {k: h.get_tensor(k) for k in h.keys()}
        for n in names:
            if n + ".weight" in t:
                w = t[n + ".weight"]
                t[n + ".weight"] = (w.float() + scale * lora[n + ".lora_B"].float() @ lora[n + ".lora_A"].float()).to(w.dtype)
                done.add(n)
        save_file(t, out / "final" / f.name, metadata=meta)
    assert done == names, f"LoRA layers not in {deliver}: {sorted(names - done)[:5]}"
    json.dump({"lora": str(lora_path), "layers": len(done), "scale": scale}, open(out / "lora_merged.json", "w"), indent=1)
    return len(done)
