"""Speed-ups that leave the training math unchanged (README "R2 속도 최적화", 10/28), off by default; the `speed` config block
(configs/student_base.yaml) turns them on once the single-GPU check (scripts/ft_speed_probe.py) shows identical losses:
  vit_sdpa_grouped  ViT attention without flash-attn (sdpa: RTX PRO 6000 / 5090 training, every gate and measurement):
                    upstream runs one sdpa call per image per layer (16 camera frames) and syncs the GPU for the lengths
                    in every layer; here the images of one size go through one batched call (3 camera sizes, 1 BEV size)
                    and the lengths are read once per ViT call. Each image still attends only to itself.
  teacher_vit_only  StudentVLA: on a micro-batch of GT rows only (R2 mixture, ~75 % at batch 1) T* runs its ViTs, not its
                    LLM: the GT rows use T* for ViT KD only (no kv, vkd, coc_kl), and the ViTs run before the LLM.
  lazy_loss_log     StudentVLA: loss components are summed on the GPU and read at the log line, not with one GPU sync
                    per component per micro-batch.
"""
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as M

_ORIG = M.Qwen3VLVisionAttention.forward


def _grouped(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs):
    """Qwen3VLVisionAttention.forward (transformers 4.57.1) with the non-flash branch batched by image length."""
    if self.config._attn_implementation == "flash_attention_2":
        return _ORIG(self, hidden_states, cu_seqlens, rotary_pos_emb, position_embeddings, **kwargs)
    n = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(n, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    cos, sin = position_embeddings
    q, k = M.apply_rotary_pos_emb_vision(q, k, cos, sin)
    q, k, v = (x.transpose(0, 1).unsqueeze(0) for x in (q, k, v))
    attn = M.eager_attention_forward if self.config._attn_implementation == "eager" else M.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    lengths = getattr(cu_seqlens, "_lengths", None)                 # every block of one ViT call gets the same tensor
    if lengths is None:
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist(); cu_seqlens._lengths = lengths
    parts = [x.split(lengths, dim=2) for x in (q, k, v)]
    out = [None] * len(lengths)
    for size in dict.fromkeys(lengths):
        idx = [i for i, s in enumerate(lengths) if s == size]
        o = attn(self, *(torch.cat([p[i] for i in idx]) for p in parts), attention_mask=None, scaling=self.scaling,
                 dropout=0.0 if not self.training else self.attention_dropout, is_causal=False, **kwargs)[0]
        for i, x in zip(idx, o.split(1)): out[i] = x
    return self.proj(torch.cat(out, dim=1).reshape(n, -1).contiguous())


def apply(model=None, cfg=None):
    """cfg: the `speed` config block (dict-like, missing keys = off). model: a StudentVLA (or None for the ViT patch only)."""
    cfg = dict(cfg or {})
    M.Qwen3VLVisionAttention.forward = _grouped if cfg.get("vit_sdpa_grouped") else _ORIG
    if model is not None:
        model.teacher_vit_only = bool(cfg.get("teacher_vit_only"))
        model.lazy_loss_log = bool(cfg.get("lazy_loss_log"))
    return cfg
