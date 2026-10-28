"""morai_vla.speed.vit_sdpa_grouped: one Qwen3-VL vision attention layer, per-image upstream vs batched by image size
(sdpa and eager, mixed sizes as in a MORAI sample: 3 camera sizes interleaved, repeated). CPU, a few ms."""
import sys
from pathlib import Path

import pytest, torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as M
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from morai_vla import speed  # noqa: E402


@pytest.mark.parametrize("impl", ["sdpa", "eager"])
def test_grouped_vision_attention_equals_per_image(impl):
    torch.manual_seed(0)
    cfg = Qwen3VLVisionConfig(hidden_size=32, num_heads=2); cfg._attn_implementation = impl
    att = M.Qwen3VLVisionAttention(cfg).eval()
    lengths = [12, 12, 8, 8, 5, 12, 8, 5]
    cu = torch.tensor([0] + lengths).cumsum(0).int()
    x = torch.randn(sum(lengths), 32)
    ang = torch.randn(sum(lengths), 8); pe = (ang.cos().repeat(1, 2), ang.sin().repeat(1, 2))
    try:
        speed.apply(cfg={}); ref = att(x, cu, position_embeddings=pe)
        speed.apply(cfg={"vit_sdpa_grouped": True}); got = att(x, cu.clone(), position_embeddings=pe)
    finally:
        speed.apply(cfg={})
    assert M.Qwen3VLVisionAttention.forward is speed._ORIG
    torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-6)


def test_gate_shards_merge_like_one_process(tmp_path):
    """scripts/gate_student.merge of k::n shards = the one-process summary (rows back in order, same means)."""
    import json
    from scripts.gate_student import merge, summarize
    rows = [dict(idx=i, key=f"e/{i}", cell="C1" if i % 3 else "C2", teacher_ade=1.0 + 0.1 * i, teacher_fde=2.0, teacher_minade=0.5,
                 student_ade=1.2 + 0.05 * i, student_fde=2.5, student_minade=0.6, st_ade=0.3) for i in range(7)]
    summarize(rows, tmp_path / "one")
    for k in range(3):
        (tmp_path / f"s{k}").mkdir(); json.dump(rows[k::3], open(tmp_path / f"s{k}" / "gate.json", "w"))
    merge(tmp_path / "merged", [tmp_path / f"s{k}" for k in range(3)])
    for f in ("gate.json", "summary.json"):
        assert json.load(open(tmp_path / "merged" / f)) == json.load(open(tmp_path / "one" / f))
