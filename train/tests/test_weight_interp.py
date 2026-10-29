"""scripts/weight_interp.py on small fake sharded checkpoints (two shards + index + config, bf16 / fp32 / int64 tensors):
alpha 0 = A and alpha 1 = B bit for bit, alpha 0.5 = the mean, the output has A's file layout (names, keys, dtypes,
shapes, safetensors metadata, non-weight files), and any key / shape / integer-buffer mismatch fails. No model, CPU."""
import json, sys
from pathlib import Path

import pytest, torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.weight_interp import interp  # noqa: E402

SHARDS = {"model-00001-of-00002.safetensors": ("vlm.w", "vlm.b"), "model-00002-of-00002.safetensors": ("expert.w", "expert.freq", "expert.step")}


def ckpt(d, seed, change=None):
    g = torch.Generator().manual_seed(seed); d.mkdir()
    make = {"vlm.w": lambda: torch.randn(8, 4, generator=g).bfloat16(), "vlm.b": lambda: torch.randn(4, generator=g).bfloat16(),
            "expert.w": lambda: torch.randn(3, 5, generator=g).bfloat16(), "expert.freq": lambda: torch.arange(6, dtype=torch.float32),
            "expert.step": lambda: torch.tensor([7, 9])}
    wm = {}
    for f, keys in SHARDS.items():
        sd = {k: make[k]() for k in keys}
        if change: change(sd)
        save_file(sd, str(d / f), metadata={"format": "pt"}); wm |= {k: f for k in sd}
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 0}, "weight_map": wm}))
    (d / "config.json").write_text(json.dumps({"model_type": "tiny"})); (d / "generation_config.json").write_text("{}")
    return d


def tensors(d):
    return {k: v for f in SHARDS for k, v in load_file(str(d / f)).items()}


@pytest.fixture
def ab(tmp_path):
    return ckpt(tmp_path / "a", 0), ckpt(tmp_path / "b", 1)


def test_alpha_endpoints_and_mean(ab, tmp_path):
    a, b = ab; A, B = tensors(a), tensors(b)
    for alpha, want in ((0.0, A), (1.0, B)):
        out = tmp_path / f"m{alpha}"; interp(a, b, alpha, out)
        M = tensors(out)
        assert M.keys() == want.keys() and all(M[k].dtype == want[k].dtype and torch.equal(M[k], want[k]) for k in M)
    out = tmp_path / "m05"; interp(a, b, 0.5, out); M = tensors(out)
    for k in M:
        assert M[k].dtype == A[k].dtype and M[k].shape == A[k].shape
        torch.testing.assert_close(M[k], ((A[k].double() + B[k].double()) / 2).to(A[k].dtype) if A[k].is_floating_point() else A[k])
    assert sorted(p.name for p in out.iterdir()) == sorted([*SHARDS, "model.safetensors.index.json", "config.json", "generation_config.json", "interp.json"])
    assert (out / "model.safetensors.index.json").read_text() == (a / "model.safetensors.index.json").read_text()
    with safe_open(out / next(iter(SHARDS)), "pt") as h: assert h.metadata() == {"format": "pt"}
    assert json.loads((out / "interp.json").read_text())["alpha"] == 0.5
    with pytest.raises(AssertionError): interp(a, b, 0.5, out)                    # never overwrites an output


@pytest.mark.parametrize("change", [
    lambda sd: sd.update({"vlm.extra": torch.zeros(2).bfloat16()}) if "vlm.w" in sd else None,      # extra key
    lambda sd: sd.pop("expert.w", None),                                                            # missing key
    lambda sd: sd.update({"vlm.b": torch.zeros(5).bfloat16()}) if "vlm.b" in sd else None,          # shape
    lambda sd: sd.update({"expert.step": torch.tensor([7, 8])}) if "expert.step" in sd else None,   # integer buffer
])
def test_mismatch_fails(tmp_path, change):
    a, b = ckpt(tmp_path / "a", 0), ckpt(tmp_path / "b", 1, change)
    with pytest.raises(AssertionError): interp(a, b, 0.5, tmp_path / "m")
    assert not (tmp_path / "m").exists()
