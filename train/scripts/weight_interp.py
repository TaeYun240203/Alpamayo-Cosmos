"""Weight interpolation of two WoWmayo checkpoints with the same parent (reports/interp_plan_20261029.md; 10/29: R2 ft
final = A, dynbal ft final = B, which started from A): every tensor theta = (1 - alpha) A + alpha B.
Floating tensors lerp in fp32 and go back to their own dtype (alpha 0 / 1 give A / B bit for bit); any other tensor
(integer / bool buffers) must be equal in A and B and is copied. Shard by shard, one output shard in memory: A and B must
have the same shard files, and per shard the same keys, shapes and dtypes. The other files of A (config.json,
model.safetensors.index.json, ...) are copied and the shards keep A's names and safetensors metadata, so
StudentVLA.build(init=out) and scripts/ft_measure.BF16(out) load the result exactly as they load A.
BF16 fine-tune finals only: QAD outputs (modelopt_state.pth) quantize each layer in a model-specific format, so mix the
ft finals and run QAD on the mix. Writes <out>.partial, renamed to <out> at the end (<out> must not exist).
Usage (CPU is enough): python scripts/weight_interp.py --a <dir> --b <dir> --alpha 0.5 --out <dir>
"""
import argparse, json, shutil, time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def interp(a, b, alpha, out):
    a, b, out = Path(a), Path(b), Path(out)
    assert 0.0 <= alpha <= 1.0, alpha
    assert not out.exists() and out.resolve() not in (a.resolve(), b.resolve()), out
    names = sorted(p.name for p in a.glob("*.safetensors"))
    assert names and names == sorted(p.name for p in b.glob("*.safetensors")), (names, sorted(p.name for p in b.glob("*.safetensors")))
    ca, cb = (json.loads((d / "config.json").read_text()) for d in (a, b))
    if ca != cb: print("WARNING config.json differs (A's is kept):", sorted(k for k in ca.keys() | cb.keys() if ca.get(k) != cb.get(k)), flush=True)
    tmp = out.with_name(out.name + ".partial"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    for name in names:
        t = time.time(); sd = {}
        with safe_open(a / name, "pt") as ha, safe_open(b / name, "pt") as hb:
            ka, kb = set(ha.keys()), set(hb.keys())
            assert ka == kb, (name, sorted(ka ^ kb)[:10])
            for k in sorted(ka):
                x, y = ha.get_tensor(k), hb.get_tensor(k)
                assert x.shape == y.shape and x.dtype == y.dtype, (k, tuple(x.shape), tuple(y.shape), x.dtype, y.dtype)
                if x.is_floating_point():
                    sd[k] = torch.lerp(x.float(), y.float(), alpha).to(x.dtype)
                else:
                    assert torch.equal(x, y), f"{k}: non-floating tensor differs between A and B"
                    sd[k] = x
            meta = ha.metadata()
        save_file(sd, str(tmp / name), metadata=meta)
        print(f"{name}: {len(sd)} tensors, {time.time() - t:.0f} s", flush=True); del sd
    for f in a.iterdir():
        if f.is_file() and f.suffix != ".safetensors": shutil.copy2(f, tmp / f.name)
        elif f.is_dir(): print("not copied (directory):", f.name, flush=True)
    (tmp / "interp.json").write_text(json.dumps({"a": str(a), "b": str(b), "alpha": alpha}, indent=1))
    tmp.rename(out)


def main():
    ap = argparse.ArgumentParser()
    for k in ("--a", "--b", "--out"): ap.add_argument(k, required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    x = ap.parse_args()
    interp(x.a, x.b, x.alpha, x.out); print("done", x.out)


if __name__ == "__main__":
    main()
