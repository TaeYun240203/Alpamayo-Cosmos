#!/usr/bin/env bash
# Training venv (validated on winpc-wsl RTX 5080, 2026-10-24). Sources: PyPI, download.pytorch.org, NVlabs GitHub only.
# Usage: bash setup_env.sh <venv dir> [--flash-attn]
set -euo pipefail
V=${1:?venv dir}; PY=${PYTHON:-python3.12}
$PY -m venv "$V"
"$V/bin/pip" install -q --upgrade pip
"$V/bin/pip" install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"$V/bin/pip" install -q transformers==4.57.1 accelerate==1.13.0 deepspeed==0.19.1 hydra-core==1.3.2 omegaconf einops colorlog \
    scipy pandas pyarrow pillow rich safetensors torchmetrics==1.8.2 bitsandbytes matplotlib pytest
# NVIDIA code at the commits the recipe pins (recipe 2e907da; its uv.lock pins alpamayo f3198c2). --no-deps: their own
# dependency lists pull flash-attn builds and the PAI SDK, which MORAI training does not use.
"$V/bin/pip" install -q --no-deps \
    "alpamayo_r1 @ git+https://github.com/NVlabs/alpamayo.git@f3198c2f6158e1158ae8cb4fc7d7716bd25085bb" \
    "alpamayo-recipes @ git+https://github.com/NVlabs/alpamayo-recipes.git@2e907dab31e5353f42b2c8a2f79d7e39b6e0d42c#subdirectory=src" \
    "alpamayo1-5-sft @ git+https://github.com/NVlabs/alpamayo-recipes.git@2e907dab31e5353f42b2c8a2f79d7e39b6e0d42c#subdirectory=recipes/alpamayo1_5_sft"
if [[ "${2:-}" == "--flash-attn" ]]; then           # H100: PyPI sdist, builds against the installed torch (needs nvcc, ~20 min)
    "$V/bin/pip" install -q ninja packaging && MAX_JOBS=8 "$V/bin/pip" install -q flash-attn==2.8.3 --no-build-isolation
fi
"$V/bin/python" -c "import torch, transformers, alpamayo_r1, alpamayo, alpamayo1_5_sft; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name())"
