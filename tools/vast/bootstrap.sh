#!/usr/bin/env bash
# Stage-B/C instance setup on nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 (VAST_SETUP.md §2.4, §5): OS tools, uv + vastai from
# PyPI, the cost watchdog (started first), and alpamayo-recipes @2e907da's a1_5_sft venv with flash-attn built from the PyPI sdist
# for sm90. Rerunnable. Usage on the instance: BUDGET_USD=40 EXTRA_USD=0 PRIOR_USD=0 bash /workspace/bootstrap.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
mkdir -p /workspace && cd /workspace
apt-get update -qq && apt-get install -y -qq --no-install-recommends git python3 python3-venv python3.12-dev build-essential ninja-build tmux rsync > /dev/null
[ -x /opt/uvtool/bin/vastai ] || { python3 -m venv /opt/uvtool && /opt/uvtool/bin/pip install -q uv vastai; }
ln -sf /opt/uvtool/bin/uv /opt/uvtool/bin/vastai /usr/local/bin/
pgrep -f idle_watchdog.sh > /dev/null || { nohup bash /workspace/idle_watchdog.sh > /dev/null 2>&1 & }
[ -d alpamayo-recipes ] || git clone -q https://github.com/NVlabs/alpamayo-recipes.git
cd alpamayo-recipes && git checkout -q 2e907da && cd recipes/alpamayo1_5_sft
export FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS:-90} MAX_JOBS=${MAX_JOBS:-16}
[ -d a1_5_sft ] || [ ! -f /workspace/envpack/a1_5_sft_venv.tar ] || tar xf /workspace/envpack/a1_5_sft_venv.tar   # venv built on stage B, from B2 env/
[ -d a1_5_sft ] || uv venv -q --python /usr/bin/python3.12 a1_5_sft
source a1_5_sft/bin/activate
uv sync --active || { uv sync --active --no-install-package flash-attn && uv sync --active; }
python -c "import torch, flash_attn; print(torch.__version__, torch.cuda.get_device_name(), flash_attn.__version__)"
