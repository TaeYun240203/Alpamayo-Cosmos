#!/usr/bin/env bash
# checkpoints for stage B (VAST_SETUP.md §4.3), in their own small venv so they download while flash-attn builds
set -euo pipefail
[ -x /opt/hfenv/bin/hf ] || { uv venv -q /opt/hfenv && uv pip install -q --python /opt/hfenv/bin/python "huggingface_hub[hf_xet]"; }
export HF_HOME=/workspace/hf HF_XET_HIGH_PERFORMANCE=1
H=/opt/hfenv/bin/hf
[ "${SKIP_RELEASE:-0}" = 1 ] || $H download nvidia/Alpamayo-1.5-10B --revision 7aba8293c09993f2e125c6819df05d7fa3e873ea --local-dir /workspace/ckpt/Alpamayo-1.5-10B   # student-only runs (R2 fine-tune): not read
# the student loads Qwen/Qwen3-VL-4B-Instruct by repo id from the HF cache (configs/student_base.yaml), not from a --local-dir
$H download Qwen/Qwen3-VL-4B-Instruct --revision ebb281ec70b05090aa6165b016eac8ec08e71b17
$H download Qwen/Qwen3-VL-8B-Instruct --include '*.json' --include '*.txt'
# --revision with a commit hash writes no refs/main, so offline from_pretrained("Qwen/...") can't resolve the repo id
for r in "Qwen--Qwen3-VL-4B-Instruct ebb281ec70b05090aa6165b016eac8ec08e71b17"; do set -- $r; d=$HF_HOME/hub/models--$1; [ -d $d/snapshots/$2 ] && mkdir -p $d/refs && printf %s $2 > $d/refs/main; done
du -sh /workspace/ckpt/* /workspace/hf 2>/dev/null || true   # ckpt/ is empty with SKIP_RELEASE: not a failure
echo CKPT_DONE
