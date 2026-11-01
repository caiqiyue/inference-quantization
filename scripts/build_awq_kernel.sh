#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AWQ_PYTHON="${AWQ_PYTHON:-/home/qiyue.cai/.conda/envs/awq/bin/python}"

cd "$ROOT_DIR/llm-awq/awq/kernels"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}" \
MAX_JOBS="${MAX_JOBS:-2}" \
"$AWQ_PYTHON" -c "import sys,runpy; sys.argv=['setup.py','build_ext','--inplace']; import torch.utils.cpp_extension as e; e._check_cuda_version=lambda *args, **kwargs: None; runpy.run_path('setup.py', run_name='__main__')"

echo "AWQ kernel built for TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
