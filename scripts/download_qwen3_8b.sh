#!/usr/bin/env bash
# Download the public, unquantized Hugging Face checkpoint outside this repo.
set -euo pipefail

model_dir="${1:-/home/qiyue.cai/host_workspace/models/Qwen3-8B}"
mkdir -p "$model_dir"
exec /home/qiyue.cai/.conda/envs/mini-sglang/bin/hf download \
  Qwen/Qwen3-8B \
  --local-dir "$model_dir"
