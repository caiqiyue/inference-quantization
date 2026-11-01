#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "[L0] Checking harness files"
test -f AGENTS.md
test -f feature_list.json
test -f configs/experiments/baseline.json

echo "[L1] Compiling Python harness"
python -m compileall -q harness tests

echo "[L2] Running no-GPU tests"
python -m unittest tests.test_harness -v
PYTHONPATH="$ROOT_DIR/mini-sglang/python" \
  /home/qiyue.cai/.conda/envs/mini-sglang/bin/python -m unittest tests.test_awq_adapter -v

echo "[L3] Validating baseline experiment"
python -m harness.cli validate --config configs/experiments/baseline.json
python -m harness.cli plan --config configs/experiments/baseline.json >/dev/null

echo "Baseline harness is healthy. See docs/HARNESS.md for a real mini-sglang run."
