"""Small environment smoke test; run with the target environment's Python."""

from __future__ import annotations

import importlib.util
import sys

import torch


def main() -> int:
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return 2
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(f"gpu[{index}]: {props.name}, capability={props.major}.{props.minor}, memory_gb={props.total_memory / 2**30:.1f}")
    x = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    y = x @ x
    torch.cuda.synchronize()
    print(f"cuda_matmul: OK ({y.shape}, {y.dtype})")
    print(f"awq_kernel_importable: {importlib.util.find_spec('awq_inference_engine') is not None}")
    print(f"sglang_importable: {importlib.util.find_spec('sglang') is not None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
