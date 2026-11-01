#!/usr/bin/env python3
"""Compare native llm-awq projections with mini-sglang's AWQ adapter.

This is a GPU validation tool, not a benchmark.  It deliberately compares the
same packed AWQ tensors through the native WQLinear modules and through the
mini-sglang fused projections.  A passing result isolates the quantization
artifact from the runtime adapter before an API smoke or quality evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mini-sglang" / "python"))

from awq.quantize.qmodule import WQLinear  # noqa: E402
from minisgl.distributed import set_tp_info  # noqa: E402
from minisgl.layers.linear import _LinearTPImpl  # noqa: E402
from minisgl.models.weight import load_awq_weight  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _native_linear(state: dict[str, torch.Tensor], prefix: str, group_size: int) -> WQLinear:
    qweight = state[f"{prefix}.qweight"]
    out_features = qweight.shape[0] * 4
    module = WQLinear(
        4,
        group_size,
        qweight.shape[1],
        out_features,
        bias=f"{prefix}.bias" in state,
        dev="cuda",
        dtype=state[f"{prefix}.scales"].dtype,
    )
    module.qweight = qweight.cuda()
    module.scales = state[f"{prefix}.scales"].cuda()
    module.scaled_zeros = state[f"{prefix}.scaled_zeros"].cuda()
    if module.bias is not None:
        module.bias = state[f"{prefix}.bias"].cuda()
    return module


def _mini_fused_linear(
    adapted: dict[str, torch.Tensor],
    prefix: str,
    group_size: int,
) -> _LinearTPImpl:
    qweight = adapted[f"{prefix}.qweight"]
    out_features = qweight.shape[0] * 4
    module = _LinearTPImpl(
        qweight.shape[1],
        out_features,
        qweight.shape[1],
        out_features,
        has_bias=f"{prefix}.bias" in adapted,
        awq=True,
        awq_group_size=group_size,
        awq_dtype=adapted[f"{prefix}.scales"].dtype,
    )
    module.qweight = qweight.cuda()
    module.scales = adapted[f"{prefix}.scales"].cuda()
    module.scaled_zeros = adapted[f"{prefix}.scaled_zeros"].cuda()
    if module.bias is not None:
        module.bias = adapted[f"{prefix}.bias"].cuda()
    return module


def _compare(
    state: dict[str, torch.Tensor],
    adapted: dict[str, torch.Tensor],
    native_prefixes: tuple[str, ...],
    fused_prefix: str,
    group_size: int,
    rows: int,
) -> dict[str, object]:
    native = [_native_linear(state, prefix, group_size) for prefix in native_prefixes]
    fused = _mini_fused_linear(adapted, fused_prefix, group_size)
    dtype = native[0].scales.dtype
    torch.manual_seed(42 + rows)
    x = torch.randn((rows, native[0].in_features), device="cuda", dtype=dtype)
    with torch.inference_mode():
        expected = torch.cat([module(x) for module in native], dim=-1)
        actual = fused.forward(x)
    difference = (expected.float() - actual.float()).abs()
    return {
        "rows": rows,
        "input_shape": list(x.shape),
        "output_shape": list(actual.shape),
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
        "allclose": bool(torch.allclose(expected, actual, rtol=2e-2, atol=2e-2)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("AWQ adapter validation requires a CUDA GPU")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    set_tp_info(0, 1)
    native_state = dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True))
    adapted_state = dict(load_awq_weight(str(args.checkpoint), args.group_size))

    comparisons: dict[str, list[dict[str, object]]] = {}
    cases = (
        (
            "qkv",
            (
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.k_proj",
                "model.layers.0.self_attn.v_proj",
            ),
            "model.layers.0.self_attn.qkv_proj",
        ),
        (
            "gate_up",
            ("model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"),
            "model.layers.0.mlp.gate_up_proj",
        ),
    )
    for name, native_prefixes, fused_prefix in cases:
        comparisons[name] = [
            _compare(native_state, adapted_state, native_prefixes, fused_prefix, args.group_size, rows)
            for rows in (1, 8)
        ]

    passed = all(item["allclose"] for results in comparisons.values() for item in results)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-awq-adapter-" + uuid.uuid4().hex[:8]
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": run_id,
        "kind": "awq-native-to-minisgl-layer-validation",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "group_size": args.group_size,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    metrics = {"passed": passed, "comparisons": comparisons}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "layer-results.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "passed": passed}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
