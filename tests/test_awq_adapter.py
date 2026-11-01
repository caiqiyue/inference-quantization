"""No-GPU structural tests for the native llm-awq v2 adapter."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "mini-sglang" / "python"))

import torch  # noqa: E402

from minisgl.layers.linear import awq_scale_rows as runtime_awq_scale_rows  # noqa: E402
from minisgl.models.weight import awq_scale_rows, load_awq_weight  # noqa: E402


GROUP_SIZE = 128
INPUT_SIZE = 128
SCALE_ROWS = 8


def _projection(rows: int, marker: int, *, bias: bool = False) -> dict[str, torch.Tensor]:
    """Build one synthetic, structurally valid AWQ v2 W4 linear layer."""
    out_features = rows * 4
    result = {
        "qweight": torch.full((rows, INPUT_SIZE), marker, dtype=torch.int16),
        "scales": torch.full((SCALE_ROWS, out_features), float(marker), dtype=torch.float16),
        "scaled_zeros": torch.full((SCALE_ROWS, out_features), float(-marker), dtype=torch.float16),
    }
    if bias:
        result["bias"] = torch.full((out_features,), float(marker), dtype=torch.float16)
    return result


def _state_dict() -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.zeros((8, INPUT_SIZE), dtype=torch.float16),
    }
    for prefix, parts, markers, bias in (
        ("model.layers.0.self_attn", ("q_proj", "k_proj", "v_proj"), (1, 2, 3), True),
        ("model.layers.0.mlp", ("gate_proj", "up_proj"), (4, 5), False),
    ):
        for part, marker in zip(parts, markers):
            for suffix, tensor in _projection(4, marker, bias=bias).items():
                result[f"{prefix}.{part}.{suffix}"] = tensor
    return result


class AWQAdapterTests(unittest.TestCase):
    def test_scale_padding_matches_llm_awq_v2(self):
        self.assertEqual(awq_scale_rows(128, 128), 8)
        self.assertEqual(awq_scale_rows(896, 128), 8)
        self.assertEqual(awq_scale_rows(4864, 128), 40)
        self.assertEqual(awq_scale_rows(896, 128), runtime_awq_scale_rows(896, 128))

    def test_merges_awq_packed_projections_and_bias(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "awq.pt"
            torch.save(_state_dict(), checkpoint)
            actual = dict(load_awq_weight(str(checkpoint), GROUP_SIZE))

        qkv = "model.layers.0.self_attn.qkv_proj"
        self.assertEqual(actual[f"{qkv}.qweight"].shape, (12, INPUT_SIZE))
        self.assertEqual(actual[f"{qkv}.scales"].shape, (SCALE_ROWS, 48))
        self.assertTrue(torch.equal(actual[f"{qkv}.qweight"][:4], torch.full((4, INPUT_SIZE), 1, dtype=torch.int16)))
        self.assertTrue(torch.equal(actual[f"{qkv}.qweight"][4:8], torch.full((4, INPUT_SIZE), 2, dtype=torch.int16)))
        self.assertEqual(actual[f"{qkv}.bias"].shape, (48,))

        gate_up = "model.layers.0.mlp.gate_up_proj"
        self.assertEqual(actual[f"{gate_up}.qweight"].shape, (8, INPUT_SIZE))
        self.assertEqual(actual[f"{gate_up}.scales"].shape, (SCALE_ROWS, 32))
        self.assertNotIn("model.layers.0.self_attn.q_proj.qweight", actual)
        self.assertNotIn("model.layers.0.mlp.gate_proj.qweight", actual)

    def test_rejects_non_awq_scale_layout(self):
        state = _state_dict()
        state["model.layers.0.self_attn.q_proj.scales"] = torch.zeros((1, 16), dtype=torch.float16)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "invalid-awq.pt"
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, "expected"):
                dict(load_awq_weight(str(checkpoint), GROUP_SIZE))


if __name__ == "__main__":
    unittest.main()
