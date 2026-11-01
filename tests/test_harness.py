import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from harness.config import ConfigError, load_config
from harness.runner import load_prompts, run_benchmark, server_command


ROOT = Path(__file__).parents[1]


class HarnessTests(unittest.TestCase):
    def test_baseline_loads(self):
        exp = load_config(ROOT / "configs/experiments/baseline.json")
        self.assertEqual(exp.name, "qwen3-8b-bf16-wikitext2-baseline")
        self.assertEqual(exp.raw["runtime"]["name"], "mini-sglang")
        self.assertEqual(exp.raw["server"]["tp"], 2)
        dataset = (exp.source.parent / exp.raw["dataset"]).resolve()
        self.assertEqual(len(load_prompts(dataset)), 32)
        self.assertIn("minisgl", server_command(exp))

    def test_torch_fallback_smoke_plan(self):
        exp = load_config(ROOT / "configs/experiments/qwen3-8b-wikitext2-smoke.json")
        command = server_command(exp)
        self.assertIn("--attn", command)
        self.assertEqual(command[command.index("--attn") + 1], "torch")
        self.assertIn("--num-pages", command)

    def test_awq_plan_passes_layout_group_size(self):
        exp = load_config(ROOT / "configs/experiments/awq.json")
        command = server_command(exp)
        self.assertIn("--awq-checkpoint", command)
        self.assertIn("--awq-group-size", command)
        self.assertEqual(command[command.index("--awq-group-size") + 1], "128")

    def test_missing_dataset_is_rejected(self):
        raw = json.loads((ROOT / "configs/experiments/baseline.json").read_text())
        raw["dataset"] = "missing.jsonl"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(raw))
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_smoke_dataset(self):
        self.assertEqual(len(load_prompts(ROOT / "data/smoke.jsonl")), 2)

    def test_benchmark_resolves_dataset_from_config_directory(self):
        exp = load_config(ROOT / "configs/experiments/qwen3-8b-wikitext2-smoke.json")
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                "harness.runner.request_completion",
                return_value={"usage": {"completion_tokens": 1}, "_client_latency_ms": 1.0},
            ):
                run_dir = run_benchmark(exp, "http://127.0.0.1:30000", directory)
        self.assertTrue(run_dir.name)


if __name__ == "__main__":
    unittest.main()
