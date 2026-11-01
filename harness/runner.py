"""mini-sglang command planning and OpenAI-compatible smoke benchmarking."""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from .config import Experiment


def server_command(exp: Experiment) -> list[str]:
    server = exp.raw["server"]
    runtime = exp.raw["runtime"]["name"]
    if runtime != "mini-sglang":
        raise ValueError(f"unsupported runtime: {runtime}")
    command = ["python", "-m", "minisgl", "--model", exp.model]
    mini_flags = {
        "host": "--host",
        "port": "--port",
        "tp": "--tp-size",
        "dtype": "--dtype",
        "mem_fraction": "--memory-ratio",
    }
    for key, flag in mini_flags.items():
        if key in server:
            command.extend([flag, str(server[key])])
    context_length = server.get("extra_args", {}).get("context_length")
    if context_length is not None:
        command.extend(["--max-seq-len-override", str(context_length)])
    mini_extra_flags = {
        "attention_backend": "--attn",
        "cuda_graph_max_bs": "--cuda-graph-max-bs",
        "num_pages": "--num-pages",
        "awq_checkpoint": "--awq-checkpoint",
        "awq_group_size": "--awq-group-size",
    }
    for key, flag in mini_extra_flags.items():
        value = server.get("extra_args", {}).get(key)
        if value is not None:
            command.extend([flag, str(value)])
    return command


def load_prompts(path: str | Path) -> list[str]:
    prompts = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        prompt = item.get("prompt") if isinstance(item, dict) else item
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"dataset line {line_no} has no non-empty prompt")
        prompts.append(prompt)
    if not prompts:
        raise ValueError("dataset contains no prompts")
    return prompts


def request_completion(base_url: str, model: str, prompt: str, sampling: dict[str, Any]) -> dict[str, Any]:
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], **sampling}
    request = Request(base_url.rstrip("/") + "/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urlopen(request, timeout=300) as response:
        body = json.loads(response.read().decode())
    body["_client_latency_ms"] = (time.perf_counter() - started) * 1000
    return body


def run_benchmark(exp: Experiment, server_url: str, output_root: str | Path = "runs") -> Path:
    dataset = Path(exp.raw["dataset"])
    if not dataset.is_absolute():
        dataset = (exp.source.parent / dataset).resolve()
    prompts = load_prompts(dataset)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for prompt in prompts:
        results.append(request_completion(server_url, exp.model, prompt, exp.raw["sampling"]))
    latencies = [float(item["_client_latency_ms"]) for item in results]
    completion_tokens = sum(item.get("usage", {}).get("completion_tokens", 0) for item in results)
    total_seconds = sum(latencies) / 1000
    manifest = {"run_id": run_id, "experiment": exp.raw, "config_source": str(exp.source), "server_url": server_url, "prompt_count": len(prompts)}
    metrics = {"request_count": len(results), "mean_latency_ms": sum(latencies) / len(latencies), "p50_latency_ms": sorted(latencies)[len(latencies) // 2], "completion_tokens": completion_tokens, "aggregate_tokens_per_second": completion_tokens / total_seconds if total_seconds else None}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "responses.jsonl").write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in results) + "\n", encoding="utf-8")
    return run_dir


def start_server(exp: Experiment) -> subprocess.Popen[bytes]:
    env = None
    if exp.raw["runtime"]["name"] == "mini-sglang" and exp.raw["server"].get("extra_args", {}).get("attention_backend") == "torch":
        env = os.environ.copy()
        env["MINISGL_TORCH_FALLBACK"] = "1"
    return subprocess.Popen(server_command(exp), env=env)
