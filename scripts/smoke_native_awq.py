#!/usr/bin/env python3
"""Run a native llm-awq W4A16 checkpoint without mini-sglang."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from awq.quantize.quantizer import real_quantize_model_weight


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--prompt", default="Explain why the sky appears blue in one sentence.")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Native AWQ smoke requires a CUDA GPU")
    if not args.model.is_dir() or not args.checkpoint.is_file():
        raise FileNotFoundError("model directory or AWQ checkpoint is missing")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    real_quantize_model_weight(
        model,
        w_bit=4,
        q_config={"zero_point": True, "q_group_size": args.group_size},
        init_only=True,
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint load result: {incompatible}")

    model = model.cuda().eval()
    encoded = tokenizer(args.prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    output = tokenizer.decode(generated[0], skip_special_tokens=True)

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-native-awq-" + uuid.uuid4().hex[:8]
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": run_id,
        "kind": "native-llm-awq-full-model-smoke",
        "model": str(args.model),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "group_size": args.group_size,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    result = {
        "prompt": args.prompt,
        "generated_text": output,
        "generated_token_count": int(generated.shape[-1] - encoded.input_ids.shape[-1]),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({"completed": True}, indent=2), encoding="utf-8")
    (run_dir / "responses.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
