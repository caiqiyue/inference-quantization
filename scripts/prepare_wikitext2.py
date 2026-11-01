#!/usr/bin/env python3
"""Create deterministic JSONL chat prompts from WikiText-2 raw text.

The source corpus remains external to this repository.  The generated JSONL is
a small, versioned benchmark fixture for the Harness OpenAI-compatible client.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def is_article_heading(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("=") and stripped.endswith("=")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/qiyue.cai/host_workspace/calibration-data/wikitext-2-raw/wiki.test.raw",
        help="WikiText-2 raw test split",
    )
    parser.add_argument("--output", default="data/wikitext2-test-32.jsonl")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--max-chars", type=int, default=768)
    args = parser.parse_args()

    if args.limit < 1 or args.max_chars < 1:
        parser.error("--limit and --max-chars must be positive")

    source = Path(args.input)
    output = Path(args.output)
    entries: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        text = " ".join(raw_line.split())
        if not text or is_article_heading(raw_line) or len(text) < 80:
            continue
        entries.append(
            {
                "id": f"wikitext2-test-{len(entries):03d}",
                "source_line": line_number,
                "prompt": text[: args.max_chars],
            }
        )
        if len(entries) == args.limit:
            break
    if len(entries) != args.limit:
        raise RuntimeError(f"only found {len(entries)} usable lines in {source}; expected {args.limit}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries), encoding="utf-8")
    print(f"wrote {len(entries)} prompts to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
