"""Command line entrypoint: validate, plan, and run."""

from __future__ import annotations

import argparse
import json
import sys

from .config import ConfigError, load_config
from .runner import run_benchmark, server_command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "plan"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--server-url", default="http://127.0.0.1:30000")
    run.add_argument("--output-root", default="runs")
    args = parser.parse_args(argv)
    try:
        exp = load_config(args.config)
        if args.command == "validate":
            print(f"OK: {exp.source} ({exp.name})")
        elif args.command == "plan":
            print(json.dumps({"experiment": exp.name, "server_command": server_command(exp), "dataset": exp.raw["dataset"], "quantization": exp.raw["quantization"]}, indent=2, ensure_ascii=False))
        else:
            output = run_benchmark(exp, args.server_url, args.output_root)
            print(f"completed: {output}")
        return 0
    except (ConfigError, OSError, ValueError, TimeoutError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
