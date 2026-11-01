"""Experiment configuration loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


REQUIRED = ("name", "model", "dataset", "sampling", "server", "quantization", "algorithm", "artifact", "runtime")


@dataclass(frozen=True)
class Experiment:
    raw: dict[str, Any]
    source: Path

    @property
    def name(self) -> str:
        return self.raw["name"]

    @property
    def model(self) -> str:
        return self.raw["model"]


def load_config(path: str | Path) -> Experiment:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read JSON config {source}: {exc}") from exc
    validate_config(raw, source)
    return Experiment(raw=raw, source=source)


def validate_config(raw: dict[str, Any], source: Path | None = None) -> None:
    label = str(source or "config")
    missing = [key for key in REQUIRED if key not in raw]
    if missing:
        raise ConfigError(f"{label}: missing fields: {', '.join(missing)}")
    if not isinstance(raw["name"], str) or not raw["name"].strip():
        raise ConfigError(f"{label}: name must be a non-empty string")
    if not isinstance(raw["sampling"], dict):
        raise ConfigError(f"{label}: sampling must be an object")
    if not isinstance(raw["server"], dict) or not raw["server"].get("port"):
        raise ConfigError(f"{label}: server.port is required")
    if not isinstance(raw["quantization"], dict) or not raw["quantization"].get("method"):
        raise ConfigError(f"{label}: quantization.method is required")
    if not isinstance(raw["algorithm"], dict) or not raw["algorithm"].get("name"):
        raise ConfigError(f"{label}: algorithm.name is required")
    if not isinstance(raw["artifact"], dict) or not raw["artifact"].get("format"):
        raise ConfigError(f"{label}: artifact.format is required")
    if not isinstance(raw["runtime"], dict) or raw["runtime"].get("name") != "mini-sglang":
        raise ConfigError(f"{label}: runtime.name must be mini-sglang")
    dataset = Path(raw["dataset"])
    if not dataset.is_absolute():
        dataset = ((source.parent if source else Path.cwd()) / dataset).resolve()
    if not dataset.exists():
        raise ConfigError(f"{label}: dataset does not exist: {dataset}")
