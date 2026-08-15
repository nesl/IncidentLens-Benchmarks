"""Configuration loading shared by benchmark and simulator modules."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        name = value[2:-1]
        if not name:
            raise ValueError("Empty environment-variable placeholder in configuration")
        try:
            return os.environ[name]
        except KeyError as exc:
            raise RuntimeError(
                f"Required configuration environment variable is not set: {name}"
            ) from exc
    return value


def get_config(filepath: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load JSON config and resolve exact ``${ENV_NAME}`` values."""
    resolved = Path(filepath or os.environ.get("URBAN_SYSTEM_CONFIG", "./config.json"))
    with resolved.open("r", encoding="utf-8") as infile:
        return _expand_environment(json.load(infile))
