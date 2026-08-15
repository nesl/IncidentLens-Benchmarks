#!/usr/bin/env python3
"""Inspect low-level and top-level news incident grouping outputs.

Prints:
  1. Every low-level incident's final name.
  2. Every non-standalone top-level incident name and the low-level incidents under it.

A "standalone" top-level incident is one whose low_level_incident_ids list contains
exactly one low-level incident and whose top-level name is effectively the same as
that low-level incident's final name.

Usage:
    python news_check.py
    python news_check.py --low-level evaluation/out/low_level.json --top-level evaluation/out/top_level.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_LOW_LEVEL_PATH = Path("evaluation/out/low_level.json")
DEFAULT_TOP_LEVEL_PATH = Path("evaluation/out/top_level.json")


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected top-level JSON object in {path}, found {type(data).__name__}"
        )

    return data


def normalize_name(name: str | None) -> str:
    """Normalize an incident name for lightweight equality checks."""
    if not name:
        return ""

    normalized = name.strip().lower()
    for ch in ["'", '"', "`", ".", ",", ":", ";", "(", ")", "[", "]", "{", "}"]:
        normalized = normalized.replace(ch, "")

    return " ".join(normalized.split())


def get_low_level_name(low_level_record: dict[str, Any], fallback_id: str) -> str:
    """Return the display name for a low-level incident."""
    for key in ("final_name", "name", "incident_name"):
        value = low_level_record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback_id


def get_top_level_name(top_level_record: dict[str, Any], fallback_id: str) -> str:
    """Return the display name for a top-level incident."""
    for key in ("top_level_incident_name", "final_name", "name", "incident_name"):
        value = top_level_record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback_id


def get_top_level_low_ids(top_level_record: dict[str, Any]) -> list[str]:
    """Return low-level incident IDs referenced by a top-level incident."""
    value = top_level_record.get("low_level_incident_ids", [])
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def is_standalone_top_level(
    top_level_name: str,
    low_ids: list[str],
    low_level_by_id: dict[str, dict[str, Any]],
) -> bool:
    """Return True if a top-level incident is just a wrapper for one low-level incident."""
    if len(low_ids) != 1:
        return False

    low_id = low_ids[0]
    low_record = low_level_by_id.get(low_id, {})
    low_name = get_low_level_name(low_record, low_id)

    return normalize_name(top_level_name) == normalize_name(low_name)


def print_low_level_incidents(low_level: dict[str, Any]) -> None:
    """Print all low-level incident final names."""
    print("LOW-LEVEL INCIDENTS")
    print("===================")

    if not low_level:
        print("(none)\n")
        return

    for low_id in sorted(low_level):
        record = low_level[low_id]
        if not isinstance(record, dict):
            print(f"- {low_id}")
            continue

        name = get_low_level_name(record, low_id)
        incident_type = record.get("incident_type")
        location = record.get("location")
        start_time = record.get("start_datetime_pacific")
        end_time = record.get("end_datetime_pacific")

        details = []
        if incident_type:
            details.append(str(incident_type))
        if location:
            details.append(str(location))
        if start_time or end_time:
            details.append(f"{start_time or '?'} to {end_time or '?'}")

        if details:
            print(f"- {low_id}: {name} ({'; '.join(details)})")
        else:
            print(f"- {low_id}: {name}")

    print()


def print_nonstandalone_top_level_incidents(
    top_level: dict[str, Any],
    low_level: dict[str, Any],
) -> None:
    """Print non-standalone top-level incident names and their low-level children."""
    low_level_by_id = {
        low_id: record
        for low_id, record in low_level.items()
        if isinstance(record, dict)
    }

    print("NON-STANDALONE TOP-LEVEL INCIDENTS")
    print("==================================")

    printed = 0

    for top_id in sorted(top_level):
        record = top_level[top_id]
        if not isinstance(record, dict):
            continue

        top_name = get_top_level_name(record, top_id)
        low_ids = get_top_level_low_ids(record)

        if is_standalone_top_level(top_name, low_ids, low_level_by_id):
            continue

        printed += 1

        start_time = record.get("start_datetime_pacific")
        end_time = record.get("end_datetime_pacific")
        time_suffix = ""
        if start_time or end_time:
            time_suffix = f" [{start_time or '?'} to {end_time or '?'}]"

        print(f"- {top_id}: {top_name}{time_suffix}")

        if not low_ids:
            print("  - (no low-level incident IDs listed)")
            continue

        for low_id in low_ids:
            low_record = low_level_by_id.get(low_id)
            if low_record is None:
                print(f"  - {low_id}: MISSING FROM LOW-LEVEL FILE")
                continue

            low_name = get_low_level_name(low_record, low_id)
            print(f"  - {low_id}: {low_name}")

    if printed == 0:
        print("(none)")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "List low-level incident final names and non-standalone top-level "
            "incident names from grouping outputs."
        )
    )
    parser.add_argument(
        "--low-level",
        type=Path,
        default=DEFAULT_LOW_LEVEL_PATH,
        help=f"Path to low_level.json. Default: {DEFAULT_LOW_LEVEL_PATH}",
    )
    parser.add_argument(
        "--top-level",
        type=Path,
        default=DEFAULT_TOP_LEVEL_PATH,
        help=f"Path to top_level.json. Default: {DEFAULT_TOP_LEVEL_PATH}",
    )
    args = parser.parse_args()

    low_level = load_json(args.low_level)
    top_level = load_json(args.top_level)

    # print_low_level_incidents(low_level)
    print_nonstandalone_top_level_incidents(top_level, low_level)


if __name__ == "__main__":
    main()
