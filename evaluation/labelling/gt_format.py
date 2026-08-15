#!/usr/bin/env python3
"""Format low_level_final.json into a compact ground-truth JSON.

Input default:
    evaluation/out/low_level_final.json

Output default:
    evaluation/out/low_level_gt.json

Each output record contains only:
    - incident_id
    - final_name
    - incident_type
    - location
    - start_datetime_pacific
    - end_datetime_pacific
    - earliest_article_datetime_pacific

Usage:
    python gt_format.py

    python gt_format.py \
        --input evaluation/out/low_level_final.json \
        --output evaluation/out/low_level_gt.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("evaluation/out_real/low_level_final.json")
DEFAULT_OUTPUT = Path("evaluation/out_real/low_level_gt.json")


OUTPUT_FIELDS = [
    "incident_id",
    "final_name",
    "incident_type",
    "location",
    "start_datetime_pacific",
    "end_datetime_pacific",
    "earliest_article_datetime_pacific",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def get_incident_id(key: str, record: dict[str, Any]) -> str:
    """Get the incident ID from the record, falling back to the top-level key."""
    value = record.get("low_level_incident_id") or record.get("incident_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return key


def format_record(key: str, record: dict[str, Any]) -> dict[str, Any]:
    """Return the compact ground-truth representation for one low-level incident."""
    incident_id = get_incident_id(key, record)

    return {
        "incident_id": incident_id,
        "final_name": record.get("final_name"),
        "incident_type": record.get("incident_type"),
        "location": record.get("location"),
        "start_datetime_pacific": record.get("start_datetime_pacific"),
        "end_datetime_pacific": record.get("end_datetime_pacific"),
        "earliest_article_datetime_pacific": record.get("earliest_article_datetime_pacific"),
    }


def format_low_level_gt(data: Any) -> dict[str, dict[str, Any]]:
    """Format low-level final data as a dict keyed by incident_id."""
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object/dict, found {type(data).__name__}")

    output: dict[str, dict[str, Any]] = {}

    for key, record in data.items():
        if not isinstance(record, dict):
            continue

        formatted = format_record(str(key), record)
        incident_id = formatted["incident_id"]
        output[incident_id] = formatted

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create low_level_gt.json from low_level_final.json."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input low_level_final.json path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output low_level_gt.json path. Default: {DEFAULT_OUTPUT}",
    )
    args = parser.parse_args()

    data = load_json(args.input)
    gt = format_low_level_gt(data)
    save_json(args.output, gt)

    print(f"Wrote {len(gt)} records to {args.output}")


if __name__ == "__main__":
    main()
