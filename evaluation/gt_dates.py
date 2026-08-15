#!/usr/bin/env python3
"""
Extract all incident dates from low_level_gt_corrected.json, including a
1-day buffer before each incident, but NOT after the event end date.

By default, this reads:
    low_level_gt_corrected.json

It prints:
    1) the sorted list of YYYYMMDD dates
    2) the number of unique dates

Example:
    python gt_dates_no_end_buffer.py
    python gt_dates_no_end_buffer.py --input low_level_gt_corrected.json --output dates.json
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable


def parse_date(value: str | None) -> date | None:
    """Parse an ISO-like datetime/date string and return only the date."""
    if not value:
        return None

    # Handles strings like:
    #   2025-01-07T00:00:00
    #   2025-01-07T00:00:00Z
    #   2025-01-07
    value = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        # Fallback for date-only strings or strings with extra content.
        return datetime.strptime(value[:10], "%Y-%m-%d").date()


def daterange(start: date, end: date) -> Iterable[date]:
    """Yield all dates from start through end, inclusive."""
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def iter_incident_entries(data: Any) -> Iterable[dict[str, Any]]:
    """
    Support either:
      - dict keyed by incident id: {"final_low_000001": {...}, ...}
      - list of incident objects: [{...}, {...}]
    """
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, dict):
                yield value
    elif isinstance(data, list):
        for value in data:
            if isinstance(value, dict):
                yield value
    else:
        raise TypeError("Expected top-level JSON object or list.")


def collect_dates(input_path: str | Path, buffer_days_before: int = 1) -> list[str]:
    input_path = Path(input_path)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    all_dates: set[date] = set()

    for entry in iter_incident_entries(data):
        start = parse_date(entry.get("start_datetime_pacific"))
        end = parse_date(entry.get("end_datetime_pacific"))

        # If end is missing, treat the incident as occurring only on start date.
        # If start is missing but earliest article exists, use earliest article date.
        if start is None:
            start = parse_date(entry.get("earliest_article_datetime_pacific"))
        if start is None:
            continue
        if end is None:
            end = start

        if end < start:
            start, end = end, start

        padded_start = start - timedelta(days=buffer_days_before)
        padded_end = end  # Do NOT add +1 day after the event end.

        for d in daterange(padded_start, padded_end):
            all_dates.add(d)

    return [d.strftime("%Y%m%d") for d in sorted(all_dates)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="evaluation/ground_truth/real/low_level_gt_corrected.json",
        help="Path to low_level_gt_corrected.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the date list as JSON.",
    )
    parser.add_argument(
        "--buffer-days-before",
        type=int,
        default=1,
        help="Number of days to include before each incident.",
    )
    args = parser.parse_args()

    dates = collect_dates(args.input, buffer_days_before=args.buffer_days_before)

    print(json.dumps(dates, indent=2))
    print(f"length: {len(dates)}")

    if args.output:
        output_path = Path(args.output)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "dates": dates,
                    "length": len(dates),
                },
                f,
                indent=2,
            )
            f.write("\n")


if __name__ == "__main__":
    main()
