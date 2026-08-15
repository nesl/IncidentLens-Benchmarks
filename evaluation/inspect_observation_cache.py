#!/usr/bin/env python3
"""Inspect cached observation-model outputs for anomaly/observation behavior.

Example:
    python inspect_observation_cache.py \
        detection/cache/observation_model/real_data/cctv/20250107

This recursively scans JSON/JSONL files under a cache directory and reports:
  * anomaly preprocessing decisions
  * anomaly scores
  * whether the heavy observation model appears to have run
  * whether the final observation output contains effects/incidents
  * example anomalous observations and skipped reports

It is intentionally tolerant of several cache shapes, e.g. either the cached
observation object itself or wrappers like {"output": ...}, {"observation": ...},
{"observation_output": ...}, or {"value": ...}.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

JSON_SUFFIXES = {".json", ".jsonl", ".ndjson"}


def iter_json_objects(path: Path) -> Iterator[Tuple[Path, Optional[int], Dict[str, Any]]]:
    """Yield JSON objects from a file.

    Supports normal JSON files and JSONL/NDJSON.  Invalid files are skipped with
    a warning printed by the caller if desired.
    """
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield path, line_no, obj
        return

    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            obj = json.load(f)
    except Exception:
        return

    if isinstance(obj, dict):
        yield path, None, obj
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            if isinstance(item, dict):
                yield path, idx, item


def unwrap_cache_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Return the likely observation output from a cache wrapper."""
    # Direct observation-model output.
    if "observed_effects" in obj or "possible_incidents" in obj or "model" in obj:
        return obj

    # Common wrappers used by archives/caches.
    for key in (
        "output",
        "observation_output",
        "observation",
        "value",
        "cached_output",
        "result",
    ):
        value = obj.get(key)
        if isinstance(value, dict):
            return unwrap_cache_object(value)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return unwrap_cache_object(parsed)

    return obj


def get_nested(obj: Dict[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def labels_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [x for x in value if isinstance(x, dict)]


def label_names(labels: List[Dict[str, Any]]) -> List[str]:
    return [str(x.get("name")) for x in labels if x.get("name")]


def float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def summarize_one(output: Dict[str, Any], source_file: Path, line_no: Optional[int]) -> Dict[str, Any]:
    model = output.get("model") if isinstance(output.get("model"), dict) else {}
    anomaly_pre = model.get("anomaly_preprocessing") if isinstance(model.get("anomaly_preprocessing"), dict) else {}

    effects = labels_list(output.get("observed_effects"))
    incidents = labels_list(output.get("possible_incidents"))

    # Try several possible locations used across versions.
    decision = (
        anomaly_pre.get("decision")
        or get_nested(output, "anomaly_only", "decision")
        or get_nested(output, "anomaly_only", "anomaly_preprocessing", "decision")
        or "<missing>"
    )
    anomaly_score = float_or_none(
        anomaly_pre.get("anomaly_score")
        if "anomaly_score" in anomaly_pre
        else anomaly_pre.get("score")
    )
    if anomaly_score is None:
        anomaly_score = float_or_none(get_nested(output, "anomaly_only", "anomaly_score"))
    if anomaly_score is None:
        anomaly_score = float_or_none(get_nested(output, "anomaly_only", "anomaly", "score"))

    candidate = bool(
        anomaly_pre.get("candidate_for_full_observation")
        or get_nested(output, "anomaly_only", "candidate_for_full_observation")
        or effects
        or incidents
        or (anomaly_score is not None and anomaly_score > 0.0)
    )

    skipped_heavy = str(decision) == "skipped_heavy_observation_model"
    heavy_used_flag = (
        anomaly_pre.get("heavy_observation_model_used")
        or model.get("heavy_observation_model_used")
        or model.get("llm_used")
    )

    # If the decision says skipped, trust it. Otherwise, if anomaly preprocessing
    # exists and did not skip, it likely admitted the report to the observation model.
    observation_model_ran = bool(heavy_used_flag) or (
        bool(anomaly_pre) and not skipped_heavy and str(decision) != "<missing>"
    )

    # Non-empty output means the final output has some effect or incident label.
    observation_nonempty = bool(effects or incidents)

    return {
        "cache_file": str(source_file),
        "line_or_index": line_no,
        "report_id": output.get("report_id"),
        "report_date": output.get("report_date"),
        "sensor_id": output.get("sensor_id"),
        "sensor_type": output.get("sensor_type"),
        "decision": str(decision),
        "anomaly_score": anomaly_score if anomaly_score is not None else "",
        "anomalous_or_candidate": candidate,
        "observation_model_ran": observation_model_ran,
        "observation_nonempty": observation_nonempty,
        "observed_effects": ";".join(label_names(effects)),
        "possible_incidents": ";".join(label_names(incidents)),
        "model_name": get_nested(output, "model", "name", default=""),
        "model_backend": get_nested(output, "model", "backend", default=""),
        "cache_status": get_nested(output, "model", "cache", "status", default=""),
    }


def scan_cache(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in JSON_SUFFIXES)
    for path in files:
        for source_file, line_no, raw_obj in iter_json_objects(path):
            output = unwrap_cache_object(raw_obj)
            rows.append(summarize_one(output, source_file, line_no))
    return rows


def print_summary(rows: List[Dict[str, Any]], *, max_examples: int) -> None:
    total = len(rows)
    decisions = Counter(row["decision"] for row in rows)
    by_source = Counter(row.get("sensor_type") or "<missing>" for row in rows)
    anomalous = [r for r in rows if r["anomalous_or_candidate"]]
    obs_ran = [r for r in rows if r["observation_model_ran"]]
    obs_nonempty = [r for r in rows if r["observation_nonempty"]]
    both = [r for r in rows if r["anomalous_or_candidate"] and r["observation_model_ran"]]
    both_nonempty = [r for r in rows if r["anomalous_or_candidate"] and r["observation_nonempty"]]

    print("\nObservation cache inspection summary")
    print("=" * 44)
    print(f"Cached objects scanned:                 {total}")
    print(f"Anomalous/candidate by anomaly fields:  {len(anomalous)}")
    print(f"Observation model appears to have run:  {len(obs_ran)}")
    print(f"Final observation output non-empty:     {len(obs_nonempty)}")
    print(f"Anomalous AND obs model ran:            {len(both)}")
    print(f"Anomalous AND non-empty observation:    {len(both_nonempty)}")

    print("\nBy sensor_type:")
    for source, count in by_source.most_common():
        print(f"  {source:32} {count:8d}")

    print("\nAnomaly preprocessing decisions:")
    for decision, count in decisions.most_common():
        print(f"  {decision:48} {count:8d}")

    def show_examples(title: str, examples: List[Dict[str, Any]]) -> None:
        print(f"\n{title}")
        if not examples:
            print("  <none>")
            return
        for row in examples[:max_examples]:
            print(
                "  "
                f"{row.get('report_date')} | {row.get('sensor_type')} | {row.get('sensor_id')} | "
                f"score={row.get('anomaly_score')} | decision={row.get('decision')} | "
                f"effects=[{row.get('observed_effects')}] incidents=[{row.get('possible_incidents')}] | "
                f"file={row.get('cache_file')}"
            )

    # Sort examples by score descending when possible.
    def score_key(row: Dict[str, Any]) -> float:
        value = row.get("anomaly_score")
        return value if isinstance(value, float) else -1.0

    show_examples("Examples: anomalous AND observation model ran", sorted(both, key=score_key, reverse=True))
    show_examples("Examples: anomalous AND non-empty final output", sorted(both_nonempty, key=score_key, reverse=True))


def write_csv(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cache_file",
        "line_or_index",
        "report_id",
        "report_date",
        "sensor_id",
        "sensor_type",
        "decision",
        "anomaly_score",
        "anomalous_or_candidate",
        "observation_model_ran",
        "observation_nonempty",
        "observed_effects",
        "possible_incidents",
        "model_name",
        "model_backend",
        "cache_status",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect cached observation-model outputs.")
    parser.add_argument(
        "cache_dir",
        type=Path,
        help="Directory such as detection/cache/observation_model/real_data/cctv/20250107",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write per-cache-object CSV details.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Maximum examples to print per category.",
    )
    args = parser.parse_args()

    if not args.cache_dir.exists():
        raise SystemExit(f"Cache directory does not exist: {args.cache_dir}")
    if not args.cache_dir.is_dir():
        raise SystemExit(f"Expected a directory: {args.cache_dir}")

    rows = scan_cache(args.cache_dir)
    print_summary(rows, max_examples=args.max_examples)

    if args.csv is not None:
        write_csv(rows, args.csv)
        print(f"\nWrote CSV details to: {args.csv}")


if __name__ == "__main__":
    main()
