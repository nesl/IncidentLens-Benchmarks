#!/usr/bin/env python3
"""Convert simulator output to the shared Urban Observations wire model.

Ground-truth labels are written only to an optional mapping sidecar. They are
never included in observations sent to enrichment or downstream consumers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
from pathlib import Path
import socket
import sys
import time
from typing import Any, Iterator

from urban_observation_model import InlineFile, Observation, SCHEMA_VERSION
from utilities.util import get_config


ROW_METADATA_FIELDS = {
    "sensor_id", "sensor_name", "sensor_type", "timestamp", "time",
    "latitude", "longitude", "lat", "lon",
    "outside_distance_km", "sensor_region_role",
}


def _first(*values: Any) -> Any:
    return next((value for value in values if value not in (None, "")), None)


def _float(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _scalar(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value
    try:
        return float(value) if any(char in value.lower() for char in (".", "e")) else int(value)
    except ValueError:
        return value


def _resolved_file(raw: dict[str, Any], folder: Path) -> Path | None:
    value = _first(
        raw.get("output_image_path"), raw.get("image_filepath"),
        raw.get("saved_input_image_path"), raw.get("input_image_path"),
    )
    if not value:
        return None
    path = Path(str(value))
    candidates = (folder / path.name, path, folder / path)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _inline_file(path: Path) -> InlineFile:
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    return InlineFile(
        name=f"synthetic-{digest[:16]}{path.suffix.lower()}",
        media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        size=len(content),
        sha256=digest,
        content=content,
    )


def _opaque_id(raw_id: Any) -> str:
    digest = hashlib.sha256(str(raw_id or "missing").encode("utf-8")).hexdigest()[:24]
    return f"synthetic:{digest}"


def to_common_observation(raw: dict[str, Any], folder: Path) -> Observation:
    """Create one portable observation without incident labels or ground truth."""
    row = raw.get("row") if isinstance(raw.get("row"), dict) else {}
    location = raw.get("sensor_location") if isinstance(raw.get("sensor_location"), dict) else {}
    latitude = _float(_first(location.get("latitude"), location.get("lat"), row.get("latitude"), row.get("lat")))
    longitude = _float(_first(location.get("longitude"), location.get("lon"), row.get("longitude"), row.get("lon")))
    source = str(raw.get("source") or raw.get("sensor_type") or "synthetic")
    sensor = str(_first(row.get("sensor_id"), raw.get("sensor_id"), raw.get("source_image_camera_description"), source))
    data = {
        key: _scalar(value)
        for key, value in row.items()
        if key not in ROW_METADATA_FIELDS and value not in (None, "")
    }
    if not data:
        for key in ("title", "subject", "body", "description", "text"):
            if raw.get(key) not in (None, ""):
                data[key] = raw[key]
    image = _resolved_file(raw, folder)
    files = (_inline_file(image),) if image is not None else ()
    original_id = str(raw.get("observation_id") or "")
    value = {
        "schema_version": SCHEMA_VERSION,
        "id": _opaque_id(original_id),
        "source": source,
        "time": str(_first(row.get("timestamp"), row.get("time"), raw.get("time"))),
        "sensor": sensor,
        "data": data,
        "end_time": None,
        "latitude": latitude,
        "longitude": longitude,
        "files": [item.to_dict() for item in files],
        "raw": {
            "format": "incidentlens-simulator.v1",
            "synthetic": True,
            "run": hashlib.sha256(folder.name.encode("utf-8")).hexdigest()[:16],
            "step": raw.get("step"),
            "modality": raw.get("modality"),
            "row_index": raw.get("row_index"),
        },
    }
    return Observation.from_dict(value)


def iter_run(folder: Path, filename: str = "observations.txt") -> Iterator[tuple[Observation, dict[str, Any]]]:
    path = folder / filename
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            observation = to_common_observation(raw, folder)
            yield observation, {
                "id": observation.id,
                "original_observation_id": raw.get("observation_id"),
                "incident_id": raw.get("incident_id"),
                "step": raw.get("step"),
                "run": folder.name,
                "line": line_number,
            }


def discover(root: Path, recursive: bool) -> list[Path]:
    if (root / "observations.txt").is_file():
        return [root]
    pattern = "**/observations.txt" if recursive else "*/observations.txt"
    return sorted({path.parent for path in root.glob(pattern)})


class ReceiverSink:
    """Stop-and-wait sender for the Urban Observations receiver."""

    def __init__(self, host: str, port: int, timeout: float):
        self.socket = socket.create_connection((host, port), timeout=timeout)
        self.reader = self.socket.makefile("r", encoding="utf-8")
        self.writer = self.socket.makefile("w", encoding="utf-8", newline="\n")

    def write(self, observation: Observation) -> None:
        self.writer.write(observation.to_json() + "\n")
        self.writer.flush()
        reply = json.loads(self.reader.readline())
        if reply.get("id") != observation.id or reply.get("accepted") is not True:
            raise RuntimeError(f"receiver rejected {observation.id}: {reply}")

    def close(self) -> None:
        self.writer.close(); self.reader.close(); self.socket.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Replay IncidentLens simulator data through shared enrichment")
    result.add_argument("root", nargs="?", type=Path, help="Override the configured simulator dataset root")
    result.add_argument("--config", type=Path, help="Configuration file (default: ./config.json)")
    recursive = result.add_mutually_exclusive_group()
    recursive.add_argument("--recursive", dest="recursive", action="store_true")
    recursive.add_argument("--no-recursive", dest="recursive", action="store_false")
    result.set_defaults(recursive=None)
    result.add_argument("--output", type=Path, help="Write common inline JSONL locally")
    result.add_argument("--mapping-output", type=Path, help="Write private ground-truth ID mapping JSONL")
    result.add_argument("--receiver-host")
    result.add_argument("--receiver-port", type=int)
    result.add_argument("--receiver-timeout", type=float)
    receiver = result.add_mutually_exclusive_group()
    receiver.add_argument("--receiver", dest="receiver", action="store_true",
                          help="Enable sending to the configured receiver")
    receiver.add_argument("--no-receiver", dest="receiver", action="store_false",
                          help="Disable sending to the configured receiver")
    result.set_defaults(receiver=None)
    result.add_argument("--interval-seconds", type=float)
    result.add_argument("--limit", type=int)
    return result


def replay_settings(args: argparse.Namespace) -> argparse.Namespace:
    """Apply config defaults while preserving explicit command-line overrides."""
    config = get_config(args.config)
    replay = config.get("replay", {})
    receiver = replay.get("receiver", {})
    host_was_overridden = args.receiver_host is not None

    root = args.root or replay.get("dataset_root")
    if not root:
        raise SystemExit("set replay.dataset_root in config.json or provide ROOT")
    args.root = Path(root)
    args.recursive = bool(replay.get("recursive", True)) if args.recursive is None else args.recursive
    args.output = args.output or (Path(replay["output"]) if replay.get("output") else None)
    args.mapping_output = args.mapping_output or (
        Path(replay["mapping_output"]) if replay.get("mapping_output") else None
    )
    args.receiver_host = args.receiver_host or receiver.get("host")
    args.receiver_port = args.receiver_port or int(receiver.get("port", 8766))
    args.receiver_timeout = args.receiver_timeout or float(receiver.get("timeout_seconds", 180.0))
    args.interval_seconds = (
        float(replay.get("interval_seconds", 0.0))
        if args.interval_seconds is None else args.interval_seconds
    )
    if args.receiver is None:
        args.receiver = host_was_overridden or bool(receiver.get("enabled", True))
    if not args.receiver:
        args.receiver_host = None
    return args


def main(argv=None) -> int:
    args = replay_settings(parser().parse_args(argv))
    folders = discover(args.root, args.recursive)
    if not folders:
        raise SystemExit(f"no observations.txt found under {args.root}")
    if not args.output and not args.receiver_host:
        raise SystemExit("provide --output and/or --receiver-host")
    output = mapping = sink = None
    count = 0
    try:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output = args.output.open("w", encoding="utf-8")
        if args.mapping_output:
            args.mapping_output.parent.mkdir(parents=True, exist_ok=True)
            mapping = args.mapping_output.open("w", encoding="utf-8")
        if args.receiver_host:
            sink = ReceiverSink(args.receiver_host, args.receiver_port, args.receiver_timeout)
        for folder in folders:
            for observation, private_mapping in iter_run(folder):
                if output:
                    output.write(observation.to_json() + "\n"); output.flush()
                if mapping:
                    mapping.write(json.dumps(private_mapping, separators=(",", ":")) + "\n"); mapping.flush()
                if sink:
                    sink.write(observation)
                count += 1
                if args.interval_seconds:
                    time.sleep(args.interval_seconds)
                if args.limit is not None and count >= args.limit:
                    print(f"emitted {count} observations", file=sys.stderr)
                    return 0
        print(f"emitted {count} observations from {len(folders)} run folders", file=sys.stderr)
        return 0
    finally:
        if sink: sink.close()
        if mapping: mapping.close()
        if output: output.close()


if __name__ == "__main__":
    raise SystemExit(main())
