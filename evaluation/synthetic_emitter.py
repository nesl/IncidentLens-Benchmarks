#!/usr/bin/env python3
"""Emit normalized simulated multimodal observations as one per-sensor REPORT.

Expected folder layout
----------------------
All files for one simulator run live in the same folder and at the same depth:

    my_run_folder/
      observations.txt
      Wildfire_cc042cce_incident_0_1_weather.csv
      Wildfire_cc042cce_incident_0_step1_image0_4f684ae0_output.png
      ...

The script reads observations.txt as newline-delimited JSON. Each input line is
converted into exactly one normalized REPORT describing a single sensor/source.

Socket mode supports backpressure: after sending each REPORT, the emitter can
wait for an ACK from full_pipeline.py. If WAIT_FOR_SOCKET_ACK=True, the emitter
will not advance to the next report until the pipeline finishes the observation
model and hypothesis proposal for the current report.

For easier development, edit the settings directly inside main().
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

from detection.report import normalize_report

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None  # type: ignore


def log(message: str) -> None:
    """Write status messages to stderr so stdout can remain JSONL if desired."""
    print(message, file=sys.stderr, flush=True)


# Fields that describe the row/sensor/report itself rather than the payload.
# These are removed from report["data"] for non-image sources.
ROW_METADATA_FIELDS = {
    "sensor_id",
    "timestamp",
    "time",
    "latitude",
    "longitude",
    "lat",
    "lon",
}

_NUMERIC_RE = re.compile(r"^-?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?$")
_INT_RE = re.compile(r"^-?\d+$")


def coerce_scalar(value: Any) -> Any:
    """Convert numeric-looking strings to int/float while preserving text."""
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if stripped == "":
        return value

    if _NUMERIC_RE.match(stripped):
        try:
            if _INT_RE.match(stripped):
                return int(stripped)
            return float(stripped)
        except ValueError:
            return value

    return value


def coerce_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce scalar values in a flat dictionary."""
    return {key: coerce_scalar(value) for key, value in payload.items()}


def first_present(*values: Any) -> Any:
    """Return the first value that is not None and not an empty string."""
    for value in values:
        if value is not None and value != "":
            return value
    return None


def as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_same_depth_path(
    path_value: Any,
    input_folder: Path,
    *,
    assume_same_depth: bool = True,
) -> Optional[str]:
    """Resolve simulator file references relative to the run folder.

    If assume_same_depth=True, any path in observations.txt is rewritten to
    INPUT_FOLDER / basename(path), matching a flat run-folder layout.
    """
    if path_value is None or path_value == "":
        return None

    raw_path = Path(str(path_value))
    input_folder = Path(input_folder)

    if assume_same_depth:
        return str(input_folder / raw_path.name)

    if raw_path.is_absolute() and raw_path.exists():
        return str(raw_path)

    candidates = [
        input_folder / raw_path.name,
        input_folder / raw_path,
        raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # Stable fallback even if the file does not exist yet.
    return str(input_folder / raw_path.name)


def extract_lat_lon(obs: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Prefer explicit sensor_location, then row latitude/longitude."""
    sensor_location = obs.get("sensor_location") or {}
    if not isinstance(sensor_location, dict):
        sensor_location = {}

    lat = first_present(
        sensor_location.get("latitude"),
        sensor_location.get("lat"),
        row.get("latitude"),
        row.get("lat"),
    )
    lon = first_present(
        sensor_location.get("longitude"),
        sensor_location.get("lon"),
        row.get("longitude"),
        row.get("lon"),
    )
    return {"latitude": as_float(lat), "longitude": as_float(lon)}


def extract_report_date(obs: Dict[str, Any], row: Dict[str, Any]) -> Optional[str]:
    """Report date/time, preferring row-level timestamps when present."""
    return first_present(row.get("timestamp"), row.get("time"), obs.get("time"))


def extract_sensor_id(obs: Dict[str, Any], row: Dict[str, Any]) -> str:
    """Find a stable-ish id/name for the source that produced this one report."""
    source = obs.get("source")
    source_slug = str(source or "sensor").replace(" ", "_")

    explicit_id = first_present(
        row.get("sensor_id"),
        row.get("station_id"),
        row.get("device_id"),
        row.get("post_id"),
        row.get("username"),
        obs.get("sensor_id"),
        obs.get("source_image_camera_description"),
    )
    if explicit_id is not None:
        return str(explicit_id)

    obs_id = obs.get("observation_id") or "unknown_observation"
    return f"{source_slug}:{obs_id}"


def extract_sensor_name(obs: Dict[str, Any], row: Dict[str, Any], sensor_id: str) -> str:
    return str(
        first_present(
            obs.get("source_image_camera_description"),
            row.get("username"),
            row.get("headline"),
            row.get("sensor_name"),
            sensor_id,
        )
    )


def extract_data_fields(
    obs: Dict[str, Any],
    row: Dict[str, Any],
    input_folder: Path,
    *,
    assume_same_depth: bool = True,
) -> Dict[str, Any]:
    """Extract only the sensor payload for this report.

    For images, the payload is intentionally just the image filepath.
    For other modalities, row metadata like lat/lon/timestamp/sensor_id is removed.
    """
    modality = obs.get("modality")
    source = obs.get("source")

    if modality == "image" or source in {"cctv", "image", "camera"}:
        image_path = first_present(
            obs.get("output_image_path"),
            obs.get("image_filepath"),
            obs.get("saved_input_image_path"),
            obs.get("input_image_path"),
        )
        return {
            "image_filepath": resolve_same_depth_path(
                image_path,
                input_folder,
                assume_same_depth=assume_same_depth,
            )
        }

    if row:
        return coerce_payload(
            {key: value for key, value in row.items() if key not in ROW_METADATA_FIELDS}
        )

    # Generic fallback for future non-row observation formats.
    excluded = {
        "observation_id",
        "incident_id",
        "source",
        "modality",
        "sensor_location",
        "time",
        "step",
        "row_index",
    }
    return coerce_payload({key: value for key, value in obs.items() if key not in excluded})


def observation_to_report(
    obs: Dict[str, Any],
    input_folder: Path,
    *,
    observations_filename: str = "observations.txt",
    raw_observation_index: Optional[int] = None,
    assume_same_depth: bool = True,
) -> Dict[str, Any]:
    """Convert one raw observation object into one single-sensor REPORT."""
    row = obs.get("row") or {}
    if not isinstance(row, dict):
        row = {}

    sensor_id = extract_sensor_id(obs, row)
    location = extract_lat_lon(obs, row)

    data_file = resolve_same_depth_path(
        obs.get("data_file"),
        input_folder,
        assume_same_depth=assume_same_depth,
    )

    observations_path = input_folder / observations_filename
    try:
        input_folder_resolved = str(input_folder.resolve())
    except Exception:
        input_folder_resolved = str(input_folder)
    try:
        observations_file_resolved = str(observations_path.resolve())
    except Exception:
        observations_file_resolved = str(observations_path)

    report = {
        "report_id": obs.get("observation_id"),
        "report_date": extract_report_date(obs, row),
        "sensor_id": sensor_id,
        "sensor_name": extract_sensor_name(obs, row, sensor_id),
        "sensor_type": obs.get("source"),
        "location": location,
        "data": extract_data_fields(
            obs,
            row,
            input_folder,
            assume_same_depth=assume_same_depth,
        ),
        # Keep useful provenance without mixing it into the sensor payload.
        "metadata": {
            "incident_id": obs.get("incident_id"),
            "simulator_run_folder": str(input_folder),
            "simulator_run_folder_resolved": input_folder_resolved,
            "observations_filename": observations_filename,
            "observations_file": str(observations_path),
            "observations_file_resolved": observations_file_resolved,
            "raw_observation_index": raw_observation_index,
            "modality": obs.get("modality"),
            "step": obs.get("step"),
            "row_index": obs.get("row_index"),
            "data_file": data_file,
            "direction": obs.get("direction"),
            "direction_bearing_degrees": obs.get("direction_bearing_degrees"),
            "direction_compass": obs.get("direction_compass"),
            "direct_observation_target": obs.get("direct_observation_target"),
        },
    }

    # Drop null metadata values to keep reports compact.
    report["metadata"] = {
        key: value for key, value in report["metadata"].items() if value is not None
    }
    return normalize_report(report)


def iter_observations(observations_path: str | Path) -> Iterator[Dict[str, Any]]:
    """Yield raw JSON observations from a JSONL observations.txt file."""
    observations_path = Path(observations_path)
    with observations_path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {observations_path} on line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_number} in {observations_path} is not a JSON object")
            yield value


def iter_reports(
    input_folder: str | Path,
    *,
    observations_filename: str = "observations.txt",
    assume_same_depth: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Yield normalized single-sensor reports, one per input observation."""
    input_folder = Path(input_folder)
    observations_path = input_folder / observations_filename

    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
    if not observations_path.exists():
        raise FileNotFoundError(f"Could not find observations file: {observations_path}")

    for raw_observation_index, obs in enumerate(iter_observations(observations_path), start=1):
        yield observation_to_report(
            obs,
            input_folder,
            observations_filename=observations_filename,
            raw_observation_index=raw_observation_index,
            assume_same_depth=assume_same_depth,
        )



def import_ground_truth_display_helpers():
    """Import GeoManager + visualization helpers from the project layout.

    The emitter is sometimes run as ``python detection/synthetic_emitter.py`` and
    sometimes as a module, so these imports intentionally try both package and
    local-module paths.
    """
    import_errors = []

    try:
        from detection.geo_manager import GeoManager  # type: ignore
    except Exception as exc:
        import_errors.append(f"detection.geo_manager: {exc!r}")
        try:
            from geo_manager import GeoManager  # type: ignore
        except Exception as fallback_exc:
            import_errors.append(f"geo_manager: {fallback_exc!r}")
            raise ImportError(
                "Could not import GeoManager. Tried detection.geo_manager and geo_manager. "
                + "; ".join(import_errors)
            ) from fallback_exc

    try:
        from detection.visualization.geo_tools import highlight_geo, hide_layer  # type: ignore
    except Exception as exc:
        import_errors.append(f"detection.visualization.geo_tools: {exc!r}")
        try:
            from visualization.geo_tools import highlight_geo, hide_layer  # type: ignore
        except Exception as fallback_exc:
            import_errors.append(f"visualization.geo_tools: {fallback_exc!r}")
            try:
                from geo_tools import highlight_geo, hide_layer  # type: ignore
            except Exception as second_fallback_exc:
                import_errors.append(f"geo_tools: {second_fallback_exc!r}")
                raise ImportError(
                    "Could not import visualization helpers. Tried "
                    "detection.visualization.geo_tools, visualization.geo_tools, and geo_tools. "
                    + "; ".join(import_errors)
                ) from second_fallback_exc

    return GeoManager, highlight_geo, hide_layer


def import_sensor_location_display_helpers():
    """Import point-display helpers from the project visualization module."""
    import_errors = []

    module_names = [
        "detection.visualization.geo_tools",
        "visualization.geo_tools",
        "geo_tools",
    ]

    for module_name in module_names:
        try:
            module = __import__(module_name, fromlist=["get_cell_data", "send_data"])
            return module.get_cell_data, module.send_data  # type: ignore[attr-defined]
        except Exception as exc:
            import_errors.append(f"{module_name}: {exc!r}")

    raise ImportError(
        "Could not import sensor location display helpers. Tried "
        + ", ".join(module_names)
        + ". "
        + "; ".join(import_errors)
    )


def extract_report_location(report: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Return the normalized report location as floats."""
    location = report.get("location") or {}
    if not isinstance(location, dict):
        location = {}

    lat = first_present(location.get("latitude"), location.get("lat"))
    lon = first_present(location.get("longitude"), location.get("lon"))
    return {"latitude": as_float(lat), "longitude": as_float(lon)}


def display_sensor_observation_location(
    report: Dict[str, Any],
    *,
    layer_name: str = "sensor_observations",
    cell_km: float = 0.25,
    color: str = "green",
    color_intensity: int = 100,
    visible: bool = True,
) -> bool:
    """Display one report's sensor location as a colored map cell.

    This is intentionally separate from the report payload so downstream
    observation processing stays unchanged. The visualization layer is only
    for showing where the observation came from.
    """
    location = extract_report_location(report)
    lat = location.get("latitude")
    lon = location.get("longitude")

    if lat is None or lon is None:
        log(f"Skipping sensor-location display for report {report.get('report_id')!r}; missing lat/lon.")
        return False

    get_cell_data, send_data = import_sensor_location_display_helpers()

    metadata = report.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    sensor_data = {
        "kind": "sensor_observation_location",
        "report_id": report.get("report_id"),
        "report_date": report.get("report_date"),
        "sensor_id": report.get("sensor_id"),
        "sensor_name": report.get("sensor_name"),
        "sensor_type": report.get("sensor_type"),
        "latitude": lat,
        "longitude": lon,
        "incident_id": metadata.get("incident_id"),
        "modality": metadata.get("modality"),
        "step": metadata.get("step"),
        "row_index": metadata.get("row_index"),
    }
    sensor_data = {key: value for key, value in sensor_data.items() if value is not None}

    text = (
        f"sensor observation\n"
        f"sensor={report.get('sensor_id')}\n"
        f"type={report.get('sensor_type')}\n"
        f"step={metadata.get('step')}"
    )

    cell = get_cell_data(lat, lon, text, color, cell_km=cell_km)
    cell["color"] = color
    cell["color_intensity"] = color_intensity
    cell["data"] = sensor_data

    try:
        send_data(layer_name, cell_km, [cell], add_to_json=False, visible=visible)
    except TypeError:
        # Backward-compatible fallback if an older geo_tools.py is on disk.
        send_data(layer_name, cell_km, [cell])

    return True


def extract_report_step(report: Dict[str, Any]) -> Optional[Any]:
    """Return the simulator step stored in the normalized report metadata."""
    metadata = report.get("metadata") or {}
    if not isinstance(metadata, dict):
        return None
    return metadata.get("step")


def extract_report_incident_id(report: Dict[str, Any]) -> Optional[str]:
    """Return incident_id from metadata, falling back to the report_id prefix."""
    metadata = report.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("incident_id"):
        return str(metadata["incident_id"])

    report_id = report.get("report_id")
    if isinstance(report_id, str) and "_step" in report_id:
        return report_id.split("_step", 1)[0]

    return None


def resolve_ground_truth_path(
    input_folder: str | Path,
    step: Any,
    *,
    incident_id: Optional[str] = None,
    ground_truth_folder: Optional[str | Path] = None,
    filename_template: Optional[str] = None,
) -> Optional[Path]:
    """Find the ground-truth JSON file for one simulator step.

    Supported names include:
      - {incident_id}_gt_{step}.json
      - {incident_id}_gt_step{step}.json
      - gt_{step}.json
      - gt_step{step}.json

    A glob fallback is included so duplicate-download suffixes like ``(1)`` or
    minor naming differences do not break development runs.
    """
    input_folder = Path(input_folder)
    search_folder = Path(ground_truth_folder) if ground_truth_folder is not None else input_folder
    step_text = str(step)

    candidates: list[Path] = []

    if filename_template:
        try:
            candidates.append(search_folder / filename_template.format(
                incident_id=incident_id or "",
                step=step_text,
            ))
        except KeyError as exc:
            raise ValueError(
                f"Unknown field in ground truth filename template {filename_template!r}: {exc}"
            ) from exc

    if incident_id:
        candidates.extend([
            search_folder / f"{incident_id}_gt_{step_text}.json",
            search_folder / f"{incident_id}_gt_step{step_text}.json",
            search_folder / f"{incident_id}_step{step_text}_gt.json",
        ])

    candidates.extend([
        search_folder / f"gt_{step_text}.json",
        search_folder / f"gt_step{step_text}.json",
        search_folder / f"step{step_text}_gt.json",
    ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback for slightly different names while avoiding gt_10 for step 1.
    step_token_re = re.compile(rf"(?:^|[_\-]|step){re.escape(step_text)}(?:$|[_.\-)])")
    glob_patterns = []
    if incident_id:
        glob_patterns.extend([
            f"{incident_id}*gt*{step_text}*.json",
            f"{incident_id}*step*{step_text}*.json",
        ])
    glob_patterns.extend([
        f"*gt*{step_text}*.json",
        f"*step*{step_text}*gt*.json",
    ])

    seen: set[Path] = set()
    glob_matches: list[Path] = []
    for pattern in glob_patterns:
        for match in search_folder.glob(pattern):
            if match in seen or not match.is_file():
                continue
            seen.add(match)
            name = match.stem
            if step_token_re.search(name) or f"gt_{step_text}" in name or f"step{step_text}" in name:
                glob_matches.append(match)

    if glob_matches:
        # Prefer incident-specific files and shorter/exacter names.
        glob_matches.sort(key=lambda p: (
            0 if incident_id and incident_id in p.name else 1,
            len(p.name),
            p.name,
        ))
        return glob_matches[0]

    return None


def iter_ground_truth_locations(gt_data: Dict[str, Any]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield GT region records, preferring the new single incident_region form.

    New simulator GT files include an explicit incident_region with the exact
    normalized/possibly-expanded polygon serialized as GeoJSON.  Older GT files
    only have a locations map, so this generator falls back to that layout.
    """
    incident_region = gt_data.get("incident_region")
    if isinstance(incident_region, dict):
        location_name = first_present(
            incident_region.get("name"),
            incident_region.get("location_name"),
            incident_region.get("query"),
            incident_region.get("place"),
            "incident_region",
        )
        yield str(location_name), incident_region
        return

    locations = gt_data.get("locations", {})

    if isinstance(locations, dict):
        for location_name, location_data in locations.items():
            if not location_name:
                continue
            if isinstance(location_data, dict):
                yield str(location_name), location_data
            else:
                yield str(location_name), {"value": location_data}
        return

    if isinstance(locations, list):
        for idx, item in enumerate(locations):
            if not isinstance(item, dict):
                continue
            location_name = first_present(
                item.get("location_name"),
                item.get("name"),
                item.get("query"),
                item.get("place"),
            )
            if location_name is None:
                log(f"Skipping ground truth location at index {idx}; no location name was found.")
                continue
            yield str(location_name), item


def geometry_from_geojson(geometry_geojson: Any) -> Any:
    """Convert a serialized GT GeoJSON geometry into a Shapely geometry."""
    if not isinstance(geometry_geojson, dict):
        return None
    try:
        from shapely.geometry import shape  # type: ignore
        geom = shape(geometry_geojson)
        if geom is None or getattr(geom, "is_empty", False):
            return None
        return geom
    except Exception as exc:
        log(f"WARNING: could not parse GT geometry_geojson: {exc}")
        return None


def ground_truth_region_metadata(
    gt_data: Dict[str, Any],
    location_name: str,
    location_data: Dict[str, Any],
    gt_path: Path,
    step: Any,
    incident_id: Optional[str],
    *,
    geo_source: Optional[str] = None,
    geo_method: Optional[str] = None,
) -> Dict[str, Any]:
    """Build compact metadata attached to highlighted GT cells."""
    region_size_check = location_data.get("region_size_check")
    region_data = {
        "kind": "ground_truth_region",
        "incident_id": gt_data.get("incident_id", incident_id),
        "step": gt_data.get("step", step),
        "time": gt_data.get("time"),
        "location_name": location_name,
        "gt_file": str(gt_path),
        "geo_source": geo_source,
        "geo_method": geo_method,
        "representative_latitude": location_data.get("representative_latitude") or location_data.get("latitude"),
        "representative_longitude": location_data.get("representative_longitude") or location_data.get("longitude"),
        "single_region_for_incident": location_data.get("single_region_for_incident"),
        "region_expanded": region_size_check.get("expanded") if isinstance(region_size_check, dict) else None,
        "minimum_region_side_km": region_size_check.get("minimum_side_km") if isinstance(region_size_check, dict) else None,
    }
    return {k: v for k, v in region_data.items() if v is not None}

def display_ground_truth_for_step(
    input_folder: str | Path,
    step: Any,
    *,
    incident_id: Optional[str] = None,
    geo_manager: Any = None,
    ground_truth_folder: Optional[str | Path] = None,
    filename_template: Optional[str] = None,
    layer_prefix: str = "ground_truth",
    cell_km: float = 0.25,
    color: str = "orange",
    color_intensity: int = 30,
    visible: bool = True,
    previous_layer_names: Optional[list[str]] = None,
) -> Tuple[Any, list[str]]:
    """Load one step's GT file and display its regions on the map.

    Returns the possibly-created GeoManager and the list of visible GT layer
    names for this step, so the caller can hide them on the next step change.
    """
    GeoManager, highlight_geo, hide_layer = import_ground_truth_display_helpers()

    for old_layer_name in previous_layer_names or []:
        try:
            hide_layer(old_layer_name)
        except Exception as exc:
            log(f"WARNING: could not hide previous ground-truth layer {old_layer_name!r}: {exc}")

    gt_path = resolve_ground_truth_path(
        input_folder,
        step,
        incident_id=incident_id,
        ground_truth_folder=ground_truth_folder,
        filename_template=filename_template,
    )
    if gt_path is None:
        log(
            f"WARNING: no ground-truth JSON found for step {step!r} "
            f"in {Path(ground_truth_folder) if ground_truth_folder else Path(input_folder)}"
        )
        return geo_manager, []

    with gt_path.open("r", encoding="utf-8") as infile:
        gt_data = json.load(infile)

    if not isinstance(gt_data, dict):
        log(f"WARNING: ground-truth file is not a JSON object: {gt_path}")
        return geo_manager, []

    layer_name = f"{layer_prefix}_step_{step}"
    current_layers: list[str] = []
    highlighted_count = 0

    for location_name, location_data in iter_ground_truth_locations(gt_data):
        geom = geometry_from_geojson(location_data.get("geometry_geojson"))
        geo_source = "ground_truth_json" if geom is not None else None
        geo_method = "geometry_geojson" if geom is not None else None

        if geom is None:
            # Backward-compatible fallback for older GT files that only stored a
            # name under locations. New files should not need this path.
            if geo_manager is None:
                geo_manager = GeoManager()
            try:
                region_record = geo_manager.get_geo_region(location_name)
            except Exception as exc:
                log(f"WARNING: GeoManager failed for GT location {location_name!r}: {exc}")
                continue

            if not isinstance(region_record, dict) or region_record.get("status") != "ok":
                log(f"WARNING: no usable GT region for {location_name!r}: {region_record}")
                continue

            geom = region_record.get("geometry")
            geo_source = region_record.get("source")
            geo_method = region_record.get("method")
            if geom is None:
                log(f"WARNING: GT region has no geometry for {location_name!r}")
                continue

        region_data = ground_truth_region_metadata(
            gt_data,
            location_name,
            location_data,
            gt_path,
            step,
            incident_id,
            geo_source=geo_source,
            geo_method=geo_method,
        )

        try:
            highlight_geo(
                geom,
                layer_name,
                cell_km,
                region_data=region_data,
                color=color,
                color_intensity=color_intensity,
                visible=visible,
                add_to_json=False,
            )
        except TypeError:
            # Backward-compatible fallback if an older geo_tools.py is on disk.
            highlight_geo(geom, layer_name, cell_km, region_data=region_data)
        except Exception as exc:
            log(f"WARNING: could not display GT region {location_name!r}: {exc}")
            continue

        highlighted_count += 1
        if layer_name not in current_layers:
            current_layers.append(layer_name)

    log(
        f"Displayed {highlighted_count} ground-truth region(s) for step {step} "
        f"from {gt_path} on layer {layer_name!r}."
    )
    return geo_manager, current_layers


# ---------------------------------------------------------------------------
# Ground-truth boundary filtering
# ---------------------------------------------------------------------------

def default_gt_grid_bounds() -> Dict[str, float]:
    """Default detector/visualization grid used to reject out-of-bounds GT runs."""
    return {
        "lat_min": 33.9,
        "lon_min": -118.95,
        "lat_max": 34.35,
        "lon_max": -118.0,
        "cell_km": 0.25,
    }


def _iter_geojson_lonlat_pairs(value: Any) -> Iterator[Tuple[float, float]]:
    """Yield lon/lat coordinate pairs from arbitrary GeoJSON coordinates."""
    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and all(isinstance(x, (int, float)) for x in value[:2]):
            yield float(value[0]), float(value[1])
            return
        for item in value:
            yield from _iter_geojson_lonlat_pairs(item)


def geojson_lonlat_bounds(geometry_geojson: Any) -> Optional[Tuple[float, float, float, float]]:
    """Return (lon_min, lat_min, lon_max, lat_max) for a GeoJSON geometry/feature."""
    if not isinstance(geometry_geojson, dict):
        return None

    # Support either a Geometry, a Feature, or a FeatureCollection-like object.
    if geometry_geojson.get("type") == "Feature":
        return geojson_lonlat_bounds(geometry_geojson.get("geometry"))
    if geometry_geojson.get("type") == "FeatureCollection":
        bounds = [geojson_lonlat_bounds(feature) for feature in geometry_geojson.get("features", [])]
        bounds = [b for b in bounds if b is not None]
        if not bounds:
            return None
        return (
            min(b[0] for b in bounds),
            min(b[1] for b in bounds),
            max(b[2] for b in bounds),
            max(b[3] for b in bounds),
        )

    coords = list(_iter_geojson_lonlat_pairs(geometry_geojson.get("coordinates")))
    if not coords:
        # Some records may expose a bbox directly as [lon_min, lat_min, lon_max, lat_max].
        bbox = geometry_geojson.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            except (TypeError, ValueError):
                return None
        return None

    lons = [lon for lon, _ in coords]
    lats = [lat for _, lat in coords]
    return min(lons), min(lats), max(lons), max(lats)


def bounds_inside_grid(bounds: Tuple[float, float, float, float], grid_bounds: Dict[str, Any]) -> bool:
    """Return True only if the entire lon/lat bounds box is inside grid_bounds."""
    lon_min, lat_min, lon_max, lat_max = bounds
    return (
        lat_min >= float(grid_bounds["lat_min"])
        and lat_max <= float(grid_bounds["lat_max"])
        and lon_min >= float(grid_bounds["lon_min"])
        and lon_max <= float(grid_bounds["lon_max"])
    )


def point_inside_grid(lat: Any, lon: Any, grid_bounds: Dict[str, Any]) -> Optional[bool]:
    """Return whether a lat/lon point is inside the grid, or None if invalid."""
    lat_f = as_float(lat)
    lon_f = as_float(lon)
    if lat_f is None or lon_f is None:
        return None
    return (
        float(grid_bounds["lat_min"]) <= lat_f <= float(grid_bounds["lat_max"])
        and float(grid_bounds["lon_min"]) <= lon_f <= float(grid_bounds["lon_max"])
    )


def discover_ground_truth_jsons(incident_folder: str | Path) -> list[Path]:
    """Find likely ground-truth JSON files inside one incident run folder."""
    folder = Path(incident_folder)
    if not folder.exists():
        return []
    paths: list[Path] = []
    for path in folder.glob("*.json"):
        stem = path.stem.lower()
        name = path.name.lower()
        if (
            "ground_truth" in stem
            or "_gt_" in stem
            or stem.startswith("gt_")
            or stem.startswith("gtstep")
            or "_gtstep" in stem
            or stem.endswith("_gt")
            or re.search(r"(?:^|[_\-])gt(?:$|[_\-])", stem) is not None
        ):
            paths.append(path)
    return sorted(paths, key=lambda p: p.name)


def ground_truth_file_outside_grid(gt_path: str | Path, grid_bounds: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Check whether any GT region in one file extends outside the target grid."""
    gt_path = Path(gt_path)
    try:
        with gt_path.open("r", encoding="utf-8") as infile:
            gt_data = json.load(infile)
    except Exception as exc:
        return False, {"gt_file": str(gt_path), "checked_regions": 0, "error": f"could_not_read_gt_json: {exc}"}

    if not isinstance(gt_data, dict):
        return False, {"gt_file": str(gt_path), "checked_regions": 0, "error": "ground_truth_json_not_object"}

    checked_regions = 0
    fallback_points_checked = 0
    for location_name, location_data in iter_ground_truth_locations(gt_data):
        if not isinstance(location_data, dict):
            continue

        geometry_geojson = location_data.get("geometry_geojson")
        bounds = geojson_lonlat_bounds(geometry_geojson)
        if bounds is not None:
            checked_regions += 1
            if not bounds_inside_grid(bounds, grid_bounds):
                return True, {
                    "gt_file": str(gt_path),
                    "location_name": location_name,
                    "reason": "ground_truth_geometry_outside_grid",
                    "geometry_bounds": {
                        "lon_min": bounds[0],
                        "lat_min": bounds[1],
                        "lon_max": bounds[2],
                        "lat_max": bounds[3],
                    },
                    "grid_bounds": grid_bounds,
                    "checked_regions": checked_regions,
                }
            continue

        # Backward-compatible fallback for older GT files without geometry.
        # This cannot prove full area containment, but it prevents obvious
        # out-of-grid point-only GT records from being streamed.
        lat = first_present(location_data.get("representative_latitude"), location_data.get("latitude"), location_data.get("lat"))
        lon = first_present(location_data.get("representative_longitude"), location_data.get("longitude"), location_data.get("lon"))
        inside = point_inside_grid(lat, lon, grid_bounds)
        if inside is not None:
            fallback_points_checked += 1
            if not inside:
                return True, {
                    "gt_file": str(gt_path),
                    "location_name": location_name,
                    "reason": "ground_truth_point_outside_grid",
                    "latitude": as_float(lat),
                    "longitude": as_float(lon),
                    "grid_bounds": grid_bounds,
                    "checked_regions": checked_regions,
                    "fallback_points_checked": fallback_points_checked,
                }

    return False, {
        "gt_file": str(gt_path),
        "checked_regions": checked_regions,
        "fallback_points_checked": fallback_points_checked,
    }


def should_skip_incident_for_ground_truth_grid(
    incident_folder: str | Path,
    *,
    grid_bounds: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Return True if any GT region in the incident folder extends outside grid."""
    grid_bounds = dict(grid_bounds or default_gt_grid_bounds())
    gt_paths = discover_ground_truth_jsons(incident_folder)
    if not gt_paths:
        return False, {
            "incident_folder": str(incident_folder),
            "reason": "no_ground_truth_json_found",
            "grid_bounds": grid_bounds,
            "checked_files": 0,
        }

    summaries = []
    total_regions = 0
    total_fallback_points = 0
    for gt_path in gt_paths:
        outside, summary = ground_truth_file_outside_grid(gt_path, grid_bounds)
        summaries.append(summary)
        total_regions += int(summary.get("checked_regions") or 0)
        total_fallback_points += int(summary.get("fallback_points_checked") or 0)
        if outside:
            summary = dict(summary)
            summary.update({
                "incident_folder": str(incident_folder),
                "checked_files": len(summaries),
                "total_gt_files": len(gt_paths),
            })
            return True, summary

    return False, {
        "incident_folder": str(incident_folder),
        "reason": "all_ground_truth_regions_inside_grid",
        "grid_bounds": grid_bounds,
        "checked_files": len(gt_paths),
        "checked_regions": total_regions,
        "fallback_points_checked": total_fallback_points,
        "gt_files": [str(p) for p in gt_paths],
    }


def open_jsonl_socket(
    host: str,
    port: int,
    *,
    connect_timeout_seconds: float = 10.0,
) -> Tuple[socket.socket, Any, Any]:
    """Open a TCP connection for newline-delimited JSON writes + ACK reads."""
    log(f"Connecting to full_pipeline.py at {host}:{port} ...")
    try:
        sock = socket.create_connection((host, port), timeout=connect_timeout_seconds)
    except OSError as exc:
        raise ConnectionError(
            f"Could not connect to full_pipeline.py at {host}:{port}. "
            "Start full_pipeline.py first and make sure SOCKET_HOST/SOCKET_PORT match."
        ) from exc

    # After connecting, wait indefinitely for ACKs. LLM/VLM calls can exceed the
    # short connection timeout, and that is exactly when we want backpressure.
    sock.settimeout(None)
    reader = sock.makefile("r", encoding="utf-8", newline="\n")
    writer = sock.makefile("w", encoding="utf-8", newline="\n")
    log(f"Connected to full_pipeline.py at {host}:{port}")
    return sock, reader, writer


def ack_is_ok(ack: Dict[str, Any]) -> bool:
    """Accept both old and new ACK schemas.

    New schema: {"type": "ack", "ok": true, ...}
    Old schema: {"type": "ack", "status": "ok", ...}
    """
    if ack.get("ok") is True:
        return True
    status = str(ack.get("status", "")).strip().lower()
    return status in {"ok", "success", "processed"}


def read_ack(socket_reader: Any, *, report_id: Any = None) -> Dict[str, Any]:
    """Block until full_pipeline.py ACKs the report.

    The ACK is sent by full_pipeline.py after observation-model inference and
    hypothesis proposal finish, so this function is the backpressure point that
    pauses the emitter while the LLM/VLM is running.
    """
    line = socket_reader.readline()
    if line == "":
        raise RuntimeError(
            f"full_pipeline.py closed the connection before ACK for report {report_id!r}"
        )

    try:
        ack = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid ACK JSON from full_pipeline.py: {line!r}") from exc

    if not isinstance(ack, dict):
        raise RuntimeError(f"ACK from full_pipeline.py was not a JSON object: {ack!r}")

    if str(ack.get("type", "ack")).lower() != "ack":
        raise RuntimeError(f"Unexpected response from full_pipeline.py: {ack}")

    ack_report_id = ack.get("report_id")
    if report_id is not None and ack_report_id not in {None, report_id}:
        raise RuntimeError(
            f"ACK report_id mismatch: expected {report_id!r}, got {ack_report_id!r}. ACK={ack}"
        )

    if not ack_is_ok(ack):
        raise RuntimeError(f"full_pipeline.py returned a non-ok ACK: {ack}")

    return ack


def stream_reports(
    input_folder: str | Path,
    *,
    observations_filename: str = "observations.txt",
    interval_seconds: float = 1.0,
    output_path: Optional[str | Path] = None,
    sleep_between_reports: bool = True,
    print_reports: bool = True,
    assume_same_depth: bool = True,
    socket_host: Optional[str] = None,
    socket_port: Optional[int] = None,
    wait_for_socket_ack: bool = True,
    display_ground_truth: bool = False,
    ground_truth_folder: Optional[str | Path] = None,
    ground_truth_filename_template: Optional[str] = None,
    ground_truth_cell_km: float = 0.25,
    ground_truth_layer_prefix: str = "ground_truth",
    ground_truth_color: str = "orange",
    ground_truth_color_intensity: int = 30,
    ground_truth_visible: bool = True,
    display_sensor_locations: bool = False,
    sensor_locations_layer_name: str = "sensor_observations",
    sensor_locations_cell_km: float = 0.25,
    sensor_locations_color: str = "green",
    sensor_locations_color_intensity: int = 100,
    sensor_locations_visible: bool = True,
    prompt_between_reports: bool = False,
    show_progress: bool = True,
    skip_if_ground_truth_outside_grid: bool = False,
    ground_truth_grid_bounds: Optional[Dict[str, Any]] = None,
    modality_filter: str = "all",
) -> int:
    """Emit one report at a time.

    A report can be printed, written to JSONL, sent over a JSONL TCP socket, or
    any combination of those. Socket mode sends one JSON object per line.

    When wait_for_socket_ack=True, each socket send blocks until full_pipeline.py
    sends an ACK. This pauses the emitter while the observation model is running.

    If display_ground_truth=True, the emitter watches report["metadata"]["step"].
    Whenever the step changes, it loads the corresponding GT JSON file, resolves
    the GT locations through GeoManager, and displays those regions on the map
    before emitting the first report for that step.

    If display_sensor_locations=True, each emitted report's sensor location is
    also displayed as a green cell on the sensor-observation layer.
    """
    output_handle = None
    socket_reader = None
    socket_writer = None
    socket_obj = None
    report_count = 0
    last_ground_truth_step = object()
    geo_manager = None
    active_ground_truth_layers: list[str] = []

    try:
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_handle = output_path.open("w", encoding="utf-8")

        if socket_host is not None and socket_port is not None:
            socket_obj, socket_reader, socket_writer = open_jsonl_socket(socket_host, socket_port)
        else:
            log("Socket output is disabled; emitting locally only.")

        for report in iter_reports(
            input_folder,
            observations_filename=observations_filename,
            assume_same_depth=assume_same_depth,
        ):
            if not report_matches_modality_filter(report, modality_filter):
                continue
            if display_ground_truth:
                step = extract_report_step(report)
                if step is not None and step != last_ground_truth_step:
                    incident_id = extract_report_incident_id(report)
                    try:
                        geo_manager, active_ground_truth_layers = display_ground_truth_for_step(
                            input_folder,
                            step,
                            incident_id=incident_id,
                            geo_manager=geo_manager,
                            ground_truth_folder=ground_truth_folder,
                            filename_template=ground_truth_filename_template,
                            layer_prefix=ground_truth_layer_prefix,
                            cell_km=ground_truth_cell_km,
                            color=ground_truth_color,
                            color_intensity=ground_truth_color_intensity,
                            visible=ground_truth_visible,
                            previous_layer_names=active_ground_truth_layers,
                        )
                    except Exception as exc:
                        log(f"WARNING: failed to display ground truth for step {step!r}: {exc}")
                    last_ground_truth_step = step

            if display_sensor_locations:
                try:
                    display_sensor_observation_location(
                        report,
                        layer_name=sensor_locations_layer_name,
                        cell_km=sensor_locations_cell_km,
                        color=sensor_locations_color,
                        color_intensity=sensor_locations_color_intensity,
                        visible=sensor_locations_visible,
                    )
                except Exception as exc:
                    log(
                        f"WARNING: failed to display sensor location for "
                        f"report {report.get('report_id')!r}: {exc}"
                    )

            line = json.dumps(report, ensure_ascii=False)

            if print_reports:
                print(line, flush=True)

            if output_handle is not None:
                output_handle.write(line + "\n")
                output_handle.flush()

            if socket_writer is not None:
                socket_writer.write(line + "\n")
                socket_writer.flush()

                if report_count == 0:
                    log("Sent first report to full_pipeline.py")

                if wait_for_socket_ack:
                    read_ack(socket_reader, report_id=report.get("report_id"))

            report_count += 1

            if sleep_between_reports and interval_seconds > 0:
                time.sleep(interval_seconds)

            if prompt_between_reports:
                input("Press Enter to emit the next report...")

    finally:
        if output_handle is not None:
            output_handle.close()
        if socket_writer is not None:
            socket_writer.close()
        if socket_reader is not None:
            socket_reader.close()
        if socket_obj is not None:
            socket_obj.close()

    return report_count



def is_incident_run_folder(path: str | Path, observations_filename: str = "observations.txt") -> bool:
    """Return True when a folder looks like one simulator incident/run folder."""
    path = Path(path)
    return path.is_dir() and (path / observations_filename).exists()


def discover_incident_folders(
    batch_roots: Sequence[str | Path],
    *,
    observations_filename: str = "observations.txt",
    recursive: bool = False,
) -> list[Path]:
    """Find incident run folders under one or more batch roots.

    A root may either be an incident folder itself or a batch folder whose
    immediate children are incident folders. Set recursive=True when the batch
    folder has deeper nesting.
    """
    found: list[Path] = []
    seen: set[str] = set()

    for root_value in batch_roots:
        root = Path(root_value)
        if is_incident_run_folder(root, observations_filename):
            candidates = [root]
        else:
            if not root.exists():
                log(f"WARNING: batch root does not exist: {root}")
                continue
            observation_paths = root.rglob(observations_filename) if recursive else root.glob(f"*/{observations_filename}")
            candidates = [path.parent for path in observation_paths]

        for candidate in candidates:
            try:
                key = str(candidate.resolve())
            except Exception:
                key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            found.append(candidate)

    return sorted(found, key=lambda p: str(p))


def incident_batch_root_name(incident_folder: str | Path) -> str:
    folder = Path(incident_folder)
    parent = folder.parent
    return parent.name if parent.name else "batch"


def send_control_message(
    socket_writer: Any,
    socket_reader: Any,
    *,
    message_type: str,
    wait_for_socket_ack: bool,
    **payload: Any,
) -> None:
    """Send an incident_start / incident_end control message to full_pipeline.py."""
    message = {"type": message_type, **payload}
    socket_writer.write(json.dumps(message, ensure_ascii=False) + "\n")
    socket_writer.flush()
    if wait_for_socket_ack:
        read_ack(socket_reader, report_id=None)



def discover_incident_groups(
    batch_roots: Sequence[str | Path],
    *,
    observations_filename: str = "observations.txt",
    recursive: bool = False,
    sync_multilevel_incidents: bool = False,
) -> list[Dict[str, Any]]:
    """Return streamable incident groups.

    Normal mode preserves the old behavior: each folder containing
    observations.txt is streamed as its own incident.  Synchronized multi-level
    mode treats each supplied parent root as one operational run when that root
    contains multiple child incident folders.  Reports from those children are
    then merged and sorted before streaming, so subincident A does not fully run
    before subincident B.
    """
    groups: list[Dict[str, Any]] = []
    seen_runs: set[str] = set()

    if not sync_multilevel_incidents:
        for folder in discover_incident_folders(
            batch_roots,
            observations_filename=observations_filename,
            recursive=recursive,
        ):
            try:
                key = str(folder.resolve())
            except Exception:
                key = str(folder)
            if key in seen_runs:
                continue
            seen_runs.add(key)
            groups.append({
                "group_folder": Path(folder),
                "run_folders": [Path(folder)],
                "multi_level": False,
            })
        return groups

    for root_value in batch_roots:
        root = Path(root_value)
        if is_incident_run_folder(root, observations_filename):
            candidate_runs = [root]
            group_folder = root
        else:
            if not root.exists():
                log(f"WARNING: batch root does not exist: {root}")
                continue
            candidate_runs = discover_incident_folders(
                [root],
                observations_filename=observations_filename,
                recursive=recursive,
            )
            group_folder = root

        # If recursive discovery finds several independent parent folders under
        # one broad root, group by each run folder's immediate parent.  For the
        # common case where BATCH_ROOTS contains one multi-level incident folder
        # directly, this produces one group containing all child subincidents.
        if recursive and not is_incident_run_folder(root, observations_filename):
            by_parent: Dict[str, list[Path]] = {}
            parent_paths: Dict[str, Path] = {}
            for run in candidate_runs:
                parent = run.parent
                try:
                    parent_key = str(parent.resolve())
                except Exception:
                    parent_key = str(parent)
                by_parent.setdefault(parent_key, []).append(run)
                parent_paths[parent_key] = parent
            parent_items = sorted(by_parent.items(), key=lambda item: item[0])
            for parent_key, runs in parent_items:
                runs = sorted(runs, key=lambda p: str(p))
                for run in runs:
                    try:
                        run_key = str(run.resolve())
                    except Exception:
                        run_key = str(run)
                    if run_key in seen_runs:
                        continue
                    seen_runs.add(run_key)
                groups.append({
                    "group_folder": parent_paths[parent_key],
                    "run_folders": runs,
                    "multi_level": len(runs) > 1,
                })
            continue

        candidate_runs = sorted(candidate_runs, key=lambda p: str(p))
        if not candidate_runs:
            continue
        for run in candidate_runs:
            try:
                run_key = str(run.resolve())
            except Exception:
                run_key = str(run)
            if run_key in seen_runs:
                continue
            seen_runs.add(run_key)
        groups.append({
            "group_folder": group_folder,
            "run_folders": candidate_runs,
            "multi_level": len(candidate_runs) > 1,
        })

    return groups


def _parse_sort_timestamp(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).timestamp()


def _sort_step_value(value: Any) -> Tuple[int, Any]:
    if value is None or value == "":
        return (2, 0)
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def report_entry_sort_key(entry: Dict[str, Any], *, mode: str = "time_then_step") -> Tuple[Any, ...]:
    """Stable ordering key for synchronized multi-level incident reports."""
    report = entry.get("report") or {}
    metadata = report.get("metadata") if isinstance(report.get("metadata"), dict) else {}
    step_key = _sort_step_value(metadata.get("step"))
    timestamp = _parse_sort_timestamp(report.get("report_date"))
    time_key = timestamp if timestamp is not None else float("inf")
    sub_index = int(entry.get("subincident_run_index") or 0)
    raw_index = int(metadata.get("raw_observation_index") or entry.get("report_index") or 0)
    source_name = str(entry.get("subincident_run_name") or "")

    mode = str(mode or "time_then_step").strip().lower()
    if mode in {"step", "step_then_time", "steps"}:
        return (step_key, time_key, sub_index, raw_index, source_name)
    if mode in {"time", "time_then_step", "datetime", "chronological"}:
        return (time_key, step_key, sub_index, raw_index, source_name)
    raise ValueError("multi_level_sort_mode must be 'time_then_step' or 'step_then_time'")


def collect_group_report_entries(
    run_folders: Sequence[str | Path],
    *,
    observations_filename: str,
    assume_same_depth: bool,
    multi_level_sort_mode: str,
) -> list[Dict[str, Any]]:
    """Collect and sort reports across subincident folders."""
    entries: list[Dict[str, Any]] = []
    for sub_index, run_folder_value in enumerate(run_folders, start=1):
        run_folder = Path(run_folder_value)
        try:
            run_folder_resolved = str(run_folder.resolve())
        except Exception:
            run_folder_resolved = str(run_folder)
        for report_index, report in enumerate(
            iter_reports(
                run_folder,
                observations_filename=observations_filename,
                assume_same_depth=assume_same_depth,
            ),
            start=1,
        ):
            metadata = report.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["subincident_run_index"] = sub_index
                metadata["subincident_run_name"] = run_folder.name
                metadata["subincident_run_folder"] = str(run_folder)
                metadata["subincident_run_folder_resolved"] = run_folder_resolved
                metadata["subincident_report_index"] = report_index
                metadata["multi_level_sync_enabled"] = len(run_folders) > 1
            entries.append({
                "report": report,
                "source_folder": run_folder,
                "subincident_run_index": sub_index,
                "subincident_run_name": run_folder.name,
                "report_index": report_index,
            })

    entries.sort(key=lambda entry: report_entry_sort_key(entry, mode=multi_level_sort_mode))
    return entries


def should_skip_group_for_ground_truth_grid(
    run_folders: Sequence[str | Path],
    *,
    grid_bounds: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Return True if any subincident GT region is outside the grid."""
    summaries = []
    for run_folder in run_folders:
        should_skip, summary = should_skip_incident_for_ground_truth_grid(
            run_folder,
            grid_bounds=grid_bounds,
        )
        summaries.append(summary)
        if should_skip:
            return True, {
                "reason": "subincident_ground_truth_outside_grid",
                "subincident_folder": str(run_folder),
                "subincident_summary": summary,
                "checked_subincident_folders": len(summaries),
                "total_subincident_folders": len(run_folders),
            }
    return False, {
        "reason": "all_subincident_ground_truth_regions_inside_grid",
        "checked_subincident_folders": len(run_folders),
        "subincident_summaries": summaries,
    }



def _stable_unit_float(*parts: Any) -> float:
    """Deterministic pseudo-random number in [0,1) from JSON-ish parts."""
    payload = json.dumps([str(part) for part in parts], ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16 ** 16)


def _report_sensor_key(report: Dict[str, Any]) -> str:
    return str(
        first_present(
            report.get("sensor_id"),
            report.get("sensor_name"),
            report.get("report_id"),
            "unknown_sensor",
        )
    )


def _report_key(report: Dict[str, Any]) -> str:
    metadata = report.get("metadata") if isinstance(report.get("metadata"), dict) else {}
    return str(
        first_present(
            report.get("report_id"),
            metadata.get("raw_observation_index"),
            metadata.get("data_file"),
            _report_sensor_key(report),
        )
    )


SENSOR_ONLY_SOURCE_ALIASES = {
    # Cameras / imagery.
    "alertcalifornia",
    "alert_california",
    "caltrans",
    "caltrans_cctv",
    "camera",
    "cctv",
    "image",
    "traffic_camera",
    "webcam",
    # Traffic time series.
    "pems",
    "pem",
    "pem_data",
    "pem_data_station_5min",
    "traffic",
    "traffic_data",
    "traffic_sensor",
    "road_sensor",
    # Air quality.
    "air",
    "air_data",
    "air_quality",
    "air_quality_data",
    "purpleair",
    "purple_air",
    "pm25",
    "pm2_5",
    "aqi",
    # Weather.
    "weather",
    "weather_data",
    "openweather",
    "open_weather",
    "weather_station",
}

OPERATIONAL_TEXT_SOURCE_ALIASES = {
    "citizen",
    "citizen_data",
    "citizenapp",
    "citizen_app",
    "citizen_app_data",
    "x",
    "x_data",
    "twitter",
    "twitter_data",
    "tweet",
    "tweets",
    "social",
    "social_media",
    "official_alert",
    "official_alerts",
    "official_alert_data",
    "official_alerts_data",
    "public_alert",
    "public_alerts",
    "emergency_alert",
    "emergency_alerts",
    "nixle",
    "la_county_alert",
    "lacounty_alert",
}

LABEL_CONSTRUCTION_TEXT_SOURCE_ALIASES = {
    "article",
    "articles",
    "article_text",
    "event_article",
    "gdelt",
    "gkg",
    "news",
    "news_article",
    "news_articles",
    "news_data",
    "label_article",
    "label_articles",
    "label_construction_article",
    "label_construction_articles",
    "label_construction_text",
}


def normalize_source_name(value: Any) -> str:
    """Normalize source names so aliases like `CitizenApp` and `citizen_app` match."""
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def report_source_aliases(report: Dict[str, Any]) -> set[str]:
    """Return normalized source/modality aliases available on one normalized report."""
    aliases: set[str] = set()

    def add(value: Any) -> None:
        norm = normalize_source_name(value)
        if norm:
            aliases.add(norm)

    add(report.get("sensor_type"))
    add(report.get("source"))
    add(report.get("modality"))
    metadata = report.get("metadata") if isinstance(report.get("metadata"), dict) else {}
    if isinstance(metadata, dict):
        for key in (
            "source",
            "sensor_type",
            "modality",
            "data_source",
            "source_name",
            "subincident_run_name",
        ):
            add(metadata.get(key))
        data_file = metadata.get("data_file")
        if data_file:
            stem = Path(str(data_file)).stem
            add(stem)
            # Many simulator files include the source as a token in the filename.
            for token in re.split(r"[^A-Za-z0-9]+", stem):
                add(token)

    # Image reports are camera/sensor observations even if the source alias is unusual.
    data = report.get("data") if isinstance(report.get("data"), dict) else {}
    if isinstance(data, dict) and data.get("image_filepath"):
        aliases.add("image")
        aliases.add("camera")

    # Common row/data field hints for numeric sensor streams.
    data_keys = {normalize_source_name(k) for k in data.keys()} if isinstance(data, dict) else set()
    if {"pm2_5", "pm25", "aqi"} & data_keys:
        aliases.add("air_quality")
    if {"avg_speed", "average_speed", "avg_occupancy", "occupancy", "speed"} & data_keys:
        aliases.add("traffic")
    if {"temperature", "wind_speed", "wind_direction", "humidity", "precipitation"} & data_keys:
        aliases.add("weather")
    if {"text", "message", "headline", "description", "body", "post"} & data_keys:
        aliases.add("text")

    return aliases


def report_matches_modality_filter(report: Dict[str, Any], modality_filter: str) -> bool:
    """Return whether a report belongs in the requested modality ablation.

    `sensor_only` keeps camera, traffic, air-quality, and weather streams.
    `operational_text_only` keeps CitizenApp, X/Twitter, and official alerts, but
    explicitly excludes news/GDELT/article text used for label construction.
    """
    mode = normalize_source_name(modality_filter) or "all"
    if mode in {"all", "none", "full"}:
        return True

    aliases = report_source_aliases(report)
    if mode in {"sensor_only", "sensors_only", "sensor"}:
        return bool(aliases & SENSOR_ONLY_SOURCE_ALIASES) and not bool(aliases & OPERATIONAL_TEXT_SOURCE_ALIASES)

    if mode in {"operational_text_only", "text_only", "operational_text"}:
        if aliases & LABEL_CONSTRUCTION_TEXT_SOURCE_ALIASES:
            return False
        return bool(aliases & OPERATIONAL_TEXT_SOURCE_ALIASES)

    raise ValueError(
        "modality_filter must be one of: all, sensor_only, operational_text_only"
    )


def filter_report_entries_by_modality(
    report_entries: list[Dict[str, Any]],
    *,
    modality_filter: str = "all",
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Filter report entries for modality-ablation experiments and return a summary."""
    mode = normalize_source_name(modality_filter) or "all"
    if mode in {"all", "none", "full"}:
        return report_entries, {
            "modality_filter": "all",
            "input_reports": len(report_entries),
            "dropped_reports": 0,
            "output_reports": len(report_entries),
            "kept_sources": {},
            "dropped_sources": {},
        }

    kept: list[Dict[str, Any]] = []
    kept_sources: Dict[str, int] = {}
    dropped_sources: Dict[str, int] = {}

    for entry in report_entries:
        report = entry.get("report") if isinstance(entry, dict) else None
        if not isinstance(report, dict):
            continue
        source_key = normalize_source_name(report.get("sensor_type")) or "unknown"
        if report_matches_modality_filter(report, mode):
            kept.append(entry)
            kept_sources[source_key] = int(kept_sources.get(source_key, 0)) + 1
        else:
            dropped_sources[source_key] = int(dropped_sources.get(source_key, 0)) + 1

    return kept, {
        "modality_filter": mode,
        "input_reports": len(report_entries),
        "dropped_reports": len(report_entries) - len(kept),
        "output_reports": len(kept),
        "kept_sources": kept_sources,
        "dropped_sources": dropped_sources,
    }


def apply_stream_perturbations(
    report_entries: list[Dict[str, Any]],
    *,
    sensor_density_fraction: float = 1.0,
    missing_observation_fraction: float = 0.0,
    corrupt_observation_fraction: float = 0.0,
    perturbation_seed: str = "0",
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Apply deterministic stress-test perturbations to a report-entry stream.

    Sensor density keeps/drops whole sensors. Missing observations drops individual
    reports. Corruption keeps the report but marks it for downstream observation-
    output label corruption in full_pipeline.py.  The raw report identity changes
    only through metadata, so cached observation outputs can still be reused and
    then perturbed after cache retrieval.
    """
    sensor_density_fraction = max(0.0, min(1.0, float(sensor_density_fraction)))
    missing_observation_fraction = max(0.0, min(1.0, float(missing_observation_fraction)))
    corrupt_observation_fraction = max(0.0, min(1.0, float(corrupt_observation_fraction)))
    seed = str(perturbation_seed)

    kept: list[Dict[str, Any]] = []
    stats = {
        "sensor_density_fraction": sensor_density_fraction,
        "missing_observation_fraction": missing_observation_fraction,
        "corrupt_observation_fraction": corrupt_observation_fraction,
        "perturbation_seed": seed,
        "input_reports": len(report_entries),
        "dropped_by_sensor_density": 0,
        "dropped_missing_observation": 0,
        "corrupted_reports": 0,
        "output_reports": 0,
    }

    for entry in report_entries:
        report = entry.get("report") if isinstance(entry, dict) else None
        if not isinstance(report, dict):
            continue
        sensor_key = _report_sensor_key(report)
        report_key = _report_key(report)

        if sensor_density_fraction < 1.0 and _stable_unit_float(seed, "sensor_density", sensor_key) >= sensor_density_fraction:
            stats["dropped_by_sensor_density"] += 1
            continue
        if missing_observation_fraction > 0.0 and _stable_unit_float(seed, "missing", report_key) < missing_observation_fraction:
            stats["dropped_missing_observation"] += 1
            continue

        if corrupt_observation_fraction > 0.0 and _stable_unit_float(seed, "corrupt", report_key) < corrupt_observation_fraction:
            # Shallow-copy entry and report so future reuse of collected entries is safe.
            entry = dict(entry)
            report = dict(report)
            metadata = dict(report.get("metadata") or {})
            metadata.update({
                "synthetic_corrupt_observation": True,
                "synthetic_corruption_seed": seed,
                "synthetic_corruption_fraction": corrupt_observation_fraction,
                "synthetic_corruption_report_key": report_key,
            })
            report["metadata"] = metadata
            entry["report"] = report
            stats["corrupted_reports"] += 1

        kept.append(entry)

    stats["output_reports"] = len(kept)
    return kept, stats


def duplicate_report_entries(
    report_entries: list[Dict[str, Any]],
    *,
    duplicate_factor: int = 1,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Duplicate each emitted report deterministically for controlled timing tests.

    The default factor of 1 preserves the stream exactly.  For factors greater
    than 1, every logical input report is emitted `duplicate_factor` times with
    a unique report_id and duplicate metadata.  This is intended for
    throughput-vs-report-count experiments where modality mix and incident
    structure should stay fixed while raw report count increases.
    """
    factor = max(1, int(duplicate_factor or 1))
    stats: Dict[str, Any] = {
        "duplicate_factor": factor,
        "input_reports": len(report_entries),
        "duplicated_reports_added": 0,
        "output_reports": len(report_entries),
    }
    if factor <= 1:
        return report_entries, stats

    duplicated: list[Dict[str, Any]] = []
    for logical_index, entry in enumerate(report_entries, start=1):
        report = entry.get("report") if isinstance(entry, dict) else None
        if not isinstance(report, dict):
            continue
        original_report_id = first_present(report.get("report_id"), f"report_{logical_index}")
        for dup_index in range(1, factor + 1):
            new_entry = dict(entry)
            new_report = dict(report)
            new_metadata = dict(new_report.get("metadata") or {})
            new_metadata.update({
                "actual_timing_duplicate_factor": factor,
                "actual_timing_duplicate_index": dup_index,
                "actual_timing_original_report_id": original_report_id,
                "actual_timing_logical_report_index": logical_index,
            })
            new_report["metadata"] = new_metadata
            # Keep the default factor=1 stream identical, but make factor>1
            # report IDs unique so downstream logs/results never collide.
            new_report["report_id"] = f"{original_report_id}__dup{dup_index:02d}of{factor}"
            new_entry["report"] = new_report
            new_entry["actual_timing_duplicate_index"] = dup_index
            new_entry["actual_timing_duplicate_factor"] = factor
            duplicated.append(new_entry)

    stats["duplicated_reports_added"] = len(duplicated) - len(report_entries)
    stats["output_reports"] = len(duplicated)
    return duplicated, stats


def stream_incident_folders(
    batch_roots: Sequence[str | Path],
    *,
    observations_filename: str = "observations.txt",
    recursive_discovery: bool = False,
    interval_seconds: float = 1.0,
    sleep_between_reports: bool = True,
    print_reports: bool = False,
    write_reports_to_incident_folders: bool = True,
    reports_output_filename: str = "reports.jsonl",
    assume_same_depth: bool = True,
    socket_host: Optional[str] = None,
    socket_port: Optional[int] = None,
    wait_for_socket_ack: bool = True,
    display_ground_truth: bool = False,
    ground_truth_filename_template: Optional[str] = None,
    ground_truth_cell_km: float = 0.25,
    ground_truth_layer_prefix: str = "ground_truth",
    ground_truth_color: str = "orange",
    ground_truth_color_intensity: int = 30,
    ground_truth_visible: bool = True,
    display_sensor_locations: bool = False,
    sensor_locations_layer_name: str = "sensor_observations",
    sensor_locations_cell_km: float = 0.25,
    sensor_locations_color: str = "green",
    sensor_locations_color_intensity: int = 100,
    sensor_locations_visible: bool = True,
    prompt_between_reports: bool = False,
    show_progress: bool = True,
    skip_if_ground_truth_outside_grid: bool = False,
    ground_truth_grid_bounds: Optional[Dict[str, Any]] = None,
    sync_multilevel_incidents: bool = False,
    multi_level_sort_mode: str = "time_then_step",
    sensor_density_fraction: float = 1.0,
    missing_observation_fraction: float = 0.0,
    corrupt_observation_fraction: float = 0.0,
    perturbation_seed: str = "0",
    modality_filter: str = "all",
    duplicate_factor: int = 1,
) -> int:
    """Discover and stream incident folders over one socket connection.

    In normal mode, each observations.txt folder is streamed independently.
    In synchronized multi-level mode, a parent folder containing several child
    incident folders is streamed as one incident_start / incident_end block;
    child reports are merged and sorted by timestamp/step so simulated
    subincidents advance together instead of one child completing before the
    next child starts.
    """
    incident_groups = discover_incident_groups(
        batch_roots,
        observations_filename=observations_filename,
        recursive=recursive_discovery,
        sync_multilevel_incidents=sync_multilevel_incidents,
    )
    total_run_folders = sum(len(group.get("run_folders", [])) for group in incident_groups)
    log(
        f"Discovered {len(incident_groups)} stream group(s) "
        f"covering {total_run_folders} incident folder(s)."
    )

    socket_reader = None
    socket_writer = None
    socket_obj = None
    total_reports = 0

    try:
        if socket_host is not None and socket_port is not None:
            socket_obj, socket_reader, socket_writer = open_jsonl_socket(socket_host, socket_port)
        else:
            log("Socket output is disabled; emitting locally only.")

        progress_iter = incident_groups
        progress_bar = None
        if show_progress and tqdm is not None:
            progress_bar = tqdm(incident_groups, total=len(incident_groups), unit="group", desc="Incident groups")
            progress_iter = progress_bar
        elif show_progress:
            log("tqdm is not installed; using plain progress logs. Install with `pip install tqdm` for ETA.")

        for run_index, group in enumerate(progress_iter, start=1):
            incident_start_time = time.monotonic()
            group_folder = Path(group["group_folder"])
            run_folders = [Path(p) for p in group.get("run_folders", [])]
            is_multi_level_group = bool(group.get("multi_level")) and len(run_folders) > 1
            incident_name = group_folder.name
            batch_root_name = group_folder.parent.name if group_folder.parent.name else "batch"
            observations_path = group_folder / observations_filename

            try:
                group_folder_resolved = str(group_folder.resolve())
            except Exception:
                group_folder_resolved = str(group_folder)
            try:
                observations_file_resolved = str(observations_path.resolve())
            except Exception:
                observations_file_resolved = str(observations_path)

            if progress_bar is not None:
                label = "multi-level" if is_multi_level_group else "incident"
                progress_bar.set_description(f"{label} {run_index}/{len(incident_groups)}: {incident_name}")
                progress_bar.set_postfix_str("starting")
            log(
                f"[{run_index}/{len(incident_groups)}] Starting "
                f"{'multi-level incident group' if is_multi_level_group else 'incident folder'}: {group_folder}"
            )

            if skip_if_ground_truth_outside_grid:
                should_skip, skip_summary = should_skip_group_for_ground_truth_grid(
                    run_folders,
                    grid_bounds=ground_truth_grid_bounds,
                )
                if should_skip:
                    if progress_bar is not None:
                        progress_bar.set_postfix_str("skipped: GT outside grid")
                    log(
                        f"SKIPPING incident group {group_folder}: "
                        f"{skip_summary.get('reason')} "
                        f"detail={json.dumps(skip_summary, ensure_ascii=False, sort_keys=True)}"
                    )
                    continue
                log(
                    f"Ground-truth grid check passed for {group_folder}: "
                    f"checked_subincident_folders={skip_summary.get('checked_subincident_folders')}"
                )

            report_entries = collect_group_report_entries(
                run_folders,
                observations_filename=observations_filename,
                assume_same_depth=assume_same_depth,
                multi_level_sort_mode=multi_level_sort_mode,
            )
            report_entries, modality_filter_summary = filter_report_entries_by_modality(
                report_entries,
                modality_filter=modality_filter,
            )
            if normalize_source_name(modality_filter) not in {"", "all", "none", "full"}:
                log(
                    f"Modality filter for {incident_name}: "
                    f"{json.dumps(modality_filter_summary, ensure_ascii=False, sort_keys=True)}"
                )
            report_entries, perturbation_summary = apply_stream_perturbations(
                report_entries,
                sensor_density_fraction=sensor_density_fraction,
                missing_observation_fraction=missing_observation_fraction,
                corrupt_observation_fraction=corrupt_observation_fraction,
                perturbation_seed=f"{perturbation_seed}|{incident_name}",
            )
            perturbation_summary["modality_filter"] = modality_filter_summary
            report_entries, duplication_summary = duplicate_report_entries(
                report_entries,
                duplicate_factor=duplicate_factor,
            )
            perturbation_summary["duplication_summary"] = duplication_summary
            if any([
                float(sensor_density_fraction) < 1.0,
                float(missing_observation_fraction) > 0.0,
                float(corrupt_observation_fraction) > 0.0,
                int(duplicate_factor or 1) > 1,
            ]):
                log(
                    f"Synthetic stress/duplication for {incident_name}: "
                    f"{json.dumps(perturbation_summary, ensure_ascii=False, sort_keys=True)}"
                )

            if socket_writer is not None:
                send_control_message(
                    socket_writer,
                    socket_reader,
                    message_type="incident_start",
                    wait_for_socket_ack=wait_for_socket_ack,
                    run_index=run_index,
                    num_runs=len(incident_groups),
                    incident_name=incident_name,
                    batch_root_name=batch_root_name,
                    incident_folder=str(group_folder),
                    incident_folder_resolved=group_folder_resolved,
                    observations_filename=observations_filename,
                    observations_file=str(observations_path),
                    observations_file_resolved=observations_file_resolved,
                    multi_level_incident=is_multi_level_group,
                    multi_level_sort_mode=multi_level_sort_mode if is_multi_level_group else None,
                    subincident_run_names=[p.name for p in run_folders] if is_multi_level_group else None,
                    subincident_run_folders=[str(p) for p in run_folders] if is_multi_level_group else None,
                    subincident_run_folders_resolved=[str(p.resolve()) for p in run_folders] if is_multi_level_group else None,
                    perturbation_summary=perturbation_summary,
                    modality_filter_summary=modality_filter_summary,
                    duplicate_factor=int(duplicate_factor or 1),
                )

            output_handle = None
            if write_reports_to_incident_folders:
                output_path = group_folder / reports_output_filename
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_handle = output_path.open("w", encoding="utf-8")

            report_count = 0
            geo_manager = None
            active_ground_truth_layers_by_source: Dict[str, list[str]] = {}
            last_ground_truth_key = object()

            try:
                for entry in report_entries:
                    report = entry["report"]
                    source_folder = Path(entry["source_folder"])
                    metadata = report.setdefault("metadata", {})
                    if isinstance(metadata, dict):
                        metadata["batch_run_index"] = run_index
                        metadata["batch_num_runs"] = len(incident_groups)
                        metadata["batch_root_name"] = batch_root_name
                        metadata["incident_run_name"] = incident_name
                        metadata["incident_run_folder"] = str(group_folder)
                        metadata["incident_run_folder_resolved"] = group_folder_resolved
                        metadata["multi_level_incident"] = is_multi_level_group
                        if is_multi_level_group:
                            metadata["multi_level_sort_mode"] = multi_level_sort_mode
                            metadata["parent_incident_name"] = incident_name
                            metadata["parent_incident_folder"] = str(group_folder)
                            metadata["parent_incident_folder_resolved"] = group_folder_resolved

                    if display_ground_truth:
                        step = extract_report_step(report)
                        gt_key = (str(source_folder), step)
                        if step is not None and gt_key != last_ground_truth_key:
                            incident_id = extract_report_incident_id(report)
                            source_key = str(source_folder)
                            layer_prefix = (
                                f"{ground_truth_layer_prefix}_{source_folder.name}"
                                if is_multi_level_group
                                else ground_truth_layer_prefix
                            )
                            try:
                                geo_manager, active_layers = display_ground_truth_for_step(
                                    source_folder,
                                    step,
                                    incident_id=incident_id,
                                    geo_manager=geo_manager,
                                    ground_truth_folder=source_folder,
                                    filename_template=ground_truth_filename_template,
                                    layer_prefix=layer_prefix,
                                    cell_km=ground_truth_cell_km,
                                    color=ground_truth_color,
                                    color_intensity=ground_truth_color_intensity,
                                    visible=ground_truth_visible,
                                    previous_layer_names=active_ground_truth_layers_by_source.get(source_key, []),
                                )
                                active_ground_truth_layers_by_source[source_key] = active_layers
                            except Exception as exc:
                                log(f"WARNING: failed to display ground truth for step {step!r}: {exc}")
                            last_ground_truth_key = gt_key

                    if display_sensor_locations:
                        try:
                            display_sensor_observation_location(
                                report,
                                layer_name=sensor_locations_layer_name,
                                cell_km=sensor_locations_cell_km,
                                color=sensor_locations_color,
                                color_intensity=sensor_locations_color_intensity,
                                visible=sensor_locations_visible,
                            )
                        except Exception as exc:
                            log(
                                f"WARNING: failed to display sensor location for "
                                f"report {report.get('report_id')!r}: {exc}"
                            )

                    line = json.dumps(report, ensure_ascii=False)
                    if print_reports:
                        print(line, flush=True)
                    if output_handle is not None:
                        output_handle.write(line + "\n")
                        output_handle.flush()
                    if socket_writer is not None:
                        socket_writer.write(line + "\n")
                        socket_writer.flush()
                        if wait_for_socket_ack:
                            read_ack(socket_reader, report_id=report.get("report_id"))

                    report_count += 1
                    total_reports += 1
                    if progress_bar is not None:
                        progress_bar.set_postfix(reports=report_count, total_reports=total_reports)

                    if sleep_between_reports and interval_seconds > 0:
                        time.sleep(interval_seconds)
                    if prompt_between_reports:
                        input("Press Enter to emit the next report...")
            finally:
                if output_handle is not None:
                    output_handle.close()

            if socket_writer is not None:
                send_control_message(
                    socket_writer,
                    socket_reader,
                    message_type="incident_end",
                    wait_for_socket_ack=wait_for_socket_ack,
                    run_index=run_index,
                    num_runs=len(incident_groups),
                    incident_name=incident_name,
                    batch_root_name=batch_root_name,
                    incident_folder=str(group_folder),
                    incident_folder_resolved=group_folder_resolved,
                    reports_sent=report_count,
                    duplicate_factor=int(duplicate_factor or 1),
                    multi_level_incident=is_multi_level_group,
                    subincident_run_names=[p.name for p in run_folders] if is_multi_level_group else None,
                )

            elapsed = max(0.0, time.monotonic() - incident_start_time)
            if progress_bar is not None:
                progress_bar.set_postfix(reports=report_count, seconds=f"{elapsed:.1f}")
            log(
                f"Finished {incident_name}: emitted {report_count} report(s) "
                f"from {len(run_folders)} subfolder(s) in {elapsed:.1f}s. "
                f"Progress {run_index}/{len(incident_groups)}."
            )

    finally:
        if socket_writer is not None:
            socket_writer.close()
        if socket_reader is not None:
            socket_reader.close()
        if socket_obj is not None:
            socket_obj.close()

    return total_reports


def parse_cli_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse optional CLI overrides for development settings.

    The file still keeps editable defaults in ``main()``, but command-line
    values can override the batch root without editing this script.
    """
    parser = argparse.ArgumentParser(
        description="Emit normalized synthetic simulator reports to full_pipeline.py."
    )
    parser.add_argument(
        "--batch-root",
        "--batch-roots",
        dest="batch_roots",
        nargs="+",
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "One or more batch/incident root folders to stream. "
            "May be repeated. Overrides the BATCH_ROOTS list in main()."
        ),
    )
    parser.add_argument(
        "batch_root_positional",
        nargs="*",
        metavar="PATH",
        help=(
            "Optional positional batch/incident root folder(s). These are also "
            "used only when provided and override the BATCH_ROOTS list in main()."
        ),
    )
    parser.add_argument(
        "--recursive-discovery",
        dest="recursive_discovery",
        action="store_true",
        default=None,
        help="Recursively discover observations.txt folders under the batch root(s).",
    )
    parser.add_argument(
        "--no-recursive-discovery",
        dest="recursive_discovery",
        action="store_false",
        help="Disable recursive discovery, overriding the default in main().",
    )
    parser.add_argument(
        "--sync-multilevel-incidents",
        dest="sync_multilevel_incidents",
        action="store_true",
        default=None,
        help="Stream child incident folders under each root as one synchronized multi-level incident.",
    )
    parser.add_argument(
        "--no-sync-multilevel-incidents",
        dest="sync_multilevel_incidents",
        action="store_false",
        help="Disable synchronized multi-level streaming, overriding the default in main().",
    )
    parser.add_argument(
        "--multi-level-sort-mode",
        choices=["time_then_step", "step_then_time"],
        default=None,
        help="Ordering mode when synchronized multi-level streaming is enabled.",
    )
    parser.add_argument(
        "--sensor-density",
        type=float,
        default=None,
        help="Synthetic stress: keep this fraction of sensors deterministically, e.g. 1.0, 0.5, 0.25.",
    )
    parser.add_argument(
        "--missing-observations",
        type=float,
        default=None,
        help="Synthetic stress: drop this fraction of individual observations deterministically.",
    )
    parser.add_argument(
        "--corrupt-observations",
        type=float,
        default=None,
        help="Synthetic stress: mark this fraction of observations for downstream label corruption.",
    )
    parser.add_argument(
        "--perturbation-seed",
        default=None,
        help="Seed for deterministic sensor-density/missing/corrupt perturbations.",
    )
    parser.add_argument(
        "--modality-filter",
        choices=["all", "sensor_only", "operational_text_only"],
        default=None,
        help=(
            "Filter emitted reports for modality ablations. "
            "sensor_only keeps camera/traffic/air-quality/weather; "
            "operational_text_only keeps CitizenApp/X/official-alert text and excludes news/article label text."
        ),
    )
    parser.add_argument(
        "--duplicate-reports",
        type=int,
        default=None,
        help=(
            "Controlled timing/scalability: emit each post-filter, post-perturbation "
            "report this many times with unique report IDs. Default 1 preserves the stream."
        ),
    )

    parser.add_argument(
        "--emit-to-socket",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Send reports to full_pipeline over the JSONL socket. Use --no-emit-to-socket to generate local JSONL only.",
    )
    parser.add_argument(
        "--wait-for-socket-ack",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Wait for full_pipeline ACKs after each socket report.",
    )
    parser.add_argument(
        "--write-reports-to-incident-folders",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write normalized reports.jsonl files into each discovered incident/group folder.",
    )
    parser.add_argument(
        "--reports-output-filename",
        default=None,
        help="Filename used when writing normalized reports into incident/group folders. Default: reports.jsonl.",
    )
    parser.add_argument(
        "--print-reports",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print normalized reports to stdout.",
    )
    parser.add_argument(
        "--display-ground-truth",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Display simulator ground-truth regions while streaming.",
    )
    parser.add_argument(
        "--display-sensor-locations",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Display each emitted sensor/report location on the visualization layer.",
    )
    parser.add_argument(
        "--skip-if-ground-truth-outside-grid",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip incident folders whose GT region extends outside the configured grid.",
    )
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show tqdm progress while generating reports.",
    )
    return parser.parse_args(argv)


def _flatten_cli_paths(value: Optional[Sequence[Sequence[str]]]) -> list[str]:
    paths: list[str] = []
    for group in value or []:
        paths.extend(str(item) for item in group)
    return paths

def main() -> int:
    cli_args = parse_cli_args()

    # ---------------------------------------------------------------------
    # Development settings: edit these values directly, or override selected
    # values from the command line.
    # ---------------------------------------------------------------------

    # Batch roots whose immediate children are incident/run folders containing
    # observations.txt. A root may also itself be one incident folder.
    # BATCH_ROOTS = [
    #     Path("simulator/generated/batch_small_area"),
    #     Path("simulator/generated/batch_incident_runs"),
    # ]

    BATCH_ROOTS = [
        Path("simulator/generated/batch_incident_runs/wildfire3"),
    ]

    cli_batch_roots = _flatten_cli_paths(cli_args.batch_roots) + list(cli_args.batch_root_positional or [])
    if cli_batch_roots:
        BATCH_ROOTS = [Path(root) for root in cli_batch_roots]
        log("Using batch root(s) from command line: " + ", ".join(str(root) for root in BATCH_ROOTS))

    RECURSIVE_DISCOVERY = False
    if cli_args.recursive_discovery is not None:
        RECURSIVE_DISCOVERY = bool(cli_args.recursive_discovery)

    # The observations file inside each incident folder.
    OBSERVATIONS_FILENAME = "observations.txt"

    # Optionally write each normalized report stream back into its incident folder.
    WRITE_REPORTS_TO_INCIDENT_FOLDERS = True
    if cli_args.write_reports_to_incident_folders is not None:
        WRITE_REPORTS_TO_INCIDENT_FOLDERS = bool(cli_args.write_reports_to_incident_folders)
    REPORTS_OUTPUT_FILENAME = "reports.jsonl"
    if cli_args.reports_output_filename is not None:
        REPORTS_OUTPUT_FILENAME = str(cli_args.reports_output_filename)

    # Emit one report every N seconds when SLEEP_BETWEEN_REPORTS is True.
    INTERVAL_SECONDS = 1.0
    SLEEP_BETWEEN_REPORTS = False
    PROMPT_BETWEEN_REPORTS = False
    SHOW_PROGRESS = True
    if cli_args.progress_bar is not None:
        SHOW_PROGRESS = bool(cli_args.progress_bar)

    # If a batch root is a multi-level incident folder whose child folders each
    # contain observations.txt, stream those children as one synchronized run.
    # Reports are merged in chronological order by default; set to
    # "step_then_time" if you want all child step-1 reports before any step-2
    # reports regardless of absolute timestamps.
    SYNC_MULTI_LEVEL_INCIDENTS = True
    MULTI_LEVEL_SORT_MODE = "time_then_step"
    if cli_args.sync_multilevel_incidents is not None:
        SYNC_MULTI_LEVEL_INCIDENTS = bool(cli_args.sync_multilevel_incidents)
    if cli_args.multi_level_sort_mode is not None:
        MULTI_LEVEL_SORT_MODE = cli_args.multi_level_sort_mode

    # Print reports to stdout as they are emitted. Usually False when using a socket.
    PRINT_REPORTS = False
    if cli_args.print_reports is not None:
        PRINT_REPORTS = bool(cli_args.print_reports)

    # Optional socket output. Start full_pipeline.py first, then set this to True.
    SEND_TO_SOCKET = True
    if cli_args.emit_to_socket is not None:
        SEND_TO_SOCKET = bool(cli_args.emit_to_socket)
    SOCKET_HOST = "127.0.0.1"
    SOCKET_PORT = 8765

    # Backpressure: wait for full_pipeline.py to ACK after observation model +
    # hypothesis proposal finish before sending the next report/control message.
    WAIT_FOR_SOCKET_ACK = True
    if cli_args.wait_for_socket_ack is not None:
        WAIT_FOR_SOCKET_ACK = bool(cli_args.wait_for_socket_ack)

    # True means any referenced path like simulator/generated/test_folder/foo.csv
    # is rewritten as INCIDENT_FOLDER / foo.csv, which matches the flat folder layout.
    ASSUME_DATA_FILES_ARE_SAME_DEPTH = True

    # Display the ground-truth region for the current simulator step. The emitter
    # loads a GT file only when the step changes within each incident folder.
    DISPLAY_GROUND_TRUTH = True
    if cli_args.display_ground_truth is not None:
        DISPLAY_GROUND_TRUTH = bool(cli_args.display_ground_truth)
    GROUND_TRUTH_FILENAME_TEMPLATE = None
    GROUND_TRUTH_CELL_KM = 0.25
    GROUND_TRUTH_LAYER_PREFIX = "ground_truth"
    GROUND_TRUTH_COLOR = "orange"
    GROUND_TRUTH_COLOR_INTENSITY = 30

    # Skip an entire incident folder if any part of any serialized GT region
    # extends outside the detector/visualization grid. This avoids evaluating
    # incidents whose ground truth cannot be represented by the configured grid.
    SKIP_IF_GROUND_TRUTH_OUTSIDE_GRID = True
    if cli_args.skip_if_ground_truth_outside_grid is not None:
        SKIP_IF_GROUND_TRUTH_OUTSIDE_GRID = bool(cli_args.skip_if_ground_truth_outside_grid)
    GROUND_TRUTH_GRID_BOUNDS = {
        "lat_min": 33.9,
        "lon_min": -118.95,
        "lat_max": 34.35,
        "lon_max": -118.0,
        "cell_km": 0.25,
    }


    # Synthetic stress-test controls.  These are deterministic for a fixed seed
    # and are intended for runtime/scalability experiments.
    SENSOR_DENSITY_FRACTION = 1.0
    MISSING_OBSERVATION_FRACTION = 0.0
    CORRUPT_OBSERVATION_FRACTION = 0.0
    PERTURBATION_SEED = "0"
    if cli_args.sensor_density is not None:
        SENSOR_DENSITY_FRACTION = float(cli_args.sensor_density)
    if cli_args.missing_observations is not None:
        MISSING_OBSERVATION_FRACTION = float(cli_args.missing_observations)
    if cli_args.corrupt_observations is not None:
        CORRUPT_OBSERVATION_FRACTION = float(cli_args.corrupt_observations)
    if cli_args.perturbation_seed is not None:
        PERTURBATION_SEED = str(cli_args.perturbation_seed)

    MODALITY_FILTER = "all"
    if cli_args.modality_filter is not None:
        MODALITY_FILTER = str(cli_args.modality_filter)

    DUPLICATE_REPORTS = 1
    if cli_args.duplicate_reports is not None:
        DUPLICATE_REPORTS = max(1, int(cli_args.duplicate_reports))

    # Display each sensor observation location as a green cell.
    DISPLAY_SENSOR_LOCATIONS = True
    if cli_args.display_sensor_locations is not None:
        DISPLAY_SENSOR_LOCATIONS = bool(cli_args.display_sensor_locations)
    SENSOR_LOCATIONS_LAYER_NAME = "sensor_observations"
    SENSOR_LOCATIONS_CELL_KM = 0.25
    SENSOR_LOCATIONS_COLOR = "green"
    SENSOR_LOCATIONS_COLOR_INTENSITY = 100

    report_count = stream_incident_folders(
        BATCH_ROOTS,
        observations_filename=OBSERVATIONS_FILENAME,
        recursive_discovery=RECURSIVE_DISCOVERY,
        interval_seconds=INTERVAL_SECONDS,
        sleep_between_reports=SLEEP_BETWEEN_REPORTS,
        print_reports=PRINT_REPORTS,
        write_reports_to_incident_folders=WRITE_REPORTS_TO_INCIDENT_FOLDERS,
        reports_output_filename=REPORTS_OUTPUT_FILENAME,
        assume_same_depth=ASSUME_DATA_FILES_ARE_SAME_DEPTH,
        socket_host=SOCKET_HOST if SEND_TO_SOCKET else None,
        socket_port=SOCKET_PORT if SEND_TO_SOCKET else None,
        wait_for_socket_ack=WAIT_FOR_SOCKET_ACK,
        display_ground_truth=DISPLAY_GROUND_TRUTH,
        ground_truth_filename_template=GROUND_TRUTH_FILENAME_TEMPLATE,
        ground_truth_cell_km=GROUND_TRUTH_CELL_KM,
        ground_truth_layer_prefix=GROUND_TRUTH_LAYER_PREFIX,
        ground_truth_color=GROUND_TRUTH_COLOR,
        ground_truth_color_intensity=GROUND_TRUTH_COLOR_INTENSITY,
        display_sensor_locations=DISPLAY_SENSOR_LOCATIONS,
        sensor_locations_layer_name=SENSOR_LOCATIONS_LAYER_NAME,
        sensor_locations_cell_km=SENSOR_LOCATIONS_CELL_KM,
        sensor_locations_color=SENSOR_LOCATIONS_COLOR,
        sensor_locations_color_intensity=SENSOR_LOCATIONS_COLOR_INTENSITY,
        prompt_between_reports=PROMPT_BETWEEN_REPORTS,
        show_progress=SHOW_PROGRESS,
        skip_if_ground_truth_outside_grid=SKIP_IF_GROUND_TRUTH_OUTSIDE_GRID,
        ground_truth_grid_bounds=GROUND_TRUTH_GRID_BOUNDS,
        sync_multilevel_incidents=SYNC_MULTI_LEVEL_INCIDENTS,
        multi_level_sort_mode=MULTI_LEVEL_SORT_MODE,
        sensor_density_fraction=SENSOR_DENSITY_FRACTION,
        missing_observation_fraction=MISSING_OBSERVATION_FRACTION,
        corrupt_observation_fraction=CORRUPT_OBSERVATION_FRACTION,
        perturbation_seed=PERTURBATION_SEED,
        modality_filter=MODALITY_FILTER,
        duplicate_factor=DUPLICATE_REPORTS,
    )

    log(f"Emitted {report_count} reports across batch roots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
