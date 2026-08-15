#!/usr/bin/env python3
"""Evaluate IncidentLens/baseline result folders against synthetic ground truth.

Expected result layout
----------------------
The evaluator assumes result folders are organized as:

    evaluation/results/<method_name>/<experiment_name>/<incident_or_date>/
      low_level_results.json
      high_level_results.json
      timing.json
      experiment_complete.json

For synthetic experiments, <incident_or_date> should match a simulator incident
folder such as wildfire1 or earthquake_damage9.  The ground-truth folder should
contain one or more *_gt_*.json files.

Main modes
----------
1. low_incident
   Evaluates low-level synthetic incident runs such as:
       evaluation/results/incidentlens/synth_low/earthquake_damage1/

2. ablation
   Evaluates architectural/modality ablation folders under IncidentLens, such as:
       evaluation/results/incidentlens/ablation_scalability_ablation_*/<incident>/
       evaluation/results/incidentlens/modality_ablation_*/<incident>/

3. scalability
   Evaluates scalability folders under results/<method>/, including:
       ablation_scalability_scalability_*/<incident>/   # performance + no-observation timing
       timing_scalability_*/<run>/                      # timing, usually with observation model

   The mode writes both system-performance summaries and timing summaries.
   Runtime summaries are split by observation-model condition so the same
   scalability setting can be compared with and without the observation model.

4. throughput
   Evaluates controlled actual_timing duplication folders such as:
       evaluation/results/incidentlens/actual_timing_dup1/wildfire1/
       evaluation/results/incidentlens/actual_timing_dup2/wildfire1/
       evaluation/results/incidentlens/actual_timing_dup3/wildfire1/
       evaluation/results/incidentlens/actual_timing_dup5/wildfire1/

   This mode is timing-only. It uses whole-incident wall-clock timing from
   timing.json when available and reports throughput by duplication level.

5. synth_composition
   Evaluates high-level incident composition on synth_comp positives and
   synth_low negatives. IncidentLens is read from high_level_results.json;
   heuristic baselines use low_level_results_denoised.json when present; otherwise
   the evaluator applies the same low-level denoising/leakage filter in memory
   before calling the baseline functions.

6. low_real
   Evaluates low-level real-data incident predictions organized by date folders,
   for example evaluation/results/incidentlens/real_all_incidents/20250110/.
   Real GT is loaded from evaluation/ground_truth/real/low_level_gt_corrected.json and is
   matched at dataset level because real incidents can span multiple date folders.

7. ablation_and_scalability
   Backward-compatible mode that evaluates both ablation and scalability folders.
   For scalability-style experiments, it can also summarize runtime components
   from timing.json and, when present, pipeline_timings.jsonl.

Outputs
-------
By default outputs are written to evaluation/evaluation_summary/<mode>/:

    run_metrics.csv/json
    aggregate_metrics.json
    aggregate_metrics_by_method_experiment.csv
    aggregate_metrics_by_ablation.csv          # only in --mode ablation
    aggregate_metrics_by_incident_type.csv
    aggregate_metrics_by_predicted_incident_type.csv
    aggregate_metrics_by_gt_incident_type.csv/json
    aggregate_metrics_by_gt_incident_name.csv/json  # only in --mode low_real
    scalability_runtime_summary.csv/json
    type_confusion_matrix.csv

Notes
-----
- Spatial IoU is computed twice: once using hypothesis_source_points and once
  using propagated_coverage_points.
- Those point sets are converted into a convex hull of approximate grid-cell
  squares.  This is intentionally conservative/filling, matching the "trace a
  line around outer cells and fill the middle" interpretation.
- Type precision/recall/F1/F2 use one-to-one matched predictions and GT records.
  A wrong-type match counts as a false positive for the predicted type and a
  false negative for the true type.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

try:
    from shapely.geometry import Point, Polygon, shape
    from shapely.geometry.base import BaseGeometry
    from shapely import wkt
    from shapely.ops import transform, unary_union
except Exception as exc:  # pragma: no cover
    raise RuntimeError("evaluate_results.py requires shapely. Install with `pip install shapely`.") from exc

try:
    try:
        from coverage_metrics import (
            compute_coverage_for_gt_items,
            compute_coverage_weighted_metrics,
            coverage_by_gt_id as coverage_scores_by_gt_id,
            load_source_records_from_jsonl,
            load_source_records_from_real_emitter_profile_cache,
            load_source_records_from_result_dirs,
            summarize_coverage_rows,
            modality_coverage as coverage_modality_coverage,
            inference_coverage as coverage_inference_coverage,
            canonical_incident_type as coverage_canonical_incident_type,
            SOURCE_TYPE_TO_MODALITY as COVERAGE_SOURCE_TYPE_TO_MODALITY,
            MODALITY_TO_WEAK_EVIDENCE as COVERAGE_MODALITY_TO_WEAK_EVIDENCE,
        )
    except Exception:
        from evaluation.coverage_metrics import (
            compute_coverage_for_gt_items,
            compute_coverage_weighted_metrics,
            coverage_by_gt_id as coverage_scores_by_gt_id,
            load_source_records_from_jsonl,
            load_source_records_from_real_emitter_profile_cache,
            load_source_records_from_result_dirs,
            summarize_coverage_rows,
            modality_coverage as coverage_modality_coverage,
            inference_coverage as coverage_inference_coverage,
            canonical_incident_type as coverage_canonical_incident_type,
            SOURCE_TYPE_TO_MODALITY as COVERAGE_SOURCE_TYPE_TO_MODALITY,
            MODALITY_TO_WEAK_EVIDENCE as COVERAGE_MODALITY_TO_WEAK_EVIDENCE,
        )
except Exception:  # pragma: no cover - coverage scoring is optional.
    compute_coverage_for_gt_items = None  # type: ignore
    compute_coverage_weighted_metrics = None  # type: ignore
    coverage_scores_by_gt_id = None  # type: ignore
    load_source_records_from_jsonl = None  # type: ignore
    load_source_records_from_real_emitter_profile_cache = None  # type: ignore
    load_source_records_from_result_dirs = None  # type: ignore
    summarize_coverage_rows = None  # type: ignore
    coverage_modality_coverage = None  # type: ignore
    coverage_inference_coverage = None  # type: ignore
    coverage_canonical_incident_type = None  # type: ignore
    COVERAGE_SOURCE_TYPE_TO_MODALITY = {}  # type: ignore
    COVERAGE_MODALITY_TO_WEAK_EVIDENCE = {}  # type: ignore


LOW_LEVEL_FILENAME = "low_level_results.json"
DENOISED_LOW_LEVEL_FILENAME = "low_level_results_denoised.json"
HIGH_LEVEL_FILENAME = "high_level_results.json"
TIMING_FILENAME = "timing.json"
COMPLETION_FILENAME = "experiment_complete.json"

# Architectural/modality ablations are now stored as experiment folders under
# evaluation/results/incidentlens/, for example:
#   evaluation/results/incidentlens/ablation_scalability_ablation_generic_propagation/<incident>/
#   evaluation/results/incidentlens/ablation_scalability_ablation_generic_clustering/<incident>/
#   evaluation/results/incidentlens/modality_ablation_sensor_only/<incident>/
#   evaluation/results/incidentlens/modality_ablation_operational_text_only/<incident>/
#
# Scalability/stress experiments may share the ablation_scalability_ prefix but
# do not contain the architectural/modality ablation marker.  Keep these modes
# separate so ablation tables do not accidentally include stress/scalability runs.
ABLATION_EXPERIMENT_PREFIXES = (
    "ablation_scalability_ablation_",
    "modality_ablation_",
)

ABLATION_AND_SCALABILITY_EXPERIMENT_PREFIXES = (
    "ablation_scalability_",
    "modality_ablation_",
    "timing_scalability_",
)

# Scalability performance folders contain low_level_results.json and timing.json
# from runs where expensive observation-model calls are usually disabled.
# Timing scalability folders are dedicated timing sweeps and are treated as the
# with-observation-model timing condition unless the name explicitly says
# otherwise.
SCALABILITY_PERFORMANCE_PREFIXES = (
    "ablation_scalability_scalability_",
)
SCALABILITY_TIMING_PREFIXES = (
    "timing_scalability_",
)

# Controlled end-to-end throughput experiments. These folders are produced by
# run_experiments.py --exp-type actual_timing and are timing-only:
#   evaluation/results/incidentlens/actual_timing_dup1/wildfire1/
#   evaluation/results/incidentlens/actual_timing_dup2/wildfire1/
#   evaluation/results/incidentlens/actual_timing_dup3/wildfire1/
#   evaluation/results/incidentlens/actual_timing_dup5/wildfire1/
ACTUAL_TIMING_EXPERIMENT_PREFIXES = (
    "actual_timing_dup",
)

# Default low-level synthetic method suite. Used by the convenience CLI flags
# --synth-low-all-baselines and --synth-low-background-baselines-only.
SYNTH_LOW_DEFAULT_BASELINES = (
    "incidentlens",
    "direct_observation",
    "space_time_clustering",
    "text_only_clustering",
    "late_fusion_voting",
    "hotspot_scan",
    "generic_propagation",
    "generic_all",
    "satscan_background",
    "hawkes_event_detector",
)
SYNTH_LOW_BACKGROUND_BASELINES = (
    "satscan_background",
    "hawkes_event_detector",
)

REAL_LOW_DEFAULT_METHODS = (
    "direct_observation",
    "hotspot_scan",
    "incidentlens",
    "late_fusion_voting",
    "space_time_clustering",
    "text_only_clustering",
    # Real-background baselines recently added under real_all_incidents.
    "satscan_background",
    "hawkes_event_detector",
)
REAL_LOW_EXCLUDED_METHODS = {"generic_propagation"}


DEFAULT_ALIASES = {
    "civil protest": "large civil protest",
    "demonstration": "large civil protest",
    "protest": "large civil protest",
    "terrorist incident": "terrorist attack",
    "terrorism": "terrorist attack",
    "earthquake_damage": "earthquake damage",
    "earthquake damage": "earthquake damage",
    "storm damage": "severe storm damage",
    "severe storm": "severe storm damage",
    "urban_fire": "urban fire",
    "hazmat": "hazardous material release",
    "hazardous material": "hazardous material release",
}

# Baseline methods sometimes emit a generic predicted type of "fire" even
# when the synthetic ground-truth ontology uses "urban fire" for non-wildfire
# fire incidents.  Keep IncidentLens and generic_propagation unchanged because
# their outputs are already part of the architecture/ablation comparison, but
# normalize this exact baseline label before one-to-one GT comparison.
BASELINE_FIRE_TO_URBAN_FIRE_EXCLUDED_METHODS = {"incidentlens", "generic_propagation"}


@dataclass
class GroundTruthIncident:
    run_name: str
    incident_id: str
    location_name: str
    incident_type: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    geometry: Optional[BaseGeometry]
    representative_latitude: Optional[float] = None
    representative_longitude: Optional[float] = None
    external_report_time: Optional[datetime] = None
    gt_files: List[str] = field(default_factory=list)


@dataclass
class PredictedIncident:
    run_name: str
    prediction_id: str
    incident_type: str
    detection_time: Optional[datetime]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    source_geometry: Optional[BaseGeometry]
    coverage_geometry: Optional[BaseGeometry]
    confidence: Optional[float] = None
    state: Optional[str] = None


@dataclass
class MatchRecord:
    run_name: str
    method: str
    experiment: str
    gt_type: str
    pred_type: str
    gt_id: str
    pred_id: str
    type_correct: bool
    source_iou: Optional[float]
    coverage_iou: Optional[float]
    best_iou: Optional[float]
    source_loc_error_km: Optional[float]
    coverage_loc_error_km: Optional[float]
    temporal_iou: Optional[float]
    detection_delay_minutes: Optional[float]
    detected_before_external_report: Optional[bool]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def progress_iter(
    iterable: Iterable[Any],
    *,
    total: Optional[int],
    desc: str,
    enabled: bool,
    unit: str = "run",
    leave: bool = True,
    position: Optional[int] = None,
    mininterval: float = 0.5,
) -> Iterable[Any]:
    """Wrap an iterable in tqdm when available and enabled."""
    if enabled and tqdm is not None:
        kwargs = {
            "total": total,
            "desc": desc,
            "unit": unit,
            "leave": leave,
            "dynamic_ncols": True,
            "mininterval": mininterval,
        }
        if position is not None:
            kwargs["position"] = position
        return tqdm(iterable, **kwargs)
    return iterable


def progress_write(message: str, *, enabled: bool) -> None:
    """Write progress-aware status messages without corrupting tqdm bars."""
    if enabled and tqdm is not None:
        tqdm.write(message, file=sys.stderr)
    else:
        log(message)


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def parse_dt(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        # Common compact date, used rarely.
        if re.fullmatch(r"\d{8}", text):
            try:
                dt = datetime.strptime(text, "%Y%m%d")
            except ValueError:
                return None
        else:
            try:
                dt = datetime.fromisoformat(text)
            except ValueError:
                # Try to pull an ISO-looking timestamp from a longer string.
                m = re.search(r"\d{4}-\d{2}-\d{2}[T ][0-9:.+-]+Z?", text)
                if not m:
                    return None
                return parse_dt(m.group(0))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def dt_to_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def norm_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_allowed_types(path: Optional[str | Path]) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    out: List[str] = []
    with p.open("r", encoding="utf-8") as infile:
        for line in infile:
            text = line.strip()
            if text and not text.startswith("#"):
                out.append(norm_label(text))
    return out


def canonical_type(value: Any, allowed_types: Sequence[str] = ()) -> str:
    raw = norm_label(value)
    if not raw:
        return "unknown"
    raw = DEFAULT_ALIASES.get(raw, raw)
    allowed = [norm_label(x) for x in allowed_types if x]
    if not allowed:
        return raw
    if raw in allowed:
        return raw
    if raw in DEFAULT_ALIASES and DEFAULT_ALIASES[raw] in allowed:
        return DEFAULT_ALIASES[raw]

    # Prefer containment, e.g. "civil protest" -> "large civil protest".
    for item in allowed:
        if raw and (raw in item or item in raw):
            return item

    # Small token-overlap fallback.
    raw_tokens = set(raw.split())
    best = None
    best_score = 0.0
    for item in allowed:
        item_tokens = set(item.split())
        if not raw_tokens or not item_tokens:
            continue
        score = len(raw_tokens & item_tokens) / len(raw_tokens | item_tokens)
        if score > best_score:
            best_score = score
            best = item
    return best if best_score >= 0.5 and best is not None else raw


def canonical_prediction_type(value: Any, method: str, allowed_types: Sequence[str] = ()) -> str:
    """Canonicalize a predicted incident type, with baseline-only fire remapping.

    For non-IncidentLens/non-generic-propagation methods, an exact predicted
    label of "fire" should be evaluated as "urban fire" rather than being
    captured by the generic containment fallback in canonical_type, which may map
    "fire" to "wildfire" depending on allowed-type ordering.
    """
    raw = norm_label(value)
    method_key = norm_label(method)
    if raw == "fire" and method_key not in BASELINE_FIRE_TO_URBAN_FIRE_EXCLUDED_METHODS:
        return canonical_type("urban fire", allowed_types)
    return canonical_type(value, allowed_types)


def infer_gt_type(gt_data: Dict[str, Any], gt_path: Path, run_name: str, allowed_types: Sequence[str]) -> str:
    for key in ("incident_type", "type", "event_type"):
        if gt_data.get(key):
            return canonical_type(gt_data.get(key), allowed_types)

    incident_id = str(gt_data.get("incident_id") or "")
    # earthquake_damage_61f1862a_incident_0 -> earthquake_damage
    m = re.match(r"(.+?)_[0-9a-fA-F]{6,}_(?:incident|WF|LA|RUN|EV)", incident_id)
    if m:
        return canonical_type(m.group(1), allowed_types)
    m = re.match(r"(.+?)_incident_\d+", incident_id)
    if m:
        return canonical_type(m.group(1), allowed_types)

    # wildfire29 -> wildfire; earthquake_damage9 -> earthquake_damage
    m = re.match(r"(.+?)(?:\d+)$", run_name)
    if m:
        return canonical_type(m.group(1), allowed_types)

    # fallback from file name prefix
    name = gt_path.name
    m = re.match(r"(.+?)_[0-9a-fA-F]{6,}_", name)
    if m:
        return canonical_type(m.group(1), allowed_types)

    return canonical_type(run_name, allowed_types)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as infile:
        value = json.load(infile)
    return value if isinstance(value, dict) else {}


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def geometry_centroid_latlon(geom: Optional[BaseGeometry]) -> Optional[Tuple[float, float]]:
    if geom is None or geom.is_empty:
        return None
    c = geom.centroid
    return (float(c.y), float(c.x))


def representative_distance_km(a: Optional[BaseGeometry], b: Optional[BaseGeometry]) -> Optional[float]:
    ca = geometry_centroid_latlon(a)
    cb = geometry_centroid_latlon(b)
    if ca is None or cb is None:
        return None
    return haversine_km(ca[0], ca[1], cb[0], cb[1])


def local_projector(geoms: Sequence[Optional[BaseGeometry]]):
    coords: List[Tuple[float, float]] = []
    for geom in geoms:
        if geom is None or geom.is_empty:
            continue
        c = geom.centroid
        coords.append((float(c.x), float(c.y)))
    if coords:
        lon0 = sum(x for x, _ in coords) / len(coords)
        lat0 = sum(y for _, y in coords) / len(coords)
    else:
        lon0, lat0 = -118.25, 34.05
    cos_lat = max(0.05, math.cos(math.radians(lat0)))

    def project_xy(lon: float, lat: float, z: Optional[float] = None):
        return ((lon - lon0) * 111.320 * cos_lat, (lat - lat0) * 110.574)

    return project_xy


def geom_area_km2(geom: Optional[BaseGeometry], ref: Optional[BaseGeometry] = None) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    project = local_projector([geom, ref])
    try:
        return max(0.0, float(transform(project, geom).area))
    except Exception:
        return 0.0


def geom_iou(a: Optional[BaseGeometry], b: Optional[BaseGeometry]) -> Optional[float]:
    if a is None or b is None or a.is_empty or b.is_empty:
        return None
    try:
        if not a.is_valid:
            a = a.buffer(0)
        if not b.is_valid:
            b = b.buffer(0)
        project = local_projector([a, b])
        pa = transform(project, a)
        pb = transform(project, b)
        inter = pa.intersection(pb).area
        union = pa.union(pb).area
        if union <= 0:
            return 0.0
        return float(max(0.0, min(1.0, inter / union)))
    except Exception:
        return None


def cell_polygon_from_point(lat: float, lon: float, cell_side_km: float) -> Polygon:
    half = max(0.025, cell_side_km / 2.0)
    dlat = half / 110.574
    dlon = half / max(1e-6, 111.320 * math.cos(math.radians(lat)))
    return Polygon([
        (lon - dlon, lat - dlat),
        (lon + dlon, lat - dlat),
        (lon + dlon, lat + dlat),
        (lon - dlon, lat + dlat),
    ])


def circle_polygon_from_point(lat: float, lon: float, radius_km: float, *, steps: int = 32) -> Polygon:
    """Approximate a geodesic point buffer in lon/lat coordinates.

    This is used for baseline outputs such as direct_observation, whose
    affected_region is stored as a point-buffer summary rather than grid cells.
    """
    radius_km = max(0.001, float(radius_km))
    coords: List[Tuple[float, float]] = []
    for idx in range(max(12, int(steps))):
        angle = 2.0 * math.pi * idx / max(12, int(steps))
        dlat = (radius_km * math.sin(angle)) / 110.574
        dlon = (radius_km * math.cos(angle)) / max(1e-6, 111.320 * math.cos(math.radians(lat)))
        coords.append((lon + dlon, lat + dlat))
    return Polygon(coords)


def affected_region_to_geometry(affected: Any, *, default_radius_km: float = 1.0) -> Optional[BaseGeometry]:
    """Reconstruct geometry from the canonical baseline affected_region field.

    Baselines may emit:
      affected_region = {
        "kind": "point_buffer_union_summary",
        "radius_km": 1.0,
        "center": {"latitude": ..., "longitude": ...},
        "support_points": [{"latitude": ..., "longitude": ...}, ...]
      }

    Earlier evaluator versions only read IncidentLens-specific
    hypothesis_source_points / propagated_coverage_points, so all these
    baseline predictions had no geometry and could never spatially match GT.
    """
    if not isinstance(affected, dict):
        return None

    explicit_cells = affected.get("cells") or affected.get("grid_cells")
    if isinstance(explicit_cells, list) and explicit_cells:
        geom = points_to_hull_region(
            explicit_cells,
            area_info=affected.get("area_coverage") if isinstance(affected.get("area_coverage"), dict) else None,
        )
        if geom is not None:
            return geom

    radius_km = safe_float(affected.get("radius_km"), default_radius_km)
    if radius_km is None:
        radius_km = default_radius_km

    support_points = affected.get("support_points")
    polygons: List[BaseGeometry] = []
    if isinstance(support_points, list):
        for point in support_points:
            if not isinstance(point, dict):
                continue
            lat = safe_float(point.get("latitude", point.get("lat")), None)
            lon = safe_float(point.get("longitude", point.get("lon")), None)
            if lat is None or lon is None:
                continue
            polygons.append(circle_polygon_from_point(lat, lon, radius_km))

    if polygons:
        try:
            union = unary_union(polygons)
            if affected.get("reconstruct_hull_from_points") and len(polygons) > 1:
                union = union.convex_hull
            return union.buffer(0) if not union.is_valid else union
        except Exception:
            pass

    center = affected.get("center") if isinstance(affected.get("center"), dict) else {}
    lat = safe_float(center.get("latitude", center.get("lat")), None)
    lon = safe_float(center.get("longitude", center.get("lon")), None)
    if lat is not None and lon is not None:
        return circle_polygon_from_point(lat, lon, radius_km)

    return None


def points_to_hull_region(
    points: Any,
    *,
    area_info: Optional[Dict[str, Any]] = None,
    default_cell_side_km: float = 0.25,
    min_coverage_normalized: float = 0.0,
) -> Optional[BaseGeometry]:
    if not isinstance(points, list) or not points:
        return None

    cell_area = safe_float((area_info or {}).get("cell_area_km2"), None)
    cell_side_km = math.sqrt(cell_area) if cell_area and cell_area > 0 else default_cell_side_km

    polys: List[BaseGeometry] = []
    raw_points: List[Point] = []
    for item in points:
        if not isinstance(item, dict):
            continue
        cov = safe_float(item.get("coverage_normalized", item.get("coverage", item.get("mass", 1.0))), 1.0)
        if cov is not None and cov < min_coverage_normalized:
            continue
        lat = safe_float(item.get("latitude", item.get("lat")), None)
        lon = safe_float(item.get("longitude", item.get("lon")), None)
        if lat is None or lon is None:
            continue
        raw_points.append(Point(lon, lat))
        polys.append(cell_polygon_from_point(lat, lon, cell_side_km))

    if not polys and not raw_points:
        return None
    try:
        if polys:
            geom = unary_union(polys)
        else:
            geom = unary_union(raw_points)
        if geom.is_empty:
            return None
        # Fill the region enclosed by outer cells, per user's requested hull behavior.
        hull = geom.convex_hull
        if hull.geom_type in {"Point", "LineString"}:
            hull = geom.buffer((cell_side_km / 2.0) / 111.0)
        return hull.buffer(0) if not hull.is_valid else hull
    except Exception:
        return None


def gt_location_geometry(location: Dict[str, Any]) -> Optional[BaseGeometry]:
    geojson = location.get("geometry_geojson")
    if isinstance(geojson, dict):
        try:
            geom = shape(geojson)
            if geom.is_valid:
                return geom
            return geom.buffer(0)
        except Exception:
            pass

    lat = safe_float(location.get("representative_latitude", location.get("latitude")), None)
    lon = safe_float(location.get("representative_longitude", location.get("longitude")), None)
    if lat is None or lon is None:
        return None

    # Fallback: a small 1-km circle if only a representative point is present.
    side = 1.0
    return cell_polygon_from_point(lat, lon, side).buffer(0)


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def build_gt_index(gt_roots: Sequence[str | Path]) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for root_value in gt_roots:
        root = Path(root_value)
        if not root.exists():
            continue
        for folder in root.rglob("*"):
            if not folder.is_dir():
                continue
            try:
                has_gt = any(p.name.endswith(".json") and "_gt_" in p.name for p in folder.iterdir())
            except Exception:
                has_gt = False
            if has_gt:
                index.setdefault(folder.name, []).append(folder)
    return index


def load_ground_truth_for_run(
    run_name: str,
    gt_index: Dict[str, List[Path]],
    *,
    allowed_types: Sequence[str],
    default_single_step_minutes: float = 5.0,
) -> List[GroundTruthIncident]:
    folders = gt_index.get(run_name, [])
    if not folders:
        return []
    # Prefer the first exact folder. If duplicates exist, they usually correspond
    # to different simulator roots; evaluating code can restrict --gt-roots.
    folder = sorted(folders, key=lambda p: str(p))[0]

    grouped: Dict[Tuple[str, str], GroundTruthIncident] = {}
    for path in sorted(folder.glob("*_gt_*.json")):
        try:
            data = read_json(path)
        except Exception:
            continue
        step_time = parse_dt(data.get("time") or data.get("timestamp") or data.get("report_time"))
        incident_id = str(data.get("incident_id") or path.stem)
        incident_type = infer_gt_type(data, path, run_name, allowed_types)
        external = first_external_report_time(data)
        locations = data.get("locations")
        if isinstance(locations, dict):
            items = list(locations.items())
        else:
            items = [(data.get("location") or run_name, data)]
        for location_name, loc in items:
            if not isinstance(loc, dict):
                continue
            key = (incident_id, str(location_name))
            geom = gt_location_geometry(loc)
            lat = safe_float(loc.get("representative_latitude", loc.get("latitude")), None)
            lon = safe_float(loc.get("representative_longitude", loc.get("longitude")), None)
            existing = grouped.get(key)
            if existing is None:
                grouped[key] = GroundTruthIncident(
                    run_name=run_name,
                    incident_id=incident_id,
                    location_name=str(location_name),
                    incident_type=incident_type,
                    start_time=step_time,
                    end_time=step_time,
                    geometry=geom,
                    representative_latitude=lat,
                    representative_longitude=lon,
                    external_report_time=external,
                    gt_files=[str(path)],
                )
            else:
                if step_time is not None:
                    if existing.start_time is None or step_time < existing.start_time:
                        existing.start_time = step_time
                    if existing.end_time is None or step_time > existing.end_time:
                        existing.end_time = step_time
                if geom is not None:
                    existing.geometry = geom if existing.geometry is None else unary_union([existing.geometry, geom]).buffer(0)
                if existing.external_report_time is None or (external is not None and external < existing.external_report_time):
                    existing.external_report_time = external
                existing.gt_files.append(str(path))

    out = list(grouped.values())
    for gt in out:
        if gt.start_time is not None and gt.end_time is not None and gt.end_time <= gt.start_time:
            gt.end_time = gt.start_time + timedelta(minutes=float(default_single_step_minutes))
    return out


def first_external_report_time(data: Dict[str, Any]) -> Optional[datetime]:
    candidate_keys = [
        "external_report_time",
        "external_report_datetime",
        "earliest_article_datetime",
        "earliest_article_datetime_pacific",
        "ground_truth_report_time",
        "report_time",
    ]
    for key in candidate_keys:
        dt = parse_dt(data.get(key))
        if dt is not None:
            return dt
    # Also look one level down for future real-data GT formats.
    for value in data.values():
        if isinstance(value, dict):
            for key in candidate_keys:
                dt = parse_dt(value.get(key))
                if dt is not None:
                    return dt
    return None


# ---------------------------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------------------------

def pred_interval(pred: Dict[str, Any]) -> Tuple[Optional[datetime], Optional[datetime]]:
    interval = pred.get("active_interval") if isinstance(pred.get("active_interval"), dict) else {}
    start = parse_dt(interval.get("start") or pred.get("start_time") or pred.get("time_first_hypothesis"))
    end = parse_dt(interval.get("end") or pred.get("end_time") or pred.get("time_last_observed") or pred.get("time_last_updated"))
    return start, end


def prediction_identity_keys(item: Dict[str, Any]) -> List[str]:
    """Stable IDs used to detect cumulative leakage across result folders.

    Some IncidentLens/generic_propagation result folders were written with the
    pipeline state still carrying low-level predictions from previous incident
    folders.  Those stale predictions keep the same prediction_id and/or
    candidate_ids when they appear in later folders.  The evaluator can repair
    this by treating the first result folder where such an identity appears as
    its owner and dropping later copies.

    The keys intentionally use fields that should be unique to a prediction or
    candidate cluster, not incident type/time/geometry, because different
    incidents can legitimately share type, location scale, or time patterns.
    """
    keys: List[str] = []

    for field in ("prediction_id", "pred_id", "baseline_internal_cluster_id", "baseline_internal_vote_window_id", "baseline_internal_hotspot_id"):
        value = item.get(field)
        if value is not None and str(value).strip():
            keys.append(f"{field}:{str(value).strip()}")

    candidate_ids = item.get("candidate_ids")
    if isinstance(candidate_ids, list):
        cleaned = sorted(str(x).strip() for x in candidate_ids if str(x).strip())
        if cleaned:
            keys.append("candidate_ids_tuple:" + "|".join(cleaned))
            for cid in cleaned:
                keys.append("candidate_id:" + cid)

    # Some variants store support/candidate IDs inside summaries.
    summary = item.get("supporting_evidence_summary") if isinstance(item.get("supporting_evidence_summary"), dict) else {}
    hyp_info = summary.get("hypothesis_info") if isinstance(summary.get("hypothesis_info"), dict) else {}
    member_ids = hyp_info.get("member_particle_ids")
    if isinstance(member_ids, list):
        cleaned = sorted(str(x).strip() for x in member_ids if str(x).strip())
        if cleaned:
            # Use a tuple key only. Individual particle IDs can be numerous and
            # are less stable than prediction/candidate IDs.
            keys.append("member_particle_ids_tuple:" + "|".join(cleaned[:200]))

    return list(dict.fromkeys(keys))


def build_prediction_first_owner_index(
    result_dirs: Sequence[Tuple[str, str, str, Path]],
) -> Dict[Tuple[str, str], Dict[str, str]]:
    """Return first owner of every prediction identity within method/experiment.

    Output shape:
      {(method, experiment): {identity_key: canonical_result_dir_string}}

    The result_dirs list is already sorted by discover_result_dirs.  That sorted
    order mirrors the experiment iteration order where the cumulative-state bug
    occurred, for example earthquake1 then earthquake10 then earthquake11, etc.
    """
    first_owner: Dict[Tuple[str, str], Dict[str, str]] = {}
    for method, experiment, run_name, result_dir in result_dirs:
        path = result_dir / LOW_LEVEL_FILENAME
        if not path.exists():
            continue
        try:
            data = read_json(path)
        except Exception:
            continue
        rows = data.get("low_level_incidents", [])
        if not isinstance(rows, list):
            continue

        group = first_owner.setdefault((method, experiment), {})
        owner = str(result_dir.resolve())
        for item in rows:
            if not isinstance(item, dict):
                continue
            for key in prediction_identity_keys(item):
                group.setdefault(key, owner)
    return first_owner


def filter_leaked_prediction_rows(
    rows: List[Dict[str, Any]],
    *,
    method: str,
    experiment: str,
    result_dir: Path,
    first_owner_index: Optional[Dict[Tuple[str, str], Dict[str, str]]] = None,
    enabled: bool = True,
    filter_methods: Sequence[str] = (),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Drop low-level predictions that first appeared in an earlier run folder."""
    if not enabled or not rows:
        return rows, {
            "enabled": bool(enabled),
            "raw_predictions": len(rows),
            "kept_predictions": len(rows),
            "dropped_predictions": 0,
            "dropped_prediction_ids": [],
        }

    if filter_methods and method not in set(filter_methods):
        return rows, {
            "enabled": False,
            "reason": "method_not_in_filter_methods",
            "raw_predictions": len(rows),
            "kept_predictions": len(rows),
            "dropped_predictions": 0,
            "dropped_prediction_ids": [],
        }

    owner_map = (first_owner_index or {}).get((method, experiment), {})
    if not owner_map:
        return rows, {
            "enabled": True,
            "reason": "no_owner_index",
            "raw_predictions": len(rows),
            "kept_predictions": len(rows),
            "dropped_predictions": 0,
            "dropped_prediction_ids": [],
        }

    current_owner = str(result_dir.resolve())
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []

    for item in rows:
        keys = prediction_identity_keys(item)
        prior_keys = [key for key in keys if owner_map.get(key) not in (None, current_owner)]

        # A prediction is stale if its stable prediction ID is already owned by
        # another result folder, or if any candidate_id has already appeared in
        # another result folder.  candidate_ids are cluster identities and should
        # not be shared across independent synthetic incident folders.
        stale_by_prediction_id = any(
            key.startswith(("prediction_id:", "pred_id:", "baseline_internal_"))
            for key in prior_keys
        )
        stale_by_candidate_id = any(
            key.startswith(("candidate_id:", "candidate_ids_tuple:"))
            for key in prior_keys
        )

        if stale_by_prediction_id or stale_by_candidate_id:
            dropped.append(item)
        else:
            kept.append(item)

    summary = {
        "enabled": True,
        "raw_predictions": len(rows),
        "kept_predictions": len(kept),
        "dropped_predictions": len(dropped),
        "dropped_prediction_ids": [
            str(item.get("prediction_id") or item.get("pred_id") or "") for item in dropped[:200]
        ],
    }
    return kept, summary



def should_write_denoised_low_level(
    *,
    method: str,
    enabled: bool,
    denoised_methods: Sequence[str] = (),
) -> bool:
    """Return whether this result directory should get a denoised copy."""
    if not enabled:
        return False
    if not denoised_methods:
        return False
    method_key = norm_label(method)
    wanted = {norm_label(x) for x in denoised_methods if str(x).strip()}
    return method_key in wanted


def write_denoised_low_level_results(
    *,
    result_dir: Path,
    original_payload: Dict[str, Any],
    kept_rows: List[Dict[str, Any]],
    leakage_summary: Dict[str, Any],
    method: str,
    experiment: str,
    run_name: str,
) -> Dict[str, Any]:
    """Write low_level_results_denoised.json next to low_level_results.json.

    The original low_level_results.json is never modified.  The denoised file
    preserves the original JSON payload but replaces low_level_incidents with the
    rows that survived the existing cumulative-leakage filter.  A small metadata
    block is added so the file is self-describing.
    """
    output_path = result_dir / DENOISED_LOW_LEVEL_FILENAME
    payload = dict(original_payload)
    payload["low_level_incidents"] = kept_rows
    payload["denoising_metadata"] = {
        "source_file": LOW_LEVEL_FILENAME,
        "method": method,
        "experiment": experiment,
        "run_name": run_name,
        "raw_predictions": leakage_summary.get("raw_predictions"),
        "kept_predictions": leakage_summary.get("kept_predictions"),
        "dropped_predictions": leakage_summary.get("dropped_predictions"),
        "dropped_prediction_ids_sample": leakage_summary.get("dropped_prediction_ids", []),
        "filter": "cumulative_prediction_identity_first_owner",
        "note": (
            "This file was generated by evaluate_results.py for convenience. "
            "The original low_level_results.json was not modified."
        ),
    }
    write_json(output_path, payload)
    return {
        "denoised_output_path": str(output_path),
        "denoised_written": True,
    }




REAL_RATIONALE_TYPE_GUESS_PATTERNS = (
    ("civil protest", ("protest", "protests", "civil unrest", "demonstration", "demonstrations")),
    ("urban fire", ("urban fire", "urban structural fire", "structural fire", "building fire", "vehicle fire")),
    ("wildfire", ("wildfire", "wild fire", "brush fire", "wildland fire")),
    ("terrorist incident", ("terrorist", "terrorism", "terror attack", "terrorist attack")),
)


def rationale_text_for_type_guesses(item: Dict[str, Any]) -> str:
    """Return the free-text rationale used for real-data alternate type guesses."""
    parts: List[str] = []
    for key in ("rationale", "reasoning", "explanation"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    return "\n".join(parts).lower()


def real_rationale_type_guesses(item: Dict[str, Any], allowed_types: Sequence[str] = ()) -> List[str]:
    """Infer alternate incident-type labels from a prediction rationale.

    Real-data IncidentLens records sometimes discuss plausible competing labels
    in the rationale even when the top-level incident_type field contains only
    one label.  For low_real evaluation we optionally expose those mentions as
    alternate labels at the same space/time footprint.  The matching code treats
    these as type alternatives for the same source prediction rather than as
    independent extra prediction records for overall FP accounting.
    """
    text = rationale_text_for_type_guesses(item)
    if not text:
        return []
    out: List[str] = []
    for label, patterns in REAL_RATIONALE_TYPE_GUESS_PATTERNS:
        if any(pattern in text for pattern in patterns):
            out.append(canonical_prediction_type(label, "incidentlens", allowed_types))
    return list(dict.fromkeys(out))


def real_prediction_group_id(pred: PredictedIncident) -> str:
    """Stable unit for real-data scoring when one prediction has type alternatives."""
    value = getattr(pred, "source_prediction_id", None) or pred.prediction_id
    return str(value)


def real_prediction_is_rationale_type_guess(pred: PredictedIncident) -> bool:
    return bool(getattr(pred, "rationale_type_guess", False))


def real_primary_prediction_by_group(pred_items: Sequence[PredictedIncident]) -> Dict[str, PredictedIncident]:
    grouped: Dict[str, List[PredictedIncident]] = {}
    for pred in pred_items:
        grouped.setdefault(real_prediction_group_id(pred), []).append(pred)
    out: Dict[str, PredictedIncident] = {}
    for gid, rows in grouped.items():
        primary = next((p for p in rows if not real_prediction_is_rationale_type_guess(p)), rows[0])
        out[gid] = primary
    return out

def load_predictions_for_result_dir(
    result_dir: Path,
    *,
    run_name: str,
    allowed_types: Sequence[str],
    min_coverage_normalized: float,
    method: str = "",
    experiment: str = "",
    first_owner_index: Optional[Dict[Tuple[str, str], Dict[str, str]]] = None,
    leakage_filter_enabled: bool = True,
    leakage_filter_methods: Sequence[str] = (),
    write_denoised_low_level: bool = False,
    denoised_methods: Sequence[str] = (),
    prefer_denoised_low_level: bool = False,
    include_rationale_type_guesses: bool = False,
) -> Tuple[List[PredictedIncident], Dict[str, Any]]:
    raw_path = result_dir / LOW_LEVEL_FILENAME
    denoised_path = result_dir / DENOISED_LOW_LEVEL_FILENAME
    using_denoised = bool(prefer_denoised_low_level and denoised_path.exists())
    path = denoised_path if using_denoised else raw_path
    if not path.exists():
        return [], {
            "enabled": leakage_filter_enabled,
            "raw_predictions": 0,
            "kept_predictions": 0,
            "dropped_predictions": 0,
            "source_file_used": path.name,
            "missing_source_file": True,
        }
    try:
        data = read_json(path)
    except Exception:
        return [], {
            "enabled": leakage_filter_enabled,
            "raw_predictions": 0,
            "kept_predictions": 0,
            "dropped_predictions": 0,
            "read_error": True,
            "source_file_used": path.name,
        }
    rows = data.get("low_level_incidents", [])
    if not isinstance(rows, list):
        return [], {
            "enabled": leakage_filter_enabled,
            "raw_predictions": 0,
            "kept_predictions": 0,
            "dropped_predictions": 0,
            "malformed": True,
            "source_file_used": path.name,
        }

    if using_denoised:
        meta = data.get("denoising_metadata") if isinstance(data.get("denoising_metadata"), dict) else {}
        leakage_summary = {
            "enabled": False,
            "reason": "used_materialized_denoised_low_level_results",
            "source_file_used": DENOISED_LOW_LEVEL_FILENAME,
            "raw_predictions": meta.get("raw_predictions", len(rows)),
            "kept_predictions": len(rows),
            "dropped_predictions": meta.get("dropped_predictions", 0),
            "dropped_prediction_ids": meta.get("dropped_prediction_ids_sample", []),
        }
    else:
        rows, leakage_summary = filter_leaked_prediction_rows(
            rows,
            method=method,
            experiment=experiment,
            result_dir=result_dir,
            first_owner_index=first_owner_index,
            enabled=leakage_filter_enabled,
            filter_methods=leakage_filter_methods,
        )
        leakage_summary["source_file_used"] = LOW_LEVEL_FILENAME

        if should_write_denoised_low_level(
            method=method,
            enabled=write_denoised_low_level,
            denoised_methods=denoised_methods,
        ):
            try:
                denoise_write_summary = write_denoised_low_level_results(
                    result_dir=result_dir,
                    original_payload=data,
                    kept_rows=rows,
                    leakage_summary=leakage_summary,
                    method=method,
                    experiment=experiment,
                    run_name=run_name,
                )
                leakage_summary.update(denoise_write_summary)
            except Exception as exc:
                leakage_summary.update({
                    "denoised_written": False,
                    "denoised_write_error": str(exc),
                })

    out: List[PredictedIncident] = []
    for idx, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        pred_id = str(item.get("prediction_id") or f"{result_dir.name}_prediction_{idx}")
        incident_type = canonical_prediction_type(item.get("incident_type"), method, allowed_types)
        detection_time = parse_dt(item.get("time_first_incident_predicted") or item.get("time_last_updated") or item.get("updated_at"))
        state_times = item.get("state_times") if isinstance(item.get("state_times"), dict) else {}
        if detection_time is None and state_times:
            # Real-data outputs often store the first system-side incident state
            # transition here rather than in time_first_incident_predicted.
            # Use the earliest state transition as the system report proxy.
            parsed_state_times = [parse_dt(v) for v in state_times.values()]
            parsed_state_times = [dt for dt in parsed_state_times if dt is not None]
            if parsed_state_times:
                detection_time = min(parsed_state_times)
        start, end = pred_interval(item)
        source_geom = points_to_hull_region(
            item.get("hypothesis_source_points"),
            area_info=item.get("hypothesis_source_area_coverage") if isinstance(item.get("hypothesis_source_area_coverage"), dict) else None,
            min_coverage_normalized=min_coverage_normalized,
        )
        cov_geom = points_to_hull_region(
            item.get("propagated_coverage_points"),
            area_info=item.get("propagated_area_coverage") if isinstance(item.get("propagated_area_coverage"), dict) else None,
            min_coverage_normalized=min_coverage_normalized,
        )

        # Baseline/canonical predictions usually store their region in
        # affected_region rather than IncidentLens-specific propagated points.
        affected = item.get("affected_region") if isinstance(item.get("affected_region"), dict) else {}
        affected_geom = affected_region_to_geometry(affected)
        if cov_geom is None:
            cov_geom = affected_geom

        # Direct-observation and other observation-stream baselines do not have
        # a separate source hypothesis set.  Use the affected-region center/
        # support points as a source-location proxy so source localization error
        # is still meaningful instead of null.
        if source_geom is None:
            source_geom = affected_geom

        base_pred = PredictedIncident(
            run_name=run_name,
            prediction_id=pred_id,
            incident_type=incident_type,
            detection_time=detection_time,
            start_time=start,
            end_time=end,
            source_geometry=source_geom,
            coverage_geometry=cov_geom,
            confidence=safe_float(item.get("confidence"), None),
            state=str(item.get("state")) if item.get("state") is not None else None,
        )
        setattr(base_pred, "source_prediction_id", pred_id)
        setattr(base_pred, "rationale_type_guess", False)
        setattr(base_pred, "source_file_used", path.name)
        out.append(base_pred)

        if include_rationale_type_guesses:
            for guess_type in real_rationale_type_guesses(item, allowed_types):
                if guess_type == incident_type:
                    continue
                guess_pred = PredictedIncident(
                    run_name=run_name,
                    prediction_id=f"{pred_id}::rationale_type::{guess_type.replace(' ', '_')}",
                    incident_type=guess_type,
                    detection_time=detection_time,
                    start_time=start,
                    end_time=end,
                    source_geometry=source_geom,
                    coverage_geometry=cov_geom,
                    confidence=safe_float(item.get("confidence"), None),
                    state=str(item.get("state")) if item.get("state") is not None else None,
                )
                setattr(guess_pred, "source_prediction_id", pred_id)
                setattr(guess_pred, "rationale_type_guess", True)
                setattr(guess_pred, "source_file_used", path.name)
                out.append(guess_pred)
    return out, leakage_summary


# ---------------------------------------------------------------------------
# Matching and metrics
# ---------------------------------------------------------------------------

def temporal_iou(
    gt_start: Optional[datetime],
    gt_end: Optional[datetime],
    pred_start: Optional[datetime],
    pred_end: Optional[datetime],
) -> Optional[float]:
    if gt_start is None or gt_end is None or pred_start is None or pred_end is None:
        return None
    if gt_end <= gt_start or pred_end <= pred_start:
        return 0.0
    inter_start = max(gt_start, pred_start)
    inter_end = min(gt_end, pred_end)
    inter = max(0.0, (inter_end - inter_start).total_seconds())
    union_start = min(gt_start, pred_start)
    union_end = max(gt_end, pred_end)
    union = max(0.0, (union_end - union_start).total_seconds())
    return float(inter / union) if union > 0 else 0.0




def union_geometries(geoms: Iterable[Optional[BaseGeometry]]) -> Optional[BaseGeometry]:
    """Return a valid union of non-empty geometries, or None."""
    valid: List[BaseGeometry] = []
    for geom in geoms:
        if geom is None or geom.is_empty:
            continue
        try:
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom is not None and not geom.is_empty:
                valid.append(geom)
        except Exception:
            continue
    if not valid:
        return None
    try:
        out = unary_union(valid)
        if out.is_empty:
            return None
        return out.buffer(0) if not out.is_valid else out
    except Exception:
        return None


def geom_overlap_metrics(pred_geom: Optional[BaseGeometry], gt_geom: Optional[BaseGeometry]) -> Dict[str, Optional[float]]:
    """Spatial union/overcoverage metrics for a predicted region and GT region.

    These metrics are intended for scenario-level evaluation.  Unlike the
    matched-TP IoU/localization metrics, they can include all same-type
    predictions in a run by passing their geometry union as pred_geom.  Extra
    predicted area outside the GT is therefore explicitly penalized.
    """
    pred_empty = pred_geom is None or pred_geom.is_empty
    gt_empty = gt_geom is None or gt_geom.is_empty

    if gt_empty and pred_empty:
        return {
            "scenario_union_coverage_iou": None,
            "scenario_coverage_precision": None,
            "scenario_coverage_recall": None,
            "scenario_pred_region_area_km2": 0.0,
            "scenario_gt_region_area_km2": 0.0,
            "scenario_intersection_area_km2": 0.0,
            "scenario_union_area_km2": 0.0,
            "scenario_extra_area_km2": 0.0,
            "scenario_missing_area_km2": 0.0,
            "scenario_overcoverage_ratio": None,
        }

    if gt_empty:
        pred_area = geom_area_km2(pred_geom)
        return {
            "scenario_union_coverage_iou": 0.0 if pred_area > 0 else None,
            "scenario_coverage_precision": 0.0 if pred_area > 0 else None,
            "scenario_coverage_recall": None,
            "scenario_pred_region_area_km2": pred_area,
            "scenario_gt_region_area_km2": 0.0,
            "scenario_intersection_area_km2": 0.0,
            "scenario_union_area_km2": pred_area,
            "scenario_extra_area_km2": pred_area,
            "scenario_missing_area_km2": 0.0,
            "scenario_overcoverage_ratio": None,
        }

    if pred_empty:
        gt_area = geom_area_km2(gt_geom)
        return {
            "scenario_union_coverage_iou": 0.0 if gt_area > 0 else None,
            "scenario_coverage_precision": None,
            "scenario_coverage_recall": 0.0 if gt_area > 0 else None,
            "scenario_pred_region_area_km2": 0.0,
            "scenario_gt_region_area_km2": gt_area,
            "scenario_intersection_area_km2": 0.0,
            "scenario_union_area_km2": gt_area,
            "scenario_extra_area_km2": 0.0,
            "scenario_missing_area_km2": gt_area,
            "scenario_overcoverage_ratio": 0.0 if gt_area > 0 else None,
        }

    try:
        a = pred_geom.buffer(0) if not pred_geom.is_valid else pred_geom
        b = gt_geom.buffer(0) if not gt_geom.is_valid else gt_geom
        project = local_projector([a, b])
        pa = transform(project, a)
        pb = transform(project, b)
        pred_area = max(0.0, float(pa.area))
        gt_area = max(0.0, float(pb.area))
        inter_area = max(0.0, float(pa.intersection(pb).area))
        union_area = max(0.0, float(pa.union(pb).area))
        extra_area = max(0.0, float(pa.difference(pb).area))
        missing_area = max(0.0, float(pb.difference(pa).area))
        return {
            "scenario_union_coverage_iou": (inter_area / union_area) if union_area > 0 else None,
            "scenario_coverage_precision": (inter_area / pred_area) if pred_area > 0 else None,
            "scenario_coverage_recall": (inter_area / gt_area) if gt_area > 0 else None,
            "scenario_pred_region_area_km2": pred_area,
            "scenario_gt_region_area_km2": gt_area,
            "scenario_intersection_area_km2": inter_area,
            "scenario_union_area_km2": union_area,
            "scenario_extra_area_km2": extra_area,
            "scenario_missing_area_km2": missing_area,
            # Area outside GT normalized by GT area.  This is unbounded: 1.0
            # means the predicted region includes one GT-area worth of extra
            # area outside the ground truth; 0.0 means no extra area.
            "scenario_overcoverage_ratio": (extra_area / gt_area) if gt_area > 0 else None,
        }
    except Exception:
        return {
            "scenario_union_coverage_iou": None,
            "scenario_coverage_precision": None,
            "scenario_coverage_recall": None,
            "scenario_pred_region_area_km2": None,
            "scenario_gt_region_area_km2": None,
            "scenario_intersection_area_km2": None,
            "scenario_union_area_km2": None,
            "scenario_extra_area_km2": None,
            "scenario_missing_area_km2": None,
            "scenario_overcoverage_ratio": None,
        }


def merge_intervals(intervals: Iterable[Tuple[Optional[datetime], Optional[datetime]]]) -> List[Tuple[datetime, datetime]]:
    clean: List[Tuple[datetime, datetime]] = []
    for start, end in intervals:
        if start is None or end is None or end <= start:
            continue
        clean.append((start, end))
    if not clean:
        return []
    clean.sort(key=lambda x: x[0])
    merged = [clean[0]]
    for start, end in clean[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def interval_duration_seconds(intervals: Iterable[Tuple[datetime, datetime]]) -> float:
    return float(sum(max(0.0, (end - start).total_seconds()) for start, end in intervals))


def interval_intersection_seconds(
    a_intervals: Sequence[Tuple[datetime, datetime]],
    b_intervals: Sequence[Tuple[datetime, datetime]],
) -> float:
    i = j = 0
    total = 0.0
    while i < len(a_intervals) and j < len(b_intervals):
        a0, a1 = a_intervals[i]
        b0, b1 = b_intervals[j]
        total += max(0.0, (min(a1, b1) - max(a0, b0)).total_seconds())
        if a1 < b1:
            i += 1
        else:
            j += 1
    return float(total)


def temporal_overlap_metrics(
    pred_intervals: Iterable[Tuple[Optional[datetime], Optional[datetime]]],
    gt_intervals: Iterable[Tuple[Optional[datetime], Optional[datetime]]],
) -> Dict[str, Optional[float]]:
    pred_merged = merge_intervals(pred_intervals)
    gt_merged = merge_intervals(gt_intervals)
    pred_dur = interval_duration_seconds(pred_merged)
    gt_dur = interval_duration_seconds(gt_merged)
    inter = interval_intersection_seconds(pred_merged, gt_merged)
    union = pred_dur + gt_dur - inter
    extra = max(0.0, pred_dur - inter)
    missing = max(0.0, gt_dur - inter)
    return {
        "scenario_temporal_iou": (inter / union) if union > 0 else None,
        "scenario_temporal_precision": (inter / pred_dur) if pred_dur > 0 else None,
        "scenario_temporal_recall": (inter / gt_dur) if gt_dur > 0 else None,
        "scenario_pred_duration_minutes": pred_dur / 60.0,
        "scenario_gt_duration_minutes": gt_dur / 60.0,
        "scenario_temporal_intersection_minutes": inter / 60.0,
        "scenario_temporal_extra_minutes": extra / 60.0,
        "scenario_temporal_missing_minutes": missing / 60.0,
        "scenario_temporal_overcoverage_ratio": (extra / gt_dur) if gt_dur > 0 else None,
    }


def scenario_union_metrics_by_gt_type(
    gt_items: List[GroundTruthIncident],
    pred_items: List[PredictedIncident],
) -> Dict[str, Dict[str, Any]]:
    """Compute scenario-level union IoU/overcoverage for each GT type in a run.

    For each ground-truth type T, this unions all predictions that also predict T
    and compares that union to the union of GT geometries of type T.  This makes
    over-fragmented or shotgun same-type predictions pay an area penalty, unlike
    matched-only IoU metrics that only summarize true-positive matches.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for typ in sorted({g.incident_type for g in gt_items}):
        gt_for_type = [g for g in gt_items if g.incident_type == typ]
        preds_for_type = [p for p in pred_items if p.incident_type == typ]
        gt_geom = union_geometries(g.geometry for g in gt_for_type)
        pred_geom = union_geometries((p.coverage_geometry if p.coverage_geometry is not None else p.source_geometry) for p in preds_for_type)
        spatial = geom_overlap_metrics(pred_geom, gt_geom)
        temporal = temporal_overlap_metrics(
            ((p.start_time, p.end_time) for p in preds_for_type),
            ((g.start_time, g.end_time) for g in gt_for_type),
        )
        missing_geom_predictions = sum(
            1 for p in preds_for_type if p.coverage_geometry is None and p.source_geometry is None
        )
        out[typ] = {
            **spatial,
            **temporal,
            "scenario_same_type_predictions": len(preds_for_type),
            "scenario_same_type_predictions_without_geometry": missing_geom_predictions,
        }
    return out

def evaluate_run(
    *,
    method: str,
    experiment: str,
    run_name: str,
    result_dir: Path,
    gt_items: List[GroundTruthIncident],
    pred_items: List[PredictedIncident],
    max_loc_error_km: float,
    min_spatial_iou: float,
) -> Tuple[Dict[str, Any], List[MatchRecord], Dict[str, Dict[str, int]]]:
    candidates: List[Tuple[float, int, int, Dict[str, Any]]] = []
    for pi, pred in enumerate(pred_items):
        for gi, gt in enumerate(gt_items):
            s_iou = geom_iou(pred.source_geometry, gt.geometry)
            c_iou = geom_iou(pred.coverage_geometry, gt.geometry)
            best_iou = max([x for x in [s_iou, c_iou] if x is not None], default=0.0)
            s_err = representative_distance_km(pred.source_geometry, gt.geometry)
            c_err = representative_distance_km(pred.coverage_geometry, gt.geometry)
            best_err = min([x for x in [s_err, c_err] if x is not None], default=None)
            t_iou = temporal_iou(gt.start_time, gt.end_time, pred.start_time, pred.end_time)
            type_correct = pred.incident_type == gt.incident_type

            spatial_ok = (best_iou is not None and best_iou >= min_spatial_iou) or (
                best_err is not None and best_err <= max_loc_error_km
            )
            if not spatial_ok:
                continue

            loc_score = 0.0
            if best_err is not None:
                loc_score = max(0.0, 1.0 - best_err / max(max_loc_error_km, 1e-6))
            score = (2.0 if type_correct else 0.0) + float(best_iou or 0.0) + loc_score + float(t_iou or 0.0)
            candidates.append((score, pi, gi, {
                "source_iou": s_iou,
                "coverage_iou": c_iou,
                "best_iou": best_iou,
                "source_loc_error_km": s_err,
                "coverage_loc_error_km": c_err,
                "temporal_iou": t_iou,
                "type_correct": type_correct,
            }))

    candidates.sort(key=lambda x: x[0], reverse=True)
    used_preds: set[int] = set()
    used_gts: set[int] = set()
    matches: List[MatchRecord] = []
    confusion: Dict[str, Dict[str, int]] = {}

    for score, pi, gi, metrics in candidates:
        if pi in used_preds or gi in used_gts:
            continue
        used_preds.add(pi)
        used_gts.add(gi)
        pred = pred_items[pi]
        gt = gt_items[gi]
        delay = None
        if pred.detection_time is not None and gt.start_time is not None:
            delay = (pred.detection_time - gt.start_time).total_seconds() / 60.0
        before_external = None
        if pred.detection_time is not None and gt.external_report_time is not None:
            before_external = pred.detection_time <= gt.external_report_time

        confusion.setdefault(gt.incident_type, {})
        confusion[gt.incident_type][pred.incident_type] = int(confusion[gt.incident_type].get(pred.incident_type, 0)) + 1

        matches.append(MatchRecord(
            run_name=run_name,
            method=method,
            experiment=experiment,
            gt_type=gt.incident_type,
            pred_type=pred.incident_type,
            gt_id=f"{gt.incident_id}::{gt.location_name}",
            pred_id=pred.prediction_id,
            type_correct=bool(metrics["type_correct"]),
            source_iou=metrics["source_iou"],
            coverage_iou=metrics["coverage_iou"],
            best_iou=metrics["best_iou"],
            source_loc_error_km=metrics["source_loc_error_km"],
            coverage_loc_error_km=metrics["coverage_loc_error_km"],
            temporal_iou=metrics["temporal_iou"],
            detection_delay_minutes=delay,
            detected_before_external_report=before_external,
        ))

    # Type precision/recall accounting.
    tp = fp = fn = 0
    per_type_counts: Dict[str, Dict[str, int]] = {}
    all_types = {g.incident_type for g in gt_items} | {p.incident_type for p in pred_items}
    for typ in all_types:
        per_type_counts[typ] = {"tp": 0, "fp": 0, "fn": 0}

    for m in matches:
        if m.type_correct:
            tp += 1
            per_type_counts[m.gt_type]["tp"] += 1
        else:
            fp += 1
            fn += 1
            per_type_counts.setdefault(m.pred_type, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
            per_type_counts.setdefault(m.gt_type, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1

    for i, pred in enumerate(pred_items):
        if i not in used_preds:
            fp += 1
            per_type_counts.setdefault(pred.incident_type, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1

    for i, gt in enumerate(gt_items):
        if i not in used_gts:
            fn += 1
            per_type_counts.setdefault(gt.incident_type, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1
            confusion.setdefault(gt.incident_type, {})
            confusion[gt.incident_type]["__missed__"] = int(confusion[gt.incident_type].get("__missed__", 0)) + 1

    for i, pred in enumerate(pred_items):
        if i not in used_preds:
            confusion.setdefault("__false_positive__", {})
            confusion["__false_positive__"][pred.incident_type] = int(confusion["__false_positive__"].get(pred.incident_type, 0)) + 1

    # Ground-truth-scenario-conditioned accounting.  The existing
    # per_type_counts above is class-wise: false positives are charged to the
    # predicted incident type.  For incident-category panels such as
    # "performance on terrorist-attack scenarios," it is often more useful to
    # condition on the GT type of the run and charge every extra prediction in
    # that run as a false positive for that scenario.  In a single-GT-type
    # synthetic run this prevents shotgun baselines from looking good by
    # predicting many labels and getting one of them right.
    gt_type_conditioned_counts: Dict[str, Dict[str, int]] = {}
    for typ in sorted({g.incident_type for g in gt_items}):
        num_gt_for_type = sum(1 for g in gt_items if g.incident_type == typ)
        correct_tp_for_type = sum(1 for m in matches if m.type_correct and m.gt_type == typ)
        gt_type_conditioned_counts[typ] = {
            "tp": correct_tp_for_type,
            # All predictions that are not correct matches for this GT scenario
            # are false positives for the scenario-conditioned metric.
            "fp": max(0, len(pred_items) - correct_tp_for_type),
            "fn": max(0, num_gt_for_type - correct_tp_for_type),
            "num_gt": num_gt_for_type,
            "num_predictions": len(pred_items),
        }

    scenario_union_by_gt_type = scenario_union_metrics_by_gt_type(gt_items, pred_items)

    correct_matches = [m for m in matches if m.type_correct]
    matched_tp_source_errors = [m.source_loc_error_km for m in correct_matches]
    matched_tp_coverage_errors = [m.coverage_loc_error_km for m in correct_matches]
    matched_tp_source_ious = [m.source_iou for m in correct_matches]
    matched_tp_coverage_ious = [m.coverage_iou for m in correct_matches]
    matched_tp_temporal_ious = [m.temporal_iou for m in correct_matches]
    scenario_union_ious = [v.get("scenario_union_coverage_iou") for v in scenario_union_by_gt_type.values()]
    scenario_overcoverage_ratios = [v.get("scenario_overcoverage_ratio") for v in scenario_union_by_gt_type.values()]
    scenario_temporal_ious = [v.get("scenario_temporal_iou") for v in scenario_union_by_gt_type.values()]
    scenario_temporal_overcoverage_ratios = [v.get("scenario_temporal_overcoverage_ratio") for v in scenario_union_by_gt_type.values()]

    run_metrics = {
        "method": method,
        "experiment": experiment,
        "run_name": run_name,
        "result_dir": str(result_dir),
        "num_gt": len(gt_items),
        "num_predictions": len(pred_items),
        "num_matches": len(matches),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "false_positives": fp,
        "false_positives_per_run": fp,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": fbeta(tp, fp, fn, beta=1.0),
        "f2": fbeta(tp, fp, fn, beta=2.0),
        "median_detection_delay_minutes": median([m.detection_delay_minutes for m in correct_matches]),
        "fraction_detected_before_external_report": fraction_true([m.detected_before_external_report for m in correct_matches]),
        # Matched-TP quality metrics. These summarize only correct one-to-one
        # matches and do not penalize unmatched false-positive predictions.
        # Keep legacy names for backward compatibility, and add explicit names
        # for paper tables/figures.
        "median_source_localization_error_km": median(matched_tp_source_errors),
        "median_coverage_localization_error_km": median(matched_tp_coverage_errors),
        "mean_source_iou": mean(matched_tp_source_ious),
        "mean_coverage_iou": mean(matched_tp_coverage_ious),
        "median_temporal_iou": median(matched_tp_temporal_ious),
        "matched_tp_median_source_or_proxy_localization_error_km": median(matched_tp_source_errors),
        "matched_tp_median_coverage_localization_error_km": median(matched_tp_coverage_errors),
        "matched_tp_mean_source_or_proxy_iou": mean(matched_tp_source_ious),
        "matched_tp_mean_coverage_iou": mean(matched_tp_coverage_ious),
        "matched_tp_median_temporal_iou": median(matched_tp_temporal_ious),
        # Scenario-level union metrics. These compare the union of all
        # same-type predictions in a GT scenario to the GT region/interval and
        # therefore penalize spatial/temporal overprediction.
        "mean_scenario_union_coverage_iou": mean(scenario_union_ious),
        "median_scenario_union_coverage_iou": median(scenario_union_ious),
        "mean_scenario_overcoverage_ratio": mean(scenario_overcoverage_ratios),
        "median_scenario_overcoverage_ratio": median(scenario_overcoverage_ratios),
        "mean_scenario_temporal_iou": mean(scenario_temporal_ious),
        "median_scenario_temporal_iou": median(scenario_temporal_ious),
        "mean_scenario_temporal_overcoverage_ratio": mean(scenario_temporal_overcoverage_ratios),
        "median_scenario_temporal_overcoverage_ratio": median(scenario_temporal_overcoverage_ratios),
        "gt_incident_types": ";".join(sorted({g.incident_type for g in gt_items})),
        "pred_incident_types": ";".join(sorted({p.incident_type for p in pred_items})),
        "_per_type_counts": per_type_counts,
        "_gt_type_conditioned_counts": gt_type_conditioned_counts,
        "_scenario_union_metrics_by_gt_type": scenario_union_by_gt_type,
    }
    return run_metrics, matches, confusion


def ratio(num: float, den: float) -> Optional[float]:
    return float(num / den) if den else None


def fbeta(tp: int, fp: int, fn: int, *, beta: float) -> Optional[float]:
    b2 = beta * beta
    den = (1 + b2) * tp + b2 * fn + fp
    return float(((1 + b2) * tp) / den) if den else None


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(sum(vals) / len(vals)) if vals else None


def median(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(statistics.median(vals)) if vals else None


def percentile(values: Iterable[Optional[float]], q: float) -> Optional[float]:
    """Return the q-th percentile using linear interpolation, q in [0, 100]."""
    vals = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0])
    q = max(0.0, min(100.0, float(q)))
    pos = (len(vals) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    weight = pos - lo
    return float(vals[lo] * (1.0 - weight) + vals[hi] * weight)


def fraction_true(values: Iterable[Optional[bool]]) -> Optional[float]:
    vals = [x for x in values if x is not None]
    if not vals:
        return None
    return sum(1 for x in vals if x) / len(vals)


# ---------------------------------------------------------------------------
# Runtime/scalability summarization
# ---------------------------------------------------------------------------

def component_stats(timing: Dict[str, Any], key: str) -> Dict[str, Optional[float]]:
    comp = timing.get("by_component", {}).get(key, {}) if isinstance(timing.get("by_component"), dict) else {}
    if not isinstance(comp, dict):
        comp = {}
    return {
        f"{key}_avg_ms": safe_float(comp.get("avg_ms"), None),
        f"{key}_total_seconds": safe_float(comp.get("total_seconds"), None),
        f"{key}_count": safe_float(comp.get("count"), None),
    }


def find_total_process_component(timing: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    comps = timing.get("by_component") if isinstance(timing.get("by_component"), dict) else {}
    for key in ["total_process_one_report", "total_with_preprocess_visualization", "process_one_report", "total"]:
        comp = comps.get(key)
        if isinstance(comp, dict):
            return comp
    return None


def completion_summary_for_result_dir(result_dir: Path) -> Dict[str, Any]:
    """Read experiment_complete.json metadata when present.

    The completion marker is the most reliable place to recover how many reports
    the emitter actually sent for a run.  timing.json sometimes uses a different
    name (num_observations) or can be missing in older/incomplete runs.
    """
    path = result_dir / COMPLETION_FILENAME
    if not path.exists():
        return {
            "experiment_completed": None,
            "reports_sent": None,
        }
    try:
        payload = read_json(path)
    except Exception:
        return {
            "experiment_completed": None,
            "reports_sent": None,
            "completion_read_error": True,
        }
    scope = payload.get("scope") if isinstance(payload.get("scope"), dict) else {}
    reports_sent = safe_float(payload.get("reports_sent"), None)
    return {
        "experiment_completed": bool(payload.get("completed")) if payload.get("completed") is not None else None,
        "reports_sent": int(reports_sent) if reports_sent is not None else None,
        "completed_at": payload.get("completed_at"),
        "completion_scope_kind": scope.get("kind"),
        "completion_scope_incident_name": scope.get("incident_name"),
        "completion_scope_batch_root_name": scope.get("batch_root_name"),
        "completion_scope_run_index": scope.get("run_index"),
    }


def _prediction_hypothesis_info(item: Dict[str, Any]) -> Dict[str, Any]:
    """Return the nested hypothesis_info block from a low-level prediction."""
    for parent_key in ("supporting_evidence_summary", "supporting_evidence"):
        parent = item.get(parent_key) if isinstance(item.get(parent_key), dict) else {}
        hyp_info = parent.get("hypothesis_info") if isinstance(parent.get("hypothesis_info"), dict) else {}
        if hyp_info:
            return hyp_info
    return {}


def low_level_prediction_complexity_summary(result_dir: Path) -> Dict[str, Any]:
    """Summarize prediction complexity from low_level_results.json.

    This captures final incident-record complexity, especially the number of
    member particles that contributed to each low-level prediction.  It is a
    useful complement to active_particles from pipeline_timings.jsonl, which is
    the online active-set size per processed report.
    """
    path = result_dir / LOW_LEVEL_FILENAME
    if not path.exists():
        return {
            "low_level_prediction_count": None,
            "member_particles_mean": None,
            "member_particles_median": None,
            "member_particles_max": None,
            "member_particles_sum": None,
            "candidate_ids_mean": None,
            "candidate_ids_max": None,
            "source_cells_mean": None,
            "source_cells_max": None,
            "coverage_cells_mean": None,
            "coverage_cells_max": None,
        }
    try:
        payload = read_json(path)
    except Exception:
        return {
            "low_level_prediction_count": None,
            "low_level_results_read_error": True,
        }
    rows = payload.get("low_level_incidents", [])
    if not isinstance(rows, list):
        return {
            "low_level_prediction_count": None,
            "low_level_results_malformed": True,
        }

    member_counts: List[float] = []
    candidate_counts: List[float] = []
    source_cell_counts: List[float] = []
    coverage_cell_counts: List[float] = []
    source_weighted_area: List[float] = []
    coverage_weighted_area: List[float] = []

    for item in rows:
        if not isinstance(item, dict):
            continue
        hyp_info = _prediction_hypothesis_info(item)
        member_count = safe_float(hyp_info.get("num_member_particles"), None)
        if member_count is None and isinstance(hyp_info.get("member_particle_ids"), list):
            member_count = float(len(hyp_info.get("member_particle_ids") or []))
        if member_count is not None:
            member_counts.append(member_count)

        candidate_ids = item.get("candidate_ids")
        if isinstance(candidate_ids, list):
            candidate_counts.append(float(len(candidate_ids)))

        source_points = item.get("hypothesis_source_points")
        if isinstance(source_points, list):
            source_cell_counts.append(float(len(source_points)))
        coverage_points = item.get("propagated_coverage_points")
        if isinstance(coverage_points, list):
            coverage_cell_counts.append(float(len(coverage_points)))

        source_area = item.get("hypothesis_source_area_coverage") if isinstance(item.get("hypothesis_source_area_coverage"), dict) else {}
        cov_area = item.get("propagated_area_coverage") if isinstance(item.get("propagated_area_coverage"), dict) else {}
        source_area_value = safe_float(source_area.get("coverage_weighted_area_km2"), None)
        cov_area_value = safe_float(cov_area.get("coverage_weighted_area_km2"), None)
        if source_area_value is not None:
            source_weighted_area.append(source_area_value)
        if cov_area_value is not None:
            coverage_weighted_area.append(cov_area_value)

    return {
        "low_level_prediction_count": len([r for r in rows if isinstance(r, dict)]),
        "member_particles_mean": mean(member_counts),
        "member_particles_median": median(member_counts),
        "member_particles_max": max(member_counts) if member_counts else None,
        "member_particles_sum": sum(member_counts) if member_counts else None,
        "candidate_ids_mean": mean(candidate_counts),
        "candidate_ids_max": max(candidate_counts) if candidate_counts else None,
        "candidate_ids_sum": sum(candidate_counts) if candidate_counts else None,
        "source_cells_mean": mean(source_cell_counts),
        "source_cells_max": max(source_cell_counts) if source_cell_counts else None,
        "coverage_cells_mean": mean(coverage_cell_counts),
        "coverage_cells_max": max(coverage_cell_counts) if coverage_cell_counts else None,
        "source_weighted_area_km2_mean": mean(source_weighted_area),
        "source_weighted_area_km2_max": max(source_weighted_area) if source_weighted_area else None,
        "coverage_weighted_area_km2_mean": mean(coverage_weighted_area),
        "coverage_weighted_area_km2_max": max(coverage_weighted_area) if coverage_weighted_area else None,
    }


def summarize_pipeline_jsonl(result_dir: Path) -> Dict[str, Any]:
    # The final result folder may or may not contain this file. Keep this robust.
    candidates = [
        result_dir / "pipeline_timings.jsonl",
        result_dir / "timings.jsonl",
    ]
    rows: List[Dict[str, Any]] = []
    for path in candidates:
        if path.exists():
            rows.extend(iter_jsonl(path))

    active_values: List[float] = []
    new_particle_values: List[float] = []
    incident_candidate_values: List[float] = []
    incident_prediction_values: List[float] = []
    composition_edge_values: List[float] = []
    total_process_values_ms: List[float] = []
    rss_values: List[float] = []
    cache_hits = 0
    cache_hit_known = 0

    for row in rows:
        # Current full_pipeline.py writes active/new particle counts inside the
        # nested hypothesis_proposal object.  Keep legacy top-level fallbacks for
        # older timing files.
        hp = row.get("hypothesis_proposal") if isinstance(row.get("hypothesis_proposal"), dict) else {}
        active = safe_float(hp.get("active_particles"), None)
        if active is None:
            for key in ["active_particles", "active_particle_count", "active_hypotheses", "active_hypotheses_count", "num_active_hypotheses"]:
                active = safe_float(row.get(key), None)
                if active is not None:
                    break
        if active is not None:
            active_values.append(active)

        new_particles = safe_float(hp.get("new_particles"), None)
        if new_particles is not None:
            new_particle_values.append(new_particles)

        ic = row.get("incident_clustering") if isinstance(row.get("incident_clustering"), dict) else {}
        ip = row.get("incident_prediction") if isinstance(row.get("incident_prediction"), dict) else {}
        ico = row.get("incident_composition") if isinstance(row.get("incident_composition"), dict) else {}
        val = safe_float(ic.get("num_candidates"), None)
        if val is not None:
            incident_candidate_values.append(val)
        val = safe_float(ip.get("num_predictions"), None)
        if val is not None:
            incident_prediction_values.append(val)
        val = safe_float(ico.get("num_edges"), None)
        if val is not None:
            composition_edge_values.append(val)

        timing_seconds = row.get("timing_seconds") if isinstance(row.get("timing_seconds"), dict) else {}
        total_s = safe_float(timing_seconds.get("total_process_one_report"), None)
        if total_s is not None:
            total_process_values_ms.append(total_s * 1000.0)

        if row.get("cache_hit") is not None:
            cache_hit_known += 1
            if bool(row.get("cache_hit")):
                cache_hits += 1

        resource = row.get("resource_usage") if isinstance(row.get("resource_usage"), dict) else {}
        rss = safe_float(resource.get("max_rss_mb") or row.get("max_rss_mb"), None)
        if rss is not None:
            rss_values.append(rss)

    return {
        "active_particles_mean": mean(active_values),
        "active_particles_median": median(active_values),
        "active_particles_max": max(active_values) if active_values else None,
        "new_particles_mean": mean(new_particle_values),
        "new_particles_max": max(new_particle_values) if new_particle_values else None,
        "incident_candidates_mean": mean(incident_candidate_values),
        "incident_candidates_max": max(incident_candidate_values) if incident_candidate_values else None,
        "incident_predictions_mean": mean(incident_prediction_values),
        "incident_predictions_max": max(incident_prediction_values) if incident_prediction_values else None,
        "composition_edges_mean": mean(composition_edge_values),
        "composition_edges_max": max(composition_edge_values) if composition_edge_values else None,
        "pipeline_row_total_latency_avg_ms": mean(total_process_values_ms),
        "pipeline_row_total_latency_median_ms": median(total_process_values_ms),
        "pipeline_row_total_latency_max_ms": max(total_process_values_ms) if total_process_values_ms else None,
        "cache_hit_fraction": (cache_hits / cache_hit_known) if cache_hit_known else None,
        "cache_hit_count": cache_hits if cache_hit_known else None,
        "cache_hit_known_count": cache_hit_known if cache_hit_known else None,
        "max_rss_mb": max(rss_values) if rss_values else None,
        "mean_rss_mb": mean(rss_values),
        "pipeline_timings_rows": len(rows),
    }


def runtime_summary_for_result_dir(method: str, experiment: str, run_name: str, result_dir: Path) -> Dict[str, Any]:
    completion_summary = completion_summary_for_result_dir(result_dir)
    complexity_summary = low_level_prediction_complexity_summary(result_dir)
    pipeline_summary = summarize_pipeline_jsonl(result_dir)

    timing_path = result_dir / TIMING_FILENAME
    base_row = {
        "method": method,
        "experiment": experiment,
        "variant": experiment_variant_name(experiment),
        "scalability_setting": scalability_setting_name(experiment) if is_scalability_experiment_name(experiment) else experiment_variant_name(experiment),
        "observation_model_condition": observation_model_condition(experiment),
        "run_name": run_name,
        "result_dir": str(result_dir),
        **completion_summary,
        **complexity_summary,
        **pipeline_summary,
    }
    if not timing_path.exists():
        return base_row
    try:
        timing = read_json(timing_path)
    except Exception:
        timing = {}

    total_comp = find_total_process_component(timing) or {}
    timing_num_obs_value = safe_float(timing.get("num_observations"), None)
    timing_num_obs = int(timing_num_obs_value) if timing_num_obs_value is not None else None

    # Prefer the completion marker for how many reports the emitter actually
    # sent, but fall back to timing.json. actual_timing runs may also store a
    # top-level reports_sent field in timing.json.
    completion_reports_sent = safe_float(completion_summary.get("reports_sent"), None)
    timing_reports_sent = safe_float(timing.get("reports_sent"), None)
    reports_sent_value = completion_reports_sent if completion_reports_sent is not None else timing_reports_sent
    reports_sent_int = int(reports_sent_value) if reports_sent_value is not None else None

    effective_num_obs = timing_num_obs if timing_num_obs and timing_num_obs > 0 else reports_sent_int
    total_seconds = safe_float(total_comp.get("total_seconds"), None)
    avg_ms = safe_float(total_comp.get("avg_ms"), None)
    throughput = (effective_num_obs / total_seconds) if (effective_num_obs and total_seconds and total_seconds > 0) else None

    # actual_timing/full end-to-end fields. These are whole-incident wall-clock
    # timings from incident_start to incident_end and should be used for the
    # controlled throughput-vs-report-count figure. Keep several fallbacks so
    # older timing.json variants are still readable.
    incident_wall_clock_seconds = safe_float(
        timing.get("incident_wall_clock_seconds")
        or timing.get("total_incident_wall_clock_seconds")
        or timing.get("wall_clock_seconds")
        or timing.get("total_wall_clock_seconds")
        or timing.get("elapsed_seconds"),
        None,
    )
    incident_wall_clock_reports_per_second = safe_float(
        timing.get("incident_wall_clock_reports_per_second")
        or timing.get("wall_clock_reports_per_second")
        or timing.get("reports_per_second"),
        None,
    )
    if incident_wall_clock_reports_per_second is None and incident_wall_clock_seconds and effective_num_obs:
        incident_wall_clock_reports_per_second = effective_num_obs / incident_wall_clock_seconds

    incident_wall_clock_reports_per_minute = safe_float(
        timing.get("incident_wall_clock_reports_per_minute")
        or timing.get("wall_clock_reports_per_minute")
        or timing.get("reports_per_minute"),
        None,
    )
    if incident_wall_clock_reports_per_minute is None and incident_wall_clock_reports_per_second is not None:
        incident_wall_clock_reports_per_minute = incident_wall_clock_reports_per_second * 60.0

    incident_wall_clock_seconds_per_100_reports = safe_float(
        timing.get("incident_wall_clock_seconds_per_100_reports")
        or timing.get("wall_clock_seconds_per_100_reports"),
        None,
    )
    if incident_wall_clock_seconds_per_100_reports is None and incident_wall_clock_reports_per_second and incident_wall_clock_reports_per_second > 0:
        incident_wall_clock_seconds_per_100_reports = 100.0 / incident_wall_clock_reports_per_second

    row = {
        **base_row,
        "duplication_level": actual_timing_duplication_level(experiment) if is_actual_timing_experiment_name(experiment) else None,
        "actual_timing_setting": actual_timing_setting_name(experiment) if is_actual_timing_experiment_name(experiment) else None,
        "timing_num_observations": timing_num_obs,
        "timing_reports_sent": int(timing_reports_sent) if timing_reports_sent is not None else None,
        # Keep the legacy column name but make it robust: timing.json first,
        # then experiment_complete.json reports_sent as fallback.
        "reports_sent": reports_sent_int,
        "num_observations": effective_num_obs,
        "end_to_end_latency_avg_ms": avg_ms,
        "end_to_end_total_seconds": total_seconds,
        "throughput_observations_per_sec": throughput,
        "incident_wall_clock_seconds": incident_wall_clock_seconds,
        "incident_wall_clock_reports_per_second": incident_wall_clock_reports_per_second,
        "incident_wall_clock_reports_per_minute": incident_wall_clock_reports_per_minute,
        "incident_wall_clock_seconds_per_100_reports": incident_wall_clock_seconds_per_100_reports,
    }
    for key in [
        "hypothesis_proposal_birth",
        "hypothesis_evaluation",
        "hypothesis_retrospective_scoring",
        "hypothesis_prune_maintain",
        "incident_clustering_update",
        "incident_prediction_update",
        "incident_composition_update",
        "active_hypotheses_write",
    ]:
        row.update(component_stats(timing, key))
    return row


# ---------------------------------------------------------------------------
# Discovery/output
# ---------------------------------------------------------------------------

def is_ablation_experiment_name(experiment: str) -> bool:
    """Return True for architectural/modality ablation experiment folders."""
    return experiment.startswith(ABLATION_EXPERIMENT_PREFIXES)


def is_scalability_performance_experiment_name(experiment: str) -> bool:
    """Return True for scalability folders with prediction outputs."""
    return experiment.startswith(SCALABILITY_PERFORMANCE_PREFIXES)


def is_timing_scalability_experiment_name(experiment: str) -> bool:
    """Return True for dedicated timing-only scalability folders."""
    return experiment.startswith(SCALABILITY_TIMING_PREFIXES)


def is_scalability_experiment_name(experiment: str) -> bool:
    """Return True for scalability performance or timing experiment folders."""
    return is_scalability_performance_experiment_name(experiment) or is_timing_scalability_experiment_name(experiment)


def is_actual_timing_experiment_name(experiment: str) -> bool:
    """Return True for controlled duplication throughput experiment folders."""
    return experiment.startswith(ACTUAL_TIMING_EXPERIMENT_PREFIXES)


def actual_timing_duplication_level(experiment: str) -> Optional[int]:
    """Extract duplication level from names such as actual_timing_dup5."""
    match = re.search(r"(?:^|_)dup(\d+)(?:$|_)", str(experiment))
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    match = re.search(r"actual_timing_dup(\d+)", str(experiment))
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def actual_timing_setting_name(experiment: str) -> str:
    """Normalize actual_timing experiment names to dupN."""
    level = actual_timing_duplication_level(experiment)
    return f"dup{level}" if level is not None else experiment


def scalability_setting_name(experiment: str) -> str:
    """Normalize paired performance/timing scalability experiment names.

    Example:
      ablation_scalability_scalability_k40 -> k40
      timing_scalability_k40              -> k40

    This lets the evaluator report timing with and without the observation
    model for the same scale setting.
    """
    for prefix in (*SCALABILITY_PERFORMANCE_PREFIXES, *SCALABILITY_TIMING_PREFIXES):
        if experiment.startswith(prefix):
            return experiment[len(prefix):]
    if experiment.startswith("ablation_scalability_"):
        return experiment[len("ablation_scalability_"):]
    return experiment


def observation_model_condition(experiment: str) -> str:
    """Classify timing rows by whether observation-model cost is included."""
    name = norm_label(experiment)
    if is_scalability_performance_experiment_name(experiment):
        return "without_observation_model"
    if "without observation" in name or "no observation" in name or "no obs" in name:
        return "without_observation_model"
    if is_timing_scalability_experiment_name(experiment) or is_actual_timing_experiment_name(experiment):
        return "with_observation_model"
    return "unknown"


def experiment_variant_name(experiment: str) -> str:
    """Human-readable variant name for aggregate ablation/scalability/throughput tables."""
    for prefix in ("ablation_scalability_ablation_", "modality_ablation_"):
        if experiment.startswith(prefix):
            return experiment[len(prefix):]
    if is_actual_timing_experiment_name(experiment):
        return actual_timing_setting_name(experiment)
    if is_scalability_experiment_name(experiment):
        return scalability_setting_name(experiment)
    if experiment.startswith("ablation_scalability_"):
        return experiment[len("ablation_scalability_"):]
    return experiment


def discover_result_dirs(
    results_root: Path,
    *,
    mode: str,
    low_experiment_name: str = "synth_low",
    experiment_glob: Optional[str] = None,
    methods: Sequence[str] = (),
) -> List[Tuple[str, str, str, Path]]:
    out: List[Tuple[str, str, str, Path]] = []
    if not results_root.exists():
        return out
    method_filter = {m for m in methods if m}
    # Ablations are currently stored as experiment folders under results/incidentlens.
    # If the caller does not specify --methods, restrict --mode ablation to
    # IncidentLens so older baseline/scalability directories are not mixed in.
    if mode == "ablation" and not method_filter:
        method_filter = {"incidentlens"}

    for method_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        method = method_dir.name
        if method_filter and method not in method_filter:
            continue
        for exp_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            experiment = exp_dir.name
            if experiment_glob:
                if not exp_dir.match(experiment_glob) and experiment_glob not in experiment:
                    continue
            elif mode == "low_incident":
                if experiment != low_experiment_name:
                    continue
            elif mode == "ablation":
                if not is_ablation_experiment_name(experiment):
                    continue
            elif mode == "scalability":
                if not is_scalability_experiment_name(experiment):
                    continue
            elif mode == "throughput":
                if not is_actual_timing_experiment_name(experiment):
                    continue
            elif mode == "coverage_smoke":
                prefix = "coverage_smoke"
                # Use the default prefix here; evaluate_coverage_smoke applies the
                # user-configurable prefix after discovery if needed.
                if not experiment.startswith(prefix + "_"):
                    continue
            elif mode == "ablation_and_scalability":
                if not experiment.startswith(ABLATION_AND_SCALABILITY_EXPERIMENT_PREFIXES):
                    continue
            else:
                continue

            for run_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
                if (run_dir / LOW_LEVEL_FILENAME).exists() or (run_dir / TIMING_FILENAME).exists():
                    out.append((method, experiment, run_dir.name, run_dir))
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key.startswith("_"):
                continue
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        json.dump(jsonable(payload), outfile, ensure_ascii=False, indent=2, sort_keys=True)
        outfile.write("\n")


def jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return dt_to_iso(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return jsonable(value.__dict__)
    return value


def aggregate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    tp = sum(int(r.get("tp") or 0) for r in rows)
    fp = sum(int(r.get("fp") or 0) for r in rows)
    fn = sum(int(r.get("fn") or 0) for r in rows)
    return {
        "num_runs": len(rows),
        "num_gt": sum(int(r.get("num_gt") or 0) for r in rows),
        "num_predictions": sum(int(r.get("num_predictions") or 0) for r in rows),
        "num_prediction_type_labels": sum(int(r.get("num_prediction_type_labels") or r.get("num_predictions") or 0) for r in rows),
        "num_rationale_type_guess_labels": sum(int(r.get("num_rationale_type_guess_labels") or 0) for r in rows),
        "num_gt_with_geometry": sum(int(r.get("num_gt_with_geometry") or 0) for r in rows),
        "num_predictions_with_geometry": sum(int(r.get("num_predictions_with_geometry") or 0) for r in rows),
        "num_matches_with_spatial_metrics": sum(int(r.get("num_matches_with_spatial_metrics") or 0) for r in rows),
        "num_matches_with_temporal_iou": sum(int(r.get("num_matches_with_temporal_iou") or 0) for r in rows),
        "scenario_metrics_num_gt_types": sum(int(r.get("scenario_metrics_num_gt_types") or 0) for r in rows),
        "scenario_metrics_num_types_with_spatial_iou": sum(int(r.get("scenario_metrics_num_types_with_spatial_iou") or 0) for r in rows),
        "scenario_metrics_num_types_with_temporal_iou": sum(int(r.get("scenario_metrics_num_types_with_temporal_iou") or 0) for r in rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": fbeta(tp, fp, fn, beta=1.0),
        "f2": fbeta(tp, fp, fn, beta=2.0),
        "micro_precision": ratio(tp, tp + fp),
        "micro_recall": ratio(tp, tp + fn),
        "micro_f1": fbeta(tp, fp, fn, beta=1.0),
        "micro_f2": fbeta(tp, fp, fn, beta=2.0),
        "false_positives_per_run": ratio(fp, len(rows)),
        "coverage_mean_gt": mean(r.get("coverage_mean_gt") for r in rows),
        "coverage_sum_gt": sum(float(r.get("coverage_sum_gt") or 0.0) for r in rows),
        "coverage_sum_matched_gt": sum(float(r.get("coverage_sum_matched_gt") or 0.0) for r in rows),
        "coverage_weighted_recall": (
            ratio(
                sum(float(r.get("coverage_sum_matched_gt") or 0.0) for r in rows),
                sum(float(r.get("coverage_sum_gt") or 0.0) for r in rows),
            )
            if any(r.get("coverage_sum_gt") is not None for r in rows)
            else None
        ),
        "coverage_weighted_f1": (
            fbeta(
                sum(float(r.get("coverage_sum_matched_gt") or 0.0) for r in rows),
                fp,
                max(
                    0.0,
                    sum(float(r.get("coverage_sum_gt") or 0.0) for r in rows)
                    - sum(float(r.get("coverage_sum_matched_gt") or 0.0) for r in rows),
                ),
                beta=1.0,
            )
            if any(r.get("coverage_sum_gt") is not None for r in rows)
            else None
        ),
        "coverage_weighted_f2": (
            fbeta(
                sum(float(r.get("coverage_sum_matched_gt") or 0.0) for r in rows),
                fp,
                max(
                    0.0,
                    sum(float(r.get("coverage_sum_gt") or 0.0) for r in rows)
                    - sum(float(r.get("coverage_sum_matched_gt") or 0.0) for r in rows),
                ),
                beta=2.0,
            )
            if any(r.get("coverage_sum_gt") is not None for r in rows)
            else None
        ),
        "median_detection_delay_minutes": median(r.get("median_detection_delay_minutes") for r in rows),
        "fraction_detected_before_external_report": mean(r.get("fraction_detected_before_external_report") for r in rows),
        "pre_report_recall": mean(r.get("pre_report_recall") for r in rows),
        "matched_pre_report_fraction": mean(r.get("matched_pre_report_fraction") for r in rows),
        "median_external_report_delay_minutes": median(r.get("median_external_report_delay_minutes") for r in rows),
        "p25_external_report_delay_minutes": median(r.get("p25_external_report_delay_minutes") for r in rows),
        "p75_external_report_delay_minutes": median(r.get("p75_external_report_delay_minutes") for r in rows),
        # Legacy matched-TP metric names retained for backward compatibility.
        "median_source_localization_error_km": median(r.get("median_source_localization_error_km") for r in rows),
        "median_coverage_localization_error_km": median(r.get("median_coverage_localization_error_km") for r in rows),
        "mean_source_iou": mean(r.get("mean_source_iou") for r in rows),
        "mean_coverage_iou": mean(r.get("mean_coverage_iou") for r in rows),
        "median_temporal_iou": median(r.get("median_temporal_iou") for r in rows),
        # Explicit matched-TP quality names.
        "matched_tp_median_source_or_proxy_localization_error_km": median(r.get("matched_tp_median_source_or_proxy_localization_error_km") for r in rows),
        "matched_tp_median_coverage_localization_error_km": median(r.get("matched_tp_median_coverage_localization_error_km") for r in rows),
        "matched_tp_mean_source_or_proxy_iou": mean(r.get("matched_tp_mean_source_or_proxy_iou") for r in rows),
        "matched_tp_mean_coverage_iou": mean(r.get("matched_tp_mean_coverage_iou") for r in rows),
        "matched_tp_median_temporal_iou": median(r.get("matched_tp_median_temporal_iou") for r in rows),
        # Scenario-level union metrics.
        "mean_scenario_union_coverage_iou": mean(r.get("mean_scenario_union_coverage_iou") for r in rows),
        "median_scenario_union_coverage_iou": median(r.get("median_scenario_union_coverage_iou") for r in rows),
        "mean_scenario_overcoverage_ratio": mean(r.get("mean_scenario_overcoverage_ratio") for r in rows),
        "median_scenario_overcoverage_ratio": median(r.get("median_scenario_overcoverage_ratio") for r in rows),
        "mean_scenario_temporal_iou": mean(r.get("mean_scenario_temporal_iou") for r in rows),
        "median_scenario_temporal_iou": median(r.get("median_scenario_temporal_iou") for r in rows),
        "mean_scenario_temporal_overcoverage_ratio": mean(r.get("mean_scenario_temporal_overcoverage_ratio") for r in rows),
        "median_scenario_temporal_overcoverage_ratio": median(r.get("median_scenario_temporal_overcoverage_ratio") for r in rows),
    }


def aggregate_by_method_experiment(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("method")), str(row.get("experiment"))), []).append(row)
    out = []
    for (method, experiment), items in sorted(groups.items()):
        agg = aggregate_metrics(items)
        agg.update({
            "method": method,
            "experiment": experiment,
            "variant": experiment_variant_name(experiment),
        })
        out.append(agg)
    return out


def aggregate_by_incident_type(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Each run row carries _per_type_counts. Aggregate by method/experiment/type.
    groups: Dict[Tuple[str, str, str], Dict[str, int]] = {}
    for row in rows:
        method = str(row.get("method"))
        experiment = str(row.get("experiment"))
        per_type = row.get("_per_type_counts") if isinstance(row.get("_per_type_counts"), dict) else {}
        for typ, counts in per_type.items():
            if not isinstance(counts, dict):
                continue
            key = (method, experiment, str(typ))
            g = groups.setdefault(key, {"tp": 0, "fp": 0, "fn": 0, "num_runs_with_type": 0})
            g["tp"] += int(counts.get("tp") or 0)
            g["fp"] += int(counts.get("fp") or 0)
            g["fn"] += int(counts.get("fn") or 0)
            if any(int(counts.get(k) or 0) for k in ("tp", "fp", "fn")):
                g["num_runs_with_type"] += 1

    out: List[Dict[str, Any]] = []
    for (method, experiment, typ), c in sorted(groups.items()):
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        out.append({
            "method": method,
            "experiment": experiment,
            "incident_type": typ,
            "num_runs_with_type": c["num_runs_with_type"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn),
            "f1": fbeta(tp, fp, fn, beta=1.0),
            "f2": fbeta(tp, fp, fn, beta=2.0),
            "classwise_f2": fbeta(tp, fp, fn, beta=2.0),
            "f2_accounting": "classwise_predicted_type_fp",
        })
    return out


def aggregate_by_gt_incident_type(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate scenario-conditioned metrics by ground-truth incident type.

    Unlike aggregate_by_incident_type, false positives here are not charged only
    to the label that a prediction emitted.  Instead, for each run containing GT
    incident type T, every prediction that is not a correct match to a GT record
    of type T is counted as an FP for T.  This is the metric to use for plots
    whose panels mean "performance on runs/scenarios of incident type T."

    This aggregation also includes scenario-level union spatial/temporal metrics
    that compare the union of all same-type predictions in those runs to the GT
    region/interval.  These metrics penalize over-fragmented or shotgun outputs
    even when one prediction happens to match the GT.
    """
    groups: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    def g_for(key: Tuple[str, str, str]) -> Dict[str, Any]:
        return groups.setdefault(key, {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "num_runs_with_gt_type": 0,
            "num_gt": 0,
            "num_predictions_in_gt_type_runs": 0,
            "scenario_union_coverage_iou_values": [],
            "scenario_coverage_precision_values": [],
            "scenario_coverage_recall_values": [],
            "scenario_overcoverage_ratio_values": [],
            "scenario_temporal_iou_values": [],
            "scenario_temporal_precision_values": [],
            "scenario_temporal_recall_values": [],
            "scenario_temporal_overcoverage_ratio_values": [],
            "scenario_same_type_predictions": 0,
            "scenario_same_type_predictions_without_geometry": 0,
            "sum_pred_region_area_km2": 0.0,
            "sum_gt_region_area_km2": 0.0,
            "sum_intersection_area_km2": 0.0,
            "sum_union_area_km2": 0.0,
            "sum_extra_area_km2": 0.0,
            "sum_missing_area_km2": 0.0,
            "sum_pred_duration_minutes": 0.0,
            "sum_gt_duration_minutes": 0.0,
            "sum_temporal_intersection_minutes": 0.0,
            "sum_temporal_extra_minutes": 0.0,
            "sum_temporal_missing_minutes": 0.0,
        })

    for row in rows:
        method = str(row.get("method"))
        experiment = str(row.get("experiment"))
        per_type_counts = row.get("_gt_type_conditioned_counts") if isinstance(row.get("_gt_type_conditioned_counts"), dict) else {}
        per_type_union = row.get("_scenario_union_metrics_by_gt_type") if isinstance(row.get("_scenario_union_metrics_by_gt_type"), dict) else {}
        for typ, counts in per_type_counts.items():
            if not isinstance(counts, dict):
                continue
            key = (method, experiment, str(typ))
            g = g_for(key)
            g["tp"] += int(counts.get("tp") or 0)
            g["fp"] += int(counts.get("fp") or 0)
            g["fn"] += int(counts.get("fn") or 0)
            g["num_gt"] += int(counts.get("num_gt") or 0)
            g["num_predictions_in_gt_type_runs"] += int(counts.get("num_predictions") or 0)
            g["num_runs_with_gt_type"] += 1

            sm = per_type_union.get(typ) if isinstance(per_type_union.get(typ), dict) else {}
            for metric_key, list_key in [
                ("scenario_union_coverage_iou", "scenario_union_coverage_iou_values"),
                ("scenario_coverage_precision", "scenario_coverage_precision_values"),
                ("scenario_coverage_recall", "scenario_coverage_recall_values"),
                ("scenario_overcoverage_ratio", "scenario_overcoverage_ratio_values"),
                ("scenario_temporal_iou", "scenario_temporal_iou_values"),
                ("scenario_temporal_precision", "scenario_temporal_precision_values"),
                ("scenario_temporal_recall", "scenario_temporal_recall_values"),
                ("scenario_temporal_overcoverage_ratio", "scenario_temporal_overcoverage_ratio_values"),
            ]:
                val = safe_float(sm.get(metric_key), None)
                if val is not None and math.isfinite(val):
                    g[list_key].append(val)

            for metric_key, sum_key in [
                ("scenario_pred_region_area_km2", "sum_pred_region_area_km2"),
                ("scenario_gt_region_area_km2", "sum_gt_region_area_km2"),
                ("scenario_intersection_area_km2", "sum_intersection_area_km2"),
                ("scenario_union_area_km2", "sum_union_area_km2"),
                ("scenario_extra_area_km2", "sum_extra_area_km2"),
                ("scenario_missing_area_km2", "sum_missing_area_km2"),
                ("scenario_pred_duration_minutes", "sum_pred_duration_minutes"),
                ("scenario_gt_duration_minutes", "sum_gt_duration_minutes"),
                ("scenario_temporal_intersection_minutes", "sum_temporal_intersection_minutes"),
                ("scenario_temporal_extra_minutes", "sum_temporal_extra_minutes"),
                ("scenario_temporal_missing_minutes", "sum_temporal_missing_minutes"),
            ]:
                val = safe_float(sm.get(metric_key), None)
                if val is not None and math.isfinite(val):
                    g[sum_key] += val
            g["scenario_same_type_predictions"] += int(sm.get("scenario_same_type_predictions") or 0)
            g["scenario_same_type_predictions_without_geometry"] += int(sm.get("scenario_same_type_predictions_without_geometry") or 0)

    out: List[Dict[str, Any]] = []
    for (method, experiment, typ), c in sorted(groups.items()):
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        f2 = fbeta(tp, fp, fn, beta=2.0)
        area_union = c["sum_union_area_km2"]
        area_gt = c["sum_gt_region_area_km2"]
        area_pred = c["sum_pred_region_area_km2"]
        area_inter = c["sum_intersection_area_km2"]
        dur_union = c["sum_pred_duration_minutes"] + c["sum_gt_duration_minutes"] - c["sum_temporal_intersection_minutes"]
        out.append({
            "method": method,
            "experiment": experiment,
            "incident_type": typ,
            "num_runs_with_gt_type": c["num_runs_with_gt_type"],
            "num_gt": c["num_gt"],
            "num_predictions_in_gt_type_runs": c["num_predictions_in_gt_type_runs"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn),
            "f1": fbeta(tp, fp, fn, beta=1.0),
            "f2": f2,
            "micro_f2": f2,
            "scenario_micro_f2": f2,
            "f2_accounting": "gt_scenario_all_extra_predictions_fp",
            # Scenario-level same-type union spatial metrics.
            "mean_scenario_union_coverage_iou": mean(c["scenario_union_coverage_iou_values"]),
            "median_scenario_union_coverage_iou": median(c["scenario_union_coverage_iou_values"]),
            "area_weighted_scenario_union_coverage_iou": (area_inter / area_union) if area_union > 0 else None,
            "mean_scenario_coverage_precision": mean(c["scenario_coverage_precision_values"]),
            "mean_scenario_coverage_recall": mean(c["scenario_coverage_recall_values"]),
            "area_weighted_scenario_coverage_precision": (area_inter / area_pred) if area_pred > 0 else None,
            "area_weighted_scenario_coverage_recall": (area_inter / area_gt) if area_gt > 0 else None,
            "mean_scenario_overcoverage_ratio": mean(c["scenario_overcoverage_ratio_values"]),
            "median_scenario_overcoverage_ratio": median(c["scenario_overcoverage_ratio_values"]),
            "area_weighted_scenario_overcoverage_ratio": (c["sum_extra_area_km2"] / area_gt) if area_gt > 0 else None,
            "sum_pred_region_area_km2": c["sum_pred_region_area_km2"],
            "sum_gt_region_area_km2": c["sum_gt_region_area_km2"],
            "sum_intersection_area_km2": c["sum_intersection_area_km2"],
            "sum_union_area_km2": c["sum_union_area_km2"],
            "sum_extra_area_km2": c["sum_extra_area_km2"],
            "sum_missing_area_km2": c["sum_missing_area_km2"],
            "scenario_same_type_predictions": c["scenario_same_type_predictions"],
            "scenario_same_type_predictions_without_geometry": c["scenario_same_type_predictions_without_geometry"],
            # Scenario-level temporal union metrics.
            "mean_scenario_temporal_iou": mean(c["scenario_temporal_iou_values"]),
            "median_scenario_temporal_iou": median(c["scenario_temporal_iou_values"]),
            "duration_weighted_scenario_temporal_iou": (c["sum_temporal_intersection_minutes"] / dur_union) if dur_union > 0 else None,
            "mean_scenario_temporal_precision": mean(c["scenario_temporal_precision_values"]),
            "mean_scenario_temporal_recall": mean(c["scenario_temporal_recall_values"]),
            "duration_weighted_scenario_temporal_precision": (c["sum_temporal_intersection_minutes"] / c["sum_pred_duration_minutes"]) if c["sum_pred_duration_minutes"] > 0 else None,
            "duration_weighted_scenario_temporal_recall": (c["sum_temporal_intersection_minutes"] / c["sum_gt_duration_minutes"]) if c["sum_gt_duration_minutes"] > 0 else None,
            "mean_scenario_temporal_overcoverage_ratio": mean(c["scenario_temporal_overcoverage_ratio_values"]),
            "median_scenario_temporal_overcoverage_ratio": median(c["scenario_temporal_overcoverage_ratio_values"]),
            "duration_weighted_scenario_temporal_overcoverage_ratio": (c["sum_temporal_extra_minutes"] / c["sum_gt_duration_minutes"]) if c["sum_gt_duration_minutes"] > 0 else None,
            "sum_pred_duration_minutes": c["sum_pred_duration_minutes"],
            "sum_gt_duration_minutes": c["sum_gt_duration_minutes"],
            "sum_temporal_intersection_minutes": c["sum_temporal_intersection_minutes"],
            "sum_temporal_extra_minutes": c["sum_temporal_extra_minutes"],
            "sum_temporal_missing_minutes": c["sum_temporal_missing_minutes"],
        })
    return out


def merge_confusions(confusions: List[Dict[str, Dict[str, int]]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, int]] = {}
    for conf in confusions:
        for gt_type, pred_counts in conf.items():
            tgt = merged.setdefault(gt_type, {})
            for pred_type, count in pred_counts.items():
                tgt[pred_type] = int(tgt.get(pred_type, 0)) + int(count)
    rows: List[Dict[str, Any]] = []
    for gt_type, pred_counts in sorted(merged.items()):
        for pred_type, count in sorted(pred_counts.items()):
            rows.append({"gt_type": gt_type, "pred_type": pred_type, "count": count})
    return rows




def aggregate_throughput_by_duplication(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate actual_timing throughput rows by duplication level.

    This mode is intended for controlled duplication experiments where the
    anomaly detector and observation model are enabled.  The preferred timing is
    incident_wall_clock_seconds because it measures the whole incident replay
    from incident_start to incident_end.  If a run lacks that field, fall back to
    end_to_end_total_seconds from timing component summaries.
    """
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        level = safe_float(row.get("duplication_level"), None)
        if level is None:
            level = safe_float(actual_timing_duplication_level(str(row.get("experiment") or "")), None)
        if level is None:
            continue
        groups.setdefault((str(row.get("method")), int(level)), []).append(row)

    out: List[Dict[str, Any]] = []
    for (method, level), items in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        reports = [safe_float(item.get("num_observations"), None) for item in items]
        reports = [v for v in reports if v is not None]
        wall_seconds = [
            safe_float(item.get("incident_wall_clock_seconds"), None)
            if safe_float(item.get("incident_wall_clock_seconds"), None) is not None
            else safe_float(item.get("end_to_end_total_seconds"), None)
            for item in items
        ]
        wall_seconds = [v for v in wall_seconds if v is not None and v > 0]

        total_reports = sum(reports)
        total_wall_seconds = sum(wall_seconds)
        aggregate_rps = (total_reports / total_wall_seconds) if total_wall_seconds > 0 else None
        aggregate_rpm = aggregate_rps * 60.0 if aggregate_rps is not None else None
        seconds_per_100 = (100.0 / aggregate_rps) if aggregate_rps and aggregate_rps > 0 else None

        row: Dict[str, Any] = {
            "method": method,
            "duplication_level": level,
            "setting": f"dup{level}",
            "num_runs": len(items),
            "run_names": ";".join(sorted({str(item.get("run_name")) for item in items if item.get("run_name")})),
            "experiments": ";".join(sorted({str(item.get("experiment")) for item in items if item.get("experiment")})),
            "total_reports": total_reports if reports else None,
            "total_wall_clock_seconds": total_wall_seconds if wall_seconds else None,
            "aggregate_reports_per_second": aggregate_rps,
            "aggregate_reports_per_minute": aggregate_rpm,
            "aggregate_seconds_per_100_reports": seconds_per_100,
            "reports_per_run_mean": mean(reports),
            "reports_per_run_median": median(reports),
            "wall_clock_seconds_mean": mean(wall_seconds),
            "wall_clock_seconds_median": median(wall_seconds),
            "wall_clock_reports_per_second_mean": mean(item.get("incident_wall_clock_reports_per_second") for item in items),
            "wall_clock_reports_per_minute_mean": mean(item.get("incident_wall_clock_reports_per_minute") for item in items),
            "end_to_end_component_seconds_mean": mean(item.get("end_to_end_total_seconds") for item in items),
            "end_to_end_component_reports_per_second_mean": mean(item.get("throughput_observations_per_sec") for item in items),
            "pipeline_row_total_latency_avg_ms_mean": mean(item.get("pipeline_row_total_latency_avg_ms") for item in items),
            "active_particles_mean": mean(item.get("active_particles_mean") for item in items),
            "active_particles_max": max([v for v in [safe_float(item.get("active_particles_max"), None) for item in items] if v is not None], default=None),
            "member_particles_mean": mean(item.get("member_particles_mean") for item in items),
            "member_particles_max": max([v for v in [safe_float(item.get("member_particles_max"), None) for item in items] if v is not None], default=None),
            "cache_hit_fraction_mean": mean(item.get("cache_hit_fraction") for item in items),
        }
        out.append(row)

    return out


def aggregate_runtime(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("method")), str(row.get("experiment"))), []).append(row)

    metric_keys = [
        "reports_sent",
        "timing_num_observations",
        "num_observations",
        "end_to_end_latency_avg_ms",
        "throughput_observations_per_sec",
        "pipeline_row_total_latency_avg_ms",
        "pipeline_row_total_latency_median_ms",
        "hypothesis_proposal_birth_avg_ms",
        "hypothesis_evaluation_avg_ms",
        "hypothesis_retrospective_scoring_avg_ms",
        "hypothesis_prune_maintain_avg_ms",
        "incident_clustering_update_avg_ms",
        "incident_prediction_update_avg_ms",
        "incident_composition_update_avg_ms",
        "active_hypotheses_write_avg_ms",
        "active_particles_mean",
        "active_particles_median",
        "active_particles_max",
        "new_particles_mean",
        "new_particles_max",
        "incident_candidates_mean",
        "incident_candidates_max",
        "incident_predictions_mean",
        "incident_predictions_max",
        "low_level_prediction_count",
        "member_particles_mean",
        "member_particles_median",
        "member_particles_max",
        "member_particles_sum",
        "candidate_ids_mean",
        "candidate_ids_max",
        "source_cells_mean",
        "source_cells_max",
        "coverage_cells_mean",
        "coverage_cells_max",
        "source_weighted_area_km2_mean",
        "coverage_weighted_area_km2_mean",
        "cache_hit_fraction",
        "max_rss_mb",
        "mean_rss_mb",
    ]
    out: List[Dict[str, Any]] = []
    for (method, experiment), items in sorted(groups.items()):
        row: Dict[str, Any] = {"method": method, "experiment": experiment, "num_runs": len(items)}
        for key in metric_keys:
            vals = [safe_float(item.get(key), None) for item in items]
            vals = [v for v in vals if v is not None]
            row[f"{key}_mean"] = mean(vals)
            row[f"{key}_median"] = median(vals)
            row[f"{key}_max"] = max(vals) if vals else None
        # Better total throughput across runs if total process seconds exists.
        total_obs = sum(int(item.get("num_observations") or 0) for item in items)
        total_time = sum(float(item.get("end_to_end_total_seconds") or 0.0) for item in items)
        row["aggregate_throughput_observations_per_sec"] = total_obs / total_time if total_time > 0 else None
        out.append(row)

    return out


def aggregate_runtime_by_scalability_setting(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate timing rows by normalized scalability setting and obs condition."""
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        method = str(row.get("method"))
        setting = str(row.get("scalability_setting") or experiment_variant_name(str(row.get("experiment") or "")))
        obs = str(row.get("observation_model_condition") or "unknown")
        groups.setdefault((method, setting, obs), []).append(row)

    metric_keys = [
        "reports_sent",
        "timing_num_observations",
        "num_observations",
        "end_to_end_latency_avg_ms",
        "throughput_observations_per_sec",
        "pipeline_row_total_latency_avg_ms",
        "pipeline_row_total_latency_median_ms",
        "hypothesis_proposal_birth_avg_ms",
        "hypothesis_evaluation_avg_ms",
        "hypothesis_retrospective_scoring_avg_ms",
        "hypothesis_prune_maintain_avg_ms",
        "incident_clustering_update_avg_ms",
        "incident_prediction_update_avg_ms",
        "incident_composition_update_avg_ms",
        "active_hypotheses_write_avg_ms",
        "active_particles_mean",
        "active_particles_median",
        "active_particles_max",
        "new_particles_mean",
        "new_particles_max",
        "incident_candidates_mean",
        "incident_candidates_max",
        "incident_predictions_mean",
        "incident_predictions_max",
        "low_level_prediction_count",
        "member_particles_mean",
        "member_particles_median",
        "member_particles_max",
        "member_particles_sum",
        "candidate_ids_mean",
        "candidate_ids_max",
        "source_cells_mean",
        "source_cells_max",
        "coverage_cells_mean",
        "coverage_cells_max",
        "source_weighted_area_km2_mean",
        "coverage_weighted_area_km2_mean",
        "cache_hit_fraction",
        "max_rss_mb",
        "mean_rss_mb",
    ]
    out: List[Dict[str, Any]] = []
    for (method, setting, obs), items in sorted(groups.items()):
        row: Dict[str, Any] = {
            "method": method,
            "scalability_setting": setting,
            "observation_model_condition": obs,
            "num_runs": len(items),
        }
        experiments = sorted({str(item.get("experiment")) for item in items if item.get("experiment")})
        row["experiments"] = ";".join(experiments)
        for key in metric_keys:
            vals = [safe_float(item.get(key), None) for item in items]
            vals = [v for v in vals if v is not None]
            row[f"{key}_mean"] = mean(vals)
            row[f"{key}_median"] = median(vals)
            row[f"{key}_max"] = max(vals) if vals else None
        total_obs = sum(int(item.get("num_observations") or 0) for item in items)
        total_time = sum(float(item.get("end_to_end_total_seconds") or 0.0) for item in items)
        row["aggregate_throughput_observations_per_sec"] = total_obs / total_time if total_time > 0 else None
        row["total_observations"] = total_obs
        row["total_process_seconds"] = total_time if total_time > 0 else None
        out.append(row)
    return out


def aggregate_performance_by_scalability_setting(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate prediction performance for scalability performance folders."""
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        method = str(row.get("method"))
        setting = str(row.get("scalability_setting") or experiment_variant_name(str(row.get("experiment") or "")))
        groups.setdefault((method, setting), []).append(row)
    out: List[Dict[str, Any]] = []
    for (method, setting), items in sorted(groups.items()):
        agg = aggregate_metrics(items)
        agg.update({"method": method, "scalability_setting": setting})
        experiments = sorted({str(item.get("experiment")) for item in items if item.get("experiment")})
        agg["experiments"] = ";".join(experiments)
        out.append(agg)
    return out


def combine_scalability_performance_and_runtime(
    performance_rows: List[Dict[str, Any]],
    runtime_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Create one row per setting with performance plus with/without timing."""
    perf_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    keys: set[Tuple[str, str]] = set()
    for row in performance_rows:
        key = (str(row.get("method")), str(row.get("scalability_setting")))
        keys.add(key)
        perf_by_key[key] = row

    runtime_by_key_obs: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in runtime_rows:
        key = (
            str(row.get("method")),
            str(row.get("scalability_setting")),
            str(row.get("observation_model_condition") or "unknown"),
        )
        keys.add((key[0], key[1]))
        runtime_by_key_obs[key] = row

    out: List[Dict[str, Any]] = []
    for method, setting in sorted(keys):
        perf = perf_by_key.get((method, setting), {})
        row: Dict[str, Any] = {
            "method": method,
            "scalability_setting": setting,
            "performance_num_runs": perf.get("num_runs"),
            "performance_num_gt": perf.get("num_gt"),
            "performance_num_predictions": perf.get("num_predictions"),
            "precision": perf.get("precision"),
            "recall": perf.get("recall"),
            "f1": perf.get("f1"),
            "f2": perf.get("f2"),
            "false_positives_per_run": perf.get("false_positives_per_run"),
            "mean_scenario_union_coverage_iou": perf.get("mean_scenario_union_coverage_iou"),
            "mean_scenario_temporal_iou": perf.get("mean_scenario_temporal_iou"),
        }
        for obs in ("with_observation_model", "without_observation_model", "unknown"):
            runtime = runtime_by_key_obs.get((method, setting, obs), {})
            if not runtime:
                continue
            prefix = "with_obs" if obs == "with_observation_model" else "without_obs" if obs == "without_observation_model" else "unknown_obs"
            row[f"{prefix}_runtime_num_runs"] = runtime.get("num_runs")
            row[f"{prefix}_latency_avg_ms_mean"] = runtime.get("end_to_end_latency_avg_ms_mean")
            row[f"{prefix}_latency_avg_ms_median"] = runtime.get("end_to_end_latency_avg_ms_median")
            row[f"{prefix}_throughput_obs_per_sec_mean"] = runtime.get("throughput_observations_per_sec_mean")
            row[f"{prefix}_throughput_obs_per_sec_median"] = runtime.get("throughput_observations_per_sec_median")
            row[f"{prefix}_aggregate_throughput_obs_per_sec"] = runtime.get("aggregate_throughput_observations_per_sec")
            row[f"{prefix}_total_observations"] = runtime.get("total_observations")
            row[f"{prefix}_num_observations_mean"] = runtime.get("num_observations_mean")
            row[f"{prefix}_reports_sent_mean"] = runtime.get("reports_sent_mean")
            row[f"{prefix}_active_particles_mean"] = runtime.get("active_particles_mean_mean")
            row[f"{prefix}_active_particles_max"] = runtime.get("active_particles_max_max")
            row[f"{prefix}_new_particles_mean"] = runtime.get("new_particles_mean_mean")
            row[f"{prefix}_incident_candidates_mean"] = runtime.get("incident_candidates_mean_mean")
            row[f"{prefix}_incident_predictions_mean"] = runtime.get("incident_predictions_mean_mean")
            row[f"{prefix}_low_level_prediction_count_mean"] = runtime.get("low_level_prediction_count_mean")
            row[f"{prefix}_member_particles_mean"] = runtime.get("member_particles_mean_mean")
            row[f"{prefix}_member_particles_max"] = runtime.get("member_particles_max_max")
            row[f"{prefix}_source_cells_mean"] = runtime.get("source_cells_mean_mean")
            row[f"{prefix}_coverage_cells_mean"] = runtime.get("coverage_cells_mean_mean")
            row[f"{prefix}_cache_hit_fraction_mean"] = runtime.get("cache_hit_fraction_mean")
            row[f"{prefix}_max_rss_mb"] = runtime.get("max_rss_mb_max")
        out.append(row)
    return out



# ---------------------------------------------------------------------------
# Real-data low-level evaluation
# ---------------------------------------------------------------------------

def pacific_tzinfo():
    """Return America/Los_Angeles tzinfo, with a fixed-offset fallback."""
    if ZoneInfo is not None:  # type: ignore[name-defined]
        try:
            return ZoneInfo("America/Los_Angeles")  # type: ignore[misc]
        except Exception:
            pass
    return timezone(timedelta(hours=-8))


def parse_pacific_dt(value: Any) -> Optional[datetime]:
    """Parse GT timestamps whose field names explicitly say *_pacific.

    parse_dt intentionally treats naive datetimes as UTC for synthetic files.
    Real GT fields such as start_datetime_pacific and
    earliest_article_datetime_pacific are local Pacific civil time, so this
    parser localizes naive timestamps to America/Los_Angeles before converting
    to UTC.  It also accepts slightly non-ISO strings such as T4:00:00.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            # A Z suffix is already UTC, so defer to the generic parser.
            return parse_dt(text)
        # Normalize a single-digit hour sometimes found in hand-corrected GT,
        # e.g. 2026-02-13T4:00:00 -> 2026-02-13T04:00:00.
        text = re.sub(r"T(\d):(\d{2}:\d{2})", r"T0\1:\2", text)
        text = re.sub(r" (\d):(\d{2}:\d{2})", r" 0\1:\2", text)
        if re.fullmatch(r"\d{8}", text):
            try:
                dt = datetime.strptime(text, "%Y%m%d")
            except ValueError:
                return None
        else:
            try:
                dt = datetime.fromisoformat(text)
            except ValueError:
                # Try the generic parser as a last resort.
                parsed = parse_dt(text)
                return parsed
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=pacific_tzinfo())
    return dt.astimezone(timezone.utc)


def _normalize_geo_cache_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _load_cached_geo_record(location_name: str, cache_path: Optional[str | Path]) -> Optional[Dict[str, Any]]:
    """Load a GeoManager-compatible cached geometry record without live geocoding."""
    if not cache_path:
        return None
    path = Path(cache_path)
    if not path.exists():
        return None
    try:
        cache = read_json(path)
    except Exception:
        return None
    if not isinstance(cache, dict):
        return None
    record = cache.get(_normalize_geo_cache_key(location_name))
    if not isinstance(record, dict):
        return None
    if record.get("status") != "ok" or not record.get("geometry_wkt"):
        return None
    try:
        geom = wkt.loads(str(record.get("geometry_wkt")))
    except Exception:
        return None
    out = dict(record)
    out["geometry"] = geom
    out["cache_hit"] = True
    out["cache_only_load"] = True
    return out


def _load_geo_manager(cache_path: Optional[str | Path] = None):
    """Best-effort import of the project GeoManager used by the real dataset."""
    try:
        from evaluation.geo_manager import GeoManager  # type: ignore
    except Exception:
        try:
            from geo_manager import GeoManager  # type: ignore
        except Exception:
            return None, "GeoManager import failed"
    try:
        if cache_path:
            return GeoManager(cache_path=str(cache_path)), None
        return GeoManager(), None
    except Exception as exc:
        return None, f"GeoManager initialization failed: {exc}"


def real_gt_location_queries(value: Dict[str, Any]) -> List[str]:
    """Generate conservative geocoding queries for a real GT row."""
    location = str(value.get("location") or "").strip()
    final_name = str(value.get("final_name") or "").strip()
    queries: List[str] = []
    for q in (location, f"{final_name}, {location}" if final_name and location else "", final_name):
        q = q.strip(" ,")
        if q:
            queries.append(q)
    expanded: List[str] = []
    for q in queries:
        expanded.append(q)
        q_lower = q.lower()
        if "california" not in q_lower and " ca" not in q_lower:
            expanded.append(f"{q}, Los Angeles County, California")
    return list(dict.fromkeys(expanded))


def geocode_real_gt_geometry(
    value: Dict[str, Any],
    *,
    geo_manager: Any = None,
    cache_path: Optional[str | Path] = None,
    cache_only: bool = False,
    force_refresh: bool = False,
) -> Tuple[Optional[BaseGeometry], Optional[float], Optional[float], Dict[str, Any]]:
    """Resolve a real GT textual location to a geometry using GeoManager/cache.

    This is intentionally best-effort: failures leave geometry as None so the
    low_real evaluator can still fall back to type+time matching.
    """
    attempts: List[Dict[str, Any]] = []
    for query in real_gt_location_queries(value):
        cached = None if force_refresh else _load_cached_geo_record(query, cache_path)
        if cached is not None:
            geom = cached.get("geometry")
            if geom is not None and not geom.is_empty:
                c = geom.centroid
                return geom, float(c.y), float(c.x), {
                    "status": "ok",
                    "query": query,
                    "source": cached.get("source"),
                    "method": cached.get("method"),
                    "cache_hit": True,
                    "attempts": attempts,
                }
        if cache_only or geo_manager is None:
            attempts.append({"query": query, "status": "not_in_cache"})
            continue
        try:
            record = geo_manager.get_geo_region(query, force_refresh=force_refresh)
        except TypeError:
            try:
                record = geo_manager.get_geo_region(query)
            except Exception as exc:
                attempts.append({"query": query, "status": "exception", "error": str(exc)})
                continue
        except Exception as exc:
            attempts.append({"query": query, "status": "exception", "error": str(exc)})
            continue
        if isinstance(record, dict) and record.get("status") == "ok" and record.get("geometry") is not None:
            geom = record.get("geometry")
            if geom is not None and not geom.is_empty:
                c = geom.centroid
                return geom, float(c.y), float(c.x), {
                    "status": "ok",
                    "query": query,
                    "source": record.get("source"),
                    "method": record.get("method"),
                    "cache_hit": record.get("cache_hit"),
                    "attempts": attempts,
                }
        attempts.append({
            "query": query,
            "status": "failed",
            "reason": record.get("reason") if isinstance(record, dict) else None,
        })
    return None, None, None, {"status": "failed", "attempts": attempts}


def load_real_low_ground_truth(
    path_value: str | Path,
    *,
    allowed_types: Sequence[str],
    default_duration_hours: float,
    geocode_locations: bool = False,
    geocode_cache_path: Optional[str | Path] = None,
    geocode_cache_only: bool = False,
    geocode_force_refresh: bool = False,
    progress_enabled: bool = False,
) -> List[GroundTruthIncident]:
    """Load corrected real-data low-level GT records.

    Expected input is evaluation/ground_truth/real/low_level_gt_corrected.json, whose entries
    contain incident_id, final_name, incident_type, location,
    start_datetime_pacific, end_datetime_pacific, and
    earliest_article_datetime_pacific.  The file currently contains textual
    locations rather than GeoJSON regions.  When geocode_locations=True, each
    textual location is resolved with GeoManager/cache so optional spatial
    matching and localization metrics can be computed.
    """
    path = Path(path_value)
    payload = read_json(path)
    rows_list: List[Tuple[str, Any]]
    if isinstance(payload, dict):
        rows_list = list(payload.items())
    elif isinstance(payload, list):
        rows_list = [(str(i), item) for i, item in enumerate(payload)]
    else:
        rows_list = []

    out: List[GroundTruthIncident] = []
    default_delta = timedelta(hours=max(0.001, float(default_duration_hours)))
    geo_manager = None
    if geocode_locations and not geocode_cache_only:
        geo_manager, geo_error = _load_geo_manager(geocode_cache_path)
        if geo_manager is None:
            log(f"low_real: GeoManager unavailable; using cached GT geometries only if present. {geo_error}")
            geocode_cache_only = True

    row_iter = progress_iter(
        rows_list,
        total=len(rows_list),
        desc="geocode low_real GT" if geocode_locations else "load low_real GT",
        enabled=bool(progress_enabled),
    )

    for key, value in row_iter:
        if not isinstance(value, dict):
            continue
        incident_id = str(value.get("incident_id") or key)
        typ = canonical_type(value.get("incident_type"), allowed_types)
        start = parse_pacific_dt(value.get("start_datetime_pacific") or value.get("start_datetime"))
        end = parse_pacific_dt(value.get("end_datetime_pacific") or value.get("end_datetime"))
        if start is not None and (end is None or end <= start):
            end = start + default_delta
        external = parse_pacific_dt(
            value.get("earliest_article_datetime_pacific")
            or value.get("earliest_article_datetime")
            or value.get("external_report_time")
        )
        gt_geometry = None
        gt_lat = None
        gt_lon = None
        geocode_summary: Dict[str, Any] = {"status": "disabled"}
        if geocode_locations:
            gt_geometry, gt_lat, gt_lon, geocode_summary = geocode_real_gt_geometry(
                value,
                geo_manager=geo_manager,
                cache_path=geocode_cache_path,
                cache_only=geocode_cache_only,
                force_refresh=geocode_force_refresh,
            )

        gt = GroundTruthIncident(
            run_name="real_all_incidents",
            incident_id=incident_id,
            location_name=str(value.get("location") or value.get("final_name") or incident_id),
            incident_type=typ,
            start_time=start,
            end_time=end,
            geometry=gt_geometry,
            representative_latitude=gt_lat,
            representative_longitude=gt_lon,
            external_report_time=external,
            gt_files=[str(path)],
        )
        setattr(gt, "geocode_summary", geocode_summary)
        # Dataclasses are not slotted; attach descriptive fields for match CSVs.
        setattr(gt, "final_name", value.get("final_name"))
        setattr(gt, "gt_key", key)
        out.append(gt)
    return out


def discover_real_low_result_dirs(
    results_root: Path,
    *,
    real_experiment_name: str,
    experiment_glob: Optional[str] = None,
    methods: Sequence[str] = (),
) -> List[Tuple[str, str, str, Path]]:
    """Discover <method>/<real_experiment>/<YYYYMMDD>/ real result folders."""
    out: List[Tuple[str, str, str, Path]] = []
    if not results_root.exists():
        return out

    if methods:
        method_filter = {m for m in methods if m}
    else:
        method_filter = set(REAL_LOW_DEFAULT_METHODS)
    method_filter -= set(REAL_LOW_EXCLUDED_METHODS)

    for method_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        method = method_dir.name
        if method in REAL_LOW_EXCLUDED_METHODS:
            continue
        if method_filter and method not in method_filter:
            continue
        for exp_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            experiment = exp_dir.name
            if experiment_glob:
                if not exp_dir.match(experiment_glob) and experiment_glob not in experiment:
                    continue
            elif experiment != real_experiment_name:
                continue
            for date_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
                if (date_dir / LOW_LEVEL_FILENAME).exists() or (date_dir / TIMING_FILENAME).exists():
                    out.append((method, experiment, date_dir.name, date_dir))
    return out


def prediction_effective_time(pred: PredictedIncident) -> Optional[datetime]:
    return pred.detection_time or pred.start_time or pred.end_time


def _prediction_centroid_signature(pred: PredictedIncident) -> str:
    geom = pred.coverage_geometry if pred.coverage_geometry is not None else pred.source_geometry
    centroid = geometry_centroid_latlon(geom)
    if centroid is None:
        return "nogeom"
    lat, lon = centroid
    return f"{lat:.3f},{lon:.3f}"


def dedupe_real_predictions(preds: Sequence[PredictedIncident]) -> Tuple[List[PredictedIncident], Dict[str, Any]]:
    """Collapse repeated saved real predictions across date folders.

    The normal leakage filter handles repeated prediction_id/candidate_ids for
    IncidentLens.  This additional pass catches exact repeated IDs for any
    method and simple same-type/same-interval/same-centroid duplicates where a
    method rewrote equivalent predictions in adjacent date folders.
    """
    def sort_key(pred: PredictedIncident) -> Tuple[datetime, str]:
        t = prediction_effective_time(pred) or datetime.max.replace(tzinfo=timezone.utc)
        return (t, pred.prediction_id)

    kept: List[PredictedIncident] = []
    seen: Dict[str, PredictedIncident] = {}
    dropped = 0
    for pred in sorted(preds, key=sort_key):
        keys: List[str] = []
        if pred.prediction_id and not re.match(r"^\d{8}_prediction_\d+$", pred.prediction_id):
            keys.append("id:" + pred.prediction_id)
        start = dt_to_iso(pred.start_time) or ""
        end = dt_to_iso(pred.end_time) or ""
        det = dt_to_iso(pred.detection_time) or ""
        keys.append(f"sig:{pred.incident_type}|{start}|{end}|{det}|{_prediction_centroid_signature(pred)}")
        if any(key in seen for key in keys):
            dropped += 1
            continue
        for key in keys:
            seen[key] = pred
        kept.append(pred)
    return kept, {
        "dedupe_real_raw_predictions": len(preds),
        "dedupe_real_kept_predictions": len(kept),
        "dedupe_real_dropped_predictions": dropped,
    }


def prediction_primary_geometry(pred: PredictedIncident) -> Optional[BaseGeometry]:
    return pred.coverage_geometry if pred.coverage_geometry is not None else pred.source_geometry


def real_prediction_spatial_relation(a: PredictedIncident, b: PredictedIncident) -> Tuple[bool, Optional[float], Optional[float]]:
    ga = prediction_primary_geometry(a)
    gb = prediction_primary_geometry(b)
    iou = geom_iou(ga, gb)
    dist = representative_distance_km(ga, gb)
    if ga is None or gb is None:
        return True, iou, dist
    return False, iou, dist


def prediction_interval_or_point(pred: PredictedIncident) -> Tuple[Optional[datetime], Optional[datetime]]:
    start = pred.start_time
    end = pred.end_time
    t = prediction_effective_time(pred)
    if start is None and end is None and t is not None:
        return t, t + timedelta(minutes=1)
    if start is None:
        start = end
    if end is None:
        end = start
    if start is not None and end is not None and end <= start:
        end = start + timedelta(minutes=1)
    return start, end


def prediction_temporal_gap_hours(a: PredictedIncident, b: PredictedIncident) -> Optional[float]:
    a0, a1 = prediction_interval_or_point(a)
    b0, b1 = prediction_interval_or_point(b)
    if a0 is None or a1 is None or b0 is None or b1 is None:
        return None
    if a1 < b0:
        return (b0 - a1).total_seconds() / 3600.0
    if b1 < a0:
        return (a0 - b1).total_seconds() / 3600.0
    return 0.0


def should_merge_real_predictions(
    a: PredictedIncident,
    b: PredictedIncident,
    *,
    max_temporal_gap_hours: float,
    max_spatial_distance_km: float,
    min_spatial_iou: float,
) -> bool:
    if a.incident_type != b.incident_type:
        return False
    gap = prediction_temporal_gap_hours(a, b)
    if gap is not None and gap > max_temporal_gap_hours:
        return False
    missing_geom, iou, dist = real_prediction_spatial_relation(a, b)
    if missing_geom:
        # Without geometry, be conservative and only merge if temporal intervals
        # overlap or nearly touch.  This prevents unrelated same-type citywide
        # detections from collapsing into one giant event.
        return gap is not None and gap <= min(1.0, max_temporal_gap_hours)
    return (iou is not None and iou >= min_spatial_iou) or (dist is not None and dist <= max_spatial_distance_km)


def merge_predicted_incidents(a: PredictedIncident, b: PredictedIncident) -> PredictedIncident:
    starts = [x for x in [a.start_time, b.start_time] if x is not None]
    ends = [x for x in [a.end_time, b.end_time] if x is not None]
    dets = [x for x in [a.detection_time, b.detection_time] if x is not None]
    source_geom = union_geometries([a.source_geometry, b.source_geometry])
    cov_geom = union_geometries([a.coverage_geometry, b.coverage_geometry])
    pred = PredictedIncident(
        run_name="real_all_incidents",
        prediction_id=f"merged:{a.prediction_id}|{b.prediction_id}",
        incident_type=a.incident_type,
        detection_time=min(dets) if dets else None,
        start_time=min(starts) if starts else None,
        end_time=max(ends) if ends else None,
        source_geometry=source_geom,
        coverage_geometry=cov_geom,
        confidence=max([x for x in [a.confidence, b.confidence] if x is not None], default=None),
        state="merged",
    )
    merged_ids = list(dict.fromkeys(
        list(getattr(a, "merged_prediction_ids", [a.prediction_id])) +
        list(getattr(b, "merged_prediction_ids", [b.prediction_id]))
    ))
    setattr(pred, "merged_prediction_ids", merged_ids)
    setattr(pred, "rationale_type_guess", bool(
        real_prediction_is_rationale_type_guess(a) or real_prediction_is_rationale_type_guess(b)
    ))
    # A merged record is one evaluation unit for FP accounting.  Keep the
    # group ID stable and distinct from individual type-alternative children.
    setattr(pred, "source_prediction_id", "merged:" + "|".join(merged_ids))
    return pred


def merge_real_prediction_updates(
    preds: Sequence[PredictedIncident],
    *,
    enabled: bool,
    max_temporal_gap_hours: float,
    max_spatial_distance_km: float,
    min_spatial_iou: float,
    progress_enabled: bool = False,
    progress_desc: Optional[str] = None,
) -> Tuple[List[PredictedIncident], Dict[str, Any]]:
    """Merge repeated/update incident records before dataset-level scoring.

    Real outputs are saved by date, and IncidentLens can emit updated records for
    the same ongoing real incident across multiple date folders.  One-to-one GT
    matching should count that as one detected event, not one TP plus many FPs.
    """
    if not enabled or not preds:
        return list(preds), {
            "merge_real_updates_enabled": bool(enabled),
            "merge_real_raw_predictions": len(preds),
            "merge_real_kept_predictions": len(preds),
            "merge_real_dropped_or_absorbed_predictions": 0,
            "merge_real_num_clusters": len(preds),
        }

    ordered = sorted(preds, key=lambda p: (prediction_effective_time(p) or datetime.max.replace(tzinfo=timezone.utc), p.prediction_id))
    clusters: List[PredictedIncident] = []
    absorbed = 0
    comparisons = 0
    merge_start = time.perf_counter()

    # This pass can dominate low_real runtime for methods that emit tens of
    # thousands of date-level predictions. It is a greedy update-merging pass:
    # each prediction is compared to existing clusters, so worst-case work is
    # roughly O(num_predictions * num_clusters), with expensive geometry checks.
    # Show progress here so the outer "match low_real methods" bar does not look
    # frozen between "after exact dedupe" and "after update merge".
    merge_iter = progress_iter(
        ordered,
        total=len(ordered),
        desc=progress_desc or "merge real updates",
        enabled=progress_enabled,
        unit="pred",
        leave=False,
        position=1,
        mininterval=0.5,
    )

    for pred_idx, pred in enumerate(merge_iter, start=1):
        best_idx = None
        best_score = -1.0
        for idx, current in enumerate(clusters):
            comparisons += 1
            if not should_merge_real_predictions(
                current,
                pred,
                max_temporal_gap_hours=max_temporal_gap_hours,
                max_spatial_distance_km=max_spatial_distance_km,
                min_spatial_iou=min_spatial_iou,
            ):
                continue
            _missing, iou, dist = real_prediction_spatial_relation(current, pred)
            gap = prediction_temporal_gap_hours(current, pred)
            score = float(iou or 0.0) + 1.0 / (1.0 + float(dist if dist is not None else max_spatial_distance_km)) + 1.0 / (1.0 + float(gap or 0.0))
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            setattr(pred, "merged_prediction_ids", list(getattr(pred, "merged_prediction_ids", [pred.prediction_id])))
            clusters.append(pred)
        else:
            clusters[best_idx] = merge_predicted_incidents(clusters[best_idx], pred)
            absorbed += 1

        if progress_enabled and tqdm is not None and hasattr(merge_iter, "set_postfix"):
            # Keep this lightweight; tqdm will throttle terminal refreshes via mininterval.
            elapsed = max(1e-9, time.perf_counter() - merge_start)
            merge_iter.set_postfix({
                "clusters": len(clusters),
                "absorbed": absorbed,
                "cmp": comparisons,
                "cmp/s": f"{comparisons / elapsed:.0f}",
            })

    return clusters, {
        "merge_real_updates_enabled": True,
        "merge_real_raw_predictions": len(preds),
        "merge_real_kept_predictions": len(clusters),
        "merge_real_dropped_or_absorbed_predictions": absorbed,
        "merge_real_num_clusters": len(clusters),
        "merge_real_cluster_comparisons": comparisons,
        "merge_real_elapsed_seconds": time.perf_counter() - merge_start,
        "merge_real_max_temporal_gap_hours": max_temporal_gap_hours,
        "merge_real_max_spatial_distance_km": max_spatial_distance_km,
        "merge_real_min_spatial_iou": min_spatial_iou,
    }


def _real_prediction_time_ok(
    pred: PredictedIncident,
    gt: GroundTruthIncident,
    *,
    window_hours: float,
) -> Tuple[bool, Optional[float], Optional[float]]:
    """Return (candidate_ok, temporal_iou, gap_hours) for real matching."""
    window = timedelta(hours=max(0.0, float(window_hours)))
    gt_start, gt_end = gt.start_time, gt.end_time
    if gt_start is None and gt_end is None:
        return True, None, None
    if gt_start is None:
        gt_start = gt_end
    if gt_end is None:
        gt_end = gt_start
    if gt_start is None or gt_end is None:
        return True, None, None
    if gt_end <= gt_start:
        gt_end = gt_start + timedelta(minutes=1)

    pred_start = pred.start_time
    pred_end = pred.end_time
    pred_time = prediction_effective_time(pred)

    t_iou = None
    if pred_start is not None and pred_end is not None and pred_end > pred_start:
        t_iou = temporal_iou(gt_start, gt_end, pred_start, pred_end)
        if t_iou is not None and t_iou > 0:
            return True, t_iou, 0.0
        # Gap between closed intervals.
        if pred_end < gt_start:
            gap = gt_start - pred_end
        elif pred_start > gt_end:
            gap = pred_start - gt_end
        else:
            gap = timedelta(0)
        gap_hours = gap.total_seconds() / 3600.0
        return gap <= window, t_iou, gap_hours

    if pred_time is None:
        return False, t_iou, None
    if gt_start - window <= pred_time <= gt_end + window:
        if gt_start <= pred_time <= gt_end:
            gap_hours = 0.0
        elif pred_time < gt_start:
            gap_hours = (gt_start - pred_time).total_seconds() / 3600.0
        else:
            gap_hours = (pred_time - gt_end).total_seconds() / 3600.0
        return True, t_iou, gap_hours
    if pred_time < gt_start:
        return False, t_iou, (gt_start - pred_time).total_seconds() / 3600.0
    return False, t_iou, (pred_time - gt_end).total_seconds() / 3600.0


def evaluate_real_low_prediction_set(
    *,
    method: str,
    experiment: str,
    gt_items: List[GroundTruthIncident],
    pred_items: List[PredictedIncident],
    temporal_match_window_hours: float,
    use_spatial_matching: bool = False,
    max_loc_error_km: float = 20.0,
    min_spatial_iou: float = 0.01,
    progress_enabled: bool = False,
    progress_desc: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """Dataset-level real-data low-level evaluation.

    Matching is one-to-one over the full real dataset, not per date folder,
    because GT incidents can span several date result folders.  Candidates are
    matched by canonical incident type and temporal overlap/window.  For low_real,
    the CLI now enables use_spatial_matching by default when GT geometries are
    available, so a candidate must also satisfy the same distance/IoU spatial gate
    used by synthetic runs unless --real-no-spatial-matching is passed.
    """
    candidates: List[Tuple[float, int, int, Dict[str, Any]]] = []
    pred_group_ids = [real_prediction_group_id(pred) for pred in pred_items]
    pred_group_set = set(pred_group_ids)
    primary_pred_by_group = real_primary_prediction_by_group(pred_items)

    # The expensive part of low_real matching is the nested prediction x GT
    # candidate scan.  Show progress over candidate pairs, not just over the
    # eight methods, so long-running methods expose a meaningful ETA.  Updating
    # once per prediction keeps tqdm overhead low even when the pair count is
    # large.
    candidate_pair_total = len(pred_items) * len(gt_items)
    pair_progress = None
    if progress_enabled and tqdm is not None:
        pair_progress = tqdm(
            total=candidate_pair_total,
            desc=progress_desc or f"candidate pairs {method}",
            unit="pair",
            leave=False,
            position=1,
            dynamic_ncols=True,
            mininterval=0.5,
        )

    for pi, pred in enumerate(pred_items):
        pred_time = prediction_effective_time(pred)
        for gi, gt in enumerate(gt_items):
            type_correct = pred.incident_type == gt.incident_type
            if not type_correct:
                continue
            ok, t_iou, gap_hours = _real_prediction_time_ok(
                pred,
                gt,
                window_hours=temporal_match_window_hours,
            )
            if not ok:
                continue

            s_iou = geom_iou(pred.source_geometry, gt.geometry)
            c_iou = geom_iou(pred.coverage_geometry, gt.geometry)
            best_iou = max([x for x in [s_iou, c_iou] if x is not None], default=0.0)
            s_err = representative_distance_km(pred.source_geometry, gt.geometry)
            c_err = representative_distance_km(pred.coverage_geometry, gt.geometry)
            best_err = min([x for x in [s_err, c_err] if x is not None], default=None)
            spatial_applicable = bool(use_spatial_matching and gt.geometry is not None)
            if spatial_applicable:
                spatial_ok = (best_iou is not None and best_iou >= min_spatial_iou) or (
                    best_err is not None and best_err <= max_loc_error_km
                )
                if not spatial_ok:
                    continue

            # Prefer temporal overlap, smaller spatial error, then small temporal gap,
            # then earlier system reports.
            loc_score = 0.0
            if best_err is not None:
                loc_score = max(0.0, 1.0 - best_err / max(max_loc_error_km, 1e-6))
            gap_score = 1.0 / (1.0 + float(gap_hours or 0.0))
            early_score = 0.0
            if pred_time is not None and gt.external_report_time is not None:
                early_score = 0.1 if pred_time <= gt.external_report_time else 0.0
            score = 2.0 + float(t_iou or 0.0) + float(best_iou or 0.0) + loc_score + gap_score + early_score
            candidates.append((score, pi, gi, {
                "temporal_iou": t_iou,
                "temporal_gap_hours": gap_hours,
                "type_correct": True,
                "source_iou": s_iou,
                "coverage_iou": c_iou,
                "best_iou": best_iou,
                "source_loc_error_km": s_err,
                "coverage_loc_error_km": c_err,
                "spatial_matching_applied": spatial_applicable,
            }))
        if pair_progress is not None:
            pair_progress.update(len(gt_items))
            # Avoid very frequent terminal writes; tqdm applies mininterval.
            pair_progress.set_postfix({
                "pred": f"{pi + 1}/{len(pred_items)}",
                "viable": len(candidates),
            })

    if pair_progress is not None:
        pair_progress.close()
    progress_write(
        f"  {method}/{experiment}: scanned candidate_pairs={candidate_pair_total} "
        f"viable_candidates={len(candidates)}; sorting/selecting matches",
        enabled=progress_enabled,
    )

    candidates.sort(key=lambda x: x[0], reverse=True)
    used_preds: set[int] = set()
    used_pred_groups: set[str] = set()
    used_gts: set[int] = set()
    matches: List[Dict[str, Any]] = []
    confusion: Dict[str, Dict[str, int]] = {}

    candidate_selection_iter = progress_iter(
        candidates,
        total=len(candidates),
        desc=f"select matches {method}",
        enabled=bool(progress_enabled and candidates),
        unit="cand",
        leave=False,
        position=1,
        mininterval=0.5,
    )
    for score, pi, gi, metrics in candidate_selection_iter:
        pred_group = real_prediction_group_id(pred_items[pi])
        if pred_group in used_pred_groups or gi in used_gts:
            continue
        used_preds.add(pi)
        used_pred_groups.add(pred_group)
        used_gts.add(gi)
        pred = pred_items[pi]
        gt = gt_items[gi]
        pred_time = prediction_effective_time(pred)
        delay_from_start = None
        if pred_time is not None and gt.start_time is not None:
            delay_from_start = (pred_time - gt.start_time).total_seconds() / 60.0
        external_delay = None
        before_external = None
        if pred_time is not None and gt.external_report_time is not None:
            external_delay = (pred_time - gt.external_report_time).total_seconds() / 60.0
            before_external = pred_time <= gt.external_report_time

        confusion.setdefault(gt.incident_type, {})
        confusion[gt.incident_type][pred.incident_type] = int(confusion[gt.incident_type].get(pred.incident_type, 0)) + 1

        matches.append({
            "run_name": "real_all_incidents",
            "method": method,
            "experiment": experiment,
            "gt_type": gt.incident_type,
            "pred_type": pred.incident_type,
            "gt_id": gt.incident_id,
            "gt_final_name": getattr(gt, "final_name", None),
            "gt_location": gt.location_name,
            "pred_id": pred.prediction_id,
            "pred_source_prediction_id": real_prediction_group_id(pred),
            "pred_is_rationale_type_guess": real_prediction_is_rationale_type_guess(pred),
            "type_correct": True,
            "temporal_iou": metrics.get("temporal_iou"),
            "temporal_gap_hours": metrics.get("temporal_gap_hours"),
            "detection_delay_minutes": delay_from_start,
            "system_report_time": dt_to_iso(pred_time),
            "gt_start_time": dt_to_iso(gt.start_time),
            "gt_end_time": dt_to_iso(gt.end_time),
            "earliest_article_time": dt_to_iso(gt.external_report_time),
            "external_report_delay_minutes": external_delay,
            "detected_before_external_report": before_external,
            "source_iou": metrics.get("source_iou"),
            "coverage_iou": metrics.get("coverage_iou"),
            "best_iou": metrics.get("best_iou"),
            "source_loc_error_km": metrics.get("source_loc_error_km"),
            "coverage_loc_error_km": metrics.get("coverage_loc_error_km"),
            "spatial_matching_applied": metrics.get("spatial_matching_applied"),
        })

    tp = len(matches)
    fp = max(0, len(pred_group_set) - len(used_pred_groups))
    fn = max(0, len(gt_items) - len(used_gts))

    per_type_counts: Dict[str, Dict[str, int]] = {}
    all_types = {g.incident_type for g in gt_items} | {p.incident_type for p in pred_items}
    for typ in all_types:
        per_type_counts[typ] = {"tp": 0, "fp": 0, "fn": 0}
    for m in matches:
        per_type_counts.setdefault(str(m["gt_type"]), {"tp": 0, "fp": 0, "fn": 0})["tp"] += 1
    for gid, pred in primary_pred_by_group.items():
        if gid not in used_pred_groups:
            per_type_counts.setdefault(pred.incident_type, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
            confusion.setdefault("__false_positive__", {})
            confusion["__false_positive__"][pred.incident_type] = int(confusion["__false_positive__"].get(pred.incident_type, 0)) + 1
    for idx, gt in enumerate(gt_items):
        if idx not in used_gts:
            per_type_counts.setdefault(gt.incident_type, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1
            confusion.setdefault(gt.incident_type, {})
            confusion[gt.incident_type]["__missed__"] = int(confusion[gt.incident_type].get("__missed__", 0)) + 1

    gt_type_conditioned_counts: Dict[str, Dict[str, int]] = {}
    for typ in sorted({g.incident_type for g in gt_items}):
        num_gt_for_type = sum(1 for g in gt_items if g.incident_type == typ)
        correct_tp_for_type = sum(1 for m in matches if m.get("gt_type") == typ)
        preds_for_type_groups = {real_prediction_group_id(p) for p in pred_items if p.incident_type == typ}
        gt_type_conditioned_counts[typ] = {
            "tp": correct_tp_for_type,
            "fp": max(0, len(preds_for_type_groups) - correct_tp_for_type),
            "fn": max(0, num_gt_for_type - correct_tp_for_type),
            "num_gt": num_gt_for_type,
            "num_predictions": len(preds_for_type_groups),
        }

    scenario_union_by_gt_type = scenario_union_metrics_by_gt_type(gt_items, pred_items)
    scenario_union_ious = [v.get("scenario_union_coverage_iou") for v in scenario_union_by_gt_type.values()]
    scenario_overcoverage_ratios = [v.get("scenario_overcoverage_ratio") for v in scenario_union_by_gt_type.values()]
    scenario_temporal_ious = [v.get("scenario_temporal_iou") for v in scenario_union_by_gt_type.values()]
    scenario_temporal_overcoverage_ratios = [v.get("scenario_temporal_overcoverage_ratio") for v in scenario_union_by_gt_type.values()]

    external_delays = [safe_float(m.get("external_report_delay_minutes"), None) for m in matches]
    start_delays = [safe_float(m.get("detection_delay_minutes"), None) for m in matches]
    pre_report_flags = [m.get("detected_before_external_report") for m in matches]
    pre_report_count = sum(1 for x in pre_report_flags if x is True)

    metrics: Dict[str, Any] = {
        "method": method,
        "experiment": experiment,
        "run_name": "real_all_incidents",
        "result_dir": None,
        "num_date_result_dirs": None,
        "num_gt": len(gt_items),
        "num_predictions": len(pred_group_set),
        "num_prediction_type_labels": len(pred_items),
        "num_rationale_type_guess_labels": sum(1 for p in pred_items if real_prediction_is_rationale_type_guess(p)),
        "num_matches": len(matches),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "false_positives": fp,
        "false_positives_per_run": fp,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": fbeta(tp, fp, fn, beta=1.0),
        "f2": fbeta(tp, fp, fn, beta=2.0),
        "median_detection_delay_minutes": median(start_delays),
        "fraction_detected_before_external_report": fraction_true(pre_report_flags),
        "pre_report_recall": ratio(pre_report_count, len(gt_items)),
        "matched_pre_report_fraction": ratio(pre_report_count, len(matches)),
        "median_external_report_delay_minutes": median(external_delays),
        "p25_external_report_delay_minutes": percentile(external_delays, 25),
        "p75_external_report_delay_minutes": percentile(external_delays, 75),
        "external_report_delay_definition": "system_report_time - earliest_article_datetime_pacific; negative means system detected before article publication",
        "gt_incident_types": ";".join(sorted({g.incident_type for g in gt_items})),
        "pred_incident_types": ";".join(sorted({p.incident_type for p in pred_items})),
        "matching_policy": "canonical_type_temporal_and_optional_spatial_gt_geometry" if use_spatial_matching else "canonical_type_and_temporal_overlap_or_window_no_spatial_gt_geometry",
        "real_temporal_match_window_hours": float(temporal_match_window_hours),
        "real_spatial_matching_used": bool(use_spatial_matching),
        "num_gt_with_geometry": sum(1 for g in gt_items if g.geometry is not None),
        "median_source_localization_error_km": median([safe_float(m.get("source_loc_error_km"), None) for m in matches]),
        "median_coverage_localization_error_km": median([safe_float(m.get("coverage_loc_error_km"), None) for m in matches]),
        "mean_source_iou": mean([safe_float(m.get("source_iou"), None) for m in matches]),
        "mean_coverage_iou": mean([safe_float(m.get("coverage_iou"), None) for m in matches]),
        "median_temporal_iou": median([safe_float(m.get("temporal_iou"), None) for m in matches]),
        "matched_tp_median_source_or_proxy_localization_error_km": median([safe_float(m.get("source_loc_error_km"), None) for m in matches]),
        "matched_tp_median_coverage_localization_error_km": median([safe_float(m.get("coverage_loc_error_km"), None) for m in matches]),
        "matched_tp_mean_source_or_proxy_iou": mean([safe_float(m.get("source_iou"), None) for m in matches]),
        "matched_tp_mean_coverage_iou": mean([safe_float(m.get("coverage_iou"), None) for m in matches]),
        "matched_tp_median_temporal_iou": median([safe_float(m.get("temporal_iou"), None) for m in matches]),
        "mean_scenario_union_coverage_iou": mean(scenario_union_ious),
        "median_scenario_union_coverage_iou": median(scenario_union_ious),
        "mean_scenario_overcoverage_ratio": mean(scenario_overcoverage_ratios),
        "median_scenario_overcoverage_ratio": median(scenario_overcoverage_ratios),
        "mean_scenario_temporal_iou": mean(scenario_temporal_ious),
        "median_scenario_temporal_iou": median(scenario_temporal_ious),
        "mean_scenario_temporal_overcoverage_ratio": mean(scenario_temporal_overcoverage_ratios),
        "median_scenario_temporal_overcoverage_ratio": median(scenario_temporal_overcoverage_ratios),
        "num_predictions_with_geometry": sum(1 for p in pred_items if prediction_primary_geometry(p) is not None),
        "num_matches_with_spatial_metrics": sum(1 for m in matches if m.get("source_loc_error_km") is not None or m.get("coverage_loc_error_km") is not None or m.get("source_iou") is not None or m.get("coverage_iou") is not None),
        "num_matches_with_temporal_iou": sum(1 for m in matches if m.get("temporal_iou") is not None),
        "scenario_metrics_num_gt_types": len(scenario_union_by_gt_type),
        "scenario_metrics_num_types_with_spatial_iou": sum(1 for v in scenario_union_by_gt_type.values() if v.get("scenario_union_coverage_iou") is not None),
        "scenario_metrics_num_types_with_temporal_iou": sum(1 for v in scenario_union_by_gt_type.values() if v.get("scenario_temporal_iou") is not None),
        "_per_type_counts": per_type_counts,
        "_gt_type_conditioned_counts": gt_type_conditioned_counts,
        "_scenario_union_metrics_by_gt_type": scenario_union_by_gt_type,
    }
    return metrics, matches, confusion



def real_low_metrics_by_gt_incident_name(
    *,
    method: str,
    experiment: str,
    gt_items: List[GroundTruthIncident],
    pred_items: List[PredictedIncident],
    matches: List[Dict[str, Any]],
    temporal_match_window_hours: float,
    use_spatial_matching: bool = False,
    max_loc_error_km: float = 20.0,
    min_spatial_iou: float = 0.01,
) -> List[Dict[str, Any]]:
    """Return one low_real performance row per GT incident name.

    The normal low_real metrics evaluate an entire real dataset per method.  For
    paper/debugging use it is also useful to ask whether each named GT incident
    (e.g., Eaton Fire, Palisades Fire) was detected.  Each row treats the GT
    incident as one label:
      * TP = 1 if the one-to-one dataset-level matcher assigned a prediction.
      * FN = 1 if it was not assigned a prediction.
      * FP = same-type predictions that pass this incident's temporal/window
        gate (and spatial gate when enabled), excluding the matched prediction.

    FP here is therefore an incident-local clutter count, not a globally
    additive FP accounting.  It is useful for understanding why an incident has
    low precision even when it is detected.
    """
    match_by_gt_id: Dict[str, Dict[str, Any]] = {}
    for m in matches:
        gt_id = str(m.get("gt_id") or "")
        if gt_id and gt_id not in match_by_gt_id:
            match_by_gt_id[gt_id] = m

    out: List[Dict[str, Any]] = []
    for gt in sorted(
        gt_items,
        key=lambda g: (
            str(getattr(g, "final_name", "") or ""),
            str(g.incident_id),
        ),
    ):
        candidate_preds: List[PredictedIncident] = []
        temporal_type_candidate_groups: set[str] = set()
        spatial_rejected_groups: set[str] = set()
        candidate_spatial_ious: List[Optional[float]] = []
        candidate_spatial_errors: List[Optional[float]] = []
        spatial_gate_applicable = bool(use_spatial_matching and gt.geometry is not None)

        for pred in pred_items:
            if pred.incident_type != gt.incident_type:
                continue
            ok, _t_iou, _gap_hours = _real_prediction_time_ok(
                pred,
                gt,
                window_hours=temporal_match_window_hours,
            )
            if not ok:
                continue
            pred_group = real_prediction_group_id(pred)
            temporal_type_candidate_groups.add(pred_group)

            s_iou = geom_iou(pred.source_geometry, gt.geometry)
            c_iou = geom_iou(pred.coverage_geometry, gt.geometry)
            best_iou = max([x for x in [s_iou, c_iou] if x is not None], default=None)
            s_err = representative_distance_km(pred.source_geometry, gt.geometry)
            c_err = representative_distance_km(pred.coverage_geometry, gt.geometry)
            best_err = min([x for x in [s_err, c_err] if x is not None], default=None)

            spatial_ok = True
            if spatial_gate_applicable:
                spatial_ok = (best_iou is not None and best_iou >= min_spatial_iou) or (
                    best_err is not None and best_err <= max_loc_error_km
                )
            if not spatial_ok:
                spatial_rejected_groups.add(pred_group)
                continue

            candidate_spatial_ious.append(best_iou)
            candidate_spatial_errors.append(best_err)
            candidate_preds.append(pred)

        # Collapse same source/update/rationale alternative group for local FP
        # accounting and union metrics.  Prefer the earliest effective record in
        # a group so delay-like summaries remain conservative.
        unique_candidate_by_group: Dict[str, PredictedIncident] = {}
        for pred in sorted(
            candidate_preds,
            key=lambda p: prediction_effective_time(p) or datetime.max.replace(tzinfo=timezone.utc),
        ):
            unique_candidate_by_group.setdefault(real_prediction_group_id(pred), pred)
        unique_candidates = list(unique_candidate_by_group.values())
        candidate_group_ids = set(unique_candidate_by_group.keys())

        match = match_by_gt_id.get(str(gt.incident_id))
        matched_group = str(match.get("pred_source_prediction_id") or match.get("pred_id") or "") if match else ""
        tp = 1 if match is not None else 0
        fn = 0 if match is not None else 1
        fp = max(0, len(candidate_group_ids) - tp)

        pred_geom_union = union_geometries(
            prediction_primary_geometry(pred) for pred in unique_candidates
        )
        spatial_metrics = geom_overlap_metrics(pred_geom_union, gt.geometry)
        temporal_metrics = temporal_overlap_metrics(
            (prediction_interval_or_point(pred) for pred in unique_candidates),
            [(gt.start_time, gt.end_time)],
        )

        external_delay = safe_float(match.get("external_report_delay_minutes"), None) if match else None
        start_delay = safe_float(match.get("detection_delay_minutes"), None) if match else None
        before_external = match.get("detected_before_external_report") if match else None

        row: Dict[str, Any] = {
            "method": method,
            "experiment": experiment,
            "run_name": "real_all_incidents",
            "gt_id": gt.incident_id,
            "gt_final_name": getattr(gt, "final_name", None) or gt.incident_id,
            "gt_type": gt.incident_type,
            "gt_location": gt.location_name,
            "gt_start_time": dt_to_iso(gt.start_time),
            "gt_end_time": dt_to_iso(gt.end_time),
            "earliest_article_time": dt_to_iso(gt.external_report_time),
            "gt_has_geometry": gt.geometry is not None,
            "gt_geometry_area_km2": geom_area_km2(gt.geometry),
            "gt_representative_latitude": gt.representative_latitude,
            "gt_representative_longitude": gt.representative_longitude,
            "matched": bool(match is not None),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn),
            "f1": fbeta(tp, fp, fn, beta=1.0),
            "f2": fbeta(tp, fp, fn, beta=2.0),
            "num_temporal_type_candidate_prediction_groups": len(temporal_type_candidate_groups),
            "num_candidate_prediction_groups": len(candidate_group_ids),
            "num_spatial_rejected_candidate_groups": len(spatial_rejected_groups),
            "num_candidate_prediction_labels": len(candidate_preds),
            "candidate_prediction_group_ids": ";".join(sorted(candidate_group_ids)[:200]),
            "matched_pred_id": match.get("pred_id") if match else None,
            "matched_pred_group_id": matched_group or None,
            "matched_pred_type": match.get("pred_type") if match else None,
            "matched_pred_is_rationale_type_guess": match.get("pred_is_rationale_type_guess") if match else None,
            "matched_system_report_time": match.get("system_report_time") if match else None,
            "detection_delay_minutes": start_delay,
            "external_report_delay_minutes": external_delay,
            "detected_before_external_report": before_external,
            "temporal_iou": safe_float(match.get("temporal_iou"), None) if match else None,
            "temporal_gap_hours": safe_float(match.get("temporal_gap_hours"), None) if match else None,
            "source_iou": safe_float(match.get("source_iou"), None) if match else None,
            "coverage_iou": safe_float(match.get("coverage_iou"), None) if match else None,
            "best_iou": safe_float(match.get("best_iou"), None) if match else None,
            "source_loc_error_km": safe_float(match.get("source_loc_error_km"), None) if match else None,
            "coverage_loc_error_km": safe_float(match.get("coverage_loc_error_km"), None) if match else None,
            "spatial_matching_applied": spatial_gate_applicable,
            "candidate_median_best_iou": median(candidate_spatial_ious),
            "candidate_median_best_loc_error_km": median(candidate_spatial_errors),
            "event_local_fp_definition": (
                "same-type predictions passing this GT incident's temporal/window gate "
                "and optional spatial gate, excluding the matched prediction"
            ),
        }
        row.update({f"event_{k}": v for k, v in spatial_metrics.items()})
        row.update({f"event_{k}": v for k, v in temporal_metrics.items()})
        out.append(row)
    return out

def aggregate_real_prediction_date_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("method")), str(row.get("experiment")), str(row.get("date")))
        g = groups.setdefault(key, {
            "method": key[0],
            "experiment": key[1],
            "date": key[2],
            "num_predictions_raw": 0,
            "num_predictions_after_filter": 0,
            "num_predictions_dropped_leaked": 0,
        })
        g["num_predictions_raw"] += int(row.get("num_predictions_raw") or 0)
        g["num_predictions_after_filter"] += int(row.get("num_predictions_after_filter") or 0)
        g["num_predictions_dropped_leaked"] += int(row.get("num_predictions_dropped_leaked") or 0)
    return list(sorted(groups.values(), key=lambda x: (x["method"], x["experiment"], x["date"])))



# ---------------------------------------------------------------------------
# Coverage source helpers and coverage/performance correlation diagnostics
# ---------------------------------------------------------------------------

LABELLING_SOURCE_TO_MODALITY = {
    "air_data": "air_quality",
    "alertcalifornia": "camera",
    "cctv": "camera",
    "citizen_data": "text_alert",
    "twitter_data": "social",
    "x_data": "social",
    "pem_data_station_5min": "traffic",
    "pems": "traffic",
    "weather_data": "weather",
}


def _normalize_loose_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _load_labelling_relevance_records(paths: Sequence[str | Path]) -> Dict[str, Dict[str, Any]]:
    """Load precomputed evaluation/labelling relevance JSON records.

    This consumes files shaped like evaluation/merged_incidents/all_merged_by_id_relevant.json,
    whose top-level keys are incident IDs and whose values include temporal_relevant
    and spatial_relevant source lists.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for raw_path in paths or []:
        path = Path(raw_path)
        if not path.exists():
            log(f"WARNING: --coverage-labelling-json path does not exist: {path}")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log(f"WARNING: could not read --coverage-labelling-json {path}: {exc}")
            continue
        if isinstance(payload, list):
            iterator = ((str(i), item) for i, item in enumerate(payload))
        elif isinstance(payload, dict):
            iterator = payload.items()
        else:
            continue
        for key, record in iterator:
            if not isinstance(record, dict):
                continue
            rec = dict(record)
            rec.setdefault("incident_id", key)
            keys = {
                str(key),
                str(rec.get("incident_id") or ""),
                str(rec.get("final_name") or ""),
                str(rec.get("representative_name") or ""),
                str(rec.get("name") or ""),
            }
            for k in list(keys):
                if k:
                    keys.add(_normalize_loose_key(k))
            for k in keys:
                if k:
                    out.setdefault(k, rec)
    return out


def _find_labelling_record_for_coverage_row(row: Mapping[str, Any], labelling_index: Mapping[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    candidates = [
        row.get("gt_id"),
        row.get("coverage_key"),
        row.get("gt_final_name"),
        row.get("gt_location"),
    ]
    for value in list(candidates):
        if value:
            candidates.append(_normalize_loose_key(value))
    for key in candidates:
        if key and str(key) in labelling_index:
            return labelling_index[str(key)]
    return None


def _noisy_or_simple(values: Iterable[float], *, cap: float = 0.98) -> float:
    miss = 1.0
    any_value = False
    for value in values:
        q = max(0.0, min(1.0, float(value)))
        miss *= 1.0 - q
        any_value = True
    if not any_value:
        return 0.0
    return max(0.0, min(float(cap), 1.0 - miss))


def _labelling_record_to_coverage_components(row: Mapping[str, Any], record: Mapping[str, Any]) -> Dict[str, Any]:
    temporal_sources = sorted({str(x) for x in (record.get("temporal_relevant") or []) if str(x)})
    spatial_sources = sorted({str(x) for x in (record.get("spatial_relevant") or []) if str(x)})

    # Text/social alert streams do not have polygons in the older labelling code;
    # if they are temporally present, count them as source evidence rather than
    # forcing them to zero spatial relevance.
    effective_spatial_sources = set(spatial_sources)
    for source in temporal_sources:
        if source in {"citizen_data", "twitter_data", "x_data"}:
            effective_spatial_sources.add(source)

    temporal_modalities = sorted({LABELLING_SOURCE_TO_MODALITY.get(s, COVERAGE_SOURCE_TYPE_TO_MODALITY.get(s, s)) for s in temporal_sources})
    spatial_modalities = sorted({LABELLING_SOURCE_TO_MODALITY.get(s, COVERAGE_SOURCE_TYPE_TO_MODALITY.get(s, s)) for s in effective_spatial_sources})
    modalities = sorted(set(temporal_modalities) | set(spatial_modalities))

    gt_type = str(row.get("gt_type") or record.get("representative_event_type") or "unknown")
    if coverage_canonical_incident_type is not None:
        gt_type = coverage_canonical_incident_type(gt_type)

    if coverage_modality_coverage is not None:
        c_mod = coverage_modality_coverage(gt_type, modalities)
    else:
        c_mod = _noisy_or_simple((0.6 for _m in modalities), cap=0.95)

    # Because the labelling file stores only source-family presence/intersection,
    # not per-sensor counts, source coverage is a family-level estimate.
    source_terms = []
    for source in effective_spatial_sources:
        source_terms.append(0.80 if source in spatial_sources else 0.45)
    for source in temporal_sources:
        if source not in effective_spatial_sources:
            source_terms.append(0.30)
    c_src = _noisy_or_simple(source_terms, cap=0.98)

    weak_evidence = []
    for modality in modalities:
        ev = COVERAGE_MODALITY_TO_WEAK_EVIDENCE.get(modality)
        if ev:
            weak_evidence.append(ev)
    if coverage_inference_coverage is not None:
        c_inf = coverage_inference_coverage(gt_type, weak_evidence)
    else:
        c_inf = _noisy_or_simple((0.45 for _ev in weak_evidence), cap=0.95)

    score = 0.25 * c_mod + 0.35 * c_src + 0.40 * c_inf
    return {
        "labelling_coverage_score": round(max(0.0, min(1.0, score)), 6),
        "labelling_modality_coverage": round(max(0.0, min(1.0, c_mod)), 6),
        "labelling_source_coverage": round(max(0.0, min(1.0, c_src)), 6),
        "labelling_inference_coverage": round(max(0.0, min(1.0, c_inf)), 6),
        "labelling_temporal_relevant_sources": ";".join(temporal_sources),
        "labelling_spatial_relevant_sources": ";".join(spatial_sources),
        "labelling_effective_spatial_sources": ";".join(sorted(effective_spatial_sources)),
        "labelling_available_modalities": ";".join(modalities),
        "labelling_coverage_formula": "0.25*modality + 0.35*source_family + 0.40*inference",
    }


def apply_labelling_coverage_overrides(
    coverage_rows: List[Dict[str, Any]],
    *,
    paths: Sequence[str | Path],
    as_primary: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Attach/optionally use older labelling relevance as coverage scores."""
    if not paths:
        return coverage_rows, {"enabled": False}
    labelling_index = _load_labelling_relevance_records(paths)
    matched = 0
    for row in coverage_rows:
        record = _find_labelling_record_for_coverage_row(row, labelling_index)
        if record is None:
            row["labelling_coverage_match"] = False
            continue
        matched += 1
        row["labelling_coverage_match"] = True
        row.update(_labelling_record_to_coverage_components(row, record))
        if as_primary:
            row["coverage_score_original"] = row.get("coverage_score")
            row["modality_coverage_original"] = row.get("modality_coverage")
            row["source_coverage_original"] = row.get("source_coverage")
            row["inference_coverage_original"] = row.get("inference_coverage")
            row["coverage_score"] = row.get("labelling_coverage_score")
            row["modality_coverage"] = row.get("labelling_modality_coverage")
            row["source_coverage"] = row.get("labelling_source_coverage")
            row["inference_coverage"] = row.get("labelling_inference_coverage")
            row["coverage_source"] = "labelling_relevance"
            row["fallback_type_default_used"] = False
    summary = {
        "enabled": True,
        "paths": [str(p) for p in paths],
        "num_labelling_index_keys": len(labelling_index),
        "num_gt_with_labelling_match": matched,
        "num_gt_without_labelling_match": max(0, len(coverage_rows) - matched),
        "used_as_primary_coverage_score": bool(as_primary),
        "interpretation": (
            "Labelling coverage is computed from temporal_relevant/spatial_relevant source-family lists. "
            "It is cleaner than prediction support, but coarser than a per-sensor source JSONL inventory."
        ),
    }
    return coverage_rows, summary


def _coverage_key_from_row(row: Mapping[str, Any]) -> str:
    gt_id = str(row.get("gt_id") or "")
    gt_location = str(row.get("gt_location") or "")
    if gt_location and gt_location != "nan" and gt_location != gt_id:
        return f"{gt_id}::{gt_location}"
    return str(row.get("coverage_key") or gt_id or row.get("gt_final_name") or "")


def attach_coverage_to_incident_rows(
    rows: List[Dict[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    cov_by_key: Dict[str, Mapping[str, Any]] = {}
    for cov in coverage_rows:
        keys = [cov.get("coverage_key"), cov.get("gt_id"), _coverage_key_from_row(cov)]
        for key in keys:
            if key:
                cov_by_key[str(key)] = cov
    for row in rows:
        key = _coverage_key_from_row(row)
        cov = cov_by_key.get(key) or cov_by_key.get(str(row.get("gt_id") or ""))
        if cov is None:
            row["coverage_score"] = None
            continue
        for col in (
            "coverage_score", "modality_coverage", "source_coverage", "inference_coverage",
            "coverage_source", "available_modalities", "evidence_types", "num_relevant_sources",
            "labelling_coverage_score", "labelling_modality_coverage", "labelling_source_coverage",
            "labelling_inference_coverage", "labelling_temporal_relevant_sources", "labelling_spatial_relevant_sources",
            "labelling_effective_spatial_sources", "labelling_available_modalities",
        ):
            if col in cov:
                row[col] = cov.get(col)
    return rows


def _pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx <= 0 or dy <= 0:
        return None
    return num / (dx * dy)


def _rank_values(values: Sequence[float]) -> List[float]:
    order = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and order[j][1] == order[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k][0]] = avg_rank
        i = j
    return ranks


def _spearman_corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    return _pearson_corr(_rank_values(xs), _rank_values(ys))


def _corr_stats(rows: Sequence[Mapping[str, Any]], x_col: str, y_col: str) -> Dict[str, Any]:
    pairs: List[Tuple[float, float]] = []
    for row in rows:
        x = safe_float(row.get(x_col), None)
        y = row.get(y_col)
        if isinstance(y, bool):
            y = 1.0 if y else 0.0
        y = safe_float(y, None)
        if x is None or y is None:
            continue
        pairs.append((float(x), float(y)))
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    return {
        "n": len(pairs),
        "pearson_r": _pearson_corr(xs, ys),
        "spearman_r": _spearman_corr(xs, ys),
        "x_mean": mean(xs),
        "y_mean": mean(ys),
    }


def coverage_bin(score: Any) -> str:
    x = safe_float(score, None)
    if x is None:
        return "unknown"
    if x < 0.33:
        return "low"
    if x < 0.66:
        return "medium"
    return "high"


def compute_coverage_performance_diagnostics(
    incident_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build per-incident coverage/performance rows, correlations, and bins."""
    per_incident: List[Dict[str, Any]] = []
    for row in incident_rows:
        out = dict(row)
        out["coverage_bin"] = coverage_bin(row.get("coverage_score"))
        # All-GT versions count unmatched incidents as zero quality.  Conditional
        # matched-only columns retain NaN/None for unmatched GTs.
        for col in (
            "temporal_iou", "best_iou", "coverage_iou", "source_iou",
            "event_scenario_coverage_recall", "event_scenario_union_coverage_iou", "event_scenario_temporal_iou",
            "recall", "f1", "f2",
        ):
            if col in out:
                out[f"{col}_all_gt"] = safe_float(out.get(col), 0.0) if out.get("matched") else 0.0
        per_incident.append(out)

    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in per_incident:
        groups.setdefault((str(row.get("method")), str(row.get("experiment"))), []).append(row)

    target_cols = [
        "matched", "recall", "f1", "f2", "temporal_iou_all_gt", "best_iou_all_gt",
        "coverage_iou_all_gt", "source_iou_all_gt", "event_scenario_coverage_recall_all_gt",
        "event_scenario_union_coverage_iou_all_gt", "event_scenario_temporal_iou_all_gt",
    ]
    conditional_cols = ["temporal_iou", "best_iou", "coverage_iou", "source_iou", "detection_delay_minutes", "external_report_delay_minutes"]
    corr_rows: List[Dict[str, Any]] = []
    for (method, experiment), rows in sorted(groups.items()):
        for target in target_cols:
            if not any(target in row for row in rows):
                continue
            stats = _corr_stats(rows, "coverage_score", target)
            corr_rows.append({
                "method": method,
                "experiment": experiment,
                "x": "coverage_score",
                "y": target,
                "condition": "all_gt",
                **stats,
            })
        matched_rows = [row for row in rows if bool(row.get("matched"))]
        for target in conditional_cols:
            if not any(target in row for row in matched_rows):
                continue
            stats = _corr_stats(matched_rows, "coverage_score", target)
            corr_rows.append({
                "method": method,
                "experiment": experiment,
                "x": "coverage_score",
                "y": target,
                "condition": "matched_only",
                **stats,
            })

    bin_groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for row in per_incident:
        bin_groups.setdefault((str(row.get("method")), str(row.get("experiment")), str(row.get("coverage_bin"))), []).append(row)
    bin_rows: List[Dict[str, Any]] = []
    for (method, experiment, bin_name), rows in sorted(bin_groups.items()):
        matched_flags = [1.0 if bool(row.get("matched")) else 0.0 for row in rows]
        bin_rows.append({
            "method": method,
            "experiment": experiment,
            "coverage_bin": bin_name,
            "num_gt": len(rows),
            "mean_coverage_score": mean([safe_float(row.get("coverage_score"), None) for row in rows]),
            "matched_recall": mean(matched_flags),
            "mean_f2": mean([safe_float(row.get("f2"), None) for row in rows]),
            "mean_temporal_iou_all_gt": mean([safe_float(row.get("temporal_iou_all_gt"), None) for row in rows]),
            "mean_best_iou_all_gt": mean([safe_float(row.get("best_iou_all_gt"), None) for row in rows]),
            "mean_event_spatial_recall_all_gt": mean([safe_float(row.get("event_scenario_coverage_recall_all_gt"), None) for row in rows]),
            "mean_event_spatial_iou_all_gt": mean([safe_float(row.get("event_scenario_union_coverage_iou_all_gt"), None) for row in rows]),
            "mean_event_temporal_iou_all_gt": mean([safe_float(row.get("event_scenario_temporal_iou_all_gt"), None) for row in rows]),
        })
    return per_incident, corr_rows, bin_rows


def default_coverage_source_jsonl_candidates(args: argparse.Namespace) -> List[Path]:
    """Return common normalized REPORT JSONL locations produced by real_emitter.

    real_emitter.py's full ordered REPORT stream is written only when
    --write-ordered-reports-jsonl is enabled.  The default path in real_emitter
    is evaluation/temp/real_emitter_ordered_reports.jsonl.  The sample report
    file is intentionally not auto-used because it contains only a few reports
    per source/date and is not a complete source inventory.
    """
    candidates: List[Path] = []
    temp_roots: List[Path] = []
    explicit = getattr(args, "coverage_auto_temp_roots", None) or []
    for value in explicit:
        temp_roots.append(Path(value))
    # Common default used by run_experiments.py/real_emitter.py.
    temp_roots.append(Path("evaluation/temp"))
    # Also try a temp root adjacent to results-root if the user runs from a
    # relocated evaluation directory.
    try:
        results_root = Path(getattr(args, "results_root", "evaluation/results"))
        if results_root.name == "results":
            temp_roots.append(results_root.parent / "temp")
    except Exception:
        pass

    seen = set()
    for root in temp_roots:
        path = root / "real_emitter_ordered_reports.jsonl"
        key = str(path)
        if key not in seen:
            seen.add(key)
            candidates.append(path)
    return [p for p in candidates if p.exists() and p.is_file()]


def default_real_emitter_profile_cache_candidates(args: argparse.Namespace) -> List[Path]:
    """Return common real_emitter profile-cache roots.

    These caches are JSON rather than JSONL, and can contain
    profile.emission_reports when real_emitter previously ran with socket
    emission or ordered-report writing enabled.
    """
    candidates: List[Path] = []
    temp_roots: List[Path] = []
    explicit = getattr(args, "coverage_auto_temp_roots", None) or []
    for value in explicit:
        temp_roots.append(Path(value))
    temp_roots.append(Path("evaluation/temp"))
    try:
        results_root = Path(getattr(args, "results_root", "evaluation/results"))
        if results_root.name == "results":
            temp_roots.append(results_root.parent / "temp")
    except Exception:
        pass

    seen = set()
    for root in temp_roots:
        path = root / "_real_emitter_profile_cache"
        key = str(path)
        if key not in seen:
            seen.add(key)
            candidates.append(path)
    return [p for p in candidates if p.exists() and p.is_dir()]

def compute_and_write_coverage_scores(
    *,
    args: argparse.Namespace,
    gt_items: Sequence[GroundTruthIncident],
    result_dirs: Sequence[Tuple[str, str, str, Path]],
    output_dir: Path,
    progress_enabled: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, Any]]:
    """Compute optional GT coverage/observability scores.

    The preferred inputs are normalized REPORT/observation JSONL files passed via
    --coverage-source-jsonl.  If absent and --coverage-use-result-support is
    enabled, this falls back to support points in low_level_results.json.  That
    fallback is explicitly marked in coverage_scores.csv because it estimates
    observed-support coverage rather than complete deployed-source coverage.
    """
    if not getattr(args, "include_coverage", False):
        return [], {}, {"enabled": False, "reason": "coverage_disabled"}
    if compute_coverage_for_gt_items is None:
        return [], {}, {"enabled": False, "reason": "coverage_metrics_import_failed"}

    source_records = []
    source_parts: List[str] = []
    explicit_source_jsonl = list(getattr(args, "coverage_source_jsonl", None) or [])
    auto_source_jsonl: List[Path] = []
    auto_profile_cache: List[Path] = []
    # Auto-discovery is intentionally restricted to low_real because the
    # default candidates are real_emitter inventories under evaluation/temp/.
    # Synthetic evaluations should either use an explicitly supplied
    # --coverage-source-jsonl or the result-support fallback; otherwise a stale
    # real-data inventory could be mixed into synthetic coverage scoring.
    if (
        not explicit_source_jsonl
        and bool(getattr(args, "coverage_auto_source_inventory", True))
        and str(getattr(args, "mode", "")) == "low_real"
    ):
        auto_source_jsonl = default_coverage_source_jsonl_candidates(args)
        auto_profile_cache = default_real_emitter_profile_cache_candidates(args)
        if auto_source_jsonl:
            log("Auto-discovered coverage source JSONL: " + ", ".join(str(p) for p in auto_source_jsonl))
        if auto_profile_cache:
            log("Auto-discovered real_emitter profile cache: " + ", ".join(str(p) for p in auto_profile_cache))
    source_jsonl_paths = explicit_source_jsonl + [str(p) for p in auto_source_jsonl]
    if source_jsonl_paths:
        try:
            source_records.extend(load_source_records_from_jsonl(source_jsonl_paths or []))
            source_parts.append("source_jsonl" if explicit_source_jsonl else "auto_source_jsonl")
        except Exception as exc:
            log(f"WARNING: could not load --coverage-source-jsonl records: {exc}")
    if auto_profile_cache and load_source_records_from_real_emitter_profile_cache is not None:
        try:
            source_records.extend(load_source_records_from_real_emitter_profile_cache(
                auto_profile_cache,
                max_records=int(getattr(args, "coverage_max_profile_cache_records", 500000)),
                include_sample_reports_when_no_emission=bool(getattr(args, "coverage_profile_cache_include_samples", False)),
                synthesize_sensor_location_records=bool(getattr(args, "coverage_profile_cache_synthesize_locations", True)),
            ))
            source_parts.append("real_emitter_profile_cache")
        except Exception as exc:
            log(f"WARNING: could not load real_emitter profile cache coverage records: {exc}")
    if getattr(args, "coverage_use_result_support", True):
        try:
            source_records.extend(load_source_records_from_result_dirs(
                result_dirs,
                prefer_denoised=getattr(args, "prefer_denoised_low_level", True),
                max_records=int(getattr(args, "coverage_max_result_support_records", 200000)),
            ))
            source_parts.append("result_support")
        except Exception as exc:
            log(f"WARNING: could not extract coverage support from result directories: {exc}")

    source_label = "+".join(source_parts) if source_parts else "type_default_no_source_inventory"
    coverage_rows = compute_coverage_for_gt_items(
        gt_items,
        source_records,
        temporal_pad_hours=float(getattr(args, "coverage_temporal_pad_hours", 24.0)),
        allow_type_default_when_no_sources=bool(getattr(args, "coverage_allow_type_default", True)),
        source_label=source_label,
    )
    labelling_summary = {"enabled": False}
    if getattr(args, "coverage_labelling_json", None):
        coverage_rows, labelling_summary = apply_labelling_coverage_overrides(
            coverage_rows,
            paths=getattr(args, "coverage_labelling_json", None) or [],
            as_primary=bool(getattr(args, "coverage_labelling_as_primary", True)),
        )
    coverage_map = coverage_scores_by_gt_id(coverage_rows) if coverage_scores_by_gt_id is not None else {}
    summary = summarize_coverage_rows(coverage_rows) if summarize_coverage_rows is not None else {"num_gt": len(coverage_rows)}
    summary.update({
        "enabled": True,
        "num_source_records": len(source_records),
        "coverage_source": source_label,
        "coverage_source_jsonl": list(getattr(args, "coverage_source_jsonl", None) or []),
        "coverage_auto_source_inventory": bool(getattr(args, "coverage_auto_source_inventory", True)),
        "coverage_auto_source_jsonl": [str(p) for p in auto_source_jsonl],
        "coverage_auto_profile_cache": [str(p) for p in auto_profile_cache],
        "coverage_use_result_support": bool(getattr(args, "coverage_use_result_support", True)),
        "coverage_labelling_json": list(getattr(args, "coverage_labelling_json", None) or []),
        "coverage_labelling_summary": labelling_summary,
        "coverage_temporal_pad_hours": float(getattr(args, "coverage_temporal_pad_hours", 24.0)),
        "coverage_interpretation": (
            "Scores estimate incident observability. If coverage_source includes result_support, "
            "the source/evidence inventory came from emitted prediction support rather than a complete deployed sensor inventory."
        ),
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "coverage_scores.csv", coverage_rows)
    write_json(output_dir / "coverage_scores.json", coverage_rows)
    write_json(output_dir / "coverage_summary.json", summary)
    return coverage_rows, coverage_map, summary


def attach_coverage_to_real_metrics(
    *,
    metrics: Dict[str, Any],
    matches: List[Dict[str, Any]],
    gt_items: Sequence[GroundTruthIncident],
    coverage_map: Dict[str, float],
) -> None:
    if not coverage_map or compute_coverage_weighted_metrics is None:
        return
    for match in matches:
        gt_id = str(match.get("gt_id") or "")
        gt_location = str(match.get("gt_location") or "")
        match["gt_coverage_score"] = coverage_map.get(f"{gt_id}::{gt_location}", coverage_map.get(gt_id))
    coverage_metrics = compute_coverage_weighted_metrics(
        gt_items,
        matches,
        coverage_map,
        fp_count=int(metrics.get("fp") or 0),
    )
    metrics.update(coverage_metrics)




def coverage_key_for_gt_item(gt: GroundTruthIncident) -> str:
    """Return the coverage-key convention used by coverage_metrics.py."""
    incident_id = str(getattr(gt, "incident_id", "") or "")
    location = str(getattr(gt, "location_name", "") or "")
    if location and location != incident_id:
        return f"{incident_id}::{location}"
    return incident_id or location or "unknown_gt"


def unique_gt_items_for_result_dirs(
    result_dirs: Sequence[Tuple[str, str, str, Path]],
    gt_index: Dict[str, List[Path]],
    *,
    allowed_types: Sequence[str],
    default_single_step_minutes: float,
) -> List[GroundTruthIncident]:
    """Load a de-duplicated GT universe for synthetic coverage scoring."""
    out: Dict[str, GroundTruthIncident] = {}
    seen_runs = sorted({run_name for _method, _experiment, run_name, _result_dir in result_dirs})
    for run_name in seen_runs:
        for gt in load_ground_truth_for_run(
            run_name,
            gt_index,
            allowed_types=allowed_types,
            default_single_step_minutes=default_single_step_minutes,
        ):
            out.setdefault(coverage_key_for_gt_item(gt), gt)
    return list(out.values())


def synthetic_incident_performance_rows(
    *,
    method: str,
    experiment: str,
    run_name: str,
    gt_items: Sequence[GroundTruthIncident],
    matches: Sequence[MatchRecord],
) -> List[Dict[str, Any]]:
    """Create per-GT rows so coverage/performance diagnostics work for synth modes.

    A row is matched only when the one-to-one match is type-correct. Wrong-type
    matches are intentionally treated as misses for the GT incident, consistent
    with the evaluator's precision/recall accounting.
    """
    match_by_gt: Dict[str, MatchRecord] = {}
    for m in matches:
        if not m.type_correct:
            continue
        match_by_gt.setdefault(str(m.gt_id), m)

    rows: List[Dict[str, Any]] = []
    for gt in gt_items:
        key = coverage_key_for_gt_item(gt)
        m = match_by_gt.get(key) or match_by_gt.get(str(gt.incident_id))
        matched = m is not None
        rows.append({
            "method": method,
            "experiment": experiment,
            "run_name": run_name,
            "gt_id": str(gt.incident_id),
            "coverage_key": key,
            "gt_location": str(gt.location_name),
            "gt_final_name": getattr(gt, "final_name", None),
            "gt_type": gt.incident_type,
            "matched": bool(matched),
            "recall": 1.0 if matched else 0.0,
            "f1": 1.0 if matched else 0.0,
            "f2": 1.0 if matched else 0.0,
            "pred_id": m.pred_id if m else None,
            "pred_type": m.pred_type if m else None,
            "type_correct": m.type_correct if m else False,
            "source_iou": m.source_iou if m else None,
            "coverage_iou": m.coverage_iou if m else None,
            "best_iou": m.best_iou if m else None,
            "source_loc_error_km": m.source_loc_error_km if m else None,
            "coverage_loc_error_km": m.coverage_loc_error_km if m else None,
            "temporal_iou": m.temporal_iou if m else None,
            "detection_delay_minutes": m.detection_delay_minutes if m else None,
            "detected_before_external_report": m.detected_before_external_report if m else None,
        })
    return rows

def evaluate_low_real(args: argparse.Namespace) -> Dict[str, Any]:
    allowed_types = load_allowed_types(args.incident_types)

    progress_enabled = bool(args.progress_bar)
    if progress_enabled and tqdm is None:
        log("tqdm is not installed; using plain progress logs. Install with `pip install tqdm` for a progress bar.")
        progress_enabled = False

    gt_items = load_real_low_ground_truth(
        args.real_gt_path,
        allowed_types=allowed_types,
        default_duration_hours=args.real_default_duration_hours,
        geocode_locations=args.real_geocode_gt_locations,
        geocode_cache_path=args.real_geo_cache_path,
        geocode_cache_only=args.real_geo_cache_only,
        geocode_force_refresh=args.real_geo_force_refresh,
        progress_enabled=progress_enabled,
    )

    result_dirs = discover_real_low_result_dirs(
        Path(args.results_root),
        real_experiment_name=args.real_experiment_name,
        experiment_glob=args.experiment_glob,
        methods=args.methods or (),
    )
    log(f"Discovered {len(result_dirs)} real low-level date result directories under {args.results_root}.")
    log(f"Loaded {len(gt_items)} corrected real GT incidents from {args.real_gt_path}.")

    output_dir = Path(args.output_dir) / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_rows, coverage_map, coverage_summary = compute_and_write_coverage_scores(
        args=args,
        gt_items=gt_items,
        result_dirs=result_dirs,
        output_dir=output_dir,
        progress_enabled=progress_enabled,
    )
    if coverage_summary.get("enabled"):
        log(f"Computed coverage scores for {len(coverage_rows)} GT incidents using {coverage_summary.get('coverage_source')}.")

    first_owner_index = build_prediction_first_owner_index(result_dirs) if args.leakage_filter else {}
    if args.leakage_filter:
        num_keys = sum(len(v) for v in first_owner_index.values())
        log(f"Built real prediction leakage first-owner index with {num_keys} identity keys.")

    grouped_predictions: Dict[Tuple[str, str], List[PredictedIncident]] = {}
    date_summary_rows: List[Dict[str, Any]] = []
    leakage_filter_summaries: List[Dict[str, Any]] = []
    runtime_rows: List[Dict[str, Any]] = []

    for method, experiment, date_name, result_dir in progress_iter(
        result_dirs,
        total=len(result_dirs),
        desc="evaluate low_real dates",
        enabled=progress_enabled,
    ):
        if args.include_runtime:
            runtime_rows.append(runtime_summary_for_result_dir(method, experiment, date_name, result_dir))

        if not (result_dir / LOW_LEVEL_FILENAME).exists():
            continue

        pred_items, leakage_summary = load_predictions_for_result_dir(
            result_dir,
            run_name=date_name,
            allowed_types=allowed_types,
            min_coverage_normalized=args.min_coverage_normalized,
            method=method,
            experiment=experiment,
            first_owner_index=first_owner_index,
            leakage_filter_enabled=args.leakage_filter,
            leakage_filter_methods=args.leakage_filter_methods or (),
            write_denoised_low_level=args.write_denoised_low_level,
            denoised_methods=args.denoised_methods or (),
            prefer_denoised_low_level=args.prefer_denoised_low_level,
            include_rationale_type_guesses=args.real_rationale_type_guesses,
        )
        # For real outputs, a missing explicit detection timestamp should not
        # erase delay metrics.  Use the predicted active start as the system
        # report proxy when needed.
        for pred in pred_items:
            if pred.detection_time is None:
                pred.detection_time = pred.start_time or pred.end_time

        grouped_predictions.setdefault((method, experiment), []).extend(pred_items)
        summary = dict(leakage_summary)
        summary.update({
            "method": method,
            "experiment": experiment,
            "date": date_name,
            "result_dir": str(result_dir),
        })
        leakage_filter_summaries.append(summary)
        date_summary_rows.append({
            "method": method,
            "experiment": experiment,
            "date": date_name,
            "result_dir": str(result_dir),
            "num_predictions_raw": leakage_summary.get("raw_predictions"),
            "num_predictions_dropped_leaked": leakage_summary.get("dropped_predictions"),
            "num_predictions_after_filter": leakage_summary.get("kept_predictions"),
            "source_file_used": leakage_summary.get("source_file_used"),
        })

    run_rows: List[Dict[str, Any]] = []
    match_rows: List[Dict[str, Any]] = []
    incident_name_rows: List[Dict[str, Any]] = []
    confusions: List[Dict[str, Dict[str, int]]] = []
    dedupe_summaries: List[Dict[str, Any]] = []
    merge_summaries: List[Dict[str, Any]] = []

    use_spatial_matching = bool((not args.real_no_spatial_matching) and any(g.geometry is not None for g in gt_items))

    grouped_items = sorted(grouped_predictions.items())
    for (method, experiment), raw_preds in progress_iter(
        grouped_items,
        total=len(grouped_items),
        desc="match low_real methods",
        enabled=progress_enabled,
    ):
        method_start = time.perf_counter()
        progress_write(
            f"Matching {method}/{experiment}: raw_labels={len(raw_preds)} gt={len(gt_items)}",
            enabled=progress_enabled,
        )
        dedupe_start = time.perf_counter()
        pred_items, dedupe_summary = dedupe_real_predictions(raw_preds)
        dedupe_summary.update({"method": method, "experiment": experiment})
        dedupe_summaries.append(dedupe_summary)
        progress_write(
            f"  {method}/{experiment}: after exact dedupe={len(pred_items)} "
            f"({time.perf_counter() - dedupe_start:.1f}s)",
            enabled=progress_enabled,
        )

        merge_start = time.perf_counter()
        pred_items, merge_summary = merge_real_prediction_updates(
            pred_items,
            enabled=args.real_merge_prediction_updates,
            max_temporal_gap_hours=args.real_merge_temporal_gap_hours,
            max_spatial_distance_km=args.real_merge_spatial_distance_km,
            min_spatial_iou=args.real_merge_min_spatial_iou,
            progress_enabled=bool(progress_enabled and getattr(args, "real_match_inner_progress", True)),
            progress_desc=f"merge updates {method}",
        )
        merge_summary.update({"method": method, "experiment": experiment})
        merge_summaries.append(merge_summary)
        progress_write(
            f"  {method}/{experiment}: after update merge={len(pred_items)} "
            f"({time.perf_counter() - merge_start:.1f}s); candidate_pairs≈{len(pred_items) * len(gt_items)}",
            enabled=progress_enabled,
        )

        match_start = time.perf_counter()
        metrics, matches, confusion = evaluate_real_low_prediction_set(
            method=method,
            experiment=experiment,
            gt_items=gt_items,
            pred_items=pred_items,
            temporal_match_window_hours=args.real_temporal_match_window_hours,
            use_spatial_matching=use_spatial_matching,
            max_loc_error_km=args.max_loc_error_km,
            min_spatial_iou=args.min_spatial_iou,
            progress_enabled=bool(progress_enabled and getattr(args, "real_match_inner_progress", True)),
            progress_desc=f"candidate pairs {method}",
        )
        match_elapsed = time.perf_counter() - match_start
        progress_write(
            f"  {method}/{experiment}: matched={len(matches)} preds_after_merge={len(pred_items)} "
            f"eval_time={match_elapsed:.1f}s total_method_time={time.perf_counter() - method_start:.1f}s",
            enabled=progress_enabled,
        )
        attach_coverage_to_real_metrics(
            metrics=metrics,
            matches=matches,
            gt_items=gt_items,
            coverage_map=coverage_map,
        )
        metrics["variant"] = experiment_variant_name(experiment)
        metrics["scalability_setting"] = experiment_variant_name(experiment)
        metrics["observation_model_condition"] = observation_model_condition(experiment)
        metrics["num_date_result_dirs"] = sum(
            1 for m, e, _d, _p in result_dirs if m == method and e == experiment
        )
        metrics["num_predictions_raw_across_dates"] = sum(
            int(r.get("num_predictions_raw") or 0)
            for r in date_summary_rows
            if r.get("method") == method and r.get("experiment") == experiment
        )
        metrics["num_predictions_after_leakage_filter_across_dates"] = sum(
            int(r.get("num_predictions_after_filter") or 0)
            for r in date_summary_rows
            if r.get("method") == method and r.get("experiment") == experiment
        )
        metrics["num_predictions_dropped_leaked_across_dates"] = sum(
            int(r.get("num_predictions_dropped_leaked") or 0)
            for r in date_summary_rows
            if r.get("method") == method and r.get("experiment") == experiment
        )
        metrics["num_predictions_dropped_real_dedupe"] = dedupe_summary.get("dedupe_real_dropped_predictions")
        metrics["num_predictions_after_real_update_merge"] = merge_summary.get("merge_real_kept_predictions")
        metrics["num_predictions_absorbed_real_update_merge"] = merge_summary.get("merge_real_dropped_or_absorbed_predictions")
        incident_name_rows.extend(real_low_metrics_by_gt_incident_name(
            method=method,
            experiment=experiment,
            gt_items=gt_items,
            pred_items=pred_items,
            matches=matches,
            temporal_match_window_hours=args.real_temporal_match_window_hours,
            use_spatial_matching=use_spatial_matching,
            max_loc_error_km=args.max_loc_error_km,
            min_spatial_iou=args.min_spatial_iou,
        ))
        run_rows.append(metrics)
        match_rows.extend(matches)
        confusions.append(confusion)

    coverage_per_incident_rows: List[Dict[str, Any]] = []
    coverage_correlation_rows: List[Dict[str, Any]] = []
    coverage_bin_rows: List[Dict[str, Any]] = []
    if coverage_rows:
        attach_coverage_to_incident_rows(incident_name_rows, coverage_rows)
        coverage_per_incident_rows, coverage_correlation_rows, coverage_bin_rows = compute_coverage_performance_diagnostics(incident_name_rows)

    progress_write("Aggregating low_real metrics...", enabled=progress_enabled)
    by_method_exp = aggregate_by_method_experiment(run_rows)
    by_type = aggregate_by_incident_type(run_rows)
    by_gt_type = aggregate_by_gt_incident_type(run_rows)
    overall = aggregate_metrics(run_rows)
    confusion_rows = merge_confusions(confusions)

    gt_geocode_rows = []
    for gt in gt_items:
        summary = getattr(gt, "geocode_summary", {}) if isinstance(getattr(gt, "geocode_summary", {}), dict) else {}
        gt_geocode_rows.append({
            "gt_id": gt.incident_id,
            "gt_final_name": getattr(gt, "final_name", None),
            "gt_type": gt.incident_type,
            "gt_location": gt.location_name,
            "has_geometry": gt.geometry is not None,
            "representative_latitude": gt.representative_latitude,
            "representative_longitude": gt.representative_longitude,
            "geocode_status": summary.get("status"),
            "geocode_query": summary.get("query"),
            "geocode_source": summary.get("source"),
            "geocode_method": summary.get("method"),
            "geocode_cache_hit": summary.get("cache_hit"),
        })

    progress_write("Writing low_real output files...", enabled=progress_enabled)
    write_csv(output_dir / "run_metrics.csv", run_rows)
    write_json(output_dir / "run_metrics.json", run_rows)
    write_csv(output_dir / "real_gt_geocode_summary.csv", gt_geocode_rows)
    write_json(output_dir / "real_gt_geocode_summary.json", gt_geocode_rows)
    write_csv(output_dir / "match_records.csv", match_rows)
    write_json(output_dir / "match_records.json", match_rows)
    write_csv(output_dir / "aggregate_metrics_by_method_experiment.csv", by_method_exp)
    write_json(output_dir / "aggregate_metrics_by_method_experiment.json", by_method_exp)
    write_csv(output_dir / "aggregate_metrics_by_incident_type.csv", by_type)
    write_csv(output_dir / "aggregate_metrics_by_predicted_incident_type.csv", by_type)
    write_csv(output_dir / "aggregate_metrics_by_gt_incident_type.csv", by_gt_type)
    write_json(output_dir / "aggregate_metrics_by_gt_incident_type.json", by_gt_type)
    write_csv(output_dir / "aggregate_metrics_by_gt_incident_name.csv", incident_name_rows)
    write_json(output_dir / "aggregate_metrics_by_gt_incident_name.json", incident_name_rows)
    if coverage_per_incident_rows:
        write_csv(output_dir / "coverage_per_incident_performance.csv", coverage_per_incident_rows)
        write_json(output_dir / "coverage_per_incident_performance.json", coverage_per_incident_rows)
        write_csv(output_dir / "coverage_performance_correlations.csv", coverage_correlation_rows)
        write_json(output_dir / "coverage_performance_correlations.json", coverage_correlation_rows)
        write_csv(output_dir / "coverage_performance_by_coverage_bin.csv", coverage_bin_rows)
        write_json(output_dir / "coverage_performance_by_coverage_bin.json", coverage_bin_rows)
    write_csv(output_dir / "type_confusion_matrix.csv", confusion_rows)
    write_json(output_dir / "prediction_leakage_filter_summary.json", leakage_filter_summaries)
    write_csv(output_dir / "prediction_leakage_filter_summary.csv", leakage_filter_summaries)
    write_json(output_dir / "real_prediction_dedupe_summary.json", dedupe_summaries)
    write_csv(output_dir / "real_prediction_dedupe_summary.csv", dedupe_summaries)
    write_json(output_dir / "real_prediction_update_merge_summary.json", merge_summaries)
    write_csv(output_dir / "real_prediction_update_merge_summary.csv", merge_summaries)
    write_csv(output_dir / "real_prediction_date_summary.csv", date_summary_rows)
    write_json(output_dir / "real_prediction_date_summary.json", date_summary_rows)
    write_csv(output_dir / "real_prediction_date_summary_by_date.csv", aggregate_real_prediction_date_summary(date_summary_rows))

    if runtime_rows:
        runtime_agg = aggregate_runtime(runtime_rows)
        write_csv(output_dir / "real_runtime_runs.csv", runtime_rows)
        write_json(output_dir / "real_runtime_runs.json", runtime_rows)
        write_csv(output_dir / "real_runtime_summary.csv", runtime_agg)
        write_json(output_dir / "real_runtime_summary.json", {
            "runs": runtime_rows,
            "by_method_experiment": runtime_agg,
        })

    aggregate_payload = {
        "mode": args.mode,
        "results_root": str(args.results_root),
        "real_gt_path": str(args.real_gt_path),
        "real_experiment_name": args.real_experiment_name,
        "num_result_date_dirs_discovered": len(result_dirs),
        "num_method_experiment_sets_evaluated": len(run_rows),
        "num_gt": len(gt_items),
        "num_gt_with_geometry": sum(1 for g in gt_items if g.geometry is not None),
        "overall": overall,
        "coverage_summary": coverage_summary,
        "coverage_performance_correlations": coverage_correlation_rows,
        "coverage_performance_by_coverage_bin": coverage_bin_rows,
        "coverage_performance_note": (
            "coverage_performance_correlations.csv reports Pearson/Spearman correlations between GT coverage_score "
            "and per-incident detection/quality metrics. coverage_performance_by_coverage_bin.csv reports recall/quality "
            "within low/medium/high coverage bands."
        ),
        "by_method_experiment": by_method_exp,
        "by_incident_type": by_type,
        "by_predicted_incident_type": by_type,
        "by_gt_incident_type": by_gt_type,
        "by_gt_incident_name": incident_name_rows,
        "delay_metric_definitions": {
            "system_report_time": "time_first_incident_predicted when present; otherwise predicted active_interval.start/end fallback.",
            "external_report_delay_minutes": "system_report_time - earliest_article_datetime_pacific; negative means the system detected before publication.",
            "pre_report_recall": "matched GT labels detected before external publication / all GT labels.",
            "matched_pre_report_fraction": "matched detections before external publication / all matched detections.",
            "median_external_report_delay_minutes": "median over matched detections; negative is earlier than the article.",
            "p25_p75_external_report_delay_minutes": "25th/75th percentiles over matched detections; negative is earlier than the article.",
        },
        "matching_notes": {
            "dataset_level_matching": "Predictions are loaded from all date folders for a method/experiment before one-to-one matching, because GT incidents can span multiple days.",
            "spatial_matching": "Enabled by default for low_real when GT geometries are available. Textual GT locations are geocoded by default through GeoManager/cache and used as a spatial gate. Use --real-no-spatial-matching to run the looser type+time-only evaluation.",
            "temporal_matching": "A prediction can match when its active interval overlaps the GT interval, or its effective system report time falls within the GT interval expanded by --real-temporal-match-window-hours.",
            "duplicate_handling": "The existing prediction identity first-owner leakage filter is applied across date folders, followed by exact/signature dedupe and optional same-event update merging.",
        },
        "settings": {
            "methods": args.methods if args.methods else list(REAL_LOW_DEFAULT_METHODS),
            "excluded_methods": sorted(REAL_LOW_EXCLUDED_METHODS),
            "real_temporal_match_window_hours": args.real_temporal_match_window_hours,
            "real_default_duration_hours": args.real_default_duration_hours,
            "real_no_spatial_matching": args.real_no_spatial_matching,
            "real_spatial_matching_used": use_spatial_matching,
            "real_geocode_gt_locations": args.real_geocode_gt_locations,
            "real_geo_cache_path": str(args.real_geo_cache_path) if args.real_geo_cache_path else None,
            "real_geo_cache_only": args.real_geo_cache_only,
            "real_merge_prediction_updates": args.real_merge_prediction_updates,
            "real_merge_temporal_gap_hours": args.real_merge_temporal_gap_hours,
            "real_merge_spatial_distance_km": args.real_merge_spatial_distance_km,
            "real_merge_min_spatial_iou": args.real_merge_min_spatial_iou,
            "prefer_denoised_low_level": args.prefer_denoised_low_level,
            "real_rationale_type_guesses": args.real_rationale_type_guesses,
            "min_coverage_normalized": args.min_coverage_normalized,
            "coverage_labelling_json": args.coverage_labelling_json,
            "coverage_labelling_as_primary": args.coverage_labelling_as_primary,
            "real_match_inner_progress": args.real_match_inner_progress,
            "leakage_filter": args.leakage_filter,
            "leakage_filter_methods": args.leakage_filter_methods,
            "write_denoised_low_level": args.write_denoised_low_level,
            "denoised_methods": args.denoised_methods,
        },
    }
    write_json(output_dir / "aggregate_metrics.json", aggregate_payload)
    log(f"Wrote low_real evaluation outputs to {output_dir}.")
    return aggregate_payload


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------



def evaluate_throughput(args: argparse.Namespace) -> Dict[str, Any]:
    """Dedicated timing-only mode for actual_timing duplication experiments."""
    result_dirs = discover_result_dirs(
        Path(args.results_root),
        mode="throughput",
        low_experiment_name=args.low_experiment_name,
        experiment_glob=args.experiment_glob,
        methods=args.methods or (),
    )
    log(f"Discovered {len(result_dirs)} actual_timing result run directories under {args.results_root}.")

    runtime_rows: List[Dict[str, Any]] = []
    progress_enabled = bool(args.progress_bar)
    if progress_enabled and tqdm is None:
        log("tqdm is not installed; using plain progress logs. Install with `pip install tqdm` for a progress bar.")
        progress_enabled = False

    for method, experiment, run_name, result_dir in progress_iter(
        result_dirs,
        total=len(result_dirs),
        desc="evaluate throughput",
        enabled=progress_enabled,
    ):
        row = runtime_summary_for_result_dir(method, experiment, run_name, result_dir)
        # Make the actual_timing fields explicit even if timing.json is older.
        row["duplication_level"] = actual_timing_duplication_level(experiment)
        row["actual_timing_setting"] = actual_timing_setting_name(experiment)
        row["observation_model_condition"] = "with_observation_model"
        runtime_rows.append(row)

    output_dir = Path(args.output_dir) / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    by_duplication = aggregate_throughput_by_duplication(runtime_rows)
    by_method_experiment = aggregate_runtime(runtime_rows)

    write_csv(output_dir / "throughput_runs.csv", runtime_rows)
    write_json(output_dir / "throughput_runs.json", runtime_rows)
    write_csv(output_dir / "throughput_by_duplication.csv", by_duplication)
    write_json(output_dir / "throughput_by_duplication.json", by_duplication)
    write_csv(output_dir / "throughput_by_method_experiment.csv", by_method_experiment)
    write_json(output_dir / "throughput_by_method_experiment.json", by_method_experiment)

    payload = {
        "mode": args.mode,
        "results_root": str(args.results_root),
        "num_result_run_dirs_discovered": len(result_dirs),
        "num_runtime_runs": len(runtime_rows),
        "by_duplication": by_duplication,
        "by_method_experiment": by_method_experiment,
        "metric_definitions": {
            "aggregate_reports_per_second": "sum(num_observations) / sum(incident_wall_clock_seconds) across runs at the duplication level.",
            "aggregate_reports_per_minute": "aggregate_reports_per_second * 60.",
            "aggregate_seconds_per_100_reports": "100 / aggregate_reports_per_second.",
            "incident_wall_clock_seconds": "Whole-incident elapsed time from incident_start to incident_end when present in timing.json; otherwise falls back to end_to_end_total_seconds for aggregation.",
        },
        "settings": {
            "experiment_glob": args.experiment_glob,
            "methods": args.methods,
        },
    }
    write_json(output_dir / "aggregate_metrics.json", payload)
    log(f"Wrote throughput outputs to {output_dir}.")
    return payload


def coverage_smoke_condition_from_experiment(experiment: str, prefix: str = "coverage_smoke") -> str:
    prefix = str(prefix or "coverage_smoke")
    if experiment.startswith(prefix + "_"):
        return experiment[len(prefix) + 1:]
    return experiment


def coverage_smoke_source_paths_for_experiment(
    *,
    experiment: str,
    result_dirs: Sequence[Tuple[str, str, str, Path]],
    gt_index: Dict[str, List[Path]],
    filename_template: str,
    prefix: str,
) -> List[Path]:
    """Find condition-specific normalized REPORT JSONL files in simulator folders."""
    out: List[Path] = []
    seen = set()
    condition = coverage_smoke_condition_from_experiment(experiment, prefix)
    for method, exp, run_name, _result_dir in result_dirs:
        if exp != experiment:
            continue
        for folder in gt_index.get(run_name, []):
            filename = filename_template.format(
                experiment=experiment,
                case=condition,
                condition=condition,
                prefix=prefix,
                method=method,
                run_name=run_name,
            )
            path = folder / filename
            key = str(path)
            if key not in seen and path.exists() and path.is_file():
                seen.add(key)
                out.append(path)
    return sorted(out, key=lambda p: str(p))


def attach_coverage_to_incident_rows_by_experiment(
    rows: List[Dict[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach coverage without mixing repeated GT keys across smoke-test conditions."""
    cov_by_key: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for cov in coverage_rows:
        experiment = str(cov.get("experiment") or "")
        keys = [cov.get("coverage_key"), cov.get("gt_id"), _coverage_key_from_row(cov)]
        for key in keys:
            if key:
                cov_by_key[(experiment, str(key))] = cov
    for row in rows:
        experiment = str(row.get("experiment") or "")
        key = _coverage_key_from_row(row)
        cov = cov_by_key.get((experiment, key)) or cov_by_key.get((experiment, str(row.get("gt_id") or "")))
        if cov is None:
            row["coverage_score"] = None
            continue
        for col in (
            "coverage_score", "modality_coverage", "source_coverage", "inference_coverage",
            "coverage_source", "available_modalities", "evidence_types", "num_relevant_sources",
            "coverage_condition", "num_source_records_for_experiment", "coverage_formula",
        ):
            if col in cov:
                row[col] = cov.get(col)
    return rows


def evaluate_coverage_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    """Evaluate the synthetic coverage/correlation smoke-test suite.

    Unlike ordinary synthetic modes, each experiment condition has its own
    source inventory JSONL.  The same GT incidents are replayed under no/low/
    medium/high report availability, so coverage must be attached by
    (experiment, GT id) rather than by GT id alone.
    """
    if compute_coverage_for_gt_items is None or load_source_records_from_jsonl is None:
        raise RuntimeError("coverage_smoke requires coverage_metrics.py to be importable")

    allowed_types = load_allowed_types(args.incident_types)
    gt_index = build_gt_index(args.gt_roots)
    results_root = Path(args.results_root)
    prefix = str(getattr(args, "coverage_smoke_results_prefix", "coverage_smoke") or "coverage_smoke")

    result_dirs: List[Tuple[str, str, str, Path]] = []
    method_filter = {m for m in (args.methods or ()) if m}
    if results_root.exists():
        for method_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
            method = method_dir.name
            if method_filter and method not in method_filter:
                continue
            for exp_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
                experiment = exp_dir.name
                if not experiment.startswith(prefix + "_"):
                    continue
                if args.experiment_glob and not (exp_dir.match(args.experiment_glob) or args.experiment_glob in experiment):
                    continue
                for run_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
                    if (run_dir / LOW_LEVEL_FILENAME).exists():
                        result_dirs.append((method, experiment, run_dir.name, run_dir))
    log(f"Discovered {len(result_dirs)} coverage-smoke result run directories under {args.results_root}.")
    log(f"Indexed {sum(len(v) for v in gt_index.values())} ground-truth folders from {args.gt_roots}.")

    progress_enabled = bool(args.progress_bar)
    if progress_enabled and tqdm is None:
        log("tqdm is not installed; using plain progress logs. Install with `pip install tqdm` for a progress bar.")
        progress_enabled = False

    output_dir = Path(args.output_dir) / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = sorted({experiment for _method, experiment, _run, _path in result_dirs})
    coverage_rows: List[Dict[str, Any]] = []
    coverage_summaries: List[Dict[str, Any]] = []
    coverage_maps_by_experiment: Dict[str, Dict[str, float]] = {}
    source_paths_by_experiment: Dict[str, List[str]] = {}

    for experiment in progress_iter(
        experiments,
        total=len(experiments),
        desc="coverage smoke sources",
        enabled=progress_enabled,
        unit="exp",
    ):
        exp_dirs = [item for item in result_dirs if item[1] == experiment]
        gt_items = unique_gt_items_for_result_dirs(
            exp_dirs,
            gt_index,
            allowed_types=allowed_types,
            default_single_step_minutes=args.default_single_step_minutes,
        )
        source_paths = coverage_smoke_source_paths_for_experiment(
            experiment=experiment,
            result_dirs=exp_dirs,
            gt_index=gt_index,
            filename_template=args.coverage_smoke_reports_filename_template,
            prefix=prefix,
        )
        source_records = load_source_records_from_jsonl(source_paths) if source_paths else []
        condition = coverage_smoke_condition_from_experiment(experiment, prefix)
        rows = compute_coverage_for_gt_items(
            gt_items,
            source_records,
            temporal_pad_hours=float(args.coverage_temporal_pad_hours),
            allow_type_default_when_no_sources=bool(args.coverage_smoke_allow_type_default),
            source_label=f"coverage_smoke:{experiment}",
        )
        for row in rows:
            row["experiment"] = experiment
            row["coverage_condition"] = condition
            row["num_source_records_for_experiment"] = len(source_records)
        coverage_rows.extend(rows)
        coverage_maps_by_experiment[experiment] = coverage_scores_by_gt_id(rows) if coverage_scores_by_gt_id is not None else {}
        source_paths_by_experiment[experiment] = [str(p) for p in source_paths]
        summary = summarize_coverage_rows(rows) if summarize_coverage_rows is not None else {"num_gt": len(rows)}
        summary.update({
            "experiment": experiment,
            "coverage_condition": condition,
            "num_source_jsonl_files": len(source_paths),
            "num_source_records": len(source_records),
            "source_jsonl_files": [str(p) for p in source_paths],
            "coverage_allow_type_default": bool(args.coverage_smoke_allow_type_default),
        })
        coverage_summaries.append(summary)

    write_csv(output_dir / "coverage_scores.csv", coverage_rows)
    write_json(output_dir / "coverage_scores.json", coverage_rows)
    write_csv(output_dir / "coverage_summary_by_experiment.csv", coverage_summaries)
    write_json(output_dir / "coverage_summary_by_experiment.json", coverage_summaries)

    first_owner_index = build_prediction_first_owner_index(result_dirs) if args.leakage_filter else {}
    if args.leakage_filter:
        num_keys = sum(len(v) for v in first_owner_index.values())
        log(f"Built prediction leakage first-owner index with {num_keys} identity keys.")

    run_rows: List[Dict[str, Any]] = []
    match_rows: List[Dict[str, Any]] = []
    incident_rows: List[Dict[str, Any]] = []
    confusions: List[Dict[str, Dict[str, int]]] = []
    missing_gt: List[Dict[str, Any]] = []
    leakage_filter_summaries: List[Dict[str, Any]] = []
    gt_summary_rows: List[Dict[str, Any]] = []

    for method, experiment, run_name, result_dir in progress_iter(
        result_dirs,
        total=len(result_dirs),
        desc="evaluate coverage_smoke",
        enabled=progress_enabled,
        unit="run",
    ):
        condition = coverage_smoke_condition_from_experiment(experiment, prefix)
        gt_items = load_ground_truth_for_run(
            run_name,
            gt_index,
            allowed_types=allowed_types,
            default_single_step_minutes=args.default_single_step_minutes,
        )
        if not gt_items:
            missing_gt.append({"method": method, "experiment": experiment, "run_name": run_name, "result_dir": str(result_dir)})
            if args.skip_missing_gt:
                continue

        for gt in gt_items:
            gt_summary_rows.append({
                "method": method,
                "experiment": experiment,
                "coverage_condition": condition,
                "run_name": run_name,
                "gt_id": gt.incident_id,
                "coverage_key": coverage_key_for_gt_item(gt),
                "gt_type": gt.incident_type,
                "gt_location": gt.location_name,
                "has_geometry": gt.geometry is not None,
                "representative_latitude": gt.representative_latitude,
                "representative_longitude": gt.representative_longitude,
            })

        pred_items, leakage_summary = load_predictions_for_result_dir(
            result_dir,
            run_name=run_name,
            allowed_types=allowed_types,
            min_coverage_normalized=args.min_coverage_normalized,
            method=method,
            experiment=experiment,
            first_owner_index=first_owner_index,
            leakage_filter_enabled=args.leakage_filter,
            leakage_filter_methods=args.leakage_filter_methods or (),
            write_denoised_low_level=args.write_denoised_low_level,
            denoised_methods=args.denoised_methods or (),
        )
        leakage_summary.update({
            "method": method,
            "experiment": experiment,
            "coverage_condition": condition,
            "run_name": run_name,
            "result_dir": str(result_dir),
        })
        leakage_filter_summaries.append(leakage_summary)

        metrics, matches, confusion = evaluate_run(
            method=method,
            experiment=experiment,
            run_name=run_name,
            result_dir=result_dir,
            gt_items=gt_items,
            pred_items=pred_items,
            max_loc_error_km=args.max_loc_error_km,
            min_spatial_iou=args.min_spatial_iou,
        )
        match_dicts = [asdict(m) for m in matches]
        cov_map = coverage_maps_by_experiment.get(experiment, {})
        if cov_map:
            attach_coverage_to_real_metrics(
                metrics=metrics,
                matches=match_dicts,
                gt_items=gt_items,
                coverage_map=cov_map,
            )
        metrics["coverage_condition"] = condition
        metrics["variant"] = condition
        metrics["scalability_setting"] = condition
        metrics["observation_model_condition"] = observation_model_condition(experiment)
        metrics["num_predictions_raw"] = leakage_summary.get("raw_predictions")
        metrics["num_predictions_dropped_leaked"] = leakage_summary.get("dropped_predictions")
        metrics["num_predictions_after_leakage_filter"] = leakage_summary.get("kept_predictions")
        run_rows.append(metrics)
        match_rows.extend(match_dicts)
        incident_rows.extend(synthetic_incident_performance_rows(
            method=method,
            experiment=experiment,
            run_name=run_name,
            gt_items=gt_items,
            matches=matches,
        ))
        confusions.append(confusion)

    attach_coverage_to_incident_rows_by_experiment(incident_rows, coverage_rows)
    coverage_per_incident_rows, coverage_correlation_rows, coverage_bin_rows = compute_coverage_performance_diagnostics(incident_rows)

    by_method_exp = aggregate_by_method_experiment(run_rows)
    for row in by_method_exp:
        row["coverage_condition"] = coverage_smoke_condition_from_experiment(str(row.get("experiment") or ""), prefix)
    by_type = aggregate_by_incident_type(run_rows)
    by_gt_type = aggregate_by_gt_incident_type(run_rows)
    overall = aggregate_metrics(run_rows)

    smoke_check_rows: List[Dict[str, Any]] = []
    for row in by_method_exp:
        condition = str(row.get("coverage_condition") or "")
        is_none = condition == "none"
        tp = int(row.get("tp") or 0)
        recall_value = safe_float(row.get("recall"), None)
        f2_value = safe_float(row.get("f2"), None)
        mean_cov = safe_float(row.get("coverage_mean_gt"), None)
        smoke_check_rows.append({
            "method": row.get("method"),
            "experiment": row.get("experiment"),
            "coverage_condition": condition,
            "num_gt": row.get("num_gt"),
            "num_predictions": row.get("num_predictions"),
            "mean_coverage_score": mean_cov,
            "tp": tp,
            "fp": row.get("fp"),
            "fn": row.get("fn"),
            "recall": recall_value,
            "f2": f2_value,
            "none_condition_zero_coverage": (mean_cov == 0.0) if is_none else None,
            "none_condition_zero_recall": (recall_value == 0.0 and f2_value == 0.0 and tp == 0) if is_none else None,
        })

    aggregate_payload = {
        "mode": args.mode,
        "results_root": str(args.results_root),
        "gt_roots": [str(x) for x in args.gt_roots],
        "coverage_smoke_results_prefix": prefix,
        "coverage_smoke_reports_filename_template": args.coverage_smoke_reports_filename_template,
        "num_result_run_dirs_discovered": len(result_dirs),
        "num_runs_evaluated": len(run_rows),
        "num_missing_gt": len(missing_gt),
        "overall": overall,
        "coverage_summary_by_experiment": coverage_summaries,
        "coverage_performance_correlations": coverage_correlation_rows,
        "coverage_performance_by_coverage_bin": coverage_bin_rows,
        "smoke_checks": smoke_check_rows,
        "source_jsonl_by_experiment": source_paths_by_experiment,
        "by_method_experiment": by_method_exp,
        "by_incident_type": by_type,
        "by_predicted_incident_type": by_type,
        "by_gt_incident_type": by_gt_type,
        "coverage_smoke_note": (
            "The no-coverage condition intentionally disables type-default coverage priors by default, "
            "so an empty REPORT stream scores C=0 instead of a type-level prior. Use this mode as a sanity "
            "check, not as a main benchmark."
        ),
        "settings": {
            "max_loc_error_km": args.max_loc_error_km,
            "min_spatial_iou": args.min_spatial_iou,
            "min_coverage_normalized": args.min_coverage_normalized,
            "default_single_step_minutes": args.default_single_step_minutes,
            "leakage_filter": args.leakage_filter,
            "coverage_temporal_pad_hours": args.coverage_temporal_pad_hours,
            "coverage_smoke_allow_type_default": args.coverage_smoke_allow_type_default,
        },
    }

    write_json(output_dir / "aggregate_metrics.json", aggregate_payload)
    write_csv(output_dir / "run_metrics.csv", run_rows)
    write_json(output_dir / "run_metrics.json", run_rows)
    write_csv(output_dir / "match_records.csv", match_rows)
    write_json(output_dir / "match_records.json", match_rows)
    write_csv(output_dir / "gt_summary.csv", gt_summary_rows)
    write_json(output_dir / "gt_summary.json", gt_summary_rows)
    write_csv(output_dir / "aggregate_metrics_by_method_experiment.csv", by_method_exp)
    write_csv(output_dir / "aggregate_metrics_by_incident_type.csv", by_type)
    write_csv(output_dir / "aggregate_metrics_by_predicted_incident_type.csv", by_type)
    write_csv(output_dir / "aggregate_metrics_by_gt_incident_type.csv", by_gt_type)
    write_json(output_dir / "aggregate_metrics_by_gt_incident_type.json", by_gt_type)
    write_csv(output_dir / "per_gt_incident_performance.csv", incident_rows)
    write_json(output_dir / "per_gt_incident_performance.json", incident_rows)
    write_csv(output_dir / "coverage_per_incident_performance.csv", coverage_per_incident_rows)
    write_json(output_dir / "coverage_per_incident_performance.json", coverage_per_incident_rows)
    write_csv(output_dir / "coverage_performance_correlations.csv", coverage_correlation_rows)
    write_json(output_dir / "coverage_performance_correlations.json", coverage_correlation_rows)
    write_csv(output_dir / "coverage_performance_by_coverage_bin.csv", coverage_bin_rows)
    write_json(output_dir / "coverage_performance_by_coverage_bin.json", coverage_bin_rows)
    write_csv(output_dir / "coverage_smoke_checks.csv", smoke_check_rows)
    write_json(output_dir / "coverage_smoke_checks.json", smoke_check_rows)
    confusion_rows = merge_confusions(confusions)
    write_csv(output_dir / "type_confusion_matrix.csv", confusion_rows)
    write_json(output_dir / "missing_ground_truth.json", missing_gt)
    write_json(output_dir / "prediction_leakage_filter_summary.json", leakage_filter_summaries)
    write_csv(output_dir / "prediction_leakage_filter_summary.csv", leakage_filter_summaries)
    log(f"Wrote coverage-smoke evaluation outputs to {output_dir}.")
    return aggregate_payload


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    if args.mode == "throughput":
        return evaluate_throughput(args)
    if args.mode == "synth_composition":
        return evaluate_synth_composition(args)
    if args.mode == "low_real":
        return evaluate_low_real(args)
    if args.mode == "coverage_smoke":
        return evaluate_coverage_smoke(args)

    allowed_types = load_allowed_types(args.incident_types)
    gt_index = build_gt_index(args.gt_roots)

    result_dirs = discover_result_dirs(
        Path(args.results_root),
        mode=args.mode,
        low_experiment_name=args.low_experiment_name,
        experiment_glob=args.experiment_glob,
        methods=args.methods or (),
    )
    log(f"Discovered {len(result_dirs)} result run directories under {args.results_root}.")
    log(f"Indexed {sum(len(v) for v in gt_index.values())} ground-truth folders from {args.gt_roots}.")

    progress_enabled = bool(args.progress_bar)
    if progress_enabled and tqdm is None:
        log("tqdm is not installed; using plain progress logs. Install with `pip install tqdm` for a progress bar.")
        progress_enabled = False

    output_dir = Path(args.output_dir) / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic coverage-aware metrics use the unique GT universe across all
    # discovered runs.  This lets the same coverage_scores.csv machinery used by
    # low_real also drive coverage-weighted recall/F-scores and coverage bins for
    # low_incident, ablation, scalability, and ablation_and_scalability modes.
    coverage_gt_items = unique_gt_items_for_result_dirs(
        result_dirs,
        gt_index,
        allowed_types=allowed_types,
        default_single_step_minutes=args.default_single_step_minutes,
    )
    coverage_rows, coverage_map, coverage_summary = compute_and_write_coverage_scores(
        args=args,
        gt_items=coverage_gt_items,
        result_dirs=result_dirs,
        output_dir=output_dir,
        progress_enabled=progress_enabled,
    )
    if coverage_summary.get("enabled"):
        log(
            f"Computed coverage scores for {len(coverage_rows)} synthetic GT records using "
            f"{coverage_summary.get('coverage_source')}."
        )

    first_owner_index = build_prediction_first_owner_index(result_dirs) if args.leakage_filter else {}
    if args.leakage_filter:
        num_keys = sum(len(v) for v in first_owner_index.values())
        log(f"Built prediction leakage first-owner index with {num_keys} identity keys.")

    run_rows: List[Dict[str, Any]] = []
    match_rows: List[Dict[str, Any]] = []
    incident_rows: List[Dict[str, Any]] = []
    runtime_rows: List[Dict[str, Any]] = []
    confusions: List[Dict[str, Dict[str, int]]] = []
    missing_gt: List[Dict[str, Any]] = []
    leakage_filter_summaries: List[Dict[str, Any]] = []
    gt_summary_rows: List[Dict[str, Any]] = []

    progress_label = f"evaluate {args.mode}"
    for method, experiment, run_name, result_dir in progress_iter(
        result_dirs,
        total=len(result_dirs),
        desc=progress_label,
        enabled=progress_enabled,
    ):
        if args.mode in {"scalability", "ablation_and_scalability"} or args.include_runtime:
            runtime_rows.append(runtime_summary_for_result_dir(method, experiment, run_name, result_dir))

        # Dedicated timing scalability folders are timing-only; keep their
        # runtime rows but do not include them in prediction-performance
        # metrics, even if they happen to contain low_level_results.json.
        if args.mode == "scalability" and is_timing_scalability_experiment_name(experiment):
            continue

        # Scalability performance folders should contain predictions.  If a
        # directory only has timing.json, keep the runtime row and skip the
        # performance evaluation rather than creating an all-FN row.
        if args.mode == "scalability" and not (result_dir / LOW_LEVEL_FILENAME).exists():
            continue

        gt_items = load_ground_truth_for_run(
            run_name,
            gt_index,
            allowed_types=allowed_types,
            default_single_step_minutes=args.default_single_step_minutes,
        )
        if not gt_items:
            missing_gt.append({"method": method, "experiment": experiment, "run_name": run_name, "result_dir": str(result_dir)})
            if args.skip_missing_gt:
                continue

        for gt in gt_items:
            gt_summary_rows.append({
                "method": method,
                "experiment": experiment,
                "run_name": run_name,
                "gt_id": gt.incident_id,
                "coverage_key": coverage_key_for_gt_item(gt),
                "gt_type": gt.incident_type,
                "gt_location": gt.location_name,
                "has_geometry": gt.geometry is not None,
                "representative_latitude": gt.representative_latitude,
                "representative_longitude": gt.representative_longitude,
            })

        pred_items, leakage_summary = load_predictions_for_result_dir(
            result_dir,
            run_name=run_name,
            allowed_types=allowed_types,
            min_coverage_normalized=args.min_coverage_normalized,
            method=method,
            experiment=experiment,
            first_owner_index=first_owner_index,
            leakage_filter_enabled=args.leakage_filter,
            leakage_filter_methods=args.leakage_filter_methods or (),
            write_denoised_low_level=args.write_denoised_low_level,
            denoised_methods=args.denoised_methods or (),
        )
        leakage_summary.update({
            "method": method,
            "experiment": experiment,
            "run_name": run_name,
            "result_dir": str(result_dir),
        })
        leakage_filter_summaries.append(leakage_summary)

        metrics, matches, confusion = evaluate_run(
            method=method,
            experiment=experiment,
            run_name=run_name,
            result_dir=result_dir,
            gt_items=gt_items,
            pred_items=pred_items,
            max_loc_error_km=args.max_loc_error_km,
            min_spatial_iou=args.min_spatial_iou,
        )
        match_dicts = [asdict(m) for m in matches]
        if coverage_map:
            attach_coverage_to_real_metrics(
                metrics=metrics,
                matches=match_dicts,
                gt_items=gt_items,
                coverage_map=coverage_map,
            )
        metrics["variant"] = experiment_variant_name(experiment)
        metrics["scalability_setting"] = scalability_setting_name(experiment) if is_scalability_experiment_name(experiment) else experiment_variant_name(experiment)
        metrics["observation_model_condition"] = observation_model_condition(experiment)
        metrics["num_predictions_raw"] = leakage_summary.get("raw_predictions")
        metrics["num_predictions_dropped_leaked"] = leakage_summary.get("dropped_predictions")
        metrics["num_predictions_after_leakage_filter"] = leakage_summary.get("kept_predictions")
        run_rows.append(metrics)
        match_rows.extend(match_dicts)
        incident_rows.extend(synthetic_incident_performance_rows(
            method=method,
            experiment=experiment,
            run_name=run_name,
            gt_items=gt_items,
            matches=matches,
        ))
        confusions.append(confusion)

    coverage_per_incident_rows: List[Dict[str, Any]] = []
    coverage_correlation_rows: List[Dict[str, Any]] = []
    coverage_bin_rows: List[Dict[str, Any]] = []
    if coverage_rows:
        attach_coverage_to_incident_rows(incident_rows, coverage_rows)
        coverage_per_incident_rows, coverage_correlation_rows, coverage_bin_rows = compute_coverage_performance_diagnostics(incident_rows)

    progress_write(f"Aggregating {args.mode} metrics...", enabled=progress_enabled)
    by_method_exp = aggregate_by_method_experiment(run_rows)
    by_type = aggregate_by_incident_type(run_rows)
    by_gt_type = aggregate_by_gt_incident_type(run_rows)
    overall = aggregate_metrics(run_rows)
    scalability_performance_by_setting = aggregate_performance_by_scalability_setting(run_rows) if args.mode == "scalability" else []

    aggregate_payload = {
        "mode": args.mode,
        "results_root": str(args.results_root),
        "gt_roots": [str(x) for x in args.gt_roots],
        "num_result_run_dirs_discovered": len(result_dirs),
        "num_runs_evaluated": len(run_rows),
        "num_missing_gt": len(missing_gt),
        "overall": overall,
        "coverage_summary": coverage_summary,
        "coverage_performance_correlations": coverage_correlation_rows,
        "coverage_performance_by_coverage_bin": coverage_bin_rows,
        "coverage_performance_note": (
            "coverage_performance_correlations.csv reports Pearson/Spearman correlations between GT coverage_score "
            "and per-incident detection/quality metrics. coverage_performance_by_coverage_bin.csv reports recall/quality "
            "within low/medium/high coverage bands. For synthetic modes, coverage is non-circular only when an explicit "
            "source inventory is supplied with --coverage-source-jsonl; otherwise the result-support fallback is diagnostic."
        ),
        "by_method_experiment": by_method_exp,
        "by_ablation": by_method_exp if args.mode == "ablation" else [],
        "by_scalability_performance_setting": scalability_performance_by_setting,
        # Legacy/class-wise view: FPs are charged to the predicted incident type.
        "by_incident_type": by_type,
        "by_predicted_incident_type": by_type,
        # Scenario-conditioned view: for each GT incident type, all extra
        # predictions in runs of that type count as FPs.  Use this for
        # incident-category result panels.
        "by_gt_incident_type": by_gt_type,
        "f2_metric_definitions": {
            "overall_micro_f2": "Micro F2 over all predictions and GT records for a method/experiment.",
            "classwise_predicted_type_f2": "Existing by_incident_type F2; false positives are charged to the predicted label.",
            "gt_scenario_micro_f2": "F2 for runs containing a given GT incident type; every non-correct prediction in those runs is a false positive for that scenario.",
        },
        "spatial_temporal_metric_definitions": {
            "matched_tp_*": "Quality metrics computed only over correct one-to-one matched predictions; unmatched false positives are not included.",
            "scenario_union_coverage_iou": "For each GT incident type in a run, IoU between the union of all same-type predicted coverage regions and the union of GT regions.",
            "scenario_overcoverage_ratio": "Predicted same-type area outside the GT region divided by GT area; unbounded, lower is better.",
            "scenario_temporal_iou": "Temporal IoU between the union of all same-type predicted active intervals and the union of GT intervals.",
            "scenario_temporal_overcoverage_ratio": "Predicted same-type time outside the GT interval divided by GT duration; unbounded, lower is better.",
        },
        "settings": {
            "max_loc_error_km": args.max_loc_error_km,
            "min_spatial_iou": args.min_spatial_iou,
            "min_coverage_normalized": args.min_coverage_normalized,
            "default_single_step_minutes": args.default_single_step_minutes,
            "leakage_filter": args.leakage_filter,
            "leakage_filter_methods": args.leakage_filter_methods,
            "write_denoised_low_level": args.write_denoised_low_level,
            "denoised_methods": args.denoised_methods,
            "include_coverage": args.include_coverage,
            "coverage_source_jsonl": args.coverage_source_jsonl,
            "coverage_use_result_support": args.coverage_use_result_support,
            "coverage_allow_type_default": args.coverage_allow_type_default,
        },
    }
    write_json(output_dir / "aggregate_metrics.json", aggregate_payload)
    write_csv(output_dir / "run_metrics.csv", run_rows)
    write_json(output_dir / "run_metrics.json", run_rows)
    write_csv(output_dir / "match_records.csv", match_rows)
    write_json(output_dir / "match_records.json", match_rows)
    write_csv(output_dir / "gt_summary.csv", gt_summary_rows)
    write_json(output_dir / "gt_summary.json", gt_summary_rows)
    write_csv(output_dir / "aggregate_metrics_by_method_experiment.csv", by_method_exp)
    if args.mode == "ablation":
        write_csv(output_dir / "aggregate_metrics_by_ablation.csv", by_method_exp)
        write_json(output_dir / "aggregate_metrics_by_ablation.json", by_method_exp)
    if args.mode == "scalability":
        write_csv(output_dir / "scalability_performance_by_setting.csv", scalability_performance_by_setting)
        write_json(output_dir / "scalability_performance_by_setting.json", scalability_performance_by_setting)
    write_csv(output_dir / "aggregate_metrics_by_incident_type.csv", by_type)
    write_csv(output_dir / "aggregate_metrics_by_predicted_incident_type.csv", by_type)
    write_csv(output_dir / "aggregate_metrics_by_gt_incident_type.csv", by_gt_type)
    write_json(output_dir / "aggregate_metrics_by_gt_incident_type.json", by_gt_type)
    write_csv(output_dir / "per_gt_incident_performance.csv", incident_rows)
    write_json(output_dir / "per_gt_incident_performance.json", incident_rows)
    if coverage_per_incident_rows:
        write_csv(output_dir / "coverage_per_incident_performance.csv", coverage_per_incident_rows)
        write_json(output_dir / "coverage_per_incident_performance.json", coverage_per_incident_rows)
        write_csv(output_dir / "coverage_performance_correlations.csv", coverage_correlation_rows)
        write_json(output_dir / "coverage_performance_correlations.json", coverage_correlation_rows)
        write_csv(output_dir / "coverage_performance_by_coverage_bin.csv", coverage_bin_rows)
        write_json(output_dir / "coverage_performance_by_coverage_bin.json", coverage_bin_rows)

    confusion_rows = merge_confusions(confusions)
    write_csv(output_dir / "type_confusion_matrix.csv", confusion_rows)
    write_json(output_dir / "missing_ground_truth.json", missing_gt)
    write_json(output_dir / "prediction_leakage_filter_summary.json", leakage_filter_summaries)
    write_csv(output_dir / "prediction_leakage_filter_summary.csv", leakage_filter_summaries)

    if runtime_rows:
        runtime_agg = aggregate_runtime(runtime_rows)
        runtime_by_setting = aggregate_runtime_by_scalability_setting(runtime_rows) if args.mode == "scalability" else []
        scalability_combined = combine_scalability_performance_and_runtime(
            scalability_performance_by_setting,
            runtime_by_setting,
        ) if args.mode == "scalability" else []
        write_csv(output_dir / "scalability_runtime_runs.csv", runtime_rows)
        write_json(output_dir / "scalability_runtime_runs.json", runtime_rows)
        write_csv(output_dir / "scalability_runtime_summary.csv", runtime_agg)
        write_json(output_dir / "scalability_runtime_summary.json", {
            "runs": runtime_rows,
            "by_method_experiment": runtime_agg,
            "by_scalability_setting": runtime_by_setting,
        })
        if args.mode == "scalability":
            write_csv(output_dir / "scalability_runtime_by_setting.csv", runtime_by_setting)
            write_json(output_dir / "scalability_runtime_by_setting.json", runtime_by_setting)
            write_csv(output_dir / "scalability_summary_by_setting.csv", scalability_combined)
            write_json(output_dir / "scalability_summary_by_setting.json", scalability_combined)

    log(f"Wrote evaluation outputs to {output_dir}.")
    return aggregate_payload


# ---------------------------------------------------------------------------
# Synthetic high-level composition evaluation
# ---------------------------------------------------------------------------

COMPOSITION_GENERIC_TYPES = {
    "operational incident complex",
    "incident complex",
    "high level incident",
    "high level incident complex",
    "composite incident",
    "composite incident complex",
}


def discover_composition_result_dirs(
    results_root: Path,
    *,
    low_experiment_name: str,
    composition_experiment_name: str,
    composition_experiments: Sequence[str] = (),
    experiment_glob: Optional[str] = None,
    methods: Sequence[str] = (),
) -> List[Tuple[str, str, str, Path]]:
    """Discover source result folders for synth composition evaluation.

    Baselines are not stored as their own result folders; they are run in memory
    on the low-level output from these source folders.  By default this uses
    IncidentLens synth_low and synth_comp folders so the same run set supplies
    positive and negative composition examples.
    """
    out: List[Tuple[str, str, str, Path]] = []
    if not results_root.exists():
        return out

    wanted_experiments = [x for x in composition_experiments if str(x).strip()]
    if not wanted_experiments:
        wanted_experiments = [low_experiment_name, composition_experiment_name]
    wanted_set = set(wanted_experiments)

    method_filter = {m for m in methods if m}
    if not method_filter:
        method_filter = {"incidentlens"}

    for method_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        method = method_dir.name
        if method_filter and method not in method_filter:
            continue
        for exp_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            experiment = exp_dir.name
            if experiment_glob:
                if not exp_dir.match(experiment_glob) and experiment_glob not in experiment:
                    continue
            elif experiment not in wanted_set:
                continue
            for run_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
                if (
                    (run_dir / DENOISED_LOW_LEVEL_FILENAME).exists()
                    or (run_dir / LOW_LEVEL_FILENAME).exists()
                    or (run_dir / HIGH_LEVEL_FILENAME).exists()
                ):
                    out.append((method, experiment, run_dir.name, run_dir))
    return out


def build_plan_index(gt_roots: Sequence[str | Path]) -> Dict[str, Path]:
    """Index simulator composition plan files by common run aliases.

    The simulator stores these as either ``plan.json`` or ``*_plan.json`` inside
    each generated run folder, for example::

        simulator/generated/batch_incident_runs/earthquake_damage1/earthquake_damage_b9bf4812_plan.json
        simulator/generated/batch_small_area/civil_protest1/civil_protest_6810be80_plan.json

    Result folders may be named either by the simulator folder
    (``earthquake_damage1``) or by the payload ``run_id``
    (``earthquake_damage_b9bf4812``), so index both along with the file stem
    without the trailing ``_plan``.
    """
    index: Dict[str, Path] = {}

    def add_alias(alias: Any, path: Path) -> None:
        text = str(alias or "").strip()
        if text:
            index.setdefault(text, path)

    for root_value in gt_roots:
        root = Path(root_value)
        if not root.exists():
            continue
        for path in root.rglob("*plan.json"):
            if not path.is_file():
                continue
            if path.name != "plan.json" and not path.name.endswith("_plan.json"):
                continue

            # Folder alias: <gt_root>/<run_folder>/<something>_plan.json.
            add_alias(path.parent.name, path)

            # File-stem alias: earthquake_damage_b9bf4812_plan -> earthquake_damage_b9bf4812.
            stem_alias = re.sub(r"_plan$", "", path.stem)
            add_alias(stem_alias, path)

            try:
                payload = read_json(path)
            except Exception:
                payload = {}

            add_alias(payload.get("run_id"), path)

            # Some future/alternate plan formats may include an explicit folder
            # or incident-name field; harmlessly index them when present.
            for key in ("run_name", "incident_name", "folder_name", "incident_id"):
                add_alias(payload.get(key), path)
    return index


def _path_has_component(path_value: Any, component: str) -> bool:
    """Return True when a path contains a component with this exact name."""
    if not path_value:
        return False
    try:
        path = Path(str(path_value))
        return component in path.parts
    except Exception:
        return component in str(path_value).split("/")


def _first_present_value(data: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return None


def _composition_child_types_from_plan(plan_payload: Dict[str, Any], allowed_types: Sequence[str]) -> List[str]:
    rows = plan_payload.get("plan")
    if not isinstance(rows, list):
        return []
    out: List[str] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        typ = item.get("incident_type") or item.get("type") or item.get("event_type")
        if typ:
            out.append(canonical_type(typ, allowed_types))
    return out


def _primary_type(types: Sequence[str]) -> Optional[str]:
    if not types:
        return None
    counts: Dict[str, int] = {}
    first_index: Dict[str, int] = {}
    for idx, typ in enumerate(types):
        counts[typ] = counts.get(typ, 0) + 1
        first_index.setdefault(typ, idx)
    return sorted(counts, key=lambda t: (-counts[t], first_index[t]))[0]


def load_composition_ground_truth_for_run(
    *,
    run_name: str,
    experiment: str,
    plan_index: Dict[str, Path],
    allowed_types: Sequence[str],
    low_experiment_name: str,
    composition_experiment_name: str,
) -> Dict[str, Any]:
    """Load high-level-composition GT from plan.json.

    synth_low runs are negative examples even if their plan does not contain an
    explicit high_level flag.  synth_comp/batch_small_area runs normally have
    high_level=true in plan.json.  If a plan contains an explicit high-level
    type, that is used. Otherwise positive high-level examples default to the
    generic IncidentLens label "operational incident complex" while also
    recording compatible child/primary types for a secondary type-compatibility
    metric.
    """
    plan_path = plan_index.get(run_name)
    payload: Dict[str, Any] = {}
    if plan_path and plan_path.exists():
        try:
            payload = read_json(plan_path)
        except Exception:
            payload = {}

    has_plan = bool(payload)
    explicit_high_level = payload.get("high_level") if has_plan else None
    if explicit_high_level is None:
        is_high_level = experiment == composition_experiment_name or experiment.startswith("synth_comp")
    else:
        is_high_level = bool(explicit_high_level)

    if experiment == low_experiment_name:
        # The low-only synthetic suite is the negative class for composition.
        is_high_level = False

    child_types = _composition_child_types_from_plan(payload, allowed_types)
    primary = _primary_type(child_types)

    explicit_type = _first_present_value(payload, [
        "high_level_incident_type",
        "high_level_type",
        "composite_incident_type",
        "top_level_incident_type",
        "parent_incident_type",
        "incident_type",
        "type",
    ])
    if explicit_type:
        gt_type = canonical_type(explicit_type, allowed_types)
    elif is_high_level:
        gt_type = "operational incident complex"
    else:
        gt_type = None

    acceptable_types = []
    if gt_type:
        acceptable_types.append(gt_type)
    if primary:
        acceptable_types.append(primary)
    acceptable_types.extend(child_types)
    # A generic operational complex label should be acceptable for any positive
    # composition example because IncidentLens currently emits that ontology
    # term for composed incidents.
    if is_high_level:
        acceptable_types.append("operational incident complex")
    acceptable_types = list(dict.fromkeys(norm_label(x) for x in acceptable_types if x))

    return {
        "run_name": run_name,
        "experiment": experiment,
        "has_plan": has_plan,
        "plan_path": str(plan_path) if plan_path else None,
        "gt_is_high_level": bool(is_high_level),
        "gt_high_level_type": gt_type,
        "gt_primary_child_type": primary,
        "gt_child_incident_types": list(dict.fromkeys(child_types)),
        "gt_acceptable_types": acceptable_types,
    }


def _composition_type_correct(pred_type: Optional[str], target_type: Optional[str]) -> Optional[bool]:
    if not pred_type or not target_type:
        return None
    return norm_label(pred_type) == norm_label(target_type)


COMPOSITION_TYPE_MODIFIER_TOKENS = {
    "active",
    "broad",
    "campaign",
    "coordinated",
    "complex",
    "composite",
    "distributed",
    "event",
    "high",
    "incident",
    "level",
    "loose",
    "multi",
    "multisite",
    "operational",
    "parent",
    "period",
    "regional",
    "related",
    "same",
    "site",
    "style",
    "system",
    "top",
    "wave",
}


def _composition_core_tokens(value: Any) -> set[str]:
    """Return ontology-bearing tokens after dropping composition modifiers."""
    label = norm_label(value)
    return {tok for tok in label.split() if tok and tok not in COMPOSITION_TYPE_MODIFIER_TOKENS}


def _composition_labels_compatible(pred_type: Any, acceptable_type: Any) -> bool:
    """Fuzzy-but-controlled compatibility for composite type labels.

    This intentionally allows labels such as ``distributed wildfire complex`` to
    match ``wildfire`` and ``distributed multi-site civil protest complex`` to
    match ``large civil protest`` while avoiding naive substring mistakes such
    as matching the token ``fire`` inside ``wildfire``.
    """
    pred = norm_label(pred_type)
    acc = norm_label(acceptable_type)
    if not pred or not acc:
        return False
    if pred == acc:
        return True

    pred_tokens = pred.split()
    acc_tokens = acc.split()

    # Exact contiguous token phrase, not character substring.
    def phrase_in(tokens_small: List[str], tokens_large: List[str]) -> bool:
        if not tokens_small or len(tokens_small) > len(tokens_large):
            return False
        n = len(tokens_small)
        return any(tokens_large[i:i + n] == tokens_small for i in range(0, len(tokens_large) - n + 1))

    if phrase_in(acc_tokens, pred_tokens) or phrase_in(pred_tokens, acc_tokens):
        return True

    pred_core = _composition_core_tokens(pred)
    acc_core = _composition_core_tokens(acc)
    if not pred_core or not acc_core:
        return False

    # One-token ontology labels such as wildfire should match a longer complex
    # label when they appear as a token. For multi-token labels, permit modifier
    # differences such as large civil protest vs distributed civil protest.
    return pred_core.issubset(acc_core) or acc_core.issubset(pred_core)


def _composition_type_compatible(pred_type: Optional[str], acceptable_types: Sequence[str]) -> Optional[bool]:
    if not pred_type or not acceptable_types:
        return None
    return any(_composition_labels_compatible(pred_type, item) for item in acceptable_types if item)


def _composition_filter_types_from_args(args: argparse.Namespace, allowed_types: Sequence[str]) -> List[str]:
    """Return canonical GT child/primary types requested by composition filters."""
    raw_types: List[str] = []
    if getattr(args, "composition_civil_protest_terrorist_only", False):
        raw_types.extend([
            "civil protest",
            "large civil protest",
            "terrorist incident",
            "terrorist attack",
        ])
    raw_types.extend(getattr(args, "composition_gt_incident_types", None) or [])

    out: List[str] = []
    for typ in raw_types:
        canonical = canonical_type(typ, allowed_types)
        if canonical and canonical != "unknown" and canonical not in out:
            out.append(canonical)
    return out


def _composition_gt_matches_type_filter(gt: Dict[str, Any], target_types: Sequence[str]) -> bool:
    """Return True when a run's GT plan matches the requested child incident types.

    The composition benchmark's parent label may be generic (for example,
    ``operational incident complex``), so filtering should be based primarily on
    the child incident types listed in ``*_plan.json``.  We also check the
    primary child type and explicit high-level type as fallbacks for alternate
    future plan schemas.
    """
    if not target_types:
        return True

    target_set = {norm_label(x) for x in target_types if x}
    candidate_values: List[str] = []
    candidate_values.extend(gt.get("gt_child_incident_types") or [])
    if gt.get("gt_primary_child_type"):
        candidate_values.append(str(gt.get("gt_primary_child_type")))
    if gt.get("gt_high_level_type"):
        candidate_values.append(str(gt.get("gt_high_level_type")))

    for value in candidate_values:
        value_norm = norm_label(value)
        if not value_norm:
            continue
        if value_norm in target_set:
            return True
        # Token-aware compatibility handles modifier labels but avoids matching
        # generic substrings such as fire inside wildfire.
        compat = _composition_type_compatible(value_norm, list(target_set))
        if compat:
            return True
    return False


def _high_level_item_is_accepted_composite(item: Dict[str, Any]) -> bool:
    """Return whether a high_level_incidents entry is accepted as composite.

    Newer IncidentLens outputs put the actual decision under
    composition_reasoning.llm_or_heuristic_decision.is_composite.  Older outputs
    may not have the flag; for backward compatibility, entries already in
    high_level_incidents are treated as accepted unless an explicit flag says
    false.  Rejected candidate groups under composition_reasoning.rejected_groups
    are never inspected by this evaluator.
    """
    direct = item.get("is_composite")
    if isinstance(direct, bool):
        return direct

    reasoning = item.get("composition_reasoning") if isinstance(item.get("composition_reasoning"), dict) else {}
    decision = reasoning.get("llm_or_heuristic_decision")
    if isinstance(decision, dict) and isinstance(decision.get("is_composite"), bool):
        return bool(decision.get("is_composite"))

    # Some diagnostics use a shorter ``decision`` key.  This is mainly useful
    # if an accepted group is ever copied into high_level_incidents from the
    # rejected/candidate representation.
    decision2 = reasoning.get("decision")
    if isinstance(decision2, dict) and isinstance(decision2.get("is_composite"), bool):
        return bool(decision2.get("is_composite"))

    return True


def _high_level_item_has_heuristic_decision(item: Dict[str, Any]) -> bool:
    """Return True when the accepted high-level item came from heuristic fallback.

    Current IncidentLens fallback outputs do not always include a separate
    backend field; the most reliable marker in saved results is the rationale
    prefix/text: ``Heuristic fallback ... enable INCIDENT_COMPOSITION_BACKEND``.
    Keep this check broad enough to cover older saved outputs, but do not treat
    ordinary use of the word heuristic in an LLM explanation as fallback unless
    it appears with fallback/backend language.
    """
    reasoning = item.get("composition_reasoning") if isinstance(item.get("composition_reasoning"), dict) else {}
    decisions: List[Dict[str, Any]] = []
    for key in ("llm_or_heuristic_decision", "decision"):
        value = reasoning.get(key)
        if isinstance(value, dict):
            decisions.append(value)

    backend_values: List[str] = []
    for container in [item, reasoning, *decisions]:
        for key in ("backend", "decision_backend", "reasoning_backend", "source", "method", "model_backend"):
            value = container.get(key) if isinstance(container, dict) else None
            if value is not None:
                backend_values.append(norm_label(value))

    if any(value in {"heuristic", "heuristic fallback", "fallback heuristic"} for value in backend_values):
        return True

    rationale_parts: List[str] = []
    for container in [item, reasoning, *decisions]:
        if not isinstance(container, dict):
            continue
        for key in ("rationale", "reasoning", "explanation", "summary"):
            value = container.get(key)
            if isinstance(value, str):
                rationale_parts.append(value.lower())
    rationale = "\n".join(rationale_parts)
    return (
        "heuristic fallback" in rationale
        or "enable incident_composition_backend" in rationale
        or "enable INCIDENT_COMPOSITION_BACKEND".lower() in rationale
    )


def _high_level_item_is_llm_decision(item: Dict[str, Any]) -> bool:
    """Return True when an IncidentLens high-level entry should count for LLM-only eval."""
    if _high_level_item_has_heuristic_decision(item):
        return False

    reasoning = item.get("composition_reasoning") if isinstance(item.get("composition_reasoning"), dict) else {}
    decision = reasoning.get("llm_or_heuristic_decision")
    decision2 = reasoning.get("decision")
    containers = [item, reasoning]
    if isinstance(decision, dict):
        containers.append(decision)
    if isinstance(decision2, dict):
        containers.append(decision2)

    # If an explicit backend/source says LLM/model/langchain, accept it.
    for container in containers:
        for key in ("backend", "decision_backend", "reasoning_backend", "source", "method", "model_backend"):
            value = container.get(key) if isinstance(container, dict) else None
            label = norm_label(value)
            if label in {"llm", "language model", "langchain", "openai", "model"}:
                return True
            if "llm" in label or "langchain" in label or "language model" in label:
                return True

    # Most saved LLM-backed outputs have a natural-language decision object but
    # no explicit backend field.  After excluding heuristic fallback text above,
    # treat llm_or_heuristic_decision entries as LLM decisions.  Older entries
    # without any decision object are excluded in LLM-only mode.
    return isinstance(decision, dict)


def _prediction_types_from_high_level_payload(
    payload: Dict[str, Any],
    allowed_types: Sequence[str],
    *,
    llm_only: bool = False,
) -> List[Dict[str, Any]]:
    rows = payload.get("high_level_incidents", [])
    if not isinstance(rows, list):
        rows = []
    out: List[Dict[str, Any]] = []
    seen: set[Tuple[str, Tuple[str, ...]]] = set()
    for idx, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        if not _high_level_item_is_accepted_composite(item):
            continue
        if llm_only and not _high_level_item_is_llm_decision(item):
            continue
        raw_type = item.get("incident_type") or item.get("high_level_type") or item.get("type")
        raw_type_norm = norm_label(raw_type)
        pred_type = raw_type_norm
        if not pred_type:
            pred_type = "operational incident complex"
        elif pred_type not in COMPOSITION_GENERIC_TYPES:
            pred_type = canonical_type(pred_type, allowed_types)
        child_ids = item.get("child_prediction_ids")
        if not isinstance(child_ids, list):
            child_ids = []
        child_tuple = tuple(sorted(str(x) for x in child_ids))
        key = (pred_type, child_tuple)
        if key in seen:
            continue
        seen.add(key)
        reasoning = item.get("composition_reasoning") if isinstance(item.get("composition_reasoning"), dict) else {}
        conf = safe_float(item.get("confidence"), None)
        if conf is None:
            conf = safe_float(reasoning.get("confidence"), None)
        out.append({
            "prediction_id": str(item.get("composite_id") or item.get("prediction_id") or f"high_level_{idx:04d}"),
            "incident_type": pred_type,
            "raw_incident_type": raw_type_norm or pred_type,
            "confidence": conf,
            "child_prediction_ids": list(child_tuple),
            "raw": item,
        })
    return out


def load_incidentlens_composition_predictions(
    result_dir: Path,
    allowed_types: Sequence[str],
    *,
    llm_only: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = result_dir / HIGH_LEVEL_FILENAME
    if not path.exists():
        return [], {"high_level_source_file": str(path), "source_exists": False}
    try:
        payload = read_json(path)
    except Exception as exc:
        return [], {"high_level_source_file": str(path), "source_exists": True, "read_error": str(exc)}
    raw_rows = payload.get("high_level_incidents", []) if isinstance(payload.get("high_level_incidents"), list) else []
    accepted_rows = [item for item in raw_rows if isinstance(item, dict) and _high_level_item_is_accepted_composite(item)]
    llm_rows = [item for item in accepted_rows if _high_level_item_is_llm_decision(item)]
    heuristic_rows = [item for item in accepted_rows if _high_level_item_has_heuristic_decision(item)]
    predictions = _prediction_types_from_high_level_payload(payload, allowed_types, llm_only=llm_only)
    return predictions, {
        "high_level_source_file": str(path),
        "source_exists": True,
        "incidentlens_llm_only": bool(llm_only),
        "raw_num_high_level_incidents": len(raw_rows),
        "accepted_num_high_level_incidents": len(accepted_rows),
        "accepted_llm_num_high_level_incidents": len(llm_rows),
        "ignored_heuristic_num_high_level_incidents": len(heuristic_rows) if llm_only else 0,
        "accepted_deduped_num_high_level_incidents": len(predictions),
        "deduped_num_high_level_incidents": len(predictions),
    }


def load_composition_baseline_low_level_rows(
    result_dir: Path,
    *,
    source_method: str,
    experiment: str,
    first_owner_index: Optional[Dict[Tuple[str, str], Dict[str, str]]],
    leakage_filter_enabled: bool,
    leakage_filter_methods: Sequence[str],
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str], Dict[str, Any]]:
    """Return optional in-memory denoised rows for composition baselines.

    If low_level_results_denoised.json already exists, the baseline module should
    read it directly, so this returns ``rows_override=None``.  If the denoised
    file is absent, this loads low_level_results.json and applies the same
    cumulative prediction-identity denoising used by the low-level evaluator, but
    keeps the result in memory so baseline evaluation does not write files.
    """
    denoised_path = result_dir / DENOISED_LOW_LEVEL_FILENAME
    if denoised_path.exists():
        return None, None, {
            "baseline_low_level_input_mode": "materialized_denoised_file",
            "baseline_low_level_source_file": str(denoised_path),
            "baseline_in_memory_denoising_applied": False,
        }

    raw_path = result_dir / LOW_LEVEL_FILENAME
    if not raw_path.exists():
        return [], str(raw_path) + "#missing", {
            "baseline_low_level_input_mode": "missing_low_level_file",
            "baseline_low_level_source_file": str(raw_path),
            "baseline_in_memory_denoising_applied": False,
            "raw_predictions": 0,
            "kept_predictions": 0,
            "dropped_predictions": 0,
        }

    try:
        payload = read_json(raw_path)
    except Exception as exc:
        return [], str(raw_path) + "#read_error", {
            "baseline_low_level_input_mode": "read_error",
            "baseline_low_level_source_file": str(raw_path),
            "baseline_in_memory_denoising_applied": False,
            "baseline_low_level_read_error": str(exc),
            "raw_predictions": 0,
            "kept_predictions": 0,
            "dropped_predictions": 0,
        }

    rows = payload.get("low_level_incidents", [])
    if not isinstance(rows, list):
        return [], str(raw_path) + "#malformed", {
            "baseline_low_level_input_mode": "malformed_low_level_file",
            "baseline_low_level_source_file": str(raw_path),
            "baseline_in_memory_denoising_applied": False,
            "raw_predictions": 0,
            "kept_predictions": 0,
            "dropped_predictions": 0,
        }

    kept_rows, leakage_summary = filter_leaked_prediction_rows(
        rows,
        method=source_method,
        experiment=experiment,
        result_dir=result_dir,
        first_owner_index=first_owner_index,
        enabled=leakage_filter_enabled,
        filter_methods=leakage_filter_methods,
    )
    summary = dict(leakage_summary)
    summary.update({
        "baseline_low_level_input_mode": "raw_file_with_in_memory_denoising",
        "baseline_low_level_source_file": str(raw_path),
        "baseline_in_memory_denoising_applied": bool(leakage_filter_enabled),
        "baseline_in_memory_source_label": str(raw_path) + "#in_memory_denoised",
    })
    return kept_rows, summary["baseline_in_memory_source_label"], summary


def load_composition_baseline_module() -> Any:
    """Import detection.baselines.composition_baselines with a local fallback."""
    try:
        from detection.baselines import composition_baselines as module  # type: ignore
        return module
    except Exception:
        import importlib
        return importlib.import_module("composition_baselines")


def _baseline_payload_to_predictions(payload: Dict[str, Any], allowed_types: Sequence[str]) -> List[Dict[str, Any]]:
    return _prediction_types_from_high_level_payload(payload, allowed_types, llm_only=False)


def _best_prediction(predictions: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not predictions:
        return None
    def key(item: Dict[str, Any]) -> Tuple[float, int]:
        conf = safe_float(item.get("confidence"), None)
        return ((conf if conf is not None else -1.0), len(item.get("child_prediction_ids") or []))
    return sorted(predictions, key=key, reverse=True)[0]


def evaluate_composition_prediction_set(
    *,
    source_method: str,
    eval_method: str,
    experiment: str,
    run_name: str,
    result_dir: Path,
    gt: Dict[str, Any],
    predictions: Sequence[Dict[str, Any]],
    source_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    pred_is_high = bool(predictions)
    gt_is_high = bool(gt.get("gt_is_high_level"))
    tp = int(gt_is_high and pred_is_high)
    fp = int((not gt_is_high) and pred_is_high)
    fn = int(gt_is_high and (not pred_is_high))
    tn = int((not gt_is_high) and (not pred_is_high))

    best = _best_prediction(predictions)
    pred_type = best.get("incident_type") if best else None

    strict_type_correct = None
    compatible_type_correct = None
    if tp:
        target_type = gt.get("gt_high_level_type")
        acceptable_types = gt.get("gt_acceptable_types") or []

        strict_values = [
            _composition_type_correct(item.get("incident_type"), target_type)
            for item in predictions
            if item.get("incident_type") and target_type
        ]
        strict_values = [v for v in strict_values if v is not None]
        strict_type_correct = any(strict_values) if strict_values else None

        compat_values = [
            _composition_type_compatible(item.get("incident_type"), acceptable_types)
            for item in predictions
            if item.get("incident_type") and acceptable_types
        ]
        compat_values = [v for v in compat_values if v is not None]
        compatible_type_correct = any(compat_values) if compat_values else None

    child_counts = [len(item.get("child_prediction_ids") or []) for item in predictions]
    row = {
        "source_method": source_method,
        "method": eval_method,
        "experiment": experiment,
        "variant": experiment_variant_name(experiment),
        "run_name": run_name,
        "result_dir": str(result_dir),
        "gt_is_high_level": gt_is_high,
        "gt_high_level_type": gt.get("gt_high_level_type"),
        "gt_primary_child_type": gt.get("gt_primary_child_type"),
        "gt_child_incident_types": ";".join(gt.get("gt_child_incident_types") or []),
        "gt_acceptable_types": ";".join(gt.get("gt_acceptable_types") or []),
        "gt_plan_path": gt.get("plan_path"),
        "pred_is_high_level": pred_is_high,
        "pred_num_high_level_incidents": len(predictions),
        "pred_high_level_type": pred_type,
        "pred_high_level_types": ";".join(sorted({str(item.get("incident_type")) for item in predictions if item.get("incident_type")})),
        "pred_raw_high_level_types": ";".join(sorted({str(item.get("raw_incident_type")) for item in predictions if item.get("raw_incident_type")})),
        "pred_max_child_count": max(child_counts) if child_counts else 0,
        "pred_total_child_links": sum(child_counts),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": fbeta(tp, fp, fn, beta=1.0),
        "f2": fbeta(tp, fp, fn, beta=2.0),
        "strict_type_correct_on_tp": strict_type_correct,
        "compatible_type_correct_on_tp": compatible_type_correct,
    }
    if source_summary:
        row.update({k: v for k, v in source_summary.items() if not isinstance(v, (dict, list))})
    return row


def aggregate_composition_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    tp = int(sum(int(row.get("tp") or 0) for row in rows))
    fp = int(sum(int(row.get("fp") or 0) for row in rows))
    fn = int(sum(int(row.get("fn") or 0) for row in rows))
    tn = int(sum(int(row.get("tn") or 0) for row in rows))
    detected_tp_rows = [row for row in rows if int(row.get("tp") or 0) == 1]
    strict_vals = [row.get("strict_type_correct_on_tp") for row in detected_tp_rows if row.get("strict_type_correct_on_tp") is not None]
    compat_vals = [row.get("compatible_type_correct_on_tp") for row in detected_tp_rows if row.get("compatible_type_correct_on_tp") is not None]
    return {
        "num_runs": len(rows),
        "num_positive_gt_runs": int(sum(1 for row in rows if row.get("gt_is_high_level"))),
        "num_negative_gt_runs": int(sum(1 for row in rows if not row.get("gt_is_high_level"))),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": fbeta(tp, fp, fn, beta=1.0),
        "f2": fbeta(tp, fp, fn, beta=2.0),
        "false_positive_rate": ratio(fp, fp + tn),
        "true_negative_rate": ratio(tn, fp + tn),
        "accuracy": ratio(tp + tn, tp + fp + fn + tn),
        "strict_type_accuracy_on_detected_tp": fraction_true(strict_vals),
        "compatible_type_accuracy_on_detected_tp": fraction_true(compat_vals),
    }


def aggregate_composition_by(rows: List[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k) for k in keys), []).append(row)
    out: List[Dict[str, Any]] = []
    for key_values, items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row = {key: value for key, value in zip(keys, key_values)}
        row.update(aggregate_composition_metrics(items))
        out.append(row)
    return out


def evaluate_synth_composition(args: argparse.Namespace) -> Dict[str, Any]:
    allowed_types = load_allowed_types(args.incident_types)
    plan_index = build_plan_index(args.gt_roots)
    result_dirs = discover_composition_result_dirs(
        Path(args.results_root),
        low_experiment_name=args.low_experiment_name,
        composition_experiment_name=args.composition_experiment_name,
        composition_experiments=args.composition_experiments or (),
        experiment_glob=args.experiment_glob,
        methods=args.methods or (),
    )
    log(f"Discovered {len(result_dirs)} composition source result run directories under {args.results_root}.")
    log(f"Indexed {len(plan_index)} simulator plan.json aliases from {args.gt_roots}.")

    first_owner_index = build_prediction_first_owner_index(result_dirs) if args.leakage_filter else {}
    if args.leakage_filter:
        num_keys = sum(len(v) for v in first_owner_index.values())
        log(f"Composition in-memory denoising index contains {num_keys} prediction identity keys.")

    baseline_module = None
    baseline_error = None
    if args.composition_baselines:
        try:
            baseline_module = load_composition_baseline_module()
        except Exception as exc:
            baseline_error = str(exc)
            log(f"Could not import composition baselines: {exc}")

    run_rows: List[Dict[str, Any]] = []
    missing_gt: List[Dict[str, Any]] = []
    skipped_non_batch_small_area: List[Dict[str, Any]] = []
    skipped_incident_type_filter: List[Dict[str, Any]] = []
    composition_target_types = _composition_filter_types_from_args(args, allowed_types)
    if composition_target_types:
        log("Composition GT incident-type filter: " + ", ".join(composition_target_types))
    progress_enabled = bool(args.progress_bar)
    if progress_enabled and tqdm is None:
        log("tqdm is not installed; using plain progress logs. Install with `pip install tqdm` for a progress bar.")
        progress_enabled = False

    for source_method, experiment, run_name, result_dir in progress_iter(
        result_dirs,
        total=len(result_dirs),
        desc="evaluate synth_composition",
        enabled=progress_enabled,
    ):
        gt = load_composition_ground_truth_for_run(
            run_name=run_name,
            experiment=experiment,
            plan_index=plan_index,
            allowed_types=allowed_types,
            low_experiment_name=args.low_experiment_name,
            composition_experiment_name=args.composition_experiment_name,
        )
        if args.composition_batch_small_area_only:
            plan_path = gt.get("plan_path")
            if not _path_has_component(plan_path, "batch_small_area"):
                skipped_non_batch_small_area.append({
                    "source_method": source_method,
                    "experiment": experiment,
                    "run_name": run_name,
                    "result_dir": str(result_dir),
                    "plan_path": plan_path,
                    "reason": "plan_not_under_batch_small_area",
                })
                continue

        if composition_target_types and not _composition_gt_matches_type_filter(gt, composition_target_types):
            skipped_incident_type_filter.append({
                "source_method": source_method,
                "experiment": experiment,
                "run_name": run_name,
                "result_dir": str(result_dir),
                "plan_path": gt.get("plan_path"),
                "gt_primary_child_type": gt.get("gt_primary_child_type"),
                "gt_child_incident_types": gt.get("gt_child_incident_types") or [],
                "target_types": composition_target_types,
                "reason": "gt_child_type_not_in_composition_incident_type_filter",
            })
            continue

        if experiment == args.composition_experiment_name and not gt.get("has_plan"):
            missing_gt.append({
                "source_method": source_method,
                "experiment": experiment,
                "run_name": run_name,
                "result_dir": str(result_dir),
                "reason": "missing_plan_json_for_positive_composition_run",
            })
            if args.skip_missing_gt:
                continue

        if args.evaluate_incidentlens_composition:
            preds, summary = load_incidentlens_composition_predictions(
                result_dir,
                allowed_types,
                llm_only=args.incidentlens_composition_llm_only,
            )
            run_rows.append(evaluate_composition_prediction_set(
                source_method=source_method,
                eval_method="incidentlens",
                experiment=experiment,
                run_name=run_name,
                result_dir=result_dir,
                gt=gt,
                predictions=preds,
                source_summary=summary,
            ))

        if baseline_module is not None:
            baseline_rows_override, baseline_source_label, baseline_input_summary = load_composition_baseline_low_level_rows(
                result_dir,
                source_method=source_method,
                experiment=experiment,
                first_owner_index=first_owner_index,
                leakage_filter_enabled=args.leakage_filter,
                leakage_filter_methods=args.leakage_filter_methods or (),
            )
            for baseline_name in args.composition_baselines:
                if baseline_name == "simple_proximity_only_linker":
                    payload = baseline_module.simple_proximity_only_linker(
                        result_dir,
                        max_distance_km=args.composition_proximity_max_distance_km,
                        min_children=args.composition_min_children,
                        prefer_denoised=True,
                        low_level_rows=baseline_rows_override,
                        source_label=baseline_source_label,
                    )
                elif baseline_name == "same_type_temporal_overlap_linker":
                    payload = baseline_module.same_type_temporal_overlap_linker(
                        result_dir,
                        max_temporal_gap_hours=args.composition_same_type_temporal_gap_hours,
                        max_distance_km=args.composition_same_type_max_distance_km,
                        min_children=args.composition_min_children,
                        prefer_denoised=True,
                        low_level_rows=baseline_rows_override,
                        source_label=baseline_source_label,
                    )
                else:
                    log(f"Skipping unknown composition baseline: {baseline_name}")
                    continue
                preds = _baseline_payload_to_predictions(payload, allowed_types)
                source_summary = dict(baseline_input_summary)
                source_summary.update({
                    "baseline_low_level_source_file": payload.get("low_level_source_file"),
                    "baseline_num_input_low_level_incidents": payload.get("num_input_low_level_incidents"),
                })
                run_rows.append(evaluate_composition_prediction_set(
                    source_method=source_method,
                    eval_method=baseline_name,
                    experiment=experiment,
                    run_name=run_name,
                    result_dir=result_dir,
                    gt=gt,
                    predictions=preds,
                    source_summary=source_summary,
                ))

    output_dir = Path(args.output_dir) / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    by_method = aggregate_composition_by(run_rows, ["method"])
    by_method_experiment = aggregate_composition_by(run_rows, ["method", "experiment"])
    by_experiment = aggregate_composition_by(run_rows, ["experiment"])
    overall = aggregate_composition_metrics(run_rows)

    progress_write("Writing low_real output files...", enabled=progress_enabled)
    write_csv(output_dir / "run_metrics.csv", run_rows)
    write_json(output_dir / "run_metrics.json", run_rows)
    write_csv(output_dir / "aggregate_metrics_by_method.csv", by_method)
    write_json(output_dir / "aggregate_metrics_by_method.json", by_method)
    write_csv(output_dir / "aggregate_metrics_by_method_experiment.csv", by_method_experiment)
    write_json(output_dir / "aggregate_metrics_by_method_experiment.json", by_method_experiment)
    write_csv(output_dir / "aggregate_metrics_by_experiment.csv", by_experiment)
    write_json(output_dir / "missing_ground_truth.json", missing_gt)
    write_json(output_dir / "skipped_non_batch_small_area.json", skipped_non_batch_small_area)
    write_json(output_dir / "skipped_composition_incident_type_filter.json", skipped_incident_type_filter)

    payload = {
        "mode": args.mode,
        "results_root": str(args.results_root),
        "gt_roots": [str(x) for x in args.gt_roots],
        "num_result_run_dirs_discovered": len(result_dirs),
        "num_runs_evaluated": len(run_rows),
        "num_missing_gt": len(missing_gt),
        "num_skipped_non_batch_small_area": len(skipped_non_batch_small_area),
        "num_skipped_composition_incident_type_filter": len(skipped_incident_type_filter),
        "overall": overall,
        "by_method": by_method,
        "by_method_experiment": by_method_experiment,
        "by_experiment": by_experiment,
        "baseline_import_error": baseline_error,
        "metric_definitions": {
            "precision_recall_f1_f2": "Binary high-level composition detection over synth_comp positives and synth_low negatives; type is not required for TP.",
            "strict_type_accuracy_on_detected_tp": "Among detected positive GT runs, at least one accepted predicted composite type equals the explicit or default GT high-level type. If plan.json lacks an explicit high-level type, positives default to operational incident complex.",
            "compatible_type_accuracy_on_detected_tp": "Among detected positive GT runs, at least one accepted predicted composite type is compatible with the GT high-level type, primary child type, any child type in *_plan.json, or operational incident complex. Modifier labels such as distributed wildfire complex are matched to wildfire by token-level compatibility.",
        },
        "settings": {
            "low_experiment_name": args.low_experiment_name,
            "composition_experiment_name": args.composition_experiment_name,
            "composition_experiments": args.composition_experiments,
            "composition_baselines": args.composition_baselines,
            "composition_proximity_max_distance_km": args.composition_proximity_max_distance_km,
            "composition_same_type_temporal_gap_hours": args.composition_same_type_temporal_gap_hours,
            "composition_same_type_max_distance_km": args.composition_same_type_max_distance_km,
            "composition_min_children": args.composition_min_children,
            "composition_batch_small_area_only": args.composition_batch_small_area_only,
            "composition_civil_protest_terrorist_only": args.composition_civil_protest_terrorist_only,
            "composition_gt_incident_types": args.composition_gt_incident_types,
            "composition_target_types_canonical": composition_target_types,
        },
    }
    write_json(output_dir / "aggregate_metrics.json", payload)
    if args.composition_batch_small_area_only:
        log(f"Skipped {len(skipped_non_batch_small_area)} non-batch_small_area composition source runs.")
    if composition_target_types:
        log(f"Skipped {len(skipped_incident_type_filter)} composition source runs outside the requested GT incident types.")
    log(f"Wrote composition evaluation outputs to {output_dir}.")
    return payload

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["low_incident", "low_real", "ablation", "scalability", "throughput", "synth_composition", "coverage_smoke", "ablation_and_scalability"],
        default="low_incident",
        help=(
            "Which result set to evaluate. ablation evaluates only architectural/modality ablation "
            "folders under results/incidentlens by default; scalability evaluates ablation_scalability_scalability_* "
            "and timing_scalability_* folders, producing performance and with/without observation timing summaries; "
            "throughput evaluates actual_timing_dup* controlled duplication folders and summarizes end-to-end "
            "reports/sec; synth_composition evaluates high-level composition from synth_comp/synth_low; "
            "low_real evaluates real-data date-folder low-level outputs against low_level_gt_corrected.json; "
            "ablation_and_scalability is the older combined mode."
        ),
    )
    parser.add_argument(
        "--results-root",
        default="evaluation/results",
        help="Root containing <method>/<experiment>/<run>/ result directories. Use 'results' if that is your actual root.",
    )
    parser.add_argument(
        "--gt-roots",
        nargs="+",
        default=[
            "simulator/generated/batch_incident_runs",
            "simulator/generated/batch_small_area",
        ],
        help="Simulator roots to search for ground-truth incident folders.",
    )
    parser.add_argument(
        "--incident-types",
        default="evaluation/incident_list_synth_batch.txt",
        help="Optional incident type list for canonicalization.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/evaluation_summary",
        help="Directory where evaluation CSV/JSON outputs are written.",
    )
    parser.add_argument(
        "--low-experiment-name",
        default="synth_low",
        help="Experiment directory name used by --mode low_incident.",
    )
    parser.add_argument(
        "--coverage-smoke-results-prefix",
        default="coverage_smoke",
        help="Experiment prefix used by --mode coverage_smoke, e.g. coverage_smoke_none/low/medium/high.",
    )
    parser.add_argument(
        "--coverage-smoke-reports-filename-template",
        default="{experiment}_reports.jsonl",
        help=(
            "Filename template for source REPORT JSONL files written into each synthetic incident folder by "
            "run_experiments.py --exp-type coverage_smoke. Available fields: experiment, case, condition, prefix, method, run_name."
        ),
    )
    parser.add_argument(
        "--coverage-smoke-allow-type-default",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For --mode coverage_smoke, allow type-default coverage priors when a condition has no source records. "
            "Disabled by default so the no-coverage condition scores C=0."
        ),
    )
    parser.add_argument(
        "--real-experiment-name",
        default="real_all_incidents",
        help="Experiment directory name used by --mode low_real.",
    )
    parser.add_argument(
        "--real-gt-path",
        default="evaluation/ground_truth/real/low_level_gt_corrected.json",
        help="Ground-truth JSON for --mode low_real.",
    )
    parser.add_argument(
        "--real-temporal-match-window-hours",
        type=float,
        default=24.0,
        help=(
            "For --mode low_real, allow a prediction point/interval to match a GT "
            "incident when it falls within this many hours of the GT active interval. "
            "This compensates for real-data date-folder boundaries."
        ),
    )
    parser.add_argument(
        "--real-default-duration-hours",
        type=float,
        default=24.0,
        help=(
            "For --mode low_real, use this duration when a GT incident has a missing "
            "or zero-length end_datetime_pacific."
        ),
    )
    parser.add_argument(
        "--real-no-spatial-matching",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For --mode low_real, disable the spatial matching gate and match by canonical "
            "incident type plus temporal overlap/window only. By default, GT textual locations "
            "are geocoded and spatial matching is applied when GT geometries are available."
        ),
    )
    parser.add_argument(
        "--real-spatial-matching",
        action="store_false",
        dest="real_no_spatial_matching",
        help=(
            "For --mode low_real, explicitly enable the spatial matching gate when GT geometries are available. "
            "This is the default; this flag is kept for backward-compatible scripts."
        ),
    )
    parser.add_argument(
        "--real-geocode-gt-locations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For --mode low_real, resolve textual GT locations from low_level_gt_corrected.json "
            "using GeoManager/cache so spatial matching and localization metrics can be computed. "
            "Enabled by default; use --no-real-geocode-gt-locations to disable."
        ),
    )
    parser.add_argument(
        "--real-geo-cache-path",
        default="evaluation/geo_cache/geo_region_cache.json",
        help="GeoManager cache path used by --real-geocode-gt-locations.",
    )
    parser.add_argument(
        "--real-geo-cache-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For --mode low_real GT geocoding, only read cached GeoManager records and do not "
            "make live OSMnx/Google geocoding calls."
        ),
    )
    parser.add_argument(
        "--real-geo-force-refresh",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force GeoManager to refresh GT location geocoding records when live geocoding is enabled.",
    )
    parser.add_argument(
        "--prefer-denoised-low-level",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Prefer low_level_results_denoised.json over low_level_results.json when it exists. "
            "This is especially useful for --mode low_real date folders."
        ),
    )
    parser.add_argument(
        "--real-rationale-type-guesses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For --mode low_real, expose alternate incident-type guesses mentioned in the "
            "prediction rationale as type alternatives at the same space/time footprint. "
            "Currently recognizes protest, urban fire, wildfire, and terrorist/terrorism mentions. "
            "Enabled by default; use --no-real-rationale-type-guesses to disable."
        ),
    )
    parser.add_argument(
        "--real-merge-prediction-updates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For --mode low_real, merge same-type spatially/temporally compatible prediction "
            "updates across date folders before scoring, so repeated IncidentLens updates are not "
            "counted as separate false positives."
        ),
    )
    parser.add_argument(
        "--real-merge-temporal-gap-hours",
        type=float,
        default=24.0,
        help="Maximum temporal gap for merging same-type real prediction updates.",
    )
    parser.add_argument(
        "--real-merge-spatial-distance-km",
        type=float,
        default=10.0,
        help="Maximum centroid distance for merging same-type real prediction updates.",
    )
    parser.add_argument(
        "--real-merge-min-spatial-iou",
        type=float,
        default=0.01,
        help="Minimum coverage/source IoU for merging same-type real prediction updates.",
    )
    parser.add_argument(
        "--experiment-glob",
        default=None,
        help="Optional substring/glob-like experiment filter.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help=(
            "Optional method names to evaluate, e.g. incidentlens generic_propagation "
            "satscan_background hawkes_event_detector. If omitted in low_real mode, "
            "the default real suite now includes satscan_background and hawkes_event_detector."
        ),
    )
    parser.add_argument(
        "--synth-low-all-baselines",
        action="store_true",
        help=(
            "Convenience shorthand for evaluating the full default synth_low baseline suite, "
            "including satscan_background and hawkes_event_detector. Missing result folders are ignored."
        ),
    )
    parser.add_argument(
        "--synth-low-background-baselines-only",
        action="store_true",
        help=(
            "Convenience shorthand for evaluating only satscan_background and hawkes_event_detector in synth_low."
        ),
    )
    parser.add_argument(
        "--include-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compute per-GT coverage/observability scores and coverage-weighted recall/F-scores. "
            "Integrated for --mode low_real and synthetic low-level modes."
        ),
    )
    parser.add_argument(
        "--coverage-source-jsonl",
        nargs="*",
        default=None,
        help=(
            "Optional normalized REPORT/observation JSONL files or directories used as the source inventory for coverage scoring. "
            "If omitted, coverage can fall back to support points in low_level_results.json."
        ),
    )
    parser.add_argument(
        "--coverage-auto-source-inventory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --coverage-source-jsonl is omitted, automatically look for real_emitter coverage source inventories, "
            "including evaluation/temp/real_emitter_ordered_reports.jsonl and evaluation/temp/_real_emitter_profile_cache/."
        ),
    )
    parser.add_argument(
        "--coverage-auto-temp-roots",
        nargs="*",
        default=None,
        help=(
            "Optional temp roots to search for auto-discovered real_emitter coverage inventories. "
            "Defaults to evaluation/temp and the temp directory adjacent to --results-root when applicable."
        ),
    )
    parser.add_argument(
        "--coverage-max-profile-cache-records",
        type=int,
        default=500000,
        help="Maximum source records to load from real_emitter profile-cache JSON files.",
    )
    parser.add_argument(
        "--coverage-profile-cache-include-samples",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If a real_emitter profile cache lacks emission_reports, also use sample_reports. "
            "Disabled by default because sample reports are not a complete inventory."
        ),
    )
    parser.add_argument(
        "--coverage-profile-cache-synthesize-locations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If a real_emitter profile cache has sensor_locations, synthesize one source record per sensor/location. "
            "This provides a deployed-source inventory even when per-report emission records are absent."
        ),
    )
    parser.add_argument(
        "--coverage-labelling-json",
        nargs="*",
        default=None,
        help=(
            "Optional precomputed evaluation/labelling relevance JSON files, such as "
            "evaluation/merged_incidents/all_merged_by_id_relevant.json. These files should contain "
            "temporal_relevant and spatial_relevant source-family lists per incident."
        ),
    )
    parser.add_argument(
        "--coverage-labelling-as-primary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --coverage-labelling-json matches a GT incident, use its labelling-derived coverage score "
            "as the primary coverage_score. If disabled, the labelling score is appended as labelling_coverage_score only."
        ),
    )
    parser.add_argument(
        "--coverage-use-result-support",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use support points from discovered low_level_results.json files as a fallback coverage source inventory. "
            "This is less pure than passing explicit source JSONL files but works with existing result folders."
        ),
    )
    parser.add_argument(
        "--coverage-temporal-pad-hours",
        type=float,
        default=24.0,
        help="Temporal padding around GT intervals when selecting source records for coverage scoring.",
    )
    parser.add_argument(
        "--coverage-max-result-support-records",
        type=int,
        default=200000,
        help="Maximum support/source records to extract from result folders for coverage scoring.",
    )
    parser.add_argument(
        "--coverage-allow-type-default",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When no source inventory is available, use a conservative incident-type default coverage prior.",
    )
    parser.add_argument(
        "--max-loc-error-km",
        type=float,
        default=20.0,
        help="A prediction can match a GT if source/coverage centroid is within this distance.",
    )
    parser.add_argument(
        "--min-spatial-iou",
        type=float,
        default=0.01,
        help="A prediction can match a GT if source/coverage IoU reaches this value.",
    )
    parser.add_argument(
        "--min-coverage-normalized",
        type=float,
        default=0.0,
        help="Drop predicted points below this coverage_normalized before building hulls.",
    )
    parser.add_argument(
        "--default-single-step-minutes",
        type=float,
        default=5.0,
        help="If a GT location appears in only one step, give it this non-zero duration for temporal IoU.",
    )
    parser.add_argument(
        "--include-runtime",
        action="store_true",
        help="Also summarize timing.json runtime fields outside scalability modes.",
    )
    parser.add_argument(
        "--skip-missing-gt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip runs whose simulator GT folder cannot be found.",
    )
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show an outer tqdm progress bar over result run directories when tqdm is installed.",
    )
    parser.add_argument(
        "--real-match-inner-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For --mode low_real, show an inner progress bar over predictions while building "
            "candidate GT matches for each method. This makes slow method-level matching visible."
        ),
    )
    parser.add_argument(
        "--leakage-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Repair cumulative low_level_results.json files by dropping predictions whose "
            "prediction_id/candidate_ids first appeared in an earlier result folder for the same method/experiment."
        ),
    )
    parser.add_argument(
        "--leakage-filter-methods",
        nargs="*",
        default=["incidentlens", "generic_propagation"],
        help=(
            "Methods to apply the cumulative-result leakage filter to. Default: incidentlens generic_propagation. "
            "Use an empty list after the flag to apply to no methods, or --no-leakage-filter to disable entirely."
        ),
    )
    parser.add_argument(
        "--write-denoised-low-level",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write low_level_results_denoised.json next to low_level_results.json for selected methods. "
            "The original low_level_results.json is never modified."
        ),
    )
    parser.add_argument(
        "--denoised-methods",
        nargs="*",
        default=["incidentlens"],
        help=(
            "Methods for which to write low_level_results_denoised.json. Default: incidentlens. "
            "For example, use --denoised-methods incidentlens generic_propagation to write both."
        ),
    )

    parser.add_argument(
        "--composition-experiment-name",
        default="synth_comp",
        help="Positive high-level-composition experiment directory used by --mode synth_composition.",
    )
    parser.add_argument(
        "--composition-experiments",
        nargs="*",
        default=None,
        help=(
            "Optional exact experiment names for --mode synth_composition. "
            "Default: --low-experiment-name and --composition-experiment-name."
        ),
    )
    parser.add_argument(
        "--composition-batch-small-area-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For --mode synth_composition, evaluate only runs whose matched *_plan.json "
            "is under a path component named batch_small_area. This excludes synth_low/"
            "batch_incident_runs and is useful when reporting only high-level composition positives."
        ),
    )
    parser.add_argument(
        "--composition-civil-protest-terrorist-only",
        "--composition-civil-protest-terrorist",
        "--composition-protest-terrorist-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="composition_civil_protest_terrorist_only",
        help=(
            "For --mode synth_composition, evaluate only runs whose *_plan.json child "
            "incident types include civil protest/large civil protest or terrorist "
            "incident/terrorist attack. This is usually used with "
            "--composition-batch-small-area-only. The shorter aliases "
            "--composition-civil-protest-terrorist and "
            "--composition-protest-terrorist-only are also accepted."
        ),
    )
    parser.add_argument(
        "--composition-gt-incident-types",
        nargs="*",
        default=None,
        help=(
            "Optional explicit GT child incident-type filter for --mode synth_composition. "
            "Values are canonicalized with the usual incident-type aliases. Example: "
            "--composition-gt-incident-types 'large civil protest' 'terrorist attack'."
        ),
    )
    parser.add_argument(
        "--composition-baselines",
        nargs="*",
        default=["simple_proximity_only_linker", "same_type_temporal_overlap_linker"],
        help=(
            "Composition baselines to run in memory from low_level_results_denoised.json. "
            "Known: simple_proximity_only_linker same_type_temporal_overlap_linker."
        ),
    )
    parser.add_argument(
        "--evaluate-incidentlens-composition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include IncidentLens high_level_results.json in --mode synth_composition.",
    )

    parser.add_argument(
        "--incidentlens-composition-llm-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For --mode synth_composition, optionally evaluate IncidentLens high_level_results.json using only "
            "LLM-backed composition decisions and ignore heuristic fallback composites. "
            "By default heuristic fallback composites are allowed, matching the deployed IncidentLens output."
        ),
    )
    parser.add_argument(
        "--composition-proximity-max-distance-km",
        type=float,
        default=20.0,
        help="Distance threshold for simple_proximity_only_linker.",
    )
    parser.add_argument(
        "--composition-same-type-temporal-gap-hours",
        type=float,
        default=0.0,
        help="Allowed active-interval gap for same_type_temporal_overlap_linker; 0 means strict overlap.",
    )
    parser.add_argument(
        "--composition-same-type-max-distance-km",
        type=float,
        default=None,
        help="Optional distance cap for same_type_temporal_overlap_linker. Default: no spatial cap.",
    )
    parser.add_argument(
        "--composition-min-children",
        type=int,
        default=2,
        help="Minimum linked low-level incidents required for a baseline high-level composite.",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if getattr(args, "synth_low_all_baselines", False):
        args.methods = list(SYNTH_LOW_DEFAULT_BASELINES)
    if getattr(args, "synth_low_background_baselines_only", False):
        args.methods = list(SYNTH_LOW_BACKGROUND_BASELINES)
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
