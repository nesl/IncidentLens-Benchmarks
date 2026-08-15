#!/usr/bin/env python3
"""Coverage/observability scoring helpers for IncidentLens evaluation.

This module estimates how observable a ground-truth incident was under an
available sensor/source stream.  It is intentionally independent from the
matching logic in evaluate_results.py: the evaluator can call this module to
write per-GT coverage rows and to add coverage-weighted recall/F-scores.

The most accurate use is to pass normalized REPORT JSONL files produced by
real_emitter.py/synthetic_emitter.py via evaluate_results.py --coverage-source-jsonl.
When those are not available, the evaluator can fall back to extracting source
hints from low_level_results.json support points.  That fallback is useful for
paper diagnostics, but it is not a pure deployment-coverage estimate because it
uses emitted prediction support rather than a complete source inventory.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

try:
    from shapely.geometry import Point
    from shapely.geometry.base import BaseGeometry
except Exception:  # pragma: no cover
    Point = None  # type: ignore
    BaseGeometry = Any  # type: ignore


# ---------------------------------------------------------------------------
# Default domain model
# ---------------------------------------------------------------------------

MODALITY_SUITABILITY: Dict[str, Dict[str, float]] = {
    "wildfire": {
        "camera": 0.80,
        "air_quality": 0.90,
        "weather": 0.40,
        "traffic": 0.20,
        "text_alert": 0.70,
        "social": 0.60,
        "official_alert": 0.85,
    },
    "urban fire": {
        "camera": 0.80,
        "air_quality": 0.60,
        "weather": 0.15,
        "traffic": 0.25,
        "text_alert": 0.70,
        "social": 0.55,
        "official_alert": 0.85,
    },
    "fire": {
        "camera": 0.80,
        "air_quality": 0.60,
        "weather": 0.15,
        "traffic": 0.25,
        "text_alert": 0.70,
        "social": 0.55,
        "official_alert": 0.85,
    },
    "road closure": {
        "traffic": 0.90,
        "camera": 0.65,
        "text_alert": 0.70,
        "social": 0.55,
        "official_alert": 0.85,
        "air_quality": 0.00,
        "weather": 0.10,
    },
    "road vehicle accident": {
        "traffic": 0.85,
        "camera": 0.75,
        "text_alert": 0.60,
        "social": 0.55,
        "official_alert": 0.70,
        "weather": 0.10,
    },
    "large civil protest": {
        "camera": 0.70,
        "traffic": 0.35,
        "text_alert": 0.70,
        "social": 0.80,
        "official_alert": 0.65,
    },
    "civil protest": {
        "camera": 0.70,
        "traffic": 0.35,
        "text_alert": 0.70,
        "social": 0.80,
        "official_alert": 0.65,
    },
    "demonstration": {
        "camera": 0.70,
        "traffic": 0.35,
        "text_alert": 0.70,
        "social": 0.80,
        "official_alert": 0.65,
    },
    "terrorist attack": {
        "camera": 0.75,
        "traffic": 0.35,
        "air_quality": 0.20,
        "text_alert": 0.75,
        "social": 0.70,
        "official_alert": 0.85,
    },
    "terrorist incident": {
        "camera": 0.75,
        "traffic": 0.35,
        "air_quality": 0.20,
        "text_alert": 0.75,
        "social": 0.70,
        "official_alert": 0.85,
    },
    "flood": {
        "camera": 0.70,
        "weather": 0.70,
        "traffic": 0.45,
        "text_alert": 0.65,
        "social": 0.50,
        "official_alert": 0.80,
    },
    "hazardous material release": {
        "air_quality": 0.80,
        "camera": 0.45,
        "weather": 0.30,
        "text_alert": 0.70,
        "official_alert": 0.85,
        "social": 0.45,
    },
}

# Spatial kernel radii in km.  These are intentionally coarse and can be
# replaced by a JSON config later if you want a stricter deployment model.
EFFECTIVE_RADIUS_KM: Dict[Tuple[str, str], float] = {
    ("wildfire", "camera"): 5.0,
    ("wildfire", "air_quality"): 15.0,
    ("wildfire", "weather"): 25.0,
    ("wildfire", "traffic"): 5.0,
    ("wildfire", "text_alert"): 30.0,
    ("wildfire", "social"): 15.0,
    ("wildfire", "official_alert"): 50.0,
    ("urban fire", "camera"): 2.0,
    ("urban fire", "air_quality"): 5.0,
    ("urban fire", "traffic"): 2.0,
    ("urban fire", "text_alert"): 10.0,
    ("road closure", "traffic"): 2.0,
    ("road closure", "camera"): 1.0,
    ("road closure", "text_alert"): 10.0,
    ("road vehicle accident", "traffic"): 2.0,
    ("road vehicle accident", "camera"): 1.5,
    ("large civil protest", "social"): 8.0,
    ("large civil protest", "camera"): 3.0,
    ("large civil protest", "traffic"): 5.0,
    ("terrorist attack", "camera"): 4.0,
    ("terrorist attack", "text_alert"): 20.0,
    ("terrorist attack", "official_alert"): 50.0,
    ("flood", "weather"): 30.0,
    ("flood", "camera"): 3.0,
    ("flood", "traffic"): 3.0,
    ("hazardous material release", "air_quality"): 8.0,
    ("hazardous material release", "weather"): 20.0,
}

DEFAULT_RADIUS_KM_BY_MODALITY = {
    "camera": 2.0,
    "air_quality": 10.0,
    "weather": 25.0,
    "traffic": 2.0,
    "text_alert": 15.0,
    "social": 10.0,
    "official_alert": 50.0,
    "unknown": 5.0,
}

EVIDENCE_STRENGTH: Dict[str, Dict[str, float]] = {
    "wildfire": {
        "flame_visible": 0.95,
        "smoke_visible": 0.75,
        "pm25_spike": 0.65,
        "wind_consistent": 0.35,
        "text_fire_report": 0.70,
        "official_fire_alert": 0.90,
        "traffic_slowdown": 0.15,
        "weather_hot_dry": 0.10,
        "camera_observation": 0.35,
        "social_report": 0.45,
    },
    "urban fire": {
        "flame_visible": 0.95,
        "smoke_visible": 0.65,
        "pm25_spike": 0.35,
        "text_fire_report": 0.70,
        "official_fire_alert": 0.90,
        "camera_observation": 0.45,
        "social_report": 0.45,
    },
    "fire": {
        "flame_visible": 0.95,
        "smoke_visible": 0.65,
        "pm25_spike": 0.35,
        "text_fire_report": 0.70,
        "official_fire_alert": 0.90,
        "camera_observation": 0.45,
        "social_report": 0.45,
    },
    "road closure": {
        "official_road_closure": 0.90,
        "traffic_speed_drop": 0.75,
        "camera_congestion": 0.55,
        "social_report": 0.45,
        "road_blockage_visible": 0.75,
    },
    "road vehicle accident": {
        "vehicle_crash_visible": 0.85,
        "traffic_speed_drop": 0.65,
        "visible_damage": 0.65,
        "emergency_vehicle_visible": 0.45,
        "social_report": 0.45,
    },
    "large civil protest": {
        "crowd_visible": 0.80,
        "social_report": 0.75,
        "text_protest_report": 0.75,
        "traffic_slowdown": 0.25,
        "official_alert": 0.60,
    },
    "civil protest": {
        "crowd_visible": 0.80,
        "social_report": 0.75,
        "text_protest_report": 0.75,
        "traffic_slowdown": 0.25,
        "official_alert": 0.60,
    },
    "terrorist attack": {
        "explosion_damage": 0.75,
        "emergency_vehicle_visible": 0.50,
        "official_alert": 0.90,
        "social_report": 0.55,
        "visible_damage": 0.45,
        "camera_observation": 0.35,
    },
    "terrorist incident": {
        "explosion_damage": 0.75,
        "emergency_vehicle_visible": 0.50,
        "official_alert": 0.90,
        "social_report": 0.55,
        "visible_damage": 0.45,
        "camera_observation": 0.35,
    },
    "flood": {
        "standing_water_visible": 0.80,
        "heavy_precipitation": 0.55,
        "traffic_speed_drop": 0.25,
        "official_alert": 0.75,
        "social_report": 0.45,
    },
    "hazardous material release": {
        "air_quality_anomaly": 0.75,
        "official_alert": 0.85,
        "visible_damage": 0.35,
        "emergency_vehicle_visible": 0.35,
        "social_report": 0.35,
    },
}

SOURCE_TYPE_TO_MODALITY = {
    "cctv": "camera",
    "camera": "camera",
    "image": "camera",
    "traffic_camera": "camera",
    "caltrans": "camera",
    "alertcalifornia": "camera",
    "alert_california": "camera",
    "air": "air_quality",
    "air_data": "air_quality",
    "air_quality": "air_quality",
    "purpleair": "air_quality",
    "pm25": "air_quality",
    "pm2_5": "air_quality",
    "weather": "weather",
    "weather_data": "weather",
    "traffic": "traffic",
    "traffic_data": "traffic",
    "pems": "traffic",
    "pem": "traffic",
    "pem_data": "traffic",
    "twitter": "social",
    "x": "social",
    "x_data": "social",
    "social": "social",
    "citizen": "text_alert",
    "citizen_data": "text_alert",
    "citizenapp": "text_alert",
    "official_alert": "official_alert",
    "official_alerts": "official_alert",
    "public_alert": "official_alert",
    "news": "text_alert",
    "article": "text_alert",
}

EFFECT_TO_EVIDENCE = {
    "possible_smoke_or_haze": "smoke_visible",
    "visible_flames_or_fire": "flame_visible",
    "standing_or_flowing_water": "standing_water_visible",
    "stopped_or_slow_traffic": "traffic_speed_drop",
    "high_vehicle_density": "traffic_speed_drop",
    "road_blockage_or_barricade": "road_blockage_visible",
    "high_pedestrian_density": "crowd_visible",
    "visible_damage_or_debris": "visible_damage",
    "emergency_vehicle_presence": "emergency_vehicle_visible",
}

MODALITY_TO_WEAK_EVIDENCE = {
    "camera": "camera_observation",
    "air_quality": "pm25_spike",
    "weather": "weather_hot_dry",
    "traffic": "traffic_speed_drop",
    "social": "social_report",
    "text_alert": "social_report",
    "official_alert": "official_alert",
}

TYPE_DEFAULT_COVERAGE = {
    "wildfire": 0.45,
    "urban fire": 0.40,
    "fire": 0.40,
    "road closure": 0.45,
    "road vehicle accident": 0.45,
    "large civil protest": 0.35,
    "civil protest": 0.35,
    "demonstration": 0.35,
    "terrorist attack": 0.30,
    "terrorist incident": 0.30,
    "flood": 0.40,
    "hazardous material release": 0.35,
}


@dataclass
class SourceRecord:
    source_id: str
    modality: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timestamp: Optional[datetime] = None
    reliability: float = 1.0
    evidence_types: Tuple[str, ...] = ()
    source_type: str = ""
    source_file: str = ""


def norm_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_source_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def canonical_incident_type(value: Any) -> str:
    raw = norm_label(value)
    aliases = {
        "civil protest": "large civil protest",
        "demonstration": "large civil protest",
        "protest": "large civil protest",
        "terrorist incident": "terrorist attack",
        "terrorism": "terrorist attack",
        "urban_fire": "urban fire",
        "hazmat": "hazardous material release",
        "hazardous material": "hazardous material release",
    }
    return aliases.get(raw, raw or "unknown")


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
        if re.fullmatch(r"\d{8}", text):
            try:
                dt = datetime.strptime(text, "%Y%m%d")
            except ValueError:
                return None
        else:
            try:
                dt = datetime.fromisoformat(text)
            except ValueError:
                return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def noisy_or(values: Iterable[float], *, cap: Optional[float] = 0.95) -> float:
    miss = 1.0
    any_value = False
    for value in values:
        q = max(0.0, min(1.0, float(value)))
        miss *= 1.0 - q
        any_value = True
    score = 1.0 - miss if any_value else 0.0
    if cap is not None:
        score = min(float(cap), score)
    return max(0.0, min(1.0, score))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def geometry_centroid_latlon(geom: Any) -> Optional[Tuple[float, float]]:
    if geom is None:
        return None
    try:
        if geom.is_empty:
            return None
        c = geom.centroid
        return float(c.y), float(c.x)
    except Exception:
        return None


def incident_centroid(gt: Any) -> Optional[Tuple[float, float]]:
    geom = getattr(gt, "geometry", None)
    c = geometry_centroid_latlon(geom)
    if c is not None:
        return c
    lat = safe_float(getattr(gt, "representative_latitude", None), None)
    lon = safe_float(getattr(gt, "representative_longitude", None), None)
    if lat is not None and lon is not None:
        return lat, lon
    return None


def infer_modality(obj: Mapping[str, Any]) -> str:
    metadata = obj.get("metadata") if isinstance(obj.get("metadata"), Mapping) else {}
    data = obj.get("data") if isinstance(obj.get("data"), Mapping) else {}
    candidates = [
        obj.get("modality"), metadata.get("modality"), obj.get("sensor_type"),
        obj.get("source"), metadata.get("source"), metadata.get("data_source"),
        data.get("sensor_type"), data.get("source"),
    ]
    for value in candidates:
        norm = normalize_source_name(value)
        if norm in SOURCE_TYPE_TO_MODALITY:
            return SOURCE_TYPE_TO_MODALITY[norm]
    keys = {normalize_source_name(k) for k in data.keys()} if isinstance(data, Mapping) else set()
    if data.get("image_filepath"):
        return "camera"
    if {"pm25", "pm2_5", "aqi", "pm10"} & keys:
        return "air_quality"
    if {"avg_speed", "data_avg_speed", "occupancy", "data_avg_occupancy"} & keys:
        return "traffic"
    if {"temperature", "temperature_f", "wind_speed", "wind_speed_mph", "humidity", "precipitation"} & keys:
        return "weather"
    if {"text", "message", "body", "content", "description", "summary", "title"} & keys:
        return "text_alert"
    return "unknown"


def extract_location(obj: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    loc = obj.get("location") if isinstance(obj.get("location"), Mapping) else {}
    data = obj.get("data") if isinstance(obj.get("data"), Mapping) else {}
    for container in (obj, loc, data):
        lat = safe_float(container.get("latitude", container.get("lat")), None) if isinstance(container, Mapping) else None
        lon = safe_float(container.get("longitude", container.get("lon")), None) if isinstance(container, Mapping) else None
        if lat is not None and lon is not None:
            return lat, lon
    return None, None


def evidence_from_obj(obj: Mapping[str, Any], modality: str) -> List[str]:
    evidence: List[str] = []
    for key in ("observed_effects", "candidate_effects"):
        values = obj.get(key)
        if isinstance(values, list):
            for item in values:
                name = item.get("name") if isinstance(item, Mapping) else item
                ev = EFFECT_TO_EVIDENCE.get(str(name))
                if ev:
                    evidence.append(ev)
    # Some low-level prediction rows include rationale text but not raw effects.
    text_parts = []
    for key in ("rationale", "reasoning", "explanation", "incident_type", "pred_type"):
        if isinstance(obj.get(key), str):
            text_parts.append(str(obj.get(key)).lower())
    text = "\n".join(text_parts)
    if "smoke" in text or "haze" in text:
        evidence.append("smoke_visible")
    if "flame" in text or "fire" in text:
        evidence.append("text_fire_report" if modality in {"social", "text_alert", "official_alert"} else "flame_visible")
    if "road closure" in text or "road closed" in text:
        evidence.append("official_road_closure")
    if "traffic" in text or "congestion" in text:
        evidence.append("traffic_speed_drop")
    if "protest" in text or "demonstration" in text or "crowd" in text:
        evidence.append("text_protest_report" if modality in {"social", "text_alert", "official_alert"} else "crowd_visible")
    weak = MODALITY_TO_WEAK_EVIDENCE.get(modality)
    if weak:
        evidence.append(weak)
    return list(dict.fromkeys(evidence))


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8", errors="replace") as infile:
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


def source_record_from_obj(obj: Mapping[str, Any], *, source_file: str = "") -> Optional[SourceRecord]:
    modality = infer_modality(obj)
    lat, lon = extract_location(obj)
    metadata = obj.get("metadata") if isinstance(obj.get("metadata"), Mapping) else {}
    source_type = str(obj.get("sensor_type") or obj.get("source") or metadata.get("source") or metadata.get("data_source") or modality)
    source_id = str(obj.get("sensor_id") or obj.get("source_id") or obj.get("report_id") or f"{source_type}:{lat}:{lon}")
    timestamp = parse_dt(obj.get("report_date") or obj.get("timestamp") or obj.get("time"))
    evidence = tuple(evidence_from_obj(obj, modality))
    if lat is None and lon is None and modality == "unknown" and not evidence:
        return None
    return SourceRecord(
        source_id=source_id,
        modality=modality,
        latitude=lat,
        longitude=lon,
        timestamp=timestamp,
        reliability=max(0.0, min(1.0, safe_float(obj.get("reliability"), 1.0) or 1.0)),
        evidence_types=evidence,
        source_type=source_type,
        source_file=source_file,
    )


def load_source_records_from_jsonl(paths: Sequence[str | Path]) -> List[SourceRecord]:
    out: List[SourceRecord] = []
    for path in paths or []:
        p = Path(path)
        if p.is_dir():
            candidates = sorted(p.rglob("*.jsonl"))
        else:
            candidates = [p]
        for candidate in candidates:
            for obj in iter_jsonl(candidate):
                rec = source_record_from_obj(obj, source_file=str(candidate))
                if rec is not None:
                    out.append(rec)
    return dedupe_source_records(out)



def load_source_records_from_real_emitter_profile_cache(
    paths: Sequence[str | Path],
    *,
    max_records: int = 500000,
    include_sample_reports_when_no_emission: bool = False,
    synthesize_sensor_location_records: bool = True,
) -> List[SourceRecord]:
    """Load normalized source records from real_emitter profile-cache JSON files.

    real_emitter.py writes source/date profile caches under a directory such as
    evaluation/temp/_real_emitter_profile_cache/.  Those JSON files are not
    JSONL, but they contain profile.emission_reports when the emitter ran with
    socket emission or ordered-report output enabled.  This loader lets
    evaluate_results.py reuse those cached normalized reports as a source
    inventory for coverage scoring even when real_emitter_ordered_reports.jsonl
    was not written.

    If a cache file lacks emission_reports, this can optionally synthesize one
    source record per sensor location from profile.sensor_locations.  That is
    weaker than per-report observations, but still useful as a deployed-source
    coverage inventory.
    """
    out: List[SourceRecord] = []

    def iter_candidate_files(root: Path) -> Iterator[Path]:
        if root.is_dir():
            for candidate in sorted(root.rglob("*.json")):
                yield candidate
        else:
            yield root

    for raw_path in paths or []:
        root = Path(raw_path)
        if not root.exists():
            continue
        for candidate in iter_candidate_files(root):
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, Mapping):
                continue
            profile = payload.get("profile")
            if not isinstance(profile, Mapping):
                continue

            source = str(profile.get("source") or payload.get("source") or "")
            date_str = str(profile.get("date_str") or payload.get("date_str") or "")

            reports = profile.get("emission_reports")
            report_source = "emission_reports"
            if (not isinstance(reports, list) or not reports) and include_sample_reports_when_no_emission:
                reports = profile.get("sample_reports")
                report_source = "sample_reports"

            if isinstance(reports, list) and reports:
                for obj in reports:
                    if not isinstance(obj, Mapping):
                        continue
                    rec = source_record_from_obj(obj, source_file=str(candidate))
                    if rec is not None:
                        out.append(rec)
                    if len(out) >= max_records:
                        return dedupe_source_records(out)

            if synthesize_sensor_location_records:
                sensor_locations = profile.get("sensor_locations")
                sensor_names = profile.get("sensor_names") if isinstance(profile.get("sensor_names"), Mapping) else {}
                if isinstance(sensor_locations, Mapping):
                    for sensor_id, loc in sensor_locations.items():
                        if not isinstance(loc, Mapping):
                            continue
                        lat = loc.get("latitude", loc.get("lat"))
                        lon = loc.get("longitude", loc.get("lon"))
                        obj = {
                            "report_id": f"profile_cache_{source}_{date_str}_{sensor_id}",
                            "report_date": date_str,
                            "sensor_id": str(sensor_id),
                            "sensor_name": str(sensor_names.get(sensor_id, sensor_id)) if isinstance(sensor_names, Mapping) else str(sensor_id),
                            "sensor_type": source,
                            "location": {"latitude": lat, "longitude": lon},
                            "metadata": {
                                "date_str": date_str,
                                "source": source,
                                "coverage_source": "real_emitter_profile_cache_sensor_location",
                                "profile_cache_report_source": report_source,
                            },
                        }
                        rec = source_record_from_obj(obj, source_file=str(candidate))
                        if rec is not None:
                            out.append(rec)
                        if len(out) >= max_records:
                            return dedupe_source_records(out)

    return dedupe_source_records(out)

def _support_point_to_obj(point: Mapping[str, Any], parent: Mapping[str, Any]) -> Dict[str, Any]:
    obj = dict(point)
    # Copy weak parent hints into the support point.
    for key in ("incident_type", "rationale", "reasoning", "explanation", "sensor_type", "source", "modality"):
        if key not in obj and key in parent:
            obj[key] = parent[key]
    return obj


def load_source_records_from_result_dirs(
    result_dirs: Sequence[Tuple[str, str, str, Path]],
    *,
    prefer_denoised: bool = True,
    max_records: int = 200000,
) -> List[SourceRecord]:
    """Extract approximate source/evidence hints from prediction support.

    This is a fallback when a complete normalized REPORT/source inventory is not
    available. It should be interpreted as observed-support coverage rather than
    pure deployed-source coverage.
    """
    records: List[SourceRecord] = []
    for method, experiment, run_name, result_dir in result_dirs:
        result_dir = Path(result_dir)
        path = result_dir / ("low_level_results_denoised.json" if prefer_denoised and (result_dir / "low_level_results_denoised.json").exists() else "low_level_results.json")
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = payload.get("low_level_incidents") if isinstance(payload, Mapping) else None
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            # Affected-region support points are the most common baseline footprint.
            affected = row.get("affected_region") if isinstance(row.get("affected_region"), Mapping) else {}
            support_points = []
            if isinstance(affected.get("support_points"), list):
                support_points.extend(affected.get("support_points"))
            for key in ("hypothesis_source_points", "propagated_coverage_points"):
                if isinstance(row.get(key), list):
                    support_points.extend(row.get(key))
            if support_points:
                for point in support_points[:1000]:
                    if not isinstance(point, Mapping):
                        continue
                    obj = _support_point_to_obj(point, row)
                    rec = source_record_from_obj(obj, source_file=str(path))
                    if rec is not None:
                        records.append(rec)
            else:
                rec = source_record_from_obj(row, source_file=str(path))
                if rec is not None:
                    records.append(rec)
            if len(records) >= max_records:
                return dedupe_source_records(records)
    return dedupe_source_records(records)


def dedupe_source_records(records: Sequence[SourceRecord]) -> List[SourceRecord]:
    seen = set()
    out: List[SourceRecord] = []
    for rec in records:
        key = (
            rec.source_id,
            rec.modality,
            round(rec.latitude, 4) if rec.latitude is not None else None,
            round(rec.longitude, 4) if rec.longitude is not None else None,
            rec.timestamp.isoformat() if rec.timestamp is not None else None,
            rec.evidence_types,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def source_relevant_to_incident(
    rec: SourceRecord,
    gt: Any,
    *,
    temporal_pad_hours: float = 24.0,
) -> bool:
    # Temporal gate, if both sides have time.
    start = getattr(gt, "start_time", None)
    end = getattr(gt, "end_time", None)
    if isinstance(start, datetime) or isinstance(end, datetime):
        start = start or end
        end = end or start
        if rec.timestamp is not None and start is not None and end is not None:
            pad = timedelta(hours=float(temporal_pad_hours))
            if rec.timestamp < start - pad or rec.timestamp > end + pad:
                return False
    # Spatial gate is soft inside source contribution; do not drop sources solely
    # for distance unless the distance is wildly outside the modality radius.
    return True


def distance_kernel(distance_km: Optional[float], radius_km: float) -> float:
    if distance_km is None:
        return 0.5
    radius_km = max(1e-6, float(radius_km))
    return math.exp(-float(distance_km) ** 2 / (2.0 * radius_km ** 2))


def source_distance_km(rec: SourceRecord, gt: Any) -> Optional[float]:
    if rec.latitude is None or rec.longitude is None:
        return None
    center = incident_centroid(gt)
    if center is None:
        return None
    return haversine_km(rec.latitude, rec.longitude, center[0], center[1])


def modality_coverage(incident_type: str, modalities: Iterable[str]) -> float:
    k = canonical_incident_type(incident_type)
    suitability = MODALITY_SUITABILITY.get(k, {})
    return noisy_or((suitability.get(m, 0.0) for m in set(modalities)), cap=0.95)


def source_coverage(incident_type: str, gt: Any, sources: Sequence[SourceRecord]) -> float:
    k = canonical_incident_type(incident_type)
    suitability = MODALITY_SUITABILITY.get(k, {})
    contributions: List[float] = []
    for rec in sources:
        q = suitability.get(rec.modality, 0.0)
        if q <= 0:
            continue
        radius = EFFECTIVE_RADIUS_KM.get((k, rec.modality), DEFAULT_RADIUS_KM_BY_MODALITY.get(rec.modality, 5.0))
        spatial = distance_kernel(source_distance_km(rec, gt), radius)
        contributions.append(q * rec.reliability * spatial)
    return noisy_or(contributions, cap=0.98)


def discriminative_strength(target_type: str, evidence_type: str) -> float:
    k = canonical_incident_type(target_type)
    target = EVIDENCE_STRENGTH.get(k, {}).get(evidence_type, 0.0)
    if target <= 0:
        return 0.0
    total = sum(strengths.get(evidence_type, 0.0) for strengths in EVIDENCE_STRENGTH.values())
    if total <= 0:
        return target
    return target * (target / total)


def inference_coverage(incident_type: str, evidence_types: Iterable[str]) -> float:
    strengths = [discriminative_strength(incident_type, ev) for ev in set(evidence_types)]
    return noisy_or(strengths, cap=0.95)


def coverage_key_for_gt(gt: Any) -> str:
    incident_id = str(getattr(gt, "incident_id", "") or "")
    location = str(getattr(gt, "location_name", "") or "")
    if location and location != incident_id:
        return f"{incident_id}::{location}"
    return incident_id or location or "unknown_gt"


def compute_coverage_for_gt_items(
    gt_items: Sequence[Any],
    source_records: Sequence[SourceRecord],
    *,
    weights: Tuple[float, float, float] = (0.25, 0.35, 0.40),
    temporal_pad_hours: float = 24.0,
    allow_type_default_when_no_sources: bool = True,
    source_label: str = "source_records",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    wm, ws, wi = weights
    total_w = max(1e-9, wm + ws + wi)
    wm, ws, wi = wm / total_w, ws / total_w, wi / total_w
    for gt in gt_items:
        k = canonical_incident_type(getattr(gt, "incident_type", "unknown"))
        relevant = [rec for rec in source_records if source_relevant_to_incident(rec, gt, temporal_pad_hours=temporal_pad_hours)]
        modalities = sorted({rec.modality for rec in relevant if rec.modality})
        evidence = sorted({ev for rec in relevant for ev in rec.evidence_types})
        c_mod = modality_coverage(k, modalities)
        c_src = source_coverage(k, gt, relevant)
        c_inf = inference_coverage(k, evidence)
        fallback_used = False
        if not relevant and allow_type_default_when_no_sources:
            fallback = TYPE_DEFAULT_COVERAGE.get(k, 0.25)
            # Keep components explicit; the final score uses the type prior only
            # because no sensor/source inventory was available.
            score = fallback
            fallback_used = True
        else:
            score = wm * c_mod + ws * c_src + wi * c_inf
        rows.append({
            "coverage_key": coverage_key_for_gt(gt),
            "gt_id": str(getattr(gt, "incident_id", "") or ""),
            "gt_final_name": getattr(gt, "final_name", None),
            "gt_type": k,
            "gt_location": str(getattr(gt, "location_name", "") or ""),
            "coverage_score": round(max(0.0, min(1.0, score)), 6),
            "modality_coverage": round(c_mod, 6),
            "source_coverage": round(c_src, 6),
            "inference_coverage": round(c_inf, 6),
            "num_relevant_sources": len(relevant),
            "available_modalities": ";".join(modalities),
            "evidence_types": ";".join(evidence),
            "fallback_type_default_used": fallback_used,
            "coverage_source": source_label,
            "coverage_formula": f"{wm:.2f}*modality + {ws:.2f}*source + {wi:.2f}*inference",
        })
    return rows


def coverage_by_gt_id(coverage_rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in coverage_rows:
        score = safe_float(row.get("coverage_score"), None)
        if score is None:
            continue
        for key in (row.get("coverage_key"), row.get("gt_id")):
            if key:
                out[str(key)] = float(score)
    return out


def fbeta_weighted(tp: float, fp: float, fn: float, beta: float = 2.0) -> Optional[float]:
    beta2 = beta * beta
    denom = (1.0 + beta2) * tp + beta2 * fn + fp
    if denom <= 0:
        return None
    return (1.0 + beta2) * tp / denom


def compute_coverage_weighted_metrics(
    gt_items: Sequence[Any],
    matches: Sequence[Mapping[str, Any]],
    coverage_scores: Mapping[str, float],
    *,
    fp_count: int = 0,
) -> Dict[str, Any]:
    gt_weights: Dict[str, float] = {}
    for gt in gt_items:
        key = coverage_key_for_gt(gt)
        score = coverage_scores.get(key, coverage_scores.get(str(getattr(gt, "incident_id", "")), 1.0))
        gt_weights[key] = max(0.0, min(1.0, float(score)))
    matched_keys = set()
    for m in matches:
        gt_id = str(m.get("gt_id") or "")
        gt_location = str(m.get("gt_location") or "")
        if f"{gt_id}::{gt_location}" in gt_weights:
            matched_keys.add(f"{gt_id}::{gt_location}")
        elif gt_id in gt_weights:
            matched_keys.add(gt_id)
        else:
            # Fallback when match rows only carry plain GT IDs.
            for key in gt_weights:
                if key.startswith(gt_id + "::"):
                    matched_keys.add(key)
                    break
    total = sum(gt_weights.values())
    matched = sum(gt_weights[k] for k in matched_keys if k in gt_weights)
    missed = max(0.0, total - matched)
    mean_cov = statistics.mean(gt_weights.values()) if gt_weights else None
    matched_values = [gt_weights[k] for k in matched_keys if k in gt_weights]
    missed_values = [v for k, v in gt_weights.items() if k not in matched_keys]
    return {
        "coverage_num_gt": len(gt_weights),
        "coverage_sum_gt": round(total, 6),
        "coverage_sum_matched_gt": round(matched, 6),
        "coverage_sum_missed_gt": round(missed, 6),
        "coverage_mean_gt": round(mean_cov, 6) if mean_cov is not None else None,
        "coverage_mean_matched_gt": round(statistics.mean(matched_values), 6) if matched_values else None,
        "coverage_mean_missed_gt": round(statistics.mean(missed_values), 6) if missed_values else None,
        "coverage_weighted_recall": (matched / total) if total > 0 else None,
        "coverage_weighted_f1": fbeta_weighted(matched, float(fp_count), missed, beta=1.0),
        "coverage_weighted_f2": fbeta_weighted(matched, float(fp_count), missed, beta=2.0),
        "coverage_weighting_note": "TP/FN are weighted by GT coverage_score; FP remains an unweighted prediction count.",
    }


def summarize_coverage_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    vals = [safe_float(r.get("coverage_score"), None) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"num_gt": len(rows), "mean_coverage_score": None, "median_coverage_score": None}
    return {
        "num_gt": len(rows),
        "mean_coverage_score": statistics.mean(vals),
        "median_coverage_score": statistics.median(vals),
        "min_coverage_score": min(vals),
        "max_coverage_score": max(vals),
        "num_fallback_type_default": sum(1 for r in rows if r.get("fallback_type_default_used")),
    }
