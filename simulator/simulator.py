from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Modality specific information
from simulator.tools.camera_query import get_closest_cameras
from simulator.tools.simulation_tools import modify_image_cloud, ToolStateAgent
from simulator.tools.time_series import generate_time_series
from simulator.tools.textual import generate_text_data
from simulator.tools.utility import append_observation_log, ensure_dir
from utilities.util import get_config
from evaluation.geo_manager import GeoManager

import argparse
import asyncio
import csv
import json
import math
import os
import random
import re
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from shapely.geometry import Point, Polygon, mapping
except Exception:  # Shapely should exist in the target project, but keep import robust.
    Point = None
    Polygon = None


SAVE_FOLDER_DEFAULT = "simulator/generated"
MIN_INCIDENT_REGION_SIDE_KM_DEFAULT = 2.0


# Keep this list focused on incidents that plausibly cover a multi-kilometer
# footprint and are observable by at least one supported physical sensor:
# air quality, images/cameras, or traffic speed/occupancy.
ALL_INCIDENTS = [
    "wildfire",
    "urban fire",
    "hazardous material release",
    "flood",
    "severe storm damage",
    "earthquake damage",
    "large civil protest",
    "terrorist attack",
]

# Outside sensors are negative-control sensors: they use a modality relevant to
# the incident type, but are placed outside the incident polygon and should show
# background/normal readings rather than the incident abnormality. Weather is
# intentionally excluded because it is contextual rather than an incident sensor.
#
# If ALL_INCIDENTS changes, update this dictionary for the new incident names.
# The helper below filters this mapping to ALL_INCIDENTS, so removed incident
# names cannot leak into run metadata or outside-sensor selection. Missing new
# incident names fall back to DEFAULT_RELEVANT_OUTSIDE_SENSOR_TYPES.
INCIDENT_RELEVANT_OUTSIDE_SENSOR_TYPES = {
    "wildfire": ["air", "alertcalifornia", "cctv"],
    "urban fire": ["air", "cctv", "alertcalifornia", "california_traffic"],
    "hazardous material release": ["air", "cctv", "california_traffic"],
    "flood": ["cctv", "california_traffic"],
    "severe storm damage": ["cctv", "california_traffic"],
    "earthquake damage": ["cctv", "california_traffic"],
    "large civil protest": ["cctv", "california_traffic"],
    "terrorist attack": ["cctv", "california_traffic", "air"],
}

DEFAULT_RELEVANT_OUTSIDE_SENSOR_TYPES = ["cctv"]

TREND_COLUMNS = ["data_avg_occupancy", "data_avg_speed"]

DEFAULT_OUTSIDE_SENSOR_TYPE_POOL = [
    "air",
    "california_traffic",
    "cctv",
    "alertcalifornia",
]



def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def _now_timer() -> float:
    return time.perf_counter()


def _print_timing(label: str, start: float, enabled: bool = True) -> None:
    if enabled:
        print(f"[TIMER] {label}: {time.perf_counter() - start:.2f}s")


def _default_simulate_images() -> bool:
    return _env_bool("SIMULATOR_SIMULATE_IMAGES", True)


def _default_fast_mode() -> bool:
    return _env_bool("SIMULATOR_FAST_MODE", True)


def _default_use_llm_camera_selection() -> bool:
    return _env_bool("SIMULATOR_USE_LLM_CAMERA_SELECTION", False)



# ---------------------------------------------------------------------------
# Generic parsing / geometry helpers
# ---------------------------------------------------------------------------

def _extract_between(text: str, start_token: str, end_token: str, default: str = "") -> str:
    if start_token not in text or end_token not in text:
        return default
    return text.split(start_token, 1)[1].split(end_token, 1)[0].strip()


def _safe_slug(value: str, fallback: str = "item") -> str:
    value = re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(value)).strip("_")
    return value[:80] or fallback


def _source_has(source: str, *needles: str) -> bool:
    source = source.lower().strip()
    return any(needle in source for needle in needles)


def _is_image_source(source: str) -> bool:
    """Return True for sources that should trigger image/camera simulation."""
    return _source_has(source, "alertcalifornia", "cctv", "camera", "image")


def _default_force_image_source() -> bool:
    """Default to forcing at least one image source when image generation is enabled.

    The planning LLM can legitimately omit camera sources in favor of air/traffic/text.
    That is bad for experiments where simulate_images=True is expected to produce image
    observations, so the simulator injects a relevant camera source unless disabled.
    """
    return _env_bool("SIMULATOR_FORCE_IMAGE_SOURCE", True)


def _preferred_image_source_for_incident(incident_type: Optional[str]) -> str:
    """Choose a camera source that is relevant for this incident type."""
    relevant = _relevant_outside_sensor_types_for_incident(incident_type)
    # ALERTCalifornia is especially useful for wildland fire/smoke; CCTV is the
    # safer general default for urban/flood/protest/damage events.
    if _normalize_incident_type(incident_type or "", default=ALL_INCIDENTS[0]) == "wildfire" and "alertcalifornia" in relevant:
        return "alertcalifornia"
    if "cctv" in relevant:
        return "cctv"
    if "alertcalifornia" in relevant:
        return "alertcalifornia"
    return "cctv"


def _ensure_image_source_available(
    sources: List[str],
    incident_type: Optional[str],
    *,
    simulate_images: bool,
    max_image_edits_per_step: int,
    force_image_source: bool = True,
) -> List[str]:
    """Make image generation robust to LLM source choices and source caps.

    If image generation is enabled, we keep image/camera sources near the front so
    max_sources_per_step does not accidentally trim them away. If the LLM omitted
    images entirely, we inject one incident-relevant camera source.
    """
    sources = [str(s).strip().lower() for s in sources if str(s).strip()]
    if not simulate_images or max_image_edits_per_step <= 0 or not force_image_source:
        return sources

    image_sources = [s for s in sources if _is_image_source(s)]
    non_image_sources = [s for s in sources if not _is_image_source(s)]

    if not image_sources:
        image_sources = [_preferred_image_source_for_incident(incident_type)]
        print(f"Injected image source {image_sources[0]!r} because simulate_images=True and the planning LLM did not include a camera source.")

    # Put image sources first so max_sources_per_step cannot silently remove all
    # image generation. Preserve relative order otherwise and deduplicate.
    ordered = image_sources + non_image_sources
    deduped: List[str] = []
    seen = set()
    for source in ordered:
        if source not in seen:
            deduped.append(source)
            seen.add(source)
    return deduped


def _is_traffic_source(source: str) -> bool:
    source = source.lower().strip().replace("-", "_").replace(" ", "_")
    return any(x in source for x in ["traffic", "california_traffic", "caltrans", "pem", "pems"])


def _source_family(source: str) -> str:
    """Collapse source aliases into a stable physical sensor family."""
    source_l = str(source or "").lower().strip().replace("-", "_").replace(" ", "_")
    if _is_traffic_source(source_l):
        return "california_traffic"
    if any(x in source_l for x in ["air", "pm25", "pm2_5", "aqi"]):
        return "air"
    if "weather" in source_l:
        return "weather"
    if "alertcalifornia" in source_l:
        return "alertcalifornia"
    if "cctv" in source_l:
        return "cctv"
    if any(x in source_l for x in ["camera", "image"]):
        return "camera"
    return source_l


def _canonical_sensor_types(sensor_types: Optional[Iterable[str]]) -> List[str]:
    if not sensor_types:
        return []
    out = []
    for sensor_type in sensor_types:
        family = _source_family(sensor_type)
        if family and family not in out:
            out.append(family)
    return out


def _incident_relevant_outside_sensor_types_map() -> Dict[str, List[str]]:
    """Return a cleaned incident->outside-sensor map aligned with ALL_INCIDENTS.

    This prevents stale keys from earlier incident lists from being used when
    ALL_INCIDENTS is edited. It also canonicalizes sensor aliases and removes
    weather from all negative-control outside sensor choices.
    """
    fallback = [x for x in _canonical_sensor_types(DEFAULT_RELEVANT_OUTSIDE_SENSOR_TYPES) if x != "weather"]
    if not fallback:
        fallback = ["cctv"]

    cleaned: Dict[str, List[str]] = {}
    for incident_type in ALL_INCIDENTS:
        sensors = INCIDENT_RELEVANT_OUTSIDE_SENSOR_TYPES.get(incident_type, fallback)
        sensors = [x for x in _canonical_sensor_types(sensors) if x != "weather"]
        cleaned[incident_type] = sensors or fallback
    return cleaned


def _relevant_outside_sensor_types_for_incident(incident_type: Optional[str]) -> List[str]:
    """Return relevant negative-control sensor families for one incident type."""
    relevant_by_incident = _incident_relevant_outside_sensor_types_map()
    normalized_incident = _normalize_incident_type(incident_type or "", default=ALL_INCIDENTS[0])
    return list(relevant_by_incident.get(normalized_incident, relevant_by_incident.get(ALL_INCIDENTS[0], ["cctv"])))


def _choose_outside_sensor_types(
    sensor_type_pool: Optional[Iterable[str]] = None,
    sensor_type_count: Optional[int] = None,
    incident_type: Optional[str] = None,
) -> List[str]:
    """Choose relevant negative-control physical sensor families for a run.

    Outside sensors should be the same kind of modality that could detect the
    incident if it were inside the incident region, but because these are outside
    the region they should produce normal/background observations.  Weather is
    intentionally filtered out: it can be useful context, but it is not a direct
    abnormality sensor for these incidents.
    """
    pool = [x for x in _canonical_sensor_types(sensor_type_pool or DEFAULT_OUTSIDE_SENSOR_TYPE_POOL) if x != "weather"]
    if incident_type:
        relevant = _relevant_outside_sensor_types_for_incident(incident_type)
        # Respect an explicit pool/env override while still keeping only relevant modalities.
        pool = [x for x in pool if x in relevant] or relevant
    if not pool:
        return []
    if sensor_type_count is None or sensor_type_count <= 0:
        count = random.randint(1, len(pool))
    else:
        count = max(1, min(int(sensor_type_count), len(pool)))
    return sorted(random.sample(pool, count))

def _source_uses_outside_sensors(source: str, outside_sensor_types: Iterable[str]) -> bool:
    family = _source_family(source)
    outside = set(_canonical_sensor_types(outside_sensor_types))
    if family in outside:
        return True
    # Treat generic camera/image as compatible with concrete camera sources.
    if family in {"camera", "cctv", "alertcalifornia"} and "camera" in outside:
        return True
    return False


def _camera_source_matches_request(camera_origin: str, requested_source: str) -> bool:
    """Return True when a camera candidate belongs to the requested camera family.

    get_closest_cameras() returns both CCTV and ALERTCalifornia candidates.  That is
    useful for broad camera requests, but when the simulator source for a step is
    explicitly "cctv" or "alertcalifornia", using both families causes duplicate
    image-edit calls across source names.
    """
    requested_family = _source_family(requested_source)
    origin_family = _source_family(camera_origin)
    if requested_family in {"cctv", "alertcalifornia"}:
        return origin_family == requested_family
    if requested_family in {"camera", "image"}:
        return origin_family in {"cctv", "alertcalifornia", "camera"}
    return True


def _location_key(location_record: Dict[str, Any]) -> str:
    """Stable-ish key for reusing fixed sensors when a location recurs across steps."""
    name = _safe_slug(location_record.get("name", "location"), fallback="location")
    lat = round(float(location_record.get("representative_lat", location_record.get("lat", 0.0))), 5)
    lon = round(float(location_record.get("representative_lon", location_record.get("lon", 0.0))), 5)
    return f"{name}_{lat}_{lon}"


def _normalize_incident_type(value: str, default: str = "wildfire") -> str:
    """Normalize only against ALL_INCIDENTS; no legacy aliases are used."""
    value_l = str(value or "").lower().strip()
    default_l = str(default or "wildfire").lower().strip()
    default_normalized = default_l if default_l in ALL_INCIDENTS else "wildfire"

    for incident_type in ALL_INCIDENTS:
        if value_l == incident_type:
            return incident_type
    for incident_type in ALL_INCIDENTS:
        if incident_type in value_l or value_l in incident_type:
            return incident_type
    return default_normalized

def _lat_lon_from_row(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    lat_keys = ["latitude", "lat", "sensor_latitude", "sensor_lat", "y"]
    lon_keys = ["longitude", "lon", "lng", "long", "sensor_longitude", "sensor_lon", "x"]
    lowered = {str(k).lower().strip(): v for k, v in row.items()}

    lat = None
    lon = None
    for key in lat_keys:
        if key in lowered:
            try:
                lat = float(lowered[key])
            except Exception:
                pass
            break
    for key in lon_keys:
        if key in lowered:
            try:
                lon = float(lowered[key])
            except Exception:
                pass
            break
    return lat, lon


def _polygon_contains_latlon(polygon: Any, lat: Optional[float], lon: Optional[float]) -> Optional[bool]:
    if polygon is None or lat is None or lon is None or Point is None:
        return None
    try:
        # Shapely uses x, y ordering, i.e. lon, lat.
        point = Point(lon, lat)
        return bool(polygon.contains(point) or polygon.touches(point))
    except Exception:
        return None


def _destination_latlon_km(lat: float, lon: float, bearing_degrees: float, distance_km: float) -> Tuple[float, float]:
    """Move from lat/lon by a distance and bearing using a spherical Earth approximation."""
    radius_km = 6371.0
    angular_distance = distance_km / radius_km
    bearing = math.radians(bearing_degrees)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
    )
    return float(math.degrees(lat2)), float(math.degrees(lon2))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two latitude/longitude points in kilometers."""
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def _polygon_bbox_dimensions_km(polygon: Any) -> Tuple[Optional[float], Optional[float]]:
    """Approximate polygon bounding-box width/height in km using lat/lon edges."""
    if polygon is None:
        return None, None
    try:
        minx, miny, maxx, maxy = polygon.bounds  # x=lon, y=lat
        mid_lat = (miny + maxy) / 2.0
        mid_lon = (minx + maxx) / 2.0
        width_km = _haversine_km(mid_lat, minx, mid_lat, maxx)
        height_km = _haversine_km(miny, mid_lon, maxy, mid_lon)
        return float(width_km), float(height_km)
    except Exception:
        return None, None


def _km_to_lat_degrees(km: float) -> float:
    return float(km) / 110.574


def _km_to_lon_degrees(km: float, latitude: float) -> float:
    # Avoid division by zero near the poles.  The simulator is LA-centric, but
    # this guard keeps the helper well-defined for unexpected locations.
    denom = 111.320 * max(0.01, abs(math.cos(math.radians(latitude))))
    return float(km) / denom


def _make_minimum_region_box(
    center_lat: float,
    center_lon: float,
    width_km: float,
    height_km: float,
) -> Any:
    """Create a Shapely lon/lat rectangle centered on the incident representative point."""
    if Polygon is None:
        return None
    half_lat = _km_to_lat_degrees(height_km / 2.0)
    half_lon = _km_to_lon_degrees(width_km / 2.0, center_lat)
    west = center_lon - half_lon
    east = center_lon + half_lon
    south = center_lat - half_lat
    north = center_lat + half_lat
    return Polygon([(west, south), (east, south), (east, north), (west, north), (west, south)])


def _ensure_minimum_region_size(
    region: Any,
    fallback_lat: float,
    fallback_lon: float,
    min_side_km: float = MIN_INCIDENT_REGION_SIDE_KM_DEFAULT,
) -> Tuple[Any, Dict[str, Any]]:
    """Ensure the incident polygon has at least a min_side_km x min_side_km footprint.

    GeoManager can return a tiny polygon for exact addresses/intersections.  That
    is useful for geocoding, but too small for this simulator because physical
    sensors need to be sampled from a meaningful incident region.  When a region
    is too small, we keep the representative point and replace the placement
    geometry with a synthetic box whose sides are at least min_side_km.
    """
    min_side_km = max(0.0, float(min_side_km or 0.0))
    original_width_km, original_height_km = _polygon_bbox_dimensions_km(region)
    rep_lat, rep_lon = _representative_latlon(region, fallback_lat, fallback_lon)

    width_ok = original_width_km is not None and original_width_km >= min_side_km
    height_ok = original_height_km is not None and original_height_km >= min_side_km
    if min_side_km <= 0 or (width_ok and height_ok):
        final_width_km, final_height_km = original_width_km, original_height_km
        return region, {
            "minimum_side_km": min_side_km,
            "original_bbox_width_km": None if original_width_km is None else round(original_width_km, 3),
            "original_bbox_height_km": None if original_height_km is None else round(original_height_km, 3),
            "final_bbox_width_km": None if final_width_km is None else round(final_width_km, 3),
            "final_bbox_height_km": None if final_height_km is None else round(final_height_km, 3),
            "expanded": False,
            "status": "passed",
        }

    target_width_km = max(min_side_km, original_width_km or 0.0)
    target_height_km = max(min_side_km, original_height_km or 0.0)
    expanded_region = _make_minimum_region_box(rep_lat, rep_lon, target_width_km, target_height_km)
    if expanded_region is None:
        # If Shapely is unexpectedly unavailable, fall back to the original
        # geometry but make the failure explicit in ground truth.
        return region, {
            "minimum_side_km": min_side_km,
            "original_bbox_width_km": None if original_width_km is None else round(original_width_km, 3),
            "original_bbox_height_km": None if original_height_km is None else round(original_height_km, 3),
            "final_bbox_width_km": None if original_width_km is None else round(original_width_km, 3),
            "final_bbox_height_km": None if original_height_km is None else round(original_height_km, 3),
            "expanded": False,
            "status": "failed_no_polygon_constructor",
        }

    final_width_km, final_height_km = _polygon_bbox_dimensions_km(expanded_region)
    return expanded_region, {
        "minimum_side_km": min_side_km,
        "original_bbox_width_km": None if original_width_km is None else round(original_width_km, 3),
        "original_bbox_height_km": None if original_height_km is None else round(original_height_km, 3),
        "final_bbox_width_km": None if final_width_km is None else round(final_width_km, 3),
        "final_bbox_height_km": None if final_height_km is None else round(final_height_km, 3),
        "expanded": True,
        "status": "expanded_to_minimum_side",
        "repair_method": "synthetic_box_centered_on_representative_point",
    }


# Default minimum spacing for high-level sub-incident seed locations.  These are
# intentionally conservative simulation priors: localized events can be 5 km
# apart, while hazards that naturally affect larger areas are forced farther
# apart so the generated run does not collapse into one neighborhood.
SUB_INCIDENT_MIN_DISTANCE_KM_BY_TYPE = {
    "wildfire": 15.0,
    "urban fire": 8.0,
    "hazardous material release": 10.0,
    "flood": 12.0,
    "severe storm damage": 12.0,
    "earthquake damage": 15.0,
    "large civil protest": 8.0,
    "terrorist attack": 10.0,
}


FALLBACK_DISTANT_SEED_LOCATIONS = [
    "Santa Monica Pier, Santa Monica, CA",
    "Griffith Observatory, Los Angeles, CA",
    "Long Beach Convention Center, Long Beach, CA",
    "Pasadena City Hall, Pasadena, CA",
    "Hollywood Bowl, Los Angeles, CA",
    "SoFi Stadium, Inglewood, CA",
    "Burbank City Hall, Burbank, CA",
    "Malibu Creek State Park, Calabasas, CA",
    "Angeles National Forest, Los Angeles County, CA",
    "Pomona Fairplex, Pomona, CA",
    "San Fernando Recreation Park, San Fernando, CA",
    "Torrance City Hall, Torrance, CA",
    "Whittier Narrows Recreation Area, South El Monte, CA",
    "Castaic Lake State Recreation Area, Castaic, CA",
    "Kenneth Hahn State Recreation Area, Los Angeles, CA",
]


def _default_sub_incident_min_distance_km(*incident_types: str) -> float:
    """Return a spacing threshold for a high-level run based on incident types."""
    thresholds = [5.0]
    for incident_type in incident_types:
        normalized = _normalize_incident_type(str(incident_type or ""), default=ALL_INCIDENTS[0])
        thresholds.append(SUB_INCIDENT_MIN_DISTANCE_KM_BY_TYPE.get(normalized, 5.0))
    return max(thresholds)


def _minimum_distance_to_records(location_record: Dict[str, Any], previous_records: List[Dict[str, Any]]) -> Optional[float]:
    """Distance from one resolved location to the nearest previous resolved sub-incident."""
    if not previous_records:
        return None
    lat = float(location_record["representative_lat"])
    lon = float(location_record["representative_lon"])
    distances = []
    for prev in previous_records:
        distances.append(
            _haversine_km(
                lat,
                lon,
                float(prev["representative_lat"]),
                float(prev["representative_lon"]),
            )
        )
    return min(distances) if distances else None


def _record_is_far_enough(
    location_record: Dict[str, Any],
    previous_records: List[Dict[str, Any]],
    min_distance_km: float,
) -> Tuple[bool, Optional[float]]:
    nearest = _minimum_distance_to_records(location_record, previous_records)
    if nearest is None:
        return True, None
    return nearest >= min_distance_km, nearest


def _resolve_seed_location_safely(
    geomanager: GeoManager,
    seed_location: str,
) -> Optional[Dict[str, Any]]:
    try:
        return _get_location_record(geomanager, seed_location)
    except Exception as e:
        print(f"Could not resolve seed location {seed_location!r}: {e}")
        return None


def _request_replacement_seed_location(
    llm: ChatOpenAI,
    incident_request: str,
    item: Dict[str, Any],
    accepted_items: List[Dict[str, Any]],
    min_distance_km: float,
) -> Optional[str]:
    """Ask the LLM for one replacement seed location that is far from accepted seeds."""
    accepted_summary = [
        {
            "incident_type": x.get("incident_type"),
            "seed_location": x.get("seed_location"),
            "sub_incident_id": x.get("sub_incident_id"),
        }
        for x in accepted_items
    ]
    token_s = "<LOCATION_JSON>"
    token_e = "</LOCATION_JSON>"
    prompt = PromptTemplate(
        template=(
            "You are repairing a Los Angeles incident simulation plan. The current sub-incident "
            "seed location is too close to another sub-incident or could not be resolved. Choose "
            "one replacement seed_location that GeoManager.get_geo_region() can resolve. It must be "
            "at least {min_distance_km} km away from every accepted seed location. Prefer one broad "
            "Los Angeles County region, neighborhood, park, roadway corridor, industrial district, "
            "campus, wildland area, stadium area, or recreation area rather than a tiny point address. "
            "Keep the replacement plausible for this incident type. Return only JSON between "
            "{token_s} and {token_e}, with exactly one key: seed_location.\n\n"
            "High-level incident request: {incident_request}\n"
            "Sub-incident to repair: {item}\n"
            "Accepted sub-incidents: {accepted_summary}"
        ),
        input_variables=[
            "incident_request",
            "item",
            "accepted_summary",
            "min_distance_km",
            "token_s",
            "token_e",
        ],
    )
    try:
        response = llm.invoke(
            prompt.format(
                incident_request=incident_request,
                item=json.dumps(item, ensure_ascii=False),
                accepted_summary=json.dumps(accepted_summary, ensure_ascii=False),
                min_distance_km=round(float(min_distance_km), 2),
                token_s=token_s,
                token_e=token_e,
            )
        )
        raw = _extract_between(response.content, token_s, token_e, default="{}")
        parsed = json.loads(raw)
        candidate = parsed.get("seed_location") if isinstance(parsed, dict) else None
        return str(candidate).strip() if candidate else None
    except Exception as e:
        print(f"Could not parse replacement seed-location response: {e}")
        return None


def _fallback_replacement_seed_location(
    geomanager: GeoManager,
    previous_records: List[Dict[str, Any]],
    min_distance_km: float,
    used_locations: Iterable[str],
) -> Optional[Tuple[str, Dict[str, Any], Optional[float]]]:
    """Find a deterministic fallback place-name that is far from previous sub-incidents."""
    used = {str(x).strip().lower() for x in used_locations if str(x).strip()}
    candidates = list(FALLBACK_DISTANT_SEED_LOCATIONS)
    random.shuffle(candidates)
    for candidate in candidates:
        if candidate.strip().lower() in used:
            continue
        record = _resolve_seed_location_safely(geomanager, candidate)
        if not record:
            continue
        ok, nearest = _record_is_far_enough(record, previous_records, min_distance_km)
        if ok:
            return candidate, record, nearest
    return None


def _ensure_sub_incident_seed_locations_far_apart(
    llm: ChatOpenAI,
    geomanager: GeoManager,
    plan: List[Dict[str, Any]],
    incident_request: str,
    min_distance_km: float,
    repair_attempts: int = 4,
) -> List[Dict[str, Any]]:
    """Validate/repair high-level sub-incident seed locations before simulation.

    The LLM is prompted to choose separated locations, but this function enforces
    the spacing after GeoManager resolution.  A repaired item keeps the same
    incident type/description and only changes seed_location.
    """
    if len(plan) <= 1:
        return plan

    accepted_items: List[Dict[str, Any]] = []
    accepted_records: List[Dict[str, Any]] = []
    used_locations = [str(x.get("seed_location", "")) for x in plan]
    repaired_plan: List[Dict[str, Any]] = []

    for idx, raw_item in enumerate(plan):
        item = dict(raw_item)
        seed_location = str(item.get("seed_location") or "Los Angeles, CA").strip()
        record = _resolve_seed_location_safely(geomanager, seed_location)
        ok = False
        nearest = None
        repaired = False

        if record is not None:
            ok, nearest = _record_is_far_enough(record, accepted_records, min_distance_km)

        attempt = 0
        while not ok and attempt < max(0, int(repair_attempts)):
            attempt += 1
            replacement = _request_replacement_seed_location(
                llm,
                incident_request,
                item,
                accepted_items,
                min_distance_km,
            )
            if not replacement or replacement.strip().lower() in {x.lower() for x in used_locations}:
                continue
            replacement_record = _resolve_seed_location_safely(geomanager, replacement)
            if replacement_record is None:
                continue
            replacement_ok, replacement_nearest = _record_is_far_enough(
                replacement_record,
                accepted_records,
                min_distance_km,
            )
            if replacement_ok:
                seed_location = replacement
                record = replacement_record
                nearest = replacement_nearest
                ok = True
                repaired = True
                used_locations.append(replacement)
                item["seed_location"] = replacement
                break

        if not ok:
            fallback = _fallback_replacement_seed_location(
                geomanager,
                accepted_records,
                min_distance_km,
                used_locations,
            )
            if fallback is not None:
                seed_location, record, nearest = fallback
                ok = True
                repaired = True
                used_locations.append(seed_location)
                item["seed_location"] = seed_location

        # If the first location could not be resolved, fall back to a known anchor.
        if record is None:
            fallback = _fallback_replacement_seed_location(
                geomanager,
                accepted_records,
                min_distance_km,
                used_locations,
            )
            if fallback is not None:
                seed_location, record, nearest = fallback
                ok = True
                repaired = True
                used_locations.append(seed_location)
                item["seed_location"] = seed_location

        item["seed_location_distance_check"] = {
            "min_required_km": round(float(min_distance_km), 3),
            "nearest_previous_seed_distance_km": None if nearest is None else round(float(nearest), 3),
            "status": "passed" if ok else "failed_unrepaired",
            "repaired": repaired,
        }

        if not ok:
            print(
                f"WARNING: sub-incident {idx} seed location {seed_location!r} did not meet "
                f"the {min_distance_km} km spacing constraint after repair attempts."
            )

        repaired_plan.append(item)
        accepted_items.append(item)
        if record is not None:
            accepted_records.append(record)

    return repaired_plan




def _largest_polygon(region: Any) -> Any:
    """Return a polygon-like object for boundary sampling, including MultiPolygons."""
    if region is None:
        return None
    geoms = getattr(region, "geoms", None)
    if geoms:
        try:
            return max(list(geoms), key=lambda g: getattr(g, "area", 0.0))
        except Exception:
            return list(geoms)[0]
    return region


def _sample_boundary_latlon(region: Any, fallback_lat: float, fallback_lon: float) -> Tuple[float, float]:
    """Sample a point on the outer boundary of a Polygon/MultiPolygon."""
    polygon = _largest_polygon(region)
    if polygon is None:
        return fallback_lat, fallback_lon
    try:
        boundary = polygon.exterior
        point = boundary.interpolate(random.random() * boundary.length)
        return float(point.y), float(point.x)
    except Exception:
        return fallback_lat, fallback_lon


def _sample_latlon_outside_region(
    region: Any,
    fallback_lat: float,
    fallback_lon: float,
    min_distance_km: float = 1.0,
    max_distance_km: float = 10.0,
    max_attempts: int = 100,
) -> Tuple[float, float, float]:
    """Sample a point outside a region, roughly min/max km beyond its boundary.

    The implementation samples a boundary point and walks outward away from the
    representative point.  It is approximate but works well for simulation
    metadata and avoids requiring projected coordinate systems.
    """
    min_distance_km = max(0.0, float(min_distance_km))
    max_distance_km = max(min_distance_km, float(max_distance_km))

    if region is None or Point is None:
        distance = random.uniform(min_distance_km, max_distance_km)
        bearing = random.uniform(0.0, 360.0)
        lat, lon = _destination_latlon_km(fallback_lat, fallback_lon, bearing, distance)
        return lat, lon, round(distance, 3)

    center_lat, center_lon = _representative_latlon(region, fallback_lat, fallback_lon)
    for _ in range(max_attempts):
        boundary_lat, boundary_lon = _sample_boundary_latlon(region, fallback_lat, fallback_lon)
        bearing = _bearing_degrees(center_lat, center_lon, boundary_lat, boundary_lon)
        distance = random.uniform(min_distance_km, max_distance_km)
        lat, lon = _destination_latlon_km(boundary_lat, boundary_lon, bearing, distance)
        inside = _polygon_contains_latlon(region, lat, lon)
        if inside is False:
            return float(lat), float(lon), round(distance, 3)

    # Fallback: move from representative point in a random direction until outside.
    for _ in range(max_attempts):
        distance = random.uniform(min_distance_km, max_distance_km)
        bearing = random.uniform(0.0, 360.0)
        lat, lon = _destination_latlon_km(center_lat, center_lon, bearing, distance)
        inside = _polygon_contains_latlon(region, lat, lon)
        if inside is False:
            return float(lat), float(lon), round(distance, 3)

    # Last resort: return a point at max distance even if containment is unknown.
    lat, lon = _destination_latlon_km(center_lat, center_lon, random.uniform(0.0, 360.0), max_distance_km)
    return float(lat), float(lon), round(max_distance_km, 3)


def _sample_latlon_in_region(region: Any, fallback_lat: float, fallback_lon: float, max_attempts: int = 1000) -> Tuple[float, float]:
    """Sample a latitude/longitude inside a Shapely Polygon or MultiPolygon.

    This is used for *simulated sensor placement*.  The source image may come
    from a real camera elsewhere, but the observation metadata is spoofed to be
    a physically plausible in-region sensor.
    """
    if region is None or Point is None:
        return fallback_lat, fallback_lon

    try:
        minx, miny, maxx, maxy = region.bounds  # x=lon, y=lat
        for _ in range(max_attempts):
            lon = random.uniform(minx, maxx)
            lat = random.uniform(miny, maxy)
            pt = Point(lon, lat)
            if region.contains(pt) or region.touches(pt):
                return float(lat), float(lon)
    except Exception:
        pass

    return _representative_latlon(region, fallback_lat, fallback_lon)


def _sample_camera_latlon_in_region(location_record: Dict[str, Any], max_attempts: int = 1000) -> Tuple[float, float]:
    """Sample a plausible in-region camera position that is not exactly at the event center."""
    region = location_record.get("region")
    target_lat = float(location_record["representative_lat"])
    target_lon = float(location_record["representative_lon"])

    best = (target_lat, target_lon)
    best_dist = -1.0
    for _ in range(max_attempts):
        lat, lon = _sample_latlon_in_region(region, target_lat, target_lon, max_attempts=100)
        # Approximate squared distance in degrees.  We only need "away from center"
        # enough to define an inward-facing camera bearing.
        dist = (lat - target_lat) ** 2 + (lon - target_lon) ** 2
        if dist > best_dist:
            best = (lat, lon)
            best_dist = dist
        if dist > 1e-8:
            return lat, lon
    return best


def _bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing from point 1 to point 2, in degrees, 0=N, 90=E."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _bearing_to_compass(bearing: float) -> str:
    dirs = [
        "North", "North-Northeast", "Northeast", "East-Northeast",
        "East", "East-Southeast", "Southeast", "South-Southeast",
        "South", "South-Southwest", "Southwest", "West-Southwest",
        "West", "West-Northwest", "Northwest", "North-Northwest",
    ]
    idx = int((bearing + 11.25) // 22.5) % 16
    return dirs[idx]


def _make_camera_pose(
    sensor_lat: float,
    sensor_lon: float,
    target_lat: float,
    target_lon: float,
    sensor_region_role: str = "inside",
    outside_distance_km: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a camera pose at a fixed sensor point facing the event center."""
    bearing = _bearing_degrees(sensor_lat, sensor_lon, target_lat, target_lon)
    compass = _bearing_to_compass(bearing)
    role_text = "outside the incident region" if sensor_region_role == "outside" else "inside the incident region"
    return {
        "latitude": float(sensor_lat),
        "longitude": float(sensor_lon),
        "target_latitude": float(target_lat),
        "target_longitude": float(target_lon),
        "bearing_degrees": round(bearing, 2),
        "compass_direction": compass,
        "sensor_region_role": sensor_region_role,
        "outside_distance_km": outside_distance_km,
        "direction_label": f"Facing {compass} ({round(bearing, 1)}° inward toward the event center from {role_text})",
    }


def _make_inward_camera_pose(location_record: Dict[str, Any]) -> Dict[str, Any]:
    """Create a spoofed camera pose inside the incident region facing the event center."""
    sensor_lat, sensor_lon = _sample_camera_latlon_in_region(location_record)
    target_lat = float(location_record["representative_lat"])
    target_lon = float(location_record["representative_lon"])
    return _make_camera_pose(sensor_lat, sensor_lon, target_lat, target_lon, sensor_region_role="inside")


def _make_outside_camera_pose(
    location_record: Dict[str, Any],
    min_distance_km: float = 1.0,
    max_distance_km: float = 10.0,
) -> Dict[str, Any]:
    """Create a spoofed camera pose outside the incident region facing the event center."""
    target_lat = float(location_record["representative_lat"])
    target_lon = float(location_record["representative_lon"])
    sensor_lat, sensor_lon, distance = _sample_latlon_outside_region(
        location_record.get("region"),
        target_lat,
        target_lon,
        min_distance_km=min_distance_km,
        max_distance_km=max_distance_km,
    )
    return _make_camera_pose(
        sensor_lat,
        sensor_lon,
        target_lat,
        target_lon,
        sensor_region_role="outside",
        outside_distance_km=distance,
    )


def _camera_extra_metadata(camera_entry: Any) -> Dict[str, Any]:
    if isinstance(camera_entry, (tuple, list)) and len(camera_entry) >= 5 and isinstance(camera_entry[4], dict):
        return camera_entry[4]
    return {}


def _spoof_camera_entry_with_pose(
    real_camera_entry: Any,
    location_record: Dict[str, Any],
    pose: Dict[str, Any],
    sensor_id: Optional[str] = None,
) -> Tuple[Any, str, str, str, Dict[str, Any]]:
    """Reuse a real camera image file but spoof the simulated camera pose."""
    original_location = real_camera_entry[0]
    original_description = real_camera_entry[1]
    origin = real_camera_entry[2]
    filepath = real_camera_entry[3]
    spoof_location = (pose["latitude"], pose["longitude"])
    sensor_region_role = pose.get("sensor_region_role", "inside")
    metadata = {
        "sensor_id": sensor_id,
        "sensor_type": _source_family(origin),
        "sensor_region_role": sensor_region_role,
        "outside_distance_km": pose.get("outside_distance_km"),
        "spoofed_sensor_pose": True,
        "source_image_sensor_location": {
            "latitude": original_location[0],
            "longitude": original_location[1],
        },
        "source_image_camera_description": original_description,
        "inward_target_location": {
            "latitude": pose["target_latitude"],
            "longitude": pose["target_longitude"],
        },
        "direction_bearing_degrees": pose["bearing_degrees"],
        "direction_compass": pose["compass_direction"],
    }
    return (spoof_location, pose["direction_label"], origin, filepath, metadata)


def _spoof_camera_entry(real_camera_entry: Any, location_record: Dict[str, Any]) -> Tuple[Any, str, str, str, Dict[str, Any]]:
    """Reuse a real camera image file but spoof the simulated camera pose into the incident region."""
    return _spoof_camera_entry_with_pose(real_camera_entry, location_record, _make_inward_camera_pose(location_record))


def _make_time_series_sensor_locations(
    location_records: List[Dict[str, Any]],
    source: str,
    step: Optional[int] = None,
    sensors_per_region: int = 3,
    sensor_region_role: str = "inside",
    min_outside_distance_km: float = 1.0,
    max_outside_distance_km: float = 10.0,
) -> List[Dict[str, Any]]:
    """Create spoofed time-series sensor locations.

    Kept for backwards compatibility, but the simulator loop now normally calls
    _get_or_create_time_series_sensor_locations() so sensors stay fixed across
    steps.  If step is None, generated sensor IDs intentionally omit the step.
    """
    sensors = []
    safe_source = _safe_slug(source, fallback="sensor")
    for loc_i, loc in enumerate(location_records):
        n = max(1, sensors_per_region)
        loc_key = _location_key(loc)
        for sensor_i in range(n):
            if sensor_region_role == "outside":
                lat, lon, outside_distance = _sample_latlon_outside_region(
                    loc.get("region"),
                    float(loc["representative_lat"]),
                    float(loc["representative_lon"]),
                    min_distance_km=min_outside_distance_km,
                    max_distance_km=max_outside_distance_km,
                )
            else:
                lat, lon = _sample_latlon_in_region(
                    loc.get("region"),
                    float(loc["representative_lat"]),
                    float(loc["representative_lon"]),
                )
                outside_distance = None

            step_part = f"_{step}" if step is not None else ""
            sensors.append(
                {
                    "sensor_id": f"{safe_source}{step_part}_{loc_key}_{sensor_region_role}_{sensor_i}",
                    "sensor_type": _source_family(source),
                    "sensor_region_role": sensor_region_role,
                    "outside_distance_km": outside_distance,
                    "latitude": round(lat, 6),
                    "longitude": round(lon, 6),
                    "location_name": loc["name"],
                    "location_index": loc_i,
                    "location_key": loc_key,
                    "spoofed_sensor_pose": True,
                }
            )
    return sensors


def _get_or_create_time_series_sensor_locations(
    fixed_sensor_layout: Dict[str, Any],
    location_records: List[Dict[str, Any]],
    source: str,
    inside_sensors_per_region: int = 3,
    outside_sensors_enabled: bool = True,
    outside_sensors_per_region: int = 2,
    outside_sensor_min_distance_km: float = 1.0,
    outside_sensor_max_distance_km: float = 10.0,
    outside_sensor_types: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Return fixed sensor positions for this source/location, creating them lazily once."""
    store = fixed_sensor_layout.setdefault("time_series", {})
    source_family = _source_family(source)
    sensor_locations: List[Dict[str, Any]] = []

    for loc in location_records:
        loc_key = _location_key(loc)

        inside_key = f"{source_family}:{loc_key}:inside"
        if inside_key not in store:
            store[inside_key] = _make_time_series_sensor_locations(
                [loc],
                source_family,
                step=None,
                sensors_per_region=inside_sensors_per_region,
                sensor_region_role="inside",
            )
        sensor_locations.extend(store[inside_key])

        if outside_sensors_enabled and outside_sensors_per_region > 0 and _source_uses_outside_sensors(source_family, outside_sensor_types or []):
            outside_key = f"{source_family}:{loc_key}:outside"
            if outside_key not in store:
                store[outside_key] = _make_time_series_sensor_locations(
                    [loc],
                    source_family,
                    step=None,
                    sensors_per_region=outside_sensors_per_region,
                    sensor_region_role="outside",
                    min_outside_distance_km=outside_sensor_min_distance_km,
                    max_outside_distance_km=outside_sensor_max_distance_km,
                )
            sensor_locations.extend(store[outside_key])

    return sensor_locations


def _get_or_create_camera_data_by_location(
    fixed_sensor_layout: Dict[str, Any],
    location_records: List[Dict[str, Any]],
    source: str,
    outside_sensors_enabled: bool = True,
    outside_sensors_per_region: int = 1,
    outside_sensor_min_distance_km: float = 1.0,
    outside_sensor_max_distance_km: float = 10.0,
    outside_sensor_types: Optional[Iterable[str]] = None,
) -> Dict[int, List[Any]]:
    """Return fixed spoofed camera entries for each location.

    The underlying source image may still come from a nearby real camera, but the
    simulated sensor pose is sampled once and then reused across all simulation
    steps for the same incident/location.
    """
    store = fixed_sensor_layout.setdefault("camera", {})
    camera_data_by_location: Dict[int, List[Any]] = {}

    for loc_i, loc in enumerate(location_records):
        loc_key = _location_key(loc)
        cache_key = f"{_source_family(source)}:{loc_key}"
        if cache_key not in store:
            raw_camera_data, _ = get_closest_cameras(loc["representative_lat"], loc["representative_lon"], limit=8)
            all_data = [cam for cam in raw_camera_data if _camera_source_matches_request(cam[2], source)]
            if not all_data and raw_camera_data:
                print(
                    f"WARNING: no camera candidates matched requested source {source!r} for {loc['name']!r}; "
                    "falling back to any available camera family so image generation is not silently skipped."
                )
                all_data = raw_camera_data
            if not all_data:
                print(f"WARNING: no camera candidates found near {loc['name']!r} for source {source!r}.")
            entries = []
            for cam_i, cam in enumerate(all_data):
                pose = _make_inward_camera_pose(loc)
                sensor_id = f"camera_{loc_key}_inside_{cam_i}"
                entries.append(_spoof_camera_entry_with_pose(cam, loc, pose, sensor_id=sensor_id))

            if outside_sensors_enabled and outside_sensors_per_region > 0 and _source_uses_outside_sensors(source, outside_sensor_types or []):
                # Reuse the first few source images from the requested camera
                # family, but place their simulated camera sensors outside the
                # incident region.
                for outside_i, cam in enumerate(all_data[:outside_sensors_per_region]):
                    pose = _make_outside_camera_pose(
                        loc,
                        min_distance_km=outside_sensor_min_distance_km,
                        max_distance_km=outside_sensor_max_distance_km,
                    )
                    sensor_id = f"camera_{loc_key}_outside_{outside_i}"
                    entries.append(_spoof_camera_entry_with_pose(cam, loc, pose, sensor_id=sensor_id))

            store[cache_key] = entries

        camera_data_by_location[loc_i] = store[cache_key]

    return camera_data_by_location

def _format_sensor_locations_for_prompt(sensor_locations: List[Dict[str, Any]]) -> str:
    if not sensor_locations:
        return ""
    lines = ["Use exactly these fixed spoofed physical sensor locations; do not invent new latitude/longitude values:"]
    for s in sensor_locations:
        role = s.get("sensor_region_role", "inside")
        distance = s.get("outside_distance_km")
        if role == "outside":
            role_text = f"outside region for {s.get('location_name', 'incident')}"
            if distance is not None:
                role_text += f" (~{distance} km beyond the incident boundary)"
            role_text += "; negative-control sensor: do NOT show incident abnormalities, use normal/background readings"
        else:
            role_text = f"inside region for {s.get('location_name', 'incident')}; should reflect the incident when this source can sense it"
        lines.append(
            f"- {s['sensor_id']}: type={s.get('sensor_type', 'sensor')}, "
            f"latitude={s['latitude']}, longitude={s['longitude']}, {role_text}"
        )
    return " ".join(lines)


def _force_time_series_rows_to_sensor_locations(parsed_result: Dict[str, Any], sensor_locations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Guarantee parsed/generated CSV rows use the simulator-sampled sensor positions."""
    if not sensor_locations:
        return parsed_result

    headers = list(parsed_result.get("headers", []))
    lower_to_idx = {h.lower().strip(): i for i, h in enumerate(headers)}

    def ensure_header(name: str) -> int:
        key = name.lower()
        if key in lower_to_idx:
            return lower_to_idx[key]
        headers.append(name)
        idx = len(headers) - 1
        lower_to_idx[key] = idx
        for row in parsed_result.get("rows", []):
            row.append("")
        return idx

    sensor_idx = ensure_header("sensor_id")
    lat_idx = ensure_header("latitude")
    lon_idx = ensure_header("longitude")
    type_idx = ensure_header("sensor_type")
    role_idx = ensure_header("sensor_region_role")
    outside_distance_idx = ensure_header("outside_distance_km")

    rows = parsed_result.get("rows", [])
    if not rows:
        return parsed_result

    for row_idx, row in enumerate(rows):
        while len(row) < len(headers):
            row.append("")
        sensor = sensor_locations[row_idx % len(sensor_locations)]
        row[sensor_idx] = sensor["sensor_id"]
        row[lat_idx] = str(sensor["latitude"])
        row[lon_idx] = str(sensor["longitude"])
        row[type_idx] = str(sensor.get("sensor_type", _source_family(row[sensor_idx])))
        row[role_idx] = str(sensor.get("sensor_region_role", "inside"))
        row[outside_distance_idx] = "" if sensor.get("outside_distance_km") is None else str(sensor.get("outside_distance_km"))

    dict_rows = []
    for row in rows:
        dict_rows.append({headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))})

    parsed_result["headers"] = headers
    parsed_result["rows"] = rows
    parsed_result["dict_rows"] = dict_rows

    filepath = parsed_result.get("filepath")
    if filepath:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

    return parsed_result


def _apply_background_values_to_outside_time_series(parsed_result: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Make outside-region time-series rows behave like negative controls.

    The LLM/fast generator receives instructions, but this deterministic pass
    enforces the invariant in the saved CSV and observation log: sensors outside
    the incident region use incident-relevant modalities but do not report the
    incident abnormality.
    """
    headers = list(parsed_result.get("headers", []))
    rows = parsed_result.get("rows", [])
    if not headers or not rows:
        return parsed_result

    lower_to_idx = {h.lower().strip(): i for i, h in enumerate(headers)}

    def set_value(row: List[Any], column_names: Iterable[str], value: Any) -> None:
        for column_name in column_names:
            idx = lower_to_idx.get(column_name.lower().strip())
            if idx is not None:
                while len(row) <= idx:
                    row.append("")
                row[idx] = str(value)
                return

    role_idx = lower_to_idx.get("sensor_region_role")
    source_family = _source_family(source)
    for row in rows:
        if role_idx is None or role_idx >= len(row):
            continue
        if str(row[role_idx]).lower().strip() != "outside":
            continue

        if source_family == "air":
            set_value(row, ["pm25", "pm2.5"], round(random.uniform(3.0, 12.0), 1))
            set_value(row, ["pm10"], round(random.uniform(8.0, 25.0), 1))
            set_value(row, ["aqi"], random.randint(15, 55))
            set_value(row, ["description", "status"], "background air quality; no incident abnormality detected")
        elif source_family == "california_traffic":
            set_value(row, ["data_avg_occupancy", "occupancy"], round(random.uniform(0.04, 0.22), 3))
            set_value(row, ["data_avg_speed", "speed", "avg_speed"], round(random.uniform(35.0, 68.0), 1))
            set_value(row, ["description", "status"], "background traffic conditions; no incident abnormality detected")
        else:
            set_value(row, ["measurement"], "background")
            set_value(row, ["value"], "normal")
            set_value(row, ["description", "status"], "outside incident region; no incident abnormality detected")

    dict_rows = []
    for row in rows:
        while len(row) < len(headers):
            row.append("")
        dict_rows.append({headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))})

    parsed_result["headers"] = headers
    parsed_result["rows"] = rows
    parsed_result["dict_rows"] = dict_rows

    filepath = parsed_result.get("filepath")
    if filepath:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

    return parsed_result


def _representative_latlon(polygon: Any, fallback_lat: float, fallback_lon: float) -> Tuple[float, float]:
    if polygon is None:
        return fallback_lat, fallback_lon
    try:
        p = polygon.representative_point()
        return float(p.y), float(p.x)
    except Exception:
        return fallback_lat, fallback_lon


def _polygon_to_prompt(polygon: Any, fallback_lat: float, fallback_lon: float, max_vertices: int = 12) -> str:
    """Return a compact human-readable polygon description for LLM prompts."""
    if polygon is None:
        return f"No polygon was available; use a point near latitude={fallback_lat}, longitude={fallback_lon}."

    try:
        polygon_for_summary = _largest_polygon(polygon) or polygon
        minx, miny, maxx, maxy = polygon.bounds
        coords = list(polygon_for_summary.exterior.coords)
        stride = max(1, len(coords) // max_vertices)
        sampled = coords[::stride][:max_vertices]
        sampled_latlon = [(round(y, 6), round(x, 6)) for x, y in sampled]
        rep_lat, rep_lon = _representative_latlon(polygon, fallback_lat, fallback_lon)
        width_km, height_km = _polygon_bbox_dimensions_km(polygon)
        dimensions_text = (
            "unknown"
            if width_km is None or height_km is None
            else f"width≈{round(width_km, 3)} km, height≈{round(height_km, 3)} km"
        )
        return (
            "Incident geo-region polygon. All physical sensors generated for this source should be "
            "inside this polygon and directly observe the incident effect. "
            f"Bounding box lat/lon: south={round(miny, 6)}, west={round(minx, 6)}, "
            f"north={round(maxy, 6)}, east={round(maxx, 6)}. "
            f"Approximate bounding-box dimensions: {dimensions_text}. "
            f"Representative point: latitude={round(rep_lat, 6)}, longitude={round(rep_lon, 6)}. "
            f"Sample boundary vertices as (latitude, longitude): {sampled_latlon}."
        )
    except Exception:
        return f"Geo-region object was available but could not be summarized; use a point near latitude={fallback_lat}, longitude={fallback_lon}."


def _get_location_record(
    geomanager: GeoManager,
    location_name: str,
    min_region_side_km: float = MIN_INCIDENT_REGION_SIDE_KM_DEFAULT,
) -> Dict[str, Any]:
    """Resolve a location using GeoManager.get_geo_region() only.

    GeoManager.get_geo_region(location_name) is expected to return a dict whose
    ["geometry"] field is a Shapely Polygon/MultiPolygon.  geocode_name() is not
    used because it is deprecated in this code path.
    """
    try:
        record = geomanager.get_geo_region(location_name)
    except Exception as e:
        raise RuntimeError(f"Could not get geo-region for {location_name!r}: {e}") from e

    if not isinstance(record, dict):
        raise TypeError(
            f"GeoManager.get_geo_region({location_name!r}) should return a dict, "
            f"but returned {type(record)}"
        )

    if "geometry" not in record:
        raise KeyError(
            f"GeoManager.get_geo_region({location_name!r}) returned a dict without a 'geometry' field. "
            f"Available keys: {list(record.keys())}"
        )

    region = record["geometry"]
    if region is None:
        raise ValueError(f"GeoManager.get_geo_region({location_name!r})['geometry'] is None")
    if getattr(region, "is_empty", False):
        raise ValueError(f"GeoManager.get_geo_region({location_name!r})['geometry'] is empty")

    # Shapely points are x=longitude, y=latitude.
    original_region = region
    original_rep_lat, original_rep_lon = _representative_latlon(original_region, 0.0, 0.0)
    region, region_size_check = _ensure_minimum_region_size(
        original_region,
        original_rep_lat,
        original_rep_lon,
        min_side_km=min_region_side_km,
    )
    rep_lat, rep_lon = _representative_latlon(region, original_rep_lat, original_rep_lon)
    region_summary = _polygon_to_prompt(region, rep_lat, rep_lon)

    if region_size_check.get("expanded"):
        print(
            f"Expanded incident region for {location_name!r} to satisfy "
            f"minimum {min_region_side_km}km x {min_region_side_km}km footprint: "
            f"original=({region_size_check.get('original_bbox_width_km')}km x "
            f"{region_size_check.get('original_bbox_height_km')}km), "
            f"final=({region_size_check.get('final_bbox_width_km')}km x "
            f"{region_size_check.get('final_bbox_height_km')}km)"
        )

    return {
        "name": location_name,
        # Backwards-compatible fields. These now mean the representative point
        # of the normalized polygon, not the old geocode_name() point.
        "lat": float(rep_lat),
        "lon": float(rep_lon),
        "representative_lat": float(rep_lat),
        "representative_lon": float(rep_lon),
        "original_representative_lat": float(original_rep_lat),
        "original_representative_lon": float(original_rep_lon),
        "record": record,
        "region": region,
        "original_region": original_region,
        "region_summary": region_summary,
        "region_size_check": region_size_check,
    }

def _build_geo_region_prompt(location_records: List[Dict[str, Any]]) -> str:
    if not location_records:
        return ""
    parts = []
    for idx, loc in enumerate(location_records):
        parts.append(f"Location {idx} ({loc['name']}): {loc['region_summary']}")
    return " Physical sensor placement constraints: " + " ".join(parts)


def _geometry_to_geojson_dict(geometry: Any) -> Optional[Dict[str, Any]]:
    """Serialize a Shapely geometry to a GeoJSON-like dict for GT consumers."""
    if geometry is None:
        return None
    try:
        return mapping(geometry)
    except Exception:
        return None

def _serialize_gt_locations(location_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for loc in location_records:
        out[loc["name"]] = {
            "latitude": loc["lat"],
            "longitude": loc["lon"],
            "representative_latitude": loc["representative_lat"],
            "representative_longitude": loc["representative_lon"],
            "original_representative_latitude": loc.get("original_representative_lat"),
            "original_representative_longitude": loc.get("original_representative_lon"),
            "region_summary": loc["region_summary"],
            "geometry_geojson": _geometry_to_geojson_dict(loc.get("region")),
            "original_geometry_geojson": _geometry_to_geojson_dict(loc.get("original_region")),
            "region_size_check": loc.get("region_size_check", {}),
            "single_region_for_incident": True,
        }
    return out


def _serialize_gt_incident_region(location_record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Convenience single-region representation for newer ground-truth consumers."""
    if location_record is None:
        return None
    return {
        "name": location_record["name"],
        "latitude": location_record["lat"],
        "longitude": location_record["lon"],
        "representative_latitude": location_record["representative_lat"],
        "representative_longitude": location_record["representative_lon"],
        "original_representative_latitude": location_record.get("original_representative_lat"),
        "original_representative_longitude": location_record.get("original_representative_lon"),
        "region_summary": location_record["region_summary"],
        "geometry_geojson": _geometry_to_geojson_dict(location_record.get("region")),
        "original_geometry_geojson": _geometry_to_geojson_dict(location_record.get("original_region")),
        "region_size_check": location_record.get("region_size_check", {}),
        "single_region_for_incident": True,
    }


# ---------------------------------------------------------------------------
# Context and high-level incident planning
# ---------------------------------------------------------------------------

def create_context(llm: ChatOpenAI, event_desc: str) -> str:
    print("Creating simulator context")

    prompt = PromptTemplate(
        template=(
            "You are an incident simulator which creates a variety of occurrences (man made or natural) "
            "which affect the lives, property, or environment of areas around Los Angeles. I will give "
            "a prompt for a type of incident (e.g. wildfire, concert, riot) and your objective is to give "
            "it enough context for synthesizing sensory data such as images, news, social media text, "
            "traffic information, and time series data such as air quality or weather. Here is the "
            "description of the event: {event_description}"
        ),
        input_variables=["event_description"],
    )
    response = llm.invoke(prompt.format(event_description=event_desc))

    print(response.content)
    return response.content


def create_high_level_incident_plan(
    llm: ChatOpenAI,
    event_desc: str,
    num_sub_incidents: int = 3,
    all_incidents: Optional[List[str]] = None,
    sub_incident_min_distance_km: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Break a higher-level incident into localized, possibly heterogeneous sub-incidents.

    A high-level incident does not have to decompose into repeated copies of the
    same type. For example, a wildfire run may contain a wildfire, hazardous
    material release, and air-quality impact if that composition is plausible.
    Each sub-incident type is constrained to ALL_INCIDENTS.
    """
    allowed = all_incidents or ALL_INCIDENTS
    if sub_incident_min_distance_km is None:
        sub_incident_min_distance_km = _default_sub_incident_min_distance_km(event_desc)
    distance_instruction = (
        f"Choose seed locations that are geographically separated: every pair of seed locations "
        f"must be at least {sub_incident_min_distance_km:.1f} km apart. For larger-area hazards "
        "such as wildfires and floods, prefer even larger separation when plausible."
    )
    plan_token_s = "<PLAN>"
    plan_token_e = "</PLAN>"

    prompt = PromptTemplate(
        template=(
            "You are creating a simulation plan for a higher-level incident run in Los Angeles. "
            "The run may contain multiple related incidents, and they do NOT all need to have "
            "the same incident type. Choose a plausible composition from this allowed incident "
            "type list only: {allowed_incidents}. Create {num_sub_incidents} localized sub-incidents "
            "that are part of the same broader run. Examples: a wildfire can also cause a major road "
            "hazardous material release; severe storm damage can cause flooding or earthquake-like damage; "
            "a large civil protest can also produce traffic disruption visible to cameras/traffic sensors. "
            "{distance_instruction} Return only "
            "JSON between {plan_token_s} and {plan_token_e}. Return a JSON list. Each object must have "
            "exactly these keys: incident_type, sub_incident_id, description, seed_location, "
            "start_time_hint. The incident_type value must exactly match one entry in the allowed "
            "incident type list. The seed_location should be one broad Los Angeles County region, such "
            "as a neighborhood, park, wildland area, industrial district, roadway corridor, floodplain, "
            "campus, or landmark region that GeoManager can resolve with get_geo_region(). Do not use "
            "multiple seed locations for a single sub-incident. Incident request: {event_desc}"
        ),
        input_variables=[
            "event_desc",
            "num_sub_incidents",
            "allowed_incidents",
            "distance_instruction",
            "plan_token_s",
            "plan_token_e",
        ],
    )
    response = llm.invoke(
        prompt.format(
            event_desc=event_desc,
            num_sub_incidents=num_sub_incidents,
            allowed_incidents=allowed,
            distance_instruction=distance_instruction,
            plan_token_s=plan_token_s,
            plan_token_e=plan_token_e,
        )
    )

    raw_plan = _extract_between(response.content, plan_token_s, plan_token_e, default="[]")
    try:
        plan = json.loads(raw_plan)
        if not isinstance(plan, list):
            raise ValueError("Plan JSON was not a list")

        cleaned = []
        for idx, item in enumerate(plan[:num_sub_incidents]):
            if not isinstance(item, dict):
                continue
            incident_type = _normalize_incident_type(item.get("incident_type", event_desc))
            sub_id = item.get("sub_incident_id") or f"{incident_type}_{idx}"
            description = item.get("description") or f"{incident_type} related to {event_desc}"
            cleaned.append(
                {
                    "incident_type": incident_type,
                    "sub_incident_id": sub_id,
                    "description": description,
                    "seed_location": item.get("seed_location") or "Los Angeles, CA",
                    "start_time_hint": item.get("start_time_hint") or "unspecified",
                }
            )

        if not cleaned:
            raise ValueError("Plan JSON had no valid objects")
        return cleaned
    except Exception as e:
        print(f"Could not parse high-level plan; falling back to one incident: {e}")
        fallback_type = _normalize_incident_type(event_desc)
        return [
            {
                "incident_type": fallback_type,
                "sub_incident_id": f"{fallback_type.replace(' ', '_')}_0",
                "description": event_desc,
                "seed_location": "Los Angeles, CA",
                "start_time_hint": "unspecified",
            }
        ]


# ---------------------------------------------------------------------------
# Image/camera generation
# ---------------------------------------------------------------------------

def _camera_inside_region(camera_entry, location_record: Dict[str, Any]) -> Optional[bool]:
    cam_lat, cam_lon = camera_entry[0]
    return _polygon_contains_latlon(location_record.get("region"), cam_lat, cam_lon)


def _camera_direction(camera_entry) -> Optional[str]:
    description = str(camera_entry[1])
    if description.lower().startswith("facing"):
        return description
    return None


def _format_camera_list_for_llm(location_records: List[Dict[str, Any]], camera_data_by_location: Dict[int, List[Any]]) -> str:
    all_location_str = ""
    for loc_i, loc in enumerate(location_records):
        all_location_str += (
            f"\n\nFor location {loc_i}: {loc['name']}\n"
            f"Region: {loc['region_summary']}\n"
            "Nearby camera candidates:\n"
        )
        for cam_i, cam_entry in enumerate(camera_data_by_location.get(loc_i, [])):
            inside = _camera_inside_region(cam_entry, loc)
            inside_text = "unknown" if inside is None else str(inside)
            direction = _camera_direction(cam_entry) or "not provided"
            meta = _camera_extra_metadata(cam_entry)
            all_location_str += (
                f"\n\tCamera {cam_i}: Simulated sensor id: {meta.get('sensor_id')}, "
                f"Spoofed sensor location: {cam_entry[0]}, Description: {cam_entry[1]}, "
                f"Origin image source: {cam_entry[2]}, Direction: {direction}, "
                f"Region role: {meta.get('sensor_region_role', 'inside')}, "
                f"Outside distance km: {meta.get('outside_distance_km')}, "
                f"Inside incident region: {inside_text}, "
                f"Original source image sensor: {meta.get('source_image_sensor_location')}"
            )
    return all_location_str


def call_image_edit(
    all_pairs,
    all_location_data: Dict[int, List[Any]],
    location_records: List[Dict[str, Any]],
    simulation_step: int,
    time_info: str,
    incident_id: str,
    save_folder: str = SAVE_FOLDER_DEFAULT,
    simulate_images: bool = True,
    max_image_edits_per_step: int = 1,
    image_every_n_steps: int = 1,
    image_sleep_seconds: float = 0.0,
):
    """Edit selected camera images and write observation metadata.

    Performance knobs:
    - simulate_images=False skips image editing entirely.
    - max_image_edits_per_step caps image API calls per simulator step.
    - image_every_n_steps lets you edit images every N steps.
    - image_sleep_seconds defaults to 0; use it only when your image API requires throttling.
    """
    observation_records = []

    if not simulate_images:
        print("Skipping image generation because simulate_images=False")
        return observation_records

    if max_image_edits_per_step <= 0:
        print("Skipping image generation because max_image_edits_per_step <= 0")
        return observation_records

    if image_every_n_steps > 1 and simulation_step % image_every_n_steps != 0:
        print(f"Skipping image generation on step {simulation_step}; image_every_n_steps={image_every_n_steps}")
        return observation_records

    for pair_idx, curr_pair in enumerate(all_pairs[:max_image_edits_per_step]):
        cam_pair = curr_pair[0]
        vlm_prompt = curr_pair[1]

        selected_location = cam_pair[0]
        cam_selection = cam_pair[1]

        if selected_location not in all_location_data:
            print(f"Skipping image pair with unknown location index {selected_location}")
            continue
        if cam_selection >= len(all_location_data[selected_location]):
            print(f"Skipping image pair with unknown camera index {cam_selection}")
            continue

        cam_data = all_location_data[selected_location][cam_selection]
        loc_record = location_records[selected_location]
        img_filepath = cam_data[3]
        origin = cam_data[2]
        direction = _camera_direction(cam_data)

        if origin == "cctv":
            vlm_prompt += " Replace the text in the top right to be {time_info}. Replace the text in the bottom gray rectangle to be {time_info}."
        elif origin == "alertcalifornia":
            vlm_prompt += " Replace the text in the bottom left box to be {time_info}."

        vlm_prompt = vlm_prompt.format(time_info=time_info)
        vlm_prompt += (
            "\nUnless explicitly specified, do not perform visual processing such as filters or saturation. "
            "Keep the image realistic and close to the original input."
        )

        observation_id = f"{_safe_slug(incident_id)}_step{simulation_step}_image{pair_idx}_{uuid.uuid4().hex[:8]}"
        print("VLM Prompt:\n " + vlm_prompt)

        image_t0 = _now_timer()
        out_filepath = modify_image_cloud(
            img_filepath,
            prompt=vlm_prompt,
            step=simulation_step,
            observation_id=observation_id,
            save_dir=save_folder,
        )
        _print_timing(
            f"{incident_id} step {simulation_step}: image edit {pair_idx + 1}/{min(len(all_pairs), max_image_edits_per_step)}",
            image_t0,
            True,
        )

        if image_sleep_seconds > 0:
            time.sleep(image_sleep_seconds)

        cam_lat, cam_lon = cam_data[0]
        cam_meta = _camera_extra_metadata(cam_data)
        record = {
            "observation_id": observation_id,
            "incident_id": incident_id,
            "step": simulation_step,
            "time": time_info,
            "modality": "image",
            "source": origin,
            "sensor_id": cam_meta.get("sensor_id"),
            "sensor_type": cam_meta.get("sensor_type", _source_family(origin)),
            "sensor_location": {"latitude": cam_lat, "longitude": cam_lon},
            "sensor_region_role": cam_meta.get("sensor_region_role", "inside"),
            "outside_distance_km": cam_meta.get("outside_distance_km"),
            "direction": direction,
            "direction_bearing_degrees": cam_meta.get("direction_bearing_degrees"),
            "direction_compass": cam_meta.get("direction_compass"),
            "what_it_senses": vlm_prompt,
            "direct_observation_target": loc_record["name"],
            "sensor_inside_incident_region": _camera_inside_region(cam_data, loc_record),
            "sensor_pose_spoofed": cam_meta.get("spoofed_sensor_pose", False),
            "source_image_sensor_location": cam_meta.get("source_image_sensor_location"),
            "source_image_camera_description": cam_meta.get("source_image_camera_description"),
            "inward_target_location": cam_meta.get("inward_target_location"),
            "incident_region_summary": loc_record["region_summary"],
            "input_image_path": img_filepath,
            "saved_input_image_path": os.path.join(save_folder, f"{observation_id}_input.{img_filepath.split('.')[-1]}"),
            "output_image_path": out_filepath,
        }
        append_observation_log(record, save_folder=save_folder)
        observation_records.append(record)

    return observation_records


def _build_deterministic_camera_pairs(
    camera_data_by_location: Dict[int, List[Any]],
    location_records: List[Dict[str, Any]],
    context: str,
    max_pairs: int,
) -> Tuple[List[Tuple[Tuple[int, int], str]], str]:
    """Choose camera pairs without an extra LLM call.

    Cameras inside the incident region are already preferred when camera_data_by_location
    is built. This function simply takes the first available candidates and creates a
    conservative edit prompt.
    """
    all_pairs = []
    prompts = []
    for loc_i, cameras in camera_data_by_location.items():
        for cam_i, cam_entry in enumerate(cameras):
            if len(all_pairs) >= max_pairs:
                return all_pairs, "\n".join(prompts)
            loc = location_records[loc_i]
            direction = _camera_direction(cam_entry)
            meta = _camera_extra_metadata(cam_entry)
            role = meta.get("sensor_region_role", "inside")
            distance = meta.get("outside_distance_km")
            if role == "outside":
                location_phrase = f"a fixed negative-control camera about {distance} km outside the incident boundary at {cam_entry[0]} near {loc['name']}"
                prompt = (
                    f"Modify this source camera image as if it was captured by {location_phrase}. "
                    f"Incident context for metadata only: {context}. "
                    "Because this camera is outside the incident region, do not add smoke, flames, flooding, crowds, road blockage, damage, emergency activity, or any other abnormal incident evidence. "
                    "Keep the scene normal/background for the area. "
                )
            else:
                location_phrase = f"a fixed in-region camera at {cam_entry[0]} observing {loc['name']}"
                prompt = (
                    f"Modify this source camera image as if it was captured by {location_phrase}. "
                    f"Incident context: {context}. "
                    "Only add visual effects that would be directly observable from this inward-facing simulated camera viewpoint. "
                )
            if direction:
                prompt += f"The camera direction is {direction}; keep the visual evidence consistent with that direction. "
            prompt += "Preserve the original scene structure as much as possible."
            all_pairs.append(((loc_i, cam_i), prompt))
            prompts.append(prompt)
    return all_pairs, "\n".join(prompts)


def select_and_edit_cameras(
    llm: ChatOpenAI,
    location_records: List[Dict[str, Any]],
    context: str,
    simulation_step: int,
    time_info: str,
    incident_id: str,
    save_folder: str = SAVE_FOLDER_DEFAULT,
    prev_vlm_prompt: str = "",
    simulate_images: bool = True,
    max_image_edits_per_step: int = 1,
    image_every_n_steps: int = 1,
    image_sleep_seconds: float = 0.0,
    use_llm_camera_selection: bool = False,
    source: str = "camera",
    fixed_sensor_layout: Optional[Dict[str, Any]] = None,
    outside_sensors_enabled: bool = True,
    outside_sensors_per_region: int = 1,
    outside_sensor_min_distance_km: float = 1.0,
    outside_sensor_max_distance_km: float = 10.0,
    outside_sensor_types: Optional[Iterable[str]] = None,
):
    if not simulate_images or max_image_edits_per_step <= 0:
        print("Skipping camera selection/image edit due to image performance settings.")
        return [], prev_vlm_prompt

    if image_every_n_steps > 1 and simulation_step % image_every_n_steps != 0:
        print(f"Skipping camera selection on step {simulation_step}; image_every_n_steps={image_every_n_steps}")
        return [], prev_vlm_prompt

    camera_tag_s = "<CAMERA>"
    camera_tag_e = "</CAMERA>"
    vlm_prompt_tag_s = "<DESCRIPTION>"
    vlm_prompt_tag_e = "</DESCRIPTION>"
    pair_tag_s = "<PAIR>"
    pair_tag_e = "</PAIR>"

    fixed_sensor_layout = fixed_sensor_layout if fixed_sensor_layout is not None else {}
    camera_data_by_location = _get_or_create_camera_data_by_location(
        fixed_sensor_layout,
        location_records,
        source=source,
        outside_sensors_enabled=outside_sensors_enabled,
        outside_sensors_per_region=outside_sensors_per_region,
        outside_sensor_min_distance_km=outside_sensor_min_distance_km,
        outside_sensor_max_distance_km=outside_sensor_max_distance_km,
        outside_sensor_types=outside_sensor_types,
    )

    if not use_llm_camera_selection:
        all_pairs, prompt_summary = _build_deterministic_camera_pairs(
            camera_data_by_location,
            location_records,
            context,
            max_pairs=max_image_edits_per_step,
        )
        if not all_pairs:
            candidate_counts = {loc_i: len(cameras) for loc_i, cameras in camera_data_by_location.items()}
            print(
                f"No image edits generated for source={source!r} on step {simulation_step}: "
                f"no camera candidates/pairs were available after filtering. candidate_counts={candidate_counts}"
            )
        records = call_image_edit(
            all_pairs,
            camera_data_by_location,
            location_records,
            simulation_step,
            time_info,
            incident_id,
            save_folder=save_folder,
            simulate_images=simulate_images,
            max_image_edits_per_step=max_image_edits_per_step,
            image_every_n_steps=image_every_n_steps,
            image_sleep_seconds=image_sleep_seconds,
        )
        return records, prompt_summary or prev_vlm_prompt

    all_location_str = _format_camera_list_for_llm(location_records, camera_data_by_location)

    camera_formatting = (
        "Format camera choices using indices only. If I provide 'location 0: camera 0, camera 1', "
        "write the chosen pair between "
        + camera_tag_s
        + " and "
        + camera_tag_e
        + " as location_index,camera_index. For example: "
        + camera_tag_s
        + "0,1"
        + camera_tag_e
        + ". You may choose multiple cameras, requiring multiple tags."
    )

    vlm_prompt_formatting = (
        "Describe each selected image between "
        + vlm_prompt_tag_s
        + " and "
        + vlm_prompt_tag_e
        + " tags. Inside-region cameras should directly observe the incident effect. Outside-region "
        "cameras are negative controls: they are modality-relevant but must show normal/background scene "
        "content with no incident abnormalities. Directionality matters: "
        "if a camera has a 'Facing ...' direction, keep the visual evidence consistent with that direction. "
        "You may switch cameras between simulation steps; do not prefer a previously used camera unless "
        "it is still the best direct observer or a useful negative-control view."
    )
    if prev_vlm_prompt:
        vlm_prompt_formatting += (
            " For visual consistency, here is a previous image-edit description from this incident: "
            + prev_vlm_prompt
            + ". You can use it to maintain temporal progression, but you may still select a different camera."
        )
    else:
        vlm_prompt_formatting += (
            " Construct a specific instantaneous image description from the event context. Avoid simply "
            "saying 'an image of a wildfire'; describe visible smoke/flames/traffic/sky/time/weather/etc."
        )

    prompt_template = PromptTemplate(
        template=(
            "You are helping select cameras for a sensory data simulator. I will give an event context, "
            "the incident geo-regions, and nearby cameras. Select up to {max_pairs} cameras that directly observe "
            "the incident effect. Here is the event context: {context}. Here are locations and nearby "
            "camera candidates: {all_location_str}. {camera_formatting} {vlm_prompt_formatting}. Each "
            "chosen camera should have its own associated description wrapped in "
            + pair_tag_s
            + pair_tag_e
            + " tags. Example: "
            + pair_tag_s
            + camera_tag_s
            + "0,1"
            + camera_tag_e
            + "\n"
            + vlm_prompt_tag_s
            + "some image description and modification steps"
            + vlm_prompt_tag_e
            + pair_tag_e
        ),
        input_variables=["context", "all_location_str", "camera_formatting", "vlm_prompt_formatting", "max_pairs"],
    )

    response = llm.invoke(
        prompt_template.format(
            context=context,
            all_location_str=all_location_str,
            camera_formatting=camera_formatting,
            vlm_prompt_formatting=vlm_prompt_formatting,
            max_pairs=max_image_edits_per_step,
        )
    )

    print("\n\n\nSELECTING CAMERAS AND CREATING VLM PROMPT:\n " + response.content)

    all_pairs = []
    vlm_prompts = []
    for pair in response.content.split(pair_tag_s)[1:]:
        try:
            camera_description = _extract_between(pair, camera_tag_s, camera_tag_e)
            loc_idx, cam_idx = [int(x.strip()) for x in camera_description.split(",")[:2]]
            vlm_prompt = _extract_between(pair, vlm_prompt_tag_s, vlm_prompt_tag_e)
            all_pairs.append(((loc_idx, cam_idx), vlm_prompt))
            vlm_prompts.append(vlm_prompt)
        except Exception as e:
            print(f"Skipping unparsable camera pair: {e}")

    records = call_image_edit(
        all_pairs,
        camera_data_by_location,
        location_records,
        simulation_step,
        time_info,
        incident_id,
        save_folder=save_folder,
        simulate_images=simulate_images,
        max_image_edits_per_step=max_image_edits_per_step,
        image_every_n_steps=image_every_n_steps,
        image_sleep_seconds=image_sleep_seconds,
    )

    return records, "\n".join(vlm_prompts)


# ---------------------------------------------------------------------------
# Simulator loop and observation logging
# ---------------------------------------------------------------------------

def _log_time_series_observations(
    parsed_result: Dict[str, Any],
    source: str,
    step: int,
    time_info: str,
    incident_id: str,
    location_records: List[Dict[str, Any]],
    save_folder: str,
):
    # Associate each row with the first region that contains the generated sensor.
    for row_idx, row in enumerate(parsed_result.get("dict_rows", [])):
        lat, lon = _lat_lon_from_row(row)
        matched_region = None
        inside_any = None
        for loc in location_records:
            inside = _polygon_contains_latlon(loc.get("region"), lat, lon)
            if inside:
                matched_region = loc
                inside_any = True
                break
            if inside is not None:
                inside_any = False

        observation_id = f"{_safe_slug(incident_id)}_step{step}_{_safe_slug(source)}_{row_idx}_{uuid.uuid4().hex[:8]}"
        append_observation_log(
            {
                "observation_id": observation_id,
                "incident_id": incident_id,
                "step": step,
                "time": time_info,
                "modality": "time_series",
                "source": source,
                "sensor_id": row.get("sensor_id"),
                "sensor_type": row.get("sensor_type", _source_family(source)),
                "sensor_location": None if lat is None or lon is None else {"latitude": lat, "longitude": lon},
                "sensor_region_role": row.get("sensor_region_role", "inside" if inside_any else "outside"),
                "outside_distance_km": row.get("outside_distance_km") or None,
                "direction": None,
                "sensor_pose_spoofed": True,
                "what_it_senses": source,
                "direct_observation_target": None if matched_region is None else matched_region["name"],
                "sensor_inside_incident_region": inside_any,
                "data_file": parsed_result.get("filepath"),
                "row_index": row_idx,
                "row": row,
            },
            save_folder=save_folder,
        )


def _log_text_observations(
    parsed_result: Dict[str, Any],
    source: str,
    step: int,
    time_info: str,
    incident_id: str,
    save_folder: str,
):
    for row_idx, row in enumerate(parsed_result.get("dict_rows", [])):
        observation_id = f"{_safe_slug(incident_id)}_step{step}_{_safe_slug(source)}_{row_idx}_{uuid.uuid4().hex[:8]}"
        append_observation_log(
            {
                "observation_id": observation_id,
                "incident_id": incident_id,
                "step": step,
                "time": time_info,
                "modality": "text",
                "source": source,
                "sensor_location": None,
                "direction": None,
                "what_it_senses": "textual report or article about the incident",
                "data_file": parsed_result.get("filepath"),
                "row_index": row_idx,
                "row": row,
            },
            save_folder=save_folder,
        )


def simulator_loop(
    llm: ChatOpenAI,
    event_context: str,
    incident_type: str = "wildfire",
    max_iterations: int = 10,
    incident_id: str = "incident_0",
    save_folder: str = SAVE_FOLDER_DEFAULT,
    geomanager: Optional[GeoManager] = None,
    simulate_images: bool = True,
    max_image_edits_per_step: int = 1,
    image_every_n_steps: int = 1,
    image_sleep_seconds: float = 0.0,
    fast_mode: bool = True,
    use_llm_camera_selection: bool = False,
    max_sources_per_step: Optional[int] = None,
    forced_sources: Optional[Iterable[str]] = None,
    force_image_source: bool = True,
    generate_news_only_at_end: bool = True,
    timing: bool = True,
    fixed_sensor_locations: bool = True,
    inside_sensors_per_region: int = 3,
    outside_sensors_enabled: bool = True,
    outside_sensors_per_region: int = 2,
    outside_sensor_min_distance_km: float = 1.0,
    outside_sensor_max_distance_km: float = 10.0,
    outside_sensor_type_pool: Optional[Iterable[str]] = None,
    outside_sensor_type_count: Optional[int] = None,
    seed_location: Optional[str] = None,
    min_incident_region_side_km: float = MIN_INCIDENT_REGION_SIDE_KM_DEFAULT,
) -> None:
    print(f"Looping simulator for {incident_id}")

    ensure_dir(save_folder)
    previous_simulation_output = "none"

    end_token = "<END_INCIDENT>"
    time_token_s = "<TIME>"
    time_token_e = "</TIME>"
    location_token_s = "<LOCATION>"
    location_token_e = "</LOCATION>"
    curr_event_context_s = "<EVENT_CONTEXT>"
    curr_event_context_e = "</EVENT_CONTEXT>"
    data_source_s = "<SOURCE>"
    data_source_e = "</SOURCE>"

    geomanager = geomanager or GeoManager()

    formatting_instructions = (
        "Write times in ISO 8601 format, e.g. 2025-05-20T15:00:00Z. Write exactly one location in "
        "the LOCATION field: one broad Los Angeles-area region, roadway corridor, park, neighborhood, "
        "campus, industrial district, floodplain, or landmark region that can be geocoded. Append "
        "', Los Angeles County, California' to the location name so geocoding cannot resolve an "
        "identically named place outside the simulation area. Do not list "
        "multiple locations for a single incident/sub-incident. The simulator will bind this entire "
        "incident/sub-incident to one physical geo-region and reuse it for all steps. Prefer a region "
        f"whose footprint is at least {min_incident_region_side_km} km by {min_incident_region_side_km} km; "
        "if GeoManager returns a smaller polygon, the simulator will expand it before placing sensors. "
        "If the incident has ended, write <END_INCIDENT>. Only simulate one step at a time."
    )

    event_description = (
        "Please also provide a concise description of the current progress of the event between "
        + curr_event_context_s
        + " and "
        + curr_event_context_e
        + " tags. Use "
        + data_source_s
        + data_source_e
        + " tags to list comma-separated data sources useful for this step, such as weather, cctv, "
        "alertcalifornia, air quality, california_traffic, news, twitter, or citizen. Use california_traffic "
        "when road speed or road occupancy may be affected. Use image/time-series sources only "
        "when a physical sensor can directly observe the chosen geo-region. Text sources such as citizen "
        "and twitter need not have physical sensor locations."
    )

    prompt_template = PromptTemplate(
        template=(
            "You are an incident simulator for occurrences that affect life, property, or the environment "
            "around Los Angeles. Given an incident description, simulate multiple steps from beginning to end. "
            "Here is the incident description: {event_context}. The last simulation output was: "
            "{previous_simulation_output}. If this is the last step, write {end_token}. Only simulate one "
            "step at a time, and {event_description}. You are on step {curr_step} of {max_steps}, so wrap up "
            "as you get close to the end. Please write output in this format:\n\n"
            "{time_token_s}{time_token_e}\n{location_token_s}{location_token_e}\n"
            "{curr_event_context_s}{curr_event_context_e}\n{data_source_s}{data_source_e}\n"
            + formatting_instructions
        ),
        input_variables=[
            "previous_simulation_output",
            "event_context",
            "end_token",
            "time_token_s",
            "time_token_e",
            "location_token_s",
            "location_token_e",
            "curr_event_context_s",
            "curr_event_context_e",
            "data_source_s",
            "data_source_e",
            "event_description",
            "curr_step",
            "max_steps",
        ],
    )

    curr_it_num = 0
    end_token_present = False
    prev_vlm_prompt = ""
    incident_location_record: Optional[Dict[str, Any]] = None
    incident_location_selection: Dict[str, Any] = {
        "seed_location": seed_location,
        "selected_location_name": None,
        "raw_llm_locations_first_seen": [],
        "selection_method": None,
    }

    if seed_location:
        incident_location_record = _get_location_record(
            geomanager,
            seed_location,
            min_region_side_km=min_incident_region_side_km,
        )
        incident_location_selection.update(
            {
                "selected_location_name": seed_location,
                "selection_method": "high_level_seed_location",
            }
        )

    if outside_sensor_max_distance_km < outside_sensor_min_distance_km:
        outside_sensor_min_distance_km, outside_sensor_max_distance_km = (
            outside_sensor_max_distance_km,
            outside_sensor_min_distance_km,
        )

    outside_sensor_types = (
        _choose_outside_sensor_types(outside_sensor_type_pool, outside_sensor_type_count, incident_type=incident_type)
        if outside_sensors_enabled
        else []
    )
    fixed_sensor_layout: Dict[str, Any] = {
        "enabled": fixed_sensor_locations,
        "outside_sensor_types": outside_sensor_types,
        "time_series": {},
        "camera": {},
    }
    print(
        f"Fixed sensor layout for {incident_id}: fixed={fixed_sensor_locations}, "
        f"outside_enabled={outside_sensors_enabled}, outside_types={outside_sensor_types}, "
        f"outside_distance_km=[{outside_sensor_min_distance_km}, {outside_sensor_max_distance_km}]"
    )

    while curr_it_num < max_iterations and not end_token_present:
        image_edits_remaining_this_step = max(0, int(max_image_edits_per_step or 0))
        image_edits_done_this_step = 0

        prompt = prompt_template.format(
            event_context=event_context,
            previous_simulation_output=previous_simulation_output,
            end_token=end_token,
            time_token_s=time_token_s,
            time_token_e=time_token_e,
            location_token_s=location_token_s,
            location_token_e=location_token_e,
            curr_event_context_s=curr_event_context_s,
            curr_event_context_e=curr_event_context_e,
            data_source_s=data_source_s,
            data_source_e=data_source_e,
            event_description=event_description,
            curr_step=curr_it_num,
            max_steps=max_iterations,
        )

        t0 = _now_timer()
        response = llm.invoke(prompt)
        _print_timing(f"{incident_id} step {curr_it_num + 1}: simulator planning LLM", t0, timing)

        previous_simulation_output = response.content
        if end_token in response.content:
            end_token_present = True
        curr_it_num += 1

        t0 = _now_timer()
        raw_locations = _extract_between(response.content, location_token_s, location_token_e)
        chosen_locations = [x.strip() for x in raw_locations.split(";") if x.strip()]

        # A sub-incident should occupy exactly one physical region.  The LLM may
        # still mention multiple locations, but only the seed location (for a
        # high-level sub-incident) or the first generated location is used.  That
        # one region is then reused across every step before any sensors are
        # sampled.
        if incident_location_record is None:
            selected_location = (
                seed_location
                or (chosen_locations[0] if chosen_locations else None)
                or "Los Angeles, CA"
            )
            incident_location_record = _get_location_record(
                geomanager,
                selected_location,
                min_region_side_km=min_incident_region_side_km,
            )
            incident_location_selection.update(
                {
                    "selected_location_name": selected_location,
                    "raw_llm_locations_first_seen": chosen_locations,
                    "selection_method": "first_llm_location" if chosen_locations else "fallback_los_angeles",
                }
            )
        elif chosen_locations and not incident_location_selection.get("raw_llm_locations_first_seen"):
            incident_location_selection["raw_llm_locations_first_seen"] = chosen_locations

        location_records = [incident_location_record]
        _print_timing(f"{incident_id} step {curr_it_num}: geo-region lookup", t0, timing)

        curr_description = _extract_between(response.content, curr_event_context_s, curr_event_context_e)
        time_info = _extract_between(response.content, time_token_s, time_token_e)
        sources = [s.strip().lower() for s in _extract_between(response.content, data_source_s, data_source_e).split(",") if s.strip()]
        if forced_sources:
            # An explicit source list makes small smoke tests reproducible while
            # retaining the LLM-generated incident context, time, and location.
            sources = list(dict.fromkeys(str(source).strip().lower() for source in forced_sources if str(source).strip()))
        sources = _ensure_image_source_available(
            sources,
            incident_type,
            simulate_images=simulate_images,
            max_image_edits_per_step=max_image_edits_per_step,
            force_image_source=force_image_source,
        )
        if max_sources_per_step is not None and max_sources_per_step > 0:
            sources = sources[:max_sources_per_step]

        geo_region_prompt = _build_geo_region_prompt(location_records)

        for source in sources:
            source_t0 = _now_timer()
            normalized_source = "california_traffic" if _is_traffic_source(source) else source

            if _is_image_source(source):
                if image_edits_remaining_this_step <= 0:
                    print(
                        f"Skipping image source {source!r} on step {curr_it_num}; "
                        f"per-step image budget already used ({image_edits_done_this_step}/{max_image_edits_per_step})."
                    )
                else:
                    image_records, prev_vlm_prompt = select_and_edit_cameras(
                        llm,
                        location_records,
                        curr_description,
                        curr_it_num,
                        time_info,
                        incident_id,
                        save_folder=save_folder,
                        prev_vlm_prompt=prev_vlm_prompt,
                        simulate_images=simulate_images,
                        max_image_edits_per_step=image_edits_remaining_this_step,
                        image_every_n_steps=image_every_n_steps,
                        image_sleep_seconds=image_sleep_seconds,
                        use_llm_camera_selection=use_llm_camera_selection,
                        source=source,
                        fixed_sensor_layout=fixed_sensor_layout if fixed_sensor_locations else {},
                        outside_sensors_enabled=outside_sensors_enabled,
                        outside_sensors_per_region=outside_sensors_per_region,
                        outside_sensor_min_distance_km=outside_sensor_min_distance_km,
                        outside_sensor_max_distance_km=outside_sensor_max_distance_km,
                        outside_sensor_types=outside_sensor_types,
                    )
                    image_edits_done_this_step += len(image_records)
                    image_edits_remaining_this_step = max(0, int(max_image_edits_per_step or 0) - image_edits_done_this_step)

            if _source_has(source, "air", "weather", "pem", "traffic", "california_traffic", "time series", "sensor"):
                if fixed_sensor_locations:
                    sensor_locations = _get_or_create_time_series_sensor_locations(
                        fixed_sensor_layout,
                        location_records,
                        normalized_source,
                        inside_sensors_per_region=inside_sensors_per_region,
                        outside_sensors_enabled=outside_sensors_enabled,
                        outside_sensors_per_region=outside_sensors_per_region,
                        outside_sensor_min_distance_km=outside_sensor_min_distance_km,
                        outside_sensor_max_distance_km=outside_sensor_max_distance_km,
                        outside_sensor_types=outside_sensor_types,
                    )
                else:
                    sensor_locations = _make_time_series_sensor_locations(
                        location_records,
                        normalized_source,
                        curr_it_num,
                        sensors_per_region=inside_sensors_per_region,
                    )
                    if outside_sensors_enabled and _source_uses_outside_sensors(normalized_source, outside_sensor_types):
                        sensor_locations.extend(
                            _make_time_series_sensor_locations(
                                location_records,
                                normalized_source,
                                curr_it_num,
                                sensors_per_region=outside_sensors_per_region,
                                sensor_region_role="outside",
                                min_outside_distance_km=outside_sensor_min_distance_km,
                                max_outside_distance_km=outside_sensor_max_distance_km,
                            )
                        )
                sensor_location_prompt = _format_sensor_locations_for_prompt(sensor_locations)
                parsed = asyncio.run(
                    generate_time_series(
                        time_info,
                        curr_description,
                        normalized_source,
                        curr_it_num,
                        geo_region_prompt=geo_region_prompt + " " + sensor_location_prompt,
                        save_folder=save_folder,
                        incident_id=incident_id,
                        fast_mode=fast_mode,
                        sensor_locations=sensor_locations,
                    )
                )
                parsed = _force_time_series_rows_to_sensor_locations(parsed, sensor_locations)
                parsed = _apply_background_values_to_outside_time_series(parsed, normalized_source)
                _log_time_series_observations(parsed, normalized_source, curr_it_num, time_info, incident_id, location_records, save_folder)

            if _source_has(source, "news", "citizen", "twitter"):
                if generate_news_only_at_end and _source_has(source, "news") and not end_token_present and curr_it_num < max_iterations:
                    print(f"Skipping news on step {curr_it_num}; generate_news_only_at_end=True")
                    continue
                parsed = asyncio.run(
                    generate_text_data(
                        time_info,
                        curr_description,
                        source,
                        curr_it_num,
                        save_folder=save_folder,
                        incident_id=incident_id,
                        fast_mode=fast_mode,
                    )
                )
                _log_text_observations(parsed, source, curr_it_num, time_info, incident_id, save_folder)

            _print_timing(f"{incident_id} step {curr_it_num}: source {source}", source_t0, timing)

        gt_data = {
            "incident_id": incident_id,
            "step": curr_it_num,
            "time": time_info,
            # Backwards-compatible map form.  This will now contain exactly one
            # location because each incident/sub-incident is bound to one region.
            "locations": _serialize_gt_locations(location_records),
            # New explicit single-region form for consumers that should not have
            # to inspect a dict keyed by location name.
            "incident_region": _serialize_gt_incident_region(incident_location_record),
            "location_selection": incident_location_selection,
            "raw_llm_locations_this_step": chosen_locations,
            "event_context": curr_description,
            "sources": sources,
            "fixed_sensor_layout_enabled": fixed_sensor_locations,
            "outside_sensor_types_for_run": outside_sensor_types,
            "outside_sensor_distance_km": {
                "min": outside_sensor_min_distance_km,
                "max": outside_sensor_max_distance_km,
            },
        }
        filepath = os.path.join(save_folder, f"{_safe_slug(incident_id)}_gt_{curr_it_num}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(gt_data, f, indent=4)


def simulate_incident_run(
    llm: ChatOpenAI,
    incident_request: str,
    max_iterations: int = 10,
    high_level: bool = False,
    num_sub_incidents: int = 3,
    save_folder: str = SAVE_FOLDER_DEFAULT,
    reset_observations: bool = True,
    simulate_images: Optional[bool] = None,
    max_image_edits_per_step: Optional[int] = None,
    image_every_n_steps: Optional[int] = None,
    image_sleep_seconds: Optional[float] = None,
    fast_mode: Optional[bool] = None,
    use_llm_camera_selection: Optional[bool] = None,
    max_sources_per_step: Optional[int] = None,
    forced_sources: Optional[Iterable[str]] = None,
    force_image_source: Optional[bool] = None,
    generate_news_only_at_end: Optional[bool] = None,
    timing: Optional[bool] = None,
    fixed_sensor_locations: Optional[bool] = None,
    inside_sensors_per_region: Optional[int] = None,
    outside_sensors_enabled: Optional[bool] = None,
    outside_sensors_per_region: Optional[int] = None,
    outside_sensor_min_distance_km: Optional[float] = None,
    outside_sensor_max_distance_km: Optional[float] = None,
    outside_sensor_type_pool: Optional[Iterable[str]] = None,
    outside_sensor_type_count: Optional[int] = None,
    sub_incident_min_distance_km: Optional[float] = None,
    sub_incident_location_repair_attempts: Optional[int] = None,
    min_incident_region_side_km: Optional[float] = None,
):
    """Simulate either one incident or a higher-level incident with sub-incidents.

    When high_level=True, a request like 'wildfire' can be decomposed into 2-3
    related incidents of different allowed types from ALL_INCIDENTS, such as
    wildfire + hazardous material release + large civil protest, all under the same output
    folder and observation log.  Each sub-incident is bound to exactly one
    normalized geo-region before physical sensors are sampled.
    """
    if simulate_images is None:
        simulate_images = _default_simulate_images()
    if max_image_edits_per_step is None:
        max_image_edits_per_step = _env_int("SIMULATOR_MAX_IMAGE_EDITS_PER_STEP", 1)
    if image_every_n_steps is None:
        image_every_n_steps = _env_int("SIMULATOR_IMAGE_EVERY_N_STEPS", 1)
    if image_sleep_seconds is None:
        image_sleep_seconds = _env_float("SIMULATOR_IMAGE_SLEEP_SECONDS", 0.0)
    if fast_mode is None:
        fast_mode = _default_fast_mode()
    if use_llm_camera_selection is None:
        use_llm_camera_selection = _default_use_llm_camera_selection()
    if force_image_source is None:
        force_image_source = _default_force_image_source()
    if generate_news_only_at_end is None:
        generate_news_only_at_end = _env_bool("SIMULATOR_GENERATE_NEWS_ONLY_AT_END", True)
    if timing is None:
        timing = _env_bool("SIMULATOR_TIMING", True)
    if fixed_sensor_locations is None:
        fixed_sensor_locations = _env_bool("SIMULATOR_FIXED_SENSOR_LOCATIONS", True)
    if inside_sensors_per_region is None:
        inside_sensors_per_region = _env_int("SIMULATOR_INSIDE_SENSORS_PER_REGION", 3)
    if outside_sensors_enabled is None:
        outside_sensors_enabled = _env_bool("SIMULATOR_OUTSIDE_SENSORS_ENABLED", True)
    if outside_sensors_per_region is None:
        outside_sensors_per_region = _env_int("SIMULATOR_OUTSIDE_SENSORS_PER_REGION", 2)
    if outside_sensor_min_distance_km is None:
        outside_sensor_min_distance_km = _env_float("SIMULATOR_OUTSIDE_SENSOR_MIN_DISTANCE_KM", 1.0)
    if outside_sensor_max_distance_km is None:
        outside_sensor_max_distance_km = _env_float("SIMULATOR_OUTSIDE_SENSOR_MAX_DISTANCE_KM", 10.0)
    if outside_sensor_type_count is None:
        env_count = os.environ.get("SIMULATOR_OUTSIDE_SENSOR_TYPE_COUNT")
        outside_sensor_type_count = int(env_count) if env_count not in {None, ""} else None
    if outside_sensor_type_pool is None:
        env_pool = os.environ.get("SIMULATOR_OUTSIDE_SENSOR_TYPE_POOL")
        if env_pool:
            outside_sensor_type_pool = [x.strip() for x in env_pool.split(",") if x.strip()]
        else:
            outside_sensor_type_pool = DEFAULT_OUTSIDE_SENSOR_TYPE_POOL
    if sub_incident_min_distance_km is None:
        env_distance = os.environ.get("SIMULATOR_SUB_INCIDENT_MIN_DISTANCE_KM")
        if env_distance not in {None, ""}:
            try:
                sub_incident_min_distance_km = float(env_distance)
            except Exception:
                sub_incident_min_distance_km = _default_sub_incident_min_distance_km(incident_request)
        else:
            sub_incident_min_distance_km = _default_sub_incident_min_distance_km(incident_request)
    if sub_incident_location_repair_attempts is None:
        sub_incident_location_repair_attempts = _env_int("SIMULATOR_SUB_INCIDENT_LOCATION_REPAIR_ATTEMPTS", 4)
    if min_incident_region_side_km is None:
        min_incident_region_side_km = _env_float(
            "SIMULATOR_MIN_INCIDENT_REGION_SIDE_KM",
            MIN_INCIDENT_REGION_SIDE_KM_DEFAULT,
        )

    print(
        "Simulator performance/settings: "
        f"fast_mode={fast_mode}, simulate_images={simulate_images}, "
        f"max_image_edits_per_step={max_image_edits_per_step}, "
        f"image_every_n_steps={image_every_n_steps}, image_sleep_seconds={image_sleep_seconds}, "
        f"use_llm_camera_selection={use_llm_camera_selection}, "
        f"force_image_source={force_image_source}, "
        f"generate_news_only_at_end={generate_news_only_at_end}, "
        f"fixed_sensor_locations={fixed_sensor_locations}, "
        f"inside_sensors_per_region={inside_sensors_per_region}, "
        f"outside_sensors_enabled={outside_sensors_enabled}, "
        f"outside_sensors_per_region={outside_sensors_per_region}, "
        f"outside_sensor_distance_km=[{outside_sensor_min_distance_km}, {outside_sensor_max_distance_km}], "
        f"outside_sensor_type_pool={list(outside_sensor_type_pool or [])}, "
        f"outside_sensor_type_count={outside_sensor_type_count}, "
        f"sub_incident_min_distance_km={sub_incident_min_distance_km}, "
        f"sub_incident_location_repair_attempts={sub_incident_location_repair_attempts}, "
        f"min_incident_region_side_km={min_incident_region_side_km}"
    )

    ensure_dir(save_folder)
    if reset_observations:
        obs_path = os.path.join(save_folder, "observations.txt")
        if os.path.exists(obs_path):
            os.remove(obs_path)

    geomanager = GeoManager()

    if high_level:
        plan = create_high_level_incident_plan(
            llm,
            incident_request,
            num_sub_incidents=num_sub_incidents,
            all_incidents=ALL_INCIDENTS,
            sub_incident_min_distance_km=sub_incident_min_distance_km,
        )
        plan = _ensure_sub_incident_seed_locations_far_apart(
            llm,
            geomanager,
            plan,
            incident_request,
            min_distance_km=sub_incident_min_distance_km,
            repair_attempts=sub_incident_location_repair_attempts,
        )
    else:
        plan = [
            {
                "incident_type": _normalize_incident_type(incident_request),
                "sub_incident_id": "incident_0",
                "description": incident_request,
                "seed_location": None,
                "start_time_hint": None,
            }
        ]

    run_id = _safe_slug(incident_request, fallback="incident_run") + "_" + uuid.uuid4().hex[:8]
    plan_path = os.path.join(save_folder, f"{run_id}_plan.json")
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "high_level": high_level,
                "sub_incident_min_distance_km": sub_incident_min_distance_km,
                "sub_incident_location_repair_attempts": sub_incident_location_repair_attempts,
                "min_incident_region_side_km": min_incident_region_side_km,
                "allowed_incidents": ALL_INCIDENTS,
                "incident_relevant_outside_sensor_types": _incident_relevant_outside_sensor_types_map(),
                "plan": plan,
            },
            f,
            indent=4,
        )

    for idx, item in enumerate(plan):
        sub_id = _safe_slug(item.get("sub_incident_id") or f"incident_{idx}")
        incident_id = f"{run_id}_{sub_id}"
        incident_type = item.get("incident_type", _normalize_incident_type(incident_request))
        desc = item.get("description", incident_request)
        seed_location = item.get("seed_location")
        start_time_hint = item.get("start_time_hint")
        context_request = f"Incident type: {incident_type}. Description: {desc}"
        if seed_location:
            context_request += f" Seed location: {seed_location}."
        if start_time_hint:
            context_request += f" Start time hint: {start_time_hint}."

        incident_context = create_context(llm, context_request)
        simulator_loop(
            llm,
            incident_context,
            incident_type=incident_type,
            max_iterations=max_iterations,
            incident_id=incident_id,
            save_folder=save_folder,
            geomanager=geomanager,
            simulate_images=simulate_images,
            max_image_edits_per_step=max_image_edits_per_step,
            image_every_n_steps=image_every_n_steps,
            image_sleep_seconds=image_sleep_seconds,
            fast_mode=fast_mode,
            use_llm_camera_selection=use_llm_camera_selection,
            max_sources_per_step=max_sources_per_step,
            forced_sources=forced_sources,
            force_image_source=force_image_source,
            generate_news_only_at_end=generate_news_only_at_end,
            timing=timing,
            fixed_sensor_locations=fixed_sensor_locations,
            inside_sensors_per_region=inside_sensors_per_region,
            outside_sensors_enabled=outside_sensors_enabled,
            outside_sensors_per_region=outside_sensors_per_region,
            outside_sensor_min_distance_km=outside_sensor_min_distance_km,
            outside_sensor_max_distance_km=outside_sensor_max_distance_km,
            outside_sensor_type_pool=outside_sensor_type_pool,
            outside_sensor_type_count=outside_sensor_type_count,
            seed_location=seed_location,
            min_incident_region_side_km=min_incident_region_side_km,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic IncidentLens observation streams."
    )
    parser.add_argument(
        "--incident-types",
        nargs="+",
        choices=ALL_INCIDENTS,
        default=None,
        help="Incident types to generate. Defaults to every supported type.",
    )
    parser.add_argument(
        "--runs-per-incident",
        type=int,
        default=None,
        help="Runs per selected incident type. Default: 25.",
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        help="Folder beneath paths.simulator_output_root. Default: batch_incident_runs.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override paths.simulator_output_root (useful for isolated smoke runs).",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Use exactly these sources at every step, e.g. cctv weather news.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Simulation steps per run. Default: 5.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned batch without calling services or writing output.",
    )
    args = parser.parse_args()

    incident_names = list(args.incident_types or ALL_INCIDENTS)
    runs_per_incident = args.runs_per_incident
    if runs_per_incident is None:
        runs_per_incident = _env_int("SIMULATOR_RUNS_PER_INCIDENT", 25)
    if runs_per_incident < 1:
        parser.error("--runs-per-incident must be at least 1")
    if args.max_iterations < 1:
        parser.error("--max-iterations must be at least 1")
    batch_output_folder = args.output_folder or os.environ.get(
        "SIMULATOR_BATCH_OUTPUT_FOLDER", "batch_incident_runs"
    )

    if args.dry_run:
        print(
            f"Would generate {len(incident_names)} incident type(s) x "
            f"{runs_per_incident} run(s) in {batch_output_folder}: "
            + ", ".join(incident_names)
        )
        raise SystemExit(0)

    config = get_config()
    openai = config["openai"]
    openai_api_key = openai["api"]

    # Disable LangSmith tracing for simulator runs by default; otherwise long
    # simulations can quickly hit trace quota. Re-enable manually if needed.
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ.pop("LANGSMITH_API_KEY", None)
    os.environ.pop("LANGCHAIN_API_KEY", None)
    os.environ["OPENAI_API_KEY"] = openai_api_key

    llm = ChatOpenAI(model="gpt-5.4", temperature=0)
    agent = ToolStateAgent()

    # Incident types to batch-generate.  These are larger-footprint events that
    # can be sensed by air quality, images/cameras, or traffic speed/occupancy.
    simulator_output_root = args.output_root or config.get("paths", {}).get("simulator_output_root", SAVE_FOLDER_DEFAULT)
    batch_save_root = os.path.join(simulator_output_root, batch_output_folder)
    ensure_dir(batch_save_root)

    total_start_time = time.time()
    total_runs = len(incident_names) * runs_per_incident
    completed_runs = 0

    print(
        f"Starting simulator batch: {len(incident_names)} incident types x "
        f"{runs_per_incident} runs = {total_runs} total runs"
    )
    print(f"Saving batch output under: {batch_save_root}")

    # Generate in round-robin order across incident types:
    #   wildfire1, urban_fire1, ..., terrorist_attack1,
    #   wildfire2, urban_fire2, ..., terrorist_attack2, ...
    # This is better for long batches because partially completed runs cover all
    # classes instead of producing many examples from one class before moving on.
    batch_run_schedule = [
        (incident_name, run_idx)
        for run_idx in range(1, runs_per_incident + 1)
        for incident_name in incident_names
    ]

    schedule_path = os.path.join(batch_save_root, "batch_run_schedule.json")
    with open(schedule_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "order": order_idx,
                    "incident_name": incident_name,
                    "run_idx": run_idx,
                    "folder_name": f"{_safe_slug(incident_name, fallback='incident')}{run_idx}",
                }
                for order_idx, (incident_name, run_idx) in enumerate(batch_run_schedule, start=1)
            ],
            f,
            indent=4,
        )
    print(f"Wrote batch run schedule to: {schedule_path}")

    for incident_name, run_idx in batch_run_schedule:
        completed_runs += 1
        safe_incident_name = _safe_slug(incident_name, fallback="incident")
        folder_name = f"{safe_incident_name}{run_idx}"
        run_save_folder = os.path.join(batch_save_root, folder_name)
        ensure_dir(run_save_folder)

        print(
            "\n" + "=" * 80 + "\n"
            f"[{completed_runs}/{total_runs}] Running incident={incident_name!r}, "
            f"run={run_idx}, save_folder={run_save_folder}\n"
            + "=" * 80
        )

        run_start_time = time.time()
        try:
            simulate_incident_run(
                llm,
                incident_name,
                max_iterations=args.max_iterations,
                high_level=False,
                num_sub_incidents=2,
                save_folder=run_save_folder,
                reset_observations=True,
                # Fast defaults. Override with environment variables or function args.
                fast_mode=None,
                simulate_images=None,
                max_image_edits_per_step=None,
                image_sleep_seconds=None,
                use_llm_camera_selection=None,
                force_image_source=None,
                forced_sources=args.sources,
                fixed_sensor_locations=None,
                inside_sensors_per_region=None,
                outside_sensors_enabled=None,
                outside_sensors_per_region=None,
                outside_sensor_min_distance_km=None,
                outside_sensor_max_distance_km=None,
                outside_sensor_type_count=None,
                sub_incident_min_distance_km=None,
                sub_incident_location_repair_attempts=None,
                min_incident_region_side_km=None,
            )
        except Exception as e:
            # Keep long batches from dying because one incident/location failed.
            error_path = os.path.join(run_save_folder, "run_error.json")
            with open(error_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "incident_name": incident_name,
                        "run_idx": run_idx,
                        "save_folder": run_save_folder,
                        "error": repr(e),
                    },
                    f,
                    indent=4,
                )
            print(f"Run failed; wrote error details to {error_path}: {e!r}")
            continue

        run_end_time = time.time()
        print(
            f"Finished {incident_name!r} run {run_idx} in "
            f"{run_end_time - run_start_time:.2f} seconds"
        )

    total_end_time = time.time()
    print(f"Total batch simulation time: {total_end_time - total_start_time:.2f} seconds")
