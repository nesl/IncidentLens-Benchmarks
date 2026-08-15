#!/usr/bin/env python3
"""Profile real extracted data and optionally replay normalized REPORT objects.

Place this file at:

    evaluation/labelling/real_emitter.py

It reads already-extracted data from:

    evaluation/temp/<data_source>/<YYYYMMDD>/...

The default behavior is profiling only. Actual emission is OFF unless you edit
main() and enable EMIT_TO_SOCKET and/or WRITE_ORDERED_REPORTS_JSONL.

Important behavior:
  * Reports use the same schema family as synthetic_emitter.py.
  * PeMS 5-minute traffic data is downsampled to one record per sensor per hour
    by default, using the first row seen for each station-hour bucket.
  * When emission/output is enabled, reports from all sources are sorted globally
    by report_date before replay/output. This avoids source-blocked replay such
    as all CCTV first, then all air data.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import json
import math
import re
import socket
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
from observation_contract import normalize_report
from utilities.util import get_config

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None  # type: ignore


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_DATA_SOURCES = [
    "air_data",
    "alertcalifornia",
    "cctv",
    "citizen_data",
    "pem_data_station_5min",
    "twitter_data",
    "weather_data",
]

# Folder aliases are useful because the coverage/profiling code has used both
# names such as weather_data and weather across experiments.  The source name
# emitted in the report stays as the logical source requested in DATA_SOURCES.
SOURCE_FOLDER_ALIASES = {
    "air_data": ["air_data", "air"],
    "alertcalifornia": ["alertcalifornia", "alert_california"],
    "cctv": ["cctv"],
    "citizen_data": ["citizen_data", "citizen"],
    "pem_data_station_5min": ["pem_data_station_5min", "pems", "traffic", "traffic_data"],
    "twitter_data": ["twitter_data", "twitter"],
    "weather_data": ["weather_data", "weather"],
}

# Camera image observations are better represented by the area viewed by the
# camera, not by the physical camera mount point.  These defaults mirror the
# assumptions used in evaluation/labelling/sensor_coverage.py.
DEFAULT_CCTV_FOV_DEG = 60.0
DEFAULT_CCTV_VIEW_DISTANCE_KM = 1.0
DEFAULT_ALERTCALIFORNIA_FOV_DEG = 60.0
DEFAULT_ALERTCALIFORNIA_VIEW_DISTANCE_KM = 50.0
EARTH_RADIUS_KM = 6371.0088

# CCTV sometimes returns an "unavailable camera" placeholder image rather than
# a real scene.  The image can contain slightly different text/metadata, so the
# filter below uses a small grayscale perceptual signature instead of exact
# bytes or filename matching.
DEFAULT_CCTV_UNAVAILABLE_IMAGE_CANDIDATES = [
    Path("evaluation/sensor_locations/unavailable.jpg"),
    Path("sensor_locations/unavailable.jpg"),
]
DEFAULT_SKIP_CCTV_UNAVAILABLE_IMAGES = True
DEFAULT_CCTV_UNAVAILABLE_SIGNATURE_SIZE = 64
DEFAULT_CCTV_UNAVAILABLE_MEAN_ABS_DIFF_THRESHOLD = 18.0
DEFAULT_CCTV_UNAVAILABLE_LOOSE_MEAN_ABS_DIFF_THRESHOLD = 35.0
DEFAULT_CCTV_UNAVAILABLE_MEAN_INTENSITY_TOLERANCE = 20.0
DEFAULT_CCTV_UNAVAILABLE_STD_TOLERANCE = 20.0
DEFAULT_CCTV_UNAVAILABLE_BRIGHT_FRACTION_TOLERANCE = 0.12

ROW_LOCATION_KEYS = {"lat", "latitude", "lon", "longitude"}
ROW_TIME_KEYS = {"timestamp", "time", "date", "datetime", "dt_txt"}

# Source/date profile cache.  This is separate from the extraction cache:
# extraction caching avoids re-untarring TARs, while this avoids rebuilding the
# normalized report list by walking/opening thousands of CCTV images on every run.
PROFILE_REPORT_CACHE_VERSION = 1
DEFAULT_PROFILE_REPORT_CACHE_DIRNAME = "_real_emitter_profile_cache"

# PeMS station metadata cache.  PeMS rows only contain station IDs; each
# source/date profile currently joins every row against pem_7_stations.txt.
# Persisting the parsed station-location table avoids rereading/parsing the
# same 4k+ station metadata file once per date and across runs.
PEMS_STATION_CACHE_VERSION = 1
DEFAULT_PEMS_STATION_CACHE_FILENAME = "pems_station_locations.json"

# Per-image unavailable-camera decision cache.  The source/date profile cache is
# fastest when it hits, but if a run is interrupted before that whole profile is
# written, this cache lets the next run resume without reopening/recomputing the
# perceptual signature for every CCTV image again.
UNAVAILABLE_IMAGE_DECISION_CACHE_VERSION = 1
DEFAULT_CCTV_UNAVAILABLE_DECISION_CACHE_FILENAME = "cctv_unavailable_image_decisions.jsonl"

# Observation-model cache root used for positive-only real-data replay.
# Expected layout: detection/cache/observation_model/real_data/<source>/<YYYYMMDD>/*.json
DEFAULT_OBSERVATION_CACHE_ROOT = Path("detection/cache/observation_model/real_data")

try:
    from shapely import wkt as shapely_wkt  # type: ignore
except Exception:  # pragma: no cover - only needed for cached text-location geometries
    shapely_wkt = None  # type: ignore

TEXT_LOCATION_KEYS = (
    "location",
    "location_name",
    "location description",
    "location_description",
    "address",
    "place",
    "place_name",
    "neighborhood",
    "city",
    "area",
)

TEXT_TIME_KEYS = tuple(sorted(ROW_TIME_KEYS | {
    "original date",
    "original_date",
    "created_at",
    "created at",
    "published_at",
    "published at",
    "event_date",
    "event date",
}))


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def progress_iter(iterable: Iterable[Any], *, desc: str, total: Optional[int] = None, unit: str = "it", leave: bool = True) -> Iterable[Any]:
    """Wrap an iterable in tqdm when available; otherwise return it unchanged."""
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit=unit, leave=leave, dynamic_ncols=True)


def resolve_existing_path(candidates: Sequence[Path]) -> Optional[Path]:
    """Return the first existing path from a list of candidate paths."""
    for candidate in candidates:
        try:
            candidate = Path(candidate)
            if candidate.exists():
                return candidate
        except Exception:
            continue
    return None


def image_unavailable_signature(path: Path, *, size: int) -> Optional[Dict[str, Any]]:
    """Return a compact grayscale signature for unavailable-camera filtering.

    The unavailable image is mostly a white placeholder with text.  Since the
    text/metadata can vary, this signature combines a downsampled grayscale
    thumbnail with coarse brightness statistics.  It intentionally avoids exact
    byte matching and avoids adding dependencies beyond Pillow.
    """
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None

    try:
        image = Image.open(path).convert("L")
        resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR)
        image = image.resize((int(size), int(size)), resample)
        pixels = list(image.getdata())
        if not pixels:
            return None
        n = float(len(pixels))
        mean = sum(float(x) for x in pixels) / n
        variance = sum((float(x) - mean) ** 2 for x in pixels) / n
        bright_fraction = sum(1 for x in pixels if int(x) >= 240) / n
        dark_fraction = sum(1 for x in pixels if int(x) <= 40) / n
        return {
            "pixels": pixels,
            "size": int(size),
            "mean": mean,
            "std": math.sqrt(variance),
            "bright_fraction": bright_fraction,
            "dark_fraction": dark_fraction,
            "source_path": str(path),
        }
    except Exception:
        return None


def compare_image_to_unavailable_signature(
    image_path: Path,
    reference_signature: Optional[Dict[str, Any]],
    *,
    size: int = DEFAULT_CCTV_UNAVAILABLE_SIGNATURE_SIZE,
    strict_mean_abs_diff_threshold: float = DEFAULT_CCTV_UNAVAILABLE_MEAN_ABS_DIFF_THRESHOLD,
    loose_mean_abs_diff_threshold: float = DEFAULT_CCTV_UNAVAILABLE_LOOSE_MEAN_ABS_DIFF_THRESHOLD,
    mean_intensity_tolerance: float = DEFAULT_CCTV_UNAVAILABLE_MEAN_INTENSITY_TOLERANCE,
    std_tolerance: float = DEFAULT_CCTV_UNAVAILABLE_STD_TOLERANCE,
    bright_fraction_tolerance: float = DEFAULT_CCTV_UNAVAILABLE_BRIGHT_FRACTION_TOLERANCE,
) -> Tuple[bool, Dict[str, Any]]:
    """Return whether image_path looks like the CCTV unavailable placeholder.

    A direct thumbnail mean-absolute-difference catches near-identical
    placeholders.  A looser statistic check catches versions where the text
    changes slightly while the overall white placeholder appearance remains.
    If Pillow or the reference image is unavailable, this fails open and does
    not skip the image.
    """
    if reference_signature is None:
        return False, {"status": "disabled_no_reference_signature"}

    signature = image_unavailable_signature(image_path, size=size)
    if signature is None:
        return False, {"status": "could_not_compute_image_signature"}

    ref_pixels = reference_signature.get("pixels") or []
    pixels = signature.get("pixels") or []
    if len(ref_pixels) != len(pixels) or not pixels:
        return False, {"status": "signature_shape_mismatch"}

    mean_abs_diff = sum(abs(float(a) - float(b)) for a, b in zip(pixels, ref_pixels)) / float(len(pixels))
    mean_delta = abs(float(signature["mean"]) - float(reference_signature["mean"]))
    std_delta = abs(float(signature["std"]) - float(reference_signature["std"]))
    bright_delta = abs(float(signature["bright_fraction"]) - float(reference_signature["bright_fraction"]))

    strict_match = mean_abs_diff <= float(strict_mean_abs_diff_threshold)
    loose_placeholder_match = (
        mean_abs_diff <= float(loose_mean_abs_diff_threshold)
        and mean_delta <= float(mean_intensity_tolerance)
        and std_delta <= float(std_tolerance)
        and bright_delta <= float(bright_fraction_tolerance)
    )
    is_unavailable = bool(strict_match or loose_placeholder_match)

    return is_unavailable, {
        "status": "ok",
        "is_unavailable_like": is_unavailable,
        "match_reason": "strict_thumbnail_diff" if strict_match else ("loose_brightness_stats" if loose_placeholder_match else "not_similar"),
        "mean_abs_pixel_diff": round(float(mean_abs_diff), 4),
        "mean_intensity_delta": round(float(mean_delta), 4),
        "std_delta": round(float(std_delta), 4),
        "bright_fraction_delta": round(float(bright_delta), 4),
        "image_mean": round(float(signature["mean"]), 4),
        "image_std": round(float(signature["std"]), 4),
        "image_bright_fraction": round(float(signature["bright_fraction"]), 4),
        "reference_path": reference_signature.get("source_path"),
        "thresholds": {
            "strict_mean_abs_diff": float(strict_mean_abs_diff_threshold),
            "loose_mean_abs_diff": float(loose_mean_abs_diff_threshold),
            "mean_intensity_tolerance": float(mean_intensity_tolerance),
            "std_tolerance": float(std_tolerance),
            "bright_fraction_tolerance": float(bright_fraction_tolerance),
        },
    }



def cctv_unavailable_decision_cache_path(cache_dir: Path) -> Path:
    """Path for append-only per-image unavailable-camera decisions."""
    return Path(cache_dir) / "cctv" / DEFAULT_CCTV_UNAVAILABLE_DECISION_CACHE_FILENAME


def _file_signature_for_cache(path: Path) -> Dict[str, Any]:
    """Small file signature used to invalidate stale per-image decisions."""
    try:
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
        }
    except Exception:
        return {
            "path": str(path),
            "size_bytes": None,
            "mtime_ns": None,
        }


def cctv_unavailable_decision_cache_key(
    image_path: Path,
    reference_signature: Optional[Dict[str, Any]],
) -> str:
    """Cache key for one image/reference/threshold combination."""
    ref_path = None
    if isinstance(reference_signature, dict):
        ref_path = reference_signature.get("source_path")
    file_sig = _file_signature_for_cache(image_path)
    payload = {
        "version": UNAVAILABLE_IMAGE_DECISION_CACHE_VERSION,
        "image": file_sig,
        "reference_path": str(ref_path or ""),
        "signature_size": DEFAULT_CCTV_UNAVAILABLE_SIGNATURE_SIZE,
        "strict_mean_abs_diff_threshold": DEFAULT_CCTV_UNAVAILABLE_MEAN_ABS_DIFF_THRESHOLD,
        "loose_mean_abs_diff_threshold": DEFAULT_CCTV_UNAVAILABLE_LOOSE_MEAN_ABS_DIFF_THRESHOLD,
        "mean_intensity_tolerance": DEFAULT_CCTV_UNAVAILABLE_MEAN_INTENSITY_TOLERANCE,
        "std_tolerance": DEFAULT_CCTV_UNAVAILABLE_STD_TOLERANCE,
        "bright_fraction_tolerance": DEFAULT_CCTV_UNAVAILABLE_BRIGHT_FRACTION_TOLERANCE,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8", errors="replace")
    return hashlib.sha1(encoded).hexdigest()


def load_cctv_unavailable_decision_cache(cache_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load append-only JSONL unavailable decisions; last duplicate wins."""
    cache: Dict[str, Dict[str, Any]] = {}
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return cache
    try:
        with cache_path.open("r", encoding="utf-8", errors="replace") as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                if int(record.get("cache_version", -1)) != UNAVAILABLE_IMAGE_DECISION_CACHE_VERSION:
                    continue
                key = str(record.get("cache_key") or "")
                if key:
                    cache[key] = record
    except Exception as exc:
        log(f"WARNING: could not load CCTV unavailable-image decision cache {cache_path}: {exc}")
    return cache


def append_cctv_unavailable_decision_cache(cache_path: Path, record: Dict[str, Any]) -> None:
    """Append one unavailable-image decision without rewriting a large cache file."""
    try:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as outfile:
            outfile.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception as exc:
        log(f"WARNING: could not append CCTV unavailable-image decision cache {cache_path}: {exc}")


def compare_image_to_unavailable_signature_cached(
    image_path: Path,
    reference_signature: Optional[Dict[str, Any]],
    *,
    decision_cache: Optional[Dict[str, Dict[str, Any]]] = None,
    decision_cache_path: Optional[Path] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Cached wrapper around compare_image_to_unavailable_signature(...).

    On a cache hit this avoids opening the image with Pillow.  On a miss it
    computes the decision once and appends it to the JSONL cache immediately, so
    interrupted runs still preserve progress.
    """
    if decision_cache is None:
        return compare_image_to_unavailable_signature(image_path, reference_signature)

    cache_key = cctv_unavailable_decision_cache_key(image_path, reference_signature)
    cached = decision_cache.get(cache_key)
    if isinstance(cached, dict):
        diag = dict(cached.get("diagnostics") or {})
        diag["cache_status"] = "hit"
        diag["cache_key"] = cache_key
        return bool(cached.get("is_unavailable_like")), diag

    is_unavailable, diag = compare_image_to_unavailable_signature(image_path, reference_signature)
    diag = dict(diag)
    diag["cache_status"] = "miss_stored"
    diag["cache_key"] = cache_key

    record = {
        "cache_version": UNAVAILABLE_IMAGE_DECISION_CACHE_VERSION,
        "cache_key": cache_key,
        "image": _file_signature_for_cache(image_path),
        "reference_path": (reference_signature or {}).get("source_path") if isinstance(reference_signature, dict) else None,
        "is_unavailable_like": bool(is_unavailable),
        "diagnostics": diag,
    }
    decision_cache[cache_key] = record
    if decision_cache_path is not None:
        append_cctv_unavailable_decision_cache(decision_cache_path, record)
    return is_unavailable, diag


def is_date_string(value: str) -> bool:
    return len(value) == 8 and value.isdigit()


def date_str_to_midnight_iso(date_str: str) -> Optional[str]:
    if not is_date_string(date_str):
        return None
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T00:00:00"


def coerce_scalar(value: Any) -> Any:
    """Convert numeric-looking strings to ints/floats while preserving text."""
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if stripped == "":
        return value

    try:
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            return int(stripped)
        return float(stripped)
    except ValueError:
        return value


def coerce_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: coerce_scalar(value) for key, value in payload.items()}

def record_get_ci(record: Mapping[str, Any], *keys: str) -> Any:
    """Case-insensitive lookup for messy CSV/JSON records."""
    if not isinstance(record, Mapping) or not keys:
        return None
    wanted = {str(key).strip().lower() for key in keys}
    for key, value in record.items():
        if str(key).strip().lower() in wanted:
            return value
    return None


def first_present_ci(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record_get_ci(record, key)
        if value is not None and value != "":
            return value
    return None


def generic_location_text_from_record(record: Mapping[str, Any]) -> Optional[str]:
    """Return a natural-language location string from citizen/twitter rows.

    Real citizen/twitter exports often use capitalized headers such as
    ``Location`` rather than lowercase ``location``.  Keep this case-insensitive
    and conservative so it also works for JSON/JSONL sources.
    """
    value = first_present_ci(record, *TEXT_LOCATION_KEYS)
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"n/a", "na", "none", "unknown", "null"}:
        return None
    return text


def normalize_text_location_query(value: str) -> str:
    """Qualify short LA-area text locations before geocoding."""
    text = str(value).strip()
    if not text:
        return text
    lower = text.lower()
    if any(token in lower for token in ["california", " ca", ",ca", "los angeles", " united states", " usa", ", us"]):
        return maybe_inject_california_local(text)
    return f"{text}, Los Angeles, California, US"


def geometry_centroid_latlon_from_wkt(geometry_wkt: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    if not geometry_wkt or shapely_wkt is None:
        return None, None
    try:
        geom = shapely_wkt.loads(geometry_wkt)
        if geom is None or geom.is_empty:
            return None, None
        c = geom.centroid
        return float(c.y), float(c.x)
    except Exception:
        return None, None


class TextLocationResolver:
    """Resolve natural-language text/social locations to lat/lon.

    This first reads the GeoManager cache file directly so cached conversions
    work even when live geocoding dependencies/API keys are unavailable.  If a
    location is not cached and cache_only=False, it lazily constructs GeoManager
    and writes the result to the same cache through GeoManager.get_geo_region().
    """

    def __init__(
        self,
        *,
        cache_path: str | Path = "evaluation/geo_cache/geo_region_cache.json",
        enabled: bool = True,
        cache_only: bool = False,
    ):
        self.cache_path = Path(cache_path)
        self.enabled = bool(enabled)
        self.cache_only = bool(cache_only)
        self._cache: Optional[Dict[str, Any]] = None
        self._geo_manager: Any = None
        self._geo_manager_attempted = False
        self.stats: Dict[str, int] = {
            "attempted": 0,
            "cache_hits": 0,
            "live_geocode_success": 0,
            "failed": 0,
            "disabled": 0,
            "cache_only_misses": 0,
        }

    def _normalize_cache_key(self, query: str) -> str:
        return str(query).strip().lower()

    def _load_cache(self) -> Dict[str, Any]:
        if self._cache is not None:
            return self._cache
        try:
            if self.cache_path.exists():
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            else:
                self._cache = {}
        except Exception as exc:
            log(f"WARNING: could not read text-location geocode cache {self.cache_path}: {exc}")
            self._cache = {}
        return self._cache

    def _record_to_location(self, *, raw_text: str, query: str, record: Mapping[str, Any], cache_hit: bool) -> Optional[Dict[str, Any]]:
        if not isinstance(record, Mapping) or record.get("status") != "ok":
            return None
        lat = as_float(record.get("lat"))
        lon = as_float(record.get("lng", record.get("lon")))
        if lat is None or lon is None:
            lat, lon = geometry_centroid_latlon_from_wkt(record.get("geometry_wkt"))
        if lat is None or lon is None:
            return None
        return {
            "latitude": lat,
            "longitude": lon,
            "location_name": raw_text,
            "location_description": raw_text,
            "location_query": query,
            "geocode_source": record.get("source"),
            "geocode_method": record.get("method"),
            "geometry_type": record.get("geometry_type"),
            "geometry_wkt": record.get("geometry_wkt"),
            "cache_hit": bool(cache_hit or record.get("cache_hit")),
        }

    def _load_geo_manager(self) -> Any:
        if self._geo_manager_attempted:
            return self._geo_manager
        self._geo_manager_attempted = True
        errors: List[str] = []
        for module_name in ("evaluation.geo_manager", "evaluation.geo_manager.geo_manager", "geo_manager"):
            try:
                module = importlib.import_module(module_name)
                GeoManager = getattr(module, "GeoManager")
                self._geo_manager = GeoManager(cache_path=str(self.cache_path))
                return self._geo_manager
            except Exception as exc:
                errors.append(f"{module_name}: {exc!r}")
        log("WARNING: text-location live geocoding unavailable: " + " | ".join(errors))
        self._geo_manager = None
        return None

    def resolve(self, raw_location: Optional[str]) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            self.stats["disabled"] += 1
            return None
        raw_text = str(raw_location or "").strip()
        if not raw_text:
            return None
        self.stats["attempted"] += 1
        query = normalize_text_location_query(raw_text)
        cache = self._load_cache()
        for candidate in (raw_text, query):
            key = self._normalize_cache_key(candidate)
            record = cache.get(key)
            loc = self._record_to_location(raw_text=raw_text, query=query, record=record or {}, cache_hit=True)
            if loc is not None:
                self.stats["cache_hits"] += 1
                return loc

        if self.cache_only:
            self.stats["cache_only_misses"] += 1
            return None

        gm = self._load_geo_manager()
        if gm is None:
            self.stats["failed"] += 1
            return None
        try:
            record = gm.get_geo_region(query)
            loc = self._record_to_location(raw_text=raw_text, query=query, record=record or {}, cache_hit=False)
            self._cache = None
            if loc is not None:
                self.stats["live_geocode_success"] += 1
                return loc
        except Exception as exc:
            log(f"WARNING: failed to geocode text location {raw_text!r}: {exc}")
        self.stats["failed"] += 1
        return None


def first_present(*values: Any) -> Any:
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


def safe_read_text(path: Path, max_chars: int = 10000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception:
        return ""


def load_sensor_coverage_module() -> Any:
    """Import evaluation.labelling.sensor_coverage.py with local fallbacks."""
    module_names = [
        "evaluation.labelling.sensor_coverage",
        "sensor_coverage",
    ]
    errors = []
    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: {exc!r}")
    raise ImportError("Could not import sensor_coverage. Tried: " + "; ".join(errors))


def maybe_load_sensor_coverage_module() -> Optional[Any]:
    try:
        return load_sensor_coverage_module()
    except Exception as exc:
        log(f"WARNING: sensor_coverage.py helpers are unavailable: {exc}")
        return None


def source_folder_names(source: str) -> List[str]:
    """Return possible folder names for a logical data source."""
    names = SOURCE_FOLDER_ALIASES.get(source, [source])
    # Preserve order while removing duplicates.
    return list(dict.fromkeys([source, *names]))


def maybe_unwrap_duplicate_date_folder(candidate: Path, date_str: str) -> Path:
    """Handle archives that extracted as <source>/<date>/<date>/... .

    Some tar files contain a top-level date directory.  If the staging script
    already extracts into <source>/<date>, the actual data may land one level
    deeper at <source>/<date>/<date>.  This helper unwraps that duplicate layer
    when it is the only child directory and there are no direct files.
    """
    nested = candidate / date_str
    if not nested.is_dir():
        return candidate

    try:
        children = list(candidate.iterdir())
    except OSError:
        return candidate

    has_direct_files = any(child.is_file() for child in children)
    child_dirs = [child for child in children if child.is_dir()]
    if not has_direct_files and len(child_dirs) == 1 and child_dirs[0].name == date_str:
        return nested
    return candidate


def infer_dates_under_folder(folder: Path) -> List[str]:
    """Best-effort YYYYMMDD discovery from filenames and small text payloads."""
    found: set[str] = set()
    date_patterns = [
        re.compile(r"(?<!\d)(20\d{6})(?!\d)"),
        re.compile(r"(?<!\d)(20\d{2})[-_](\d{2})[-_](\d{2})(?!\d)"),
    ]

    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        haystacks = [path.name]
        # Only peek into small, likely text-like files to avoid expensive scans.
        if path.suffix.lower() in {".json", ".jsonl", ".csv", ".txt"}:
            try:
                if path.stat().st_size <= 512_000:
                    haystacks.append(path.read_text(encoding="utf-8", errors="replace")[:512_000])
            except Exception:
                pass
        for haystack in haystacks:
            for match in date_patterns[0].finditer(haystack):
                found.add(match.group(1))
            for match in date_patterns[1].finditer(haystack):
                found.add("".join(match.groups()))
    return sorted(found)


def source_root_looks_like_flat_weather_layout(source_root: Path) -> bool:
    """Return True for layouts like evaluation/temp/weather_data/<location_name>/... ."""
    if not source_root.exists() or not source_root.is_dir():
        return False

    try:
        child_dirs = [child for child in source_root.iterdir() if child.is_dir()]
    except OSError:
        return False

    if not child_dirs:
        return False

    # The weather folders the user described are city strings such as
    # "Pasadena, US", not YYYYMMDD date folders.  Treat that as the canonical
    # real-weather layout.
    return any(("," in child.name or not is_date_string(child.name)) for child in child_dirs)


def resolve_source_date_folder(temp_root: Path, source: str, date_str: str) -> Optional[Path]:
    for folder_name in source_folder_names(source):
        candidate = temp_root / folder_name / date_str
        if candidate.exists() and candidate.is_dir():
            return maybe_unwrap_duplicate_date_folder(candidate, date_str)

    # Weather real data may be stored as:
    #   evaluation/temp/weather_data/<location_name>/...
    # instead of:
    #   evaluation/temp/weather_data/<YYYYMMDD>/<location_name>/...
    # In that case return the source root and let the weather parser group files
    # by the first location folder.
    if source == "weather_data":
        for folder_name in source_folder_names(source):
            source_root = temp_root / folder_name
            if source_root_looks_like_flat_weather_layout(source_root):
                return source_root

    return None


def discover_dates_for_source(temp_root: Path, source: str) -> List[str]:
    dates: set[str] = set()
    for folder_name in source_folder_names(source):
        source_root = temp_root / folder_name
        if not source_root.exists():
            continue
        dates.update(child.name for child in source_root.iterdir() if child.is_dir() and is_date_string(child.name))

        # Flat weather folders may not include a date folder at all.  Try to
        # infer dates from filenames/payload timestamps, but only as a fallback.
        if source == "weather_data" and not dates and source_root_looks_like_flat_weather_layout(source_root):
            inferred = infer_dates_under_folder(source_root)
            dates.update(inferred)
    return sorted(dates)


def resolve_dates(temp_root: Path, sources: Sequence[str], requested_dates: Sequence[str]) -> List[str]:
    if requested_dates:
        return sorted(set(requested_dates))

    discovered: set[str] = set()
    for source in sources:
        discovered.update(discover_dates_for_source(temp_root, source))
    return sorted(discovered)


def percentile(sorted_values: Sequence[int], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    weight = pos - lo
    return float(sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight)


def summarize_counts(sensor_record_counts: Dict[str, int]) -> Dict[str, Any]:
    counts = list(sensor_record_counts.values())
    sorted_counts = sorted(counts)
    if not counts:
        return {
            "num_sensors": 0,
            "total_records": 0,
            "records_per_sensor_avg": 0.0,
            "records_per_sensor_median": 0.0,
            "records_per_sensor_min": 0,
            "records_per_sensor_max": 0,
            "records_per_sensor_p90": 0.0,
        }

    return {
        "num_sensors": len(counts),
        "total_records": int(sum(counts)),
        "records_per_sensor_avg": float(statistics.mean(counts)),
        "records_per_sensor_median": float(statistics.median(counts)),
        "records_per_sensor_min": int(min(counts)),
        "records_per_sensor_max": int(max(counts)),
        "records_per_sensor_p90": percentile(sorted_counts, 0.90),
    }


def top_sensor_counts(sensor_record_counts: Dict[str, int], n: int = 20) -> List[Dict[str, Any]]:
    return [
        {"sensor_id": sensor_id, "records": count}
        for sensor_id, count in Counter(sensor_record_counts).most_common(n)
    ]


# ---------------------------------------------------------------------------
# Timestamp helpers for temporal replay
# ---------------------------------------------------------------------------


_COMMON_TIME_FORMATS = [
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y%m%d_%H%M%S",
    "%Y%m%d%H%M%S",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y%m%d",
]


def parse_datetime(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)

    text = str(value).strip()
    if not text:
        return None

    # Normalize common ISO variants.
    iso_text = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso_text)
        return dt.replace(tzinfo=None)
    except ValueError:
        pass

    for fmt in _COMMON_TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    return None


def normalize_report_date(value: Any, fallback_date_str: Optional[str] = None) -> Optional[str]:
    dt = parse_datetime(value)
    if dt is not None:
        return dt.isoformat(timespec="seconds")
    if fallback_date_str:
        return date_str_to_midnight_iso(fallback_date_str)
    return None


def infer_timestamp_from_path(path: Path, fallback_date_str: Optional[str] = None) -> Optional[str]:
    """Best-effort timestamp inference from filenames or fallback YYYYMMDD."""
    name = path.name

    patterns = [
        # 20250102_134500 or 20250102-134500
        r"(?P<date>20\d{6})[_\-T]?(?P<hms>\d{6})",
        # 2025-01-02_13-45-00 or 2025_01_02_13_45_00
        r"(?P<year>20\d{2})[-_](?P<mon>\d{2})[-_](?P<day>\d{2})[T_\- ](?P<h>\d{2})[-_:](?P<m>\d{2})(?:[-_:](?P<s>\d{2}))?",
    ]

    for pattern in patterns:
        match = re.search(pattern, name)
        if not match:
            continue
        groupdict = match.groupdict()
        try:
            if "date" in groupdict and groupdict.get("date"):
                date = groupdict["date"]
                hms = groupdict["hms"]
                dt = datetime(
                    int(date[:4]), int(date[4:6]), int(date[6:8]),
                    int(hms[:2]), int(hms[2:4]), int(hms[4:6]),
                )
                return dt.isoformat(timespec="seconds")
            dt = datetime(
                int(groupdict["year"]),
                int(groupdict["mon"]),
                int(groupdict["day"]),
                int(groupdict["h"]),
                int(groupdict["m"]),
                int(groupdict.get("s") or 0),
            )
            return dt.isoformat(timespec="seconds")
        except Exception:
            continue

    return date_str_to_midnight_iso(fallback_date_str) if fallback_date_str else None


def pems_hour_bucket(timestamp_value: Any, fallback_date_str: str) -> Optional[str]:
    dt = parse_datetime(timestamp_value)
    if dt is None:
        fallback = parse_datetime(fallback_date_str)
        if fallback is None:
            return None
        dt = fallback
    dt = dt.replace(minute=0, second=0, microsecond=0)
    return dt.isoformat(timespec="seconds")


def report_sort_key(report: Dict[str, Any]) -> Tuple[datetime, str, str, str]:
    metadata = report.get("metadata") if isinstance(report.get("metadata"), dict) else {}
    fallback_date_str = metadata.get("date_str") if isinstance(metadata, dict) else None
    dt = parse_datetime(report.get("report_date"))
    if dt is None and fallback_date_str:
        dt = parse_datetime(fallback_date_str)
    if dt is None:
        dt = datetime.max
    return (
        dt,
        str(report.get("sensor_type") or ""),
        str(report.get("sensor_id") or ""),
        str(report.get("report_id") or ""),
    )


# ---------------------------------------------------------------------------
# Normalized report schema helpers
# ---------------------------------------------------------------------------


def make_report(
    *,
    report_id: str,
    report_date: Optional[str],
    sensor_id: str,
    sensor_name: Optional[str],
    sensor_type: str,
    latitude: Optional[float],
    longitude: Optional[float],
    data: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    report = {
        "report_id": report_id,
        "report_date": report_date,
        "sensor_id": str(sensor_id),
        "sensor_name": str(sensor_name if sensor_name is not None else sensor_id),
        "sensor_type": sensor_type,
        "location": {"latitude": latitude, "longitude": longitude},
        "data": data,
        "metadata": metadata,
    }
    report["metadata"] = {key: value for key, value in metadata.items() if value is not None}
    return normalize_report(report)


def point_in_bounds(
    latitude: Optional[float],
    longitude: Optional[float],
    bounds: Optional[Dict[str, float]],
) -> bool:
    """Return True only if a point is present and within the configured bbox."""
    if bounds is None:
        return True
    if latitude is None or longitude is None:
        return False
    return (
        float(bounds["lat_min"]) <= latitude <= float(bounds["lat_max"])
        and float(bounds["lon_min"]) <= longitude <= float(bounds["lon_max"])
    )


def normalize_bearing_degrees(value: Any) -> Optional[float]:
    """Normalize a bearing-like value to [0, 360), returning None on failure."""
    bearing = as_float(value)
    if bearing is None or not math.isfinite(bearing):
        return None
    return bearing % 360.0


def svg_angle_to_compass_bearing_local(svg_angle: Any) -> Optional[float]:
    """Convert AlertCalifornia SVG/math-style angles into compass bearings.

    This mirrors sensor_coverage.svg_angle_to_compass_bearing():
      SVG/math angle 0=east, 90=north -> compass 0=north, 90=east.
    """
    angle = as_float(svg_angle)
    if angle is None or not math.isfinite(angle):
        return None
    return (90.0 - angle) % 360.0


def alertcalifornia_direction_to_bearing(
    raw_direction: Any,
    sensor_coverage: Optional[Any],
    *,
    direction_is_svg_angle: bool = True,
) -> Optional[float]:
    """Convert an AlertCalifornia .location/.direction value to compass bearing."""
    if raw_direction is None:
        return None
    if not direction_is_svg_angle:
        return normalize_bearing_degrees(raw_direction)
    if sensor_coverage is not None:
        try:
            return normalize_bearing_degrees(sensor_coverage.svg_angle_to_compass_bearing(raw_direction))
        except Exception:
            pass
    return svg_angle_to_compass_bearing_local(raw_direction)


def destination_point(
    latitude: float,
    longitude: float,
    bearing_degrees: float,
    distance_km: float,
) -> Dict[str, float]:
    """Return the lat/lon reached by moving distance_km along a bearing.

    Uses a spherical-earth destination-point formula.  This keeps the emitter
    independent of Shapely/GeoManager imports while matching the camera-cone
    geometry concept used by sensor_coverage.py.
    """
    lat1 = math.radians(float(latitude))
    lon1 = math.radians(float(longitude))
    bearing = math.radians(float(bearing_degrees) % 360.0)
    angular_distance = float(distance_km) / EARTH_RADIUS_KM

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
    )
    lon2 = (lon2 + math.pi) % (2.0 * math.pi) - math.pi
    return {"latitude": math.degrees(lat2), "longitude": math.degrees(lon2)}


def camera_view_triangle_metadata(
    *,
    camera_latitude: Optional[float],
    camera_longitude: Optional[float],
    bearing_degrees: Optional[float],
    fov_degrees: float,
    distance_km: float,
    method: str = "view_triangle_centroid",
) -> Optional[Dict[str, Any]]:
    """Build view-triangle vertices and centroid for a camera observation.

    The emitted report location should be the centroid of the viewed triangle.
    The physical camera mount point is preserved in metadata.
    """
    lat = as_float(camera_latitude)
    lon = as_float(camera_longitude)
    bearing = normalize_bearing_degrees(bearing_degrees)
    if lat is None or lon is None or bearing is None:
        return None

    fov = float(fov_degrees)
    distance = float(distance_km)
    left_bearing = (bearing - fov / 2.0) % 360.0
    right_bearing = (bearing + fov / 2.0) % 360.0

    camera = {"latitude": lat, "longitude": lon}
    left_edge = destination_point(lat, lon, left_bearing, distance)
    right_edge = destination_point(lat, lon, right_bearing, distance)

    # For these local/regional camera cones, the arithmetic centroid of the
    # lat/lon triangle vertices is adequate and avoids a Shapely dependency.
    centroid = {
        "latitude": (camera["latitude"] + left_edge["latitude"] + right_edge["latitude"]) / 3.0,
        "longitude": (camera["longitude"] + left_edge["longitude"] + right_edge["longitude"]) / 3.0,
    }

    return {
        "observation_latitude": centroid["latitude"],
        "observation_longitude": centroid["longitude"],
        "observation_location_method": method,
        "camera_latitude": lat,
        "camera_longitude": lon,
        "camera_bearing_degrees": bearing,
        "camera_fov_degrees": fov,
        "camera_view_distance_km": distance,
        "camera_left_edge_bearing_degrees": left_bearing,
        "camera_right_edge_bearing_degrees": right_bearing,
        "camera_view_triangle_vertices": {
            "camera": camera,
            "left_edge": left_edge,
            "right_edge": right_edge,
            "centroid": centroid,
        },
    }


def choose_alertcalifornia_camera_view_metadata(
    camera_folder: Path,
    camera_name: str,
    base_latitude: Optional[float],
    base_longitude: Optional[float],
    sensor_coverage: Optional[Any],
    *,
    fov_degrees: float = DEFAULT_ALERTCALIFORNIA_FOV_DEG,
    distance_km: float = DEFAULT_ALERTCALIFORNIA_VIEW_DISTANCE_KM,
) -> Optional[Dict[str, Any]]:
    """Resolve AlertCalifornia camera view centroid from .location/.direction files.

    .location files contain lat, lon, direction.  .direction files contain only
    direction, so they are paired with the saved/location-derived camera lat/lon.
    """
    candidate_files = sorted(camera_folder.glob("*.location")) + sorted(camera_folder.glob("*.direction"))
    for metadata_file in candidate_files:
        lat, lon, raw_direction = parse_alertcalifornia_location_or_direction_file(metadata_file)
        camera_lat = lat if lat is not None else base_latitude
        camera_lon = lon if lon is not None else base_longitude
        bearing = alertcalifornia_direction_to_bearing(raw_direction, sensor_coverage, direction_is_svg_angle=True)
        view = camera_view_triangle_metadata(
            camera_latitude=camera_lat,
            camera_longitude=camera_lon,
            bearing_degrees=bearing,
            fov_degrees=fov_degrees,
            distance_km=distance_km,
        )
        if view is not None:
            view.update({
                "camera_view_metadata_file": str(metadata_file),
                "camera_raw_direction": raw_direction,
                "camera_direction_format": "alertcalifornia_svg_angle_converted_to_compass_bearing",
            })
            return view

    # If we have a mount point but no direction, keep the old point behavior and
    # explain why.  The caller will use the mount point as a fallback.
    return None


def maybe_inject_california_local(location_name: str) -> str:
    """Convert city folders like 'Pasadena, US' into 'Pasadena, California, US'."""
    text = str(location_name).strip()
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) < 2:
        return text

    last = parts[-1].lower()
    if last not in {"us", "usa", "united states", "united states of america"}:
        return text

    middle = [part.lower() for part in parts[1:-1]]
    if any(part in {"california", "ca"} for part in middle):
        return text

    return ", ".join(parts[:-1] + ["California", parts[-1]])


def weather_location_query_from_name(
    location_name: str,
    sensor_coverage: Optional[Any],
    *,
    inject_california: bool = True,
) -> str:
    """Build the area/city query used to represent weather spatial support."""
    if not inject_california:
        return str(location_name)

    if sensor_coverage is not None:
        try:
            return str(sensor_coverage.maybe_inject_california(location_name))
        except Exception:
            pass

    return maybe_inject_california_local(location_name)


def weather_area_location_object(area_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Return a non-point location object for city/area weather reports.

    observation_model.py already includes report["location"] and report["data"]
    in the compact prompt it sends to the LLM.  By placing the weather
    location query directly in report["location"] (and not only in metadata),
    we can keep the observation model unchanged while still giving it the
    area context it needs.
    """
    return {
        "latitude": None,
        "longitude": None,
        "location_name": area_metadata.get("location_name"),
        "location_description": area_metadata.get("location_description"),
        "location_query": area_metadata.get("location_query"),
        "weather_location_representation": area_metadata.get(
            "weather_location_representation",
            "area_description",
        ),
    }


def add_weather_area_context_to_payload(
    payload: Dict[str, Any],
    area_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Expose weather area context in data as well as location.

    The observation model filters metadata down to a small safe list before LLM
    prompting, so metadata-only fields may not be visible.  Keeping these
    non-leaky fields in data makes weather reports self-contained.
    """
    enriched = dict(payload)
    for key in (
        "location_name",
        "location_description",
        "location_query",
        "weather_location_representation",
    ):
        value = area_metadata.get(key)
        if value is not None and value != "":
            enriched[key] = value
    return enriched


@dataclass
class DateProfile:
    source: str
    date_str: str
    date_folder: str
    sensor_record_counts: Dict[str, int] = field(default_factory=dict)
    sensor_locations: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    sensor_names: Dict[str, str] = field(default_factory=dict)
    file_counts_by_extension: Dict[str, int] = field(default_factory=dict)
    invalid_records: int = 0
    warnings: List[str] = field(default_factory=list)
    sample_reports: List[Dict[str, Any]] = field(default_factory=list)
    emission_reports: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def add_report(self, report: Dict[str, Any], *, sample_limit: int, collect_for_replay: bool) -> None:
        if len(self.sample_reports) < sample_limit:
            self.sample_reports.append(report)
        if collect_for_replay:
            self.emission_reports.append(report)

    def to_json(self, top_n: int = 20) -> Dict[str, Any]:
        return {
            "source": self.source,
            "date_str": self.date_str,
            "date_folder": self.date_folder,
            **summarize_counts(self.sensor_record_counts),
            "top_sensors_by_records": top_sensor_counts(self.sensor_record_counts, n=top_n),
            "num_sensor_locations": len(self.sensor_locations),
            "file_counts_by_extension": dict(sorted(self.file_counts_by_extension.items())),
            "invalid_records": self.invalid_records,
            "warnings": self.warnings,
            "extra": self.extra,
        }


# ---------------------------------------------------------------------------
# Source-specific parsing/profile functions
# ---------------------------------------------------------------------------


def profile_air_data(
    date_folder: Path,
    source: str,
    date_str: str,
    *,
    sample_limit: int,
    collect_for_replay: bool,
) -> DateProfile:
    profile = DateProfile(source=source, date_str=date_str, date_folder=str(date_folder))
    csv_files = sorted(date_folder.rglob("*.csv"))
    profile.file_counts_by_extension[".csv"] = len(csv_files)

    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row_index, row in enumerate(reader):
                if not row:
                    continue
                if len(row) < 4:
                    profile.invalid_records += 1
                    continue

                sensor_id = str(row[0]).strip()
                lat = as_float(row[1])
                lon = as_float(row[2])
                pm25 = coerce_scalar(row[3])
                if not sensor_id or lat is None or lon is None:
                    profile.invalid_records += 1
                    continue

                profile.sensor_record_counts[sensor_id] = profile.sensor_record_counts.get(sensor_id, 0) + 1
                profile.sensor_locations.setdefault(sensor_id, {"latitude": lat, "longitude": lon})
                profile.sensor_names.setdefault(sensor_id, sensor_id)

                if collect_for_replay or len(profile.sample_reports) < sample_limit:
                    report_date = infer_timestamp_from_path(csv_path, fallback_date_str=date_str)
                    report = make_report(
                        report_id=f"real_{source}_{date_str}_{sensor_id}_{profile.sensor_record_counts[sensor_id]}",
                        report_date=report_date,
                        sensor_id=sensor_id,
                        sensor_name=sensor_id,
                        sensor_type=source,
                        latitude=lat,
                        longitude=lon,
                        data={"pm25": pm25, "raw_row": row},
                        metadata={
                            "date_str": date_str,
                            "source_file": str(csv_path),
                            "source_row_index": row_index,
                            "report_timestamp_source": "air_csv_filename",
                            "real_data": True,
                        },
                    )
                    profile.add_report(report, sample_limit=sample_limit, collect_for_replay=collect_for_replay)

    return profile


def build_alertcalifornia_location_lookup(
    date_folder: Path,
    sensor_coverage: Optional[Any],
    locations_path: Path,
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    lookup: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    if locations_path.exists() and sensor_coverage is not None:
        try:
            raw_lookup = sensor_coverage.load_alertcalifornia_camera_locs(locations_path)
            lookup.update({camera: (lat, lon) for camera, (lat, lon) in raw_lookup.items()})
        except Exception as exc:
            log(f"WARNING: could not load AlertCalifornia saved locations from {locations_path}: {exc}")

    for camera_folder in sorted(date_folder.iterdir()):
        if not camera_folder.is_dir():
            continue
        for location_file in sorted(camera_folder.glob("*.location")):
            try:
                parts = [x.strip() for x in location_file.read_text(encoding="utf-8").strip().split(",")]
                if len(parts) >= 2:
                    lat = as_float(parts[0])
                    lon = as_float(parts[1])
                    if lat is not None and lon is not None:
                        lookup[camera_folder.name] = (lat, lon)
                        break
            except Exception:
                continue

    return lookup


def parse_alertcalifornia_location_or_direction_file(path: Path) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        parts = [x.strip() for x in path.read_text(encoding="utf-8").strip().split(",")]
        if path.suffix.lower() == ".location":
            lat = as_float(parts[0]) if len(parts) >= 1 else None
            lon = as_float(parts[1]) if len(parts) >= 2 else None
            direction = as_float(parts[2]) if len(parts) >= 3 else None
            return lat, lon, direction
        if path.suffix.lower() == ".direction":
            direction = as_float(parts[-1]) if parts else None
            return None, None, direction
    except Exception:
        pass
    return None, None, None


def profile_alertcalifornia(
    date_folder: Path,
    source: str,
    date_str: str,
    *,
    sample_limit: int,
    collect_for_replay: bool,
    sensor_coverage: Optional[Any],
    alertcalifornia_locations_path: Path,
) -> DateProfile:
    profile = DateProfile(source=source, date_str=date_str, date_folder=str(date_folder))
    location_lookup = build_alertcalifornia_location_lookup(date_folder, sensor_coverage, alertcalifornia_locations_path)

    camera_folders = [child for child in sorted(date_folder.iterdir()) if child.is_dir()]
    for camera_folder in camera_folders:
        camera_name = camera_folder.name
        all_files = sorted(path for path in camera_folder.rglob("*") if path.is_file())
        if not all_files:
            continue

        # Only image files are observation records.  .location/.direction files
        # are sensor metadata used for lat/lon/direction, not emissions.
        files = [path for path in all_files if path.suffix.lower() in IMAGE_EXTENSIONS]
        ignored_non_image_files = len(all_files) - len(files)
        if not files:
            profile.extra["alertcalifornia_non_image_files_ignored"] = (
                profile.extra.get("alertcalifornia_non_image_files_ignored", 0) + ignored_non_image_files
            )
            continue

        profile.extra["alertcalifornia_non_image_files_ignored"] = (
            profile.extra.get("alertcalifornia_non_image_files_ignored", 0) + ignored_non_image_files
        )

        base_lat, base_lon = location_lookup.get(camera_name, (None, None))
        view_metadata = choose_alertcalifornia_camera_view_metadata(
            camera_folder,
            camera_name,
            base_lat,
            base_lon,
            sensor_coverage,
        )
        emitted_lat = view_metadata.get("observation_latitude") if view_metadata else base_lat
        emitted_lon = view_metadata.get("observation_longitude") if view_metadata else base_lon
        if emitted_lat is not None or emitted_lon is not None:
            profile.sensor_locations[camera_name] = {"latitude": emitted_lat, "longitude": emitted_lon}
        profile.sensor_names[camera_name] = camera_name
        if view_metadata is not None:
            profile.extra["alertcalifornia_camera_view_centroid_locations"] = (
                profile.extra.get("alertcalifornia_camera_view_centroid_locations", 0) + 1
            )
        elif base_lat is not None or base_lon is not None:
            profile.extra["alertcalifornia_camera_mount_location_fallbacks"] = (
                profile.extra.get("alertcalifornia_camera_mount_location_fallbacks", 0) + 1
            )

        file_iter = files
        for file_index, data_file in enumerate(file_iter):
            ext = data_file.suffix.lower() or "<no_ext>"
            profile.file_counts_by_extension[ext] = profile.file_counts_by_extension.get(ext, 0) + 1
            profile.sensor_record_counts[camera_name] = profile.sensor_record_counts.get(camera_name, 0) + 1

            if collect_for_replay or len(profile.sample_reports) < sample_limit:
                lat, lon = emitted_lat, emitted_lon
                metadata: Dict[str, Any] = {
                    "date_str": date_str,
                    "source_file": str(data_file),
                    "source_file_index": file_index,
                    "report_timestamp_source": "alertcalifornia_image_filename",
                    "modality": "image",
                    "real_data": True,
                }
                if view_metadata is not None:
                    metadata.update(view_metadata)
                else:
                    metadata.update({
                        "observation_location_method": "camera_mount_point_fallback_no_view_direction",
                        "camera_latitude": base_lat,
                        "camera_longitude": base_lon,
                    })
                data = {"image_filepath": str(data_file)}

                report = make_report(
                    report_id=f"real_{source}_{date_str}_{camera_name}_{profile.sensor_record_counts[camera_name]}",
                    report_date=infer_timestamp_from_path(data_file, fallback_date_str=date_str),
                    sensor_id=camera_name,
                    sensor_name=camera_name,
                    sensor_type=source,
                    latitude=lat,
                    longitude=lon,
                    data=data,
                    metadata=metadata,
                )
                profile.add_report(report, sample_limit=sample_limit, collect_for_replay=collect_for_replay)

    profile.extra["locations_path"] = str(alertcalifornia_locations_path)
    return profile


def normalize_camera_key_fallback(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def normalize_cctv_key(value: str, sensor_coverage: Optional[Any]) -> str:
    """Normalize CCTV folder/KML names with the same spirit as sensor_coverage.py.

    This deliberately removes punctuation, so both of these collapse to the
    same key:

        I-10 - (101) Mcclure West
        I-10 : (101) Mcclure West
    """
    if sensor_coverage is not None:
        try:
            return str(sensor_coverage.normalize_camera_key(value))
        except Exception:
            pass
    return normalize_camera_key_fallback(value)


def cctv_name_variants(name: str) -> List[str]:
    """Generate common CCTV folder/KML spelling variants.

    The pulled folders often sanitize the Caltrans KML colon separator as a
    dash.  Example:

        folder: I-10 - (101) Mcclure West
        KML:    I-10 : (101) Mcclure West
    """
    text = str(name).strip()
    variants = [text]

    replacements = [
        (" : ", " - "),
        (" - ", " : "),
        (":", "-"),
        ("-", ":"),
    ]
    for old, new in replacements:
        if old in text:
            variants.append(text.replace(old, new))

    # Specific variant for the common route/index separator just before the
    # parenthesized Caltrans camera index.
    variants.append(re.sub(r"\s+-\s+(?=\()", " : ", text))
    variants.append(re.sub(r"\s*:\s*(?=\()", " - ", text))

    # Deduplicate while preserving order.
    return list(dict.fromkeys(v for v in variants if v))


def build_cctv_lookup(sensor_coverage: Optional[Any], cctv_kml_path: Path) -> Dict[str, Dict[str, Any]]:
    if sensor_coverage is None or not cctv_kml_path.exists():
        return {}
    try:
        cctv_lookup, entries = sensor_coverage.parse_cctv_kml(cctv_kml_path, include_only_in_service=False)

        # Add explicit aliases for sanitized folder names.  parse_cctv_kml()
        # already normalizes many names, but adding these aliases makes the
        # match robust when folders use " - " and KML uses " : ".
        enhanced_lookup = dict(cctv_lookup)
        for entry in entries:
            for field in ("name", "image_slug"):
                value = entry.get(field)
                if not value:
                    continue
                for variant in cctv_name_variants(str(value)):
                    enhanced_lookup[normalize_cctv_key(variant, sensor_coverage)] = entry
        return enhanced_lookup
    except Exception as exc:
        log(f"WARNING: could not parse CCTV KML from {cctv_kml_path}: {exc}")
        return {}


def lookup_cctv_entry(camera_name: str, sensor_coverage: Optional[Any], cctv_lookup: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not cctv_lookup:
        return None

    for variant in cctv_name_variants(camera_name):
        key = normalize_cctv_key(variant, sensor_coverage)
        entry = cctv_lookup.get(key)
        if entry is not None:
            return entry

    return None


def profile_cctv(
    date_folder: Path,
    source: str,
    date_str: str,
    *,
    sample_limit: int,
    collect_for_replay: bool,
    sensor_coverage: Optional[Any],
    cctv_kml_path: Path,
    skip_unavailable_images: bool = DEFAULT_SKIP_CCTV_UNAVAILABLE_IMAGES,
    unavailable_signature: Optional[Dict[str, Any]] = None,
    unavailable_decision_cache: Optional[Dict[str, Dict[str, Any]]] = None,
    unavailable_decision_cache_path: Optional[Path] = None,
    use_unavailable_decision_cache: bool = True,
) -> DateProfile:
    profile = DateProfile(source=source, date_str=date_str, date_folder=str(date_folder))
    cctv_lookup = build_cctv_lookup(sensor_coverage, cctv_kml_path)
    unmatched = 0
    profile.extra["cctv_unavailable_filter_enabled"] = bool(skip_unavailable_images and unavailable_signature is not None)
    profile.extra["cctv_unavailable_decision_cache_enabled"] = bool(
        use_unavailable_decision_cache and unavailable_decision_cache is not None
    )
    if unavailable_decision_cache_path is not None:
        profile.extra["cctv_unavailable_decision_cache_path"] = str(unavailable_decision_cache_path)
    if unavailable_signature is not None:
        profile.extra["cctv_unavailable_reference_image"] = unavailable_signature.get("source_path")

    camera_folders = [child for child in sorted(date_folder.iterdir()) if child.is_dir()]
    for camera_folder in camera_folders:
        camera_name = camera_folder.name
        all_files = sorted(path for path in camera_folder.rglob("*") if path.is_file())
        image_files = [path for path in all_files if path.suffix.lower() in IMAGE_EXTENSIONS]
        ignored_non_image_files = len(all_files) - len(image_files)
        profile.extra["cctv_non_image_files_ignored"] = profile.extra.get("cctv_non_image_files_ignored", 0) + ignored_non_image_files
        files = image_files
        if not files:
            continue

        entry = lookup_cctv_entry(camera_name, sensor_coverage, cctv_lookup)
        view_metadata: Optional[Dict[str, Any]] = None
        if entry is None:
            unmatched += 1
            camera_lat = camera_lon = None
            lat = lon = None
            sensor_name = camera_name
        else:
            camera_lat = as_float(entry.get("lat"))
            camera_lon = as_float(entry.get("lon"))
            sensor_name = entry.get("name") or camera_name
            bearing = normalize_bearing_degrees(entry.get("bearing"))
            view_metadata = camera_view_triangle_metadata(
                camera_latitude=camera_lat,
                camera_longitude=camera_lon,
                bearing_degrees=bearing,
                fov_degrees=DEFAULT_CCTV_FOV_DEG,
                distance_km=DEFAULT_CCTV_VIEW_DISTANCE_KM,
            )
            if view_metadata is not None:
                view_metadata.update({
                    "camera_direction": entry.get("direction"),
                    "camera_direction_format": "cctv_kml_direction_compass_bearing",
                })
                lat = view_metadata["observation_latitude"]
                lon = view_metadata["observation_longitude"]
                profile.extra["cctv_camera_view_centroid_locations"] = (
                    profile.extra.get("cctv_camera_view_centroid_locations", 0) + 1
                )
            else:
                lat = camera_lat
                lon = camera_lon
                if camera_lat is not None or camera_lon is not None:
                    profile.extra["cctv_camera_mount_location_fallbacks"] = (
                        profile.extra.get("cctv_camera_mount_location_fallbacks", 0) + 1
                    )
            profile.sensor_locations[camera_name] = {"latitude": lat, "longitude": lon}

        profile.sensor_names[camera_name] = str(sensor_name)

        file_iter = files
        for file_index, data_file in enumerate(file_iter):
            ext = data_file.suffix.lower() or "<no_ext>"
            profile.file_counts_by_extension[ext] = profile.file_counts_by_extension.get(ext, 0) + 1
            profile.extra["cctv_image_files_seen"] = profile.extra.get("cctv_image_files_seen", 0) + 1

            if skip_unavailable_images and unavailable_signature is not None:
                if use_unavailable_decision_cache and unavailable_decision_cache is not None:
                    unavailable_like, unavailable_diag = compare_image_to_unavailable_signature_cached(
                        data_file,
                        unavailable_signature,
                        decision_cache=unavailable_decision_cache,
                        decision_cache_path=unavailable_decision_cache_path,
                    )
                else:
                    unavailable_like, unavailable_diag = compare_image_to_unavailable_signature(
                        data_file,
                        unavailable_signature,
                    )

                cache_status = unavailable_diag.get("cache_status")
                if cache_status == "hit":
                    profile.extra["cctv_unavailable_decision_cache_hits"] = (
                        profile.extra.get("cctv_unavailable_decision_cache_hits", 0) + 1
                    )
                elif cache_status == "miss_stored":
                    profile.extra["cctv_unavailable_decision_cache_misses"] = (
                        profile.extra.get("cctv_unavailable_decision_cache_misses", 0) + 1
                    )

                if unavailable_diag.get("status") != "ok":
                    key = f"cctv_unavailable_filter_{unavailable_diag.get('status', 'unknown')}"
                    profile.extra[key] = profile.extra.get(key, 0) + 1
                if unavailable_like:
                    profile.extra["cctv_unavailable_images_skipped"] = profile.extra.get("cctv_unavailable_images_skipped", 0) + 1
                    if len(profile.extra.get("cctv_unavailable_skip_examples", [])) < 20:
                        profile.extra.setdefault("cctv_unavailable_skip_examples", []).append({
                            "camera_name": camera_name,
                            "source_file": str(data_file),
                            "diagnostics": unavailable_diag,
                        })
                    continue

            profile.extra["cctv_images_emitted"] = profile.extra.get("cctv_images_emitted", 0) + 1
            profile.sensor_record_counts[camera_name] = profile.sensor_record_counts.get(camera_name, 0) + 1

            if collect_for_replay or len(profile.sample_reports) < sample_limit:
                data = {"image_filepath": str(data_file)}

                metadata = {
                    "date_str": date_str,
                    "source_file": str(data_file),
                    "source_file_index": file_index,
                    "report_timestamp_source": "cctv_image_filename",
                    "modality": "image",
                    "kml_path": str(cctv_kml_path),
                    "real_data": True,
                }
                if view_metadata is not None:
                    metadata.update(view_metadata)
                elif entry is not None:
                    metadata.update({
                        "observation_location_method": "camera_mount_point_fallback_no_kml_bearing",
                        "camera_latitude": camera_lat,
                        "camera_longitude": camera_lon,
                        "camera_direction": entry.get("direction"),
                    })

                report = make_report(
                    report_id=f"real_{source}_{date_str}_{camera_name}_{profile.sensor_record_counts[camera_name]}",
                    report_date=infer_timestamp_from_path(data_file, fallback_date_str=date_str),
                    sensor_id=camera_name,
                    sensor_name=str(sensor_name),
                    sensor_type=source,
                    latitude=lat,
                    longitude=lon,
                    data=data,
                    metadata=metadata,
                )
                profile.add_report(report, sample_limit=sample_limit, collect_for_replay=collect_for_replay)

    profile.extra["kml_path"] = str(cctv_kml_path)
    profile.extra["num_unmatched_camera_folders"] = unmatched
    return profile


def traffic_file_date(date_str: str) -> str:
    if len(date_str) != 8:
        return date_str
    return f"{date_str[:4]}_{date_str[4:6]}_{date_str[6:8]}"


def find_traffic_files(date_folder: Path, date_str: str) -> List[Path]:
    file_date = traffic_file_date(date_str)
    gz_files = sorted(date_folder.rglob(f"*{file_date}*.txt.gz"))
    txt_files = sorted(date_folder.rglob(f"*{file_date}*.txt"))
    if not gz_files and not txt_files:
        gz_files = sorted(date_folder.rglob("*.txt.gz"))
        txt_files = sorted(date_folder.rglob("*.txt"))
    return gz_files + txt_files


PEMS_STATION_FIELDS = [
    "ID",
    "Fwy",
    "Dir",
    "District",
    "County",
    "City",
    "State_PM",
    "Abs_PM",
    "Latitude",
    "Longitude",
    "Length",
    "Type",
    "Lanes",
    "Name",
    "User_ID_1",
    "User_ID_2",
    "User_ID_3",
    "User_ID_4",
]


def normalize_pems_station_id(value: Any) -> str:
    """Normalize station IDs from station metadata and 5-minute rows.

    PeMS rows usually store station IDs as strings like ``715910``.  This also
    handles values accidentally read as ``715910.0`` or surrounded by quotes.
    """
    text = str(value).strip().strip('"').strip("'")
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text


def _clean_fieldname(value: Any) -> str:
    return str(value or "").strip().lstrip("\ufeff")


def _field_lookup(row: Dict[str, Any]) -> Dict[str, str]:
    """Map lowercase field names to the original field names in a DictReader row."""
    return {_clean_fieldname(key).lower(): key for key in row.keys()}


def _get_row_value(row: Dict[str, Any], lookup: Dict[str, str], *names: str) -> Any:
    for name in names:
        key = lookup.get(name.lower())
        if key is not None:
            return row.get(key)
    return None


def station_record_from_values(values: Dict[str, Any], *, metadata_format: str) -> Optional[Dict[str, Any]]:
    station_id = normalize_pems_station_id(values.get("ID") or values.get("station_id"))
    if not station_id:
        return None

    lat = as_float(values.get("Latitude") or values.get("lat"))
    lon = as_float(values.get("Longitude") or values.get("lon"))
    if lat is None or lon is None:
        return None

    return {
        "station_id": station_id,
        "lat": lat,
        "lon": lon,
        "fwy": values.get("Fwy"),
        "direction": values.get("Dir"),
        "district": values.get("District"),
        "county": values.get("County"),
        "city": values.get("City"),
        "state_pm": values.get("State_PM"),
        "abs_pm": values.get("Abs_PM"),
        "length": values.get("Length"),
        "type": values.get("Type"),
        "lanes": values.get("Lanes"),
        "name": values.get("Name"),
        "user_id_1": values.get("User_ID_1"),
        "user_id_2": values.get("User_ID_2"),
        "user_id_3": values.get("User_ID_3"),
        "user_id_4": values.get("User_ID_4"),
        "station_metadata_format": metadata_format,
    }


def load_pems_station_locations_tsv(stations_path: Path) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Load a headered PeMS station metadata TSV.

    This is the authoritative location source for PeMS reports.  The 5-minute
    PeMS data files usually only contain station IDs, so each row must be
    cross-referenced against this metadata file to obtain lat/lon.
    """
    station_locations: Dict[str, Dict[str, Any]] = {}
    invalid_rows: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Any] = {
        "loader": "headered_tsv",
        "stations_path": str(stations_path),
        "fieldnames": [],
        "rows_read": 0,
    }

    with stations_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        diagnostics["fieldnames"] = [_clean_fieldname(x) for x in (reader.fieldnames or [])]

        for row_index, row in enumerate(reader):
            diagnostics["rows_read"] += 1
            lookup = _field_lookup(row)

            values = {
                "ID": _get_row_value(row, lookup, "ID", "station_id", "station"),
                "Fwy": _get_row_value(row, lookup, "Fwy", "freeway"),
                "Dir": _get_row_value(row, lookup, "Dir", "direction"),
                "District": _get_row_value(row, lookup, "District"),
                "County": _get_row_value(row, lookup, "County"),
                "City": _get_row_value(row, lookup, "City"),
                "State_PM": _get_row_value(row, lookup, "State_PM"),
                "Abs_PM": _get_row_value(row, lookup, "Abs_PM"),
                "Latitude": _get_row_value(row, lookup, "Latitude", "lat"),
                "Longitude": _get_row_value(row, lookup, "Longitude", "lon", "lng"),
                "Length": _get_row_value(row, lookup, "Length"),
                "Type": _get_row_value(row, lookup, "Type"),
                "Lanes": _get_row_value(row, lookup, "Lanes"),
                "Name": _get_row_value(row, lookup, "Name"),
                "User_ID_1": _get_row_value(row, lookup, "User_ID_1"),
                "User_ID_2": _get_row_value(row, lookup, "User_ID_2"),
                "User_ID_3": _get_row_value(row, lookup, "User_ID_3"),
                "User_ID_4": _get_row_value(row, lookup, "User_ID_4"),
            }

            station = station_record_from_values(values, metadata_format="headered_tab_delimited")
            if station is None:
                invalid_rows.append({
                    "row_index": row_index,
                    "row": row,
                    "reason": "missing station ID or invalid Latitude/Longitude",
                })
                continue

            station_locations[station["station_id"]] = station

    diagnostics["stations_loaded"] = len(station_locations)
    diagnostics["invalid_rows"] = len(invalid_rows)
    return station_locations, invalid_rows, diagnostics


def load_pems_station_locations_positional(stations_path: Path) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Fallback loader for PeMS station files using the standard column order.

    It works for truly headerless files and also for headered files if the
    headered parser could not be used for some reason.
    """
    station_locations: Dict[str, Dict[str, Any]] = {}
    invalid_rows: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Any] = {
        "loader": "positional_tsv_fallback",
        "stations_path": str(stations_path),
        "rows_read": 0,
    }

    with stations_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row_index, row in enumerate(reader):
            if not row:
                continue
            diagnostics["rows_read"] += 1

            # Skip a header row if present.
            if row_index == 0 and row[0].strip().lower() in {"id", "station", "station_id"}:
                continue

            if len(row) < 10:
                invalid_rows.append({
                    "row_index": row_index,
                    "row": row,
                    "reason": "expected at least 10 tab-delimited columns",
                })
                continue

            values = {
                field: row[i].strip() if i < len(row) else None
                for i, field in enumerate(PEMS_STATION_FIELDS)
            }
            station = station_record_from_values(values, metadata_format="positional_tab_delimited")
            if station is None:
                invalid_rows.append({
                    "row_index": row_index,
                    "row": row,
                    "reason": "missing station ID or invalid columns 8/9 Latitude/Longitude",
                })
                continue

            station_locations[station["station_id"]] = station

    diagnostics["stations_loaded"] = len(station_locations)
    diagnostics["invalid_rows"] = len(invalid_rows)
    return station_locations, invalid_rows, diagnostics



def pems_station_cache_path(cache_dir: Path) -> Path:
    """Return the persistent cache path for parsed PeMS station metadata."""
    return Path(cache_dir) / "pem_data_station_5min" / DEFAULT_PEMS_STATION_CACHE_FILENAME


def _path_stat_signature(path: Path) -> Dict[str, Any]:
    """Return a small invalidation signature for a metadata file."""
    try:
        stat = path.stat()
        return {
            "path": str(path),
            "exists": True,
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
        }
    except OSError:
        return {"path": str(path), "exists": False}


def _candidate_path_matches_signature(path: Path, signature: Dict[str, Any]) -> bool:
    """Return True if path still matches the cached metadata signature."""
    current = _path_stat_signature(path)
    return (
        bool(current.get("exists"))
        and str(current.get("path")) == str(signature.get("path"))
        and int(current.get("size_bytes", -1)) == int(signature.get("size_bytes", -2))
        and int(current.get("mtime_ns", -1)) == int(signature.get("mtime_ns", -2))
    )


def load_pems_station_locations_cache(
    cache_dir: Optional[Path],
    candidates: Sequence[Path],
) -> Optional[Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]]:
    """Load cached PeMS station metadata when the source file is unchanged."""
    if cache_dir is None:
        return None
    cache_path = pems_station_cache_path(Path(cache_dir))
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as infile:
            payload = json.load(infile)
        if not isinstance(payload, dict):
            return None
        if int(payload.get("cache_version", -1)) != PEMS_STATION_CACHE_VERSION:
            return None
        selected_path_text = str(payload.get("selected_path") or "")
        selected_signature = payload.get("selected_signature") if isinstance(payload.get("selected_signature"), dict) else {}
        matching_candidate: Optional[Path] = None
        for candidate in candidates:
            candidate = Path(candidate)
            if str(candidate) == selected_path_text and _candidate_path_matches_signature(candidate, selected_signature):
                matching_candidate = candidate
                break
        if matching_candidate is None:
            return None
        station_locations_raw = payload.get("station_locations")
        if not isinstance(station_locations_raw, dict):
            return None
        station_locations: Dict[str, Dict[str, Any]] = {
            normalize_pems_station_id(station_id): dict(record) if isinstance(record, dict) else {}
            for station_id, record in station_locations_raw.items()
        }
        diagnostics = dict(payload.get("diagnostics") or {})
        diagnostics.update({
            "selected_path": selected_path_text,
            "selected_loader": diagnostics.get("selected_loader") or payload.get("selected_loader") or "pems_station_cache",
            "stations_loaded": len(station_locations),
            "loaded_from_pems_station_cache": True,
            "pems_station_cache_path": str(cache_path),
            "pems_station_cache_signature": selected_signature,
        })
        log(
            f"Loaded {len(station_locations)} PeMS station locations from cache "
            f"{cache_path} for {selected_path_text}."
        )
        return station_locations, diagnostics
    except Exception as exc:
        log(f"WARNING: could not load PeMS station metadata cache {cache_path}: {exc}")
        return None


def save_pems_station_locations_cache(
    cache_dir: Optional[Path],
    *,
    selected_path: Path,
    selected_loader: str,
    station_locations: Dict[str, Dict[str, Any]],
    diagnostics: Dict[str, Any],
) -> None:
    """Persist parsed PeMS station metadata for future dates/runs."""
    if cache_dir is None:
        return
    cache_path = pems_station_cache_path(Path(cache_dir))
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_version": PEMS_STATION_CACHE_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "selected_path": str(selected_path),
            "selected_loader": selected_loader,
            "selected_signature": _path_stat_signature(selected_path),
            "station_locations": station_locations,
            "diagnostics": diagnostics,
        }
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as outfile:
            json.dump(payload, outfile, ensure_ascii=False)
        tmp_path.replace(cache_path)
        log(f"Saved {len(station_locations)} PeMS station locations to cache {cache_path}.")
    except Exception as exc:
        log(f"WARNING: could not save PeMS station metadata cache {cache_path}: {exc}")


def load_pems_locations(
    sensor_coverage: Optional[Any],
    candidates: Sequence[Path],
    *,
    pems_station_cache_dir: Optional[Path] = None,
    pems_station_cache_enabled: bool = True,
    pems_station_cache_read: bool = True,
    pems_station_cache_write: bool = True,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Load PeMS station metadata from the first usable candidate path.

    The 5-minute data files are not expected to have locations.  This function
    loads locations from ``pem_7_stations.txt``/station metadata and later row
    parsing uses row[1] as the station-id join key.  Parsed station metadata is
    cached persistently because it is reused for every date and every baseline.
    """
    diagnostics: Dict[str, Any] = {
        "candidate_paths": [str(path) for path in candidates],
        "attempts": [],
        "selected_path": None,
        "selected_loader": None,
        "pems_station_cache_enabled": bool(pems_station_cache_enabled),
        "pems_station_cache_dir": str(pems_station_cache_dir) if pems_station_cache_dir else None,
    }

    if pems_station_cache_enabled and pems_station_cache_read:
        cached = load_pems_station_locations_cache(pems_station_cache_dir, candidates)
        if cached is not None:
            station_locations, cache_diag = cached
            diagnostics.update(cache_diag)
            return station_locations, diagnostics

    def _save_and_return(
        *,
        station_locations: Dict[str, Dict[str, Any]],
        selected_path: Path,
        selected_loader: str,
        invalid_rows_count: int,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        diagnostics["selected_path"] = str(selected_path)
        diagnostics["selected_loader"] = selected_loader
        diagnostics["invalid_rows"] = int(invalid_rows_count)
        diagnostics["stations_loaded"] = len(station_locations)
        diagnostics["loaded_from_pems_station_cache"] = False
        if pems_station_cache_enabled and pems_station_cache_write:
            save_pems_station_locations_cache(
                pems_station_cache_dir,
                selected_path=selected_path,
                selected_loader=selected_loader,
                station_locations=station_locations,
                diagnostics=diagnostics,
            )
        return station_locations, diagnostics

    for stations_path in candidates:
        stations_path = Path(stations_path)
        attempt: Dict[str, Any] = {"path": str(stations_path), "exists": stations_path.exists()}
        diagnostics["attempts"].append(attempt)
        if not stations_path.exists():
            continue

        # Prefer our local explicit parser so this file works even if importing
        # sensor_coverage.py fails due unrelated project dependencies.
        try:
            station_locations, invalid_rows, loader_diag = load_pems_station_locations_tsv(stations_path)
            attempt["headered_tsv"] = loader_diag
            if station_locations:
                log(
                    f"Loaded {len(station_locations)} PeMS station locations from {stations_path} "
                    "using headered TSV parser."
                )
                return _save_and_return(
                    station_locations=station_locations,
                    selected_path=stations_path,
                    selected_loader="headered_tsv",
                    invalid_rows_count=len(invalid_rows),
                )
        except Exception as exc:
            attempt["headered_tsv_error"] = repr(exc)

        # The project helper should behave similarly; keep it as another option.
        if sensor_coverage is not None:
            try:
                station_locations, invalid_rows = sensor_coverage.load_pems_station_locations(stations_path)
                normalized_locations = {normalize_pems_station_id(k): v for k, v in station_locations.items()}
                attempt["sensor_coverage_loader"] = {
                    "stations_loaded": len(normalized_locations),
                    "invalid_rows": len(invalid_rows),
                }
                if normalized_locations:
                    log(
                        f"Loaded {len(normalized_locations)} PeMS station locations from {stations_path} "
                        "using sensor_coverage.py."
                    )
                    return _save_and_return(
                        station_locations=normalized_locations,
                        selected_path=stations_path,
                        selected_loader="sensor_coverage.load_pems_station_locations",
                        invalid_rows_count=len(invalid_rows),
                    )
            except Exception as exc:
                attempt["sensor_coverage_loader_error"] = repr(exc)

        try:
            station_locations, invalid_rows, loader_diag = load_pems_station_locations_positional(stations_path)
            attempt["positional_tsv_fallback"] = loader_diag
            if station_locations:
                log(
                    f"Loaded {len(station_locations)} PeMS station locations from {stations_path} "
                    "using positional TSV fallback."
                )
                return _save_and_return(
                    station_locations=station_locations,
                    selected_path=stations_path,
                    selected_loader="positional_tsv_fallback",
                    invalid_rows_count=len(invalid_rows),
                )
        except Exception as exc:
            attempt["positional_tsv_fallback_error"] = repr(exc)

    log("WARNING: no usable PeMS station metadata file was found; PeMS rows with bounds enabled will be skipped.")
    return {}, diagnostics


def iter_csv_rows_maybe_gzip(path: Path) -> Iterator[Tuple[int, List[str]]]:
    if path.name.endswith(".gz"):
        opener = lambda p: gzip.open(p, "rt", encoding="utf-8", newline="")  # noqa: E731
    else:
        opener = lambda p: open(p, "r", encoding="utf-8", newline="")  # noqa: E731

    with opener(path) as f:
        reader = csv.reader(f)
        for row_index, row in enumerate(reader):
            if row:
                yield row_index, row


PEMS_ROW_COLUMNS = {
    # PeMS station 5-minute rows are positional and do not contain locations.
    # The location comes from pem_7_stations.txt, joined by station_id=row[1].
    # These zero-based indices correspond to:
    #   timestamp (0), station id (1), avg occupancy (10), avg speed (11)
    "timestamp": 0,
    "station_id": 1,
    "district": 2,
    "freeway": 3,
    "direction": 4,
    "data_total_flow": 9,
    "data_avg_occupancy": 10,
    "data_avg_speed": 11,
}


def row_value(row: Sequence[Any], index: int) -> Any:
    return row[index] if index < len(row) else None


def is_missing_measurement(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"na", "n/a", "none", "null", "nan"}


def parse_numeric_measurement(value: Any) -> Optional[float]:
    """Return a finite numeric measurement, or None when the field is missing/unusable."""
    if is_missing_measurement(value):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def pems_required_measurements(row: Sequence[Any]) -> Tuple[Optional[float], Optional[float]]:
    """Return required PeMS traffic measurements: avg occupancy and avg speed.

    Rows missing either field are ignored for profiling and emission because a
    traffic report without these measurements is not useful downstream.
    """
    avg_occupancy = parse_numeric_measurement(row_value(row, PEMS_ROW_COLUMNS["data_avg_occupancy"]))
    avg_speed = parse_numeric_measurement(row_value(row, PEMS_ROW_COLUMNS["data_avg_speed"]))
    return avg_occupancy, avg_speed


def pems_row_has_required_traffic_measurements(row: Sequence[Any]) -> bool:
    avg_occupancy, avg_speed = pems_required_measurements(row)
    return avg_occupancy is not None and avg_speed is not None


def pems_row_to_data(row: List[str], station: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Format a PeMS row like the synthetic traffic reports.

    The synthetic emitter typically exposes traffic measurements as named fields
    such as ``station_id``, ``data_avg_occupancy``, and ``data_avg_speed`` rather
    than an opaque ``raw_row``.  Keep the row timestamp in report_date/metadata,
    not inside report["data"], because the downstream report schema already has
    a top-level report_date.
    """
    station = station or {}
    station_id = normalize_pems_station_id(row_value(row, PEMS_ROW_COLUMNS["station_id"]))

    data: Dict[str, Any] = {
        "station_id": station_id,
    }

    avg_occupancy, avg_speed = pems_required_measurements(row)
    total_flow = parse_numeric_measurement(row_value(row, PEMS_ROW_COLUMNS["data_total_flow"]))

    # profile_traffic() skips rows missing either required measurement before
    # this function is called, so these fields should normally be present.
    if avg_occupancy is not None:
        data["data_avg_occupancy"] = avg_occupancy
    if avg_speed is not None:
        data["data_avg_speed"] = avg_speed
    if total_flow is not None:
        data["data_total_flow"] = total_flow

    # Optional context fields that are commonly useful downstream and mirror the
    # synthetic traffic reports when station metadata is available.
    lanes = station.get("lanes")
    if lanes not in {None, ""}:
        data["lanes"] = coerce_scalar(lanes)

    fwy = station.get("fwy") or row_value(row, PEMS_ROW_COLUMNS["freeway"])
    direction = station.get("direction") or row_value(row, PEMS_ROW_COLUMNS["direction"])
    if fwy not in {None, ""}:
        data["fwy"] = coerce_scalar(fwy)
    if direction not in {None, ""}:
        data["direction"] = direction

    return data


def profile_traffic(
    date_folder: Path,
    source: str,
    date_str: str,
    *,
    sample_limit: int,
    collect_for_replay: bool,
    sensor_coverage: Optional[Any],
    pems_station_path_candidates: Sequence[Path],
    downsample_to_hourly: bool,
    spatial_bounds: Optional[Dict[str, float]],
    require_spatial_bounds: bool,
    pems_station_cache_dir: Optional[Path] = None,
    pems_station_cache_enabled: bool = True,
    pems_station_cache_read: bool = True,
    pems_station_cache_write: bool = True,
) -> DateProfile:
    profile = DateProfile(source=source, date_str=date_str, date_folder=str(date_folder))
    station_locations, station_loader_diagnostics = load_pems_locations(
        sensor_coverage,
        pems_station_path_candidates,
        pems_station_cache_dir=pems_station_cache_dir,
        pems_station_cache_enabled=pems_station_cache_enabled,
        pems_station_cache_read=pems_station_cache_read,
        pems_station_cache_write=pems_station_cache_write,
    )
    traffic_files = find_traffic_files(date_folder, date_str)
    profile.extra["traffic_files"] = [str(path) for path in traffic_files]
    profile.extra["num_station_locations_loaded"] = len(station_locations)
    profile.extra["pems_station_loader"] = station_loader_diagnostics
    profile.extra["pems_downsample_to_hourly"] = downsample_to_hourly
    profile.extra["pems_downsample_policy"] = "first_record_per_station_per_hour" if downsample_to_hourly else "none"
    profile.extra["pems_output_format"] = {
        "data_fields": ["station_id", "data_avg_occupancy", "data_avg_speed", "data_total_flow", "lanes", "fwy", "direction"],
        "row_indices_zero_based": PEMS_ROW_COLUMNS,
        "raw_row_in_data": False,
        "rows_missing_avg_occupancy_or_avg_speed": "skipped_not_counted_not_emitted",
        "location_source": "station metadata joined by station_id=row[1]",
    }
    profile.extra["pems_spatial_bounds_filter_enabled"] = require_spatial_bounds and spatial_bounds is not None
    profile.extra["pems_spatial_bounds"] = spatial_bounds
    if spatial_bounds is not None:
        in_bounds_station_ids = [
            station_id
            for station_id, station in station_locations.items()
            if point_in_bounds(as_float(station.get("lat")), as_float(station.get("lon")), spatial_bounds)
        ]
        profile.extra["num_station_locations_in_bounds"] = len(in_bounds_station_ids)

    seen_station_hour: set[Tuple[str, str]] = set()
    raw_rows_read = 0
    raw_valid_station_rows = 0
    skipped_by_hourly_downsample = 0
    skipped_by_spatial_bounds = 0
    skipped_missing_required_traffic_metrics = 0
    missing_station_location_for_bounds = 0
    missing_hour_bucket = 0
    missing_station_location_examples: List[str] = []
    out_of_bounds_station_examples: List[Dict[str, Any]] = []
    missing_required_traffic_metrics_examples: List[Dict[str, Any]] = []

    for traffic_path in traffic_files:
        ext = ".txt.gz" if traffic_path.name.endswith(".txt.gz") else traffic_path.suffix.lower()
        profile.file_counts_by_extension[ext] = profile.file_counts_by_extension.get(ext, 0) + 1

        try:
            row_iter = iter_csv_rows_maybe_gzip(traffic_path)
            for row_index, row in row_iter:
                raw_rows_read += 1
                if len(row) < 2:
                    profile.invalid_records += 1
                    continue

                station_id = normalize_pems_station_id(row[1])
                if not station_id:
                    profile.invalid_records += 1
                    continue
                raw_valid_station_rows += 1

                avg_occupancy, avg_speed = pems_required_measurements(row)
                if avg_occupancy is None or avg_speed is None:
                    skipped_missing_required_traffic_metrics += 1
                    if len(missing_required_traffic_metrics_examples) < 20:
                        missing_required_traffic_metrics_examples.append({
                            "station_id": station_id,
                            "source_file": str(traffic_path),
                            "source_row_index": row_index,
                            "raw_avg_occupancy_value": row_value(row, PEMS_ROW_COLUMNS["data_avg_occupancy"]),
                            "raw_avg_speed_value": row_value(row, PEMS_ROW_COLUMNS["data_avg_speed"]),
                            "raw_row_prefix": row[:14],
                        })
                    continue

                station = station_locations.get(station_id, {})
                lat = as_float(station.get("lat"))
                lon = as_float(station.get("lon"))

                if require_spatial_bounds and spatial_bounds is not None:
                    if lat is None or lon is None:
                        missing_station_location_for_bounds += 1
                        skipped_by_spatial_bounds += 1
                        if len(missing_station_location_examples) < 20:
                            missing_station_location_examples.append(station_id)
                        continue
                    if not point_in_bounds(lat, lon, spatial_bounds):
                        skipped_by_spatial_bounds += 1
                        if len(out_of_bounds_station_examples) < 20:
                            out_of_bounds_station_examples.append({
                                "station_id": station_id,
                                "lat": lat,
                                "lon": lon,
                            })
                        continue

                timestamp_raw = row[0] if len(row) > 0 else None
                hour_bucket = pems_hour_bucket(timestamp_raw, fallback_date_str=date_str)
                if hour_bucket is None:
                    missing_hour_bucket += 1

                if downsample_to_hourly and hour_bucket is not None:
                    station_hour_key = (station_id, hour_bucket)
                    if station_hour_key in seen_station_hour:
                        skipped_by_hourly_downsample += 1
                        continue
                    seen_station_hour.add(station_hour_key)

                profile.sensor_record_counts[station_id] = profile.sensor_record_counts.get(station_id, 0) + 1

                if lat is not None or lon is not None:
                    profile.sensor_locations.setdefault(station_id, {"latitude": lat, "longitude": lon})
                profile.sensor_names.setdefault(station_id, str(station.get("name") or station_id))

                if collect_for_replay or len(profile.sample_reports) < sample_limit:
                    data = pems_row_to_data(row, station=station)
                    report_date = normalize_report_date(timestamp_raw, fallback_date_str=date_str)
                    report = make_report(
                        report_id=f"real_{source}_{date_str}_{station_id}_{profile.sensor_record_counts[station_id]}",
                        report_date=report_date,
                        sensor_id=station_id,
                        sensor_name=str(station.get("name") or station_id),
                        sensor_type=source,
                        latitude=lat,
                        longitude=lon,
                        data=data,
                        metadata={
                            "date_str": date_str,
                            "source_file": str(traffic_path),
                            "source_row_index": row_index,
                            "source_timestamp_raw": timestamp_raw,
                            "report_timestamp_source": "pems_row_timestamp",
                            "hour_bucket": hour_bucket,
                            "downsampled_from_5min_to_hourly": downsample_to_hourly,
                            "spatial_bounds_filter": spatial_bounds if require_spatial_bounds else None,
                            "station_metadata_path": station_loader_diagnostics.get("selected_path"),
                            "station_metadata_loader": station_loader_diagnostics.get("selected_loader"),
                            "station_fwy": station.get("fwy"),
                            "station_direction": station.get("direction"),
                            "station_city": station.get("city"),
                            "station_lanes": station.get("lanes"),
                            "real_data": True,
                        },
                    )
                    profile.add_report(report, sample_limit=sample_limit, collect_for_replay=collect_for_replay)
        except Exception as exc:
            profile.warnings.append(f"failed to read {traffic_path}: {exc!r}")

    profile.extra["raw_rows_read_before_downsample"] = raw_rows_read
    profile.extra["raw_valid_station_rows_before_downsample"] = raw_valid_station_rows
    profile.extra["rows_skipped_missing_required_traffic_metrics"] = skipped_missing_required_traffic_metrics
    profile.extra["missing_required_traffic_metrics_examples"] = missing_required_traffic_metrics_examples
    profile.extra["required_traffic_metrics"] = ["data_avg_occupancy", "data_avg_speed"]
    profile.extra["rows_skipped_by_hourly_downsample"] = skipped_by_hourly_downsample
    profile.extra["rows_skipped_by_spatial_bounds"] = skipped_by_spatial_bounds
    profile.extra["rows_missing_station_location_for_bounds"] = missing_station_location_for_bounds
    profile.extra["missing_station_location_examples"] = missing_station_location_examples
    profile.extra["out_of_bounds_station_examples"] = out_of_bounds_station_examples
    profile.extra["rows_missing_hour_bucket"] = missing_hour_bucket
    profile.extra["rows_kept_after_downsample_and_bounds_filter"] = sum(profile.sensor_record_counts.values())
    return profile


def infer_report_date_from_payload(payload: Dict[str, Any], fallback_date_str: str) -> Optional[str]:
    for key in ("timestamp", "time", "datetime", "date", "dt_txt"):
        value = payload.get(key)
        if value is not None and value != "":
            return normalize_report_date(value, fallback_date_str=fallback_date_str)
    return date_str_to_midnight_iso(fallback_date_str)


def weather_csv_row_to_payload(row: Sequence[Any], *, path: Path) -> Optional[Dict[str, Any]]:
    """Parse one real weather CSV row.

    Real weather files are timestamp-named one-row CSVs under:

        evaluation/temp/weather_data/<YYYYMMDD>/<location_name>/<YYYYMMDDHHMMSS>.csv

    The row schema is:

        temperature_f, weather_description, humidity_percent, wind_speed_mps

    Example:

        58.64,scattered clouds,39,0.01
    """
    if len(row) < 4:
        return None

    temperature_f = as_float(row[0])
    weather_description = str(row[1]).strip()
    humidity = as_float(row[2])
    wind_speed_mps = as_float(row[3])

    if temperature_f is None or humidity is None or wind_speed_mps is None:
        return None

    timestamp = infer_timestamp_from_path(path)

    # Keep canonical units explicit, while also providing short aliases that are
    # easy for the downstream observation model to consume.
    payload: Dict[str, Any] = {
        "temperature_f": temperature_f,
        "temperature": temperature_f,
        "temperature_unit": "F",
        "weather_description": weather_description,
        "condition": weather_description,
        "humidity": humidity,
        "humidity_unit": "percent",
        "wind_speed_mps": wind_speed_mps,
        "wind_speed": wind_speed_mps,
        "wind_speed_unit": "m/s",
        "raw_row": list(row),
    }

    if timestamp is not None:
        payload["timestamp"] = timestamp

    return payload


def csv_row_looks_like_header(row: Sequence[Any]) -> bool:
    text = " ".join(str(cell).strip().lower() for cell in row)
    header_tokens = {
        "temp", "temperature", "weather", "description", "humidity",
        "wind", "windspeed", "wind_speed", "timestamp", "time",
    }
    return any(token in text for token in header_tokens)


def iter_weather_payloads(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            for idx, item in enumerate(raw):
                yield idx, item if isinstance(item, dict) else {"value": item}
        elif isinstance(raw, dict):
            # If it looks like a response containing a list field, count each item.
            yielded = False
            for key in ("hourly", "daily", "list", "data", "records"):
                value = raw.get(key)
                if isinstance(value, list):
                    for idx, item in enumerate(value):
                        yield idx, item if isinstance(item, dict) else {"value": item}
                    yielded = True
                    break
            if not yielded:
                yield 0, raw
        else:
            yield 0, {"value": raw}
    elif suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield idx, obj if isinstance(obj, dict) else {"value": obj}
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = [row for row in csv.reader(f) if row]

        if not rows:
            return

        # Most real weather files have exactly one headerless row:
        #   58.64,scattered clouds,39,0.01
        payload_index = 0
        for row_index, row in enumerate(rows):
            payload = weather_csv_row_to_payload(row, path=path)
            if payload is not None:
                yield payload_index, payload
                payload_index += 1
                continue

            # Also tolerate a header row followed by one or more regular rows.
            if row_index == 0 and csv_row_looks_like_header(row):
                continue

            # Fallback for unexpected CSV rows: still surface the file, but this
            # should be rare for the weather_data source.
            yield payload_index, {
                "raw_row": row,
                "timestamp": infer_timestamp_from_path(path),
                "parse_warning": "unexpected_weather_csv_row_schema",
            }
            payload_index += 1
    else:
        yield 0, {"file_path": str(path), "file_extension": suffix}



WEATHER_TECHNICAL_DIR_NAMES = {
    "weather",
    "weather_data",
    "pulled_data",
    "__macosx",
}


def looks_like_weather_location_folder(name: str) -> bool:
    """Return True for any direct child folder that can represent a weather area.

    Weather real data is discovered directly from:

        evaluation/temp/weather_data/<YYYYMMDD>/<location_folder>/...

    There is intentionally no hardcoded city allowlist.  Any non-empty,
    non-date, non-technical child folder under the date folder is treated as a
    location description.  Names like "Pasadena, US" are later rewritten to
    "Pasadena, California, US" for the location query.
    """
    text = str(name).strip()
    if not text or is_date_string(text):
        return False
    if text.lower() in WEATHER_TECHNICAL_DIR_NAMES:
        return False
    return True


def infer_weather_location_name_for_file(path: Path, date_folder: Path, date_str: str) -> str:
    """Infer the city/area folder represented by one weather data file.

    The expected layout is <date>/<location_name>/<files>, but extracted
    archives sometimes add wrapper folders.  This picks the first directory
    below the date folder that is not a technical wrapper/date name.
    """
    try:
        rel = path.relative_to(date_folder)
        dir_parts = list(rel.parts[:-1])
    except Exception:
        dir_parts = list(path.parent.parts)

    for part in dir_parts:
        cleaned = str(part).strip()
        if not cleaned:
            continue
        if cleaned == date_str or cleaned.lower() in WEATHER_TECHNICAL_DIR_NAMES:
            continue
        return cleaned

    # Direct files under the date folder are still real weather records, but
    # they lack a city/area wrapper in the filesystem.
    return date_folder.name


def group_weather_files_by_location(date_folder: Path, date_str: str) -> Dict[str, List[Path]]:
    """Group weather files by area folder.

    Canonical layout for the real data is:

        evaluation/temp/weather_data/<YYYYMMDD>/<location_name>/...

    where location_name is any direct child folder under the date folder, for
    example "Anaheim, US".  This function does not use a hardcoded location
    allowlist.  If no direct location folders are present, it falls back to
    recursively inferring the first non-technical directory below the date
    folder.
    """
    groups: Dict[str, List[Path]] = defaultdict(list)

    direct_location_dirs: List[Path] = []
    try:
        direct_location_dirs = [
            child
            for child in sorted(date_folder.iterdir())
            if child.is_dir() and looks_like_weather_location_folder(child.name)
        ]
    except OSError:
        direct_location_dirs = []

    if direct_location_dirs:
        for location_dir in direct_location_dirs:
            files = sorted(path for path in location_dir.rglob("*") if path.is_file())
            for path in files:
                groups[location_dir.name].append(path)
        return groups

    for path in sorted(date_folder.rglob("*")):
        if not path.is_file():
            continue
        location_name = infer_weather_location_name_for_file(path, date_folder, date_str)
        groups[location_name].append(path)
    return groups

def profile_weather(
    date_folder: Path,
    source: str,
    date_str: str,
    *,
    sample_limit: int,
    collect_for_replay: bool,
    geocode_weather_locations: bool,
    sensor_coverage: Optional[Any],
) -> DateProfile:
    profile = DateProfile(source=source, date_str=date_str, date_folder=str(date_folder))
    # Weather is intentionally represented as an area/location description, not
    # as a point sensor.  report["location"] carries a California-qualified
    # location_query instead of a point lat/lon so observation_model.py can see
    # the area context without being modified.
    weather_area_metadata: Dict[str, Dict[str, Any]] = {}

    if geocode_weather_locations and sensor_coverage is not None:
        try:
            gm = sensor_coverage.GeoManager()
            _polygons, weather_map, metadata = sensor_coverage.retrieve_by_date_weather(
                extracted_path=date_folder.parent,
                date_str=date_str,
                gm=gm,
                inject_california=True,
                return_location_map=True,
            )
            for location_name, entry in weather_map.items():
                weather_area_metadata[location_name] = {
                    "location_name": entry.get("location_name", location_name),
                    "location_description": location_name,
                    "location_query": entry.get("location_query"),
                    "weather_location_representation": "area_description",
                    "geometry_type": entry.get("geometry_type"),
                    "geometry_wkt": entry.get("geometry_wkt"),
                    "geo_source": entry.get("source"),
                    "geo_method": entry.get("method"),
                    "geo_cache_hit": entry.get("cache_hit"),
                    "geo_fallback_used": entry.get("fallback_used"),
                }
            profile.extra["weather_geocode_metadata"] = {
                key: value for key, value in metadata.items() if key not in {"failed_locations"}
            }
        except Exception as exc:
            profile.warnings.append(f"weather geocoding failed: {exc!r}")

    weather_file_groups = group_weather_files_by_location(date_folder, date_str)
    profile.extra["weather_expected_layout"] = "evaluation/temp/weather_data/<YYYYMMDD>/<location_name>/<YYYYMMDDHHMMSS>.csv"
    profile.extra["weather_location_discovery"] = "all_non_date_child_folders_under_weather_data_date"
    profile.extra["weather_csv_schema"] = [
        "temperature_f",
        "weather_description",
        "humidity_percent",
        "wind_speed_mps",
    ]
    profile.extra["weather_location_representation"] = "area_description_not_point"
    profile.extra["weather_timestamp_source"] = "timestamp inferred from CSV filename when payload has no timestamp field"
    profile.extra["num_weather_location_groups"] = len(weather_file_groups)
    profile.extra["weather_location_group_names_sample"] = list(weather_file_groups.keys())[:20]
    try:
        profile.extra["weather_date_folder_child_dirs_sample"] = [
            child.name for child in sorted(date_folder.iterdir()) if child.is_dir()
        ][:50]
        profile.extra["weather_date_folder_file_count_recursive"] = sum(
            1 for path in date_folder.rglob("*") if path.is_file()
        )
    except Exception as exc:
        profile.extra["weather_date_folder_scan_error"] = repr(exc)

    weather_groups = sorted(weather_file_groups.items())
    for location_name, files in weather_groups:
        if not files:
            continue

        area_metadata = weather_area_metadata.get(location_name)
        if area_metadata is None:
            area_metadata = {
                "location_name": location_name,
                "location_description": location_name,
                "location_query": weather_location_query_from_name(location_name, sensor_coverage),
                "weather_location_representation": "area_description",
            }

        # Keep this location explicitly non-point: weather supports a city/area.
        lat_lon = {"latitude": None, "longitude": None}
        profile.sensor_locations.setdefault(location_name, lat_lon)
        profile.sensor_names.setdefault(location_name, location_name)

        file_iter = files
        for file_index, data_file in enumerate(file_iter):
            ext = data_file.suffix.lower() or "<no_ext>"
            profile.file_counts_by_extension[ext] = profile.file_counts_by_extension.get(ext, 0) + 1
            try:
                for payload_index, payload in iter_weather_payloads(data_file):
                    profile.sensor_record_counts[location_name] = profile.sensor_record_counts.get(location_name, 0) + 1
                    if collect_for_replay or len(profile.sample_reports) < sample_limit:
                        report_date = infer_report_date_from_payload(payload, fallback_date_str=date_str)
                        weather_payload = add_weather_area_context_to_payload(payload, area_metadata)
                        report = make_report(
                            report_id=f"real_{source}_{date_str}_{location_name}_{profile.sensor_record_counts[location_name]}",
                            report_date=report_date,
                            sensor_id=location_name,
                            sensor_name=location_name,
                            sensor_type=source,
                            latitude=lat_lon.get("latitude"),
                            longitude=lat_lon.get("longitude"),
                            data=weather_payload,
                            metadata={
                                "date_str": date_str,
                                "source_file": str(data_file),
                                "source_file_index": file_index,
                                "payload_index": payload_index,
                                "report_timestamp_source": "weather_csv_filename_or_payload_timestamp",
                                **area_metadata,
                                "real_data": True,
                            },
                        )
                        # Weather is an area/city observation, not a point sensor.
                        # Keep latitude/longitude null, but make the California-
                        # qualified area query visible to observation_model.py via
                        # report["location"].
                        report["location"] = weather_area_location_object(area_metadata)
                        profile.add_report(report, sample_limit=sample_limit, collect_for_replay=collect_for_replay)
            except Exception as exc:
                profile.invalid_records += 1
                profile.warnings.append(f"failed to parse weather file {data_file}: {exc!r}")

    return profile


def generic_sensor_id_from_record(record: Dict[str, Any], fallback: str) -> str:
    for key in ("sensor_id", "station_id", "device_id", "post_id", "username", "user", "id", "location", "location_name", "Location"):
        value = record_get_ci(record, key)
        if value is not None and value != "":
            return str(value)
    return fallback


def generic_report_date_from_record(record: Dict[str, Any], fallback_date_str: str, fallback_file: Path) -> Optional[str]:
    for key in TEXT_TIME_KEYS:
        value = record_get_ci(record, key)
        if value is not None and value != "":
            return normalize_report_date(value, fallback_date_str=fallback_date_str)
    return infer_timestamp_from_path(fallback_file, fallback_date_str=fallback_date_str)


def generic_data_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    excluded = {x.lower() for x in set(ROW_LOCATION_KEYS) | set(ROW_TIME_KEYS)}
    return {key: value for key, value in record.items() if str(key).strip().lower() not in excluded}


def profile_generic_source(
    date_folder: Path,
    source: str,
    date_str: str,
    *,
    sample_limit: int,
    collect_for_replay: bool,
    text_location_resolver: Optional[TextLocationResolver] = None,
    geocode_text_locations: bool = True,
) -> DateProfile:
    """Best-effort profiler for citizen/twitter/unknown extracted layouts."""
    profile = DateProfile(source=source, date_str=date_str, date_folder=str(date_folder))

    files = sorted(path for path in date_folder.rglob("*") if path.is_file())
    file_iter = files
    for file_index, data_file in enumerate(file_iter):
        ext = data_file.suffix.lower() or "<no_ext>"
        profile.file_counts_by_extension[ext] = profile.file_counts_by_extension.get(ext, 0) + 1

        try:
            records: List[Tuple[int, Dict[str, Any]]] = []
            if ext == ".csv":
                with data_file.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames:
                        records = [(idx, coerce_payload(dict(row))) for idx, row in enumerate(reader)]
                    else:
                        f.seek(0)
                        reader2 = csv.reader(f)
                        records = [(idx, {"raw_row": row}) for idx, row in enumerate(reader2) if row]
            elif ext == ".jsonl":
                with data_file.open("r", encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        records.append((idx, obj if isinstance(obj, dict) else {"value": obj}))
            elif ext == ".json":
                obj = json.loads(data_file.read_text(encoding="utf-8"))
                values = obj if isinstance(obj, list) else [obj]
                records = [(idx, value if isinstance(value, dict) else {"value": value}) for idx, value in enumerate(values)]
            else:
                records = [(0, {"file_path": str(data_file), "file_extension": ext})]

            for row_index, row in records:
                sensor_id = generic_sensor_id_from_record(row, fallback=data_file.stem)
                profile.sensor_record_counts[sensor_id] = profile.sensor_record_counts.get(sensor_id, 0) + 1
                lat = as_float(first_present_ci(row, "latitude", "lat"))
                lon = as_float(first_present_ci(row, "longitude", "lon", "lng"))
                raw_location_text = generic_location_text_from_record(row)
                resolved_location: Optional[Dict[str, Any]] = None
                if (lat is None or lon is None) and geocode_text_locations and raw_location_text and text_location_resolver is not None:
                    resolved_location = text_location_resolver.resolve(raw_location_text)
                    if resolved_location is not None:
                        lat = as_float(resolved_location.get("latitude"))
                        lon = as_float(resolved_location.get("longitude"))

                if lat is not None or lon is not None:
                    profile.sensor_locations.setdefault(sensor_id, {"latitude": lat, "longitude": lon})
                profile.sensor_names.setdefault(sensor_id, sensor_id)

                if collect_for_replay or len(profile.sample_reports) < sample_limit:
                    metadata = {
                        "date_str": date_str,
                        "source_file": str(data_file),
                        "source_file_index": file_index,
                        "source_row_index": row_index,
                        "real_data": True,
                    }
                    if raw_location_text:
                        metadata["raw_text_location"] = raw_location_text
                    if resolved_location is not None:
                        metadata.update({
                            "text_location_geocoded": True,
                            "text_location_query": resolved_location.get("location_query"),
                            "text_location_geocode_source": resolved_location.get("geocode_source"),
                            "text_location_geocode_method": resolved_location.get("geocode_method"),
                            "text_location_cache_hit": resolved_location.get("cache_hit"),
                            "text_location_geometry_type": resolved_location.get("geometry_type"),
                            "text_location_geometry_wkt": resolved_location.get("geometry_wkt"),
                        })
                    elif raw_location_text:
                        metadata["text_location_geocoded"] = False

                    report = make_report(
                        report_id=f"real_{source}_{date_str}_{sensor_id}_{profile.sensor_record_counts[sensor_id]}",
                        report_date=generic_report_date_from_record(row, fallback_date_str=date_str, fallback_file=data_file),
                        sensor_id=sensor_id,
                        sensor_name=sensor_id,
                        sensor_type=source,
                        latitude=lat,
                        longitude=lon,
                        data=generic_data_from_record(row),
                        metadata=metadata,
                    )
                    if raw_location_text:
                        report["location"].update({
                            "location_name": raw_location_text,
                            "location_description": raw_location_text,
                            "location_query": resolved_location.get("location_query") if resolved_location else normalize_text_location_query(raw_location_text),
                            "location_representation": "text_geocoded_point" if resolved_location else "text_unresolved",
                        })
                    profile.add_report(report, sample_limit=sample_limit, collect_for_replay=collect_for_replay)

        except Exception as exc:
            profile.invalid_records += 1
            profile.warnings.append(f"failed to parse {data_file}: {exc!r}")

    if text_location_resolver is not None and source in {"citizen_data", "twitter_data"}:
        profile.extra["text_location_geocoding"] = dict(text_location_resolver.stats)

    return profile



# ---------------------------------------------------------------------------
# Optional raw-data preparation via data_process.py
# ---------------------------------------------------------------------------


def load_data_process_module(data_process_path: Optional[Path] = None) -> Any:
    """Load data_process.py so real_emitter can prepare evaluation/temp itself."""
    import importlib.util

    if data_process_path is not None:
        candidates = [Path(data_process_path)]
    else:
        here = Path(__file__).resolve()
        candidates = [
            Path("data_process.py"),
            Path("evaluation/data_process.py"),
            Path("evaluation/labelling/data_process.py"),
            here.parent / "data_process.py",
            here.parent.parent / "data_process.py",
            here.parent.parent.parent / "data_process.py",
        ]

    errors: List[str] = []

    # First try normal imports for project layouts where data_process.py is on
    # PYTHONPATH.  Then try explicit filesystem paths.
    for module_name in ["data_process", "evaluation.data_process", "evaluation.labelling.data_process"]:
        try:
            return importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: {exc!r}")

    for candidate in candidates:
        if not candidate.exists():
            errors.append(f"{candidate}: does not exist")
            continue
        try:
            spec = importlib.util.spec_from_file_location("real_emitter_data_process", candidate)
            if spec is None or spec.loader is None:
                raise ImportError(f"could not create import spec for {candidate}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            return module
        except Exception as exc:
            errors.append(f"{candidate}: {exc!r}")

    raise ImportError("Could not import data_process.py. Tried: " + "; ".join(errors))



def temp_source_date_has_extracted_data(temp_root: Path, source: str, date_str: str) -> bool:
    """Return True when evaluation/temp already has extracted files for source/date."""
    for folder_name in source_folder_names(source):
        date_folder = Path(temp_root) / folder_name / str(date_str)
        if not date_folder.exists() or not date_folder.is_dir():
            continue
        try:
            for child in date_folder.rglob("*"):
                if child.is_file():
                    return True
        except Exception:
            # Preserve data when in doubt.
            return True
    return False


def missing_source_date_pairs(
    *,
    temp_root: Path,
    data_sources: Sequence[str],
    date_strings: Sequence[str],
) -> List[Tuple[str, str]]:
    missing: List[Tuple[str, str]] = []
    for source in data_sources:
        for date_str in date_strings:
            if not temp_source_date_has_extracted_data(temp_root, str(source), str(date_str)):
                missing.append((str(source), str(date_str)))
    return missing


def filter_sources_and_dates_for_missing_temp_data(
    *,
    temp_root: Path,
    data_sources: Sequence[str],
    date_strings: Sequence[str],
) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """Return sources/dates still requiring extraction plus exact missing pairs.

    data_process.copy_and_untar_raw_data_to_temp accepts separate source/date
    lists, so this returns the source/date rectangle covering the missing pairs.
    The updated data_process.py will still skip already-extracted pairs inside
    that rectangle.
    """
    missing = missing_source_date_pairs(
        temp_root=temp_root,
        data_sources=data_sources,
        date_strings=date_strings,
    )
    return (
        sorted({source for source, _ in missing}),
        sorted({date_str for _, date_str in missing}),
        missing,
    )



def prepare_temp_data_with_data_process(
    *,
    date_strings: Sequence[str],
    data_sources: Sequence[str],
    raw_root: Path,
    temp_root: Path,
    data_process_path: Optional[Path] = None,
    keep_tar: bool = False,
    strict: bool = False,
    clear_temp: bool = False,
    skip_existing: bool = True,
) -> None:
    """Call data_process.copy_and_untar_raw_data_to_temp before profiling/replay.

    Default behavior is non-destructive:
      * do not clear evaluation/temp,
      * skip source/date folders that already contain extracted files,
      * refuse to call legacy data_process.py if it could delete existing temp data.
    """
    if not date_strings:
        raise ValueError("AUTO_PREPARE_TEMP_DATA requires DATE_STRINGS to be non-empty.")
    if not data_sources:
        raise ValueError("AUTO_PREPARE_TEMP_DATA requires DATA_SOURCES to be non-empty.")

    temp_root = Path(temp_root)
    raw_root = Path(raw_root)

    if clear_temp:
        log(
            "WARNING: clear_temp=True was requested. Existing extracted temp data may be deleted. "
            "Use --no-clear-temp-before-prepare to preserve evaluation/temp."
        )
    elif temp_root.exists():
        log(f"Preserving existing temp root: {temp_root}")

    effective_sources = list(data_sources)
    effective_dates = list(date_strings)

    if skip_existing:
        effective_sources, effective_dates, missing_pairs = filter_sources_and_dates_for_missing_temp_data(
            temp_root=temp_root,
            data_sources=data_sources,
            date_strings=date_strings,
        )
        total_pairs = len(list(data_sources)) * len(list(date_strings))
        existing_pairs = total_pairs - len(missing_pairs)
        log(
            "Temp-data precheck: "
            f"existing_source_date_pairs={existing_pairs}, missing_source_date_pairs={len(missing_pairs)}"
        )
        if not missing_pairs:
            log("All requested source/date temp folders already contain extracted data; skipping raw TAR preparation.")
            return
        log(f"Preparing only missing temp coverage: sources={effective_sources}, dates={effective_dates}")

    module = load_data_process_module(data_process_path)
    prepare_fn = getattr(module, "copy_and_untar_raw_data_to_temp", None)
    if not callable(prepare_fn):
        raise AttributeError("data_process.py does not define copy_and_untar_raw_data_to_temp(...).")

    log(
        "Preparing evaluation/temp from raw TARs using data_process.py: "
        f"dates={effective_dates}, sources={effective_sources}, raw_root={raw_root}, "
        f"temp_root={temp_root}, clear_temp={clear_temp}, skip_existing={skip_existing}"
    )
    try:
        prepare_fn(
            date_strings=list(effective_dates),
            data_sources=list(effective_sources),
            raw_root=str(raw_root),
            temp_root=str(temp_root),
            keep_tar=keep_tar,
            strict=strict,
            clear_temp=clear_temp,
            skip_existing=skip_existing,
        )
    except TypeError as exc:
        if "clear_temp" not in str(exc) and "skip_existing" not in str(exc):
            raise

        any_existing_requested = any(
            temp_source_date_has_extracted_data(temp_root, str(source), str(date_str))
            for source in data_sources
            for date_str in date_strings
        )
        if any_existing_requested and not clear_temp:
            raise RuntimeError(
                "Refusing to call legacy data_process.copy_and_untar_raw_data_to_temp(...) "
                "because it does not support clear_temp/skip_existing and existing extracted "
                f"temp data was found under {temp_root}. This protects existing TAR extractions. "
                "Use the updated data_process.py, or run with --no-auto-prepare-temp-data."
            ) from exc

        log(
            "WARNING: data_process.py does not support clear_temp/skip_existing; "
            "calling legacy function only because no existing temp data is at risk "
            "or clear_temp=True was explicitly requested."
        )
        prepare_fn(
            date_strings=list(effective_dates),
            data_sources=list(effective_sources),
            raw_root=str(raw_root),
            temp_root=str(temp_root),
            keep_tar=keep_tar,
            strict=strict,
        )

# ---------------------------------------------------------------------------
# Main orchestration and ordered replay
# ---------------------------------------------------------------------------


def profile_source_date(
    *,
    temp_root: Path,
    source: str,
    date_str: str,
    sample_limit: int,
    collect_for_replay: bool,
    sensor_coverage: Optional[Any],
    cctv_kml_path: Path,
    cctv_skip_unavailable_images: bool,
    cctv_unavailable_signature: Optional[Dict[str, Any]],
    cctv_unavailable_decision_cache: Optional[Dict[str, Dict[str, Any]]],
    cctv_unavailable_decision_cache_path: Optional[Path],
    cctv_use_unavailable_decision_cache: bool,
    alertcalifornia_locations_path: Path,
    pems_station_path_candidates: Sequence[Path],
    pems_downsample_to_hourly: bool,
    pems_spatial_bounds: Optional[Dict[str, float]],
    pems_require_spatial_bounds: bool,
    pems_station_cache_dir: Optional[Path],
    pems_station_cache_enabled: bool,
    pems_station_cache_read: bool,
    pems_station_cache_write: bool,
    geocode_weather_locations: bool,
    text_location_resolver: Optional[TextLocationResolver],
    geocode_text_locations: bool,
) -> Optional[DateProfile]:
    date_folder = resolve_source_date_folder(temp_root, source, date_str)
    if date_folder is None:
        return None

    if source == "air_data":
        return profile_air_data(date_folder, source, date_str, sample_limit=sample_limit, collect_for_replay=collect_for_replay)
    if source == "alertcalifornia":
        return profile_alertcalifornia(
            date_folder,
            source,
            date_str,
            sample_limit=sample_limit,
            collect_for_replay=collect_for_replay,
            sensor_coverage=sensor_coverage,
            alertcalifornia_locations_path=alertcalifornia_locations_path,
        )
    if source == "cctv":
        return profile_cctv(
            date_folder,
            source,
            date_str,
            sample_limit=sample_limit,
            collect_for_replay=collect_for_replay,
            sensor_coverage=sensor_coverage,
            cctv_kml_path=cctv_kml_path,
            skip_unavailable_images=cctv_skip_unavailable_images,
            unavailable_signature=cctv_unavailable_signature,
            unavailable_decision_cache=cctv_unavailable_decision_cache,
            unavailable_decision_cache_path=cctv_unavailable_decision_cache_path,
            use_unavailable_decision_cache=cctv_use_unavailable_decision_cache,
        )
    if source == "pem_data_station_5min":
        return profile_traffic(
            date_folder,
            source,
            date_str,
            sample_limit=sample_limit,
            collect_for_replay=collect_for_replay,
            sensor_coverage=sensor_coverage,
            pems_station_path_candidates=pems_station_path_candidates,
            downsample_to_hourly=pems_downsample_to_hourly,
            spatial_bounds=pems_spatial_bounds,
            require_spatial_bounds=pems_require_spatial_bounds,
            pems_station_cache_dir=pems_station_cache_dir,
            pems_station_cache_enabled=pems_station_cache_enabled,
            pems_station_cache_read=pems_station_cache_read,
            pems_station_cache_write=pems_station_cache_write,
        )
    if source == "weather_data":
        return profile_weather(
            date_folder,
            source,
            date_str,
            sample_limit=sample_limit,
            collect_for_replay=collect_for_replay,
            geocode_weather_locations=geocode_weather_locations,
            sensor_coverage=sensor_coverage,
        )

    return profile_generic_source(
        date_folder,
        source,
        date_str,
        sample_limit=sample_limit,
        collect_for_replay=collect_for_replay,
        text_location_resolver=text_location_resolver,
        geocode_text_locations=geocode_text_locations and source in {"citizen_data", "twitter_data"},
    )


def first_example_report_by_source(profiles: Sequence[DateProfile]) -> Dict[str, Dict[str, Any]]:
    """Return one normalized-report example per data source for summary/display."""
    examples: Dict[str, Dict[str, Any]] = {}
    for profile in profiles:
        if profile.source in examples:
            continue
        if profile.sample_reports:
            examples[profile.source] = profile.sample_reports[0]
    return examples


def print_example_reports_by_source(examples: Dict[str, Dict[str, Any]]) -> None:
    if not examples:
        print("No example reports were generated.")
        return
    print("Example emitted REPORT by source:")
    for source in sorted(examples):
        print(f"\n--- {source} ---")
        print(json.dumps(examples[source], indent=2, ensure_ascii=False))


def aggregate_profiles(profiles: Sequence[DateProfile]) -> Dict[str, Any]:
    by_source: Dict[str, Dict[str, Any]] = {}

    for profile in profiles:
        source_entry = by_source.setdefault(
            profile.source,
            {
                "source": profile.source,
                "dates": [],
                "sensor_record_counts": defaultdict(int),
                "file_counts_by_extension": defaultdict(int),
                "invalid_records": 0,
                "warnings": [],
                "extra": defaultdict(int),
            },
        )
        source_entry["dates"].append(profile.date_str)
        for sensor_id, count in profile.sensor_record_counts.items():
            source_entry["sensor_record_counts"][sensor_id] += count
        for ext, count in profile.file_counts_by_extension.items():
            source_entry["file_counts_by_extension"][ext] += count
        source_entry["invalid_records"] += profile.invalid_records
        source_entry["warnings"].extend(profile.warnings)

        # Carry forward numeric extras like raw/kept PeMS rows.
        for key, value in profile.extra.items():
            if isinstance(value, int):
                source_entry["extra"][key] += value

    result: Dict[str, Any] = {}
    for source, entry in by_source.items():
        counts = dict(entry["sensor_record_counts"])
        result[source] = {
            "source": source,
            "dates": sorted(set(entry["dates"])),
            **summarize_counts(counts),
            "top_sensors_by_records": top_sensor_counts(counts, n=20),
            "file_counts_by_extension": dict(sorted(dict(entry["file_counts_by_extension"]).items())),
            "invalid_records": entry["invalid_records"],
            "num_warnings": len(entry["warnings"]),
            "warnings_sample": entry["warnings"][:10],
            "extra_numeric_totals": dict(entry["extra"]),
        }

    return result


def print_summary_table(aggregate: Dict[str, Any]) -> None:
    columns = [
        ("source", 24),
        ("dates", 7),
        ("sensors", 9),
        ("records", 10),
        ("avg/sensor", 12),
        ("median", 10),
        ("max", 8),
    ]
    header = "".join(name.ljust(width) for name, width in columns)
    print(header)
    print("-" * len(header))
    for source in sorted(aggregate):
        row = aggregate[source]
        values = {
            "source": source,
            "dates": str(len(row["dates"])),
            "sensors": str(row["num_sensors"]),
            "records": str(row["total_records"]),
            "avg/sensor": f"{row['records_per_sensor_avg']:.2f}",
            "median": f"{row['records_per_sensor_median']:.2f}",
            "max": str(row["records_per_sensor_max"]),
        }
        print("".join(values[name].ljust(width) for name, width in columns))




# ---------------------------------------------------------------------------
# Source/date profile cache
# ---------------------------------------------------------------------------


def profile_cache_filename(source: str, date_str: str) -> str:
    return f"{slugify_cache_component(source)}__{slugify_cache_component(date_str)}.json"


def slugify_cache_component(value: Any) -> str:
    text = str(value or "unknown").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def profile_cache_path(cache_dir: Path, source: str, date_str: str) -> Path:
    return cache_dir / slugify_cache_component(source) / profile_cache_filename(source, date_str)


def profile_to_cache_payload(profile: DateProfile, *, cache_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "cache_version": PROFILE_REPORT_CACHE_VERSION,
        "cache_metadata": cache_metadata or {},
        "profile": {
            "source": profile.source,
            "date_str": profile.date_str,
            "date_folder": profile.date_folder,
            "sensor_record_counts": profile.sensor_record_counts,
            "sensor_locations": profile.sensor_locations,
            "sensor_names": profile.sensor_names,
            "file_counts_by_extension": profile.file_counts_by_extension,
            "invalid_records": profile.invalid_records,
            "warnings": profile.warnings,
            "sample_reports": profile.sample_reports,
            "emission_reports": profile.emission_reports,
            "extra": profile.extra,
        },
    }


def profile_from_cache_payload(payload: Dict[str, Any]) -> Optional[DateProfile]:
    if int(payload.get("cache_version", -1)) != PROFILE_REPORT_CACHE_VERSION:
        return None
    data = payload.get("profile")
    if not isinstance(data, dict):
        return None
    try:
        profile = DateProfile(
            source=str(data["source"]),
            date_str=str(data["date_str"]),
            date_folder=str(data.get("date_folder") or ""),
        )
        profile.sensor_record_counts = {str(k): int(v) for k, v in (data.get("sensor_record_counts") or {}).items()}
        profile.sensor_locations = {
            str(k): dict(v) if isinstance(v, dict) else {"latitude": None, "longitude": None}
            for k, v in (data.get("sensor_locations") or {}).items()
        }
        profile.sensor_names = {str(k): str(v) for k, v in (data.get("sensor_names") or {}).items()}
        profile.file_counts_by_extension = {str(k): int(v) for k, v in (data.get("file_counts_by_extension") or {}).items()}
        profile.invalid_records = int(data.get("invalid_records") or 0)
        profile.warnings = list(data.get("warnings") or [])
        profile.sample_reports = list(data.get("sample_reports") or [])
        profile.emission_reports = list(data.get("emission_reports") or [])
        profile.extra = dict(data.get("extra") or {})
        profile.extra["loaded_from_profile_report_cache"] = True
        if isinstance(payload.get("cache_metadata"), dict):
            profile.extra["profile_report_cache_metadata"] = dict(payload.get("cache_metadata") or {})
        return profile
    except Exception as exc:
        log(f"WARNING: failed to deserialize profile cache payload: {exc}")
        return None


def load_profile_report_cache(cache_dir: Path, source: str, date_str: str) -> Optional[DateProfile]:
    path = profile_cache_path(cache_dir, source, date_str)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return None
        profile = profile_from_cache_payload(payload)
        if profile is not None:
            profile.extra["profile_report_cache_path"] = str(path)
        return profile
    except Exception as exc:
        log(f"WARNING: could not load profile/report cache {path}: {exc}")
        return None


def save_profile_report_cache(cache_dir: Path, profile: DateProfile, *, cache_metadata: Optional[Dict[str, Any]] = None) -> None:
    path = profile_cache_path(cache_dir, profile.source, profile.date_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = profile_to_cache_payload(profile, cache_metadata=cache_metadata)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    tmp_path.replace(path)

def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def open_jsonl_socket(host: str, port: int) -> Tuple[socket.socket, Any, Any]:
    sock = socket.create_connection((host, port), timeout=10.0)
    sock.settimeout(None)
    reader = sock.makefile("r", encoding="utf-8", newline="\n")
    writer = sock.makefile("w", encoding="utf-8", newline="\n")
    return sock, reader, writer


def read_ack(reader: Any, report_id: Any = None) -> Dict[str, Any]:
    line = reader.readline()
    if line == "":
        raise RuntimeError(f"socket closed before ACK for report {report_id!r}")
    ack = json.loads(line)
    if not isinstance(ack, dict):
        raise RuntimeError(f"ACK was not a JSON object: {ack!r}")
    if str(ack.get("type", "ack")).lower() != "ack":
        raise RuntimeError(f"Unexpected socket response: {ack}")
    if ack.get("ok") is False or str(ack.get("status", "ok")).lower() not in {"ok", "success", "processed", ""}:
        raise RuntimeError(f"Non-ok ACK: {ack}")
    ack_report_id = ack.get("report_id")
    if report_id is not None and ack_report_id not in {None, report_id}:
        raise RuntimeError(f"ACK report_id mismatch: expected {report_id!r}, got {ack_report_id!r}")
    return ack


def send_control_message(
    writer: Any,
    reader: Any,
    *,
    message_type: str,
    wait_for_socket_ack: bool,
    **payload: Any,
) -> None:
    """Send an optional stream-level control message to full_pipeline.py."""
    message = {"type": message_type, **payload}
    writer.write(json.dumps(message, ensure_ascii=False) + "\n")
    writer.flush()
    if wait_for_socket_ack:
        read_ack(reader, report_id=None)



# ---------------------------------------------------------------------------
# Observation-cache positive filtering for real-data replay
# ---------------------------------------------------------------------------


def _observation_cache_output_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract an observation-model output from one cache JSON record."""
    if not isinstance(record, dict):
        return {}
    output = record.get("observation_output")
    if isinstance(output, dict):
        return output
    # Some cache files are written directly as the downstream observation object.
    if "observed_effects" in record or "possible_incidents" in record or "provenance" in record:
        return record
    return {}


def observation_output_is_semantic_positive(output: Dict[str, Any]) -> bool:
    """Return True for cached outputs that should be replayed online.

    The fast real-data path streams only observations that already produced
    semantic evidence in the observation cache.  Non-semantic reports remain on
    disk and can later be sampled/retrieved as background evidence, but they are
    not pushed through the per-report online loop.
    """
    effects = output.get("observed_effects")
    incidents = output.get("possible_incidents")
    if isinstance(effects, list) and len(effects) > 0:
        return True
    if isinstance(incidents, list) and len(incidents) > 0:
        return True

    # Be conservative for anomaly-preprocessing metadata if it was stored inside
    # the observation output.  Different versions used slightly different field
    # names, so treat explicit anomalous/heavy-model decisions as positives but
    # do not treat skipped/low-relevance decisions as positives.
    model = output.get("model") if isinstance(output.get("model"), dict) else {}
    anomaly = None
    if isinstance(model, dict):
        anomaly = model.get("anomaly_preprocessing")
    if anomaly is None and isinstance(output.get("_anomaly_preprocessing"), dict):
        anomaly = output.get("_anomaly_preprocessing")
    if isinstance(anomaly, dict):
        decision = str(anomaly.get("decision") or anomaly.get("status") or "").lower()
        if decision and not decision.startswith("skipped") and not decision.startswith("skip"):
            if any(token in decision for token in ["anomal", "heavy", "run", "selected", "positive"]):
                return True
        for key in ["is_anomalous", "anomalous", "heavy_model_used", "ran_heavy_model"]:
            if anomaly.get(key) is True:
                return True
    return False


def _cache_key_date_from_output(output: Dict[str, Any]) -> str:
    prov = output.get("provenance") if isinstance(output.get("provenance"), dict) else {}
    for value in [prov.get("date_str"), output.get("date_str"), output.get("report_date")]:
        if value is None:
            continue
        digits = re.sub(r"[^0-9]", "", str(value))
        if len(digits) >= 8 and digits[:4].startswith("20"):
            return digits[:8]
    return ""


def _positive_match_keys_from_output(output: Dict[str, Any]) -> set[str]:
    """Build conservative match keys for a positive cached observation output."""
    keys: set[str] = set()
    prov = output.get("provenance") if isinstance(output.get("provenance"), dict) else {}
    sensor_type = str(output.get("sensor_type") or prov.get("sensor_type") or "")
    date_str = _cache_key_date_from_output(output)
    report_id = str(output.get("report_id") or prov.get("report_id") or "")
    report_date = str(output.get("report_date") or prov.get("report_date") or "")
    sensor_id = str(output.get("sensor_id") or prov.get("sensor_id") or "")
    source_file = str(prov.get("source_file") or "")
    source_base = Path(source_file).name if source_file else ""
    source_row_index = str(prov.get("source_row_index") or "")
    source_file_index = str(prov.get("source_file_index") or "")

    if report_id:
        keys.add(f"report_id::{report_id}")
    if sensor_type and date_str and sensor_id and report_date:
        keys.add(f"real_sensor_time::{sensor_type}|{date_str}|{sensor_id}|{report_date}")
    if sensor_type and date_str and source_base and source_row_index:
        keys.add(f"real_source_row::{sensor_type}|{date_str}|{source_base}|{source_row_index}")
    if sensor_type and date_str and source_base and source_file_index:
        keys.add(f"real_source_file_index::{sensor_type}|{date_str}|{source_base}|{source_file_index}")
    return keys


def _positive_match_keys_from_report(report: Dict[str, Any]) -> set[str]:
    """Build match keys for one normalized real-emitter report."""
    keys: set[str] = set()
    metadata = report.get("metadata") if isinstance(report.get("metadata"), dict) else {}
    report_id = str(report.get("report_id") or "")
    sensor_type = str(report.get("sensor_type") or metadata.get("source") or metadata.get("data_source") or "")
    date_str = str(metadata.get("date_str") or "")
    if not date_str:
        dt = parse_datetime(report.get("report_date"))
        date_str = dt.strftime("%Y%m%d") if dt is not None else ""
    report_date = str(report.get("report_date") or "")
    sensor_id = str(report.get("sensor_id") or "")
    source_file = str(metadata.get("source_file") or "")
    source_base = Path(source_file).name if source_file else ""
    source_row_index = str(metadata.get("source_row_index") or "")
    source_file_index = str(metadata.get("source_file_index") or "")

    if report_id:
        keys.add(f"report_id::{report_id}")
    if sensor_type and date_str and sensor_id and report_date:
        keys.add(f"real_sensor_time::{sensor_type}|{date_str}|{sensor_id}|{report_date}")
    if sensor_type and date_str and source_base and source_row_index:
        keys.add(f"real_source_row::{sensor_type}|{date_str}|{source_base}|{source_row_index}")
    if sensor_type and date_str and source_base and source_file_index:
        keys.add(f"real_source_file_index::{sensor_type}|{date_str}|{source_base}|{source_file_index}")
    return keys


def load_positive_observation_cache_keys(
    *,
    observation_cache_root: Path,
    data_sources: Sequence[str],
    date_strings: Sequence[str],
) -> Tuple[set[str], Dict[str, Any]]:
    """Load match keys for positive cached observation outputs."""
    root = Path(observation_cache_root)
    keys: set[str] = set()
    stats: Dict[str, Any] = {
        "cache_root": str(root),
        "source_date_dirs_checked": 0,
        "source_date_dirs_missing": 0,
        "json_files_seen": 0,
        "cache_records_read": 0,
        "positive_cache_records": 0,
        "positive_match_keys": 0,
        "errors": [],
    }
    for source in data_sources:
        source_name = str(source)
        for date_str in date_strings:
            cache_dir = root / source_name / str(date_str)
            stats["source_date_dirs_checked"] += 1
            if not cache_dir.exists():
                stats["source_date_dirs_missing"] += 1
                continue
            for path in cache_dir.glob("*.json"):
                if path.name in {"index.json", "summary.json"}:
                    continue
                stats["json_files_seen"] += 1
                try:
                    with path.open("r", encoding="utf-8") as infile:
                        record = json.load(infile)
                except Exception as exc:
                    errors = stats.setdefault("errors", [])
                    if len(errors) < 20:
                        errors.append({"path": str(path), "error": str(exc)})
                    continue
                output = _observation_cache_output_from_record(record)
                if not output:
                    continue
                stats["cache_records_read"] += 1
                if not observation_output_is_semantic_positive(output):
                    continue
                stats["positive_cache_records"] += 1
                keys.update(_positive_match_keys_from_output(output))
    stats["positive_match_keys"] = len(keys)
    return keys, stats


def filter_reports_to_cached_positives(
    reports: Sequence[Dict[str, Any]],
    *,
    observation_cache_root: Path,
    data_sources: Sequence[str],
    date_strings: Sequence[str],
    missing_policy: str = "keep_all",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return only reports whose observation-cache entry is semantically positive."""
    positive_keys, cache_stats = load_positive_observation_cache_keys(
        observation_cache_root=observation_cache_root,
        data_sources=data_sources,
        date_strings=date_strings,
    )
    if not positive_keys:
        # Distinguish "cache exists and everything was negative" from
        # "no usable cache exists".  The former should emit zero reports; the
        # latter follows missing_policy so users do not accidentally run an
        # empty experiment when the observation cache has not been built yet.
        cache_records_read = int(cache_stats.get("cache_records_read") or 0)
        if cache_records_read > 0:
            kept = []
            positive_cache_available = True
            reason = "cache_scanned_no_positive_records"
        else:
            kept = list(reports) if missing_policy == "keep_all" else []
            positive_cache_available = False
            reason = "no_cache_records_found"
        stats = {
            "enabled": True,
            "missing_policy": missing_policy,
            "input_reports": len(reports),
            "kept_reports": len(kept),
            "dropped_reports": len(reports) - len(kept),
            "positive_cache_available": positive_cache_available,
            "filter_reason": reason,
            "cache": cache_stats,
        }
        return kept, stats

    kept_reports: List[Dict[str, Any]] = []
    for report in reports:
        if _positive_match_keys_from_report(report) & positive_keys:
            kept_reports.append(report)
    stats = {
        "enabled": True,
        "missing_policy": missing_policy,
        "input_reports": len(reports),
        "kept_reports": len(kept_reports),
        "dropped_reports": len(reports) - len(kept_reports),
        "positive_cache_available": True,
        "cache": cache_stats,
    }
    return kept_reports, stats


# ---------------------------------------------------------------------------
# Anomaly-only top-K replay filtering
# ---------------------------------------------------------------------------


def _selected_anomaly_report_from_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract a normalized report from an anomaly-only top-K selected row.

    Current full_pipeline anomaly-only mode writes raw normalized reports to
    batch_runs/<YYYYMMDD>/selected_observations.jsonl.  Older/debug files may
    wrap the report under a "report" key, so accept both shapes.
    """
    if not isinstance(row, dict):
        return None
    report = row.get("report")
    if isinstance(report, dict):
        return dict(report)
    if "report_id" in row or "sensor_type" in row or "report_date" in row:
        return dict(row)
    return None


def _report_source_matches_filter(report: Dict[str, Any], data_sources: Optional[Sequence[str]]) -> bool:
    if not data_sources:
        return True
    wanted = {str(src).strip().lower() for src in data_sources if str(src).strip()}
    if not wanted:
        return True
    metadata = report.get("metadata") if isinstance(report.get("metadata"), dict) else {}
    candidates = {
        str(report.get("sensor_type") or "").strip().lower(),
        str(report.get("source") or "").strip().lower(),
        str(metadata.get("source") or "").strip().lower(),
        str(metadata.get("data_source") or "").strip().lower(),
        str(metadata.get("sensor_type") or "").strip().lower(),
    }
    candidates.discard("")
    return bool(candidates & wanted)


def load_anomaly_topk_selected_reports(
    *,
    anomaly_topk_root: Path,
    date_strings: Sequence[str],
    data_sources: Optional[Sequence[str]] = None,
    missing_policy: str = "error",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load anomaly-only top-K selected observations for replay.

    Expected layout is produced by full_pipeline anomaly-only mode:

        detection/outputs/anomaly/batch_runs/<YYYYMMDD>/selected_observations.jsonl

    Each selected row is a normalized report that can be sent directly to
    full_pipeline for cached observation-model lookup and baseline updates.
    """
    root = Path(anomaly_topk_root)
    reports: List[Dict[str, Any]] = []
    seen_report_ids: set[str] = set()
    stats: Dict[str, Any] = {
        "enabled": True,
        "root": str(root),
        "missing_policy": str(missing_policy),
        "dates_requested": list(date_strings),
        "date_files_checked": 0,
        "date_files_missing": 0,
        "rows_read": 0,
        "rows_invalid": 0,
        "duplicate_reports_dropped": 0,
        "source_filtered_reports_dropped": 0,
        "reports_loaded": 0,
        "missing_files": [],
        "errors": [],
    }
    for date_str in date_strings:
        path = root / str(date_str) / "selected_observations.jsonl"
        stats["date_files_checked"] += 1
        if not path.exists():
            stats["date_files_missing"] += 1
            missing = stats.setdefault("missing_files", [])
            if len(missing) < 50:
                missing.append(str(path))
            continue
        try:
            with path.open("r", encoding="utf-8") as infile:
                for line_number, line in enumerate(infile, start=1):
                    text = line.strip()
                    if not text:
                        continue
                    stats["rows_read"] += 1
                    try:
                        row = json.loads(text)
                    except json.JSONDecodeError as exc:
                        stats["rows_invalid"] += 1
                        errors = stats.setdefault("errors", [])
                        if len(errors) < 20:
                            errors.append({"path": str(path), "line": line_number, "error": str(exc)})
                        continue
                    report = _selected_anomaly_report_from_row(row)
                    if report is None:
                        stats["rows_invalid"] += 1
                        continue
                    if not _report_source_matches_filter(report, data_sources):
                        stats["source_filtered_reports_dropped"] += 1
                        continue
                    report_id = str(report.get("report_id") or "")
                    if report_id and report_id in seen_report_ids:
                        stats["duplicate_reports_dropped"] += 1
                        continue
                    if report_id:
                        seen_report_ids.add(report_id)
                    reports.append(report)
        except Exception as exc:
            errors = stats.setdefault("errors", [])
            if len(errors) < 20:
                errors.append({"path": str(path), "error": str(exc)})

    stats["reports_loaded"] = len(reports)
    if not reports and missing_policy == "error":
        missing_preview = ", ".join(stats.get("missing_files", [])[:5])
        raise FileNotFoundError(
            "No anomaly top-K selected observations were loaded. "
            f"root={root}, dates={list(date_strings)}, missing_files={missing_preview}"
        )
    return reports, stats

def replay_reports_in_temporal_order(
    reports: Sequence[Dict[str, Any]],
    *,
    output_jsonl_path: Optional[Path] = None,
    emit_to_socket: bool = False,
    socket_host: str = "127.0.0.1",
    socket_port: int = 8765,
    wait_for_socket_ack: bool = True,
    interval_seconds: float = 0.0,
    print_reports: bool = False,
    send_stream_control_messages: bool = True,
) -> int:
    """Replay reports globally sorted by report_date across all sources."""
    ordered_reports = sorted(reports, key=report_sort_key)

    output_handle = None
    sock = None
    reader = None
    writer = None
    emitted = 0

    try:
        if output_jsonl_path is not None:
            output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            output_handle = output_jsonl_path.open("w", encoding="utf-8")

        if emit_to_socket:
            sock, reader, writer = open_jsonl_socket(socket_host, socket_port)
            log(f"Connected to {socket_host}:{socket_port}; replaying reports in temporal order.")
            if send_stream_control_messages:
                send_control_message(
                    writer,
                    reader,
                    message_type="stream_start",
                    wait_for_socket_ack=wait_for_socket_ack,
                    num_reports=len(ordered_reports),
                    source="real_emitter",
                )

        report_iter = progress_iter(ordered_reports, desc="Replay ordered reports", total=len(ordered_reports), unit="report", leave=True)
        for report in report_iter:
            line = json.dumps(report, ensure_ascii=False)
            if print_reports:
                print(line, flush=True)
            if output_handle is not None:
                output_handle.write(line + "\n")
                output_handle.flush()
            if writer is not None:
                writer.write(line + "\n")
                writer.flush()
                if wait_for_socket_ack:
                    read_ack(reader, report_id=report.get("report_id"))
            emitted += 1
            if interval_seconds > 0:
                time.sleep(interval_seconds)

        if writer is not None and send_stream_control_messages:
            send_control_message(
                writer,
                reader,
                message_type="stream_end",
                wait_for_socket_ack=wait_for_socket_ack,
                reports_sent=emitted,
                source="real_emitter",
            )
    finally:
        if output_handle is not None:
            output_handle.close()
        if writer is not None:
            writer.close()
        if reader is not None:
            reader.close()
        if sock is not None:
            sock.close()

    return emitted



def _read_date_file(path: str | Path) -> List[str]:
    """Read YYYYMMDD values from a file, ignoring blanks/comments."""
    out: List[str] = []
    with Path(path).open("r", encoding="utf-8") as infile:
        for line in infile:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            for part in re.split(r"[\s,]+", text):
                part = part.strip()
                if part:
                    out.append(part)
    return out


def _normalize_date_list(values: Sequence[Any]) -> List[str]:
    """Normalize and validate date strings."""
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        digits = re.sub(r"[^0-9]", "", text)
        if len(digits) != 8:
            raise ValueError(f"Expected YYYYMMDD date, got {value!r}")
        if digits not in seen:
            seen.add(digits)
            out.append(digits)
    return sorted(out)

def load_run_experiments_date_set(date_set: str, fallback: Sequence[Any]) -> List[str]:
    """Load default real-data date sets from evaluation.run_experiments.

    This keeps real_emitter aligned with the date lists used by
    evaluation/run_experiments.py, while preserving a local fallback so the
    emitter can still run standalone.
    """
    candidates = (
        "evaluation.run_experiments",
        "run_experiments",
    )
    errors: List[str] = []
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
            if date_set == "non":
                values = getattr(module, "DEFAULT_REAL_NON_DATES", None)
            else:
                values = getattr(module, "DEFAULT_REAL_DATES", None)
            if values:
                return _normalize_date_list(values)
        except Exception as exc:
            errors.append(f"{module_name}: {exc!r}")
    if errors:
        log("WARNING: could not import run_experiments date defaults; using local real_emitter fallback. " + " | ".join(errors))
    return _normalize_date_list(fallback)


def compact_coverage_source_record(report: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a compact source-inventory row for coverage scoring.

    The goal is to keep the JSONL small: no image bytes, no large payloads, and
    no full raw rows.  coverage_metrics.py only needs sensor/source identity,
    modality/source type, approximate time, location, and small evidence hints.
    """
    if not isinstance(report, Mapping):
        return None

    metadata = report.get("metadata") if isinstance(report.get("metadata"), Mapping) else {}
    data = report.get("data") if isinstance(report.get("data"), Mapping) else {}
    loc = report.get("location") if isinstance(report.get("location"), Mapping) else {}

    sensor_type = str(report.get("sensor_type") or metadata.get("source") or metadata.get("data_source") or "unknown")
    sensor_id = str(report.get("sensor_id") or report.get("report_id") or sensor_type)
    report_date = report.get("report_date") or report.get("timestamp") or metadata.get("date_str")

    # Keep only compact, coverage-relevant data keys.
    compact_data: Dict[str, Any] = {}
    interesting_data_keys = {
        "pm25", "pm2_5", "pm10", "aqi",
        "avg_speed", "speed", "occupancy", "flow",
        "temperature", "temperature_f", "humidity", "wind_speed", "wind_speed_mph", "precipitation",
        "title", "text", "message", "body", "content", "description", "summary",
    }
    for key, value in data.items():
        key_text = str(key)
        if key_text.lower() not in interesting_data_keys:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            if isinstance(value, str) and len(value) > 500:
                value = value[:500]
            compact_data[key_text] = value

    # A small hint helps coverage_metrics infer camera modality, without storing
    # or copying the actual image.
    if data.get("image_filepath"):
        compact_data["image_filepath"] = "<image_filepath_present>"

    compact_metadata: Dict[str, Any] = {}
    interesting_metadata_keys = {
        "date_str", "source_file", "source_row_index", "source_file_index",
        "modality", "real_data", "observation_location_method",
        "raw_text_location", "text_location_geocoded", "text_location_query",
        "text_location_geocode_source", "text_location_geocode_method", "text_location_cache_hit",
        "camera_latitude", "camera_longitude", "camera_bearing_degrees",
        "camera_fov_degrees", "camera_view_distance_km",
    }
    for key, value in metadata.items():
        key_text = str(key)
        if key_text not in interesting_metadata_keys:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            if isinstance(value, str) and len(value) > 500:
                value = value[:500]
            compact_metadata[key_text] = value

    compact_loc = {
        "latitude": loc.get("latitude"),
        "longitude": loc.get("longitude"),
    }
    for key in ("location_name", "location_description", "location_query", "location_representation", "weather_location_representation"):
        if key in loc and loc.get(key) not in (None, ""):
            value = loc.get(key)
            compact_loc[key] = value[:500] if isinstance(value, str) and len(value) > 500 else value

    return {
        "report_id": str(report.get("report_id") or f"{sensor_type}_{sensor_id}_{report_date}"),
        "report_date": report_date,
        "sensor_id": sensor_id,
        "sensor_name": str(report.get("sensor_name") or sensor_id),
        "sensor_type": sensor_type,
        "location": compact_loc,
        "data": compact_data,
        "metadata": compact_metadata,
        "coverage_source_compact": True,
    }


def _coverage_record_dedupe_key(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    loc = record.get("location") if isinstance(record.get("location"), Mapping) else {}
    metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    report_date = str(record.get("report_date") or metadata.get("date_str") or "")
    date_key = report_date[:10] if len(report_date) >= 10 else report_date
    lat = as_float(loc.get("latitude")) if isinstance(loc, Mapping) else None
    lon = as_float(loc.get("longitude")) if isinstance(loc, Mapping) else None
    return (
        record.get("sensor_type"),
        record.get("sensor_id"),
        date_key,
        round(lat, 4) if lat is not None else None,
        round(lon, 4) if lon is not None else None,
    )


def write_coverage_source_jsonl(path: str | Path, reports: Sequence[Mapping[str, Any]], *, dedupe: bool = True) -> int:
    """Write compact non-circular source inventory rows for coverage metrics."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    written = 0
    with out_path.open("w", encoding="utf-8") as outfile:
        for report in reports or []:
            record = compact_coverage_source_record(report)
            if record is None:
                continue
            if dedupe:
                key = _coverage_record_dedupe_key(record)
                if key in seen:
                    continue
                seen.add(key)
            outfile.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
    return written


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile/replay real extracted data from evaluation/temp/<source>/<YYYYMMDD>. "
            "When emitting, reports are globally sorted by report_date and sent to full_pipeline."
        )
    )
    parser.add_argument("--dates", nargs="+", default=None, help="One or more YYYYMMDD dates to replay/profile.")
    parser.add_argument("--dates-file", default=None, help="Text file containing YYYYMMDD dates.")
    parser.add_argument(
        "--date-set",
        choices=["default", "non"],
        default="default",
        help="Use built-in default incident dates or built-in non-incident dates when --dates is not supplied.",
    )
    parser.add_argument("--temp-root", default=None, help="Root containing <data_source>/<YYYYMMDD>/ folders.")
    parser.add_argument(
        "--raw-root",
        default=None,
        help="Archive root containing <data_source>/<YYYYMMDD>.tar files.",
    )
    parser.add_argument("--data-sources", nargs="+", default=None, help="Data sources to include.")
    parser.add_argument("--emit-to-socket", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--write-ordered-reports-jsonl", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ordered-reports-jsonl", default=None)
    parser.add_argument("--write-coverage-source-jsonl", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--coverage-source-jsonl", default=None)
    parser.add_argument("--coverage-source-dedupe", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--geocode-text-locations", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--text-location-cache-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--text-location-cache-path", default=None)
    parser.add_argument("--socket-host", default=None)
    parser.add_argument("--socket-port", type=int, default=None)
    parser.add_argument("--wait-for-socket-ack", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--replay-interval-seconds", type=float, default=None)
    parser.add_argument("--print-replayed-reports", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--send-stream-control-messages",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Send stream_start/stream_end control messages around the replay. "
            "Keep enabled with updated full_pipeline; disable for older pipelines."
        ),
    )
    parser.add_argument("--auto-prepare-temp-data", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--clear-temp-before-prepare",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether raw-data preparation may clear temp_root before extracting. Default: false.",
    )
    parser.add_argument(
        "--skip-existing-temp-data",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip source/date temp folders that already contain extracted files. Default: true.",
    )
    parser.add_argument(
        "--cctv-unavailable-image-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Cache per-image CCTV unavailable-placeholder decisions. "
            "This avoids reopening every CCTV image when a full source/date profile cache is missing."
        ),
    )
    parser.add_argument(
        "--pems-station-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Cache parsed PeMS station metadata for reuse across dates and future runs. Default: true.",
    )
    parser.add_argument(
        "--pems-profile-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Cache normalized PeMS source/date profiles, including hourly downsampled reports. "
            "This avoids reparsing pem_data_station_5min files on later runs. Default: true."
        ),
    )
    parser.add_argument(
        "--emit-cached-positive-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "For real-data replay, emit only reports whose existing observation-model cache "
            "contains observed_effects or possible_incidents. This is intended for fast "
            "real-data experiments after the observation cache has already been built."
        ),
    )
    parser.add_argument(
        "--observation-cache-root",
        default=None,
        help=(
            "Observation-model real-data cache root used by --emit-cached-positive-only. "
            "Default: detection/cache/observation_model/real_data."
        ),
    )
    parser.add_argument(
        "--cached-positive-missing-policy",
        choices=["keep_all", "drop_all"],
        default="keep_all",
        help=(
            "What to do when --emit-cached-positive-only finds no positive cache keys. "
            "keep_all preserves old behavior; drop_all is useful for cache-only baseline runs."
        ),
    )
    parser.add_argument(
        "--emit-anomaly-topk-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "For real-data replay, emit only observations selected by a prior "
            "full_pipeline anomaly-only top-K run. Reads "
            "<anomaly-topk-root>/<YYYYMMDD>/selected_observations.jsonl and sends "
            "those normalized reports directly to full_pipeline."
        ),
    )
    parser.add_argument(
        "--anomaly-topk-root",
        default=None,
        help=(
            "Root containing anomaly-only selected observations. Default: "
            "detection/outputs/anomaly/batch_runs."
        ),
    )
    parser.add_argument(
        "--anomaly-topk-missing-policy",
        choices=["error", "drop_all"],
        default="error",
        help=(
            "What to do when --emit-anomaly-topk-only finds no selected observations. "
            "error fails loudly; drop_all emits zero reports."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    # ------------------------------------------------------------------
    # Edit these settings directly. Command-line arguments below can override
    # dates, sources, socket emission, and TEMP_ROOT.
    # ------------------------------------------------------------------

    config = get_config()
    configured_paths = config.get("paths", {})
    TEMP_ROOT = Path(configured_paths.get("evaluation_temp_root", "evaluation/temp"))

    # Optional: prepare TEMP_ROOT directly from raw TARs before profiling/replay.
    # This calls data_process.copy_and_untar_raw_data_to_temp(...). By default it
    # preserves TEMP_ROOT and skips source/date folders that already contain
    # extracted data, so repeated runs do not re-untar existing dates.
    AUTO_PREPARE_TEMP_DATA = True
    RAW_ROOT = Path(configured_paths.get("raw_archive_root", "./raw_data"))
    DATA_PROCESS_PATH: Optional[Path] = None
    KEEP_COPIED_TARS = False
    STRICT_MISSING_TARS = False
    CLEAR_TEMP_BEFORE_PREPARE = False
    SKIP_EXISTING_TEMP_DATA = True

    # Cache normalized source/date profiles and report objects after the first
    # scan.  This is especially important for CCTV: without this cache, each run
    # still walks every image and opens each file for the unavailable-camera
    # similarity check, even when the extracted data already exists.
    PROFILE_REPORT_CACHE_ENABLED = True
    PROFILE_REPORT_CACHE_DIR = TEMP_ROOT / DEFAULT_PROFILE_REPORT_CACHE_DIRNAME
    PROFILE_REPORT_CACHE_SOURCES = {"cctv", "pem_data_station_5min"}
    PROFILE_REPORT_CACHE_READ = True
    PROFILE_REPORT_CACHE_WRITE = True
    CCTV_UNAVAILABLE_IMAGE_DECISION_CACHE_ENABLED = True
    CCTV_UNAVAILABLE_IMAGE_DECISION_CACHE_PATH = cctv_unavailable_decision_cache_path(PROFILE_REPORT_CACHE_DIR)
    PEMS_STATION_CACHE_ENABLED = True
    PEMS_STATION_CACHE_READ = True
    PEMS_STATION_CACHE_WRITE = True
    PEMS_STATION_CACHE_DIR = PROFILE_REPORT_CACHE_DIR

    # Leave empty to auto-detect date folders under evaluation/temp/<source>/.
    # If AUTO_PREPARE_TEMP_DATA=True, this must be filled explicitly.
    # DATE_STRINGS: List[str] = []
    # Example:
    DATE_STRINGS = ["20250106",
  "20250107",
  "20250108",
  "20250109",
  "20250110",
  "20250111",
  "20250112",
  "20250113",
  "20250114",
  "20250115",
  "20250116",
  "20250117",
  "20250118",
  "20250119",
  "20250120",
  "20250121",
  "20250122",
  "20250123",
  "20250124",
  "20250125",
  "20250126",
  "20250127",
  "20250128",
  "20250129",
  "20250130",
  "20250131",
  "20250201",
  "20250202",
  "20250328",
  "20250329",
  "20250404",
  "20250405",
  "20250430",
  "20250501",
  "20250605",
  "20250606",
  "20250607",
  "20250608",
  "20250609",
  "20250610",
  "20250611",
  "20250612",
  "20250630",
  "20250701",
  "20250702",
  "20250703",
  "20250704",
  "20250706",
  "20250707",
  "20250708",
  "20250710",
  "20250711",
  "20250712",
  "20250811",
  "20250812",
  "20260129",
  "20260130",
  "20260131",
  "20260212",
  "20260213",
  "20260214"]

    # Built-in non-incident control dates.
    DATE_STRINGS_NON = [
        "20250221",
        "20250222",
        "20250223",
        "20250224",
        "20250225",
        "20250226",
        "20250227",
        "20250228",
        "20250301",
        "20250302",
        "20250303",
        "20250304",
        "20250305",
        "20250306",
    ]

    # Keep real_emitter aligned with evaluation/run_experiments.py defaults.
    # The local lists above are retained only as a fallback if run_experiments
    # cannot be imported in the current environment.
    DATE_STRINGS = load_run_experiments_date_set("default", DATE_STRINGS)
    DATE_STRINGS_NON = load_run_experiments_date_set("non", DATE_STRINGS_NON)

    DATA_SOURCES = [
        "air_data",
        "alertcalifornia",
        "cctv",
        "citizen_data",
        "pem_data_station_5min",
        "twitter_data",
        "weather_data",
    ]

    # ------------------------------------------------------------------
    # Command-line overrides for experiment orchestration.
    # ------------------------------------------------------------------
    if args.temp_root:
        TEMP_ROOT = Path(args.temp_root)
        PROFILE_REPORT_CACHE_DIR = TEMP_ROOT / DEFAULT_PROFILE_REPORT_CACHE_DIRNAME
        CCTV_UNAVAILABLE_IMAGE_DECISION_CACHE_PATH = cctv_unavailable_decision_cache_path(PROFILE_REPORT_CACHE_DIR)
        PEMS_STATION_CACHE_DIR = PROFILE_REPORT_CACHE_DIR
    if args.raw_root:
        RAW_ROOT = Path(args.raw_root)

    if args.auto_prepare_temp_data is not None:
        AUTO_PREPARE_TEMP_DATA = bool(args.auto_prepare_temp_data)
    if args.clear_temp_before_prepare is not None:
        CLEAR_TEMP_BEFORE_PREPARE = bool(args.clear_temp_before_prepare)
    if args.skip_existing_temp_data is not None:
        SKIP_EXISTING_TEMP_DATA = bool(args.skip_existing_temp_data)
    if args.cctv_unavailable_image_cache is not None:
        CCTV_UNAVAILABLE_IMAGE_DECISION_CACHE_ENABLED = bool(args.cctv_unavailable_image_cache)
    if args.pems_station_cache is not None:
        PEMS_STATION_CACHE_ENABLED = bool(args.pems_station_cache)
    if args.pems_profile_cache is not None:
        if bool(args.pems_profile_cache):
            PROFILE_REPORT_CACHE_SOURCES.add("pem_data_station_5min")
        else:
            PROFILE_REPORT_CACHE_SOURCES.discard("pem_data_station_5min")

    cli_dates: List[str] = []
    if args.dates_file:
        cli_dates.extend(_read_date_file(args.dates_file))
    if args.dates:
        cli_dates.extend(args.dates)
    if cli_dates:
        DATE_STRINGS = _normalize_date_list(cli_dates)
    elif args.date_set == "non":
        DATE_STRINGS = list(DATE_STRINGS_NON)

    if args.data_sources:
        DATA_SOURCES = [str(x) for x in args.data_sources]

    # If top-K replay is requested, it can run directly from
    # detection/outputs/anomaly/batch_runs and does not need raw TAR extraction
    # or evaluation/temp profiling.  Use the parsed arg here rather than
    # EMIT_ANOMALY_TOPK_ONLY because the replay-mode globals are initialized
    # later in this function.
    if bool(args.emit_anomaly_topk_only) and args.auto_prepare_temp_data is None:
        AUTO_PREPARE_TEMP_DATA = False

    if AUTO_PREPARE_TEMP_DATA:
        prepare_temp_data_with_data_process(
            date_strings=DATE_STRINGS,
            data_sources=DATA_SOURCES,
            raw_root=RAW_ROOT,
            temp_root=TEMP_ROOT,
            data_process_path=DATA_PROCESS_PATH,
            keep_tar=KEEP_COPIED_TARS,
            strict=STRICT_MISSING_TARS,
            clear_temp=CLEAR_TEMP_BEFORE_PREPARE,
            skip_existing=SKIP_EXISTING_TEMP_DATA,
        )

    # Profiling output files.
    OUTPUT_SUMMARY_JSON = Path("evaluation/temp/real_emitter_data_summary.json")
    OUTPUT_SAMPLE_REPORTS_JSONL = Path("evaluation/temp/real_emitter_sample_reports.jsonl")
    SAMPLE_REPORTS_PER_SOURCE_DATE = 3

    # PeMS 5-minute traffic downsampling. When True, each station keeps only the
    # first row in each hour, so a full day is about 24 records per station.
    PEMS_DOWNSAMPLE_TO_HOURLY = True

    # Only PeMS stations with metadata locations inside this bounding box are
    # counted for replay/emission. This also affects PeMS profiling so the
    # profile reflects the stream you would actually send downstream.
    PEMS_REQUIRE_SPATIAL_BOUNDS = True
    PEMS_SPATIAL_BOUNDS = {
        "lat_min": 33.9,
        "lon_min": -118.95,
        "lat_max": 34.35,
        "lon_max": -118.0,
    }

    # Actual emission remains off by default. If either of these is enabled, the
    # script will collect full normalized reports and sort them globally by time.
    EMIT_TO_SOCKET = True
    WRITE_ORDERED_REPORTS_JSONL = False
    ORDERED_REPORTS_JSONL = Path("evaluation/temp/real_emitter_ordered_reports.jsonl")
    WRITE_COVERAGE_SOURCE_JSONL = False
    COVERAGE_SOURCE_JSONL = Path("evaluation/temp/real_emitter_coverage_sources.jsonl")
    COVERAGE_SOURCE_DEDUPE = True
    GEOCODE_TEXT_LOCATIONS = False
    TEXT_LOCATION_CACHE_ONLY = False
    TEXT_LOCATION_CACHE_PATH = Path("evaluation/geo_cache/geo_region_cache.json")

    SOCKET_HOST = "127.0.0.1"
    SOCKET_PORT = 8765
    WAIT_FOR_SOCKET_ACK = True
    REPLAY_INTERVAL_SECONDS = 0.0
    PRINT_REPLAYED_REPORTS = False
    SEND_STREAM_CONTROL_MESSAGES = True
    EMIT_CACHED_POSITIVE_ONLY = False
    OBSERVATION_CACHE_ROOT = Path(configured_paths.get("real_observation_cache_root", DEFAULT_OBSERVATION_CACHE_ROOT))
    CACHED_POSITIVE_MISSING_POLICY = "keep_all"
    EMIT_ANOMALY_TOPK_ONLY = False
    ANOMALY_TOPK_ROOT = Path("detection/outputs/anomaly/batch_runs")
    ANOMALY_TOPK_MISSING_POLICY = "error"

    if args.emit_to_socket is not None:
        EMIT_TO_SOCKET = bool(args.emit_to_socket)
    if args.write_ordered_reports_jsonl is not None:
        WRITE_ORDERED_REPORTS_JSONL = bool(args.write_ordered_reports_jsonl)
    if args.ordered_reports_jsonl:
        ORDERED_REPORTS_JSONL = Path(args.ordered_reports_jsonl)
    if args.write_coverage_source_jsonl is not None:
        WRITE_COVERAGE_SOURCE_JSONL = bool(args.write_coverage_source_jsonl)
    if args.coverage_source_jsonl:
        COVERAGE_SOURCE_JSONL = Path(args.coverage_source_jsonl)
    if args.coverage_source_dedupe is not None:
        COVERAGE_SOURCE_DEDUPE = bool(args.coverage_source_dedupe)
    if args.geocode_text_locations is not None:
        GEOCODE_TEXT_LOCATIONS = bool(args.geocode_text_locations)
    elif WRITE_COVERAGE_SOURCE_JSONL:
        GEOCODE_TEXT_LOCATIONS = True
    if args.text_location_cache_only is not None:
        TEXT_LOCATION_CACHE_ONLY = bool(args.text_location_cache_only)
    if args.text_location_cache_path:
        TEXT_LOCATION_CACHE_PATH = Path(args.text_location_cache_path)
    if args.socket_host:
        SOCKET_HOST = str(args.socket_host)
    if args.socket_port is not None:
        SOCKET_PORT = int(args.socket_port)
    if args.wait_for_socket_ack is not None:
        WAIT_FOR_SOCKET_ACK = bool(args.wait_for_socket_ack)
    if args.replay_interval_seconds is not None:
        REPLAY_INTERVAL_SECONDS = float(args.replay_interval_seconds)
    if args.print_replayed_reports is not None:
        PRINT_REPLAYED_REPORTS = bool(args.print_replayed_reports)
    if args.send_stream_control_messages is not None:
        SEND_STREAM_CONTROL_MESSAGES = bool(args.send_stream_control_messages)

    if args.emit_cached_positive_only is not None:
        EMIT_CACHED_POSITIVE_ONLY = bool(args.emit_cached_positive_only)
    if args.observation_cache_root:
        OBSERVATION_CACHE_ROOT = Path(args.observation_cache_root)
    if args.cached_positive_missing_policy:
        CACHED_POSITIVE_MISSING_POLICY = str(args.cached_positive_missing_policy)
    if args.emit_anomaly_topk_only is not None:
        EMIT_ANOMALY_TOPK_ONLY = bool(args.emit_anomaly_topk_only)
    if args.anomaly_topk_root:
        ANOMALY_TOPK_ROOT = Path(args.anomaly_topk_root)
    if args.anomaly_topk_missing_policy:
        ANOMALY_TOPK_MISSING_POLICY = str(args.anomaly_topk_missing_policy)


    # Location metadata used by the source-specific profilers.
    CCTV_KML_PATH = Path("evaluation/sensor_locations/cctv.kml")

    # Skip CCTV placeholder images such as sensor_locations/unavailable.jpg.
    # This uses a perceptual grayscale similarity check, not exact bytes, because
    # the text rendered on the unavailable image can differ across pulls.
    SKIP_CCTV_UNAVAILABLE_IMAGES = True
    CCTV_UNAVAILABLE_IMAGE_PATH_CANDIDATES = [
        Path("evaluation/sensor_locations/unavailable.jpg"),
        Path("sensor_locations/unavailable.jpg"),
    ]

    ALERTCALIFORNIA_LOCATIONS_PATH = Path("evaluation/sensor_locations/alertcalifornia_locations.json")
    PEMS_STATION_PATH_CANDIDATES = [
        Path("evaluation/sensor_locations/pem_7_stations.txt"),
        Path("evaluation/sensor_locations/pems_y_stations.txt"),
        Path("evaluation/sensor_locations/y_stations.txt"),
    ]

    # Weather geocoding can be slow and is not needed to answer "how much data".
    GEOCODE_WEATHER_LOCATIONS = False

    # ------------------------------------------------------------------
    # Execution.
    # ------------------------------------------------------------------

    collect_for_replay = EMIT_TO_SOCKET or WRITE_ORDERED_REPORTS_JSONL or WRITE_COVERAGE_SOURCE_JSONL

    if not TEMP_ROOT.exists() and not (collect_for_replay and EMIT_ANOMALY_TOPK_ONLY):
        raise FileNotFoundError(f"TEMP_ROOT does not exist: {TEMP_ROOT}")

    dates = list(DATE_STRINGS) if (collect_for_replay and EMIT_ANOMALY_TOPK_ONLY) else resolve_dates(TEMP_ROOT, DATA_SOURCES, DATE_STRINGS)
    if not dates:
        if collect_for_replay and EMIT_ANOMALY_TOPK_ONLY:
            raise FileNotFoundError(
                "No dates were supplied for anomaly top-K replay. Pass --dates or --dates-file."
            )
        raise FileNotFoundError(
            f"No date folders found under {TEMP_ROOT}/<source>/. "
            "Expected paths like evaluation/temp/cctv/20250102/..."
        )

    if collect_for_replay and EMIT_ANOMALY_TOPK_ONLY:
        replay_reports, topk_stats = load_anomaly_topk_selected_reports(
            anomaly_topk_root=ANOMALY_TOPK_ROOT,
            date_strings=dates,
            data_sources=DATA_SOURCES,
            missing_policy=ANOMALY_TOPK_MISSING_POLICY,
        )
        OUTPUT_SUMMARY_JSON = Path("evaluation/temp/real_emitter_data_summary.json")
        ORDERED_REPORTS_JSONL = Path("evaluation/temp/real_emitter_ordered_reports.jsonl")
        if args.ordered_reports_jsonl:
            ORDERED_REPORTS_JSONL = Path(args.ordered_reports_jsonl)

        output = {
            "temp_root": str(TEMP_ROOT),
            "dates": dates,
            "sources": DATA_SOURCES,
            "emission_enabled": EMIT_TO_SOCKET,
            "ordered_replay_enabled": collect_for_replay,
            "ordered_reports_path": str(ORDERED_REPORTS_JSONL) if WRITE_ORDERED_REPORTS_JSONL else None,
            "send_stream_control_messages": SEND_STREAM_CONTROL_MESSAGES,
            "emit_cached_positive_only": False,
            "emit_anomaly_topk_only": True,
            "anomaly_topk_root": str(ANOMALY_TOPK_ROOT),
            "anomaly_topk_missing_policy": ANOMALY_TOPK_MISSING_POLICY,
            "anomaly_topk_filter": topk_stats,
        }
        write_json(OUTPUT_SUMMARY_JSON, output)
        log(
            "Anomaly top-K replay filter: "
            f"loaded={topk_stats.get('reports_loaded')}, "
            f"rows_read={topk_stats.get('rows_read')}, "
            f"missing_files={topk_stats.get('date_files_missing')}, "
            f"root={ANOMALY_TOPK_ROOT}"
        )
        emitted_count = replay_reports_in_temporal_order(
            replay_reports,
            output_jsonl_path=ORDERED_REPORTS_JSONL if WRITE_ORDERED_REPORTS_JSONL else None,
            emit_to_socket=EMIT_TO_SOCKET,
            socket_host=SOCKET_HOST,
            socket_port=SOCKET_PORT,
            wait_for_socket_ack=WAIT_FOR_SOCKET_ACK,
            interval_seconds=REPLAY_INTERVAL_SECONDS,
            print_reports=PRINT_REPLAYED_REPORTS,
            send_stream_control_messages=SEND_STREAM_CONTROL_MESSAGES,
        )
        print(f"Replayed/wrote {emitted_count} anomaly top-K selected reports in global temporal order.")
        print(f"Wrote summary JSON: {OUTPUT_SUMMARY_JSON}")
        if WRITE_ORDERED_REPORTS_JSONL:
            print(f"Wrote ordered reports JSONL: {ORDERED_REPORTS_JSONL}")
        return 0

    sensor_coverage = maybe_load_sensor_coverage_module()
    cctv_unavailable_reference_path = resolve_existing_path(CCTV_UNAVAILABLE_IMAGE_PATH_CANDIDATES)
    cctv_unavailable_signature = (
        image_unavailable_signature(
            cctv_unavailable_reference_path,
            size=DEFAULT_CCTV_UNAVAILABLE_SIGNATURE_SIZE,
        )
        if SKIP_CCTV_UNAVAILABLE_IMAGES and cctv_unavailable_reference_path is not None
        else None
    )
    if SKIP_CCTV_UNAVAILABLE_IMAGES and cctv_unavailable_signature is None:
        log(
            "WARNING: CCTV unavailable-image filter is enabled, but no usable "
            f"reference image was found. Tried: {CCTV_UNAVAILABLE_IMAGE_PATH_CANDIDATES}"
        )

    cctv_unavailable_decision_cache: Optional[Dict[str, Dict[str, Any]]] = None
    if (
        SKIP_CCTV_UNAVAILABLE_IMAGES
        and cctv_unavailable_signature is not None
        and CCTV_UNAVAILABLE_IMAGE_DECISION_CACHE_ENABLED
    ):
        cctv_unavailable_decision_cache = load_cctv_unavailable_decision_cache(
            CCTV_UNAVAILABLE_IMAGE_DECISION_CACHE_PATH
        )
        log(
            "Loaded CCTV unavailable-image decision cache: "
            f"path={CCTV_UNAVAILABLE_IMAGE_DECISION_CACHE_PATH}, "
            f"entries={len(cctv_unavailable_decision_cache)}"
        )

    text_location_resolver = TextLocationResolver(
        cache_path=TEXT_LOCATION_CACHE_PATH,
        enabled=GEOCODE_TEXT_LOCATIONS,
        cache_only=TEXT_LOCATION_CACHE_ONLY,
    )

    profiles: List[DateProfile] = []

    log(f"Profiling sources={DATA_SOURCES}")
    log(f"Profiling dates={dates}")
    log(f"Reading from TEMP_ROOT={TEMP_ROOT}")
    log(f"PeMS hourly downsampling enabled={PEMS_DOWNSAMPLE_TO_HOURLY}")
    log(f"PeMS spatial bounds enabled={PEMS_REQUIRE_SPATIAL_BOUNDS}; bounds={PEMS_SPATIAL_BOUNDS}")
    log(f"Collecting full reports for ordered replay/source inventory={collect_for_replay}")
    log(f"Writing compact coverage source JSONL={WRITE_COVERAGE_SOURCE_JSONL}; path={COVERAGE_SOURCE_JSONL}")
    log(f"Text-location geocoding enabled={GEOCODE_TEXT_LOCATIONS}; cache_only={TEXT_LOCATION_CACHE_ONLY}; cache={TEXT_LOCATION_CACHE_PATH}")

    source_date_pairs = [(source, date_str) for source in DATA_SOURCES for date_str in dates]
    for source, date_str in source_date_pairs:
        use_profile_cache = (
            PROFILE_REPORT_CACHE_ENABLED
            and source in PROFILE_REPORT_CACHE_SOURCES
        )

        profile: Optional[DateProfile] = None
        if use_profile_cache and PROFILE_REPORT_CACHE_READ:
            profile = load_profile_report_cache(PROFILE_REPORT_CACHE_DIR, source, date_str)
            if profile is not None:
                cache_metadata = profile.extra.get("profile_report_cache_metadata") if isinstance(profile.extra, dict) else {}
                cache_metadata = cache_metadata if isinstance(cache_metadata, dict) else {}
                cache_usable = True
                cache_reasons: List[str] = []
                if collect_for_replay and not profile.emission_reports:
                    cache_usable = False
                    cache_reasons.append("cache_has_no_emission_reports_but_replay_is_enabled")
                if source == "pem_data_station_5min":
                    expected_pems_metadata = {
                        "pems_downsample_to_hourly": PEMS_DOWNSAMPLE_TO_HOURLY,
                        "pems_require_spatial_bounds": PEMS_REQUIRE_SPATIAL_BOUNDS,
                        "pems_spatial_bounds": PEMS_SPATIAL_BOUNDS,
                    }
                    for meta_key, expected_value in expected_pems_metadata.items():
                        if cache_metadata.get(meta_key) != expected_value:
                            cache_usable = False
                            cache_reasons.append(f"{meta_key}_changed")
                if cache_usable:
                    log(
                        f"{source}/{date_str}: loaded normalized reports from cache "
                        f"({profile.extra.get('profile_report_cache_path')})"
                    )
                else:
                    log(
                        f"{source}/{date_str}: ignoring stale/incompatible normalized report cache "
                        f"({profile.extra.get('profile_report_cache_path')}): {cache_reasons}"
                    )
                    profile = None

        if profile is None:
            profile = profile_source_date(
                temp_root=TEMP_ROOT,
                source=source,
                date_str=date_str,
                sample_limit=SAMPLE_REPORTS_PER_SOURCE_DATE,
                collect_for_replay=collect_for_replay,
                sensor_coverage=sensor_coverage,
                cctv_kml_path=CCTV_KML_PATH,
                cctv_skip_unavailable_images=SKIP_CCTV_UNAVAILABLE_IMAGES,
                cctv_unavailable_signature=cctv_unavailable_signature,
                cctv_unavailable_decision_cache=cctv_unavailable_decision_cache,
                cctv_unavailable_decision_cache_path=CCTV_UNAVAILABLE_IMAGE_DECISION_CACHE_PATH,
                cctv_use_unavailable_decision_cache=CCTV_UNAVAILABLE_IMAGE_DECISION_CACHE_ENABLED,
                alertcalifornia_locations_path=ALERTCALIFORNIA_LOCATIONS_PATH,
                pems_station_path_candidates=PEMS_STATION_PATH_CANDIDATES,
                pems_downsample_to_hourly=PEMS_DOWNSAMPLE_TO_HOURLY,
                pems_spatial_bounds=PEMS_SPATIAL_BOUNDS,
                pems_require_spatial_bounds=PEMS_REQUIRE_SPATIAL_BOUNDS,
                pems_station_cache_dir=PEMS_STATION_CACHE_DIR,
                pems_station_cache_enabled=PEMS_STATION_CACHE_ENABLED,
                pems_station_cache_read=PEMS_STATION_CACHE_READ,
                pems_station_cache_write=PEMS_STATION_CACHE_WRITE,
                geocode_weather_locations=GEOCODE_WEATHER_LOCATIONS,
                text_location_resolver=text_location_resolver,
                geocode_text_locations=GEOCODE_TEXT_LOCATIONS,
            )
            if (
                profile is not None
                and use_profile_cache
                and PROFILE_REPORT_CACHE_WRITE
            ):
                save_profile_report_cache(
                    PROFILE_REPORT_CACHE_DIR,
                    profile,
                    cache_metadata={
                        "source": source,
                        "date_str": date_str,
                        "collect_for_replay": collect_for_replay,
                        "sample_limit": SAMPLE_REPORTS_PER_SOURCE_DATE,
                        "skip_cctv_unavailable_images": SKIP_CCTV_UNAVAILABLE_IMAGES,
                        "cctv_unavailable_reference_image": str(cctv_unavailable_reference_path) if cctv_unavailable_reference_path else None,
                        "pems_downsample_to_hourly": PEMS_DOWNSAMPLE_TO_HOURLY,
                        "pems_require_spatial_bounds": PEMS_REQUIRE_SPATIAL_BOUNDS,
                        "pems_spatial_bounds": PEMS_SPATIAL_BOUNDS,
                        "pems_station_cache_enabled": PEMS_STATION_CACHE_ENABLED,
                        "note": (
                            "This cache stores normalized reports, not raw data. "
                            "Delete this file if the extracted source/date folder changes."
                        ),
                    },
                )
                profile.extra["profile_report_cache_path"] = str(profile_cache_path(PROFILE_REPORT_CACHE_DIR, source, date_str))
                profile.extra["profile_report_cache_status"] = "miss_stored"

        if profile is None:
            continue
        profiles.append(profile)
        summary = summarize_counts(profile.sensor_record_counts)
        cache_note = " cache=hit" if profile.extra.get("loaded_from_profile_report_cache") else ""
        log(
            f"{source}/{date_str}: sensors={summary['num_sensors']} "
            f"records={summary['total_records']} "
            f"avg_per_sensor={summary['records_per_sensor_avg']:.2f}"
            f"{cache_note}"
        )

    aggregate = aggregate_profiles(profiles)
    per_date = [profile.to_json() for profile in profiles]
    sample_reports = [report for profile in profiles for report in profile.sample_reports]
    replay_reports = [report for profile in profiles for report in profile.emission_reports]
    example_report_by_source = first_example_report_by_source(profiles)

    output = {
        "temp_root": str(TEMP_ROOT),
        "dates": dates,
        "sources": DATA_SOURCES,
        "aggregate_by_source": aggregate,
        "per_source_date": per_date,
        "sample_reports_path": str(OUTPUT_SAMPLE_REPORTS_JSONL),
        "example_report_by_source": example_report_by_source,
        "ordered_reports_path": str(ORDERED_REPORTS_JSONL) if WRITE_ORDERED_REPORTS_JSONL else None,
        "coverage_source_jsonl_path": str(COVERAGE_SOURCE_JSONL) if WRITE_COVERAGE_SOURCE_JSONL else None,
        "coverage_source_dedupe": COVERAGE_SOURCE_DEDUPE,
        "text_location_geocoding_enabled": GEOCODE_TEXT_LOCATIONS,
        "text_location_cache_only": TEXT_LOCATION_CACHE_ONLY,
        "text_location_cache_path": str(TEXT_LOCATION_CACHE_PATH),
        "text_location_geocoding_stats": dict(text_location_resolver.stats),
        "emission_enabled": EMIT_TO_SOCKET,
        "ordered_replay_enabled": collect_for_replay,
        "pems_downsample_to_hourly": PEMS_DOWNSAMPLE_TO_HOURLY,
        "pems_downsample_policy": "first_record_per_station_per_hour" if PEMS_DOWNSAMPLE_TO_HOURLY else "none",
        "pems_require_spatial_bounds": PEMS_REQUIRE_SPATIAL_BOUNDS,
        "pems_spatial_bounds": PEMS_SPATIAL_BOUNDS,
        "weather_location_representation": "area_description_not_point",
        "skip_cctv_unavailable_images": SKIP_CCTV_UNAVAILABLE_IMAGES,
        "cctv_unavailable_reference_image": str(cctv_unavailable_reference_path) if cctv_unavailable_reference_path else None,
        "cctv_unavailable_filter_active": cctv_unavailable_signature is not None,
        "profile_report_cache_enabled": PROFILE_REPORT_CACHE_ENABLED,
        "profile_report_cache_dir": str(PROFILE_REPORT_CACHE_DIR),
        "profile_report_cache_sources": sorted(PROFILE_REPORT_CACHE_SOURCES),
        "cctv_unavailable_image_decision_cache_enabled": CCTV_UNAVAILABLE_IMAGE_DECISION_CACHE_ENABLED,
        "cctv_unavailable_image_decision_cache_path": str(CCTV_UNAVAILABLE_IMAGE_DECISION_CACHE_PATH),
        "cctv_unavailable_image_decision_cache_entries_loaded": (
            len(cctv_unavailable_decision_cache) if isinstance(cctv_unavailable_decision_cache, dict) else 0
        ),
        "pems_station_cache_enabled": PEMS_STATION_CACHE_ENABLED,
        "pems_station_cache_dir": str(PEMS_STATION_CACHE_DIR),
        "pems_station_cache_path": str(pems_station_cache_path(PEMS_STATION_CACHE_DIR)),
        "pems_profile_cache_enabled": "pem_data_station_5min" in PROFILE_REPORT_CACHE_SOURCES,
        "send_stream_control_messages": SEND_STREAM_CONTROL_MESSAGES,
        "emit_cached_positive_only": EMIT_CACHED_POSITIVE_ONLY,
        "observation_cache_root": str(OBSERVATION_CACHE_ROOT),
        "cached_positive_missing_policy": CACHED_POSITIVE_MISSING_POLICY,
        "emit_anomaly_topk_only": EMIT_ANOMALY_TOPK_ONLY,
        "anomaly_topk_root": str(ANOMALY_TOPK_ROOT),
        "anomaly_topk_missing_policy": ANOMALY_TOPK_MISSING_POLICY,
    }

    write_json(OUTPUT_SUMMARY_JSON, output)
    write_jsonl(OUTPUT_SAMPLE_REPORTS_JSONL, sample_reports)
    if WRITE_COVERAGE_SOURCE_JSONL:
        coverage_count = write_coverage_source_jsonl(
            COVERAGE_SOURCE_JSONL,
            replay_reports,
            dedupe=COVERAGE_SOURCE_DEDUPE,
        )
        output["coverage_source_jsonl_records"] = coverage_count
        output["text_location_geocoding_stats"] = dict(text_location_resolver.stats)
        write_json(OUTPUT_SUMMARY_JSON, output)

    print_summary_table(aggregate)
    print()
    print_example_reports_by_source(example_report_by_source)
    print()
    print(f"Wrote summary JSON: {OUTPUT_SUMMARY_JSON}")
    print(f"Wrote sample normalized reports JSONL: {OUTPUT_SAMPLE_REPORTS_JSONL}")
    if WRITE_COVERAGE_SOURCE_JSONL:
        print(f"Wrote compact coverage source JSONL: {COVERAGE_SOURCE_JSONL}")

    if collect_for_replay and EMIT_CACHED_POSITIVE_ONLY:
        replay_reports, positive_filter_stats = filter_reports_to_cached_positives(
            replay_reports,
            observation_cache_root=OBSERVATION_CACHE_ROOT,
            data_sources=DATA_SOURCES,
            date_strings=dates,
            missing_policy=CACHED_POSITIVE_MISSING_POLICY,
        )
        output["cached_positive_filter"] = positive_filter_stats
        write_json(OUTPUT_SUMMARY_JSON, output)
        log(
            "Cached-positive replay filter: "
            f"input={positive_filter_stats.get('input_reports')}, "
            f"kept={positive_filter_stats.get('kept_reports')}, "
            f"dropped={positive_filter_stats.get('dropped_reports')}, "
            f"positive_cache_available={positive_filter_stats.get('positive_cache_available')}, "
            f"cache_root={OBSERVATION_CACHE_ROOT}"
        )

    if collect_for_replay:
        emitted_count = replay_reports_in_temporal_order(
            replay_reports,
            output_jsonl_path=ORDERED_REPORTS_JSONL if WRITE_ORDERED_REPORTS_JSONL else None,
            emit_to_socket=EMIT_TO_SOCKET,
            socket_host=SOCKET_HOST,
            socket_port=SOCKET_PORT,
            wait_for_socket_ack=WAIT_FOR_SOCKET_ACK,
            interval_seconds=REPLAY_INTERVAL_SECONDS,
            print_reports=PRINT_REPLAYED_REPORTS,
            send_stream_control_messages=SEND_STREAM_CONTROL_MESSAGES,
        )
        print(f"Replayed/wrote {emitted_count} reports in global temporal order.")
        if WRITE_ORDERED_REPORTS_JSONL:
            print(f"Wrote ordered reports JSONL: {ORDERED_REPORTS_JSONL}")
    else:
        print("Emission is disabled; this run only profiles data and writes sample reports.")
        print("When EMIT_TO_SOCKET or WRITE_ORDERED_REPORTS_JSONL is enabled, reports are globally sorted by parsed report_date datetime before replay; date-only fallbacks sort at midnight.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
