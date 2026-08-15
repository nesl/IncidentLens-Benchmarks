#!/usr/bin/env python3
"""Build cleaned low-level and optional top-level incident files from all_merged_by_id_relevant.json.

This version intentionally starts from the source-event evidence in
`all_merged_by_id_relevant.json` rather than from a previous `low_level.json`, because
the first-pass low-level file can lose direct evidence for incidents such as the
Eaton Fire or Hurst Fire.

Default behavior:
  1. Read evaluation/merged_incidents/all_merged_by_id_relevant.json.
  2. Extract concrete source-event candidates.
  3. Filter out non-events, adjacent coverage, threats, hearings/trials, and out-of-window events.
  4. Keep aggregate candidates separately, e.g. Los Angeles wildfires, ICE protests.
  5. Merge duplicate concrete low-level incidents, with hard guards against merging distinct places/fires.
  6. Write evaluation/out/low_level.json.
  7. Pause so you can inspect the low-level count.
  8. If you type "continue", write evaluation/out/top_level.json with only multi-child top-level incidents.

Examples:
  python -m evaluation.labelling.news_refine --openai-api-key "$OPENAI_API_KEY" --force

  python evaluation/labelling/news_refine.py \
      --all-merged evaluation/merged_incidents/all_merged_by_id_relevant.json \
      --openai-api-key "$OPENAI_API_KEY" \
      --force

  python evaluation/labelling/news_refine.py --no-llm --force
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

SCRIPT_PATH = Path(__file__).resolve()
if SCRIPT_PATH.parent.name == "labelling" and SCRIPT_PATH.parent.parent.name == "evaluation":
    REPO_ROOT = SCRIPT_PATH.parents[2]
else:
    REPO_ROOT = Path.cwd()

DEFAULT_ALL_MERGED = REPO_ROOT / "evaluation/merged_incidents/all_merged_by_id_relevant.json"
DEFAULT_FILTERED_INCIDENTS = REPO_ROOT / "evaluation/filtered_incidents.txt"
DEFAULT_OUT_DIR = REPO_ROOT / "evaluation/out"
DEFAULT_START_DATE = "2025-01-01"
DEFAULT_END_DATE = "2026-05-31"
PACIFIC = ZoneInfo("America/Los_Angeles")

TARGET_DEFAULTS = ["civil protest", "urban fire", "wildfire", "terrorist incident"]

# Broad Greater LA hints. This is intentionally broad because all_merged_by_id_relevant
# should already be spatially relevant, but we still want to reject obvious non-LA events.
LA_HINTS = {
    "los angeles", "l.a.", "la county", "south la", "east la", "west la", "dtla",
    "pacific palisades", "palisades", "malibu", "topanga", "calabasas", "eaton",
    "altadena", "pasadena", "hurst", "sylmar", "san fernando", "hollywood",
    "west hollywood", "north hollywood", "burbank", "glendale", "santa monica",
    "venice", "long beach", "compton", "inglewood", "torrance", "gardena",
    "hawthorne", "culver city", "santa clarita", "woodland hills", "encino",
    "agoura hills", "la cañada", "la canada", "arcadia", "monrovia", "azusa",
    "pomona", "whittier", "el monte", "montebello", "beverly hills", "san pedro",
    "wilmington", "ventura county", "orange county", "southern california",
}

# Used only as a fallback if text is ambiguous.
LA_BBOX = {
    "lat_min": 33.3,
    "lat_max": 35.0,
    "lon_min": -119.4,
    "lon_max": -117.2,
}

PROCEDURAL_OR_ADJACENT_TERMS = {
    "trial", "lawsuit", "court", "sentencing", "plea", "hearing", "council chamber",
    "committee", "meeting", "board meeting", "conference", "briefing", "press conference",
    "fundraiser", "memorial", "vigil", "donation", "relief", "shelter", "intake",
    "animal rescue", "pet rescue", "horse intake", "equestrian center", "recovery center",
    "assistance center", "resource center", "insurance", "claims", "celebrity", "celebrities",
    "lost homes", "homeowners sue", "leadership test", "response questions", "social media trial",
}

THREAT_OR_NON_OCCURRED_TERMS = {
    "threatened", "threat of", "could strike", "may strike", "planned strike",
    "strike authorization", "vote to authorize", "mediation", "negotiation", "warning of",
    "potential", "possible", "preparing for", "forecast", "expected to", "could happen",
}

DIRECT_EVENT_TERMS = {
    "wildfire", "brush fire", "fire broke out", "fire started", "fire erupted",
    "burning", "evacuation order", "structures destroyed", "protest", "demonstration",
    "rally", "march", "walkout", "strike began", "clash", "riot", "attack", "shooting",
    "bomb", "explosion", "terrorist", "arson",
}

# Specific named fires should not be merged with each other. The regex also catches
# names not listed here, but these are useful top-level linking hints.
KNOWN_FIRE_NAMES = {
    "palisades fire", "eaton fire", "hurst fire", "kenneth fire", "hughes fire",
    "sunset fire", "lydia fire", "franklin fire", "bridge fire", "line fire",
    "airport fire", "lake fire", "post fire", "borel fire", "vista fire",
}

AGGREGATE_HINTS = {
    "los angeles wildfires", "los angeles fires", "la wildfires", "la fires",
    "southern california wildfires", "california wildfires", "palisades and eaton fires",
    "palisades, eaton and hurst fires", "january 2025 los angeles wildfires",
    "ice protests", "anti ice protests", "anti-ice protests", "los angeles ice protests",
    "la ice protests", "immigration protests", "anti immigration protests",
}

WEAK_INDIRECT_DOMAINS = {
    "iraqinews.com", "nbcchicago.com", "nbcmiami.com", "people.com", "pagesix.com",
    "tmz.com", "hollywoodreporter.com", "variety.com",
}

STRONG_NEWS_DOMAINS = {
    "latimes.com", "nbclosangeles.com", "abc7.com", "ktla.com", "cbsnews.com",
    "apnews.com", "reuters.com", "nbcnews.com", "cnn.com", "nytimes.com",
    "washingtonpost.com", "foxla.com", "dailynews.com", "pasadenastarnews.com",
    "sgvtribune.com", "fire.ca.gov", "lafd.org", "lacounty.gov", "lacity.gov",
    "yahoo.com", "courthousenews.com",
}


def load_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=False)
    tmp.replace(path)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = text.replace("&amp;", " and ")
    text = re.sub(r"[^a-z0-9\s.-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_key(value: Any) -> str:
    text = normalize_text(value)
    drop = {
        "the", "a", "an", "incident", "event", "reported", "report", "area", "region",
        "california", "ca", "usa", "united", "states", "county", "city", "of", "in",
        "near", "at", "on", "and", "los", "angeles",
    }
    return " ".join(t for t in text.split() if t not in drop)


def domain_of(url: str | None) -> str:
    if not url:
        return ""
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def load_labels(path: Path) -> list[str]:
    if not path.exists():
        return TARGET_DEFAULTS[:]
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return labels or TARGET_DEFAULTS[:]


def parse_to_pacific_naive(value: Any) -> tuple[str | None, str]:
    """Return (YYYY-MM-DDTHH:MM:SS or None, precision)."""
    if value is None:
        return None, "unknown"
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None, "unknown"

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return f"{text}T00:00:00", "date"

    # Some upstream fields use compact GDELT-like article dates.
    if re.fullmatch(r"\d{14}", text):
        try:
            dt = datetime.strptime(text, "%Y%m%d%H%M%S").replace(tzinfo=ZoneInfo("UTC"))
            pt = dt.astimezone(PACIFIC).replace(tzinfo=None)
            return pt.isoformat(timespec="seconds"), "exact"
        except Exception:
            pass

    clean = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is not None:
            dt = dt.astimezone(PACIFIC).replace(tzinfo=None)
        else:
            dt = dt.replace(tzinfo=None)
        precision = "exact" if (dt.hour or dt.minute or dt.second) else "date"
        return dt.isoformat(timespec="seconds"), precision
    except Exception:
        pass

    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}T00:00:00", "date"
    return None, "unknown"


def date_part(dt_text: str | None) -> date | None:
    if not dt_text:
        return None
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", dt_text)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except Exception:
        return None


def in_window(start_dt: str | None, end_dt: str | None, start: date, end: date) -> bool:
    dates = [d for d in (date_part(start_dt), date_part(end_dt)) if d is not None]
    if not dates:
        return True
    return not (all(d < start for d in dates) or all(d > end for d in dates))


def source_event_text(incident: dict[str, Any], ev: dict[str, Any]) -> str:
    pieces = [
        ev.get("event_name"), ev.get("event_type"), ev.get("event_description"),
        ev.get("reasoning"), ev.get("location"), ev.get("location_name"), ev.get("url"),
        incident.get("representative_event_type"), incident.get("representative_location_name"),
        incident.get("incident_key"), incident.get("incident_names"), incident.get("event_types"),
    ]
    return " ".join(str(p) for p in pieces if p not in (None, "", [], {}))


def map_incident_type(raw_type: Any, text: str, labels: list[str]) -> str | None:
    t = normalize_text(f"{raw_type} {text}")
    label_norm = {normalize_text(x): x for x in labels}

    if normalize_text(raw_type) in label_norm:
        return label_norm[normalize_text(raw_type)]

    if "wildfire" in t or "brush fire" in t or any(name in t for name in KNOWN_FIRE_NAMES):
        return "wildfire" if "wildfire" in labels else None
    if "fire" in t and "wild" not in t and not any(name in t for name in KNOWN_FIRE_NAMES):
        return "urban fire" if "urban fire" in labels else ("wildfire" if "wildfire" in labels else None)
    if any(x in t for x in ["civil protest", "protest", "demonstration", "rally", "march", "walkout"]):
        return "civil protest" if "civil protest" in labels else None
    # Strike is only a protest if it actually occurs, not if it is merely threatened.
    if "strike" in t and not any(x in t for x in ["threatened", "could strike", "may strike", "mediation"]):
        return "civil protest" if "civil protest" in labels else None
    if any(x in t for x in ["terrorist", "terrorism", "bomb", "explosion", "attack"]):
        return "terrorist incident" if "terrorist incident" in labels else None
    return None


def has_la_evidence(incident: dict[str, Any], ev: dict[str, Any]) -> bool:
    text = normalize_text(source_event_text(incident, ev))
    if any(h in text for h in LA_HINTS):
        return True
    lat = incident.get("centroid_lat")
    lon = incident.get("centroid_lon")
    try:
        lat_f = float(lat)
        lon_f = float(lon)
        return LA_BBOX["lat_min"] <= lat_f <= LA_BBOX["lat_max"] and LA_BBOX["lon_min"] <= lon_f <= LA_BBOX["lon_max"]
    except Exception:
        return False


def extract_specific_fire_names(text: Any) -> set[str]:
    t = normalize_text(text)
    out = {name for name in KNOWN_FIRE_NAMES if name in t}
    # Catch other "X Fire" names with capitalized original text as fallback.
    raw = str(text or "")
    for m in re.finditer(r"\b([A-Z][A-Za-z0-9.'-]{2,}(?:\s+[A-Z][A-Za-z0-9.'-]{2,}){0,2})\s+Fire\b", raw):
        name = normalize_text(m.group(0))
        if name not in {"structure fire", "urban fire", "brush fire"}:
            out.add(name)
    return out


def is_aggregate_name(name: Any, text: str, incident_type: str | None) -> bool:
    n = normalize_text(name)
    t = normalize_text(text)
    if not n:
        return False
    if n in AGGREGATE_HINTS:
        return True
    if any(h in n for h in AGGREGATE_HINTS):
        return True
    # "Palisades and Eaton fires" or "Palisades, Eaton and Hurst fires".
    fire_names = extract_specific_fire_names(text)
    if incident_type == "wildfire" and len(fire_names) >= 2:
        return True
    if incident_type == "wildfire" and ("wildfires" in n or "fires" in n) and not extract_specific_fire_names(name):
        return True
    if incident_type == "civil protest" and ("protests" in n or "demonstrations" in n) and any(x in n for x in ["ice", "immigration", "los angeles", "la"]):
        return True
    return False


def is_direct_incident_evidence(ev: dict[str, Any], text: str, incident_type: str | None) -> bool:
    t = normalize_text(text)
    name = normalize_text(ev.get("event_name"))
    desc = normalize_text(ev.get("event_description"))
    reason = normalize_text(ev.get("reasoning"))

    if incident_type == "wildfire":
        if extract_specific_fire_names(text):
            return True
        return any(x in t for x in ["wildfire", "brush fire", "major blaze", "fire broke out", "fire started", "structures", "burned", "evacuation"])
    if incident_type == "urban fire":
        return "fire" in t and not any(x in name for x in ["hearing", "trial", "shelter", "intake"])
    if incident_type == "civil protest":
        return any(x in t for x in ["protest", "demonstration", "rally", "march", "walkout", "strike began", "clashed"])
    if incident_type == "terrorist incident":
        return any(x in t for x in ["terrorist", "terrorism", "attack", "bomb", "explosion"])
    return any(x in t for x in DIRECT_EVENT_TERMS)


def procedural_or_adjacent_reject(ev: dict[str, Any], text: str, incident_type: str | None) -> str | None:
    t = normalize_text(text)
    name_loc = normalize_text(f"{ev.get('event_name', '')} {ev.get('location', '')} {ev.get('location_name', '')}")

    if any(x in name_loc for x in PROCEDURAL_OR_ADJACENT_TERMS):
        # Keep actual direct fires/protests even if the article also mentions response.
        if not is_direct_incident_evidence(ev, text, incident_type):
            return "procedural_or_adjacent_not_direct_incident"

    if any(x in normalize_text(f"{ev.get('event_name', '')} {ev.get('event_description', '')}") for x in THREAT_OR_NON_OCCURRED_TERMS):
        if not any(x in t for x in ["strike began", "workers walked out", "protesters gathered", "demonstrators marched"]):
            return "threat_or_planned_event_not_actual_occurrence"

    # Weak/indirect URL alone is not an automatic reject if the source_event itself is direct.
    domain = domain_of(ev.get("url"))
    if domain in WEAK_INDIRECT_DOMAINS and not is_direct_incident_evidence(ev, text, incident_type):
        return "weak_or_indirect_source_without_direct_event_evidence"

    return None


def best_event_name(incident: dict[str, Any], ev: dict[str, Any], incident_type: str) -> str:
    name = ev.get("event_name")
    if isinstance(name, str) and name.strip() and name.strip().upper() != "N/A":
        return name.strip()
    names = incident.get("incident_names")
    if isinstance(names, list):
        for x in names:
            if isinstance(x, str) and x.strip() and x.strip().upper() != "N/A":
                return x.strip()
    loc = ev.get("location") or ev.get("location_name") or incident.get("representative_location_name") or "Los Angeles"
    return f"{incident_type} | {loc}"


def build_candidate_from_source_event(incident_id: str, incident: dict[str, Any], idx: int, ev: dict[str, Any], labels: list[str], start_window: date, end_window: date) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Return (candidate_id, low_candidate, aggregate_candidate, reject)."""
    cid = f"{incident_id}__event_{idx}"
    text = source_event_text(incident, ev)
    incident_type = map_incident_type(ev.get("event_type"), text, labels)
    start_dt, start_precision = parse_to_pacific_naive(ev.get("start_time_str") or ev.get("start_time") or ev.get("parsed_start_date") or incident.get("canonical_start_datetime"))
    end_dt, end_precision = parse_to_pacific_naive(ev.get("end_time_str") or ev.get("end_time"))
    article_dt, _ = parse_to_pacific_naive(ev.get("article_date"))

    base = {
        "candidate_id": cid,
        "source_incident_id": incident_id,
        "source_event_index": idx,
        "source_event_url": ev.get("url"),
        "article_date_raw": ev.get("article_date"),
        "article_datetime_pacific": article_dt,
        "source_file": ev.get("source_file"),
        "article_index": ev.get("article_index"),
        "event_index": ev.get("event_index", idx),
        "raw_event_type": ev.get("event_type"),
        "raw_event_name": ev.get("event_name"),
        "raw_location": ev.get("location") or ev.get("location_name"),
        "raw_start_time": ev.get("start_time") or ev.get("start_time_str"),
        "raw_end_time": ev.get("end_time") or ev.get("end_time_str"),
        "event_description": ev.get("event_description"),
        "reasoning": ev.get("reasoning"),
        "relationships": ev.get("relationships") or [],
        "representative_event_type": incident.get("representative_event_type"),
        "representative_location_name": incident.get("representative_location_name"),
        "centroid_lat": incident.get("centroid_lat"),
        "centroid_lon": incident.get("centroid_lon"),
    }

    if not incident_type:
        return cid, None, None, {**base, "reject_reason": "not_target_incident_type"}
    if not in_window(start_dt, end_dt, start_window, end_window):
        return cid, None, None, {**base, "reject_reason": "outside_target_date_window", "start_datetime_pacific": start_dt, "end_datetime_pacific": end_dt}
    if not has_la_evidence(incident, ev):
        return cid, None, None, {**base, "reject_reason": "not_in_or_near_los_angeles"}

    final_name = best_event_name(incident, ev, incident_type)
    location = ev.get("location") or ev.get("location_name") or incident.get("representative_location_name")

    if is_aggregate_name(final_name, text, incident_type):
        return cid, None, {
            **base,
            "aggregate_candidate_id": cid,
            "aggregate_name": final_name,
            "incident_type": incident_type,
            "location": location,
            "start_datetime_pacific": start_dt,
            "end_datetime_pacific": end_dt,
            "start_time_precision": start_precision,
            "end_time_precision": end_precision,
            "specific_fire_names_mentioned": sorted(extract_specific_fire_names(text)),
            "urls_used": [ev.get("url")] if ev.get("url") else [],
        }, None

    reject_reason = procedural_or_adjacent_reject(ev, text, incident_type)
    if reject_reason:
        return cid, None, None, {**base, "reject_reason": reject_reason, "incident_type": incident_type, "final_name": final_name, "start_datetime_pacific": start_dt}

    if not is_direct_incident_evidence(ev, text, incident_type):
        return cid, None, None, {**base, "reject_reason": "no_direct_incident_evidence", "incident_type": incident_type, "final_name": final_name}

    low = {
        **base,
        "low_candidate_id": cid,
        "final_name": final_name,
        "incident_type": incident_type,
        "location": location,
        "start_datetime_pacific": start_dt,
        "end_datetime_pacific": end_dt,
        "start_time_precision": start_precision,
        "end_time_precision": end_precision,
        "time_notes": ev.get("reasoning"),
        "location_notes": f"Source event location: {location}",
        "urls_used": [ev.get("url")] if ev.get("url") else [],
        "source_event_refs": [
            {
                "source_incident_id": incident_id,
                "source_event_index": idx,
                "url": ev.get("url"),
                "article_date_raw": ev.get("article_date"),
                "article_datetime_pacific": article_dt,
            }
        ],
        "specific_fire_names": sorted(extract_specific_fire_names(text)),
        "evidence_text_compact": " ".join(str(x) for x in [final_name, location, ev.get("event_description"), ev.get("reasoning")] if x)[:1500],
    }
    return cid, low, None, None


class UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> list[list[str]]:
        out: dict[str, list[str]] = collections.defaultdict(list)
        for x in self.parent:
            out[self.find(x)].append(x)
        return [sorted(v) for v in out.values()]


def same_start_day(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ad = date_part(a.get("start_datetime_pacific"))
    bd = date_part(b.get("start_datetime_pacific"))
    if ad is None or bd is None:
        return True
    return ad == bd


def location_tokens(location: Any) -> set[str]:
    text = normalize_key(location)
    tokens = {t for t in text.split() if len(t) >= 4}
    generic = {"southern", "county", "area", "region", "downtown"}
    return tokens - generic


def compatible_locations(a: dict[str, Any], b: dict[str, Any]) -> bool:
    al, bl = location_tokens(a.get("location")), location_tokens(b.get("location"))
    if not al or not bl:
        return True
    # If one says "Pacific Palisades" and another says "Los Angeles", allow when names match.
    generic_la = {"los", "angeles", "southern"}
    if al <= generic_la or bl <= generic_la:
        return True
    overlap = al & bl
    return bool(overlap) or normalize_key(a.get("location")) in normalize_key(b.get("location")) or normalize_key(b.get("location")) in normalize_key(a.get("location"))


def merge_allowed(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a.get("incident_type") != b.get("incident_type"):
        return False

    af, bf = set(a.get("specific_fire_names") or []), set(b.get("specific_fire_names") or [])
    if af and bf and af.isdisjoint(bf):
        return False

    # If both have named fires, same named fire is enough even if one location is broad.
    if af and bf and not af.isdisjoint(bf):
        return True

    if not same_start_day(a, b):
        return False
    if not compatible_locations(a, b):
        return False

    an, bn = normalize_key(a.get("final_name")), normalize_key(b.get("final_name"))
    if an and bn and (an == bn or an in bn or bn in an):
        return True

    # For protests, avoid merging different locations unless the name/location/day really match.
    if a.get("incident_type") == "civil protest":
        return False

    return False


def merge_key_candidates(c: dict[str, Any]) -> list[tuple[str, str]]:
    keys = []
    typ = c.get("incident_type") or ""
    day = str(c.get("start_datetime_pacific") or "")[:10]
    fires = c.get("specific_fire_names") or []
    for f in fires:
        keys.append(("fire", f))
    if typ and day:
        name = normalize_key(c.get("final_name"))
        loc = " ".join(sorted(location_tokens(c.get("location"))))
        if name and loc:
            keys.append(("exactish", f"{typ}|{day}|{name}|{loc}"))
    return keys


def dedupe(values: Iterable[Any]) -> list[Any]:
    out, seen = [], set()
    for v in values:
        if v in (None, "", [], {}):
            continue
        key = json.dumps(v, sort_keys=True, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def most_common(values: Iterable[Any]) -> Any:
    vals = [v for v in values if v not in (None, "", [], {})]
    if not vals:
        return None
    return collections.Counter(str(v) for v in vals).most_common(1)[0][0]


def earliest(values: Iterable[Any]) -> str | None:
    pairs = []
    for v in values:
        d = parse_dt_sortable(v)
        if d:
            pairs.append((d, str(v)))
    return min(pairs, key=lambda x: x[0])[1] if pairs else None


def latest(values: Iterable[Any]) -> str | None:
    pairs = []
    for v in values:
        d = parse_dt_sortable(v)
        if d:
            pairs.append((d, str(v)))
    return max(pairs, key=lambda x: x[0])[1] if pairs else None


def parse_dt_sortable(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        d = date_part(str(value))
        return datetime(d.year, d.month, d.day) if d else None


def merge_group_to_low(clean_id: str, ids: list[str], cands: dict[str, dict[str, Any]]) -> dict[str, Any]:
    group = [cands[i] for i in ids]
    # Prefer named fires/protests over generic type | location names.
    names = [g.get("final_name") for g in group]
    fire_names = sorted({f for g in group for f in (g.get("specific_fire_names") or [])})
    if fire_names:
        final_name = fire_names[0].title().replace(" La ", " LA ")
    else:
        final_name = most_common(names) or clean_id

    refs = []
    urls = []
    source_incident_ids = []
    raw_names = []
    for g in group:
        refs.extend(g.get("source_event_refs") or [])
        urls.extend(g.get("urls_used") or [])
        source_incident_ids.append(g.get("source_incident_id"))
        raw_names.append(g.get("final_name"))

    return {
        "low_level_incident_id": clean_id,
        "final_name": final_name,
        "incident_type": most_common(g.get("incident_type") for g in group),
        "location": most_common(g.get("location") for g in group),
        "start_datetime_pacific": earliest(g.get("start_datetime_pacific") for g in group),
        "end_datetime_pacific": latest(g.get("end_datetime_pacific") for g in group),
        "start_time_precision": most_common(g.get("start_time_precision") for g in group),
        "end_time_precision": most_common(g.get("end_time_precision") for g in group),
        "source_candidate_ids": ids,
        "source_incident_ids": dedupe(source_incident_ids),
        "original_incident_names": dedupe(raw_names),
        "urls_used": dedupe(urls),
        "source_event_refs": dedupe(refs),
        "specific_fire_names": fire_names,
        "evidence_source_event_count": len(refs),
    }


def merge_low_candidates(cands: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, str], list[list[str]]]:
    uf = UnionFind(cands.keys())
    buckets: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for cid, c in cands.items():
        for key in merge_key_candidates(c):
            buckets[key].append(cid)

    for ids in buckets.values():
        if len(ids) < 2:
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if merge_allowed(cands[ids[i]], cands[ids[j]]):
                    uf.union(ids[i], ids[j])

    groups = uf.groups()
    groups.sort(key=lambda ids: (str(cands[ids[0]].get("start_datetime_pacific") or ""), normalize_key(cands[ids[0]].get("final_name")), ids[0]))
    low: dict[str, Any] = {}
    merge_map: dict[str, str] = {}
    for idx, ids in enumerate(groups):
        clean_id = f"low_{idx:06d}"
        low[clean_id] = merge_group_to_low(clean_id, ids, cands)
        for old in ids:
            merge_map[old] = clean_id
    return low, merge_map, groups


class LLMClient:
    def __init__(self, api_key: str | None, model: str) -> None:
        self.model = model
        self.client = None
        if api_key:
            if OpenAI is None:
                raise RuntimeError("The openai package is not installed. Run: pip install openai")
            self.client = OpenAI(api_key=api_key)

    def enabled(self) -> bool:
        return self.client is not None

    def json_call(self, system: str, payload: dict[str, Any], retries: int = 3) -> dict[str, Any] | None:
        if not self.client:
            return None
        prompt = "Return only valid JSON. Payload:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
        last = None
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                )
                text = (resp.choices[0].message.content or "").strip()
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
                return json.loads(text)
            except Exception as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        print(f"WARNING: LLM call failed: {last}", file=sys.stderr)
        return None


TOP_LEVEL_SYSTEM = """
You are building a two-level incident hierarchy for Los Angeles incident records.
Given one aggregate candidate and a list of concrete low-level incidents, identify which
low-level incidents are part of the aggregate. Return only JSON:
{
  "top_level_name": "string",
  "child_low_level_ids": ["low_..."],
  "reason": "brief"
}
Rules:
- Top-level incidents must have at least two children.
- Los Angeles wildfires can include distinct wildfire incidents such as Palisades Fire,
  Eaton Fire, Hurst Fire, Hughes Fire, etc.
- ICE protests can include distinct protest incidents at different locations if they are
  all part of the same broader anti-ICE/immigration protest wave.
- Do not include unrelated incident types.
- Do not include a low-level incident merely because it is nearby in time; it must be
  substantively part of the broader incident family.
"""


def low_text(low: dict[str, Any]) -> str:
    return normalize_text(" ".join(str(x) for x in [low.get("final_name"), low.get("incident_type"), low.get("location"), low.get("original_incident_names"), low.get("urls_used")] if x))


def build_top_level_candidates(aggregates: dict[str, dict[str, Any]], low: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    # Seed known aggregates even if they were not explicitly extracted.
    seeds = [
        {"aggregate_name": "Los Angeles wildfires", "incident_type": "wildfire", "seed": True},
        {"aggregate_name": "ICE protests", "incident_type": "civil protest", "seed": True},
    ]
    candidates.extend(seeds)
    for agg in aggregates.values():
        candidates.append(agg)
    # Dedupe by normalized name/type.
    out, seen = [], set()
    for c in candidates:
        fp = (normalize_key(c.get("aggregate_name")), c.get("incident_type"))
        if fp not in seen:
            seen.add(fp)
            out.append(c)
    return out


def heuristic_children_for_top(agg: dict[str, Any], low: dict[str, dict[str, Any]]) -> list[str]:
    name = normalize_text(agg.get("aggregate_name"))
    typ = agg.get("incident_type")
    children = []
    mentioned_fires = set(agg.get("specific_fire_names_mentioned") or [])

    for low_id, rec in low.items():
        if typ and rec.get("incident_type") != typ:
            continue
        text = low_text(rec)
        fires = set(rec.get("specific_fire_names") or [])

        if typ == "wildfire":
            if "los angeles" in name or "la fires" in name or "wildfires" in name:
                # Broad Jan 2025 LA wildfire complex. Keep distinct named LA wildfires.
                if fires or any(x in text for x in ["palisades", "eaton", "hurst", "hughes", "kenneth", "sunset"]):
                    children.append(low_id)
                    continue
            if mentioned_fires and fires and not mentioned_fires.isdisjoint(fires):
                children.append(low_id)
                continue
        elif typ == "civil protest":
            if any(x in name for x in ["ice", "immigration"]):
                if any(x in text for x in ["ice", "immigration", "federal", "la protest", "anti ice", "anti-ice"]):
                    children.append(low_id)
                    continue
            if "protest" in name or "demonstration" in name:
                if "protest" in text or "demonstration" in text or "rally" in text or "march" in text:
                    children.append(low_id)
                    continue
    return sorted(set(children))


def build_top_level(aggregates: dict[str, dict[str, Any]], low: dict[str, dict[str, Any]], llm: LLMClient, use_llm: bool, output_path: Path, decisions_path: Path, no_progress: bool) -> dict[str, Any]:
    top: dict[str, Any] = {}
    decisions = load_json(decisions_path, default={})
    candidates = build_top_level_candidates(aggregates, low)
    iterator = tqdm(candidates, desc="Building top-level incidents", unit="aggregate") if tqdm and not no_progress else candidates

    idx = 0
    used_child_sets = set()
    for agg in iterator:
        agg_name = str(agg.get("aggregate_name") or "").strip() or "Aggregate incident"
        children = heuristic_children_for_top(agg, low)

        # Optional LLM review when heuristic has enough possible children or aggregate is nonstandard.
        decision_key = normalize_key(agg_name) + "|" + str(agg.get("incident_type"))
        if use_llm and llm.enabled() and decision_key not in decisions and len(children) >= 2:
            payload = {
                "aggregate_candidate": agg,
                "heuristic_child_low_level_ids": children,
                "candidate_low_level_incidents": [
                    {
                        "low_level_incident_id": cid,
                        "final_name": low[cid].get("final_name"),
                        "incident_type": low[cid].get("incident_type"),
                        "location": low[cid].get("location"),
                        "start_datetime_pacific": low[cid].get("start_datetime_pacific"),
                        "original_incident_names": low[cid].get("original_incident_names"),
                        "urls_used": low[cid].get("urls_used"),
                    }
                    for cid in children[:80]
                ],
            }
            decision = llm.json_call(TOP_LEVEL_SYSTEM, payload)
            if isinstance(decision, dict):
                decisions[decision_key] = decision
                atomic_save_json(decisions_path, decisions)
        elif decision_key in decisions:
            decision = decisions[decision_key]
        else:
            decision = None

        if isinstance(decision, dict) and isinstance(decision.get("child_low_level_ids"), list):
            llm_children = [c for c in decision["child_low_level_ids"] if c in low]
            if len(llm_children) >= 2:
                children = sorted(set(llm_children))
                agg_name = str(decision.get("top_level_name") or agg_name)

        if len(children) < 2:
            continue
        fp = tuple(sorted(children))
        if fp in used_child_sets:
            continue
        used_child_sets.add(fp)

        starts = [low[c].get("start_datetime_pacific") for c in children]
        ends = [low[c].get("end_datetime_pacific") for c in children]
        top_id = f"top_{idx:06d}"
        idx += 1
        top[top_id] = {
            "top_level_incident_id": top_id,
            "top_level_incident_name": agg_name,
            "low_level_incident_ids": children,
            "start_datetime_pacific": earliest(starts),
            "end_datetime_pacific": latest(ends),
            "incident_types": dedupe(low[c].get("incident_type") for c in children),
            "source_aggregate_candidate_ids": dedupe([agg.get("aggregate_candidate_id")]),
        }
        atomic_save_json(output_path, top)

    atomic_save_json(output_path, top)
    return top


def process_all_merged(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    labels = load_labels(args.filtered_incidents)
    start_window, end_window = date.fromisoformat(args.start_date), date.fromisoformat(args.end_date)
    all_merged = load_json(args.all_merged)
    if not isinstance(all_merged, dict):
        raise ValueError(f"Expected JSON object/dict in {args.all_merged}")

    candidates = {} if args.force else load_json(args.candidates_output, default={})
    aggregates = {} if args.force else load_json(args.aggregates_output, default={})
    rejects = {} if args.force else load_json(args.rejects_output, default={})
    progress = {} if args.force else load_json(args.progress_output, default={})
    processed = set(progress.get("processed_incident_ids", [])) if isinstance(progress, dict) else set()

    items = list(all_merged.items())
    iterator = tqdm(items, desc="Extracting source-event candidates", unit="incident") if tqdm and not args.no_progress else items

    for incident_id, incident in iterator:
        if not args.force and incident_id in processed:
            continue
        if not isinstance(incident, dict):
            rejects[incident_id] = {"reject_reason": "incident_record_not_object"}
            processed.add(incident_id)
            continue
        events = incident.get("source_events") or []
        if not isinstance(events, list):
            events = []
        for idx, ev in enumerate(events):
            if not isinstance(ev, dict):
                continue
            cid, low, agg, reject = build_candidate_from_source_event(incident_id, incident, idx, ev, labels, start_window, end_window)
            if low:
                candidates[cid] = low
            elif agg:
                aggregates[cid] = agg
            elif reject:
                rejects[cid] = reject

        processed.add(incident_id)
        # Frequent checkpoints so a crash can resume after the last completed incident.
        atomic_save_json(args.candidates_output, candidates)
        atomic_save_json(args.aggregates_output, aggregates)
        atomic_save_json(args.rejects_output, rejects)
        atomic_save_json(args.progress_output, {
            "stage": "extracting_source_event_candidates",
            "processed_incident_ids": sorted(processed),
            "processed_count": len(processed),
            "total_incidents": len(all_merged),
            "candidate_count": len(candidates),
            "aggregate_candidate_count": len(aggregates),
            "reject_count": len(rejects),
        })

    return candidates, aggregates, rejects


def print_low_level_summary(low: dict[str, Any], aggregates: dict[str, Any], rejects: dict[str, Any]) -> None:
    by_type = collections.Counter(rec.get("incident_type") for rec in low.values())
    print("\nLow-level incident build complete.")
    print(f"  low_level count: {len(low)}")
    print(f"  aggregate candidates: {len(aggregates)}")
    print(f"  rejected source events: {len(rejects)}")
    print("  by type:")
    for typ, count in sorted(by_type.items(), key=lambda x: str(x[0])):
        print(f"    {typ}: {count}")
    print("\nSample low-level incidents:")
    for low_id, rec in list(low.items())[:20]:
        print(f"  {low_id}: {rec.get('final_name')} | {rec.get('incident_type')} | {rec.get('location')} | {rec.get('start_datetime_pacific')}")


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(args.filtered_incidents)

    llm = LLMClient(args.openai_api_key or os.environ.get("OPENAI_API_KEY"), args.model)
    use_llm = (not args.no_llm) and llm.enabled()

    candidates, aggregates, rejects = process_all_merged(args)

    low, merge_map, merge_groups = merge_low_candidates(candidates)
    atomic_save_json(args.low_output, low)
    atomic_save_json(args.merge_map_output, merge_map)
    atomic_save_json(args.merge_groups_output, merge_groups)

    summary = {
        "stage": "low_level_complete",
        "all_merged_input": str(args.all_merged),
        "source_event_candidate_count": len(candidates),
        "aggregate_candidate_count": len(aggregates),
        "rejected_source_event_count": len(rejects),
        "low_level_count": len(low),
        "low_level_by_type": dict(collections.Counter(rec.get("incident_type") for rec in low.values())),
        "low_output": str(args.low_output),
        "top_output": str(args.top_output),
        "notes": "Top-level output is only written after pause/continue unless --no-pause is used.",
    }
    atomic_save_json(args.summary_output, summary)
    print_low_level_summary(low, aggregates, rejects)

    should_continue = args.no_pause
    if not args.no_pause:
        reply = input("\nType 'continue' to build top_level.json, or anything else to stop here: ").strip().lower()
        should_continue = reply == "continue"
    if not should_continue:
        print(f"Stopped after writing low-level output: {args.low_output}")
        return

    top = build_top_level(
        aggregates=aggregates,
        low=low,
        llm=llm,
        use_llm=use_llm,
        output_path=args.top_output,
        decisions_path=args.top_decisions_output,
        no_progress=args.no_progress,
    )
    summary.update({
        "stage": "complete",
        "top_level_count": len(top),
        "top_output": str(args.top_output),
    })
    atomic_save_json(args.summary_output, summary)
    print(f"Wrote {len(top)} top-level incidents to {args.top_output}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Refine news incidents directly from all_merged_by_id_relevant.json")
    p.add_argument("--all-merged", type=Path, default=DEFAULT_ALL_MERGED, help=f"Input all_merged_by_id_relevant.json. Default: {DEFAULT_ALL_MERGED}")
    p.add_argument("--filtered-incidents", type=Path, default=DEFAULT_FILTERED_INCIDENTS)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--start-date", default=DEFAULT_START_DATE)
    p.add_argument("--end-date", default=DEFAULT_END_DATE)

    p.add_argument("--low-output", type=Path, default=DEFAULT_OUT_DIR / "low_level.json")
    p.add_argument("--top-output", type=Path, default=DEFAULT_OUT_DIR / "top_level.json")
    p.add_argument("--candidates-output", type=Path, default=DEFAULT_OUT_DIR / "source_event_candidates.json")
    p.add_argument("--aggregates-output", type=Path, default=DEFAULT_OUT_DIR / "aggregate_candidates.json")
    p.add_argument("--rejects-output", type=Path, default=DEFAULT_OUT_DIR / "source_event_rejects.json")
    p.add_argument("--progress-output", type=Path, default=DEFAULT_OUT_DIR / "refine_from_all_progress.json")
    p.add_argument("--merge-map-output", type=Path, default=DEFAULT_OUT_DIR / "low_level_merge_map.json")
    p.add_argument("--merge-groups-output", type=Path, default=DEFAULT_OUT_DIR / "low_level_merge_groups.json")
    p.add_argument("--top-decisions-output", type=Path, default=DEFAULT_OUT_DIR / "top_level_decisions.json")
    p.add_argument("--summary-output", type=Path, default=DEFAULT_OUT_DIR / "refine_from_all_summary.json")

    p.add_argument("--openai-api-key", default=None)
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--no-llm", action="store_true", help="Disable LLM review for top-level hierarchy construction.")
    p.add_argument("--no-pause", action="store_true", help="Do not pause after writing low_level.json; build top_level.json immediately.")
    p.add_argument("--force", action="store_true", help="Ignore extraction checkpoints and rebuild candidates/rejects from scratch.")
    p.add_argument("--no-progress", action="store_true")

    # Compatibility no-ops from older versions, so old commands do not immediately fail.
    p.add_argument("--low-level", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--top-level", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--no-fetch-web", action="store_true", help=argparse.SUPPRESS)
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
