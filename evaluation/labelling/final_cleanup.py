#!/usr/bin/env python3
"""Low-level-only final cleanup for incident labels.

This script rewrites the final cleanup stage to focus ONLY on creating
`low_level_final.json`. It intentionally ignores top-level construction.

It starts from `low_level_cleaned.json`, optionally uses one or more reference
low-level files such as the original `low_level.json`, and looks back into
`all_merged_by_id_relevant.json` to recover source-event/article metadata.

Key behavior:
  * Merges duplicate low-level incidents.
  * Uses parent/original names only as hints, not as identity. For example, an
    Eaton Fire record with parent/original name "Palisades and Eaton fires" is
    still only Eaton Fire if its final_name is Eaton Fire.
  * Searches article evidence by exact source refs/incident IDs/URLs AND by
    similar concrete incident name, especially named fires.
  * Adds earliest article publication metadata to each final low-level record.
  * Uses LLMs only for ambiguous merge blocks, after cheap candidate generation.

Typical run:
  python final_cleanup.py \
    --low-input evaluation/out/low_level_cleaned.json \
    --reference-low evaluation/out/low_level.json \
    --all-merged evaluation/merged_incidents/all_merged_by_id_relevant.json \
    --openai-api-key "$OPENAI_API_KEY" \
    --force

Outputs:
  evaluation/out/low_level_final.json
  evaluation/out/final_low_merge_map.json
  evaluation/out/final_low_article_audit.json
  evaluation/out/final_low_duplicate_report.json
  evaluation/out/final_low_summary.json
"""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import html
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


DEFAULT_LOW_INPUT = Path("evaluation/out/low_level_cleaned.json")
DEFAULT_REFERENCE_LOW = Path("evaluation/out/low_level.json")
DEFAULT_ALL_MERGED = Path("evaluation/merged_incidents/all_merged_by_id_relevant.json")
DEFAULT_LOW_OUTPUT = Path("evaluation/out/low_level_final.json")
DEFAULT_TOP_OUTPUT = Path("evaluation/out/top_level.json")
DEFAULT_MERGE_CACHE = Path("evaluation/out/final_low_merge_decisions.json")
DEFAULT_MERGE_MAP = Path("evaluation/out/final_low_merge_map.json")
DEFAULT_ARTICLE_AUDIT = Path("evaluation/out/final_low_article_audit.json")
DEFAULT_DUPLICATE_REPORT = Path("evaluation/out/final_low_duplicate_report.json")
DEFAULT_SUMMARY = Path("evaluation/out/final_low_summary.json")
DEFAULT_TOP_AUDIT = Path("evaluation/out/final_top_audit.json")
DEFAULT_TOP_CACHE = Path("evaluation/out/final_top_decisions.json")

PACIFIC = ZoneInfo("America/Los_Angeles")
TARGET_TYPES = {"civil protest", "urban fire", "wildfire", "terrorist incident"}

# Includes typo normalization: Lidia -> Lydia.
FIRE_NAMES = {
    "palisades": "Palisades Fire",
    "eaton": "Eaton Fire",
    "hurst": "Hurst Fire",
    "kenneth": "Kenneth Fire",
    "sunset": "Sunset Fire",
    "lydia": "Lydia Fire",
    "lidia": "Lydia Fire",
    "hughes": "Hughes Fire",
    "franklin": "Franklin Fire",
    "bridge": "Bridge Fire",
    "airport": "Airport Fire",
    "mountain": "Mountain Fire",
    "post": "Post Fire",
    "lake": "Lake Fire",
    "borel": "Borel Fire",
}

AGGREGATE_TERMS = {
    "wildfires", "los angeles wildfires", "los angeles fires", "la fires",
    "palisades and eaton", "eaton and palisades", "palisades fire and eaton fire",
    "ice protests", "anti ice protests", "anti-ice protests", "immigration protests",
    "protests", "demonstrations", "rallies", "marches",
}

LOW_QUALITY_CONTEXT_TERMS = {
    "school closing", "school reopening", "trial", "hearing", "lawsuit", "court",
    "celebrity", "meghan", "prince harry", "paris hilton", "grammys", "oscars",
    "animal rescue", "horse intake", "shelter", "relief center", "fundraiser",
    "health hazards", "cleanup", "reopening", "damaged schools",
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
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value)).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"https?://\S+", " ", text)
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_name(value: Any) -> str:
    text = normalize_text(value)
    stop = {
        "the", "a", "an", "incident", "event", "reported", "report", "usa",
        "united", "states", "america", "california", "ca", "county", "city",
    }
    return " ".join(t for t in text.split() if t not in stop)


def normalize_location(value: Any) -> str:
    text = normalize_text(value)
    drop = {"usa", "united", "states", "america", "california", "ca", "county", "city", "of", "the"}
    return " ".join(t for t in text.split() if t not in drop)


def dedupe_list(values: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for v in values:
        if v in (None, "", []):
            continue
        key = json.dumps(v, sort_keys=True, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).replace(tzinfo=None)
    except Exception:
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
        if m:
            try:
                return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except Exception:
                return None
    return None


def earliest_dt(values: Iterable[Any]) -> str | None:
    pairs = []
    for v in values:
        dt = parse_dt(v)
        if dt:
            pairs.append((dt, str(v)))
    return min(pairs, key=lambda x: x[0])[1] if pairs else None


def latest_dt(values: Iterable[Any]) -> str | None:
    pairs = []
    for v in values:
        dt = parse_dt(v)
        if dt:
            pairs.append((dt, str(v)))
    return max(pairs, key=lambda x: x[0])[1] if pairs else None


def parse_article_date_raw(raw: Any) -> tuple[datetime | None, str | None]:
    """Parse upstream YYYYMMDDHHMMSS article_date as UTC and convert to Pacific."""
    if raw is None:
        return None, None
    s = str(raw).strip()
    if not re.match(r"^\d{14}$", s):
        return None, None
    try:
        dt_utc = datetime.strptime(s, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        dt_pacific = dt_utc.astimezone(PACIFIC).replace(tzinfo=None)
        return dt_pacific, dt_pacific.isoformat(timespec="seconds")
    except Exception:
        return None, None


def domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def extract_urls(record: dict[str, Any]) -> list[str]:
    urls = []
    for key in ("urls_used", "all_candidate_urls", "source_urls"):
        val = record.get(key)
        if isinstance(val, list):
            urls.extend(str(u) for u in val if isinstance(u, str) and u.startswith(("http://", "https://")))
    for ref in record.get("source_event_refs") or []:
        if isinstance(ref, dict):
            url = ref.get("url")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                urls.append(url)
    return dedupe_list(urls)


def fire_names_in_text(text: Any) -> set[str]:
    norm = normalize_text(text)
    found = set()
    for token, canonical in FIRE_NAMES.items():
        if token in {"lake", "post", "bridge", "airport", "mountain"}:
            if re.search(rf"\b{token}\s+fire\b", norm):
                found.add(canonical)
        elif re.search(rf"\b{re.escape(token)}\b", norm):
            found.add(canonical)
    return found


def aggregate_like_text(text: Any) -> bool:
    n = normalize_text(text)
    return any(t in n for t in AGGREGATE_TERMS)


def record_blob(record: dict[str, Any], include_parent_hints: bool = True) -> str:
    fields = [
        record.get("final_name"),
        record.get("incident_type"),
        record.get("location"),
        record.get("start_datetime_pacific"),
        record.get("end_datetime_pacific"),
        record.get("time_notes"),
        record.get("location_notes"),
        record.get("cleanup_notes"),
        record.get("iter_cleanup_notes"),
        record.get("refine_notes"),
        record.get("final_cleanup_notes"),
    ]
    keys = ["original_incident_names", "urls_used", "all_candidate_urls", "specific_fire_names"]
    if include_parent_hints:
        keys += ["source_parent_names", "suggested_parent_names"]
    for key in keys:
        val = record.get(key)
        if isinstance(val, list):
            fields.extend(str(x) for x in val[:60])
    return " ".join(str(x) for x in fields if x)


def primary_fire_names(record: dict[str, Any]) -> set[str]:
    """Concrete fire identity for one low-level record.

    Parent fields are intentionally ignored unless needed later for evidence search.
    This prevents an Eaton Fire record from inheriting Palisades Fire from a parent
    phrase like "Palisades and Eaton fires".
    """
    final_name = str(record.get("final_name") or "")
    final_fires = fire_names_in_text(final_name)

    if len(final_fires) == 1 and not aggregate_like_text(final_name):
        return final_fires
    if aggregate_like_text(final_name) and len(final_fires) >= 2:
        return set()

    focused = " ".join([
        str(record.get("final_name") or ""),
        str(record.get("location") or ""),
        " ".join(str(x) for x in record.get("specific_fire_names") or []),
    ])
    focused_fires = fire_names_in_text(focused)
    if len(focused_fires) == 1:
        return focused_fires
    if len(focused_fires) > 1:
        return final_fires if len(final_fires) == 1 else set()
    return set()


def all_fire_names_anywhere(record: dict[str, Any]) -> set[str]:
    return fire_names_in_text(record_blob(record, include_parent_hints=True))


def canonical_type(record: dict[str, Any] | str | None) -> str:
    if isinstance(record, dict):
        raw = normalize_text(record.get("incident_type"))
        blob = normalize_text(record_blob(record, include_parent_hints=False))
        fires = primary_fire_names(record)
    else:
        raw = normalize_text(record)
        blob = raw
        fires = fire_names_in_text(raw)
    if fires or "wildfire" in blob or "brush fire" in blob:
        return "wildfire"
    if raw in TARGET_TYPES:
        return raw
    if "urban fire" in blob or "structure fire" in blob or "building fire" in blob:
        return "urban fire"
    if any(t in blob for t in ["civil protest", "protest", "demonstration", "rally", "march", "walkout", "strike"]):
        return "civil protest"
    if any(t in blob for t in ["terrorist", "terror", "bomb", "explosion", "attack"]):
        return "terrorist incident"
    return raw


def record_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    typ = canonical_type(record)
    fires = sorted(primary_fire_names(record))
    name_key = "|".join(fires) if fires else normalize_name(record.get("final_name"))
    loc_key = normalize_location(record.get("location"))
    start_key = str(record.get("start_datetime_pacific") or "")[:10]
    return typ, name_key, loc_key, start_key


def record_text_for_embedding(record: dict[str, Any]) -> str:
    bits = [
        str(record.get("final_name") or ""),
        canonical_type(record),
        str(record.get("location") or ""),
        str(record.get("start_datetime_pacific") or "")[:10],
        " ".join(str(x) for x in record.get("original_incident_names") or []),
        " ".join(str(x) for x in record.get("source_parent_names") or []),
        " ".join(str(x) for x in record.get("suggested_parent_names") or []),
    ]
    return " | ".join(x for x in bits if x)


# ---------------------------------------------------------------------------
# Article evidence
# ---------------------------------------------------------------------------

def build_article_index(all_merged: dict[str, Any]) -> dict[str, Any]:
    by_incident: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    by_url: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    by_ref: dict[str, dict[str, Any]] = {}
    by_fire: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    by_name_key: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)

    for incident_id, rec in all_merged.items():
        if not isinstance(rec, dict):
            continue
        for idx, ev in enumerate(rec.get("source_events") or []):
            if not isinstance(ev, dict):
                continue
            dt, dt_pacific = parse_article_date_raw(ev.get("article_date"))
            item = {
                "source_incident_id": incident_id,
                "source_event_index": idx,
                "url": ev.get("url"),
                "article_date_raw": ev.get("article_date"),
                "article_datetime_pacific": dt_pacific,
                "_article_dt_sort": dt.timestamp() if dt else None,
                "event_name": ev.get("event_name"),
                "event_type": ev.get("event_type"),
                "event_description": ev.get("event_description"),
                "location": ev.get("location") or ev.get("location_name"),
                "start_time": ev.get("start_time") or ev.get("start_time_str"),
                "end_time": ev.get("end_time") or ev.get("end_time_str"),
                "reasoning": ev.get("reasoning"),
                "source_file": ev.get("source_file"),
                "article_index": ev.get("article_index"),
            }
            by_incident[incident_id].append(item)
            if isinstance(item["url"], str):
                by_url[item["url"]].append(item)
            by_ref[f"{incident_id}__event_{idx}"] = item

            text = " ".join(str(x) for x in [
                ev.get("event_name"), ev.get("event_type"), ev.get("event_description"),
                ev.get("location"), ev.get("location_name"), ev.get("reasoning"), ev.get("url"),
            ] if x)
            for fire in fire_names_in_text(text):
                by_fire[fire].append(item)
            name = ev.get("event_name")
            if isinstance(name, str) and normalize_text(name) not in {"", "n a", "na", "none", "unknown"}:
                by_name_key[normalize_name(name)].append(item)

    return {"by_incident": dict(by_incident), "by_url": dict(by_url), "by_ref": by_ref, "by_fire": dict(by_fire), "by_name_key": dict(by_name_key)}


def event_public(item: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in item.items() if not k.startswith("_")}


def article_evidence_score(record: dict[str, Any], item: dict[str, Any], source: str) -> tuple[int, list[str]]:
    score = 0
    reasons = [source]
    text = normalize_text(" ".join(str(x) for x in [
        item.get("event_name"), item.get("event_type"), item.get("event_description"),
        item.get("location"), item.get("reasoning"), item.get("url"),
    ] if x))

    fires = primary_fire_names(record)
    if fires:
        for fire in fires:
            ft = normalize_text(fire)
            if ft in text or all(t in text for t in ft.split()):
                score += 80
                reasons.append(f"mentions_primary_fire:{fire}")
            else:
                score -= 20
    else:
        name_tokens = [t for t in normalize_name(record.get("final_name")).split() if len(t) >= 4]
        overlap = sum(1 for t in name_tokens if t in text)
        if overlap:
            score += min(overlap * 8, 40)
            reasons.append(f"name_token_overlap:{overlap}")

    typ = canonical_type(record)
    ev_type = normalize_text(item.get("event_type"))
    if typ == "wildfire":
        if any(t in text for t in ["wildfire", "brush fire", "fire", "blaze", "burn scar"]):
            score += 25
            reasons.append("wildfire_terms")
        if ev_type in {"wildfire", "fire"}:
            score += 25
            reasons.append("event_type_fire")
    elif typ == "civil protest" and any(t in text for t in ["protest", "demonstration", "rally", "march", "strike", "walkout"]):
        score += 25
        reasons.append("protest_terms")
    elif typ == "urban fire" and any(t in text for t in ["structure fire", "building fire", "apartment fire", "firefighters"]):
        score += 25
        reasons.append("urban_fire_terms")

    if source in {"source_event_ref", "source_incident_id", "url"}:
        score += 25
        reasons.append("exact_record_link")
    if any(t in text for t in LOW_QUALITY_CONTEXT_TERMS):
        score -= 20
        reasons.append("low_quality_context_terms")
    if ev_type and ev_type not in {"wildfire", "fire", "civil protest", "urban fire", "terrorist incident", "demonstration"} and typ in TARGET_TYPES:
        score -= 10
        reasons.append(f"non_target_event_type:{ev_type}")
    return score, reasons


def gather_article_evidence(record: dict[str, Any], article_index: dict[str, Any], reference_records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[tuple[dict[str, Any], str]] = []

    for ref in record.get("source_event_refs") or []:
        if not isinstance(ref, dict):
            continue
        sid = ref.get("source_incident_id")
        idx = ref.get("source_event_index")
        if sid is not None and idx is not None:
            item = article_index["by_ref"].get(f"{sid}__event_{idx}")
            if item:
                candidates.append((item, "source_event_ref"))

    for sid in record.get("source_incident_ids") or []:
        for item in article_index["by_incident"].get(str(sid), []):
            candidates.append((item, "source_incident_id"))

    for url in extract_urls(record):
        for item in article_index["by_url"].get(url, []):
            candidates.append((item, "url"))

    fires = primary_fire_names(record)
    name_key = normalize_name(record.get("final_name"))
    for ref_id, ref in reference_records.items():
        if fires:
            # Match reference records by concrete fire names anywhere, including their refs.
            if not (fires & primary_fire_names(ref) or fires & all_fire_names_anywhere(ref)):
                continue
        elif normalize_name(ref.get("final_name")) != name_key:
            continue

        for sref in ref.get("source_event_refs") or []:
            if not isinstance(sref, dict):
                continue
            sid = sref.get("source_incident_id")
            idx = sref.get("source_event_index")
            item = article_index["by_ref"].get(f"{sid}__event_{idx}")
            if item:
                candidates.append((item, "reference_source_event_ref"))
        for sid in ref.get("source_incident_ids") or []:
            for item in article_index["by_incident"].get(str(sid), []):
                candidates.append((item, "reference_source_incident_id"))
        for url in extract_urls(ref):
            for item in article_index["by_url"].get(url, []):
                candidates.append((item, "reference_url"))

    if fires:
        for fire in fires:
            for item in article_index["by_fire"].get(fire, []):
                candidates.append((item, "all_merged_fire_name"))
    elif name_key:
        for item in article_index["by_name_key"].get(name_key, []):
            candidates.append((item, "all_merged_event_name"))

    scored = []
    seen = set()
    for item, source in candidates:
        key = (item.get("source_incident_id"), item.get("source_event_index"), item.get("url"), item.get("article_date_raw"))
        if key in seen:
            continue
        seen.add(key)
        score, reasons = article_evidence_score(record, item, source)
        public = event_public(item)
        public["evidence_score"] = score
        public["evidence_reasons"] = reasons
        public["_article_dt_sort"] = item.get("_article_dt_sort")
        scored.append(public)

    scored.sort(key=lambda e: (-(e.get("evidence_score") or 0), e.get("_article_dt_sort") is None, e.get("_article_dt_sort") or math.inf))
    return scored


def choose_earliest_article(evidence: list[dict[str, Any]], min_score: int) -> dict[str, Any] | None:
    direct = [e for e in evidence if e.get("_article_dt_sort") is not None and (e.get("evidence_score") or 0) >= min_score]
    if not direct:
        dated = [e for e in evidence if e.get("_article_dt_sort") is not None]
        if not dated:
            return None
        best_score = max(e.get("evidence_score") or -999 for e in dated)
        direct = [e for e in dated if (e.get("evidence_score") or -999) == best_score]
    chosen = min(direct, key=lambda e: e["_article_dt_sort"])
    return {k: v for k, v in chosen.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, ids: Iterable[str]) -> None:
        self.parent = {x: x for x in ids}

    def find(self, x: str) -> str:
        p = self.parent.setdefault(x, x)
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = collections.defaultdict(list)
        for x in self.parent:
            out[self.find(x)].append(x)
        return dict(out)


def do_not_merge(a: dict[str, Any], b: dict[str, Any]) -> tuple[bool, str]:
    if canonical_type(a) != canonical_type(b):
        return True, "different_types"
    fa, fb = primary_fire_names(a), primary_fire_names(b)
    if fa and fb and fa.isdisjoint(fb):
        return True, f"different_primary_fires:{sorted(fa)} vs {sorted(fb)}"
    return False, ""


def should_merge_deterministic(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if do_not_merge(a, b)[0]:
        return False
    fa, fb = primary_fire_names(a), primary_fire_names(b)
    if fa and fa == fb:
        return True
    if record_key(a) == record_key(b) and all(record_key(a)):
        return True
    if canonical_type(a) != canonical_type(b):
        return False
    sa, sb = str(a.get("start_datetime_pacific") or "")[:10], str(b.get("start_datetime_pacific") or "")[:10]
    if sa and sb and sa != sb:
        return False
    name_sim = difflib.SequenceMatcher(None, normalize_name(a.get("final_name")), normalize_name(b.get("final_name"))).ratio()
    loc_sim = difflib.SequenceMatcher(None, normalize_location(a.get("location")), normalize_location(b.get("location"))).ratio()
    return name_sim >= 0.94 and loc_sim >= 0.70


def lexical_blocks(records: dict[str, dict[str, Any]]) -> list[list[str]]:
    buckets: dict[str, list[str]] = collections.defaultdict(list)
    for rid, rec in records.items():
        typ, name_key, loc_key, start_key = record_key(rec)
        for fire in primary_fire_names(rec):
            buckets[f"fire:{fire}"].append(rid)
        if typ and name_key:
            buckets[f"name:{typ}:{name_key}"].append(rid)
        if typ and name_key and start_key:
            buckets[f"name-date:{typ}:{name_key}:{start_key}"].append(rid)
        if typ and name_key and loc_key:
            buckets[f"name-loc:{typ}:{name_key}:{loc_key}"].append(rid)
        for url in extract_urls(rec)[:3]:
            buckets[f"url:{typ}:{url}"].append(rid)
    return [sorted(set(v)) for v in buckets.values() if len(set(v)) >= 2]


def load_embedding_model(disabled: bool) -> Any | None:
    if disabled or SentenceTransformer is None or np is None:
        return None
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"WARNING: sentence-transformers unavailable: {e}", file=sys.stderr)
        return None


def embedding_blocks(records: dict[str, dict[str, Any]], model: Any | None, threshold: float, top_k: int) -> list[list[str]]:
    if model is None or np is None or len(records) < 2:
        return []
    ids = list(records)
    texts = [record_text_for_embedding(records[rid]) for rid in ids]
    try:
        emb = np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False))
    except Exception as e:
        print(f"WARNING: embedding encode failed: {e}", file=sys.stderr)
        return []
    blocks = []
    by_type: dict[str, list[int]] = collections.defaultdict(list)
    for i, rid in enumerate(ids):
        by_type[canonical_type(records[rid])].append(i)
    for typ, idxs in by_type.items():
        if len(idxs) < 2:
            continue
        sub = emb[idxs]
        sim = sub @ sub.T
        for local_i, global_i in enumerate(idxs):
            order = np.argsort(-sim[local_i])
            block = [ids[global_i]]
            for local_j in order[1: top_k + 1]:
                if sim[local_i, local_j] < threshold:
                    continue
                block.append(ids[idxs[local_j]])
            if len(block) >= 2:
                blocks.append(sorted(set(block)))
    return blocks


def block_fingerprint(ids: list[str], records: dict[str, dict[str, Any]]) -> str:
    material = [
        {
            "id": rid,
            "name": records[rid].get("final_name"),
            "type": canonical_type(records[rid]),
            "loc": records[rid].get("location"),
            "start": records[rid].get("start_datetime_pacific"),
            "fires": sorted(primary_fire_names(records[rid])),
        }
        for rid in ids
    ]
    raw = json.dumps(material, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def json_from_text(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    s, e = raw.find("{"), raw.rfind("}")
    if s >= 0 and e > s:
        try:
            data = json.loads(raw[s:e + 1])
            return data if isinstance(data, dict) else None
        except Exception:
            return None
    return None


class LLMClient:
    def __init__(self, api_key: str | None, model: str) -> None:
        if api_key and OpenAI is None:
            raise RuntimeError("openai package missing. Run: pip install openai")
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.model = model

    def enabled(self) -> bool:
        return self.client is not None

    def complete_json(self, system: str, payload: dict[str, Any], retries: int = 3) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("LLM disabled")
        msg = "Return only valid JSON. Payload:\n" + json.dumps(payload, indent=2, ensure_ascii=False)
        last = None
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": msg}],
                )
                data = json_from_text(resp.choices[0].message.content or "")
                if data is None:
                    raise ValueError("non-JSON LLM response")
                return data
            except Exception as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM failed after retries: {last}")


MERGE_PROMPT = """
You are merging low-level incident records.

Merge records only if they refer to the SAME concrete low-level incident.

Hard rules:
- Do NOT merge different named fires: Palisades Fire, Eaton Fire, Hurst Fire, Lydia Fire, etc.
- Parent/aggregate phrases such as "Palisades and Eaton fires" or "Los Angeles wildfires"
  are not evidence that a single low-level record is both fires.
- Do NOT merge different incident types.
- Prefer no merge when uncertain.

Return exactly:
{
  "merge_groups": [
    {"record_ids": ["id1", "id2"], "reason": "why same incident"}
  ],
  "notes": "brief notes"
}
"""


def compact_merge_record(rid: str, rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": rid,
        "final_name": rec.get("final_name"),
        "incident_type": canonical_type(rec),
        "raw_incident_type": rec.get("incident_type"),
        "location": rec.get("location"),
        "start_datetime_pacific": rec.get("start_datetime_pacific"),
        "end_datetime_pacific": rec.get("end_datetime_pacific"),
        "primary_fire_names": sorted(primary_fire_names(rec)),
        "all_fire_names_anywhere": sorted(all_fire_names_anywhere(rec)),
        "original_incident_names": (rec.get("original_incident_names") or [])[:10],
        "source_parent_names": (rec.get("source_parent_names") or [])[:10],
        "source_incident_ids": (rec.get("source_incident_ids") or [])[:10],
        "urls": extract_urls(rec)[:8],
    }


def merge_group_records(final_id: str, ids: list[str], records: dict[str, dict[str, Any]], article_index: dict[str, Any], reference_records: dict[str, dict[str, Any]], min_article_score: int) -> tuple[dict[str, Any], dict[str, Any]]:
    group = [records[i] for i in ids]
    fires = sorted(set().union(*(primary_fire_names(r) for r in group)))
    if len(fires) == 1:
        final_name = fires[0]
        incident_type = "wildfire"
    else:
        names = [str(r.get("final_name")) for r in group if r.get("final_name")]
        names = sorted(names, key=lambda n: ("|" in n, aggregate_like_text(n), len(n)))
        final_name = names[0] if names else final_id
        incident_type = collections.Counter(canonical_type(r) for r in group).most_common(1)[0][0]

    locs = [str(r.get("location")) for r in group if r.get("location")]
    locs = sorted(locs, key=lambda x: (normalize_location(x) in {"los angeles", "la", "l a", ""}, -len(x)))
    location = locs[0] if locs else None

    base = {
        "final_name": final_name,
        "incident_type": incident_type,
        "location": location,
        "start_datetime_pacific": earliest_dt(r.get("start_datetime_pacific") for r in group),
        "end_datetime_pacific": latest_dt(r.get("end_datetime_pacific") for r in group),
        "specific_fire_names": fires,
    }

    source_clean_ids, source_low_ids, source_incident_ids = [], [], []
    original_names, source_parent_names, suggested_parent_names = [], [], []
    source_event_refs, urls, all_urls, notes = [], [], [], []
    for old_id, r in zip(ids, group):
        source_clean_ids.append(old_id)
        source_low_ids.extend(r.get("source_low_level_ids") or [])
        source_incident_ids.extend(r.get("source_incident_ids") or [])
        original_names.extend(r.get("original_incident_names") or [])
        if r.get("final_name"):
            original_names.append(r.get("final_name"))
        source_parent_names.extend(r.get("source_parent_names") or [])
        suggested_parent_names.extend(r.get("suggested_parent_names") or [])
        source_event_refs.extend(r.get("source_event_refs") or [])
        urls.extend(r.get("urls_used") or [])
        all_urls.extend(extract_urls(r))
        if r.get("final_cleanup_notes"):
            notes.append(r["final_cleanup_notes"])

    evidence_record = dict(base)
    evidence_record.update({
        "source_incident_ids": dedupe_list(source_incident_ids),
        "source_event_refs": dedupe_list(source_event_refs),
        "urls_used": dedupe_list(urls),
        "all_candidate_urls": dedupe_list(all_urls),
        "original_incident_names": dedupe_list(original_names),
        "source_parent_names": dedupe_list(source_parent_names),
        "suggested_parent_names": dedupe_list(suggested_parent_names),
        "source_low_level_ids": dedupe_list(source_low_ids),
    })

    evidence = gather_article_evidence(evidence_record, article_index, reference_records)
    earliest_article = choose_earliest_article(evidence, min_score=min_article_score)
    public_evidence = [{k: v for k, v in e.items() if not k.startswith("_")} for e in evidence]

    evidence_urls = [e.get("url") for e in public_evidence if e.get("url")]
    final = {
        "low_level_incident_id": final_id,
        **base,
        "source_clean_low_level_ids": dedupe_list(source_clean_ids),
        "source_low_level_ids": dedupe_list(source_low_ids),
        "source_incident_ids": dedupe_list(source_incident_ids),
        "original_incident_names": dedupe_list(original_names),
        "source_parent_names": dedupe_list(source_parent_names),
        "suggested_parent_names": dedupe_list(suggested_parent_names),
        "urls_used": dedupe_list(urls + evidence_urls),
        "all_candidate_urls": dedupe_list(all_urls + evidence_urls),
        "source_event_refs": dedupe_list(source_event_refs),
        "evidence_record_count": len(group),
        "article_evidence": public_evidence,
        "earliest_article": earliest_article,
        "earliest_article_date_raw": earliest_article.get("article_date_raw") if earliest_article else None,
        "earliest_article_datetime_pacific": earliest_article.get("article_datetime_pacific") if earliest_article else None,
        "earliest_article_url": earliest_article.get("url") if earliest_article else None,
        "final_cleanup_notes": " | ".join(dedupe_list(notes)),
    }
    audit = {
        "source_record_ids": ids,
        "primary_fire_names": fires,
        "earliest_article": earliest_article,
        "article_evidence_count": len(public_evidence),
        "top_article_evidence": public_evidence[:20],
    }
    return final, audit


def build_duplicate_report(final_low: dict[str, dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[str]] = collections.defaultdict(list)
    for rid, rec in final_low.items():
        buckets["|".join(record_key(rec))].append(rid)
    dupes = {k: v for k, v in buckets.items() if len(v) > 1}
    return {"duplicate_key_count": len(dupes), "duplicate_keys": dupes}


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object/dict at {path}")
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def load_reference_records(paths: list[Path]) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for p in paths:
        if not p.exists():
            print(f"WARNING: reference file not found: {p}", file=sys.stderr)
            continue
        for rid, rec in load_records(p).items():
            refs[f"{p.name}:{rid}"] = rec
    return refs


# ---------------------------------------------------------------------------
# Focused top-level grouping: Los Angeles Wildfires and Anti-ICE Protests
# ---------------------------------------------------------------------------

LA_LOCATION_HINTS = {
    "los angeles", "l a", "l.a.", "la county", "los angeles county", "pacific palisades",
    "palisades", "eaton", "pasadena", "altadena", "hurst", "sylmar", "malibu",
    "hollywood", "west hollywood", "north hollywood", "santa monica", "calabasas",
    "topanga", "burbank", "glendale", "long beach", "compton", "inglewood",
    "torrance", "venice", "san pedro", "wilmington", "santa clarita", "san fernando",
    "woodland hills", "encino", "west hills", "castaic", "acton", "arcadia",
    "monrovia", "azusa", "el monte", "whittier", "beverly hills",
}

ICE_TERMS = {
    "ice", "immigration", "deportation", "federal agents", "federal agent",
    "federal immigration", "immigration raid", "raids", "national guard",
    "customs enforcement", "homeland security", "dhs",
}

TOP_PROMPT = """
You are building a two-level incident hierarchy for a Los Angeles sensing dataset.

There are only two possible top-level incident families in this stage:
1. Los Angeles wildfires: a January 2025 Los Angeles-area wildfire complex containing distinct
   low-level fires such as Palisades Fire, Eaton Fire, Hurst Fire, Kenneth Fire, Sunset Fire,
   Lydia/Lidia Fire, Hughes Fire, etc.
2. Los Angeles anti-ICE protests: a Los Angeles-area protest wave about ICE, immigration raids,
   deportation, federal agents, or National Guard deployment. Children should be distinct concrete
   protest occurrences/locations, not the same protest duplicated.

Given one proposed parent and candidate children, select only the low-level incident IDs that truly
belong to that parent. Use incident type, location, dates, names, and evidence fields. Exclude
ambiguous children. A valid parent must have at least two children.

Return exactly JSON:
{
  "is_valid_top_level": true/false,
  "top_level_name": "string",
  "child_low_level_ids": ["final_low_..."],
  "reason": "brief reason"
}
"""


def is_la_area_text(text: Any) -> bool:
    n = normalize_text(text)
    return any(h in n for h in LA_LOCATION_HINTS)


def low_record_date(record: dict[str, Any]) -> datetime | None:
    return parse_dt(record.get("start_datetime_pacific"))


def in_date_range(record: dict[str, Any], start: str, end: str) -> bool:
    dt = low_record_date(record)
    if not dt:
        return False
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end)
    return s <= dt <= e


def wildfire_parent_score(record: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    blob = record_blob(record, include_parent_hints=True)
    nblob = normalize_text(blob)

    if canonical_type(record) != "wildfire":
        return -100, ["not_wildfire"]

    if primary_fire_names(record):
        score += 50
        reasons.append("named_fire")

    if is_la_area_text(blob):
        score += 25
        reasons.append("la_area")

    # The main LA wildfire complex is January 2025, but allow a little margin for
    # late January / early February reporting-derived start dates.
    if in_date_range(record, "2025-01-01T00:00:00", "2025-02-15T23:59:59"):
        score += 35
        reasons.append("jan_2025_window")
    elif in_date_range(record, "2025-01-01T00:00:00", "2025-03-31T23:59:59"):
        score += 15
        reasons.append("near_jan_2025_window")

    if any(t in nblob for t in ["los angeles wildfires", "los angeles fires", "la fires", "palisades and eaton"]):
        score += 20
        reasons.append("parent_hint")

    # Demote aftermath-only records if they survived as low-level records.
    if any(t in nblob for t in ["school closing", "school reopening", "reopening", "cleanup", "lawsuit", "trial", "relief center"]):
        score -= 20
        reasons.append("aftermath_or_procedural_hint")

    return score, reasons


def ice_parent_score(record: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    blob = record_blob(record, include_parent_hints=True)
    nblob = normalize_text(blob)

    if canonical_type(record) != "civil protest":
        return -100, ["not_civil_protest"]

    if any(t in nblob for t in ICE_TERMS):
        score += 45
        reasons.append("ice_or_immigration_terms")

    if any(t in nblob for t in ["protest", "demonstration", "rally", "march", "walkout"]):
        score += 20
        reasons.append("protest_terms")

    if is_la_area_text(blob):
        score += 25
        reasons.append("la_area")

    # Keep this broad because the source dataset may contain 2025/2026 protest entries,
    # but favor the known LA anti-ICE protest period around June 2025.
    if in_date_range(record, "2025-06-01T00:00:00", "2025-07-31T23:59:59"):
        score += 30
        reasons.append("june_2025_window")
    elif in_date_range(record, "2025-01-01T00:00:00", "2026-05-31T23:59:59"):
        score += 10
        reasons.append("study_window")

    if any(t in nblob for t in ["ice protests", "anti ice protests", "anti ice", "immigration protests"]):
        score += 20
        reasons.append("parent_hint")

    if any(t in nblob for t in ["threatened", "could strike", "trial", "hearing", "lawsuit"]):
        score -= 25
        reasons.append("non_occurrence_or_procedural_hint")

    return score, reasons


def compact_top_child(rid: str, record: dict[str, Any], score: int, reasons: list[str]) -> dict[str, Any]:
    return {
        "low_level_id": rid,
        "final_name": record.get("final_name"),
        "incident_type": record.get("incident_type"),
        "location": record.get("location"),
        "start_datetime_pacific": record.get("start_datetime_pacific"),
        "end_datetime_pacific": record.get("end_datetime_pacific"),
        "specific_fire_names": record.get("specific_fire_names"),
        "source_parent_names": (record.get("source_parent_names") or [])[:8],
        "suggested_parent_names": (record.get("suggested_parent_names") or [])[:8],
        "original_incident_names": (record.get("original_incident_names") or [])[:8],
        "earliest_article_datetime_pacific": record.get("earliest_article_datetime_pacific"),
        "urls_used": (record.get("urls_used") or [])[:5],
        "heuristic_score": score,
        "heuristic_reasons": reasons,
    }


def top_decision_fingerprint(parent_name: str, candidates: list[tuple[str, dict[str, Any], int, list[str]]]) -> str:
    material = {
        "parent_name": parent_name,
        "children": [
            {
                "id": rid,
                "name": rec.get("final_name"),
                "type": rec.get("incident_type"),
                "loc": rec.get("location"),
                "start": rec.get("start_datetime_pacific"),
                "score": score,
            }
            for rid, rec, score, _ in candidates
        ],
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def review_top_with_llm(
    parent_name: str,
    parent_type: str,
    candidates: list[tuple[str, dict[str, Any], int, list[str]]],
    llm: LLMClient,
    cache: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    fp = top_decision_fingerprint(parent_name, candidates)
    if fp in cache and not args.force_top:
        return cache[fp]

    if not llm.enabled() or args.no_llm_top:
        child_ids = [rid for rid, _, score, _ in candidates if score >= args.top_score_threshold]
        decision = {
            "is_valid_top_level": len(child_ids) >= 2,
            "top_level_name": parent_name,
            "child_low_level_ids": child_ids,
            "reason": "heuristic top-level grouping because LLM top review is disabled",
        }
        cache[fp] = decision
        atomic_save_json(args.top_cache, cache)
        return decision

    payload = {
        "top_level_candidate": {
            "name": parent_name,
            "incident_type": parent_type,
        },
        "candidate_low_level_children": [
            compact_top_child(rid, rec, score, reasons)
            for rid, rec, score, reasons in candidates
        ],
    }
    try:
        decision = llm.complete_json(TOP_PROMPT, payload)
    except Exception as e:
        print(f"WARNING: top-level LLM failed for {parent_name}: {e}", file=sys.stderr)
        child_ids = [rid for rid, _, score, _ in candidates if score >= args.top_score_threshold]
        decision = {
            "is_valid_top_level": len(child_ids) >= 2,
            "top_level_name": parent_name,
            "child_low_level_ids": child_ids,
            "reason": f"LLM failed; heuristic fallback: {e}",
        }

    cache[fp] = decision
    atomic_save_json(args.top_cache, cache)
    return decision


def build_focused_top_level(final_low: dict[str, dict[str, Any]], llm: LLMClient, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build top_level.json for only the two expected aggregate families."""
    top_cache = {} if args.force_top else load_json(args.top_cache, default={})

    wildfire_candidates: list[tuple[str, dict[str, Any], int, list[str]]] = []
    ice_candidates: list[tuple[str, dict[str, Any], int, list[str]]] = []

    for rid, rec in final_low.items():
        w_score, w_reasons = wildfire_parent_score(rec)
        if w_score >= args.top_candidate_min_score:
            wildfire_candidates.append((rid, rec, w_score, w_reasons))

        i_score, i_reasons = ice_parent_score(rec)
        if i_score >= args.top_candidate_min_score:
            ice_candidates.append((rid, rec, i_score, i_reasons))

    wildfire_candidates.sort(key=lambda x: (-x[2], str(x[1].get("start_datetime_pacific") or ""), str(x[1].get("final_name") or "")))
    ice_candidates.sort(key=lambda x: (-x[2], str(x[1].get("start_datetime_pacific") or ""), str(x[1].get("final_name") or "")))

    audit: dict[str, Any] = {
        "Los Angeles wildfires": {
            "candidate_count": len(wildfire_candidates),
            "candidates": [compact_top_child(*c) for c in wildfire_candidates],
        },
        "Los Angeles anti-ICE protests": {
            "candidate_count": len(ice_candidates),
            "candidates": [compact_top_child(*c) for c in ice_candidates],
        },
    }

    top: dict[str, Any] = {}

    for parent_name, parent_type, candidates in [
        ("Los Angeles wildfires", "wildfire", wildfire_candidates),
        ("Los Angeles anti-ICE protests", "civil protest", ice_candidates),
    ]:
        if len(candidates) < 2:
            audit[parent_name]["decision"] = {
                "is_valid_top_level": False,
                "reason": "fewer than two heuristic candidates",
            }
            continue

        # Bound the LLM payload but retain all high-scoring candidates.
        bounded = candidates[: args.max_top_candidates]
        decision = review_top_with_llm(parent_name, parent_type, bounded, llm, top_cache, args)
        audit[parent_name]["decision"] = decision

        child_ids = [rid for rid in decision.get("child_low_level_ids", []) if rid in final_low]
        child_ids = dedupe_list(child_ids)
        if not decision.get("is_valid_top_level") or len(child_ids) < 2:
            continue

        recs = [final_low[rid] for rid in child_ids]
        tid = f"top_{len(top):06d}"
        top[tid] = {
            "top_level_incident_id": tid,
            "top_level_incident_name": decision.get("top_level_name") or parent_name,
            "low_level_incident_ids": child_ids,
            "start_datetime_pacific": earliest_dt(r.get("start_datetime_pacific") for r in recs),
            "end_datetime_pacific": latest_dt(r.get("end_datetime_pacific") for r in recs),
            "incident_types": dedupe_list(r.get("incident_type") for r in recs),
            "reason": decision.get("reason"),
        }
        atomic_save_json(args.top_output, top)

    atomic_save_json(args.top_audit_output, audit)
    atomic_save_json(args.top_cache, top_cache)
    return top, audit


def run(args: argparse.Namespace) -> None:
    llm = LLMClient(args.openai_api_key or os.environ.get("OPENAI_API_KEY"), args.model)

    if args.top_only:
        final_low = load_records(args.low_output)
        top_level, top_audit = build_focused_top_level(final_low, llm, args)
        summary = {
            "mode": "top_only",
            "input_low_level_final": str(args.low_output),
            "low_level_final_count": len(final_low),
            "top_level_count": len(top_level),
            "llm_top_enabled": llm.enabled() and not args.no_llm_top,
            "outputs": {
                "top_level": str(args.top_output),
                "top_audit": str(args.top_audit_output),
            },
        }
        atomic_save_json(args.summary_output, summary)
        print(json.dumps(summary, indent=2))
        return

    low = load_records(args.low_input)
    refs = load_reference_records(args.reference_low)
    all_merged = load_json(args.all_merged)
    if not isinstance(all_merged, dict):
        raise ValueError(f"Expected JSON object/dict at {args.all_merged}")
    article_index = build_article_index(all_merged)

    embed_model = load_embedding_model(args.no_embedding)
    merge_cache = {} if args.force else load_json(args.merge_cache, default={})

    uf = UnionFind(low.keys())
    blocks = lexical_blocks(low)
    blocks.extend(embedding_blocks(low, embed_model, args.embedding_threshold, args.embedding_top_k))

    unique_blocks = []
    seen_blocks = set()
    for block in blocks:
        block = sorted(set(block))
        if len(block) < 2 or len(block) > args.max_merge_block_size:
            continue
        fp = tuple(block)
        if fp in seen_blocks:
            continue
        seen_blocks.add(fp)
        unique_blocks.append(block)

    # Deterministic safe merges first.
    for block in unique_blocks:
        for i in range(len(block)):
            for j in range(i + 1, len(block)):
                if should_merge_deterministic(low[block[i]], low[block[j]]):
                    uf.union(block[i], block[j])

    iterator = unique_blocks
    if tqdm and not args.no_progress:
        iterator = tqdm(iterator, desc="LLM merge review", unit="block")

    for block in iterator:
        fp = block_fingerprint(block, low)
        if fp in merge_cache and not args.force:
            decision = merge_cache[fp]
        elif llm.enabled() and not args.no_llm_merge:
            payload = {"candidate_records": [compact_merge_record(rid, low[rid]) for rid in block]}
            try:
                decision = llm.complete_json(MERGE_PROMPT, payload)
            except Exception as e:
                print(f"WARNING: LLM merge failed for block {fp}: {e}", file=sys.stderr)
                decision = {"merge_groups": [], "notes": f"LLM failed: {e}"}
            merge_cache[fp] = decision
            atomic_save_json(args.merge_cache, merge_cache)
        else:
            decision = {"merge_groups": [], "notes": "LLM merge disabled"}

        for group in decision.get("merge_groups", []):
            ids = [rid for rid in group.get("record_ids", []) if rid in low]
            if len(ids) < 2:
                continue
            first = ids[0]
            for other in ids[1:]:
                if do_not_merge(low[first], low[other])[0]:
                    continue
                uf.union(first, other)

    groups = [sorted(g) for g in uf.groups().values()]
    groups.sort(key=lambda g: (
        str(low[g[0]].get("start_datetime_pacific") or ""),
        sorted(primary_fire_names(low[g[0]]))[0] if primary_fire_names(low[g[0]]) else normalize_name(low[g[0]].get("final_name")),
        g[0],
    ))

    final_low: dict[str, dict[str, Any]] = {}
    merge_map: dict[str, str] = {}
    article_audit: dict[str, Any] = {}

    iterator2 = groups
    if tqdm and not args.no_progress:
        iterator2 = tqdm(iterator2, desc="Writing low_level_final", unit="group")

    for idx, ids in enumerate(iterator2):
        fid = f"final_low_{idx:06d}"
        rec, audit = merge_group_records(fid, ids, low, article_index, refs, args.min_article_score)
        final_low[fid] = rec
        article_audit[fid] = audit
        for old_id in ids:
            merge_map[old_id] = fid
        atomic_save_json(args.low_output, final_low)
        atomic_save_json(args.merge_map_output, merge_map)
        atomic_save_json(args.article_audit_output, article_audit)

    dup_report = build_duplicate_report(final_low)
    atomic_save_json(args.duplicate_report_output, dup_report)

    top_level, top_audit = build_focused_top_level(final_low, llm, args)

    summary = {
        "input_count": len(low),
        "reference_count": len(refs),
        "candidate_merge_blocks": len(unique_blocks),
        "output_count": len(final_low),
        "top_level_count": len(top_level),
        "duplicate_key_count": dup_report["duplicate_key_count"],
        "llm_merge_enabled": llm.enabled() and not args.no_llm_merge,
        "llm_top_enabled": llm.enabled() and not args.no_llm_top,
        "embedding_enabled": embed_model is not None,
        "outputs": {
            "low_level_final": str(args.low_output),
            "top_level": str(args.top_output),
            "top_audit": str(args.top_audit_output),
            "merge_map": str(args.merge_map_output),
            "article_audit": str(args.article_audit_output),
            "duplicate_report": str(args.duplicate_report_output),
        },
    }
    atomic_save_json(args.summary_output, summary)
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Low-level-only final cleanup with similar-name article evidence recovery.")
    p.add_argument("--low-input", type=Path, default=DEFAULT_LOW_INPUT)
    p.add_argument("--reference-low", type=Path, action="append", default=[DEFAULT_REFERENCE_LOW])
    p.add_argument("--all-merged", type=Path, default=DEFAULT_ALL_MERGED)
    p.add_argument("--low-output", type=Path, default=DEFAULT_LOW_OUTPUT)
    p.add_argument("--top-output", type=Path, default=DEFAULT_TOP_OUTPUT)
    p.add_argument("--top-audit-output", type=Path, default=DEFAULT_TOP_AUDIT)
    p.add_argument("--top-cache", type=Path, default=DEFAULT_TOP_CACHE)
    p.add_argument("--merge-cache", type=Path, default=DEFAULT_MERGE_CACHE)
    p.add_argument("--merge-map-output", type=Path, default=DEFAULT_MERGE_MAP)
    p.add_argument("--article-audit-output", type=Path, default=DEFAULT_ARTICLE_AUDIT)
    p.add_argument("--duplicate-report-output", type=Path, default=DEFAULT_DUPLICATE_REPORT)
    p.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--openai-api-key", default=None)
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--no-llm-merge", action="store_true")
    p.add_argument("--no-llm-top", action="store_true")
    p.add_argument("--top-only", action="store_true", help="Use existing low_level_final.json and rebuild only top_level.json.")
    p.add_argument("--force", action="store_true", help="Recompute LLM merge decisions.")
    p.add_argument("--force-top", action="store_true", help="Recompute LLM top-level decisions.")
    p.add_argument("--no-embedding", action="store_true")
    p.add_argument("--embedding-threshold", type=float, default=0.82)
    p.add_argument("--embedding-top-k", type=int, default=8)
    p.add_argument("--max-merge-block-size", type=int, default=40)
    p.add_argument("--min-article-score", type=int, default=45)
    p.add_argument("--top-candidate-min-score", type=int, default=55)
    p.add_argument("--top-score-threshold", type=int, default=65)
    p.add_argument("--max-top-candidates", type=int, default=80)
    p.add_argument("--no-progress", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
