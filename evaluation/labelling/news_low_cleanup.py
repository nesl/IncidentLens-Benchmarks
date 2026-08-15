#!/usr/bin/env python3
"""Iteratively clean and regroup low-level incident JSON.

This script is intentionally standalone. It does not import or modify news_refine.py.

It is designed for repeated cleanup passes:
    input low_level.json or low_level_cleaned.json
        -> filter bad records
        -> recover better evidence URLs from optional reference files
        -> merge duplicate concrete incidents
        -> rebuild top_level.json
        -> output a new low_level_cleaned.json

The main design choices are intentionally stricter than the earlier scripts:
  * Default LLM mode is "all" when an OpenAI key is provided.
  * Type/name contradictions are rejected or corrected before merging.
  * Named wildfires are canonicalized to incident_type="wildfire".
  * Records naming multiple concrete fires, e.g. "Palisades and Eaton fires",
    are promoted to top-level candidates, not merged into low-level incidents.
  * URL evidence is ranked. Direct incident URLs are preferred; lifestyle,
    celebrity, political-debate, legal, shelter/intake, and other adjacent URLs
    are demoted or removed from urls_used when better direct URLs exist.
  * The script can accept reference low-level files to recover good URLs that
    may have been dropped by a previous cleanup pass.

Typical usage:
    python news_low_iterative_cleanup.py \
      --input evaluation/out/low_level_cleaned.json \
      --reference evaluation/out/low_level.json \
      --openai-api-key "$OPENAI_API_KEY" \
      --force

First pass from raw low-level:
    python news_low_iterative_cleanup.py \
      --input evaluation/out/low_level.json \
      --openai-api-key "$OPENAI_API_KEY" \
      --force

Heuristic-only:
    python news_low_iterative_cleanup.py --input evaluation/out/low_level_cleaned.json --no-llm --no-fetch-web --force

Outputs by default:
    evaluation/out/low_level_cleaned.json
    evaluation/out/top_level.json
    evaluation/out/low_level_iter_rejects.json
    evaluation/out/low_level_iter_merge_map.json
    evaluation/out/low_level_iter_top_candidates.json
    evaluation/out/low_level_iter_url_audit.json
    evaluation/out/low_level_iter_duplicate_report.json
    evaluation/out/low_level_iter_summary.json
"""

from __future__ import annotations

import argparse
import collections
import difflib
import html
import json
import os
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


DEFAULT_INPUT = Path("evaluation/out/low_level_cleaned.json")
FALLBACK_INPUT = Path("evaluation/out/low_level.json")

DEFAULT_LOW_OUTPUT = Path("evaluation/out/low_level_cleaned.json")
DEFAULT_TOP_OUTPUT = Path("evaluation/out/top_level.json")
DEFAULT_REJECTS_OUTPUT = Path("evaluation/out/low_level_iter_rejects.json")
DEFAULT_MERGE_MAP_OUTPUT = Path("evaluation/out/low_level_iter_merge_map.json")
DEFAULT_TOP_CANDIDATES_OUTPUT = Path("evaluation/out/low_level_iter_top_candidates.json")
DEFAULT_URL_AUDIT_OUTPUT = Path("evaluation/out/low_level_iter_url_audit.json")
DEFAULT_DUPLICATE_REPORT_OUTPUT = Path("evaluation/out/low_level_iter_duplicate_report.json")
DEFAULT_DECISIONS_CACHE = Path("evaluation/out/low_level_iter_decisions.json")
DEFAULT_MERGE_DECISIONS_CACHE = Path("evaluation/out/low_level_iter_merge_decisions.json")
DEFAULT_LLM_MERGE_AUDIT_OUTPUT = Path("evaluation/out/low_level_iter_llm_merge_audit.json")
DEFAULT_WEB_CACHE = Path("evaluation/out/low_level_iter_article_cache.json")
DEFAULT_SUMMARY_OUTPUT = Path("evaluation/out/low_level_iter_summary.json")

DEFAULT_START_DATE = "2025-01-01"
DEFAULT_END_DATE = "2026-05-31"

TARGET_TYPES = {"civil protest", "urban fire", "wildfire", "terrorist incident"}

# These are canonical named wildfires that repeatedly appear in Los Angeles Jan 2025 data.
# They are used only for disambiguation/canonicalization. They prevent, for example,
# "Palisades and Eaton fires" from becoming a low-level incident.
SPECIFIC_FIRE_CANONICAL = {
    "palisades": "Palisades Fire",
    "eaton": "Eaton Fire",
    "hurst": "Hurst Fire",
    "kenneth": "Kenneth Fire",
    "sunset": "Sunset Fire",
    "lydia": "Lydia Fire",
    "hughes": "Hughes Fire",
    "franklin": "Franklin Fire",
    "bridge": "Bridge Fire",
    "airport": "Airport Fire",
    "mountain": "Mountain Fire",
    "post": "Post Fire",
    "lake": "Lake Fire",
    "borel": "Borel Fire",
}

KNOWN_FIRE_LOCATION = {
    "Palisades Fire": "Pacific Palisades, Los Angeles, California",
    "Eaton Fire": "Eaton Canyon / Altadena-Pasadena area, Los Angeles County, California",
    "Hurst Fire": "Sylmar, Los Angeles, California",
    "Kenneth Fire": "West Hills / Calabasas area, Los Angeles County, California",
    "Sunset Fire": "Hollywood Hills, Los Angeles, California",
    "Lydia Fire": "Acton area, Los Angeles County, California",
    "Hughes Fire": "Castaic Lake area, Los Angeles County, California",
    "Franklin Fire": "Malibu, California",
}

KNOWN_FIRE_START = {
    "Palisades Fire": "2025-01-07T00:00:00",
    "Eaton Fire": "2025-01-07T00:00:00",
    "Hurst Fire": "2025-01-07T00:00:00",
    "Kenneth Fire": "2025-01-09T00:00:00",
    "Sunset Fire": "2025-01-08T00:00:00",
    "Lydia Fire": "2025-01-08T00:00:00",
    "Hughes Fire": "2025-01-22T00:00:00",
}

LA_HINTS = {
    "los angeles", "l a", "l.a.", "la county", "los angeles county",
    "pacific palisades", "palisades", "eaton", "eaton canyon", "altadena",
    "pasadena", "hurst", "sylmar", "malibu", "topanga", "calabasas",
    "hollywood", "west hollywood", "north hollywood", "hollywood hills",
    "santa monica", "burbank", "glendale", "long beach", "compton",
    "inglewood", "torrance", "venice", "san pedro", "wilmington",
    "santa clarita", "san fernando", "woodland hills", "encino",
    "west hills", "canyon country", "castaic", "acton", "arcadia",
    "monrovia", "azusa", "el monte", "whittier", "beverly hills",
    "ucla", "usc", "la city", "city of los angeles",
}

NON_TARGET_NAME_TERMS = {
    "shooting", "officer involved shooting", "officer-involved shooting",
    "lapd shooting", "crash", "collision", "traffic crash", "hit and run",
    "stabbing", "robbery", "burglary", "homicide", "murder",
}

ADJACENT_TERMS = {
    "horse intake", "animal intake", "equestrian center", "animal rescue",
    "pet rescue", "pets", "shelter", "relief center", "resource center",
    "assistance center", "donation", "fundraiser", "aid center",
    "evacuation center", "recovery center", "celebrity", "celebrities",
    "paris hilton", "meghan markle", "prince harry", "grammys", "oscars",
    "awards", "red carpet", "benefit concert", "lost homes", "home loss",
}

PROCEDURAL_TERMS = {
    "trial", "hearing", "lawsuit", "court", "courtroom", "sentencing",
    "plea", "council chamber", "committee", "city council", "plum hearing",
    "meeting", "board meeting", "debate", "mayoral election", "election",
    "campaign", "press conference", "briefing", "report released", "study",
    "claims", "landmark social media trial",
}

THREAT_TERMS = {
    "threatened", "threat of", "could strike", "may strike", "might strike",
    "planned strike", "strike authorization", "vote to authorize", "mediation",
    "negotiation", "warning of", "potential", "possible", "preparing for",
    "expected to", "could happen", "threaten to strike", "would strike",
}

DIRECT_EVENT_TERMS = {
    "fire", "wildfire", "brush fire", "structure fire", "burn scar", "evacuation order",
    "evacuation warning", "protest", "demonstration", "march", "rally", "walkout",
    "strike", "attack", "bomb", "explosion", "terror", "terrorist",
}

AGGREGATE_TERMS = {
    "wildfires", "fires", "la fires", "los angeles fires", "los angeles wildfires",
    "palisades and eaton", "eaton and palisades",
    "protests", "demonstrations", "marches", "rallies",
    "ice protests", "anti ice protests", "anti-ice protests", "immigration protests",
}

DIRECT_DOMAINS = {
    "latimes.com", "ktla.com", "nbclosangeles.com", "abc7.com", "cbsnews.com",
    "abcnews.go.com", "apnews.com", "reuters.com", "nbcnews.com", "cnn.com",
    "nytimes.com", "washingtonpost.com", "foxla.com", "dailynews.com",
    "pasadenastarnews.com", "sgvtribune.com", "fire.ca.gov", "lafd.org",
    "lacity.gov", "lacounty.gov", "yahoo.com", "courthousenews.com",
}

INDIRECT_DOMAINS = {
    "iraqinews.com", "tmz.com", "people.com", "pagesix.com", "thenews.com.pk",
    "lasvegassun.com", "nbcchicago.com", "nbcmiami.com",
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
    stop = {"the", "a", "an", "incident", "event", "reported", "report", "usa", "united", "states", "america", "california", "ca", "county"}
    return " ".join(t for t in text.split() if t not in stop)


def normalize_location(value: Any) -> str:
    text = normalize_text(value)
    # Keep "los angeles"; only remove country/state suffix tokens.
    drop = {"usa", "united", "states", "america", "california", "ca", "county", "city", "of", "the"}
    return " ".join(t for t in text.split() if t not in drop)


def dedupe_list(values: Iterable[Any]) -> list[Any]:
    out = []
    seen = set()
    for v in values:
        if v in (None, "", []):
            continue
        key = json.dumps(v, sort_keys=True, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def parse_date_prefix(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", value)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).replace(tzinfo=None)
    except Exception:
        d = parse_date_prefix(text)
        if d:
            return datetime(d.year, d.month, d.day)
    return None


def earliest(values: Iterable[Any]) -> str | None:
    pairs = []
    for v in values:
        dt = parse_dt(v)
        if dt:
            pairs.append((dt, str(v)))
    return min(pairs, key=lambda x: x[0])[1] if pairs else None


def latest(values: Iterable[Any]) -> str | None:
    pairs = []
    for v in values:
        dt = parse_dt(v)
        if dt:
            pairs.append((dt, str(v)))
    return max(pairs, key=lambda x: x[0])[1] if pairs else None


def domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def extract_urls(record: dict[str, Any], max_urls: int | None = None) -> list[str]:
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
    urls = dedupe_list(urls)
    return urls[:max_urls] if max_urls is not None else urls


def record_blob(record: dict[str, Any]) -> str:
    fields = [
        record.get("final_name"), record.get("incident_type"), record.get("location"),
        record.get("start_datetime_pacific"), record.get("end_datetime_pacific"),
        record.get("time_notes"), record.get("location_notes"), record.get("refine_notes"),
        record.get("cleanup_notes"),
    ]
    for key in ("original_incident_names", "source_parent_names", "urls_used", "all_candidate_urls", "specific_fire_names"):
        val = record.get(key)
        if isinstance(val, list):
            fields.extend(str(x) for x in val[:50])
    return " ".join(str(x) for x in fields if x)


def canonical_type(record: dict[str, Any] | Any) -> str:
    if isinstance(record, dict):
        blob = normalize_text(record_blob(record))
        raw = normalize_text(record.get("incident_type"))
    else:
        blob = normalize_text(record)
        raw = blob

    fires = fire_names_in_text(blob)
    if fires or "wildfire" in blob or "brush fire" in blob:
        return "wildfire"
    if raw in TARGET_TYPES:
        return raw
    if "urban fire" in blob or "structure fire" in blob:
        return "urban fire"
    if "civil protest" in blob or "protest" in blob or "demonstration" in blob or "rally" in blob or "strike" in blob or "walkout" in blob:
        return "civil protest"
    if "terror" in blob or "bomb" in blob or "attack" in blob:
        return "terrorist incident"
    if raw:
        return raw
    return ""


def fire_names_in_text(text: Any) -> set[str]:
    norm = normalize_text(text)
    found = set()
    for token, canonical in SPECIFIC_FIRE_CANONICAL.items():
        if token in {"lake", "post", "bridge", "airport", "mountain"}:
            if re.search(rf"\b{token}\s+fire\b", norm):
                found.add(canonical)
        else:
            if re.search(rf"\b{re.escape(token)}\b", norm):
                found.add(canonical)
    return found


def specific_fire_names(record: dict[str, Any]) -> set[str]:
    vals = [record_blob(record)]
    if isinstance(record.get("specific_fire_names"), list):
        vals.extend(str(x) for x in record["specific_fire_names"])
    return fire_names_in_text(" ".join(vals))


def is_la_area(record: dict[str, Any]) -> bool:
    blob = normalize_text(record_blob(record))
    return any(normalize_text(h) in blob for h in LA_HINTS)


def in_date_window(record: dict[str, Any], start: date, end: date) -> tuple[bool, str]:
    known = [parse_date_prefix(record.get("start_datetime_pacific")), parse_date_prefix(record.get("end_datetime_pacific"))]
    known = [d for d in known if d is not None]
    if known:
        if all(d < start for d in known):
            return False, "date_before_window"
        if all(d > end for d in known):
            return False, "date_after_window"
        return True, "date_in_window"

    blob = normalize_text(record_blob(record))
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", blob)]
    if years and not any(start.year <= y <= end.year for y in years):
        return False, "mentioned_year_outside_window"
    return True, "date_unknown_or_in_window"


def is_generic_broad(record: dict[str, Any]) -> bool:
    name = normalize_name(record.get("final_name"))
    loc = normalize_location(record.get("location"))
    typ = canonical_type(record)
    fires = specific_fire_names(record)
    if fires:
        return False
    if "|" in str(record.get("final_name", "")) and loc in {"los angeles", "la", "l a", ""}:
        return True
    generic = {
        "wildfire los angeles", "fire los angeles", "urban fire los angeles",
        "civil protest los angeles", "protest los angeles", "terrorist incident los angeles",
    }
    if name in generic:
        return True
    if typ in TARGET_TYPES and loc in {"los angeles", "la", "l a", ""} and len(name.split()) <= 3:
        return True
    return False


def aggregate_kind(record: dict[str, Any]) -> tuple[bool, str]:
    name = normalize_text(record.get("final_name"))
    blob = normalize_text(record_blob(record))
    fires = specific_fire_names(record)
    typ = canonical_type(record)
    if len(fires) >= 2:
        return True, "multiple_named_fires"
    if typ == "wildfire" and any(term in name for term in ["los angeles wildfires", "los angeles fires", "la fires"]):
        return True, "la_wildfires"
    if typ == "wildfire" and ("wildfires" in name or re.search(r"\bfires\b", name)) and not fires:
        return True, "wildfire_plural"
    if typ == "civil protest" and any(term in blob for term in ["ice protests", "anti ice protests", "anti ice demonstration", "immigration protests"]):
        return True, "ice_protests"
    if typ == "civil protest" and ("protests" in name or "demonstrations" in name or "rallies" in name):
        return True, "protest_plural"
    return False, ""


def deterministic_action(record: dict[str, Any], start: date, end: date) -> tuple[str, str]:
    blob = normalize_text(record_blob(record))
    name_loc = normalize_text(f"{record.get('final_name', '')} {record.get('location', '')}")

    agg, why = aggregate_kind(record)
    if agg:
        return "top_candidate", why

    typ = canonical_type(record)
    if typ not in TARGET_TYPES:
        return "reject", f"type_not_target:{record.get('incident_type')}"

    ok_time, why_time = in_date_window(record, start, end)
    if not ok_time:
        return "reject", why_time

    if not is_la_area(record):
        return "reject", "not_la_area"

    if is_generic_broad(record):
        return "reject", "generic_broad_low_level"

    # Strong type/name contradiction: a shooting is not an urban fire, and shooting
    # is not in the current filtered target labels unless there is explicit terror/bomb/attack context.
    if any(term in name_loc for term in NON_TARGET_NAME_TERMS):
        if not any(term in name_loc for term in ["terror", "terrorist", "bomb", "explosion"]):
            return "reject", "name_indicates_non_target_incident_type"

    if any(term in name_loc for term in ADJACENT_TERMS):
        return "reject", "adjacent_aftermath_or_support_activity"
    if any(term in name_loc for term in PROCEDURAL_TERMS):
        return "reject", "procedural_legal_political_not_incident"
    if any(term in name_loc for term in THREAT_TERMS):
        return "reject", "threat_or_planned_event_not_actual_occurrence"

    if any(term in blob for term in ADJACENT_TERMS | PROCEDURAL_TERMS | THREAT_TERMS):
        return "suspicious", "suspicious_adjacent_procedural_or_threat_terms"

    urls = extract_urls(record)
    domains = {domain_of(u) for u in urls}
    if domains and all(d in INDIRECT_DOMAINS or any(d.endswith("." + x) for x in INDIRECT_DOMAINS) for d in domains):
        return "suspicious", "only_indirect_domains"

    return "keep", "deterministic_keep"


def fetch_article(url: str, timeout: int = 8, max_chars: int = 5000) -> dict[str, Any]:
    result = {"url": url, "domain": domain_of(url), "ok": False, "status_code": None, "title": None, "published_time_raw": None, "text_snippet": "", "error": None}
    if requests is None or BeautifulSoup is None:
        result["error"] = "requests_or_bs4_missing"
        return result
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"},
            timeout=timeout,
        )
        result["status_code"] = resp.status_code
        if resp.status_code >= 400:
            result["error"] = f"http_{resp.status_code}"
            return result
        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.find("title")
        if title:
            result["title"] = title.get_text(" ", strip=True)
        for attr, val in [
            ("property", "article:published_time"),
            ("name", "pubdate"),
            ("name", "datePublished"),
            ("itemprop", "datePublished"),
        ]:
            tag = soup.find("meta", attrs={attr: val})
            if tag and tag.get("content"):
                result["published_time_raw"] = str(tag["content"]).strip()
                break
        for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav"]):
            tag.decompose()
        main = soup.find("article") or soup.find("main") or soup.body or soup
        text = re.sub(r"\s+", " ", main.get_text(" ", strip=True))
        result["text_snippet"] = text[:max_chars]
        result["ok"] = bool(result["text_snippet"])
        return result
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        return result


def fetch_articles(record: dict[str, Any], web_cache: dict[str, Any], fetch_web: bool, timeout: int) -> list[dict[str, Any]]:
    out = []
    for url in extract_urls(record, max_urls=8):
        if url in web_cache:
            out.append(web_cache[url])
            continue
        item = fetch_article(url, timeout=timeout) if fetch_web else {"url": url, "domain": domain_of(url), "ok": False, "text_snippet": "", "error": "fetch_disabled"}
        web_cache[url] = item
        out.append(item)
    return out


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
                    raise ValueError("non-json response")
                return data
            except Exception as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM failed: {last}")


FILTER_PROMPT = """
You are cleaning low-level incident records for a Los Angeles sensing-system evaluation.

Allowed low-level incident types are ONLY:
- civil protest
- urban fire
- wildfire
- terrorist incident

Return action="keep" only for a concrete real-world incident that actually happened,
is in/near Los Angeles, occurred from Jan 1 2025 through May 31 2026, and is directly
relevant to one of the allowed types.

Reject records if:
- the name/type contradict, e.g. "LAPD shooting" labeled "urban fire";
- the record is a shooting/crash/trial/hearing/lawsuit/debate/election story and not one of the allowed types;
- it is adjacent or aftermath only, e.g. animal intake, shelters, relief centers, celebrity homes, Meghan/Prince Harry/Paris Hilton lifestyle stories;
- it is only a threat/planned/possible event rather than an actual event;
- it is a generic broad placeholder like "wildfire | Los Angeles".

Return action="top_candidate" for aggregate records such as:
- Los Angeles wildfires
- Palisades and Eaton fires
- ICE protests / anti-ICE protests
These are not low-level incidents.

Correct obvious metadata:
- named fires such as Eaton Fire, Palisades Fire, Hurst Fire should be incident_type="wildfire";
- do not invent dates; use null if unsupported.

Return exactly JSON:
{
  "action": "keep|reject|top_candidate",
  "reject_reason": "string or null",
  "confidence": 0.0-1.0,
  "corrected_final_name": "string or null",
  "corrected_incident_type": "civil protest|urban fire|wildfire|terrorist incident|null",
  "corrected_location": "string or null",
  "corrected_start_datetime_pacific": "YYYY-MM-DDTHH:MM:SS or null",
  "corrected_end_datetime_pacific": "YYYY-MM-DDTHH:MM:SS or null",
  "notes": "brief rationale"
}
"""



MERGE_PROMPT = """
You are deduplicating low-level incident records for a Los Angeles sensing-system evaluation.

You will receive a small block of candidate low-level records that were selected by
blocking/embedding similarity. Return groups of records that refer to the SAME concrete
low-level incident.

Merge only when the records describe the same actual incident at the same real-world
place/time, possibly with slightly different names, URLs, or metadata.

Rules:
- Merge duplicate Palisades Fire records with Palisades Fire records.
- Merge duplicate Eaton Fire records with Eaton Fire records.
- Merge duplicate Hurst Fire records with Hurst Fire records.
- Do NOT merge different named fires, e.g. Palisades Fire vs Eaton Fire vs Hurst Fire.
- Do NOT merge a broad aggregate like Los Angeles wildfires with a specific low-level fire.
- Do NOT merge protests at different specific locations, even if they are both ICE protests.
- Do NOT merge records of different incident types.
- Prefer no merge when unsure.

Return exactly JSON:
{
  "merge_groups": [
    {"record_ids": ["id1", "id2"], "reason": "why these are same incident"}
  ],
  "no_merge_reasoning": "brief notes"
}
"""

def llm_decide(record_id: str, record: dict[str, Any], deterministic: tuple[str, str], articles: list[dict[str, Any]], llm: LLMClient) -> dict[str, Any]:
    payload = {
        "record_id": record_id,
        "deterministic_action": deterministic[0],
        "deterministic_reason": deterministic[1],
        "record": {
            "final_name": record.get("final_name"),
            "incident_type": record.get("incident_type"),
            "location": record.get("location"),
            "start_datetime_pacific": record.get("start_datetime_pacific"),
            "end_datetime_pacific": record.get("end_datetime_pacific"),
            "time_notes": record.get("time_notes"),
            "location_notes": record.get("location_notes"),
            "source_low_level_ids": record.get("source_low_level_ids"),
            "source_incident_ids": record.get("source_incident_ids"),
            "original_incident_names": record.get("original_incident_names"),
            "source_parent_names": record.get("source_parent_names"),
            "urls": extract_urls(record),
            "specific_fire_names_detected": sorted(specific_fire_names(record)),
        },
        "article_snippets": [
            {
                "url": a.get("url"),
                "domain": a.get("domain"),
                "title": a.get("title"),
                "published_time_raw": a.get("published_time_raw"),
                "ok": a.get("ok"),
                "text_snippet": (a.get("text_snippet") or "")[:1800],
                "error": a.get("error"),
            }
            for a in articles[:5]
        ],
    }
    return llm.complete_json(FILTER_PROMPT, payload)


def apply_decision(record: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    for src, dst in [
        ("corrected_final_name", "final_name"),
        ("corrected_incident_type", "incident_type"),
        ("corrected_location", "location"),
        ("corrected_start_datetime_pacific", "start_datetime_pacific"),
        ("corrected_end_datetime_pacific", "end_datetime_pacific"),
    ]:
        if decision.get(src):
            out[dst] = decision[src]
    out["incident_type"] = canonical_type(out)
    fires = sorted(specific_fire_names(out))
    if len(fires) == 1:
        out["final_name"] = fires[0]
        out["incident_type"] = "wildfire"
        out.setdefault("specific_fire_names", fires)
    out["iter_cleanup_notes"] = decision.get("notes")
    out["iter_cleanup_confidence"] = decision.get("confidence")
    return out


def url_score(url: str, record: dict[str, Any], article: dict[str, Any] | None = None) -> tuple[int, list[str]]:
    score = 0
    reasons = []
    domain = domain_of(url)
    text = normalize_text(url)
    if article:
        text += " " + normalize_text(article.get("title")) + " " + normalize_text(article.get("text_snippet"))

    fires = specific_fire_names(record)
    typ = canonical_type(record)
    name_terms = set(normalize_name(record.get("final_name")).split())

    if any(domain == d or domain.endswith("." + d) for d in DIRECT_DOMAINS):
        score += 3
        reasons.append("direct_or_major_domain")
    if any(domain == d or domain.endswith("." + d) for d in INDIRECT_DOMAINS):
        score -= 5
        reasons.append("indirect_domain")

    for fire in fires:
        tokens = normalize_text(fire)
        if tokens in text or all(t in text for t in tokens.split()):
            score += 8
            reasons.append(f"mentions_{fire}")

    if typ == "wildfire" and any(t in text for t in ["wildfire", "brush fire", "fire", "burn scar", "evacuation warning", "evacuation order"]):
        score += 3
        reasons.append("wildfire_terms")
    if typ == "civil protest" and any(t in text for t in ["protest", "demonstration", "rally", "march", "walkout", "strike"]):
        score += 3
        reasons.append("protest_terms")
    if typ == "urban fire" and any(t in text for t in ["structure fire", "apartment fire", "building fire", "firefighters"]):
        score += 3
        reasons.append("urban_fire_terms")

    if name_terms:
        overlap = sum(1 for t in name_terms if len(t) >= 4 and t in text)
        score += min(overlap, 4)
        if overlap:
            reasons.append(f"name_overlap_{overlap}")

    if any(term in text for term in ADJACENT_TERMS):
        score -= 8
        reasons.append("adjacent_lifestyle_or_support_terms")
    if any(term in text for term in PROCEDURAL_TERMS):
        score -= 6
        reasons.append("procedural_terms")
    if any(term in text for term in ["meghan", "prince harry", "paris hilton", "celebrity", "grammys"]):
        score -= 10
        reasons.append("celebrity_lifestyle_terms")
    if any(term in text for term in ["mayoral election", "debate", "candidate", "campaign"]):
        score -= 8
        reasons.append("political_debate_terms")

    return score, reasons


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
    ta, tb = canonical_type(a), canonical_type(b)
    if ta != tb:
        return True, "different_types"
    fa, fb = specific_fire_names(a), specific_fire_names(b)
    if fa and fb and fa.isdisjoint(fb):
        return True, f"different_named_fires:{sorted(fa)} vs {sorted(fb)}"
    if ta == "civil protest":
        la, lb = normalize_location(a.get("location")), normalize_location(b.get("location"))
        broad = {"los angeles", "la", "l a", ""}
        if la not in broad and lb not in broad:
            if difflib.SequenceMatcher(None, la, lb).ratio() < 0.55:
                return True, "different_specific_protest_locations"
    return False, ""


def merge_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    typ = canonical_type(record)
    fires = sorted(specific_fire_names(record))
    name = "|".join(fires) if fires else normalize_name(record.get("final_name"))
    loc = normalize_location(record.get("location"))
    start = str(record.get("start_datetime_pacific") or "")[:10]
    return typ, name, loc, start


def should_merge(a: dict[str, Any], b: dict[str, Any]) -> bool:
    blocked, _ = do_not_merge(a, b)
    if blocked:
        return False
    fa, fb = specific_fire_names(a), specific_fire_names(b)
    if fa and fa == fb:
        return True
    if merge_key(a) == merge_key(b) and all(merge_key(a)):
        return True
    if canonical_type(a) != canonical_type(b):
        return False

    sa, sb = str(a.get("start_datetime_pacific") or "")[:10], str(b.get("start_datetime_pacific") or "")[:10]
    if sa and sb and sa != sb:
        return False

    name_sim = difflib.SequenceMatcher(None, normalize_name(a.get("final_name")), normalize_name(b.get("final_name"))).ratio()
    loc_sim = difflib.SequenceMatcher(None, normalize_location(a.get("location")), normalize_location(b.get("location"))).ratio()
    if name_sim >= 0.92 and loc_sim >= 0.70:
        return True

    urls_a, urls_b = set(extract_urls(a)), set(extract_urls(b))
    if urls_a & urls_b and name_sim >= 0.78 and loc_sim >= 0.60:
        return True

    return False


def build_merge_blocks(records: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    blocks = collections.defaultdict(list)
    for rid, rec in records.items():
        typ = canonical_type(rec)
        start = str(rec.get("start_datetime_pacific") or "")[:10]
        fires = specific_fire_names(rec)
        if fires:
            for fire in fires:
                blocks[f"fire:{fire}"].append(rid)
                if start:
                    blocks[f"fire-date:{fire}:{start}"].append(rid)
        key = merge_key(rec)
        blocks["exact:" + "|".join(key)].append(rid)
        urls = extract_urls(rec, max_urls=3)
        for url in urls:
            blocks[f"url:{typ}:{start}:{url}"].append(rid)
        nn = normalize_name(rec.get("final_name"))
        loc = normalize_location(rec.get("location"))
        if nn and start:
            blocks[f"name-date:{typ}:{start}:{'-'.join(nn.split()[:4])}"].append(rid)
        if loc and start:
            blocks[f"loc-date:{typ}:{start}:{'-'.join(loc.split()[:3])}"].append(rid)
    return blocks



def merge_candidate_text(record_id: str, record: dict[str, Any]) -> str:
    """Compact text used for embedding/blocking duplicate merge candidates."""
    parts = [
        f"id={record_id}",
        f"name={record.get('final_name')}",
        f"type={canonical_type(record)}",
        f"location={record.get('location')}",
        f"start={record.get('start_datetime_pacific')}",
        f"end={record.get('end_datetime_pacific')}",
        f"fires={sorted(specific_fire_names(record))}",
        "original_names=" + "; ".join(str(x) for x in (record.get("original_incident_names") or [])[:8]),
        "parents=" + "; ".join(str(x) for x in (record.get("source_parent_names") or [])[:8]),
    ]
    urls = extract_urls(record, max_urls=4)
    if urls:
        parts.append("domains=" + "; ".join(domain_of(u) for u in urls))
    return " | ".join(parts)


def summarize_merge_candidate(record_id: str, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "final_name": record.get("final_name"),
        "incident_type": canonical_type(record),
        "raw_incident_type": record.get("incident_type"),
        "location": record.get("location"),
        "start_datetime_pacific": record.get("start_datetime_pacific"),
        "end_datetime_pacific": record.get("end_datetime_pacific"),
        "specific_fire_names": sorted(specific_fire_names(record)),
        "original_incident_names": (record.get("original_incident_names") or [])[:10],
        "source_parent_names": (record.get("source_parent_names") or [])[:10],
        "urls": extract_urls(record, max_urls=6),
        "source_incident_ids": (record.get("source_incident_ids") or [])[:10],
    }


def date_bucket_for_merge(record: dict[str, Any]) -> str:
    """Bucket dates coarsely for semantic merge candidate generation."""
    d = str(record.get("start_datetime_pacific") or "")
    if len(d) >= 7:
        return d[:7]  # YYYY-MM
    return "unknown"


def lexical_merge_candidate_blocks(records: dict[str, dict[str, Any]], max_block_size: int) -> list[list[str]]:
    """Fallback candidate blocks when sentence-transformers is unavailable."""
    raw_blocks = build_merge_blocks(records)
    out: list[list[str]] = []
    for ids in raw_blocks.values():
        ids = sorted(set(ids))
        if 2 <= len(ids) <= max_block_size:
            out.append(ids)
    return out


def semantic_merge_candidate_blocks(
    records: dict[str, dict[str, Any]],
    model_name: str,
    similarity_threshold: float,
    top_k: int,
    max_bucket_size: int,
    max_block_size: int,
) -> list[list[str]]:
    """Generate plausible duplicate blocks using sentence-transformer embeddings.

    This avoids all-pairs LLM calls. We embed compact record summaries and only ask
    the LLM to review connected components formed by high-similarity nearest neighbors
    inside type/month buckets. If sentence-transformers/numpy is unavailable, callers
    should fall back to lexical blocks.
    """
    try:
        import numpy as np  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        print(f"WARNING: sentence-transformers unavailable; using lexical merge blocks only: {e}", file=sys.stderr)
        return []

    buckets: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for rid, rec in records.items():
        typ = canonical_type(rec)
        # Named fires are already covered by deterministic blocks, but include them in
        # a type-only bucket so metadata variants can still be reviewed.
        if specific_fire_names(rec):
            buckets[(typ, "named-fire")].append(rid)
        else:
            buckets[(typ, date_bucket_for_merge(rec))].append(rid)

    model = SentenceTransformer(model_name)
    all_blocks: list[list[str]] = []

    for (_typ, _bucket), ids in buckets.items():
        ids = sorted(set(ids))
        if len(ids) < 2:
            continue
        # Avoid quadratic blowups in giant generic buckets. Those records should mostly
        # have been filtered as broad/generic anyway.
        if len(ids) > max_bucket_size:
            # Split by first normalized name token to keep work bounded.
            sub: dict[str, list[str]] = collections.defaultdict(list)
            for rid in ids:
                name_tokens = normalize_name(records[rid].get("final_name")).split()
                key = name_tokens[0] if name_tokens else "unknown"
                sub[key].append(rid)
            sublists = [v for v in sub.values() if 2 <= len(v) <= max_bucket_size]
        else:
            sublists = [ids]

        for sub_ids in sublists:
            texts = [merge_candidate_text(rid, records[rid]) for rid in sub_ids]
            emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            emb = np.asarray(emb, dtype="float32")
            sim = emb @ emb.T

            uf = UnionFind(sub_ids)
            n = len(sub_ids)
            for i in range(n):
                # Top-k nearest neighbors above threshold, excluding self.
                row = sim[i].copy()
                row[i] = -1.0
                if top_k < n:
                    cand_idx = np.argpartition(row, -top_k)[-top_k:]
                else:
                    cand_idx = np.arange(n)
                for j in cand_idx:
                    if j <= i:
                        continue
                    if float(row[j]) < similarity_threshold:
                        continue
                    a, b = records[sub_ids[i]], records[sub_ids[j]]
                    blocked, _ = do_not_merge(a, b)
                    if not blocked:
                        uf.union(sub_ids[i], sub_ids[j])

            for comp in uf.groups().values():
                comp = sorted(set(comp))
                if 2 <= len(comp) <= max_block_size:
                    all_blocks.append(comp)

    return all_blocks


def build_llm_merge_review_blocks(records: dict[str, dict[str, Any]], args: argparse.Namespace) -> list[list[str]]:
    blocks: list[list[str]] = []
    blocks.extend(lexical_merge_candidate_blocks(records, args.max_merge_block_size))
    if not args.no_embedding_merge:
        blocks.extend(
            semantic_merge_candidate_blocks(
                records=records,
                model_name=args.embedding_model,
                similarity_threshold=args.embedding_similarity_threshold,
                top_k=args.embedding_top_k,
                max_bucket_size=args.max_embedding_bucket_size,
                max_block_size=args.max_merge_block_size,
            )
        )

    # Deduplicate exact block membership and avoid blocks already fully disconnected
    # by hard guards.
    deduped: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for block in blocks:
        block = sorted(set(block))
        if len(block) < 2 or len(block) > args.max_merge_block_size:
            continue
        # Keep a block only if at least one pair is not hard-blocked.
        plausible = False
        for i in range(len(block)):
            for j in range(i + 1, len(block)):
                blocked, _ = do_not_merge(records[block[i]], records[block[j]])
                if not blocked:
                    plausible = True
                    break
            if plausible:
                break
        if not plausible:
            continue
        key = tuple(block)
        if key not in seen:
            seen.add(key)
            deduped.append(block)
    return deduped


def merge_block_cache_key(block: list[str], records: dict[str, dict[str, Any]]) -> str:
    payload = [
        {
            "id": rid,
            "name": records[rid].get("final_name"),
            "type": canonical_type(records[rid]),
            "loc": records[rid].get("location"),
            "start": records[rid].get("start_datetime_pacific"),
            "fires": sorted(specific_fire_names(records[rid])),
        }
        for rid in sorted(block)
    ]
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    import hashlib
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def apply_llm_merge_review(
    uf: UnionFind,
    records: dict[str, dict[str, Any]],
    llm: LLMClient,
    args: argparse.Namespace,
) -> tuple[int, int]:
    """Ask the LLM to review only plausible merge blocks and union approved groups.

    Returns (blocks_reviewed, unions_added).
    """
    if args.no_llm_merge or not llm.enabled():
        return 0, 0

    merge_decisions = {} if args.force else load_json(args.merge_decisions_cache, default={})
    merge_audit = {} if args.force else load_json(args.llm_merge_audit_output, default={})
    blocks = build_llm_merge_review_blocks(records, args)

    iterator = blocks
    if tqdm and not args.no_progress:
        iterator = tqdm(blocks, desc="LLM merge review", unit="block")

    reviewed = 0
    unions = 0
    for block in iterator:
        key = merge_block_cache_key(block, records)
        if key in merge_decisions and not args.force:
            decision = merge_decisions[key]
        else:
            payload = {
                "instructions": "Group only records that are the same concrete low-level incident.",
                "candidate_records": [summarize_merge_candidate(rid, records[rid]) for rid in block],
                "hard_rules": {
                    "do_not_merge_different_named_fires": True,
                    "do_not_merge_different_incident_types": True,
                    "do_not_merge_protests_at_different_specific_locations": True,
                    "prefer_no_merge_when_unsure": True,
                },
            }
            try:
                decision = llm.complete_json(MERGE_PROMPT, payload)
            except Exception as e:
                print(f"WARNING: LLM merge review failed for block {key}: {e}", file=sys.stderr)
                decision = {"merge_groups": [], "no_merge_reasoning": f"LLM failed: {e}"}
            merge_decisions[key] = decision
            atomic_save_json(args.merge_decisions_cache, merge_decisions)

        reviewed += 1
        audit_groups = []
        for group in decision.get("merge_groups", []) or []:
            ids = [rid for rid in group.get("record_ids", []) if rid in records and rid in block]
            ids = sorted(set(ids))
            if len(ids) < 2:
                continue
            # Enforce hard rules even if the LLM suggests an unsafe merge.
            safe_ids = [ids[0]]
            rejected_pairs = []
            for candidate in ids[1:]:
                safe = True
                for existing in safe_ids:
                    blocked, why = do_not_merge(records[existing], records[candidate])
                    if blocked:
                        safe = False
                        rejected_pairs.append({"a": existing, "b": candidate, "reason": why})
                        break
                if safe:
                    safe_ids.append(candidate)
            if len(safe_ids) >= 2:
                first = safe_ids[0]
                for other in safe_ids[1:]:
                    if uf.find(first) != uf.find(other):
                        unions += 1
                    uf.union(first, other)
            audit_groups.append({
                "requested_ids": ids,
                "applied_ids": safe_ids,
                "reason": group.get("reason"),
                "rejected_pairs": rejected_pairs,
            })

        merge_audit[key] = {
            "block_ids": block,
            "decision": decision,
            "applied_groups": audit_groups,
        }
        atomic_save_json(args.llm_merge_audit_output, merge_audit)

    return reviewed, unions


def merge_records(clean_id: str, old_ids: list[str], records: dict[str, dict[str, Any]], article_cache: dict[str, Any], disable_known_fire_corrections: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    group = [records[x] for x in old_ids]
    fires = sorted(set().union(*(specific_fire_names(r) for r in group)))

    if len(fires) == 1:
        final_name = fires[0]
        incident_type = "wildfire"
    else:
        names = [str(r.get("final_name")) for r in group if r.get("final_name")]
        names = sorted(names, key=lambda n: ("|" in n, len(n)))
        final_name = names[0] if names else clean_id
        incident_type = collections.Counter(canonical_type(r) for r in group).most_common(1)[0][0]

    locs = [str(r.get("location")) for r in group if r.get("location")]
    locs = sorted(locs, key=lambda x: (normalize_location(x) in {"los angeles", "la", "l a", ""}, -len(x)))
    location = locs[0] if locs else None

    starts = [r.get("start_datetime_pacific") for r in group]
    ends = [r.get("end_datetime_pacific") for r in group]
    if len(fires) == 1 and not disable_known_fire_corrections:
        location = KNOWN_FIRE_LOCATION.get(fires[0], location)
        if fires[0] in KNOWN_FIRE_START:
            starts.append(KNOWN_FIRE_START[fires[0]])

    all_urls = dedupe_list(u for r in group for u in extract_urls(r))
    source_low_ids = dedupe_list(x for r in group for x in ([*r.get("source_low_level_ids", [])] if isinstance(r.get("source_low_level_ids"), list) else []) + [r.get("low_level_incident_id")] if False)
    # Simpler because the above expression is unreadable with mixed source formats:
    source_low_ids = []
    source_incident_ids = []
    original_names = []
    source_parents = []
    source_refs = []
    notes = []
    for old_id, r in zip(old_ids, group):
        source_low_ids.append(old_id)
        if isinstance(r.get("source_low_level_ids"), list):
            source_low_ids.extend(r["source_low_level_ids"])
        if isinstance(r.get("source_incident_ids"), list):
            source_incident_ids.extend(r["source_incident_ids"])
        if isinstance(r.get("original_incident_names"), list):
            original_names.extend(r["original_incident_names"])
        if r.get("final_name"):
            original_names.append(r["final_name"])
        if isinstance(r.get("source_parent_names"), list):
            source_parents.extend(r["source_parent_names"])
        if isinstance(r.get("source_event_refs"), list):
            source_refs.extend(r["source_event_refs"])
        if r.get("iter_cleanup_notes"):
            notes.append(r["iter_cleanup_notes"])

    temp_record = {
        "final_name": final_name,
        "incident_type": incident_type,
        "location": location,
        "start_datetime_pacific": earliest(starts),
        "end_datetime_pacific": latest(ends),
        "specific_fire_names": fires,
    }

    scored_urls = []
    for url in all_urls:
        article = article_cache.get(url)
        score, reasons = url_score(url, temp_record, article)
        scored_urls.append({"url": url, "score": score, "reasons": reasons, "domain": domain_of(url)})

    scored_urls.sort(key=lambda x: x["score"], reverse=True)
    # Keep direct-ish URLs first. If none score positive, keep best two for traceability.
    positive = [x["url"] for x in scored_urls if x["score"] > 0]
    urls_used = positive[:8] if positive else [x["url"] for x in scored_urls[:2]]

    record = {
        "low_level_incident_id": clean_id,
        "final_name": final_name,
        "incident_type": incident_type,
        "location": location,
        "start_datetime_pacific": earliest(starts),
        "end_datetime_pacific": latest(ends),
        "source_low_level_ids": dedupe_list(source_low_ids),
        "source_incident_ids": dedupe_list(source_incident_ids),
        "original_incident_names": dedupe_list(original_names),
        "source_parent_names": dedupe_list(source_parents),
        "urls_used": urls_used,
        "all_candidate_urls": all_urls,
        "source_event_refs": dedupe_list(source_refs),
        "specific_fire_names": fires,
        "evidence_record_count": len(group),
        "iter_cleanup_notes": " | ".join(dedupe_list(notes)),
    }
    return record, scored_urls


def build_duplicate_report(cleaned: dict[str, dict[str, Any]]) -> dict[str, Any]:
    buckets = collections.defaultdict(list)
    for cid, rec in cleaned.items():
        buckets["|".join(merge_key(rec))].append(cid)
    dupes = {k: v for k, v in buckets.items() if len(v) > 1}
    return {"duplicate_key_count": len(dupes), "duplicate_keys": dupes}


def build_top_level(cleaned: dict[str, dict[str, Any]], top_candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    top = {}
    idx = 0

    def add(name: str, children: list[str], reason: str) -> None:
        nonlocal idx
        children = dedupe_list(children)
        if len(children) < 2:
            return
        tid = f"top_{idx:06d}"
        idx += 1
        recs = [cleaned[c] for c in children]
        top[tid] = {
            "top_level_incident_id": tid,
            "top_level_incident_name": name,
            "low_level_incident_ids": children,
            "start_datetime_pacific": earliest(r.get("start_datetime_pacific") for r in recs),
            "end_datetime_pacific": latest(r.get("end_datetime_pacific") for r in recs),
            "incident_types": dedupe_list(r.get("incident_type") for r in recs),
            "reason": reason,
        }

    jan_la_fires = []
    for cid, rec in cleaned.items():
        if canonical_type(rec) == "wildfire" and specific_fire_names(rec):
            d = parse_date_prefix(rec.get("start_datetime_pacific"))
            if d and date(2025, 1, 1) <= d <= date(2025, 1, 31):
                jan_la_fires.append(cid)
    add("Los Angeles wildfires", jan_la_fires, "named_january_2025_la_wildfires")

    ice = []
    for cid, rec in cleaned.items():
        if canonical_type(rec) == "civil protest":
            blob = normalize_text(record_blob(rec))
            if any(t in blob for t in ["ice", "immigration", "deportation", "federal agents", "national guard", "raids"]):
                ice.append(cid)
    add("Los Angeles ICE protests", ice, "ice_or_immigration_protest_keywords")

    existing = {tuple(sorted(t["low_level_incident_ids"])) for t in top.values()}

    for aid, agg in top_candidates.items():
        name = str(agg.get("final_name") or aid)
        typ = canonical_type(agg)
        agg_blob = normalize_text(record_blob(agg))
        agg_fires = specific_fire_names(agg)
        children = []
        for cid, rec in cleaned.items():
            if canonical_type(rec) != typ:
                continue
            if typ == "wildfire":
                rec_fires = specific_fire_names(rec)
                if agg_fires:
                    if rec_fires and not rec_fires.isdisjoint(agg_fires):
                        children.append(cid)
                elif any(t in agg_blob for t in ["wildfire", "wildfires", "fires", "la fires", "los angeles fires"]):
                    if rec_fires:
                        children.append(cid)
            elif typ == "civil protest":
                rec_blob = normalize_text(record_blob(rec))
                if any(t in agg_blob for t in ["ice", "immigration"]):
                    if any(t in rec_blob for t in ["ice", "immigration", "deportation", "federal agents", "national guard", "raids"]):
                        children.append(cid)
        fp = tuple(sorted(dedupe_list(children)))
        if len(fp) >= 2 and fp not in existing:
            add(name, list(fp), f"from_top_candidate:{aid}")
            existing.add(fp)

    return top


def load_reference_records(paths: list[Path]) -> dict[str, dict[str, Any]]:
    refs = {}
    for p in paths:
        if not p.exists():
            print(f"WARNING: reference file not found: {p}", file=sys.stderr)
            continue
        data = load_json(p)
        if not isinstance(data, dict):
            continue
        for rid, rec in data.items():
            if isinstance(rec, dict):
                refs[f"{p.name}:{rid}"] = rec
    return refs


def enrich_with_reference_urls(records: dict[str, dict[str, Any]], refs: dict[str, dict[str, Any]]) -> None:
    """Add URLs from reference records that clearly match the same concrete incident."""
    if not refs:
        return
    ref_by_fire = collections.defaultdict(list)
    ref_by_key = collections.defaultdict(list)
    for rid, rec in refs.items():
        for fire in specific_fire_names(rec):
            ref_by_fire[fire].append((rid, rec))
        ref_by_key[(canonical_type(rec), normalize_name(rec.get("final_name")), normalize_location(rec.get("location")))].append((rid, rec))

    for rid, rec in records.items():
        candidates = []
        fires = specific_fire_names(rec)
        for fire in fires:
            candidates.extend(ref_by_fire.get(fire, []))
        if not fires:
            candidates.extend(ref_by_key.get((canonical_type(rec), normalize_name(rec.get("final_name")), normalize_location(rec.get("location"))), []))

        extra_urls = []
        extra_refs = []
        extra_original_names = []
        for ref_id, ref in candidates:
            blocked, _ = do_not_merge(rec, ref)
            if blocked:
                continue
            extra_urls.extend(extract_urls(ref))
            extra_refs.extend(ref.get("source_event_refs") or [])
            if ref.get("final_name"):
                extra_original_names.append(ref["final_name"])
            extra_original_names.extend(ref.get("original_incident_names") or [])

        if extra_urls:
            rec["all_candidate_urls"] = dedupe_list(extract_urls(rec) + extra_urls)
            rec["urls_used"] = dedupe_list(extract_urls(rec) + extra_urls)
        if extra_refs:
            rec["source_event_refs"] = dedupe_list((rec.get("source_event_refs") or []) + extra_refs)
        if extra_original_names:
            rec["original_incident_names"] = dedupe_list((rec.get("original_incident_names") or []) + extra_original_names)


def run(args: argparse.Namespace) -> None:
    input_path = args.input
    if input_path == DEFAULT_INPUT and not input_path.exists() and FALLBACK_INPUT.exists():
        input_path = FALLBACK_INPUT

    data = load_json(input_path)
    if not isinstance(data, dict):
        raise ValueError("Input low-level file must be a JSON object/dict.")

    refs = load_reference_records(args.reference)
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)

    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    llm = LLMClient(api_key, args.model)
    use_llm = (not args.no_llm) and llm.enabled()
    llm_mode = "off" if args.no_llm else args.llm_mode

    decisions = {} if args.force else load_json(args.decisions_cache, default={})
    web_cache = {} if args.force_web_cache else load_json(args.web_cache, default={})

    # Work copy.
    raw_records = {rid: dict(rec) for rid, rec in data.items() if isinstance(rec, dict)}
    if refs:
        enrich_with_reference_urls(raw_records, refs)

    kept = {}
    rejects = {}
    top_candidates = {}

    items = list(raw_records.items())
    iterator = items
    if tqdm and not args.no_progress:
        iterator = tqdm(items, desc="Filtering records", unit="record")

    for rid, rec in iterator:
        det = deterministic_action(rec, start, end)
        action, reason = det

        should_llm = False
        if use_llm:
            if llm_mode == "all":
                should_llm = True
            elif llm_mode == "suspicious" and action in {"suspicious", "top_candidate", "reject"}:
                should_llm = True

        if rid in decisions and not args.force:
            decision = decisions[rid]
        elif should_llm:
            articles = fetch_articles(rec, web_cache, fetch_web=not args.no_fetch_web, timeout=args.fetch_timeout)
            atomic_save_json(args.web_cache, web_cache)
            try:
                decision = llm_decide(rid, rec, det, articles, llm)
            except Exception as e:
                print(f"WARNING: LLM failed for {rid}: {e}", file=sys.stderr)
                decision = {
                    "action": "top_candidate" if action == "top_candidate" else ("reject" if action == "reject" else "keep"),
                    "reject_reason": reason if action == "reject" else None,
                    "confidence": 0.4,
                    "notes": f"LLM failed; deterministic fallback: {action}:{reason}",
                }
            decisions[rid] = decision
            atomic_save_json(args.decisions_cache, decisions)
        else:
            decision = {
                "action": "top_candidate" if action == "top_candidate" else ("reject" if action == "reject" else "keep"),
                "reject_reason": reason if action == "reject" else None,
                "confidence": 0.6,
                "notes": f"deterministic:{action}:{reason}",
            }
            decisions[rid] = decision
            atomic_save_json(args.decisions_cache, decisions)

        final_action = decision.get("action")
        if final_action == "keep":
            kept[rid] = apply_decision(rec, decision)
        elif final_action == "top_candidate":
            top_candidates[rid] = apply_decision(rec, decision)
        else:
            rejects[rid] = {
                "source_record_id": rid,
                "final_name": rec.get("final_name"),
                "incident_type": rec.get("incident_type"),
                "location": rec.get("location"),
                "start_datetime_pacific": rec.get("start_datetime_pacific"),
                "urls_used": extract_urls(rec),
                "reject_reason": decision.get("reject_reason") or reason,
                "notes": decision.get("notes"),
            }

        atomic_save_json(args.rejects_output, rejects)
        atomic_save_json(args.top_candidates_output, top_candidates)

    # Merge.
    # Stage 1: deterministic safety-preserving unions for exact/name/location/URL matches.
    uf = UnionFind(kept.keys())
    blocks = build_merge_blocks(kept)
    deterministic_unions = 0
    for ids in blocks.values():
        ids = sorted(set(ids))
        if len(ids) < 2 or len(ids) > args.max_merge_block_size:
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if should_merge(kept[ids[i]], kept[ids[j]]):
                    if uf.find(ids[i]) != uf.find(ids[j]):
                        deterministic_unions += 1
                    uf.union(ids[i], ids[j])

    # Stage 2: LLM merge review on efficient candidate blocks. Candidate blocks come
    # from lexical blocking plus optional sentence-transformer nearest neighbors; this
    # avoids O(n^2) LLM pairwise comparisons.
    llm_merge_blocks_reviewed, llm_merge_unions = apply_llm_merge_review(uf, kept, llm, args)

    groups = [sorted(g) for g in uf.groups().values()]
    groups.sort(key=lambda g: (
        str(kept[g[0]].get("start_datetime_pacific") or ""),
        sorted(specific_fire_names(kept[g[0]]))[0] if specific_fire_names(kept[g[0]]) else normalize_name(kept[g[0]].get("final_name")),
        g[0],
    ))

    cleaned = {}
    merge_map = {}
    url_audit = {}

    iterator2 = groups
    if tqdm and not args.no_progress:
        iterator2 = tqdm(groups, desc="Writing merged records", unit="group")

    for idx, old_ids in enumerate(iterator2):
        cid = f"clean_low_{idx:06d}"
        rec, scores = merge_records(cid, old_ids, kept, web_cache, args.disable_known_fire_corrections)
        cleaned[cid] = rec
        url_audit[cid] = scores
        for old_id in old_ids:
            merge_map[old_id] = cid

        atomic_save_json(args.low_output, cleaned)
        atomic_save_json(args.merge_map_output, merge_map)
        atomic_save_json(args.url_audit_output, url_audit)

    duplicate_report = build_duplicate_report(cleaned)
    atomic_save_json(args.duplicate_report_output, duplicate_report)

    top = build_top_level(cleaned, top_candidates)
    atomic_save_json(args.top_output, top)

    summary = {
        "input_path": str(input_path),
        "reference_paths": [str(p) for p in args.reference],
        "input_count": len(data),
        "kept_premerge_count": len(kept),
        "rejected_count": len(rejects),
        "top_candidate_count": len(top_candidates),
        "cleaned_low_level_count": len(cleaned),
        "top_level_count": len(top),
        "duplicate_key_count_after_cleaning": duplicate_report["duplicate_key_count"],
        "llm_enabled": use_llm,
        "llm_mode": llm_mode,
        "deterministic_merge_unions": deterministic_unions,
        "llm_merge_enabled": (not args.no_llm_merge) and llm.enabled(),
        "llm_merge_blocks_reviewed": llm_merge_blocks_reviewed,
        "llm_merge_unions_added": llm_merge_unions,
        "embedding_merge_enabled": not args.no_embedding_merge,
        "outputs": {
            "low_level_cleaned": str(args.low_output),
            "top_level": str(args.top_output),
            "rejects": str(args.rejects_output),
            "merge_map": str(args.merge_map_output),
            "top_candidates": str(args.top_candidates_output),
            "url_audit": str(args.url_audit_output),
            "duplicate_report": str(args.duplicate_report_output),
            "merge_decisions_cache": str(args.merge_decisions_cache),
            "llm_merge_audit": str(args.llm_merge_audit_output),
        },
    }
    atomic_save_json(args.summary_output, summary)
    print(json.dumps(summary, indent=2))
    if duplicate_report["duplicate_key_count"]:
        print("WARNING: duplicate-like keys remain; inspect duplicate report.", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Iteratively clean low-level incident JSON and rebuild top-level JSON.")

    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input low-level JSON. Defaults to low_level_cleaned.json, falling back to low_level.json if needed.")
    p.add_argument("--reference", type=Path, action="append", default=[], help="Optional reference low-level JSON used only to recover better URLs/evidence before merging. Can be repeated.")

    p.add_argument("--low-output", type=Path, default=DEFAULT_LOW_OUTPUT)
    p.add_argument("--top-output", type=Path, default=DEFAULT_TOP_OUTPUT)
    p.add_argument("--rejects-output", type=Path, default=DEFAULT_REJECTS_OUTPUT)
    p.add_argument("--merge-map-output", type=Path, default=DEFAULT_MERGE_MAP_OUTPUT)
    p.add_argument("--top-candidates-output", type=Path, default=DEFAULT_TOP_CANDIDATES_OUTPUT)
    p.add_argument("--url-audit-output", type=Path, default=DEFAULT_URL_AUDIT_OUTPUT)
    p.add_argument("--duplicate-report-output", type=Path, default=DEFAULT_DUPLICATE_REPORT_OUTPUT)
    p.add_argument("--decisions-cache", type=Path, default=DEFAULT_DECISIONS_CACHE)
    p.add_argument("--web-cache", type=Path, default=DEFAULT_WEB_CACHE)
    p.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)

    p.add_argument("--start-date", default=DEFAULT_START_DATE)
    p.add_argument("--end-date", default=DEFAULT_END_DATE)

    p.add_argument("--openai-api-key", default=None)
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--llm-mode", choices=["all", "suspicious", "off"], default="all", help="Default is all when an API key is available.")
    p.add_argument("--no-fetch-web", action="store_true")
    p.add_argument("--fetch-timeout", type=int, default=8)
    p.add_argument("--force", action="store_true", help="Ignore cached LLM decisions.")
    p.add_argument("--force-web-cache", action="store_true", help="Ignore cached article snippets.")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--max-merge-block-size", type=int, default=150)
    p.add_argument("--no-llm-merge", action="store_true", help="Disable LLM review of plausible duplicate merge blocks.")
    p.add_argument("--merge-decisions-cache", type=Path, default=DEFAULT_MERGE_DECISIONS_CACHE)
    p.add_argument("--llm-merge-audit-output", type=Path, default=DEFAULT_LLM_MERGE_AUDIT_OUTPUT)
    p.add_argument("--no-embedding-merge", action="store_true", help="Disable sentence-transformer candidate generation for LLM merge review.")
    p.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="SentenceTransformer model for semantic merge candidate generation.")
    p.add_argument("--embedding-similarity-threshold", type=float, default=0.78)
    p.add_argument("--embedding-top-k", type=int, default=8)
    p.add_argument("--max-embedding-bucket-size", type=int, default=600)
    p.add_argument("--disable-known-fire-corrections", action="store_true", help="Disable curated corrections for known LA named wildfires.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.llm_mode == "off":
        args.no_llm = True
    run(args)


if __name__ == "__main__":
    main()
