import json
import calendar
from pathlib import Path
from datetime import datetime, date, timezone
from collections import defaultdict, Counter
import random
from tqdm import tqdm

from evaluation.geo_manager import GeoManager

from math import radians, sin, cos, sqrt, atan2

def count_event_types(filtered_events, ignore_no_event=True):
    return Counter(
        event["event_type"]
        for event in filtered_events
        if not is_na(event.get("event_type"))
        and (
            not ignore_no_event
            or normalize_text(event.get("event_type")) != "no event"
        )
    )

def tokenize_name(name):
    return re.findall(r"[A-Za-z0-9]+", str(name))

def is_capitalized_token(token):
    """
    True for tokens like:
      Palisades
      Eaton
      LA
      UCLA

    False for:
      high
      wind
      warning
      fire
    """
    if not token:
        return False

    # Acronyms / all-caps abbreviations
    if len(token) >= 2 and token.isupper():
        return True

    # Regular proper-case word
    return token[0].isupper() and not token.isupper()

def classify_parent_name_by_capitalization(
    name,
    manual_distinctive_names=None,
    manual_generic_names=None,
):
    """
    Classifies a parent name using capitalization only.

    Returns:
      - "distinctive" if the name contains capitalized/proper-name tokens
      - "generic" otherwise

    Manual overrides are optional.
    """
    manual_distinctive_names = {
        normalize_text(x)
        for x in (manual_distinctive_names or set())
    }

    manual_generic_names = {
        normalize_text(x)
        for x in (manual_generic_names or set())
    }

    name_norm = normalize_text(name)

    if name_norm in manual_distinctive_names:
        return "distinctive"

    if name_norm in manual_generic_names:
        return "generic"

    tokens = tokenize_name(name)

    capitalized_tokens = [
        token
        for token in tokens
        if is_capitalized_token(token)
    ]

    if capitalized_tokens:
        return "distinctive"

    return "generic"


def split_part_of_by_capitalization(
    part_of_by_parent_name,
    manual_distinctive_names=None,
    manual_generic_names=None,
):
    distinctive = {}
    generic = {}

    for parent_name, entry in part_of_by_parent_name.items():
        label = classify_parent_name_by_capitalization(
            parent_name,
            manual_distinctive_names=manual_distinctive_names,
            manual_generic_names=manual_generic_names,
        )

        entry_with_classification = {
            **entry,
            "name_classification": label,
        }

        if label == "distinctive":
            distinctive[parent_name] = entry_with_classification
        else:
            generic[parent_name] = entry_with_classification

    return distinctive, generic


def normalize_text(value):
    return str(value).strip().lower()


def is_na(value):
    return value is None or normalize_text(value) in {"n/a", "na", "none", ""}


def parse_event_datetime(event):
    """
    Parse event start time into a datetime.

    Assumes your filtered events have either:
      - start_time_str
      - start_time
      - parsed_start_date

    Handles:
      - YYYY-MM-DD
      - ISO datetime like 2025-05-15T15:15:00-07:00
    """
    candidates = [
        event.get("start_time_str"),
        event.get("start_time"),
        event.get("parsed_start_date"),
    ]

    for value in candidates:
        if is_na(value):
            continue

        value = str(value).strip()

        # YYYY-MM-DD
        try:
            return datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            pass

        # ISO datetime
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))

            # Convert timezone-aware datetimes to naive UTC
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)

            return dt
        except ValueError:
            pass

    return None


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Great-circle distance between two lat/lon points.
    """
    radius_km = 6371.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return radius_km * c


def geometry_centroid_latlon(geom):
    """
    Returns centroid as (lat, lon) from a Shapely geometry in EPSG:4326.
    """
    centroid = geom.centroid
    return centroid.y, centroid.x

class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1

    def groups(self):
        grouped = defaultdict(list)

        for item in self.parent:
            grouped[self.find(item)].append(item)

        return list(grouped.values())


def print_top_event_types(filtered_events, top_n=15, ignore_no_event=True):
    event_type_counts = count_event_types(
        filtered_events,
        ignore_no_event=ignore_no_event,
    )

    if ignore_no_event:
        print(f"Top {top_n} event types after pruning, ignoring 'no event':")
    else:
        print(f"Top {top_n} event types after pruning:")

    for event_type, count in event_type_counts.most_common(top_n):
        print(f"{event_type}: {count}")

def get_unique_locations(filtered_events, ignore_no_event=True):
    locations = set()

    for event in filtered_events:
        if ignore_no_event and normalize_text(event.get("event_type")) == "no event":
            continue

        # Prefer explicit location_name if present, otherwise fall back to location
        location = event.get("location_name") or event.get("location")

        if not is_na(location):
            locations.add(location.strip())

    return sorted(locations)


def normalize_text(value):
    return str(value).strip().lower()


def is_na(value):
    return value is None or normalize_text(value) in {"n/a", "na", "none", ""}


def parse_start_time(start_time):
    """
    Accepts:
      - YYYY-MM-DD
      - ISO datetime like 2025-05-15T15:15:00-07:00

    Returns a date object, or None if unparseable.
    """
    if is_na(start_time):
        return None

    start_time = str(start_time).strip()

    # Format: 2024-12-26
    try:
        return datetime.strptime(start_time, "%Y-%m-%d").date()
    except ValueError:
        pass

    # Format: 2025-05-15T15:15:00-07:00
    try:
        # Handles trailing Z if it appears
        cleaned = start_time.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned).date()
    except ValueError:
        return None


def parse_article_date(article_date):
    """
    Parses article date like:
      20250103070000

    Returns a date object, or None if unparseable.
    """
    if is_na(article_date):
        return None

    try:
        return datetime.strptime(str(article_date), "%Y%m%d%H%M%S").date()
    except ValueError:
        return None


def add_months(d, months):
    """
    Add/subtract calendar months while clamping invalid days.
    Example: Jan 31 + 1 month -> Feb 28 or Feb 29.
    """
    month_index = d.month - 1 + months
    year = d.year + month_index // 12
    month = month_index % 12 + 1

    last_day = calendar.monthrange(year, month)[1]
    day = min(d.day, last_day)

    return date(year, month, day)


def within_three_months(article_date, start_date):
    lower_bound = add_months(article_date, -3)
    upper_bound = add_months(article_date, 3)
    return lower_bound <= start_date <= upper_bound


def make_error_entry(
    *,
    file_path,
    article_index,
    event_index,
    article,
    event,
    reason=None,
):
    return {
        "file": str(file_path),
        "article_index": article_index,
        "event_index": event_index,
        "link": article.get("link"),
        "article_date": article.get("date"),
        "event_type": event.get("event_type"),
        "location": event.get("location"),
        "start_time": event.get("start_time"),
        "reason": reason,
    }


def collect_filtered_events(news_folder="news_by_date"):
    news_folder = Path(news_folder)

    filtered_events = []

    errors = {
        "invalid_event_type_or_location": [],
        "invalid_start_time": [],
        "invalid_article_date": [],
        "start_time_too_far_from_article_date": [],
        "malformed_article_or_events": [],
        "file_or_json_error": [],
    }

    for file_path in news_folder.iterdir():
        if not file_path.is_file():
            continue

        if "_data" not in file_path.name or file_path.suffix != ".json":
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                articles = json.load(f)
        except Exception as e:
            errors["file_or_json_error"].append({
                "file": str(file_path),
                "reason": repr(e),
            })
            continue

        if not isinstance(articles, list):
            errors["file_or_json_error"].append({
                "file": str(file_path),
                "reason": "Top-level JSON object is not a list",
            })
            continue

        for article_index, article in enumerate(articles):
            if not isinstance(article, dict):
                errors["malformed_article_or_events"].append({
                    "file": str(file_path),
                    "article_index": article_index,
                    "reason": "Article is not a dictionary",
                })
                continue

            article_date = parse_article_date(article.get("date"))
            events = article.get("events")

            if not isinstance(events, list):
                errors["malformed_article_or_events"].append({
                    "file": str(file_path),
                    "article_index": article_index,
                    "link": article.get("link"),
                    "article_date": article.get("date"),
                    "reason": "Missing or non-list events field",
                })
                continue

            for event_index, event in enumerate(events):
                if not isinstance(event, dict):
                    errors["malformed_article_or_events"].append({
                        "file": str(file_path),
                        "article_index": article_index,
                        "event_index": event_index,
                        "link": article.get("link"),
                        "article_date": article.get("date"),
                        "reason": "Event is not a dictionary",
                    })
                    continue

                event_type = event.get("event_type")
                location = event.get("location")

                # Ignore if event_type or location is N/A
                if is_na(event_type) or is_na(location):
                    errors["invalid_event_type_or_location"].append(
                        make_error_entry(
                            file_path=file_path,
                            article_index=article_index,
                            event_index=event_index,
                            article=article,
                            event=event,
                            reason="event_type or location is N/A",
                        )
                    )
                    continue

                start_date = parse_start_time(event.get("start_time"))

                # Ignore if start_time is not parseable
                if start_date is None:
                    errors["invalid_start_time"].append(
                        make_error_entry(
                            file_path=file_path,
                            article_index=article_index,
                            event_index=event_index,
                            article=article,
                            event=event,
                            reason="start_time is not parseable as YYYY-MM-DD or ISO datetime",
                        )
                    )
                    continue

                # Ignore if article date is malformed
                if article_date is None:
                    errors["invalid_article_date"].append(
                        make_error_entry(
                            file_path=file_path,
                            article_index=article_index,
                            event_index=event_index,
                            article=article,
                            event=event,
                            reason="article date is not parseable as YYYYMMDDHHMMSS",
                        )
                    )
                    continue

                # Ignore if start_time and article date differ by > 3 months
                if not within_three_months(article_date, start_date):
                    errors["start_time_too_far_from_article_date"].append(
                        make_error_entry(
                            file_path=file_path,
                            article_index=article_index,
                            event_index=event_index,
                            article=article,
                            event=event,
                            reason=(
                                f"start_time date {start_date} is more than "
                                f"3 months away from article date {article_date}"
                            ),
                        )
                    )
                    continue

                # Keep every filtered event. No deduplication.
                filtered_event = {
                    **event,

                    # Article/source metadata
                    "url": article.get("link"),
                    "article_date": article.get("date"),
                    "source_file": str(file_path),
                    "article_index": article_index,
                    "event_index": event_index,

                    # Dates as original strings
                    "start_time_str": event.get("start_time"),
                    "end_time_str": event.get("end_time"),

                    # Parsed helper field, useful later
                    "parsed_start_date": start_date.isoformat(),

                    # Location names
                    # Keeping this explicit makes later geo steps easier.
                    "location_name": location,
                }

                filtered_events.append(filtered_event)

    return filtered_events, errors

def sample_locations(unique_locations, n=100, seed=None):
    if seed is not None:
        random.seed(seed)

    n = min(n, len(unique_locations))
    return random.sample(unique_locations, n)

def get_events_with_names(filtered_events):
    named_events = []

    for event in filtered_events:
        event_name = event.get("event_name")

        if not is_na(event_name):
            named_events.append(event)

    return named_events

def cluster_filtered_events_by_type_space_time(
    filtered_events,
    incident_types,
    gm,
    distance_km=1.0,
    time_diff_hours=24,
):
    """
    Cluster filtered events into merged incidents.

    Parameters
    ----------
    filtered_events:
        List of event dictionaries from collect_filtered_events(...).

    incident_types:
        List of event types to include, e.g. ["wildfire", "fire"].

    gm:
        GeoManager instance with gm.get_geo_region(location_name).

    distance_km:
        Max centroid distance for two events to be spatially mergeable.

    time_diff_hours:
        Max start-time difference in hours for two events to be temporally mergeable.

    Returns
    -------
    merged_incidents:
        List of merged incident clusters.

    skipped:
        Dict of skipped events and reasons.
    """
    incident_type_set = {
        normalize_text(event_type)
        for event_type in incident_types
    }

    candidate_events = []
    skipped = {
        "non_matching_event_type": [],
        "missing_location": [],
        "missing_or_unparseable_start_time": [],
        "geocode_failed": [],
    }

    # -------------------------
    # 1. Filter and geocode
    # -------------------------
    print("Filter and geocode")
    for idx, event in tqdm(enumerate(filtered_events)):
        event_type = event.get("event_type")

        if normalize_text(event_type) not in incident_type_set:
            skipped["non_matching_event_type"].append(idx)
            continue

        location = event.get("location_name") or event.get("location")

        if is_na(location):
            skipped["missing_location"].append(idx)
            continue

        start_dt = parse_event_datetime(event)

        if start_dt is None:
            skipped["missing_or_unparseable_start_time"].append(idx)
            continue

        geo_record = gm.get_geo_region(location)

        if geo_record.get("status") != "ok" or geo_record.get("geometry") is None:
            skipped["geocode_failed"].append({
                "filtered_event_index": idx,
                "location": location,
                "event_type": event_type,
                "geo_record": geo_record,
            })
            continue

        geom = geo_record["geometry"]
        centroid_lat, centroid_lon = geometry_centroid_latlon(geom)

        candidate_events.append({
            "filtered_event_index": idx,
            "event": event,
            "event_type": event_type,
            "normalized_event_type": normalize_text(event_type),
            "location_name": location,
            "start_dt": start_dt,
            "geometry": geom,
            "geo_record": geo_record,
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
        })

    # Nothing to cluster
    if not candidate_events:
        return [], skipped

    # -------------------------
    # 2. Build merge graph
    # -------------------------
    candidate_indices = list(range(len(candidate_events)))
    uf = UnionFind(candidate_indices)

    print("Merge Graph")
    for a_pos in tqdm(range(len(candidate_events))):
        a = candidate_events[a_pos]

        for b_pos in range(a_pos + 1, len(candidate_events)):
            b = candidate_events[b_pos]

            # Require same normalized event type.
            # If you want "wildfire" and "fire" to merge with each other,
            # remove this condition.
            # if a["normalized_event_type"] != b["normalized_event_type"]:
            #     continue

            time_diff = abs(
                (a["start_dt"] - b["start_dt"]).total_seconds()
            ) / 3600.0

            if time_diff > time_diff_hours:
                continue

            spatial_dist = haversine_km(
                a["centroid_lat"],
                a["centroid_lon"],
                b["centroid_lat"],
                b["centroid_lon"],
            )

            if spatial_dist > distance_km:
                continue

            uf.union(a_pos, b_pos)

    # -------------------------
    # 3. Convert connected components into merged incidents
    # -------------------------
    merged_incidents = []

    print("Clustering")
    for cluster_id, group in tqdm(enumerate(uf.groups())):
        cluster_items = [candidate_events[pos] for pos in group]

        filtered_event_indices = [
            item["filtered_event_index"]
            for item in cluster_items
        ]

        source_events = [
            item["event"]
            for item in cluster_items
        ]

        event_types = sorted({
            item["event_type"]
            for item in cluster_items
        })

        location_names = sorted({
            item["location_name"]
            for item in cluster_items
        })

        urls = sorted({
            event.get("url")
            for event in source_events
            if event.get("url")
        })

        source_files = sorted({
            event.get("source_file")
            for event in source_events
            if event.get("source_file")
        })

        start_datetimes = [
            item["start_dt"]
            for item in cluster_items
        ]

        centroid_lat = sum(
            item["centroid_lat"] for item in cluster_items
        ) / len(cluster_items)

        centroid_lon = sum(
            item["centroid_lon"] for item in cluster_items
        ) / len(cluster_items)

        representative = cluster_items[0]

        merged_incidents.append({
            "cluster_id": cluster_id,
            "num_mentions": len(cluster_items),

            # Main cluster fields
            "event_types": event_types,
            "representative_event_type": representative["event_type"],
            "representative_location_name": representative["location_name"],
            "canonical_start_datetime": min(start_datetimes).isoformat(),

            # Spatial summary
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "location_names": location_names,

            # Provenance
            "filtered_event_indices": filtered_event_indices,
            "urls": urls,
            "source_files": source_files,

            # Full references
            "source_events": source_events,

            # Useful for debugging
            "geo_methods": sorted({
                item["geo_record"].get("method")
                for item in cluster_items
                if item["geo_record"].get("method")
            }),
        })

    return merged_incidents, skipped

def get_list_of_incidents(filepath):

    with open(filepath, "r") as f:
        data = f.readlines()
        data = [x.strip() for x in data]
    
    return data

def canonical_name(value):
    if is_na(value):
        return None
    return str(value).strip().lower()

def make_incident_key(incident):
    """
    Fallback human-readable key for incidents without names.
    This is not the unique ID. It is just a descriptive label.
    """
    event_type = incident.get("representative_event_type", "unknown_event")
    location = incident.get("representative_location_name", "unknown_location")
    start = incident.get("canonical_start_datetime", "unknown_time")

    return f"{event_type} | {location} | {start}"


def get_event_names(event):
    """
    Names are optional. Return an empty list if no names exist.
    """
    names = []

    for key in [
        "event_name",
        "incident_name",
        "name",
        "title",
    ]:
        value = event.get(key)
        if not is_na(value):
            names.append(str(value).strip())

    return names


def get_merged_incident_names(merged_incident):
    """
    Collect all names associated with a merged incident from its source events.
    """
    names = set()

    for event in merged_incident.get("source_events", []):
        for name in get_event_names(event):
            names.add(name)

    return sorted(names)


def add_incident_ids(all_merged, prefix="incident"):
    """
    Assign IDs to every merged incident, even unnamed ones.
    """
    all_merged_by_id = {}
    name_to_incident_ids = defaultdict(set)
    filtered_event_index_to_incident_id = {}

    for idx, incident in enumerate(all_merged):
        incident_id = f"{prefix}_{idx:06d}"

        incident_names = set()
        for event in incident.get("source_events", []):
            incident_names.update(get_event_names(event))

        incident_with_id = {
            **incident,
            "incident_id": incident_id,
            "incident_names": sorted(incident_names),
            "incident_key": make_incident_key(incident),
        }

        all_merged_by_id[incident_id] = incident_with_id

        for name in incident_names:
            name_to_incident_ids[normalize_text(name)].add(incident_id)

        for filtered_idx in incident.get("filtered_event_indices", []):
            filtered_event_index_to_incident_id[filtered_idx] = incident_id

    name_to_incident_ids = {
        name: sorted(ids)
        for name, ids in name_to_incident_ids.items()
    }

    return (
        all_merged_by_id,
        name_to_incident_ids,
        filtered_event_index_to_incident_id,
    )

def iter_part_of_relationships(event):
    """
    Yield part_of relationships from an event.

    Handles your observed schema:

        {
            "relationship_type": "part_of",
            "other_event_name": "Palisades Fire"
        }

    Also handles several other common relationship schemas.
    """

    # Case 1: direct field on the event
    direct_parent = event.get("part_of")
    if not is_na(direct_parent):
        yield {
            "child_name": None,
            "parent_name": str(direct_parent).strip(),
            "raw_relationship": {
                "format": "direct_part_of",
                "part_of": direct_parent,
            },
        }

    # Case 2: relationship objects
    relationship_fields = [
        "relationships",
        "relations",
        "compositional_relationships",
        "incident_relationships",
    ]

    for field in relationship_fields:
        relationships = event.get(field)

        if not isinstance(relationships, list):
            continue

        for rel in relationships:
            if not isinstance(rel, dict):
                continue

            rel_type = (
                rel.get("relationship_type")      # your schema
                or rel.get("relationship")
                or rel.get("relation")
                or rel.get("type")
                or rel.get("predicate")
            )

            if normalize_text(rel_type) != "part_of":
                continue

            # In your schema, the current event is the child,
            # and other_event_name is the parent.
            child_name = (
                rel.get("subject")
                or rel.get("source")
                or rel.get("from")
                or rel.get("child")
                or rel.get("sub_event")
                or event.get("event_name")
            )

            parent_name = (
                rel.get("other_event_name")       # your schema
                or rel.get("object")
                or rel.get("target")
                or rel.get("to")
                or rel.get("parent")
                or rel.get("higher_level_event")
            )

            yield {
                "child_name": None if is_na(child_name) else str(child_name).strip(),
                "parent_name": None if is_na(parent_name) else str(parent_name).strip(),
                "raw_relationship": rel,
            }

def resolve_unique_incident_id(name, name_to_incident_ids):
    if is_na(name):
        return None, "missing_name"

    ids = name_to_incident_ids.get(normalize_text(name), [])

    if len(ids) == 1:
        return ids[0], "resolved"

    if len(ids) == 0:
        return None, "not_found"

    return None, "ambiguous"

def build_part_of_incident_dictionary(all_merged_by_id, name_to_incident_ids):
    """
    Builds two dictionaries:

    1. resolved_by_parent:
       Parent incident must resolve to exactly one incident ID.

    2. mentions_by_parent_name:
       Keeps every extracted part_of relationship, even if unresolved.
    """
    resolved_by_parent = {}
    mentions_by_parent_name = {}
    unresolved = []

    for current_incident_id, merged_incident in all_merged_by_id.items():
        for event in merged_incident.get("source_events", []):
            for rel in iter_part_of_relationships(event):

                child_name = rel["child_name"]
                parent_name = rel["parent_name"]

                if is_na(parent_name):
                    unresolved.append({
                        "current_incident_id": current_incident_id,
                        "current_incident_key": merged_incident.get("incident_key"),
                        "reason": "missing_parent_name",
                        "child_name": child_name,
                        "parent_name": parent_name,
                        "raw_relationship": rel["raw_relationship"],
                        "source_event_url": event.get("url"),
                    })
                    continue

                # Child can always be the current merged cluster.
                child_id = current_incident_id

                child_incident = all_merged_by_id[child_id]
                child_label = (
                    child_incident["incident_names"][0]
                    if child_incident.get("incident_names")
                    else child_incident.get("incident_key")
                )

                parent_id, parent_status = resolve_unique_incident_id(
                    parent_name,
                    name_to_incident_ids,
                )

                # Always save the mention, even if parent ID resolution fails.
                if parent_name not in mentions_by_parent_name:
                    mentions_by_parent_name[parent_name] = {
                        "parent_name": parent_name,
                        "parent_incident_id": parent_id,
                        "parent_resolution_status": parent_status,
                        "child_incident_ids": [],
                        "children": [],
                        "source_relationships": [],
                    }

                mention_entry = mentions_by_parent_name[parent_name]

                if child_id not in mention_entry["child_incident_ids"]:
                    mention_entry["child_incident_ids"].append(child_id)
                    mention_entry["children"].append({
                        "name": child_label,
                        "incident_id": child_id,
                    })

                mention_entry["source_relationships"].append({
                    "child_name": child_label,
                    "child_incident_id": child_id,
                    "parent_name": parent_name,
                    "parent_incident_id": parent_id,
                    "parent_resolution_status": parent_status,
                    "source_event_url": event.get("url"),
                    "raw_relationship": rel["raw_relationship"],
                })

                # Only put it in the resolved dictionary if parent is uniquely resolved.
                if parent_status != "resolved":
                    unresolved.append({
                        "current_incident_id": current_incident_id,
                        "current_incident_key": merged_incident.get("incident_key"),
                        "reason": f"parent_{parent_status}",
                        "child_name": child_name,
                        "parent_name": parent_name,
                        "parent_candidate_ids": name_to_incident_ids.get(
                            normalize_text(parent_name),
                            [],
                        ),
                        "source_event_url": event.get("url"),
                        "raw_relationship": rel["raw_relationship"],
                    })
                    continue

                parent_incident = all_merged_by_id[parent_id]
                parent_label = (
                    parent_incident["incident_names"][0]
                    if parent_incident.get("incident_names")
                    else parent_incident.get("incident_key")
                )

                if parent_label not in resolved_by_parent:
                    resolved_by_parent[parent_label] = {
                        "incident_id": parent_id,
                        "children": [],
                        "child_incident_ids": [],
                        "source_relationships": [],
                    }

                parent_entry = resolved_by_parent[parent_label]

                if child_id not in parent_entry["child_incident_ids"]:
                    parent_entry["child_incident_ids"].append(child_id)
                    parent_entry["children"].append({
                        "name": child_label,
                        "incident_id": child_id,
                    })

                parent_entry["source_relationships"].append({
                    "child_name": child_label,
                    "child_incident_id": child_id,
                    "parent_name": parent_label,
                    "parent_incident_id": parent_id,
                    "source_event_url": event.get("url"),
                    "raw_relationship": rel["raw_relationship"],
                })

    return resolved_by_parent, mentions_by_parent_name, unresolved

def build_part_of_by_parent_name(all_merged_by_id, name_to_incident_ids):
    """
    Build a name-level part_of dictionary.

    This treats parent names as canonical incident identities.
    Multiple merged incident IDs may share the same parent name.
    """
    part_of_by_parent_name = {}
    unresolved = []

    for current_incident_id, merged_incident in all_merged_by_id.items():
        for event in merged_incident.get("source_events", []):
            for rel in iter_part_of_relationships(event):

                child_name = rel.get("child_name")
                parent_name = rel.get("parent_name")

                if is_na(parent_name):
                    unresolved.append({
                        "current_incident_id": current_incident_id,
                        "current_incident_key": merged_incident.get("incident_key"),
                        "reason": "missing_parent_name",
                        "child_name": child_name,
                        "parent_name": parent_name,
                        "source_event_url": event.get("url"),
                        "raw_relationship": rel.get("raw_relationship"),
                    })
                    continue

                # The current merged incident is the child evidence.
                child_id = current_incident_id

                child_incident = all_merged_by_id[child_id]
                child_label = (
                    child_incident["incident_names"][0]
                    if child_incident.get("incident_names")
                    else child_incident.get("incident_key")
                )

                normalized_parent_name = normalize_text(parent_name)

                parent_candidate_ids = name_to_incident_ids.get(
                    normalized_parent_name,
                    [],
                )

                if parent_name not in part_of_by_parent_name:
                    part_of_by_parent_name[parent_name] = {
                        "parent_name": parent_name,

                        # There may be 0, 1, or many merged clusters
                        # that use this same parent name.
                        "parent_candidate_ids": parent_candidate_ids,

                        # This is name-level, not strict ID-level.
                        "parent_resolution_status": (
                            "name_found"
                            if parent_candidate_ids
                            else "name_not_found_as_merged_incident"
                        ),

                        "child_incident_ids": [],
                        "children": [],
                        "source_relationships": [],
                    }

                parent_entry = part_of_by_parent_name[parent_name]

                if child_id not in parent_entry["child_incident_ids"]:
                    parent_entry["child_incident_ids"].append(child_id)
                    parent_entry["children"].append({
                        "name": child_label,
                        "incident_id": child_id,
                    })

                parent_entry["source_relationships"].append({
                    "child_name": child_label,
                    "child_incident_id": child_id,
                    "parent_name": parent_name,
                    "parent_candidate_ids": parent_candidate_ids,
                    "source_event_url": event.get("url"),
                    "raw_relationship": rel.get("raw_relationship"),
                })

    # Stable ordering
    for parent_entry in part_of_by_parent_name.values():
        parent_entry["parent_candidate_ids"] = sorted(
            set(parent_entry["parent_candidate_ids"])
        )
        parent_entry["child_incident_ids"] = sorted(
            set(parent_entry["child_incident_ids"])
        )
        parent_entry["children"] = sorted(
            parent_entry["children"],
            key=lambda x: (x["name"], x["incident_id"]),
        )

    return part_of_by_parent_name, unresolved

def save_json(data, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def get_top_level_names(part_of_incidents_path):
    """
    Get all top-level parent names from:
      1. part_of_incidents_by_parent.json

    Returns a dictionary with names from each file and their union.
    """
    part_of_incidents = load_json(part_of_incidents_path)

    part_of_incident_names = sorted(part_of_incidents.keys())

    all_top_level_names = sorted(
        set(part_of_incident_names)
    )
    return part_of_incident_names

if __name__ == "__main__":


    gm = GeoManager() # Get geomanager for geocoding locations

    filtered_events, errors = collect_filtered_events("evaluation/news_by_date")
    # print_top_event_types(filtered_events, top_n=15)

    print("Filtered events:", len(filtered_events))



    unique_locations = get_unique_locations(filtered_events)

    print("Number of unique locations:", len(unique_locations))

    # Iterate through each unique location and get it
    # for location in tqdm(unique_locations):
    #     print(location)
    #     record = gm.get_geo_region(location)

    
    # # For every event 
    # named_events = get_events_with_names(filtered_events)
    # print("Named events:", len(named_events))


    incident_list = get_list_of_incidents("detection/emergency_types_filtered.txt")


    all_merged = []
    incident_counts = {}

    for incident_name in tqdm(incident_list):

        # Iterate through each item in the incident list
        merged_for_incident, skipped = cluster_filtered_events_by_type_space_time(
            filtered_events=filtered_events,
            incident_types=[incident_name],
            gm=gm,
            distance_km=2.0,
            time_diff_hours=24,
        )

        print("Merged incidents:", len(merged_for_incident))
        print("Skipped due to geocode failure:", len(skipped["geocode_failed"]))

        all_merged.extend(merged_for_incident)

        incident_counts[incident_name] = len(merged_for_incident)
    


    print("All merged incident count ", len(all_merged))
    print("Incident counts: ", incident_counts)

    output_folder = Path("evaluation/merged_incidents")
    output_folder.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Add globally unique IDs to all merged incidents
    # --------------------------------------------------
    all_merged_by_id, name_to_incident_ids, filtered_event_index_to_incident_id = (
    add_incident_ids(all_merged)
)
    
    # Save merged incidents
    save_json(
        all_merged_by_id,
        output_folder / "all_merged_by_id.json",
    )
    
    save_json(
        name_to_incident_ids,
        output_folder / "name_to_incident_ids.json",
    )
    
    
    # --------------------------------------------------
    # Build part_of dictionary keyed by higher-level incident
    # --------------------------------------------------
    part_of_by_parent_name, unresolved_part_of = build_part_of_by_parent_name(
        all_merged_by_id,
        name_to_incident_ids,
    )


    # Save part_of incidents
    part_of_by_parent_name_path = output_folder / "part_of_by_parent_name.json"
    part_of_mentions_path = output_folder / "part_of_mentions_by_parent_name.json"
    unresolved_mentions_path = output_folder / "unresolved_part_of_relationships.json"
    save_json(
        part_of_by_parent_name,
        part_of_by_parent_name_path,
    )

    save_json(
        unresolved_part_of,
        unresolved_mentions_path,
    )

    

    print("Saved all merged incidents to:", output_folder / "all_merged_by_id.json")
    print("Saved part_of relationships to:", part_of_by_parent_name_path)
    print("Saved unresolved part_of relationships to:", unresolved_mentions_path)
    print("Saved name index to:", output_folder / "name_to_incident_ids.json")

    top_level_names = get_top_level_names(
        part_of_by_parent_name_path,
    )
        
    print("Number incidents: ")
    print(len(top_level_names))