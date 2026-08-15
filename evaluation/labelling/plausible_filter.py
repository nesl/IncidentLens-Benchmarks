
import json
import copy
from pathlib import Path
import tarfile
import shutil
from datetime import datetime

from shapely.ops import unary_union

from evaluation.labelling.sensor_coverage import untar_date_file, retrieve_by_date_air, retrieve_by_date_alertcalifornia, retrieve_by_date_cctv, retrieve_by_date_traffic_coverage, retrieve_by_date_weather

from evaluation.geo_manager import GeoManager

from tqdm import tqdm
from utilities.util import get_config

INCIDENT_TO_DATA_SOURCES = {
    "active shooter situation": [
        "cctv",
        "citizen_data",
        "twitter_data",
    ],

    "civil protest": [
        "cctv",
        "citizen_data",
        "pem_data_station_5min",
        "twitter_data",
    ],

    "demonstration": [
        "cctv",
        "citizen_data",
        "pem_data_station_5min",
        "twitter_data",
    ],

    "fire": [
        "air_data",
        "alertcalifornia",
        "cctv",
        "citizen_data",
        "twitter_data",
        "weather_data",
    ],

    "home crime": [
        "citizen_data",
        "twitter_data",
    ],

    "road vehicle accident": [
        "cctv",
        "citizen_data",
        "pem_data_station_5min",
        "twitter_data",
    ],

    "wildfire": [
        "air_data",
        "alertcalifornia",
        "cctv",
        "citizen_data",
        "twitter_data",
        "weather_data",
    ],

    "bomb threat": [
        "cctv",
        "citizen_data",
        "twitter_data",
    ],

    "dangerous person threat": [
        "cctv",
        "citizen_data",
        "twitter_data",
    ],

    "flood": [
        "cctv",
        "citizen_data",
        "pem_data_station_5min",
        "twitter_data",
        "weather_data",
    ],

    "industrial crime": [
        "air_data",
        "cctv",
        "citizen_data",
        "twitter_data",
    ],

    "road closure": [
        "cctv",
        "citizen_data",
        "pem_data_station_5min",
        "twitter_data",
        "weather_data",
    ],

    "school closing": [
        "citizen_data",
        "twitter_data",
        "weather_data",
    ],

    "terrorist incident": [
        "cctv",
        "citizen_data",
        "pem_data_station_5min",
        "twitter_data",
    ],
}

def parse_incident_date_to_yyyymmdd(value):
# Iterate through the all_merged_id file
    """
    Convert an ISO-ish datetime string into YYYYMMDD.

    Examples:
      2025-05-14T07:00:00 -> 20250514
      2025-05-14 -> 20250514
    """
    if value is None:
        return None

    value = str(value).strip()

    if not value or value.lower() in {"n/a", "na", "none"}:
        return None

    # Handle ISO strings with Z
    value = value.replace("Z", "+00:00")

    try:
        dt = datetime.fromisoformat(value)
        return dt.strftime("%Y%m%d")
    except ValueError:
        pass

    # Fallback for date-only strings
    try:
        dt = datetime.strptime(value[:10], "%Y-%m-%d")
        return dt.strftime("%Y%m%d")
    except ValueError:
        return None


def parse_incident_summary(incident_id, incident):
    """
    Parse one incident record.
    """
    canonical_start_datetime = incident.get("canonical_start_datetime")

    return {
        "incident_id": incident_id,
        "representative_event_type": incident.get("representative_event_type"),
        "location_names": incident.get("location_names", []),
        "canonical_start_datetime": canonical_start_datetime,
        "canonical_start_date": parse_incident_date_to_yyyymmdd(
            canonical_start_datetime
        ),
    }


# Given a list of geo names, return the region which is the union of the two
def get_union_region(list_of_locations, geomanager):
    geoms = []

    for loc_name in list_of_locations:
        geom_data = geomanager.get_geo_region(loc_name)

        if geom_data.get("status") != "ok" or geom_data.get("geometry") is None:
            print(f"Skipping failed geocode: {loc_name}")
            continue

        geoms.append(geom_data["geometry"])

    if not geoms:
        return None

    return unary_union(geoms)
    
def clean_geom(geom):
    """
    Fix minor invalid geometry issues if possible.
    """
    if geom is None:
        return None

    if geom.is_empty:
        return None

    if not geom.is_valid:
        geom = geom.buffer(0)

    if geom.is_empty:
        return None

    return geom


def polygons_intersect_location(polygons, location_geom):
    """
    Return True if any polygon intersects the incident location geometry.
    Otherwise return False.
    """
    if location_geom is None or location_geom.is_empty:
        return False

    if not location_geom.is_valid:
        location_geom = location_geom.buffer(0)

    for poly in polygons:
        if poly is None or poly.is_empty:
            continue

        if not poly.is_valid:
            poly = poly.buffer(0)

        if poly.intersects(location_geom):
            return True

    return False
    


def atomic_save_json(data, output_path):
    """
    Save JSON safely by writing to a temporary file first,
    then replacing the original file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    tmp_path.replace(output_path)


def load_existing_json(output_path):
    """
    Load existing output JSON if it exists.
    Otherwise return an empty dictionary.
    """
    output_path = Path(output_path)

    if not output_path.exists():
        return {}

    with open(output_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_incident_summaries_from_file(
    input_path,
    geo_manager,
    temp_folder,
    data_folder,
    output_path,
    polygon_cell_km=0.25,
):
    """
    Load incident JSON, compute temporal/spatial relevance, and incrementally
    save relevant incidents to output_path.

    If an incident_id already exists in output_path, skip it.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    with open(input_path, "r", encoding="utf-8") as f:
        incidents = json.load(f)

    # Load existing output so we can resume.
    relevant_incidents = load_existing_json(output_path)

    print(f"Loaded {len(relevant_incidents)} existing processed incidents.")

    for incident_id, incident in tqdm(incidents.items()):
        # Skip if already saved
        if incident_id in relevant_incidents:
            continue

        incident_data = parse_incident_summary(
            incident_id,
            incident,
        )

        event_type = incident_data["representative_event_type"]
        locations = incident_data["location_names"]
        start_date = incident_data["canonical_start_date"]

        updated_incident = copy.deepcopy(incident)

        updated_incident["temporal_relevant"] = []
        updated_incident["spatial_relevant"] = []

        # Get union region
        location_geom = get_union_region(
            locations,
            geo_manager,
        )

        # Get associated sensory data
        data_source_choices = INCIDENT_TO_DATA_SOURCES.get(event_type, [])

        for data_source in data_source_choices:
            print(f"  Checking data source: {data_source}")

            temporal_present, polygons = get_polygons_for_data_source(
                data_source_name=data_source,
                date_str=start_date,
                temp_folder=temp_folder,
                data_folder=data_folder,
                geo_manager=geo_manager,
                polygon_cell_km=polygon_cell_km,
            )

            if temporal_present:
                updated_incident["temporal_relevant"].append(data_source)

            polygon_intersects = polygons_intersect_location(
                polygons=polygons,
                location_geom=location_geom,
            )

            if polygon_intersects:
                updated_incident["spatial_relevant"].append(data_source)

        updated_incident["any_temporal_relevant"] = bool(
            updated_incident["temporal_relevant"]
        )

        updated_incident["any_spatial_relevant"] = bool(
            updated_incident["spatial_relevant"]
        )

        # Save only relevant incidents
        if (
            updated_incident["any_temporal_relevant"]
            or updated_incident["any_spatial_relevant"]
        ):
            relevant_incidents[incident_id] = updated_incident

        # Save after every processed incident
        atomic_save_json(relevant_incidents, output_path)

    return relevant_incidents


# ["air_data", "alertcalifornia", "cctv", "citizen_data", "pem_data_station_5min", "twitter_data", "weather_data"]
def get_polygons_for_data_source(
    data_source_name,
    date_str,
    temp_folder,
    data_folder,
    geo_manager,
    polygon_cell_km=0.25,
):
    try:
        extracted_path = untar_date_file(
            date_str=date_str,
            data_folder=Path(data_folder) / data_source_name,
            temp_folder=temp_folder,
            clear_existing=True,
        )

    except FileNotFoundError as e:
        print(f"[missing tar] {data_source_name} for {date_str}: {e}")
        return False, []

    except tarfile.ReadError as e:
        print(f"[bad tar] {data_source_name} for {date_str}: {e}")
        return False, []

    except EOFError as e:
        print(f"[incomplete tar] {data_source_name} for {date_str}: {e}")
        return False, []

    polygons = []

    if data_source_name == "air_data":
        polygons, sensor_id_to_polygon, metadata = retrieve_by_date_air(
            extracted_path=extracted_path,
            date_str=date_str,
            radius_km=1.0,
            cell_km=polygon_cell_km,
            return_sensor_map=True,
        )

    elif data_source_name == "alertcalifornia":
        polygons, camera_direction_to_polygon, metadata = retrieve_by_date_alertcalifornia(
            extracted_path=extracted_path,
            date_str=date_str,
            locations_path="evaluation/sensor_locations/alertcalifornia_locations.json",
            fov_deg=60.0,
            distance_km=50.0,
            cell_km=polygon_cell_km,
            direction_is_svg_angle=True,
            return_camera_map=True,
        )

    elif data_source_name == "cctv":
        polygons, cctv_camera_to_polygon, metadata = retrieve_by_date_cctv(
            extracted_path=extracted_path,
            date_str=date_str,
            kml_path="evaluation/sensor_locations/cctv.kml",
            fov_deg=60.0,
            distance_km=1.0,
            cell_km=polygon_cell_km,
            unknown_direction_mode="circle",
            return_camera_map=True,
        )

    elif data_source_name == "pem_data_station_5min":
        polygons, station_id_to_polygon, station_id_to_info, metadata = (
            retrieve_by_date_traffic_coverage(
                extracted_path=extracted_path,
                date_str=date_str,
                stations_path="evaluation/sensor_locations/pem_7_stations.txt",
                radius_km=0.25,
                cell_km=polygon_cell_km,
                return_sensor_map=True,
            )
        )

    elif data_source_name == "weather_data":
        polygons, weather_location_to_polygon, metadata = retrieve_by_date_weather(
            extracted_path=extracted_path,
            date_str=date_str,
            gm=geo_manager,
            inject_california=True,
            return_location_map=True,
        )

    elif data_source_name in {"citizen_data", "twitter_data"}:
        polygons = []

    else:
        raise ValueError(f"Unknown data source: {data_source_name}")

    return True, polygons


if __name__ == "__main__":

    # Important note - remember that we have to treat twitter and citizen as full polygons over LA

    gm = GeoManager()
    merged_incidents_filepath = "evaluation/merged_incidents/all_merged_by_id_original.json"

    paths = get_config().get("paths", {})
    temp_folder = paths.get("evaluation_temp_root", "evaluation/temp")
    data_folder = paths.get("raw_archive_root", "./raw_data")
    output_path = "evaluation/merged_incidents/all_merged_by_id_relevant.json"

    parsed = parse_incident_summaries_from_file(
        input_path=merged_incidents_filepath, geo_manager=gm, temp_folder=temp_folder, data_folder=data_folder, output_path=output_path
    )

    
    # data_type = "weather_data"
    # date_str = "20260203"
    # cell_km = 0.25

    
