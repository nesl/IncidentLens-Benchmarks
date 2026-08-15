
import tarfile
import csv
import math
from pathlib import Path
import shutil
from evaluation.geo_manager import make_cone, make_circle
from collections import defaultdict, Counter
import json

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import csv
import gzip

from tqdm import tqdm
from utilities.util import get_config


from evaluation.geo_manager import GeoManager



DATA_SOURCES = ["air_data", "alertcalifornia", "cctv", "citizen_data", "pem_data_station_5min", "twitter_data", "weather_data"]



def safe_extract_tar(tar, extract_to):
    """
    Safely extract a tar file, preventing path traversal attacks.
    """
    extract_to = Path(extract_to).resolve()

    for member in tar.getmembers():
        member_path = (extract_to / member.name).resolve()

        if not str(member_path).startswith(str(extract_to)):
            raise RuntimeError(f"Unsafe tar path detected: {member.name}")

    tar.extractall(extract_to)

def untar_date_file(date_str, data_folder, temp_folder="temp", clear_existing=False):
    data_folder = Path(data_folder)
    temp_folder = Path(temp_folder)

    tar_path = data_folder / f"{date_str}.tar"
    extract_folder = temp_folder / data_folder.name

    if not tar_path.exists():
        raise FileNotFoundError(f"Could not find tar file: {tar_path}")

    if not tar_path.is_file():
        raise ValueError(f"Path exists but is not a file: {tar_path}")

    if tar_path.stat().st_size == 0:
        raise tarfile.ReadError(f"Tar file is empty: {tar_path}")

    if clear_existing and extract_folder.exists():
        shutil.rmtree(extract_folder)

    extract_folder.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(tar_path, "r:*") as tar:
            safe_extract_tar(tar, extract_folder)

    except tarfile.ReadError as e:
        raise tarfile.ReadError(
            f"Could not read tar file {tar_path}. "
            f"File size={tar_path.stat().st_size} bytes. "
            f"Original error: {e}"
        )

    return extract_folder


def retrieve_by_date_air(
    extracted_path,
    date_str,
    radius_km=1.0,
    cell_km=1.0,
    filter_files_by_date=False,
    return_sensor_map=False,
):
    """
    Iterate through CSV files in an extracted air-data folder and create one
    coverage polygon per unique sensor.

    Expected CSV row format:
        sensor_id,lat,lon,pm25

    Example row:
        168449,34.192623,-118.179146,8.5

    Parameters
    ----------
    extracted_path : str or Path
        Path to extracted folder, e.g. "temp/air_data".

    date_str : str
        Date string like "20250608".

    radius_km : float
        Assumed sensing/coverage radius for each sensor.
        Default is 1.0 km.

    cell_km : float
        Approximate polygon edge resolution.

    filter_files_by_date : bool
        If True, only reads CSV files whose filenames contain date_str.
        If False, reads every CSV under extracted_path.

    return_sensor_map : bool
        If False, returns only a list of polygons.
        If True, returns:
            polygons, sensor_id_to_polygon, metadata

    Returns
    -------
    list[Polygon]
        One polygon per unique sensor ID.

    Or, if return_sensor_map=True:
        polygons, sensor_id_to_polygon, metadata
    """
    extracted_path = Path(extracted_path)

    if not extracted_path.exists():
        raise FileNotFoundError(f"Extracted path does not exist: {extracted_path}")

    if not extracted_path.is_dir():
        raise ValueError(f"Extracted path is not a directory: {extracted_path}")

    csv_files = sorted(extracted_path.rglob("*.csv"))

    if filter_files_by_date:
        csv_files = [
            path for path in csv_files
            if date_str in path.name
        ]

    sensor_id_to_polygon = {}
    sensor_id_to_info = {}

    metadata = {
        "date_str": date_str,
        "extracted_path": str(extracted_path),
        "num_csv_files": len(csv_files),
        "num_rows_read": 0,
        "num_invalid_rows": 0,
        "num_duplicate_sensor_rows": 0,
        "num_location_conflicts": 0,
        "invalid_rows": [],
        "location_conflicts": [],
    }

    for csv_path in csv_files:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)

            for row_index, row in enumerate(reader):
                metadata["num_rows_read"] += 1

                # Skip empty rows
                if not row:
                    continue

                # Expected: sensor_id, lat, lon, pm25
                if len(row) < 4:
                    metadata["num_invalid_rows"] += 1
                    metadata["invalid_rows"].append({
                        "file": str(csv_path),
                        "row_index": row_index,
                        "row": row,
                        "reason": "expected at least 4 columns",
                    })
                    continue

                sensor_id = str(row[0]).strip()

                try:
                    lat = float(row[1])
                    lon = float(row[2])
                    pm25 = float(row[3])
                except ValueError:
                    metadata["num_invalid_rows"] += 1
                    metadata["invalid_rows"].append({
                        "file": str(csv_path),
                        "row_index": row_index,
                        "row": row,
                        "reason": "lat, lon, or pm25 could not be parsed as float",
                    })
                    continue

                if not sensor_id:
                    metadata["num_invalid_rows"] += 1
                    metadata["invalid_rows"].append({
                        "file": str(csv_path),
                        "row_index": row_index,
                        "row": row,
                        "reason": "missing sensor_id",
                    })
                    continue

                # If this sensor has already been seen, do not recompute polygon.
                if sensor_id in sensor_id_to_polygon:
                    metadata["num_duplicate_sensor_rows"] += 1

                    previous = sensor_id_to_info[sensor_id]
                    if (
                        abs(previous["lat"] - lat) > 1e-6
                        or abs(previous["lon"] - lon) > 1e-6
                    ):
                        metadata["num_location_conflicts"] += 1
                        metadata["location_conflicts"].append({
                            "sensor_id": sensor_id,
                            "first_lat": previous["lat"],
                            "first_lon": previous["lon"],
                            "new_lat": lat,
                            "new_lon": lon,
                            "file": str(csv_path),
                            "row_index": row_index,
                        })

                    continue

                polygon = make_circle(
                    lat=lat,
                    lon=lon,
                    radius_km=radius_km,
                    cell_km=cell_km,
                )

                sensor_id_to_polygon[sensor_id] = polygon
                sensor_id_to_info[sensor_id] = {
                    "sensor_id": sensor_id,
                    "lat": lat,
                    "lon": lon,
                    "first_pm25": pm25,
                    "source_file": str(csv_path),
                    "source_row_index": row_index,
                    "radius_km": radius_km,
                }

    polygons = list(sensor_id_to_polygon.values())

    metadata["num_unique_sensors"] = len(sensor_id_to_polygon)
    metadata["num_polygons"] = len(polygons)

    if return_sensor_map:
        return polygons, sensor_id_to_polygon, metadata

    return polygons

def get_alertcalifornia_camera_locs(
    root_folder="evaluation/temp/alertcalifornia",
    date_str="20251001",
    conflict_mode="most_common",
):
    """
    Returns:
        camera_locs:
            {camera_name: (lat, lon)}

        conflicts:
            {
                camera_name: {
                    "locations": [((lat, lon), count), ...],
                    "files": [...]
                }
            }

    conflict_mode:
        - "most_common": use the most frequently occurring lat/lon
        - "first": use the first valid lat/lon
        - "error": raise if a camera has multiple lat/lon values
    """
    date_folder = Path(root_folder) / date_str

    if not date_folder.exists():
        raise FileNotFoundError(f"Date folder does not exist: {date_folder}")

    camera_locs = {}
    conflicts = {}

    for camera_folder in sorted(date_folder.iterdir()):
        if not camera_folder.is_dir():
            continue

        camera_name = camera_folder.name
        location_files = sorted(camera_folder.glob("*.location"))

        observed_locations = []
        files_by_location = defaultdict(list)

        for location_file in location_files:
            line = location_file.read_text(encoding="utf-8").strip()

            if not line:
                continue

            parts = [x.strip() for x in line.split(",")]

            if len(parts) < 2:
                continue

            lat = float(parts[0])
            lon = float(parts[1])

            # Round lightly to avoid tiny floating-point noise.
            lat_lon = (round(lat, 6), round(lon, 6))

            observed_locations.append(lat_lon)
            files_by_location[lat_lon].append(str(location_file))

        if not observed_locations:
            continue

        location_counts = Counter(observed_locations)

        if len(location_counts) > 1:
            conflicts[camera_name] = {
                "locations": [
                    {
                        "lat_lon": lat_lon,
                        "count": count,
                        "files": files_by_location[lat_lon],
                    }
                    for lat_lon, count in location_counts.most_common()
                ]
            }

            if conflict_mode == "error":
                raise ValueError(
                    f"Conflicting lat/lon for camera {camera_name}: "
                    f"{location_counts}"
                )

        if conflict_mode == "first":
            chosen_lat_lon = observed_locations[0]
        elif conflict_mode == "most_common":
            chosen_lat_lon = location_counts.most_common(1)[0][0]
        else:
            raise ValueError(f"Unknown conflict_mode: {conflict_mode}")

        camera_locs[camera_name] = chosen_lat_lon

    return camera_locs, conflicts

def save_alertcalifornia_camera_locs(
    camera_locs,
    output_folder="sensor_locations",
    filename="alertcalifornia_locations.json",
):
    """
    Save AlertCalifornia camera location mapping to JSON.

    Input:
        camera_locs = {
            "Baldwin Hills 1": (34.01, -118.36),
            ...
        }

    Output JSON:
        {
            "Baldwin Hills 1": {
                "lat": 34.01,
                "lon": -118.36
            }
        }
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / filename

    json_ready = {
        camera_name: {
            "lat": lat,
            "lon": lon,
        }
        for camera_name, (lat, lon) in camera_locs.items()
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_ready, f, indent=2, ensure_ascii=False)

    return output_path

def load_alertcalifornia_camera_locs(
    locations_path="sensor_locations/alertcalifornia_locations.json",
):
    """
    Load saved AlertCalifornia camera locations.

    Supports JSON shaped like:

        {
          "Camera Name": {"lat": 34.0, "lon": -118.0}
        }

    or:

        {
          "Camera Name": [34.0, -118.0]
        }
    """
    locations_path = Path(locations_path)

    if not locations_path.exists():
        raise FileNotFoundError(
            f"Camera locations file does not exist: {locations_path}"
        )

    with open(locations_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    camera_locs = {}

    for camera_name, value in raw.items():
        if isinstance(value, dict):
            lat = float(value["lat"])
            lon = float(value["lon"])
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            lat = float(value[0])
            lon = float(value[1])
        else:
            raise ValueError(
                f"Unrecognized location format for camera {camera_name}: {value}"
            )

        camera_locs[camera_name] = (lat, lon)

    return camera_locs


def parse_location_file(location_file):
    """
    Parse a .location file.

    Expected one-line format:
        lat,lon,direction

    Example:
        33.9639,-118.2898,140.19442890773482
    """
    line = Path(location_file).read_text(encoding="utf-8").strip()

    parts = [x.strip() for x in line.split(",")]

    if len(parts) < 3:
        raise ValueError(f"Malformed .location file {location_file}: {line}")

    lat = float(parts[0])
    lon = float(parts[1])
    direction = float(parts[2])

    return lat, lon, direction


def parse_direction_file(direction_file):
    """
    Parse a .direction file.

    Expected one-line format:
        direction

    Example:
        140.19442890773482
    """
    line = Path(direction_file).read_text(encoding="utf-8").strip()

    if not line:
        raise ValueError(f"Empty .direction file: {direction_file}")

    return float(line)


def svg_angle_to_compass_bearing(svg_angle):
    """
    Convert the angle returned by your get_line_angle(...) function into
    a compass bearing usable by destination_point(...).

    Your scraped angle behaves like a math angle:
      0   = east
      90  = north
      -90 = south

    Compass bearing expects:
      0   = north
      90  = east
      180 = south
      270 = west

    Conversion:
      compass = (90 - svg_angle) % 360
    """
    return (90.0 - svg_angle) % 360.0


def resolve_alertcalifornia_date_folder(extracted_path, date_str):
    """
    Supports either:
      extracted_path = "temp/alertcalifornia"
    or:
      extracted_path = "temp/alertcalifornia/20251001"
    """
    extracted_path = Path(extracted_path)

    if extracted_path.name == date_str:
        date_folder = extracted_path
    else:
        date_folder = extracted_path / date_str

    if not date_folder.exists():
        raise FileNotFoundError(f"Date folder does not exist: {date_folder}")

    if not date_folder.is_dir():
        raise ValueError(f"Expected date folder to be a directory: {date_folder}")

    return date_folder


def retrieve_by_date_alertcalifornia(
    extracted_path,
    date_str,
    locations_path="sensor_locations/alertcalifornia_locations.json",
    fov_deg=60.0,
    distance_km=50.0,
    cell_km=1.0,
    direction_is_svg_angle=True,
    round_direction_decimals=2,
    return_camera_map=False,
):
    """
    Retrieve spatial coverage cones for AlertCalifornia cameras on a date.

    Expected folder structure:
        temp/alertcalifornia/20251001/<camera_name>/*.location

    or:
        temp/alertcalifornia/20251001/<camera_name>/*.direction

    The two file types are assumed exclusive per camera folder.

    .location format:
        lat,lon,direction

    .direction format:
        direction

    If only .direction files exist, this function looks up lat/lon from:
        sensor_locations/alertcalifornia_locations.json

    Parameters
    ----------
    extracted_path : str or Path
        Either the AlertCalifornia root folder or the date folder.
        Examples:
            "temp/alertcalifornia"
            "temp/alertcalifornia/20251001"

    date_str : str
        Date string like "20251001".

    locations_path : str or Path
        Saved camera location mapping.

    fov_deg : float
        Field of view angle in degrees.
        Default 60 degrees.

    distance_km : float
        Maximum visual coverage distance.
        Default 50 km. This is a modeling assumption.

    cell_km : float
        Approximate resolution for cone arc construction.

    direction_is_svg_angle : bool
        If True, converts direction using svg_angle_to_compass_bearing.
        Use True if directions come from your get_line_angle(...) function.
        Use False if directions are already compass bearings.

    round_direction_decimals : int
        Used to avoid recomputing nearly identical directions.

    return_camera_map : bool
        If False, return only list of polygons.
        If True, return:
            polygons, camera_direction_to_polygon, metadata

    Returns
    -------
    list[Polygon]
        One cone polygon per unique camera-direction pair.

    Or, if return_camera_map=True:
        polygons, camera_direction_to_polygon, metadata
    """
    date_folder = resolve_alertcalifornia_date_folder(extracted_path, date_str)

    saved_camera_locs = load_alertcalifornia_camera_locs(locations_path)

    camera_direction_to_polygon = {}

    metadata = {
        "date_str": date_str,
        "date_folder": str(date_folder),
        "locations_path": str(locations_path),
        "fov_deg": fov_deg,
        "distance_km": distance_km,
        "cell_km": cell_km,
        "direction_is_svg_angle": direction_is_svg_angle,
        "num_camera_folders": 0,
        "num_location_files": 0,
        "num_direction_files": 0,
        "num_duplicate_camera_directions": 0,
        "num_missing_saved_locations": 0,
        "num_invalid_files": 0,
        "missing_saved_locations": [],
        "invalid_files": [],
        "mixed_file_type_cameras": [],
    }

    for camera_folder in sorted(date_folder.iterdir()):
        if not camera_folder.is_dir():
            continue

        metadata["num_camera_folders"] += 1

        camera_name = camera_folder.name

        location_files = sorted(camera_folder.glob("*.location"))
        direction_files = sorted(camera_folder.glob("*.direction"))

        metadata["num_location_files"] += len(location_files)
        metadata["num_direction_files"] += len(direction_files)

        if location_files and direction_files:
            metadata["mixed_file_type_cameras"].append({
                "camera_name": camera_name,
                "camera_folder": str(camera_folder),
                "num_location_files": len(location_files),
                "num_direction_files": len(direction_files),
            })

        # Prefer .location files if present because they include lat/lon.
        if location_files:
            files_to_process = location_files
            file_mode = "location"
        elif direction_files:
            files_to_process = direction_files
            file_mode = "direction"
        else:
            continue

        for data_file in files_to_process:
            try:
                if file_mode == "location":
                    lat, lon, raw_direction = parse_location_file(data_file)

                else:
                    raw_direction = parse_direction_file(data_file)

                    if camera_name not in saved_camera_locs:
                        metadata["num_missing_saved_locations"] += 1
                        metadata["missing_saved_locations"].append({
                            "camera_name": camera_name,
                            "camera_folder": str(camera_folder),
                            "direction_file": str(data_file),
                        })
                        continue

                    lat, lon = saved_camera_locs[camera_name]

            except Exception as e:
                metadata["num_invalid_files"] += 1
                metadata["invalid_files"].append({
                    "camera_name": camera_name,
                    "file": str(data_file),
                    "reason": repr(e),
                })
                continue

            if direction_is_svg_angle:
                bearing = svg_angle_to_compass_bearing(raw_direction)
            else:
                bearing = raw_direction % 360.0

            rounded_bearing = round(bearing, round_direction_decimals)

            coverage_key = (camera_name, rounded_bearing)

            if coverage_key in camera_direction_to_polygon:
                metadata["num_duplicate_camera_directions"] += 1
                continue

            polygon = make_cone(
                lat=lat,
                lon=lon,
                bearing=bearing,
                fov_deg=fov_deg,
                distance_km=distance_km,
                cell_km=cell_km,
            )

            camera_direction_to_polygon[coverage_key] = {
                "camera_name": camera_name,
                "lat": lat,
                "lon": lon,
                "raw_direction": raw_direction,
                "bearing": bearing,
                "rounded_bearing": rounded_bearing,
                "source_file": str(data_file),
                "file_mode": file_mode,
                "polygon": polygon,
            }

    polygons = [
        entry["polygon"]
        for entry in camera_direction_to_polygon.values()
    ]

    metadata["num_unique_camera_directions"] = len(camera_direction_to_polygon)
    metadata["num_polygons"] = len(polygons)

    if return_camera_map:
        return polygons, camera_direction_to_polygon, metadata

    return polygons

def normalize_text(value):
    return str(value).strip().lower()

def normalize_camera_key(value):
    """
    Normalize camera names so folders and KML names match.

    Examples:
      "I-5 - (25) Meadowdale" -> "i525meadowdale"
      "I-5 : (25) Meadowdale" -> "i525meadowdale"
    """
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def xml_local_name(tag):
    """
    Strip XML namespace from tag.

    Example:
      "{http://www.opengis.net/kml/2.2}Placemark" -> "Placemark"
    """
    return tag.split("}", 1)[-1] if "}" in tag else tag


def get_simple_data_dict(placemark):
    """
    Extract all ExtendedData/SimpleData fields from a Placemark.

    Returns:
        {
            "locationName": "I-5 : (25) Meadowdale",
            "direction": "South",
            "latitude": "34.08842",
            "longitude": "-118.2369",
            ...
        }
    """
    data = {}

    for elem in placemark.iter():
        if xml_local_name(elem.tag) != "SimpleData":
            continue

        key = elem.attrib.get("name")
        value = elem.text.strip() if elem.text else None

        if key:
            data[key] = value

    return data


def find_first_text_by_local_name(parent, local_name):
    """
    Find the first descendant with a given local tag name.
    Namespace-agnostic.
    """
    for elem in parent.iter():
        if xml_local_name(elem.tag) == local_name:
            return elem.text.strip() if elem.text else None

    return None


def direction_to_bearing(direction):
    """
    Convert KML direction strings into compass bearings.

    Bearing convention:
      0   = north
      90  = east
      180 = south
      270 = west
    """
    if direction is None:
        return None

    direction = str(direction).strip().lower()

    direction_map = {
        "north": 0.0,
        "n": 0.0,
        "northbound": 0.0,
        "nb": 0.0,

        "northeast": 45.0,
        "north east": 45.0,
        "ne": 45.0,

        "east": 90.0,
        "e": 90.0,
        "eastbound": 90.0,
        "eb": 90.0,

        "southeast": 135.0,
        "south east": 135.0,
        "se": 135.0,

        "south": 180.0,
        "s": 180.0,
        "southbound": 180.0,
        "sb": 180.0,

        "southwest": 225.0,
        "south west": 225.0,
        "sw": 225.0,

        "west": 270.0,
        "w": 270.0,
        "westbound": 270.0,
        "wb": 270.0,

        "northwest": 315.0,
        "north west": 315.0,
        "nw": 315.0,
    }

    return direction_map.get(direction)


def extract_image_slug_from_url(url):
    """
    Example:
      https://cwwp2.dot.ca.gov/data/d7/cctv/image/i525meadowdale/i525meadowdale.jpg

    Returns:
      i525meadowdale
    """
    if not url:
        return None

    match = re.search(r"/image/([^/]+)/", url)
    if match:
        return match.group(1)

    return None


def parse_cctv_kml(kml_path, include_only_in_service=True):
    """
    Parse CCTV KML using ExtendedData fields.

    Expected KML fields:
      locationName
      longitude
      latitude
      direction
      inService
      currentImageURL

    Returns
    -------
    cctv_lookup:
        {
            normalized_key: {
                "name": ...,
                "lat": ...,
                "lon": ...,
                "direction": ...,
                "bearing": ...,
                "image_slug": ...,
                "current_image_url": ...,
                ...
            }
        }

    entries:
        list of parsed KML entries
    """
    kml_path = Path(kml_path)

    if not kml_path.exists():
        raise FileNotFoundError(f"KML file does not exist: {kml_path}")

    tree = ET.parse(kml_path)
    root = tree.getroot()

    placemarks = [
        elem for elem in root.iter()
        if xml_local_name(elem.tag) == "Placemark"
    ]

    entries = []
    cctv_lookup = {}

    for placemark in placemarks:
        simple_data = get_simple_data_dict(placemark)

        location_name = simple_data.get("locationName")

        # Fallback to <name> if locationName is absent.
        if not location_name:
            location_name = find_first_text_by_local_name(placemark, "name")

        if not location_name:
            continue

        in_service = simple_data.get("inService")

        if include_only_in_service and in_service is not None:
            if str(in_service).strip().lower() != "true":
                continue

        # Prefer SimpleData lon/lat.
        lon = simple_data.get("longitude")
        lat = simple_data.get("latitude")

        # Fallback to Point/coordinates.
        if lon is None or lat is None:
            coordinates = find_first_text_by_local_name(placemark, "coordinates")

            if not coordinates:
                continue

            coord_parts = [x.strip() for x in coordinates.split(",")]

            if len(coord_parts) < 2:
                continue

            lon = coord_parts[0]
            lat = coord_parts[1]

        try:
            lon = float(lon)
            lat = float(lat)
        except ValueError:
            continue

        direction = simple_data.get("direction")
        bearing = direction_to_bearing(direction)

        current_image_url = simple_data.get("currentImageURL")
        streaming_video_url = simple_data.get("streamingVideoURL")

        image_slug = extract_image_slug_from_url(current_image_url)

        entry = {
            "name": location_name,
            "lat": lat,
            "lon": lon,
            "direction": direction,
            "bearing": bearing,
            "image_slug": image_slug,
            "current_image_url": current_image_url,
            "streaming_video_url": streaming_video_url,
            "district": simple_data.get("district"),
            "county": simple_data.get("county"),
            "route": simple_data.get("route"),
            "postmile": simple_data.get("postmile"),
            "index": simple_data.get("index_"),
            "in_service": in_service,
            "kml_path": str(kml_path),
        }

        entries.append(entry)

        # Build multiple keys so folder names can match even with different formatting.
        keys = {
            normalize_camera_key(location_name),
        }

        if image_slug:
            keys.add(normalize_camera_key(image_slug))
            keys.add(image_slug.lower())

        # For your example:
        #   locationName: "I-5 : (25) Meadowdale"
        #   folder:       "I-5 - (25) Meadowdale"
        #
        # Both normalize to:
        #   i525meadowdale
        #
        # Also add route + index + nearbyPlace style keys when available.
        route = simple_data.get("route")
        index = simple_data.get("index_")
        nearby_place = simple_data.get("nearbyPlace")

        if route and index and nearby_place:
            keys.add(normalize_camera_key(f"{route} - ({index}) {nearby_place}"))
            keys.add(normalize_camera_key(f"{route} : ({index}) {nearby_place}"))

        if route and index:
            keys.add(normalize_camera_key(f"{route} - ({index})"))
            keys.add(normalize_camera_key(f"{route} : ({index})"))

        for key in keys:
            cctv_lookup[key] = entry

    return cctv_lookup, entries


def resolve_cctv_date_folder(extracted_path, date_str):
    """
    Supports either:
      extracted_path = "temp/cctv"
    or:
      extracted_path = "temp/cctv/20251001"
    """
    extracted_path = Path(extracted_path)

    if extracted_path.name == date_str:
        date_folder = extracted_path
    else:
        date_folder = extracted_path / date_str

    if not date_folder.exists():
        raise FileNotFoundError(f"Date folder does not exist: {date_folder}")

    if not date_folder.is_dir():
        raise ValueError(f"Expected date folder to be a directory: {date_folder}")

    return date_folder


def retrieve_by_date_cctv(
    extracted_path,
    date_str,
    kml_path="sensor_locations/cctv.kml",
    fov_deg=60.0,
    distance_km=1.0,
    cell_km=0.1,
    unknown_direction_mode="skip",
    unknown_direction_radius_km=0.25,
    include_only_in_service=True,
    return_camera_map=False,
):
    """
    Create CCTV spatial coverage polygons for a date.

    Expected folder structure:
        temp/cctv/20251001/<camera_name>/*.jpg

    KML provides:
        - locationName
        - latitude
        - longitude
        - direction

    Example folder/KML mismatch handled:
        Folder: "I-5 - (25) Meadowdale"
        KML:    "I-5 : (25) Meadowdale"

    Both normalize to:
        "i525meadowdale"

    Parameters
    ----------
    extracted_path : str or Path
        CCTV root folder or date folder.
        Examples:
            "temp/cctv"
            "temp/cctv/20251001"

    date_str : str
        Date string like "20251001".

    kml_path : str or Path
        Path to cctv.kml.

    fov_deg : float
        Assumed camera field of view.

    distance_km : float
        Assumed visible distance for traffic CCTV.

    cell_km : float
        Polygon arc resolution.

    unknown_direction_mode : str
        What to do if KML direction is missing/unparseable:
          - "skip": skip camera
          - "circle": use small circular coverage around camera

    unknown_direction_radius_km : float
        Radius used when unknown_direction_mode="circle".

    include_only_in_service : bool
        If True, skip KML entries where inService is not True.

    return_camera_map : bool
        If False, return only list of polygons.
        If True, return:
            polygons, cctv_camera_to_polygon, metadata

    Returns
    -------
    list[Polygon]
        One polygon per active CCTV camera.

    Or, if return_camera_map=True:
        polygons, cctv_camera_to_polygon, metadata
    """
    date_folder = resolve_cctv_date_folder(extracted_path, date_str)

    cctv_lookup, kml_entries = parse_cctv_kml(
        kml_path=kml_path,
        include_only_in_service=include_only_in_service,
    )

    cctv_camera_to_polygon = {}

    metadata = {
        "date_str": date_str,
        "date_folder": str(date_folder),
        "kml_path": str(kml_path),
        "fov_deg": fov_deg,
        "distance_km": distance_km,
        "cell_km": cell_km,
        "unknown_direction_mode": unknown_direction_mode,
        "unknown_direction_radius_km": unknown_direction_radius_km,
        "include_only_in_service": include_only_in_service,
        "num_kml_entries": len(kml_entries),
        "num_camera_folders": 0,
        "num_matched_cameras": 0,
        "num_unmatched_camera_folders": 0,
        "num_unknown_direction": 0,
        "num_skipped_unknown_direction": 0,
        "num_circle_fallbacks": 0,
        "num_polygons": 0,
        "unmatched_camera_folders": [],
        "unknown_direction_cameras": [],
    }

    for camera_folder in sorted(date_folder.iterdir()):
        if not camera_folder.is_dir():
            continue

        metadata["num_camera_folders"] += 1

        folder_name = camera_folder.name
        folder_key = normalize_camera_key(folder_name)

        entry = cctv_lookup.get(folder_key)

        if entry is None:
            metadata["num_unmatched_camera_folders"] += 1
            metadata["unmatched_camera_folders"].append({
                "folder_name": folder_name,
                "folder_path": str(camera_folder),
                "normalized_key": folder_key,
            })
            continue

        metadata["num_matched_cameras"] += 1

        lat = entry["lat"]
        lon = entry["lon"]
        bearing = entry["bearing"]

        if bearing is None:
            metadata["num_unknown_direction"] += 1
            metadata["unknown_direction_cameras"].append({
                "folder_name": folder_name,
                "kml_name": entry.get("name"),
                "direction": entry.get("direction"),
                "lat": lat,
                "lon": lon,
            })

            if unknown_direction_mode == "skip":
                metadata["num_skipped_unknown_direction"] += 1
                continue

            elif unknown_direction_mode == "circle":
                metadata["num_circle_fallbacks"] += 1

                polygon = make_circle(
                    lat=lat,
                    lon=lon,
                    radius_km=unknown_direction_radius_km,
                    cell_km=cell_km,
                )

                coverage_type = "circle_fallback"

            else:
                raise ValueError(
                    f"Unknown unknown_direction_mode: {unknown_direction_mode}"
                )

        else:
            polygon = make_cone(
                lat=lat,
                lon=lon,
                bearing=bearing,
                fov_deg=fov_deg,
                distance_km=distance_km,
                cell_km=cell_km,
            )

            coverage_type = "cone"

        cctv_camera_to_polygon[folder_name] = {
            "folder_name": folder_name,
            "kml_name": entry.get("name"),
            "lat": lat,
            "lon": lon,
            "direction": entry.get("direction"),
            "bearing": bearing,
            "fov_deg": fov_deg,
            "distance_km": distance_km,
            "coverage_type": coverage_type,
            "image_slug": entry.get("image_slug"),
            "current_image_url": entry.get("current_image_url"),
            "streaming_video_url": entry.get("streaming_video_url"),
            "district": entry.get("district"),
            "county": entry.get("county"),
            "route": entry.get("route"),
            "postmile": entry.get("postmile"),
            "polygon": polygon,
        }

    polygons = [
        entry["polygon"]
        for entry in cctv_camera_to_polygon.values()
    ]

    metadata["num_polygons"] = len(polygons)

    if return_camera_map:
        return polygons, cctv_camera_to_polygon, metadata

    return polygons


def date_str_to_traffic_file_date(date_str):
    """
    Convert:
        20250203
    into:
        2025_02_03
    """
    if len(date_str) != 8:
        raise ValueError(f"Expected date_str like YYYYMMDD, got: {date_str}")

    return f"{date_str[:4]}_{date_str[4:6]}_{date_str[6:8]}"


def resolve_traffic_date_folder(extracted_path, date_str):
    """
    Supports either:
      extracted_path = "temp/traffic"
    or:
      extracted_path = "temp/traffic/20250203"
    """
    extracted_path = Path(extracted_path)

    if extracted_path.name == date_str:
        date_folder = extracted_path
    else:
        date_folder = extracted_path / date_str

    if not date_folder.exists():
        raise FileNotFoundError(f"Date folder does not exist: {date_folder}")

    if not date_folder.is_dir():
        raise ValueError(f"Expected a directory, got: {date_folder}")

    return date_folder


def find_traffic_files_by_date(extracted_path, date_str):
    """
    Find traffic .txt.gz files for a date.

    Example file:
        d07_text_station_5min_2025_02_03.txt.gz

    Expected folder:
        temp/traffic/20250203/
    """
    date_folder = resolve_traffic_date_folder(extracted_path, date_str)
    file_date = date_str_to_traffic_file_date(date_str)

    gz_files = sorted(date_folder.rglob(f"*{file_date}*.txt.gz"))
    txt_files = sorted(date_folder.rglob(f"*{file_date}*.txt"))

    return gz_files, txt_files


def extract_gz_to_txt(gz_path, output_folder):
    """
    Extract one .txt.gz file into a .txt file.

    Example:
        d07_text_station_5min_2025_02_03.txt.gz
    becomes:
        d07_text_station_5min_2025_02_03.txt
    """
    gz_path = Path(gz_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if gz_path.suffix != ".gz":
        raise ValueError(f"Expected .gz file, got: {gz_path}")

    txt_filename = gz_path.name[:-3]  # remove ".gz"
    txt_path = output_folder / txt_filename

    with gzip.open(gz_path, "rb") as f_in:
        with open(txt_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    return txt_path


def read_traffic_txt_file(txt_path, max_rows=None):
    """
    Read an uncompressed traffic .txt file.

    Assumes comma-separated rows.
    Returns a list of rows, where each row is a list of strings.
    """
    rows = []

    with open(txt_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)

        for row_index, row in enumerate(reader):
            if max_rows is not None and row_index >= max_rows:
                break

            if not row:
                continue

            rows.append(row)

    return rows


def read_traffic_gz_file(gz_path, max_rows=None):
    """
    Read a compressed traffic .txt.gz file directly.

    Assumes comma-separated rows.
    Returns a list of rows, where each row is a list of strings.
    """
    rows = []

    with gzip.open(gz_path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)

        for row_index, row in tqdm(enumerate(reader)):
            if max_rows is not None and row_index >= max_rows:
                break

            if not row:
                continue

            rows.append(row)

    return rows

def date_str_to_traffic_file_date(date_str):
    """
    Convert:
        20250203
    into:
        2025_02_03
    """
    if len(date_str) != 8:
        raise ValueError(f"Expected date_str like YYYYMMDD, got: {date_str}")

    return f"{date_str[:4]}_{date_str[4:6]}_{date_str[6:8]}"


def resolve_traffic_date_folder(extracted_path, date_str):
    """
    Supports either:
      extracted_path = "evaluation/temp/pem_data_station_5min"
    or:
      extracted_path = "evaluation/temp/pem_data_station_5min/20250203"
    """
    extracted_path = Path(extracted_path)

    if extracted_path.name == date_str:
        date_folder = extracted_path
    else:
        date_folder = extracted_path / date_str

    if not date_folder.exists():
        raise FileNotFoundError(f"Date folder does not exist: {date_folder}")

    if not date_folder.is_dir():
        raise ValueError(f"Expected a directory, got: {date_folder}")

    return date_folder


def find_traffic_gz_files_by_date(extracted_path, date_str):
    """
    Find traffic .txt.gz files for a date.

    Example:
        d07_text_station_5min_2025_02_03.txt.gz
    """
    date_folder = resolve_traffic_date_folder(extracted_path, date_str)
    file_date = date_str_to_traffic_file_date(date_str)

    return sorted(date_folder.rglob(f"*{file_date}*.txt.gz"))


def load_pems_station_locations(stations_path):
    """
    Load PeMS station metadata.

    Expected tab-delimited file:

        ID  Fwy  Dir  District  County  City  State_PM  Abs_PM
        Latitude  Longitude  Length  Type  Lanes  Name ...

    Returns
    -------
    dict
        {
            "715898": {
                "station_id": "715898",
                "lat": 33.880069,
                "lon": -118.021261,
                "fwy": "5",
                "direction": "N",
                ...
            }
        }
    """
    stations_path = Path(stations_path)

    if not stations_path.exists():
        raise FileNotFoundError(f"Stations file does not exist: {stations_path}")

    station_locations = {}
    invalid_rows = []

    with open(stations_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row_index, row in enumerate(reader):
            station_id = str(row.get("ID", "")).strip()

            if not station_id:
                invalid_rows.append({
                    "row_index": row_index,
                    "row": row,
                    "reason": "missing station ID",
                })
                continue

            try:
                lat = float(row["Latitude"])
                lon = float(row["Longitude"])
            except Exception as e:
                invalid_rows.append({
                    "row_index": row_index,
                    "station_id": station_id,
                    "row": row,
                    "reason": f"invalid lat/lon: {repr(e)}",
                })
                continue

            station_locations[station_id] = {
                "station_id": station_id,
                "lat": lat,
                "lon": lon,
                "fwy": row.get("Fwy"),
                "direction": row.get("Dir"),
                "district": row.get("District"),
                "county": row.get("County"),
                "city": row.get("City"),
                "state_pm": row.get("State_PM"),
                "abs_pm": row.get("Abs_PM"),
                "length": row.get("Length"),
                "type": row.get("Type"),
                "lanes": row.get("Lanes"),
                "name": row.get("Name"),
            }

    return station_locations, invalid_rows


def iter_traffic_rows_from_gz(gz_path):
    """
    Stream rows from a PeMS .txt.gz file.

    Each yielded item:
        row_index, row

    Example row:
        [
          '02/03/2025 00:00:00',
          '715989',
          '7',
          '5',
          'S',
          ...
        ]
    """
    with gzip.open(gz_path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)

        for row_index, row in enumerate(reader):
            if not row:
                continue

            yield row_index, row


def retrieve_by_date_traffic_coverage(
    extracted_path,
    date_str,
    stations_path="sensor_location/pems_y_stations.txt",
    radius_km=0.25,
    cell_km=0.05,
    return_sensor_map=False,
):
    """
    Get spatial coverage polygons for PeMS traffic sensors active on a date.

    This function:
      1. Loads all station locations from pems_y_stations.txt.
      2. Streams through the date's traffic .txt.gz file(s).
      3. For each row, checks row[1] as the station ID.
      4. If the station is in the station metadata file and has not already
         been processed, creates a circular coverage polygon.
      5. Stops early once all stations from the station metadata file have
         appeared in the traffic rows.

    Parameters
    ----------
    extracted_path : str or Path
        Root folder or date folder.
        Example:
            "evaluation/temp/pem_data_station_5min"
            "evaluation/temp/pem_data_station_5min/20250203"

    date_str : str
        Date string like "20250203".

    stations_path : str or Path
        Path to PeMS station metadata file.

    radius_km : float
        Spatial coverage radius around each station.
        Default 0.25 km.

    cell_km : float
        Polygon boundary resolution.

    return_sensor_map : bool
        If False, return only list of polygons.
        If True, return:
            polygons, station_id_to_polygon, metadata

    Returns
    -------
    list[Polygon]
        One polygon per unique station found in both the traffic data and
        the station metadata file.

    Or, if return_sensor_map=True:
        polygons, station_id_to_polygon, metadata
    """
    station_locations, invalid_station_rows = load_pems_station_locations(
        stations_path
    )

    target_station_ids = set(station_locations.keys())

    gz_files = find_traffic_gz_files_by_date(
        extracted_path=extracted_path,
        date_str=date_str,
    )

    if not gz_files:
        raise FileNotFoundError(
            f"No traffic .txt.gz files found for date {date_str} under {extracted_path}"
        )

    station_id_to_polygon = {}
    station_id_to_info = {}

    found_station_ids = set()

    metadata = {
        "date_str": date_str,
        "extracted_path": str(extracted_path),
        "stations_path": str(stations_path),
        "radius_km": radius_km,
        "cell_km": cell_km,
        "num_station_metadata_rows": len(station_locations),
        "num_invalid_station_rows": len(invalid_station_rows),
        "invalid_station_rows": invalid_station_rows,
        "num_gz_files": len(gz_files),
        "gz_files": [str(path) for path in gz_files],
        "num_rows_read": 0,
        "num_invalid_traffic_rows": 0,
        "num_rows_with_station_metadata": 0,
        "num_rows_without_station_metadata": 0,
        "num_duplicate_station_rows": 0,
        "stopped_early": False,
        "invalid_traffic_rows": [],
    }

    for gz_path in gz_files:
        for row_index, row in iter_traffic_rows_from_gz(gz_path):
            metadata["num_rows_read"] += 1

            # PeMS station ID is row[1].
            if len(row) < 2:
                metadata["num_invalid_traffic_rows"] += 1
                metadata["invalid_traffic_rows"].append({
                    "source_file": str(gz_path),
                    "row_index": row_index,
                    "row": row,
                    "reason": "expected station ID at row[1]",
                })
                continue

            station_id = str(row[1]).strip()

            if station_id not in station_locations:
                metadata["num_rows_without_station_metadata"] += 1
                continue

            metadata["num_rows_with_station_metadata"] += 1

            if station_id in station_id_to_polygon:
                metadata["num_duplicate_station_rows"] += 1
                continue

            station = station_locations[station_id]
            lat = station["lat"]
            lon = station["lon"]

            polygon = make_circle(
                lat=lat,
                lon=lon,
                radius_km=radius_km,
                cell_km=cell_km,
            )

            station_id_to_polygon[station_id] = polygon
            station_id_to_info[station_id] = {
                **station,
                "first_seen_source_file": str(gz_path),
                "first_seen_row_index": row_index,
                "coverage_radius_km": radius_km,
            }

            found_station_ids.add(station_id)

            # Stop once every station in the station metadata file has appeared.
            if found_station_ids == target_station_ids:
                metadata["stopped_early"] = True
                break

        if metadata["stopped_early"]:
            break

    polygons = list(station_id_to_polygon.values())

    missing_station_ids = sorted(target_station_ids - found_station_ids)

    metadata["num_unique_stations_found"] = len(found_station_ids)
    metadata["num_polygons"] = len(polygons)
    metadata["num_missing_station_ids"] = len(missing_station_ids)
    metadata["missing_station_ids"] = missing_station_ids

    if return_sensor_map:
        return polygons, station_id_to_polygon, station_id_to_info, metadata

    return polygons

def is_na(value):
    return value is None or normalize_text(value) in {"", "n/a", "na", "none"}


def resolve_weather_date_folder(extracted_path, date_str):
    """
    Supports either:
      extracted_path = "evaluation/temp/weather"
    or:
      extracted_path = "evaluation/temp/weather/20251001"
    """
    extracted_path = Path(extracted_path)

    if extracted_path.name == date_str:
        date_folder = extracted_path
    else:
        date_folder = extracted_path / date_str

    if not date_folder.exists():
        raise FileNotFoundError(f"Date folder does not exist: {date_folder}")

    if not date_folder.is_dir():
        raise ValueError(f"Expected date folder to be a directory: {date_folder}")

    return date_folder


def maybe_inject_california(location_name):
    """
    Convert location strings like:
        Pasadena, US
    into:
        Pasadena, California, US

    Leaves more specific locations unchanged:
        Pasadena, California, US
        Los Angeles County, California, US
        Santa Monica, CA, US
    """
    location_name = str(location_name).strip()

    parts = [
        part.strip()
        for part in location_name.split(",")
        if part.strip()
    ]

    if len(parts) < 2:
        return location_name

    last = parts[-1].lower()

    # Only adjust locations that end with US/USA/United States.
    if last not in {"us", "usa", "united states", "united states of america"}:
        return location_name

    middle_parts = [part.lower() for part in parts[1:-1]]

    already_has_california = any(
        part in {"california", "ca"}
        for part in middle_parts
    )

    if already_has_california:
        return location_name

    # Example:
    #   ["Pasadena", "US"]
    # becomes:
    #   ["Pasadena", "California", "US"]
    return ", ".join(parts[:-1] + ["California", parts[-1]])


def get_weather_location_names(date_folder):
    """
    Weather data structure:
        date/location_name/

    Example:
        20251001/Pasadena, US/
        20251001/Los Angeles, US/

    Returns folder names as location names.
    """
    date_folder = Path(date_folder)

    location_names = []

    for child in sorted(date_folder.iterdir()):
        if child.is_dir():
            location_names.append(child.name)

    return location_names


def retrieve_by_date_weather(
    extracted_path,
    date_str,
    gm=None,
    inject_california=True,
    force_refresh=False,
    return_location_map=False,
):
    """
    Retrieve spatial coverage polygons for weather data on a date.

    Weather data is organized as:
        extracted_path/date_str/location_name/

    Example:
        evaluation/temp/weather/20251001/Pasadena, US/

    This function:
      1. Finds all location_name folders for the date.
      2. Optionally rewrites "Pasadena, US" as "Pasadena, California, US".
      3. Calls gm.get_geo_region(location_query).
      4. Returns one geometry per successfully geocoded location.

    Parameters
    ----------
    extracted_path : str or Path
        Weather root folder or date folder.
        Examples:
            "evaluation/temp/weather"
            "evaluation/temp/weather/20251001"

    date_str : str
        Date string like "20251001".

    gm : GeoManager or None
        Existing GeoManager instance. If None, creates one.

    inject_california : bool
        If True, turns "Pasadena, US" into "Pasadena, California, US".

    force_refresh : bool
        Passed to gm.get_geo_region(...).

    return_location_map : bool
        If False, return only list of geometries.
        If True, return:
            polygons, weather_location_to_polygon, metadata

    Returns
    -------
    list[shapely geometry]
        One region geometry per weather location.

    Or, if return_location_map=True:
        polygons, weather_location_to_polygon, metadata
    """
    if gm is None:
        from evaluation.geo_manager import GeoManager
        gm = GeoManager()

    date_folder = resolve_weather_date_folder(extracted_path, date_str)

    location_names = get_weather_location_names(date_folder)

    weather_location_to_polygon = {}

    metadata = {
        "date_str": date_str,
        "date_folder": str(date_folder),
        "num_location_folders": len(location_names),
        "num_successful_geocodes": 0,
        "num_failed_geocodes": 0,
        "num_duplicate_queries": 0,
        "inject_california": inject_california,
        "failed_locations": [],
        "duplicate_queries": [],
    }

    seen_queries = set()

    for location_name in location_names:
        if is_na(location_name):
            continue

        location_query = (
            maybe_inject_california(location_name)
            if inject_california
            else location_name
        )

        query_key = normalize_text(location_query)

        if query_key in seen_queries:
            metadata["num_duplicate_queries"] += 1
            metadata["duplicate_queries"].append({
                "location_name": location_name,
                "location_query": location_query,
            })
            continue

        seen_queries.add(query_key)

        record = gm.get_geo_region(
            location_query,
            force_refresh=force_refresh,
        )

        if record.get("status") != "ok" or record.get("geometry") is None:
            metadata["num_failed_geocodes"] += 1
            metadata["failed_locations"].append({
                "location_name": location_name,
                "location_query": location_query,
                "record": {
                    k: v
                    for k, v in record.items()
                    if k != "geometry"
                },
            })
            continue

        geometry = record["geometry"]

        weather_location_to_polygon[location_name] = {
            "location_name": location_name,
            "location_query": location_query,
            "geometry": geometry,
            "geometry_type": record.get("geometry_type"),
            "geometry_wkt": record.get("geometry_wkt"),
            "source": record.get("source"),
            "method": record.get("method"),
            "cache_hit": record.get("cache_hit"),
            "fallback_used": record.get("fallback_used"),
        }

        metadata["num_successful_geocodes"] += 1

    polygons = [
        entry["geometry"]
        for entry in weather_location_to_polygon.values()
    ]

    metadata["num_polygons"] = len(polygons)

    if return_location_map:
        return polygons, weather_location_to_polygon, metadata

    return polygons


# Example usage
if __name__ == "__main__":


    paths = get_config().get("paths", {})
    temp_folder = paths.get("evaluation_temp_root", "evaluation/temp")
    data_folder = paths.get("raw_archive_root", "./raw_data")
    data_type = "weather_data"
    date_str = "20260203"
    cell_km = 0.25

    gm = GeoManager()

    extracted_path = untar_date_file(
        date_str=date_str,
        data_folder=data_folder + data_type,
        temp_folder=temp_folder,
        clear_existing=True,
    )

    # # Get polygons for air data at a particular data
    # polygons, sensor_id_to_polygon, metadata = retrieve_by_date_air(
    #     extracted_path=extracted_path,
    #     date_str=date_str,
    #     radius_km=1.0,
    #     cell_km=cell_km,
    #     return_sensor_map=True,
    # )

    # alert california
    #  Sometimes it is 88.87944304619052 in a .direction file
    #  Sometimes it is 33.9639,-118.2898,140.61970935770947 with a .location file.
    # camera_locs, conflicts = get_alertcalifornia_camera_locs(
    #     root_folder=temp_folder + "/" + data_type,
    #     date_str="20251001",
    # )
    # output_path = save_alertcalifornia_camera_locs(camera_locs, output_folder="evaluation/sensor_locations",
    # filename="alertcalifornia_locations.json",)
    

    # Get polygons for alertcalifornia cameras
    # polygons, camera_direction_to_polygon, metadata = retrieve_by_date_alertcalifornia(
    #     extracted_path=extracted_path,
    #     date_str=date_str,
    #     locations_path="evaluation/sensor_locations/alertcalifornia_locations.json",
    #     fov_deg=60.0,
    #     distance_km=50.0,
    #     cell_km=cell_km,
    #     direction_is_svg_angle=True,
    #     return_camera_map=True,
    # )

    # # Get polygons for cctv
    # polygons, cctv_camera_to_polygon, metadata = retrieve_by_date_cctv(
    #     extracted_path=extracted_path,
    #     date_str=date_str,
    #     kml_path="evaluation/sensor_locations/cctv.kml",
    #     fov_deg=60.0,
    #     distance_km=1.0,
    #     cell_km=0.25,
    #     unknown_direction_mode="circle",
    #     return_camera_map=True,
    # )

    # Get polygons for traffic
    # polygons, station_id_to_polygon, station_id_to_info, metadata = (
    #     retrieve_by_date_traffic_coverage(
    #         extracted_path=extracted_path,
    #         date_str=date_str,
    #         stations_path="evaluation/sensor_locations/pem_7_stations.txt",
    #         radius_km=0.25,
    #         cell_km=0.25,
    #         return_sensor_map=True,
    #     )
    # )

    # Get polygons for weather
    # polygons, weather_location_to_polygon, metadata = retrieve_by_date_weather(
    #     extracted_path=extracted_path,
    #     date_str=date_str,
    #     gm=gm,
    #     inject_california=True,
    #     return_location_map=True,
    # )


    # print("Polygons:", len(polygons))
    # print("Extracted to:", extracted_path)




# Two types of sensors:
#   - sensors whose location varies (alert california)
#   - sensors whose location is fixed (everything else)

#  Get a retrieval function by date for each sensor
#   For fixed sensors, check each sensor and its coverage
#     This may require going back to the locations file for it
#     Then mapping it to a coverage
#   For moving sensors, we'd probably have to aggregate coverage the whole day
#     and see if there's overlap.

# REMEMBER TO ITERATE THROUGH INCIDENTS REVERSE THROUGH TIME
#   THIS IS BECAUSE YOUR ALERTCALIFORNIA CAMERA DATA IS ONLY SAVED LATER AND NEEDS TO BE CACHED

#  Then iterate through each incident, and get the spatial area
#   identify the RELEVANT sensors which may get info about this incident
#   Iterate through all RELEVANT sensors to see if there's overlap with spatial area (and at this date/time)
#   Side note - twitter and citizen are always 'relevant', since they don't have geo information.  I'm ignoring PeMs CHP incidents since it's a bit unfair
