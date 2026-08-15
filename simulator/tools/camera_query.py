import os
from utilities.util import get_config

from simulator.sensor_catalog import load_cctv_sensor_locations
import math
from PIL import Image
import numpy as np
import hashlib

from datetime import datetime

import heapq
from functools import lru_cache
import time

def _timing_enabled():
    return os.environ.get("SIMULATOR_TIMING", "true").strip().lower() not in {"0", "false", "no", "off"}


def _print_timing(label, start):
    if _timing_enabled():
        print(f"[TIMER] {label}: {time.perf_counter() - start:.2f}s")


def _resolve_unavailable_image_path():
    """Find the CCTV unavailable placeholder robustly across working directories."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "unavailable.jpg"),
        os.path.join(os.getcwd(), "simulator", "tools", "unavailable.jpg"),
        "./simulator/tools/unavailable.jpg",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    print(
        "WARNING: simulator/tools/unavailable.jpg was not found; "
        "CCTV unavailable-image filtering is disabled instead of dropping every CCTV frame."
    )
    return None


def get_closest_locations(lat, lon, data_dict, k=4):
    return heapq.nsmallest(
        k,
        (
            ((lat2, lon2), value, haversine(lat, lon, lat2, lon2))
            for coords, value in data_dict.items()
            for lat2, lon2 in [tuple(coords)]
        ),
        key=lambda x: x[2]
    )


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius (km)

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# merge dictionaries from alertcalifornia and cctv
def merge_location_dicts(d1, d2):
    merged = dict(d1)

    for k, v in d2.items():
        if k in merged:
            merged[k] += v   # or custom logic
        else:
            merged[k] = v

    return merged

def in_time_window(filename):
    ts = filename[:14]  # grab timestamp part
    dt = datetime.strptime(ts, "%Y%m%d%H%M%S")

    return 10 <= dt.hour < 14   # 10:00–13:59 (2pm exclusive)

def get_compass_direction(angle):

    # Note - the angle from alertcalfornia is math angle (0 is east)

    bearing = (90 - angle) % 360
    dirs = ["North","Northeast","East","Southeast","South","Southwest","West","Northwest"]
    idx = round(bearing / 45) % 8
    return "Facing " + dirs[idx]


def get_alertcalifornia_sensor_locations():
    start = time.perf_counter()
    
    # Open the config file
    config = get_config()
    save_folder = config["save_folder"]

    alertcalifornia_folder = os.path.join(save_folder, "alertcalifornia")
    if not os.path.isdir(alertcalifornia_folder):
        print(f"WARNING: ALERTCalifornia folder not found: {alertcalifornia_folder}")
        return {}

    # Get the latest folder in the alertcalifornia folder
    latest_folder = sorted(os.listdir(alertcalifornia_folder))[-1]
    full_latest_folder = os.path.join(alertcalifornia_folder, latest_folder)

    camera_location_dict = {} # latlong -> [camera id, direction]

    # Iterate through every camera, and get the location + direction
    for camera_folder in os.listdir(full_latest_folder):
        camera_folder_path = os.path.join(full_latest_folder, camera_folder)
        
        for file in os.listdir(camera_folder_path):
            if file.endswith(".location"):

                # Get the location file and read it
                location_file = os.path.join(camera_folder_path, file)
                with open(location_file, "r") as f:
                    location = f.read().strip()
                    location = location.split(",")
                    lat = float(location[0])
                    long = float(location[1])
                    direction = get_compass_direction(float(location[2]))

                    image_name = file.split(".")[0] + ".jpg"
                    if in_time_window(image_name):
                        
                        image_path = os.path.join(camera_folder_path, image_name) 

                        dict_key = (lat, long)

                        if dict_key in camera_location_dict:

                            # One check - do not add if the direction has already been included
                            if direction not in [x[1] for x in camera_location_dict[dict_key]]:
                                camera_location_dict[dict_key].append([image_path, direction])
                        else:
                            camera_location_dict[dict_key] = [[image_path, direction]]
            
    _print_timing(f"build ALERTCalifornia camera index ({len(camera_location_dict)} locations)", start)
    return camera_location_dict


@lru_cache(maxsize=16)
def _image_array_for_similarity(path):
    with Image.open(path) as img:
        img = img.convert("RGB")
        return img.size, np.array(img)


def images_similar(path1, path2, match_threshold=0.8):
    # The unavailable placeholder is compared against many CCTV frames during
    # the first camera-index build. Cache decoded arrays so we do not re-open
    # the placeholder for every camera image.
    if not path1 or not os.path.exists(path1):
        # Important: missing placeholder should not make every CCTV frame look
        # unavailable. Skip the unavailable-image filter instead.
        return False

    try:
        size1, arr1 = _image_array_for_similarity(path1)
    except Exception:
        return False

    try:
        with Image.open(path2) as img2:
            img2 = img2.convert("RGB")
            if img2.size != size1:
                return False
            arr2 = np.array(img2)
    except Exception:
        # The candidate image itself is unreadable, so drop it.
        return True

    # Exact pixel match ratio
    matches = np.all(arr1 == arr2, axis=2)
    similarity = matches.mean()  # between 0 and 1
    return similarity >= match_threshold


def get_cctv_sensor_locations():
    start = time.perf_counter()


    config = get_config()
    save_folder = config["save_folder"]

    # Some CCTV images show 'unavailable' as the image, so filter out of simulation.
    # Resolve this robustly; if the placeholder is missing, do not drop all CCTV frames.
    unavailable_image = _resolve_unavailable_image_path()

    cctv_folder = os.path.join(save_folder, "cctv")
    if not os.path.isdir(cctv_folder):
        print(f"WARNING: CCTV folder not found: {cctv_folder}")
        return {}

    # Pick a date
    latest_folder = sorted(os.listdir(cctv_folder))[-1]
    full_latest_folder = os.path.join(cctv_folder, latest_folder)

    cam_entries = load_cctv_sensor_locations()
    # Have to reformat into lat/long -> [camera id]

    camera_location_dict = {}
    for sensor in cam_entries:  # Iterate through each camera name

        camera_folder = os.path.join(full_latest_folder, sensor)

        if os.path.isdir(camera_folder):
            for file in os.listdir(camera_folder):

                image_path = os.path.join(camera_folder, file)

                if file.endswith(".jpg") and in_time_window(file) and not images_similar(unavailable_image, image_path):

                    latlong = cam_entries[sensor]

                    dict_key = tuple(latlong)

                    if dict_key not in camera_location_dict:
                        camera_location_dict[dict_key] = [[image_path]]
                            
    
    _print_timing(f"build CCTV camera index ({len(camera_location_dict)} locations)", start)
    return camera_location_dict



@lru_cache(maxsize=1)
def get_alertcalifornia_sensor_locations_cached():
    return get_alertcalifornia_sensor_locations()


@lru_cache(maxsize=1)
def get_cctv_sensor_locations_cached():
    return get_cctv_sensor_locations()


def clear_camera_location_cache():
    """Clear cached camera dictionaries, useful after refreshing pulled_data."""
    get_alertcalifornia_sensor_locations_cached.cache_clear()
    get_cctv_sensor_locations_cached.cache_clear()


def get_closest_cameras(lat, lon, limit=5):
    start = time.perf_counter()

     # Obtain all sensor locations once per process instead of rescanning folders each step.
    alert_california_locations = get_alertcalifornia_sensor_locations_cached()
    cctv_locations = get_cctv_sensor_locations_cached()
    
    # Merge location dictionaries
    # camera_locations = merge_location_dicts(alert_california_locations, cctv_locations)


    # Do a query for both alertcalifornia and cctv
    closest_cctv = get_closest_locations(lat, lon, cctv_locations, k=limit)

    closest_alert_california = get_closest_locations(lat, lon, alert_california_locations, k=limit)

    # This data needs to be organized into [ (latlong, description/direction, origin, img_filepath) ]
    #  However, when we display to the LLM, it will not see the filepath.

    all_data = []  # Will have latlong, description/direction, origin, img_filepath
    llm_data = []  # Will have everything except the img_filepath, which is not visible to the LLM but will be used for retrieval later 

    for x in closest_cctv:
        for cam_data in x[1]:
            # Get the entry
            location = x[0]
            description = cam_data[0].split("/")[-2]
            filepath = cam_data[0]
            origin = "cctv"
            all_data.append((location, description, origin, filepath))
            llm_data.append((location, description, origin))
    
    for x in closest_alert_california:
        for cam_data in x[1]:
            # Get the entry
            location = x[0]
            description = cam_data[1]
            filepath = cam_data[0]
            origin = "alertcalifornia"
            all_data.append((location, description, origin, filepath))
            llm_data.append((location, description, origin))

    if not all_data:
        print(
            f"WARNING: get_closest_cameras found no usable camera images near "
            f"({float(lat):.5f}, {float(lon):.5f}). Check pulled_data/cctv, "
            "pulled_data/alertcalifornia, filename time windows, and unavailable-image filtering."
        )

    _print_timing(f"get closest cameras near ({float(lat):.5f}, {float(lon):.5f})", start)
    return all_data, llm_data




if __name__ == "__main__":

    


    # # Obtain all sensor locations
    # alert_california_locations = get_alertcalifornia_sensor_locations()
    # cctv_locations = get_cctv_sensor_locations()
    
    # # Merge location dictionaries
    # # camera_locations = merge_location_dicts(alert_california_locations, cctv_locations)


    # # Create a geomanager
    # geomanager = GeoManager()
    # query_loc = geomanager.geocode_name("The Santa Monica Mountains, Los Angeles County, California")

    # # Do a query for both alertcalifornia and cctv
    # closest_cctv = get_closest_locations(query_loc[0], query_loc[1], cctv_locations)

    # closest_alert_california = get_closest_locations(query_loc[0], query_loc[1], alert_california_locations)

    get_closest_cameras("Devil's Punchbowl")

