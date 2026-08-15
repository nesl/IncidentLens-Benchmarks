"""Read immutable sensor metadata without importing a live extractor."""

from pathlib import Path
from typing import Dict, List
import xml.etree.ElementTree as ET


def load_cctv_sensor_locations(path: Path | None = None) -> Dict[str, List[float]]:
    if path is None:
        path = Path(__file__).resolve().parent / "data" / "cctv.kml"
    tree = ET.parse(path)
    namespace = {"kml": "http://www.opengis.net/kml/2.2"}
    locations: Dict[str, List[float]] = {}
    for placemark in tree.getroot().findall(".//kml:Placemark", namespace):
        name = placemark.find("kml:name", namespace)
        coords = placemark.find(".//kml:coordinates", namespace)
        if name is None or coords is None or not name.text or not coords.text:
            continue
        longitude, latitude, *_ = coords.text.strip().split(",")
        locations[name.text.strip()] = [float(latitude), float(longitude)]
    return locations
