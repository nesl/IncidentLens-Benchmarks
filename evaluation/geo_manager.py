from utilities.util import get_config
import numpy as np

# Add geocoding
import googlemaps
import osmnx as ox

import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point, box, Polygon
from shapely import wkt
import math

def destination_point(lat, lon, bearing_deg, distance_km):
    R = 6371.0  # Earth radius in km

    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    bearing = math.radians(bearing_deg)

    d = distance_km / R

    lat2 = math.asin(
        math.sin(lat1)*math.cos(d) +
        math.cos(lat1)*math.sin(d)*math.cos(bearing)
    )

    lon2 = lon1 + math.atan2(
        math.sin(bearing)*math.sin(d)*math.cos(lat1),
        math.cos(d) - math.sin(lat1)*math.sin(lat2)
    )

    return math.degrees(lat2), math.degrees(lon2)
    
def make_cone(lat, lon, bearing, fov_deg, distance_km, cell_km=1):
    half = fov_deg / 2

    arc_length = 2 * math.pi * distance_km * (fov_deg / 360.0)

    steps = max(8, int(arc_length / cell_km))

    angles = [
        bearing - half + i * (fov_deg / steps)
        for i in range(steps + 1)
    ]

    arc_points = [
        destination_point(lat, lon, ang, distance_km)
        for ang in angles
    ]

    coords = (
        [(lon, lat)] +
        [(lon2, lat2) for lat2, lon2 in arc_points] +
        [(lon, lat)]
    )

    return Polygon(coords)

def make_circle(lat, lon, radius_km, cell_km=1):
    circumference = 2 * math.pi * radius_km

    # ensure enough resolution relative to grid size
    steps = max(12, int(circumference / cell_km))

    angles = [i * (360 / steps) for i in range(steps)]

    points = [
        destination_point(lat, lon, angle, radius_km)
        for angle in angles
    ]

    coords = [(lon2, lat2) for lat2, lon2 in points]

    return Polygon(coords)

# For visualization
# from detection.visualization.geo_tools import highlight_geo

class GeoManager:

    # -------------------------
    # Cache helpers
    # -------------------------

    def _normalize_geo_name(self, geo_name):
        return str(geo_name).strip().lower()

    def _load_geo_region_cache(self):
        if not self.cache_path.exists():
            return {}

        with open(self.cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_geo_region_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(
                self.geo_region_cache,
                f,
                indent=2,
                ensure_ascii=False,
            )

    def _cache_record(self, cache_key, record):
        """
        Store a JSON-serializable version of the record.
        The live Shapely geometry object is removed; geometry_wkt is kept.
        """
        serializable_record = {
            k: v
            for k, v in record.items()
            if k != "geometry"
        }

        self.geo_region_cache[cache_key] = serializable_record
        self._save_geo_region_cache()


    def _record_from_cache(self, cache_key):
        cached = self.geo_region_cache[cache_key].copy()

        if cached.get("geometry_wkt"):
            cached["geometry"] = wkt.loads(cached["geometry_wkt"])
        else:
            cached["geometry"] = None

        cached["cache_hit"] = True
        return cached

    # -------------------------
    # Geometry helpers
    # -------------------------

    def _make_region_record(
        self,
        *,
        location_name,
        geometry,
        source,
        method,
        extra=None,
    ):
        extra = extra or {}

        return {
            "status": "ok",
            "location_name": location_name,
            "source": source,
            "method": method,
            "geometry": geometry,
            "geometry_wkt": geometry.wkt,
            "geometry_type": geometry.geom_type,
            **extra,
        }

    def bbox_to_polygon(self, bbox):
        sw = bbox["southwest"]
        ne = bbox["northeast"]

        return box(
            sw["lng"],  # min longitude
            sw["lat"],  # min latitude
            ne["lng"],  # max longitude
            ne["lat"],  # max latitude
        )

    def point_buffer(self, lat, lon, meters=300):
        point = gpd.GeoSeries(
            [Point(lon, lat)],
            crs="EPSG:4326",
        )

        return (
            point
            .to_crs(epsg=3857)
            .buffer(meters)
            .to_crs(epsg=4326)
            .iloc[0]
        )

    # -------------------------
    # OSMnx region lookup
    # -------------------------

    def osmnx_get_geo(self, query):
        """
        Try to get a region geometry from OSMnx/Nominatim.

        Returns a common region record:
            {
                status,
                location_name,
                source,
                method,
                geometry,
                geometry_wkt,
                geometry_type,
                ...
            }

        If OSMnx cannot geocode the query, this function lets the error raise.
        get_geo_region catches that and falls back to Google.
        """
        gdf = ox.geocode_to_gdf(query)
        geom = gdf.geometry.iloc[0]

        if geom.is_empty:
            raise ValueError(f"OSMnx returned an empty geometry for {query!r}")

        if geom.geom_type in ["Polygon", "MultiPolygon"]:
            return self._make_region_record(
                location_name=query,
                geometry=geom,
                source="osmnx",
                method="osmnx_polygon_or_multipolygon",
                extra={
                    "cache_hit": False,
                },
            )

        gdf_proj = gdf.to_crs(epsg=3310)
        geom_proj = gdf_proj.geometry.iloc[0]

        if geom.geom_type == "Point":
            buffer_meters = 300
            buffered = geom_proj.buffer(buffer_meters)
            method = "osmnx_point_buffer"

        elif geom.geom_type in ["LineString", "MultiLineString"]:
            buffer_meters = 200
            buffered = geom_proj.buffer(buffer_meters)
            method = "osmnx_line_buffer"

        else:
            raise ValueError(
                f"Unsupported OSMnx geometry type for {query!r}: {geom.geom_type}"
            )

        buffered_gdf = gdf_proj.copy()
        buffered_gdf["geometry"] = [buffered]

        buffered_geom = buffered_gdf.to_crs(epsg=4326).geometry.iloc[0]

        return self._make_region_record(
            location_name=query,
            geometry=buffered_geom,
            source="osmnx",
            method=method,
            extra={
                "buffer_meters": buffer_meters,
                "original_geometry_type": geom.geom_type,
                "cache_hit": False,
            },
        )

    # -------------------------
    # Google fallback region lookup
    # -------------------------

    def google_get_geo(self, location_name):
        """
        Use Google Geocoding as an approximate area fallback.

        Safer rule:
        - POIs/businesses/addresses/intersections/routes -> point + buffer
        - true named regions -> bounds/viewport bbox
        """
        if self.gmaps_client is None:
            return {
                "status": "failed",
                "reason": "No Google Maps client was provided",
                "location_name": location_name,
            }

        geocode_result = self.gmaps_client.geocode(location_name)

        if not geocode_result:
            return {
                "status": "failed",
                "reason": "No Google geocoding result",
                "location_name": location_name,
            }

        result = geocode_result[0]

        geom = result["geometry"]
        types = set(result.get("types", []))
        location_type = geom.get("location_type")
        loc = geom["location"]

        base_extra = {
            "lat": loc["lat"],
            "lng": loc["lng"],
            "types": list(types),
            "location_type": location_type,
            "formatted_address": result.get("formatted_address"),
            "place_id": result.get("place_id"),
            "partial_match": result.get("partial_match", False),
            "cache_hit": False,
        }

        # -------------------------
        # 1. Precise addresses / buildings / POIs
        # -------------------------
        # Important: do this BEFORE viewport/bounds.
        if (
            location_type == "ROOFTOP"
            or "street_address" in types
            or "premise" in types
            or "subpremise" in types
        ):
            buffer_meters = 10
            polygon = self.point_buffer(loc["lat"], loc["lng"], meters=buffer_meters)
            method = "google_address_or_premise_buffer"

            return self._make_region_record(
                location_name=location_name,
                geometry=polygon,
                source="google",
                method=method,
                extra={
                    **base_extra,
                    "buffer_meters": buffer_meters,
                },
            )

        # Businesses / landmarks / stores
        if "establishment" in types or "point_of_interest" in types:
            buffer_meters = 25
            polygon = self.point_buffer(loc["lat"], loc["lng"], meters=buffer_meters)
            method = "google_poi_or_establishment_buffer"

            return self._make_region_record(
                location_name=location_name,
                geometry=polygon,
                source="google",
                method=method,
                extra={
                    **base_extra,
                    "buffer_meters": buffer_meters,
                },
            )

        # Intersections
        if "intersection" in types:
            buffer_meters = 25
            polygon = self.point_buffer(loc["lat"], loc["lng"], meters=buffer_meters)
            method = "google_intersection_buffer"

            return self._make_region_record(
                location_name=location_name,
                geometry=polygon,
                source="google",
                method=method,
                extra={
                    **base_extra,
                    "buffer_meters": buffer_meters,
                },
            )

        # Routes / roads
        if "route" in types:
            buffer_meters = 50
            polygon = self.point_buffer(loc["lat"], loc["lng"], meters=buffer_meters)
            method = "google_route_center_buffer"

            return self._make_region_record(
                location_name=location_name,
                geometry=polygon,
                source="google",
                method=method,
                extra={
                    **base_extra,
                    "buffer_meters": buffer_meters,
                },
            )

        # -------------------------
        # 2. True named regions
        # -------------------------
        region_types = {
            "locality",
            "neighborhood",
            "sublocality",
            "sublocality_level_1",
            "administrative_area_level_2",
            "administrative_area_level_1",
            "postal_code",
            "park",
            "airport",
        }

        is_region_like = bool(types & region_types)

        if is_region_like and "bounds" in geom:
            polygon = self.bbox_to_polygon(geom["bounds"])

            return self._make_region_record(
                location_name=location_name,
                geometry=polygon,
                source="google",
                method="google_bounds_bbox",
                extra=base_extra,
            )

        if is_region_like and "viewport" in geom:
            polygon = self.bbox_to_polygon(geom["viewport"])

            return self._make_region_record(
                location_name=location_name,
                geometry=polygon,
                source="google",
                method="google_viewport_bbox",
                extra=base_extra,
            )

        # -------------------------
        # 3. Default fallback
        # -------------------------
        buffer_meters = 50
        polygon = self.point_buffer(loc["lat"], loc["lng"], meters=buffer_meters)

        return self._make_region_record(
            location_name=location_name,
            geometry=polygon,
            source="google",
            method="google_default_point_buffer",
            extra={
                **base_extra,
                "buffer_meters": buffer_meters,
            },
        )

    # -------------------------
    # Main function
    # -------------------------

    def get_geo_region(self, location_name, force_refresh=False):
        """
        Main entry point.

        1. Check file-backed cache.
        2. Try OSMnx first.
        3. If OSMnx fails, fall back to Google.
        4. Save successful OR fully failed result to cache file.
        5. Return common region record.

        Use force_refresh=True to retry locations that were previously cached as failed.
        """
        cache_key = self._normalize_geo_name(location_name)

        if not force_refresh and cache_key in self.geo_region_cache:
            # print("Found cached location!")
            return self._record_from_cache(cache_key)

        osmnx_error = None

        try:
            record = self.osmnx_get_geo(location_name)
            record["fallback_used"] = False
            record["cache_hit"] = False

        except Exception as e:
            osmnx_error = repr(e)

            # print("osmnx failed, using google!")
            record = self.google_get_geo(location_name)

            # Google also failed
            if record.get("status") != "ok":
                failed_record = {
                    "status": "failed",
                    "location_name": location_name,
                    "source": None,
                    "method": "osmnx_then_google_failed",
                    "geometry_wkt": None,
                    "geometry_type": None,
                    "osmnx_error": osmnx_error,
                    "google_error": record.get("reason"),
                    "google_record": record,
                    "fallback_used": True,
                    "cache_hit": False,
                }

                self._cache_record(cache_key, failed_record)
                return failed_record

            # Google succeeded
            record["fallback_used"] = True
            record["osmnx_error"] = osmnx_error
            record["cache_hit"] = False

        self._cache_record(cache_key, record)

        return record


    def __init__(self, cache_path="./evaluation/geo_cache/geo_region_cache.json"):
        self.config = get_config()

        # Obtain geocoding api
        config_data = get_config()
        google_key = config_data["google_places_key"]["key"]
        self.gmaps_client = googlemaps.Client(key=google_key)

        self.cache_path = Path(cache_path)
        self.geo_region_cache = self._load_geo_region_cache()


if __name__ == "__main__":

    gm = GeoManager()

    location = "1200 block of Harper Avenue, West Hollywood"
    # location = "Culver City, Los Angeles"

    record = gm.get_geo_region(location)
    
    print(record["status"])
    print(record["source"])
    print(record["method"])
    print(record["geometry_type"])
    # print(record["geometry"])

    # Visualize it
    # highlight_geo(record["geometry"], location, cell_km=0.25)

