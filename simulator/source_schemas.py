"""Source schemas used to preview simulator inputs.

These describe the normalized source files, not SIGMUS database internals. They
live with the benchmark so simulation does not import the KG implementation.
"""

TABLE_TEMPLATES = {
    "cctv": "row_id INTEGER, time TIMESTAMPTZ, sensor_name TEXT, seaweed_image_ref TEXT, caption TEXT",
    "alertcalifornia": "row_id INTEGER, time TIMESTAMPTZ, sensor_name TEXT, latitude FLOAT, longitude FLOAT, direction FLOAT, seaweed_image_ref TEXT, caption TEXT",
    "pem_incidents": "row_id INTEGER, time TIMESTAMPTZ, latitude FLOAT, longitude FLOAT, data_description TEXT, data_severity TEXT, data_duration FLOAT",
    "pem_station": "row_id INTEGER, time TIMESTAMPTZ, latitude FLOAT, longitude FLOAT, sensor_name INTEGER, data_avg_occupancy FLOAT, data_avg_speed FLOAT",
    "gdelt_gkg": "row_id INTEGER, time TIMESTAMPTZ, sensor_name TEXT",
    "gdelt_events": "row_id INTEGER, time TIMESTAMPTZ, actor1_name TEXT, actor1_type TEXT, actor1_geo_name TEXT, actor1_latitude FLOAT, actor1_longitude FLOAT, actor2_name TEXT, actor2_type TEXT, actor2_geo_name TEXT, actor2_latitude FLOAT, actor2_longitude FLOAT, event_code TEXT, event_name TEXT, event_description TEXT, event_geo_name TEXT, event_latitude FLOAT, event_longitude FLOAT, event_date TEXT, original_link TEXT",
    "air_quality": "row_id INTEGER, time TIMESTAMPTZ, latitude FLOAT, longitude FLOAT, sensor_name INTEGER, data_pm25 FLOAT",
    "weather": "row_id INTEGER, time TIMESTAMPTZ, sensor_name TEXT, latitude FLOAT, longitude FLOAT, location_description TEXT, data_temp_f FLOAT, data_weather_description TEXT, data_humidity FLOAT, data_wind_mps FLOAT",
    "seismic": "row_id INTEGER, time TIMESTAMPTZ, time_end TIMESTAMPTZ, latitude FLOAT, longitude FLOAT, data_total_samples INTEGER, data_avg_sample_rate_hz FLOAT, data_channel_type TEXT, seaweed_file_ref TEXT",
    "citizen": "row_id INTEGER, time TIMESTAMPTZ, latitude FLOAT, longitude FLOAT, data_event_name TEXT, data_event_type TEXT, data_event_desc TEXT",
    "twitter": "row_id INTEGER, time TIMESTAMPTZ, time_end TIMESTAMPTZ, latitude FLOAT, longitude FLOAT, data_body TEXT, data_event_type TEXT",
}
