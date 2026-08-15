from langchain_core.prompts import PromptTemplate

from simulator.tools.utility import parse_model_output, load_mcp_with_agent, get_relevant_messages
from utilities.util import get_config
from langchain_openai import ChatOpenAI
import asyncio
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple


time_series_agent = None
sensor_token_s = "<SENSOR>"
sensor_token_e = "</SENSOR>"
header_token_s = "<HEADER>"
header_token_e = "</HEADER>"


TRAFFIC_SOURCE_ALIASES = ("traffic", "california_traffic", "caltrans", "pem", "pems")
TRAFFIC_HEADER = "sensor_id,timestamp,latitude,longitude,data_avg_occupancy,data_avg_speed"
TRAFFIC_FIELD_DESCRIPTIONS = (
    "data_avg_occupancy: Average occupancy across all lanes over the 5-minute period, "
    "expressed as a decimal number between 0 and 1. "
    "data_avg_speed: Flow-weighted average speed over the 5-minute period across all lanes; "
    "if flow is 0, mathematical average of 5-minute station speeds."
)


def _is_california_traffic_source(source: str) -> bool:
    source_l = source.lower().strip().replace("-", "_").replace(" ", "_")
    return any(alias in source_l for alias in TRAFFIC_SOURCE_ALIASES)


def _make_direct_llm() -> ChatOpenAI:
    config = get_config()
    openai = config["openai"]
    os.environ["OPENAI_API_KEY"] = openai["api"]
    return ChatOpenAI(model=openai.get("model", "gpt-4"), temperature=0)


def _extract_representative_latlon(geo_region_prompt: str) -> Tuple[float, float]:
    """Best-effort parse of representative latitude/longitude from simulator.py prompt text."""
    if not geo_region_prompt:
        return 34.0522, -118.2437

    patterns = [
        r"Representative point:\s*latitude=([-+]?\d+(?:\.\d+)?),\s*longitude=([-+]?\d+(?:\.\d+)?)",
        r"latitude=([-+]?\d+(?:\.\d+)?),\s*longitude=([-+]?\d+(?:\.\d+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, geo_region_prompt)
        if m:
            return float(m.group(1)), float(m.group(2))
    return 34.0522, -118.2437


def _sensor_locations_to_prompt(sensor_locations: Optional[List[Dict[str, Any]]]) -> str:
    if not sensor_locations:
        return ""
    lines = ["Use exactly these simulator-provided fixed physical sensor locations; do not invent latitude/longitude values:"]
    for s in sensor_locations:
        role = s.get("sensor_region_role", "inside")
        distance = s.get("outside_distance_km")
        if role == "outside":
            role_text = f"outside incident region {s.get('location_name', '')}"
            if distance is not None:
                role_text += f" (~{distance} km beyond the incident boundary)"
        else:
            role_text = f"inside incident region {s.get('location_name', '')}"
        lines.append(
            f"- {s.get('sensor_id')}: type={s.get('sensor_type', '')}, "
            f"latitude={s.get('latitude')}, longitude={s.get('longitude')}, {role_text}"
        )
    return "\n".join(lines)


def _format_fast_csv_response(header: str, rows: list[list[object]]) -> str:
    out = [f"{header_token_s}{header}{header_token_e}"]
    for row in rows:
        out.append(f"{sensor_token_s}" + ",".join(str(x) for x in row) + f"{sensor_token_e}")
    return "\n".join(out)


def _traffic_values(event_desc: str) -> Tuple[float, float]:
    desc = event_desc.lower()
    severe = ["closure", "closed", "accident", "crash", "protest", "demonstration", "fire", "wildfire", "flood", "threat", "evacuation"]
    if any(w in desc for w in severe):
        # Congestion/blocked-road signature.
        return round(random.uniform(0.45, 0.95), 3), round(random.uniform(3.0, 25.0), 1)
    return round(random.uniform(0.05, 0.35), 3), round(random.uniform(30.0, 70.0), 1)


async def generate_california_traffic(
    time_str: str,
    event_desc: str,
    step: int,
    geo_region_prompt: str = "",
    save_folder: str = "simulator/generated",
    incident_id: str = "incident_0",
    n_sensors: int = 3,
    sensor_locations: Optional[List[Dict[str, Any]]] = None,
):
    """Generate California traffic trend data without historic lookup or MCP.

    This is intentionally deterministic/Python-based for speed. It keeps the two
    requested trend columns and adds sensor/time/location fields so the simulator
    can log observation metadata externally.
    """
    base_lat, base_lon = _extract_representative_latlon(geo_region_prompt)
    rows = []
    if sensor_locations:
        selected_sensors = sensor_locations[: max(1, min(len(sensor_locations), 5))]
    else:
        selected_sensors = []
        for i in range(max(1, min(n_sensors, 5))):
            # Fallback only. simulator.py normally supplies exact in-polygon locations.
            selected_sensors.append({
                "sensor_id": f"traffic_{step}_{i}",
                "latitude": round(base_lat + random.uniform(-0.002, 0.002), 6),
                "longitude": round(base_lon + random.uniform(-0.002, 0.002), 6),
            })

    for i, sensor in enumerate(selected_sensors):
        occupancy, speed = _traffic_values(event_desc)
        rows.append([
            sensor.get("sensor_id", f"traffic_{step}_{i}"),
            time_str,
            sensor.get("latitude", base_lat),
            sensor.get("longitude", base_lon),
            occupancy,
            speed,
        ])

    response_text = _format_fast_csv_response(TRAFFIC_HEADER, rows)
    return parse_model_output(
        response_text,
        "california_traffic",
        step,
        [header_token_s, header_token_e],
        [sensor_token_s, sensor_token_e],
        save_folder=save_folder,
        filename_prefix=incident_id,
    )




def _is_air_quality_source(source: str) -> bool:
    source_l = source.lower().strip().replace("-", "_").replace(" ", "_")
    return any(alias in source_l for alias in ["air", "air_quality", "pm25", "pm2_5", "aqi"])


def _air_values(event_desc: str, sensor_region_role: str = "inside") -> Tuple[float, float, int, str]:
    desc = event_desc.lower()
    if sensor_region_role == "outside":
        return (
            round(random.uniform(3.0, 12.0), 1),
            round(random.uniform(8.0, 25.0), 1),
            random.randint(15, 55),
            "background air quality; no incident abnormality detected",
        )
    smoky = ["wildfire", "fire", "smoke", "hazardous material", "chemical", "terrorist", "explosion"]
    if any(w in desc for w in smoky):
        return (
            round(random.uniform(55.0, 220.0), 1),
            round(random.uniform(70.0, 280.0), 1),
            random.randint(120, 300),
            "elevated particulate/air-quality anomaly consistent with incident",
        )
    return (
        round(random.uniform(6.0, 18.0), 1),
        round(random.uniform(12.0, 35.0), 1),
        random.randint(25, 75),
        "mildly elevated or background air quality",
    )


async def generate_air_quality(
    time_str: str,
    event_desc: str,
    step: int,
    geo_region_prompt: str = "",
    save_folder: str = "simulator/generated",
    incident_id: str = "incident_0",
    n_sensors: int = 3,
    sensor_locations: Optional[List[Dict[str, Any]]] = None,
):
    """Generate air-quality data without an LLM call.

    Air data is one of the common sources in this simulator, and using an LLM
    once per step for a fixed schema adds avoidable runtime.  This deterministic
    path preserves the same CSV/logging shape and respects outside negative
    controls.
    """
    base_lat, base_lon = _extract_representative_latlon(geo_region_prompt)
    rows = []
    if sensor_locations:
        selected_sensors = sensor_locations[: max(1, min(len(sensor_locations), 5))]
    else:
        selected_sensors = []
        for i in range(max(1, min(n_sensors, 5))):
            selected_sensors.append({
                "sensor_id": f"air_{step}_{i}",
                "latitude": round(base_lat + random.uniform(-0.002, 0.002), 6),
                "longitude": round(base_lon + random.uniform(-0.002, 0.002), 6),
                "sensor_region_role": "inside",
            })

    header = "sensor_id,timestamp,latitude,longitude,pm25,pm10,aqi,description"
    for i, sensor in enumerate(selected_sensors):
        role = str(sensor.get("sensor_region_role", "inside")).lower().strip()
        pm25, pm10, aqi, desc = _air_values(event_desc, role)
        rows.append([
            sensor.get("sensor_id", f"air_{step}_{i}"),
            time_str,
            sensor.get("latitude", base_lat),
            sensor.get("longitude", base_lon),
            pm25,
            pm10,
            aqi,
            desc,
        ])

    response_text = _format_fast_csv_response(header, rows)
    return parse_model_output(
        response_text,
        "air",
        step,
        [header_token_s, header_token_e],
        [sensor_token_s, sensor_token_e],
        save_folder=save_folder,
        filename_prefix=incident_id,
    )


async def _ensure_time_series_agent():
    global time_series_agent
    if time_series_agent is None:
        time_series_agent = await load_mcp_with_agent()
    return time_series_agent


def _fast_schema_for_source(source: str) -> str:
    source_l = source.lower()
    if "weather" in source_l:
        return "sensor_id,timestamp,latitude,longitude,temperature_f,wind_speed_mph,wind_direction,precipitation_in,description"
    if "air" in source_l:
        return "sensor_id,timestamp,latitude,longitude,pm25,pm10,aqi,description"
    return "sensor_id,timestamp,latitude,longitude,measurement,value,unit,description"


async def generate_time_series_fast_direct(
    time_str: str,
    event_desc: str,
    source: str,
    step: int,
    geo_region_prompt: str = "",
    save_folder: str = "simulator/generated",
    incident_id: str = "incident_0",
    sensor_locations: Optional[List[Dict[str, Any]]] = None,
):
    """One direct LLM call, no MCP/ReAct/web tools.

    This trades historical-schema matching for speed. Use fast_mode=False in
    generate_time_series() when you want the older MCP/schema lookup behavior.
    """
    llm = _make_direct_llm()
    header = _fast_schema_for_source(source)
    exact_sensor_prompt = _sensor_locations_to_prompt(sensor_locations)
    prompt = (
        "Generate synthetic physical time-series sensor data for an incident. "
        "Use exactly the required CSV schema and no extra columns. "
        "When simulator-provided sensors are listed, generate one row per listed sensor, up to 5 rows. "
        "Use the exact simulator-provided sensor coordinates when provided, and make values "
        "consistent with the incident context and whether each sensor is inside or outside the incident region. "
        "Do not call tools.\n\n"
        f"Source: {source}\n"
        f"Time: {time_str}\n"
        f"Incident context: {event_desc}\n"
        f"Geo-region constraints: {geo_region_prompt or 'Use a plausible Los Angeles-area location near the incident.'}\n"
        f"{exact_sensor_prompt}\n\n"
        f"Required header exactly: {header}\n"
        f"Format the header between {header_token_s} and {header_token_e}. "
        f"Format each CSV row between {sensor_token_s} and {sensor_token_e}."
    )
    response = llm.invoke(prompt)
    print("AI messages:", [response.content])
    return parse_model_output(
        response.content,
        source,
        step,
        [header_token_s, header_token_e],
        [sensor_token_s, sensor_token_e],
        save_folder=save_folder,
        filename_prefix=incident_id,
    )


async def generate_time_series(
    time_str: str,
    event_desc: str,
    source: str,
    step: int,
    geo_region_prompt: str = "",
    save_folder: str = "simulator/generated",
    incident_id: str = "incident_0",
    fast_mode: bool = True,
    sensor_locations: Optional[List[Dict[str, Any]]] = None,
):
    """Generate time-series sensor data for one simulator step.

    fast_mode=True avoids MCP/ReAct/tool lookup and is much faster. Set
    fast_mode=False to use the historical-data/schema lookup path.
    """
    if _is_california_traffic_source(source):
        return await generate_california_traffic(
            time_str=time_str,
            event_desc=event_desc,
            step=step,
            geo_region_prompt=geo_region_prompt,
            save_folder=save_folder,
            incident_id=incident_id,
            sensor_locations=sensor_locations,
        )

    if fast_mode and _is_air_quality_source(source):
        return await generate_air_quality(
            time_str=time_str,
            event_desc=event_desc,
            step=step,
            geo_region_prompt=geo_region_prompt,
            save_folder=save_folder,
            incident_id=incident_id,
            sensor_locations=sensor_locations,
        )

    if fast_mode:
        return await generate_time_series_fast_direct(
            time_str=time_str,
            event_desc=event_desc,
            source=source,
            step=step,
            geo_region_prompt=geo_region_prompt,
            save_folder=save_folder,
            incident_id=incident_id,
            sensor_locations=sensor_locations,
        )

    agent = await _ensure_time_series_agent()

    prompt = PromptTemplate(
        template=(
            "You are helping simulate an incident by producing physical sensory time-series data. "
            "There are tools which you can use as reference to produce data of a similar schema "
            "(for example, air quality has a particular format). You may be producing data at a "
            "particular time step, in which case I will also provide the time and previous simulated "
            "step context. Here is the time: {time}. Here is the event description: {event_description}. "
            "Here is the data source you should try to generate data for: {source}. Your objective is "
            "to use the given tools to simulate data which matches the description. You may need to call "
            "multiple tools. Try to match the schema and format from the reference data when relevant. "
            "Importantly, while you may be given the schema for the data, you only need to simulate the "
            "data given by the example; do not simulate extra fields from the schema unless needed for "
            "sensor placement. Return the simulated data as text output and verify that the output format "
            "matches the reference data format. {geo_region_prompt}"
        ),
        input_variables=["time", "event_description", "source", "geo_region_prompt"],
    )
    prompt = prompt.format(
        time=time_str,
        event_description=event_desc,
        source=source,
        geo_region_prompt=geo_region_prompt + " " + _sensor_locations_to_prompt(sensor_locations),
    )

    formatting = (
        " Please format your output as comma-separated rows between "
        + sensor_token_s
        + " and "
        + sensor_token_e
        + " tags. You may use multiple tags if you generate data from multiple sensors, "
        "but generate only up to 5 sensors at any one time. If fixed simulator-provided sensors "
        "are listed, generate one row per listed sensor, up to 5 rows. For physical sensors, include "
        "sensor_id, latitude, and longitude columns if the reference example does not already "
        "identify the sensor location. Sensors may be inside the provided incident geo-region or "
        "outside it but facing/near the incident; keep the values consistent with that placement. "
        "Keep in mind that a single location should only have one sensor. Please also use the "
        + header_token_s
        + header_token_e
        + " tags to generate a comma-separated list of column headers. Remember that you should "
        "not generate more data beyond the example, except for sensor_id/latitude/longitude when "
        "needed to make the physical sensor placement explicit."
    )

    response = await agent.ainvoke({"messages": [{"role": "user", "content": prompt + formatting}]})

    tool_messages, ai_messages = get_relevant_messages(response)
    print("AI messages:", ai_messages)

    return parse_model_output(
        ai_messages[-1],
        source,
        step,
        [header_token_s, header_token_e],
        [sensor_token_s, sensor_token_e],
        save_folder=save_folder,
        filename_prefix=incident_id,
    )


if __name__ == "__main__":
    asyncio.run(
        generate_time_series(
            "May 20, 2025 at 3pm",
            "The start of a wildfire in Culver City, Los Angeles.",
            "weather",
            1,
        )
    )
