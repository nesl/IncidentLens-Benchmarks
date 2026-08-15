import os
import io

from utilities.util import get_config
from neo4j import GraphDatabase
from PIL import Image
import requests
from pathlib import Path
import shutil

# MCP server stuff
import json
from fastmcp import FastMCP
from typing import Tuple, List, Literal
from pydantic import BaseModel
import uuid

import requests
from requests_html import HTMLSession, AsyncHTMLSession
from datetime import datetime
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time
import random

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
# import vertexai
# from vertexai import types

# Source-file schemas belong to the benchmark, not the SIGMUS database layer.
from simulator.source_schemas import TABLE_TEMPLATES

# Class for source input
class Source(BaseModel):
    source_type: Literal["air_data", "alertcalifornia", "cctv", "citizen_data", "gkg", "pem_data_chp_incidents_day", "pem_data_station_5min", "twitter_data", "weather_data", "news", "california_traffic", "traffic_data"]


# Mapping between table formats and source types:
# Table keys:     'pem_station', 'gdelt_gkg', 'gdelt_events', 'air_quality', 'weather', 'seismic', 'citizen', 'twitter'
TABLE_MAP = {
    "pem_data_chp_incidents_day": "pem_incidents", 
    "pem_data_station_5min": "pem_station",
    "air_data": "air_quality",
    "weather_data": "weather",
    "california_traffic": "california_traffic",
    "traffic_data": "california_traffic",
    "twitter_data": "twitter",
    "citizen_data": "citizen"
}



mcp = FastMCP(name="Simulation Tools MCP")

#  Manages the state and connections to databases
class ToolStateAgent:
    _instance = None

    def __init__(self):
        self.config = get_config()
        self.neo4j_config = self.config["neo4j_config"]
        uri = self.neo4j_config["uri"]
        username = self.neo4j_config["username"]
        password = self.neo4j_config["password"]

        self.img_edit_server = self.config["img_edit"]

        gcloud_api_key = self.config["google"]["api"]
        self.cloud_edit_server = genai.Client(vertexai=True, api_key=gcloud_api_key)

        self.mcp_simulation_config = self.config["mcp"]["simulation"]

        self.driver = GraphDatabase.driver(uri, auth=(username, password))


        ToolStateAgent._instance = self

    @classmethod
    def get_driver(cls):
        return cls._instance.driver
    
    @classmethod
    def get_img_edit_server(cls, server_type="qwen"):
        if server_type == "qwen":
            return cls._instance.img_edit_server
        elif server_type == "cloud":
            return cls._instance.cloud_edit_server
    
    @classmethod
    def get_mcp_config(cls):
        return cls._instance.mcp_simulation_config

    

# A simple search which can look up historic sensory data from the most recent file
#  Right now, just at the file level but later can be at the DB or KG level.
@mcp.tool(
    name="lookup_historic_data",           # Custom tool name for the LLM
    description="Takes in a string describing the data source.  The value must be air_data, weather_data, cctv, citizen, twitter, news, or other source types.  It will return the text of the contents of the file (if the file is text), and otherwise return just the filepath. In addition, it will also return the column names, which may be useful in identifying the meaning of each value.  This is useful if you want to identify the schema of the data to generate new data similar to it." # Custom description
)
def lookup_historic_data(source: Source) -> str:
    "Simple function for looking up data filepaths"
    # Given a source type, look through that data.

    if source.source_type == "news":
        return "We do not have historic data for news - please use web search instead."

    if source.source_type in {"california_traffic", "traffic_data"}:
        return (
            "Data type: california_traffic\n"
            "Columns: data_avg_occupancy, data_avg_speed\n\n"
            "data_avg_occupancy: Average occupancy across all lanes over the 5-minute period "
            "expressed as a decimal number between 0 and 1.\n"
            "data_avg_speed: Flow-weighted average speed over the 5-minute period across all lanes. "
            "If flow is 0, mathematical average of 5-minute station speeds.\n"
            "No local historical traffic sample is available; synthesize plausible values from "
            "incident context, location, and time."
        )
    
    modality_path = os.path.join("pulled_data", source.source_type)
    latest_day = sorted(os.listdir(modality_path))[-1]
    latest_path = Path(os.path.join(modality_path, latest_day))

    while latest_path.is_dir():
        entries = sorted(latest_path.iterdir())  # sort for determinism
        if not entries:
            raise RuntimeError(f"Directory is empty: {latest_path}")
        latest_path = entries[-1]  # pick the "last" entry
    

    # Get the schema:
    data_format = TABLE_MAP.get(source.source_type, "unknown")
    out_str = ""
    if data_format in TABLE_TEMPLATES:
        schema = TABLE_TEMPLATES[data_format]
        schema_str = ", ".join([x.strip() for x in schema])

        out_str += f"Data type: {data_format}\nColumns: {schema_str}\n\n"


    if latest_path.suffix in [".txt", ".csv", ".json", ".nt"]:
        with open(latest_path, "r") as f:
        
            out_str += f"File data: {f.read()[:500]}" " -- preview" # Only read 500 characters
    else:
        out_str += f"Filepath: {latest_path}\n\n"
    
    return out_str


# This tool will save the sensory data 
# @mcp.tool(
#     name="read_webpage",           # Custom tool name for the LLM
#     description="A tool for getting the text from a web page given a link" # Custom description
# )
def read_webpage(link: str, timeout: int = 20) -> str:

    options = Options()
    options.headless = True
    # options.binary_location = "/snap/firefox/current/usr/lib/firefox/firefox"

    # Set realistic user agent
    options.set_preference(
        "general.useragent.override",
        "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) "
        "Gecko/20100101 Firefox/120.0"
    )

    # Reduce automation fingerprinting
    options.set_preference("dom.webdriver.enabled", False)
    options.set_preference("useAutomationExtension", False)

    driver = webdriver.Firefox(options=options)



    try:
        driver.get(link)

        # Wait for page body
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # Optional wait for JS-heavy pages
        time.sleep(3)

        # Optional scroll for lazy-loaded content
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        # Remove script/style tags
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        cleaned = "\n".join(line for line in lines if line)

        return cleaned[:5000] # Don't include whole web page

    finally:
        driver.quit()



def is_retryable_genai_error(err: Exception) -> bool:
    """Return True for Gemini/Vertex transient quota/rate/capacity errors."""
    err_str = str(err)
    retryable_markers = [
        "429",
        "RESOURCE_EXHAUSTED",
        "rateLimitExceeded",
        "quota",
        "503",
        "UNAVAILABLE",
        "500",
        "INTERNAL",
    ]
    return any(marker in err_str for marker in retryable_markers)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def genai_call_with_backoff(
    call_fn,
    *,
    max_attempts: int = 3,
    base_delay: float = 5.0,
    max_delay: float = 60.0,
):
    """Retry Gemini calls with exponential backoff and jitter."""
    last_err = None
    for attempt in range(max_attempts):
        try:
            return call_fn()
        except (ClientError, ServerError) as err:
            last_err = err
            if not is_retryable_genai_error(err):
                raise
            if attempt == max_attempts - 1:
                raise
            delay = min(max_delay, base_delay * (2 ** attempt))
            delay += random.uniform(0, delay * 0.25)
            print(
                f"[Gemini retry] attempt {attempt + 1}/{max_attempts} failed with retryable error: {err}"
            )
            print(f"[Gemini retry] sleeping {delay:.1f}s before retrying...")
            time.sleep(delay)
    raise last_err


def modify_image_cloud(
    img_filepath: str,
    cam_latlong: str = "",
    cam_description: str = "",
    prompt: str = "",
    step: int = 0,
    observation_id: str | None = None,
    save_dir: str = "simulator/generated",
    max_attempts: int = 4,
    base_delay: float = 15,
    max_delay: float = 60.0,
    fail_soft: bool = True,
) -> str | None:
    """Edit an image with Gemini/Vertex, with quota-aware backoff.

    Keeps backwards compatibility with the old positional API while supporting
    observation_id/save_dir from the simulator refactor.
    """
    if not prompt:
        raise ValueError("modify_image_cloud requires a prompt")

    # Keep image failures from dominating full simulator runtime.  The old
    # defaults could sleep 30s, then 60s, then 120s... per image on quota or
    # transient capacity errors.  These environment variables let long batch
    # runs tune retries without changing simulator.py.
    max_attempts = _env_int("SIMULATOR_IMAGE_EDIT_MAX_ATTEMPTS", max_attempts)
    base_delay = _env_float("SIMULATOR_IMAGE_EDIT_BASE_DELAY_SECONDS", base_delay)
    max_delay = _env_float("SIMULATOR_IMAGE_EDIT_MAX_DELAY_SECONDS", max_delay)

    os.makedirs(save_dir, exist_ok=True)
    file_extension = img_filepath.split(".")[-1]
    filetype = "jpeg" if file_extension == "jpg" else file_extension

    safe_id = observation_id or f"image_step{step}_{uuid.uuid4().hex[:8]}"
    save_filepath = os.path.join(save_dir, f"{safe_id}_input." + file_extension)
    out_filepath = os.path.join(save_dir, f"{safe_id}_output.png")

    img_edit_server = ToolStateAgent.get_img_edit_server(server_type="cloud")

    image_to_edit = Image.open(img_filepath)
    shutil.copy(img_filepath, save_filepath)

    def call_gemini():
        return img_edit_server.models.generate_content(
            model=os.environ.get("SIMULATOR_IMAGE_EDIT_MODEL", "gemini-2.5-flash-image"),
            contents=[prompt, image_to_edit],
        )

    request_start = time.perf_counter()
    try:
        response = genai_call_with_backoff(
            call_gemini,
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
        )
    except Exception as err:
        if fail_soft and is_retryable_genai_error(err):
            print(f"[WARN] Gemini image edit failed after retries due to quota/rate limit: {err}")
            print("[WARN] Continuing simulation without this generated image.")
            return None
        raise

    for part in response.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            image = part.as_image()
            image.save(out_filepath)
            print("saving image to: " + str(out_filepath))
            print(f"[TIMER] Gemini image edit request: {time.perf_counter() - request_start:.2f}s")
            return out_filepath

    if fail_soft:
        print(f"[TIMER] Gemini image edit request: {time.perf_counter() - request_start:.2f}s")
        print("[WARN] Gemini returned no image output. Continuing simulation.")
        return None
    raise RuntimeError("Gemini returned no image output.")

    


# @mcp.tool(
#     name="modify_image",           # Custom tool name for the LLM
#     description="Takes in an image filepath and prompt to produce a modified image based on the prompt request.  The input image is represented as a string filepath, but the output is represented as the image bytes",
# )
def modify_image_qwen(img_filepath: str, cam_latlong: str = "", cam_description: str = "", prompt: str = "", step: int = 0, observation_id: str | None = None, save_dir: str = "simulator/generated") -> str:

    os.makedirs(save_dir, exist_ok=True)
    file_extension = img_filepath.split(".")[-1]
    filetype = "jpeg" if file_extension == "jpg" else file_extension

    safe_id = observation_id or f"image_step{step}_{uuid.uuid4().hex[:8]}"
    save_filepath = os.path.join(save_dir, f"{safe_id}_input." + file_extension)
    out_filepath = os.path.join(save_dir, f"{safe_id}_output.png")  # qwen server returns png

    # Get image edit server details
    img_edit_server = ToolStateAgent.get_img_edit_server()
    api_url = img_edit_server["api"]
    status_url = img_edit_server["status"]

    # Submit job
    
    with open(img_filepath, "rb") as f:
        files = {"image": (img_filepath, f, "image/" + filetype)}
        data = {"prompt": prompt}
        resp = requests.post(api_url, files=files, data=data)
        resp.raise_for_status()
        job_id = resp.json()["job_id"]

    # Copy input elsewhere
    shutil.copy(img_filepath, save_filepath)

    print("Job submitted, ID:", job_id)

    # Poll status
    while True:
        r = requests.get(status_url.format(job_id))
        if r.status_code == 200 and r.headers.get("Content-Type") == "image/png":
            img = Image.open(io.BytesIO(r.content))
            img.save(out_filepath)
            print("Image received!")
            break
        elif r.status_code == 500:
            print("Error:", r.json())
            break
        else:
            print("Still processing...")
            time.sleep(10)

    return out_filepath


if __name__ == "__main__":

    # Initialize database connection once
    agent = ToolStateAgent()


    # Test image edit
    # results = modify_image_cloud("simulator/test/test2.png", "test", "test", "make the image have a rainbow", 1)


    # print(lookup_historic_data(source="air_data"))

    mcp_config = agent.get_mcp_config()
    print(mcp_config)
    mcp.run(transport="http", host=mcp_config["host"], port=int(mcp_config["port"]), path=mcp_config["path"])
