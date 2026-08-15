import os
import tarfile
import csv

import time
import re

from bs4 import BeautifulSoup
import json
from langchain_core.messages import SystemMessage, HumanMessage

from selenium import webdriver

from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

from selenium.webdriver.common.by import By

from utilities.util import get_config
from langchain_openai import ChatOpenAI
from openai import OpenAI

import requests
from requests.exceptions import RequestException, HTTPError, ReadTimeout

from tqdm import tqdm

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from urllib.parse import urlparse

from typing import Literal, List, Tuple
from pydantic import BaseModel, Field
import shutil


def extract_tar_to_news_temp(tar_path, output_dir="/evaluation/temp/news"):
    """
    Extracts all contents of tar_path directly into:
        /evaluation/temp/news

    It does NOT create:
        /evaluation/temp/news/<tarfile_name>
    """

    os.makedirs(output_dir, exist_ok=True)

    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(path=output_dir)

    return output_dir



def filter_url_terms(url):

    to_ignore = ["entertainment"]
    to_filter = False
    for x in to_ignore:
        if x in url:
            to_filter = True
    
    return to_filter


# Additional filtering based on themes
# Theme list is here: http://data.gdeltproject.org/api/v2/guides/LOOKUP-GKGTHEMES.TXT


    

def url_topic_key(url):
    """
    Extract a rough topic key from a URL path.

    Example:
    https://fox4beaumont.com/news/entertainment/jennifer-aniston-11-wildest-confessions-birthday
    ->
    jennifer-aniston-11-wildest-confessions-birthday
    """

    if not url:
        return ""

    path = urlparse(url.lower()).path.strip("/")

    if not path:
        return ""

    parts = [p for p in path.split("/") if p]

    # Usually the final path component is the article slug
    slug = parts[-1]

    # Remove common file endings
    slug = re.sub(r"\.(html|htm|php|aspx)$", "", slug)

    # Remove trailing numeric IDs if desired
    slug = re.sub(r"-\d+$", "", slug)

    return slug.strip()

def theme_present(filter_terms, theme_data):

    # print(filter_terms)
    # print(theme_data)
    # input()
    

    theme_present = False
    
    for x in filter_terms:
        if x in theme_data:
            theme_present = True
            # print(x)
            # print(theme_data)
            break

    return theme_present


def loc_name_proportion(v2locations):
    """
    Returns the proportion of V2Locations entries
    that refer to Los Angeles.

    Example:
        2 LA entries out of 5 total -> 0.4
    """

    if not v2locations:
        return 0.0

    entries = [
        x.strip()
        for x in v2locations.split(";")
        if x.strip()
    ]

    if not entries:
        return 0.0

    la_count = 0

    for entry in entries:
        parts = entry.split("#")

        if len(parts) < 8:
            continue

        location_name = parts[1].lower()
        feature_id = parts[7]

        is_la = (
            "los angeles" in location_name
            or feature_id == "1662328"
        )

        if is_la:
            la_count += 1

    return la_count / len(entries)


def extract_unique_news_urls(root_dir, filter_terms):
    """
    Extract unique URLs from column 4.
    Deduplicates by exact URL and exact normalized article slug/topic.
    """

    unique_urls = set()
    seen_topic_keys = set()

    total_rows = 0
    exact_duplicate_count = 0
    topic_duplicate_count = 0
    filtered_urls_by_theme = 0
    filtered_by_location = 0
    filtered_by_urlterm = 0

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".csv"):
                continue

            file_path = os.path.join(root, file)
            # print(f"Processing: {file_path}")

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)

                    for line_num, row in enumerate(reader, start=1):
                        total_rows += 1

                        if len(row) <= 4:
                            print(f"  Skipping malformed row (line {line_num})")
                            continue

                        news_url = row[4].strip()

                        datetime_str = row[1].strip()

                        # item 7 and 8 are gkg themes
                        if not (theme_present(filter_terms, row[7]) or theme_present(filter_terms, row[8])):
                            filtered_urls_by_theme += 1
                            continue

                        # Item 10 is a location
                        if not loc_name_proportion(row[10]) > 0.5:
                            filtered_by_location += 1
                            continue 

                        # Filter by terms in the url
                        if filter_url_terms(news_url):
                            filtered_by_urlterm += 1
                            continue



                        if not news_url:
                            continue

                        topic_key = url_topic_key(news_url)

                        if news_url in unique_urls:
                            exact_duplicate_count += 1
                            continue

                        if topic_key and topic_key in seen_topic_keys:
                            topic_duplicate_count += 1
                            continue

                        unique_urls.add((news_url, datetime_str))
                        

                        if topic_key:
                            seen_topic_keys.add(topic_key)

            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    print("\nFinished.")
    print(f"Total rows processed : {total_rows}")
    print(f"Unique URLs found    : {len(unique_urls)}")
    print(f"Exact duplicates     : {exact_duplicate_count}")
    print(f"Topic duplicates     : {topic_duplicate_count}")
    print(f"Filtered by theme    : {filtered_urls_by_theme}")
    print(f"Filtered by location : {filtered_by_location}")
    print(f"Filtered by url term : {filtered_by_urlterm}")

    return unique_urls





def fetch_webpage(url, timeout=10):
    """
    Fetch webpage HTML content.

    Returns:
        html_content (str)

    Raises:
        Exception if request fails.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=timeout
    )

    # Raise exception for bad status codes
    response.raise_for_status()

    return response.text

def fetch_html_selenium(url, wait_time=5):
    """
    Firefox Selenium fetch.

    Useful for:
      - 403 protection
      - JavaScript-rendered pages
      - Some anti-bot protections
    """

    firefox_options = Options()

    # Headless mode
    firefox_options.add_argument("--headless")

    # Some anti-detection improvements
    firefox_options.set_preference(
        "general.useragent.override",
        (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64; rv:136.0) "
            "Gecko/20100101 Firefox/136.0"
        )
    )

    firefox_options.set_preference(
        "dom.webdriver.enabled",
        False
    )

    firefox_options.set_preference(
        "useAutomationExtension",
        False
    )

    firefox_options.set_preference(
        "media.peerconnection.enabled",
        False
    )

    # Create driver
    driver = webdriver.Firefox(
        options=firefox_options
    )

    try:

        driver.get(url)

        # Wait for JS rendering
        time.sleep(wait_time)

        html = driver.page_source

        return html

    finally:

        driver.quit()

def extract_article_text(html):
    """
    Extract likely article text from HTML.
    """

    soup = BeautifulSoup(html, "html.parser")

    # Remove obvious junk
    for tag in soup([
        "script",
        "style",
        "noscript",
        "iframe",
        "svg",
        "form",
        "button",
        "footer",
        "nav",
        "aside",
    ]):
        tag.decompose()

    # Common article selectors
    selectors = [
        "article",
        '[role="main"]',
        ".article-body",
        ".story-body",
        ".post-content",
        ".entry-content",
        ".article-content",
        ".main-content",
    ]

    article = None

    for selector in selectors:
        article = soup.select_one(selector)

        if article:
            break

    # Fallback to body
    if article is None:
        article = soup.body

    if article is None:
        return ""

    # Extract paragraph text
    paragraphs = article.find_all(["p", "h1", "h2", "h3"])

    lines = []

    for p in paragraphs:
        text = p.get_text(" ", strip=True)

        # Skip tiny junk fragments
        if len(text) < 30:
            continue

        lines.append(text)

    text = "\n\n".join(lines)

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


MIN_ARTICLE_CHARS = 1000


def try_selenium(url, original_error, error_urls):
    """
    Attempt Selenium fallback.
    """

    print("  Trying Selenium...")

    try:
        html = fetch_html_selenium(url)
        html = extract_article_text(html)

        if len(html) < MIN_ARTICLE_CHARS:

            error_urls.append({
                "url": url,
                "error": "selenium html too short",
                "original_error": str(original_error),
                "status_code": 200,
                "selenium_tried": True,
            })

            print(f"  SELENIUM SHORT ({len(html)} chars)")

            return False, ""

        print(f"  SELENIUM SUCCESS ({len(html)} chars)")

        return True, html

    except Exception as selenium_error:

        error_urls.append({
            "url": url,
            "error": str(selenium_error),
            "original_error": str(original_error),
            "status_code": None,
            "selenium_tried": True,
        })

        print(f"  SELENIUM ERROR: {selenium_error}")

        return False, ""




def get_html_from_urls(url_list, llm_parser, timeout=5):

    success_count = 0
    error_count = 0
    selenium_success_count = 0

    error_urls = []
    short_article_urls = []

    total = len(url_list)
    all_event_data = []
    token_counts = []

    for i, url_tup in tqdm(
        enumerate(url_list, start=1),
        total=total
    ):
        url, date_str = url_tup

        curr_event_data = {}
        

        print(f"[{i}/{total}] Fetching: {url}")

        try:
            html = fetch_webpage(url, timeout=timeout)
            html = extract_article_text(html)

            if len(html) < MIN_ARTICLE_CHARS:

                short_article_urls.append(url)

                error_count += 1

                error_urls.append({
                    "url": url,
                    "error": "html too short",
                    "status_code": 200,
                    "selenium_tried": False,
                })

                print(f"  SHORT ARTICLE ({len(html)} chars)")

                continue

            success_count += 1

            # Parse with LLM
            curr_event_data, usage = llm_parser.get_event_data(html, url, date_str)
            all_event_data.append(curr_event_data)
            token_counts.append(usage)
            print(f"  SUCCESS ({len(html)} chars)")

        except HTTPError as e:

            status_code = (
                e.response.status_code
                if e.response is not None
                else None
            )

            print(f"  HTTP ERROR status_code={status_code}")

            if status_code == 403:
                print("  Got 403. Trying Selenium...")

                selenium_ok, html = try_selenium(
                    url,
                    e,
                    error_urls
                )

                if selenium_ok:
                    success_count += 1
                    selenium_success_count += 1
                    curr_event_data, usage = llm_parser.get_event_data(html, url, date_str)

                    all_event_data.append(curr_event_data)
                    token_counts.append(usage)
                else:
                    error_count += 1

            else:
                error_count += 1

                error_urls.append({
                    "url": url,
                    "error": str(e),
                    "status_code": status_code,
                    "selenium_tried": False,
                })

                print(f"  ERROR: {e}")

        except ReadTimeout as e:

            print("  Read timeout.")

            selenium_ok, html = try_selenium(
                url,
                e,
                error_urls
            )

            if selenium_ok:
                success_count += 1
                selenium_success_count += 1
                curr_event_data, usage = llm_parser.get_event_data(html, url, date_str)
                all_event_data.append(curr_event_data)
                token_counts.append(usage)
            else:
                error_count += 1

        except RequestException as e:

            error_count += 1

            error_urls.append({
                "url": url,
                "error": str(e),
                "status_code": None,
                "selenium_tried": False,
            })

            print(f"  ERROR: {e}")

        except Exception as e:

            error_count += 1

            error_urls.append({
                "url": url,
                "error": str(e),
                "status_code": None,
                "selenium_tried": False,
            })

            print(f"  ERROR: {e}")

    print("\nFinished URL checking.")

    print(f"All Successes          : {success_count}")
    print(f"Selenium successes : {selenium_success_count}")
    print(f"Errors             : {error_count}")
    print(f"Short articles     : {len(short_article_urls)}")

    error_data = {
        "basic_success_count": (
            success_count - selenium_success_count
        ),
        "selenium_success_count": selenium_success_count,
        "total_success_count": success_count,
        "error_count": error_count,
        "short_article_urls": short_article_urls,
        "error_urls": error_urls,
    }
    

    return error_data, all_event_data, token_counts


# Open the news filter terms
def get_filter_terms(filepath="evaluation/news_filter_terms.txt"):

    with open(filepath, "r") as f:
        filter_terms = f.readlines()
    
    # Remove description and whitespace
    filter_terms = filter_terms[1:]
    filter_terms = [x.strip() for x in filter_terms]
    return filter_terms


class EventRelationship(BaseModel):
    relationship_type: str = Field(
        description="Relationship type, such as part_of or caused_by"
    )
    other_event_name: str = Field(
        description="Name of the other event explicitly mentioned in the article"
    )

class EventData(BaseModel):
    event_type: str = Field(
        description="One event type from the provided event_terms list, or N/A"
    )
    location: str = Field(
        description="Specific human-readable English location such as address, neighborhood, landmark, region, city subarea, or N/A"
    )
    start_time: str = Field(
        description="ISO8601 date/time/timezone string, date-only ISO string, or N/A"
    )
    end_time: str = Field(
        description="ISO8601 date/time/timezone string, date-only ISO string, or N/A"
    )
    reasoning: str = Field(
        description="Brief explanation of why the event_type, location, start_time, and end_time were chosen"
    )
    event_name: str = Field(
        description="Name of the event exactly as stated in the news article if available, otherwise N/A"
    )
    event_description: str = Field(
        description="One line summary of how the news article describes the event"
    )
    relationships: List[EventRelationship] = Field(
        default_factory=list,
        description="Describes of the current event to other events mentioned in the article, in the form of (relationship_type, other event name).  Example relationship types are part_of, caused_by.  This field is an empty list if no explicit relationships are mentioned."
    )

class EventDataList(BaseModel):
    events: List[EventData] = Field(
        description="List of emergency or incident events described directly by the article"
    )

def safe_json_loads(text: str) -> dict:
    """
    Tries to parse an LLM response as JSON.
    Handles cases where the model wraps JSON in markdown fences.
    """
    text = text.strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    return json.loads(text)

# This parses news articles and catgorizes them
class LLM_news_parser:

    def get_event_terms(self, term_filepath="detection/emergency_types_filtered.txt"):

        # Open filepath and read line by line
        with open(term_filepath, "r") as f:
            event_terms = f.read()
        
        return event_terms

    def __init__(self):
        # Setup for parsing
        config = get_config()
        # Add LLM client
        openai = config["openai"]
        openai_api_key = openai["api"]
        langsmith_api_key = config["langsmith"]["api"]

        # Set up langsmith
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key

        self.llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
        # self.llm = ChatOpenAI(model="gpt-5.4", temperature=0)
        self.event_terms = self.get_event_terms()

    
    # Get event information from html
    #  {
    #     "event_type": ,
    #     "location": ,
    #     "start_time": ,
    #     "end_time": ,   # ISO8601 with date and time and timezone
    #   }
    def get_event_data(self, article_text: str, url: str, date_str: str) -> dict:
        """
        Extract one or more event records from an HTML document.

        Returns:
            {
                "events": [
                    {
                        "event_type": str,
                        "location": str,
                        "start_time": str,
                        "end_time": str,
                        "reasoning": str,
                        ...
                    },
                    ...
                ]
            }

        Notes:
        - event_type must be selected from self.event_terms or "N/A".
        - location should be specific and human-readable, not lat/lon.
        - start_time and end_time should be ISO8601 when available.
        - Fields that cannot be directly identified should be "N/A".
        """

        # Prevent very long pages from exceeding model context.
        max_chars = 25000
        article_text = article_text[:max_chars]

        # event_terms_text = "\n".join(
        #     f"- {term}"
        #     for term in self.event_terms
        # )
        event_terms_text = self.event_terms

        structured_llm = self.llm.with_structured_output(EventDataList, include_raw=True)

        prompt = f"""
    You extract structured emergency/event information from news article text.

    The article may describe one event or multiple events happening at different times and locations.

    Return a JSON-compatible object with this structure:

    {{
    "events": [
        {{
        "event_type": "...",
        "location": "...",
        "start_time": "...",
        "end_time": "...",
        "event_name: "...",
        "relationships": "...",
        "reasoning": "...",
        "event_description": "..."
        }}
    ]
    }}

    Each event must contain these fields:
    - event_type
    - location
    - start_time
    - end_time
    - event_name
    - relationships
    - reasoning
    - event_description
    


    Rules:
    - event_type must be exactly one item from the allowed event type list.
    - Do not invent event types.
    - If no allowed event_type clearly matches the article, do not return an event.
    - If event_type would be "N/A", do not return that event.
    - Only return events that are directly described in the article which have stated start times and locations
    - Do not extract background examples, unrelated historical events, or hypothetical risks unless they are directly described as occurring.

    Location rules:
    - location should be a human-readable English place description.
    - The location should be as specific as possible, such as:
    - a street address
    - a named road or intersection
    - a landmark
    - a neighborhood, e.g. Palisades
    - a region within a county, e.g. Los Angeles National Forest
    - If you cannot obtain location information as specific as a neighborhood or region, do not return that event.  
    - You must only return real location names that can be reverse-geocoded, otherwise N/A
    - Do not use latitude/longitude.
    - If you need to use two locations, then you should create two events.  In other words, this field should not include more than a single location.

    Time rules:
    - start_time is required.
    - If you cannot identify the start_time of the event directly, do not return that event.  The start time must be directly stated in the article to be used.
    - start_time and end_time should use ISO8601 with timezone when possible.
    - Example ISO8601 datetime: 2025-01-08T10:30:00-08:00
    - If only the date is known, use date-only ISO format, e.g. 2025-01-08.
    - If the event is ongoing and no end time is provided, use "N/A" for end_time.
    - If end_time is unavailable, use "N/A".

    Consistency rules:
    - The time and location should correspond to the specific event_type.
    - For example, if the event_type is wildfire, the location should describe the wildfire location,
    and the times should describe the start and end time of the wildfire.
    - Include a reasoning field for the selected time and location
    - The reasoning should briefly explain why each field's value was chosen.
    - also include a short event_description field summarizing how the news article describes the event
    - For the event_name, check if the article gives a name for your extracted event_type and location.  For example, the article may name it "Culver City Bombing". Do not come up with your own name, and use "N/A" if that's the case.

    Other rules:
    - For the relationships field, this should be structured as a list of pairs (relationship_type, other_event).  This should describe how this current event relates to other events mentioned in the article.  For example, if your event name is "Culver City Bombing", the article may state that it's part of another larger event called "Los Angeles Attacks", so you would have [("part_of", "Los Angeles Attacks")].  You may have multiple pairs.  Only add a pair if the other_event is explicitly mentioned.  Allowed relationship_types are ['part_of', 'caused_by'], but you may use other terms as you see fit. Use empty list if there are no clearly stated relationships. 
    
    


    If no valid events meet all requirements, return:
    {{
    "events": []
    }}

    Allowed event types:
    {event_terms_text}

    Article text:
    {article_text}
    """.strip()

        try:
            response = structured_llm.invoke(prompt)
            parsed = response["parsed"]
            raw = response["raw"]
            result = parsed.model_dump()

            usage = getattr(raw, "usage_metadata", None)
            response_metadata = getattr(raw, "response_metadata", None)

        except Exception as e:
            return {
                "events": [],
                "error": f"Extraction failed: {str(e)}"
            }

        cleaned_events = []

        for event in result.get("events", []):
            cleaned_event = {
                "event_type": event.get("event_type", "N/A"),
                "location": event.get("location", "N/A"),
                "start_time": event.get("start_time", "N/A"),
                "end_time": event.get("end_time", "N/A"),
                "reasoning": event.get("reasoning", "N/A"),
                "event_name": event.get("event_name", "N/A"),
                "event_description": event.get("event_description", "N/A"),
                "relationships": event.get("relationships", []),
            }

            # Normalize null / blank values
            for key, value in cleaned_event.items():
                if value is None:
                    cleaned_event[key] = "N/A"

                elif key == "relationships":
                    if not isinstance(value, list):
                        cleaned_event[key] = []
                    else:
                        cleaned_event[key] = value

                elif str(value).strip() == "":
                    cleaned_event[key] = "N/A"

                else:
                    cleaned_event[key] = str(value).strip()

            # Hard validation:
            # If event_type is missing or invalid, skip this event.
            if cleaned_event["event_type"] == "N/A":
                continue

            if cleaned_event["event_type"] not in self.event_terms:
                continue

            # If location is not specific enough / unavailable, skip.
            if cleaned_event["location"] == "N/A":
                continue

            # If start_time is unavailable, skip.
            if cleaned_event["start_time"] == "N/A":
                continue

            cleaned_events.append(cleaned_event)


        return {
            "events": cleaned_events,
            "date": date_str,
            "link": url
        }, usage


def clear_news_temp(output_dir="./evaluation/temp/news"):
    """
    Clears all files and folders inside /evaluation/temp/news,
    but keeps the /evaluation/temp/news directory itself.
    """

    os.makedirs(output_dir, exist_ok=True)

    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)

        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def save_or_load_unique_urls(news_data_folder, parsed_data_folder, extracted_temp_folder):
    """
    If unique_urls.csv already exists in parsed_data_folder, load and return it.
    Otherwise, get unique_urls, save to unique_urls.csv and return it.

    unique_urls is expected to be a set of tuples:
        {(news_url, date_str), ...}
    """

    os.makedirs(parsed_data_folder, exist_ok=True)

    csv_path = os.path.join(parsed_data_folder, "unique_urls.csv")

    # If CSV already exists, load it
    if os.path.exists(csv_path):
        loaded_urls = set()

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                loaded_urls.add((row["news_url"], row["date_str"]))

        print(f"Loaded {len(loaded_urls)} URLs from {csv_path}")
        return loaded_urls

    # Otherwise, we have to load them
    filter_terms = get_filter_terms()
    unique_url_with_dates = set()

    tarfiles = sorted(os.listdir(news_folder))
    for date_tarfile in tqdm(tarfiles):
        tar_filepath = os.path.join(news_folder, date_tarfile)


        # Clear the directory before un-tar
        clear_news_temp(output_dir=extracted_temp_folder)

        # Extract tarfile to this directory
        extract_tar_to_news_temp(
            tar_filepath,
            output_dir=extracted_temp_folder
        )

        found_urls_with_dates = extract_unique_news_urls(extracted_temp_folder, filter_terms)

        #  Merge the urls
        unique_url_with_dates = unique_url_with_dates | found_urls_with_dates

        print(f"Unique urls: {len(found_urls_with_dates)}" )


    # Otherwise, save the provided set
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["news_url", "date_str"])

        for news_url, date_str in sorted(unique_url_with_dates):
            writer.writerow([news_url, date_str])

    print(f"Saved {len(unique_url_with_dates)} URLs to {csv_path}")
    return unique_url_with_dates


def group_urls_by_month(unique_urls):
    """
    Groups (news_url, date_str) tuples by YYYYMM month.

    Args:
        unique_urls: iterable of tuples:
            {(news_url, date_str), ...}

    Returns:
        List of tuples:
            [
                ("202501", [(url1, date_str1), (url2, date_str2), ...]),
                ("202502", [(url3, date_str3), ...]),
                ...
            ]
    """

    grouped = defaultdict(list)

    for news_url, date_str in unique_urls:
        date_str = str(date_str)

        if len(date_str) < 6:
            raise ValueError(f"Invalid date_str: {date_str}")

        month_key = date_str[:6]  # YYYYMM

        grouped[month_key].append((news_url, date_str))

    # Sort months, and sort URLs within each month by full date string
    return [
        (month, sorted(items, key=lambda x: x[1]))
        for month, items in sorted(grouped.items())
    ]


def save_month_results(
    month: str,
    error_data: dict,
    event_data: list[dict],
    news_by_date_folder: str,
):
    """
    Saves monthly error/event data under news_by_date_folder.

    Example outputs:
        news_by_date_folder/202505_error.json
        news_by_date_folder/202505_data.json
    """

    os.makedirs(news_by_date_folder, exist_ok=True)

    error_path = os.path.join(news_by_date_folder, f"{month}_error.json")
    data_path = os.path.join(news_by_date_folder, f"{month}_data.json")

    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(error_data, f, indent=2, ensure_ascii=False)

    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(event_data, f, indent=2, ensure_ascii=False)

    print(f"Saved error data to {error_path}")
    print(f"Saved event data to {data_path}")


def filter_url_date_pairs_by_keyword(url_date_pairs, keyword, case_sensitive=False):
    """
    Return all (url, date_str) pairs where keyword appears in the URL.

    Args:
        url_date_pairs: iterable of (url, date_str)
        keyword: string to search for in the URL
        case_sensitive: whether matching should be case-sensitive

    Returns:
        list of (url, date_str)
    """
    if not case_sensitive:
        keyword = keyword.lower()

    matches = []

    for url, date_str in url_date_pairs:
        url_to_check = url if case_sensitive else url.lower()

        if keyword in url_to_check:
            matches.append((url, date_str))

    return matches


if __name__ == "__main__":

    llm_parser = LLM_news_parser()

    # Example usage

    # Iterate through every tar file
    # Example archive: <raw_archive_root>/gkg/20260213.tar
    paths = get_config().get("paths", {})
    news_folder = os.path.join(paths.get("raw_archive_root", "./raw_data"), "gkg")
    extracted_temp_folder = os.path.join(paths.get("evaluation_temp_root", "./evaluation/temp"), "news")
    # Place to save data
    news_by_date_folder = "./evaluation/news_by_date/"
    
    unique_url_with_dates = save_or_load_unique_urls(news_folder, news_by_date_folder, extracted_temp_folder)

    print(f"\nFound {len(unique_url_with_dates)} total unique URLs.\n")

    # out_urls = list(found_urls)[:10]

    # Examples - wildfire URLS
    # out_urls = [, 'https://www.theblaze.com/news/fire-chief-dei-wildfire-lafd', 'https://www.vox.com/climate/394005/palisades-eaton-wildfire-los-angeles-santa-ana-winds-california-explainer', 'https://www.nbcmiami.com/news/local/people-impacted-california-palisades-eaton-wildfires/3510499/', 'https://uk.news.yahoo.com/live/los-angeles-wildfires-live-updates-5-killed-palisades-and-eaton-fires-spread-sunset-fire-erupts-in-hollywood-hills-141555082.html','https://tribune.com.pk/story/2520994/five-killed-thousands-displaced-as-eaton-fire-burns-10600-acres-in-la', 'https://www.kcra.com/article/eaton-fire-altadena-los-angeles-january-14/63422185']

    # Examples - ice protests
    # out_urls = ["https://abc7.com/live-updates/tensions-flare-downtown-la-anti-ice-protesters-clash-agents-live-updates/16692645/"]

    grouped_articles_by_month = group_urls_by_month(unique_url_with_dates)

    for month, url_date_pairs in grouped_articles_by_month:
        
        # print(month)
        # matches = filter_url_date_pairs_by_keyword(
        #     url_date_pairs,
        #     keyword="explosion",
        # )
        # print(len(matches))
        # for url, date_str in matches:
        #     print(date_str, url)

        # We process the unique urls from the day
        error_data, event_data, token_counts = get_html_from_urls(list(url_date_pairs), llm_parser, timeout=5)


        save_month_results(
            month=month,
            error_data=error_data,
            event_data=event_data,
            news_by_date_folder=news_by_date_folder,
        )

        print("\nURLs with errors:\n")
        for item in error_data["error_urls"]:
            print(item)




    
