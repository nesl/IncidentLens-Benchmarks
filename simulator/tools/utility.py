import csv
import json
import os
from io import StringIO
from typing import Any, Dict, List, Optional, get_type_hints

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages.tool import ToolMessage
from langchain_core.messages.ai import AIMessage

from utilities.util import get_config


field_separator = ","
OBSERVATION_LOG_NAME = "observations.txt"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def append_observation_log(record: Dict[str, Any], save_folder: str = "simulator/generated") -> str:
    """Append one observation metadata record as JSONL to observations.txt.

    The file remains human-readable while still being easy to parse later.  Each
    line is one generated observation or sensor used by the simulator.
    """
    ensure_dir(save_folder)
    log_path = os.path.join(save_folder, OBSERVATION_LOG_NAME)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return log_path


def get_relevant_messages(agent_response):
    tool_messages = []
    ai_messages = []

    for x in agent_response["messages"]:
        if isinstance(x, ToolMessage):
            print(vars(x))
            tool_messages.append({"content": x.content, "name": x.name})
        elif isinstance(x, AIMessage):
            ai_messages.append(x.content)

    return tool_messages, ai_messages


def tool_check(tools):
    for t in tools:
        func = getattr(t, "func", None)
        if not func:
            continue
        try:
            get_type_hints(func, include_extras=True)
        except Exception as e:
            print(f"\n❌ Tool failed type-hint eval: {t.name}")
            print("Function:", func)
            print("Error:", repr(e))


def _extract_between(text: str, start_token: str, end_token: str) -> str:
    if start_token not in text or end_token not in text:
        raise ValueError(f"Missing expected tokens {start_token!r} / {end_token!r} in model output")
    return text.split(start_token, 1)[1].split(end_token, 1)[0]


def _parse_csv_row(row: str) -> List[str]:
    reader = csv.reader(StringIO(row), skipinitialspace=True)
    return next(reader)


def parse_model_output(
    simulated_response: str,
    source: str,
    step: int,
    header_tokens,
    text_tokens,
    save_folder: str = "simulator/generated",
    filename_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse an LLM response, save the generated CSV, and return parsed rows.

    The return value lets simulator.py create external observation metadata
    without encoding metadata in the filename.
    """
    header_token_s, header_token_e = header_tokens
    text_token_s, text_token_e = text_tokens

    header_info = _extract_between(simulated_response, header_token_s, header_token_e)
    headers = [h.strip() for h in header_info.split(field_separator)]
    print(header_info)

    text_data = []
    for sensor_data in simulated_response.split(text_token_s)[1:]:
        data_str = sensor_data.split(text_token_e, 1)[0]
        if data_str.strip():
            text_data.append(data_str.strip())

    rows = []
    for row in text_data:
        parsed_row = _parse_csv_row(row)
        rows.append(parsed_row)

    ensure_dir(save_folder)
    safe_source = source.strip().replace(" ", "_")
    if filename_prefix:
        filename = f"{filename_prefix}_{step}_{safe_source}.csv"
    else:
        filename = f"{step}_{safe_source}.csv"
    save_filepath = os.path.join(save_folder, filename)

    with open(save_filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)

    dict_rows = []
    for row in rows:
        dict_rows.append({headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))})

    return {
        "filepath": save_filepath,
        "headers": headers,
        "rows": rows,
        "dict_rows": dict_rows,
    }


async def load_mcp_with_agent():
    config = get_config()
    openai = config["openai"]
    openai_api_key = openai["api"]
    mcp_config = config["mcp"]["simulation"]
    mcp_url = "http://" + mcp_config["host"] + ":" + mcp_config["port"] + mcp_config["path"]

    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ.pop("LANGSMITH_API_KEY", None)
    os.environ.pop("LANGCHAIN_API_KEY", None)

    mcp_client = MultiServerMCPClient(
        {
            "simulation": {
                "url": mcp_url,
                "transport": "streamable_http",
            }
        }
    )
    tools = await mcp_client.get_tools()

    ddg_search = DuckDuckGoSearchResults()
    tools.append(ddg_search)
    tool_check(tools)

    os.environ["OPENAI_API_KEY"] = openai_api_key
    agent = create_react_agent(
        "openai:gpt-4",
        tools,
    )

    return agent


if __name__ == "__main__":
    example_text = '1, "Tue, 20 May 2025 15:00:00 +0000 (UTC)", 34.0219, -118.4814, "Culver City Wildfire", "wildfire", "A wildfire has started in Culver City, Los Angeles, with several buildings burnt down."'

    parsed_row = _parse_csv_row(example_text)

    with open("test.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(parsed_row)
