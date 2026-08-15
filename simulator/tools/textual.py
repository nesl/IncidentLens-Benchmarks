from langchain_core.prompts import PromptTemplate

from simulator.tools.utility import parse_model_output, load_mcp_with_agent, get_relevant_messages
from utilities.util import get_config
from langchain_openai import ChatOpenAI
import asyncio
import os


text_agent = None
text_token_s = "<TEXT>"
text_token_e = "</TEXT>"
header_token_s = "<HEADER>"
header_token_e = "</HEADER>"


def _make_direct_llm() -> ChatOpenAI:
    config = get_config()
    openai = config["openai"]
    os.environ["OPENAI_API_KEY"] = openai["api"]
    return ChatOpenAI(model=openai.get("model", "gpt-4"), temperature=0)


async def _ensure_text_agent():
    global text_agent
    if text_agent is None:
        text_agent = await load_mcp_with_agent()
    return text_agent


def _fast_text_header(source: str) -> str:
    source_l = source.lower().strip()
    if source_l == "news":
        return "time,headline,article_text,mentioned_location"
    if "twitter" in source_l:
        return "time,username,text,mentioned_location"
    if "citizen" in source_l:
        return "time,post_id,text,mentioned_location"
    return "time,source,text,mentioned_location"


async def generate_text_data_fast_direct(
    time_str: str,
    event_desc: str,
    source: str,
    step: int,
    save_folder: str = "simulator/generated",
    incident_id: str = "incident_0",
):
    """One direct LLM call, no MCP/ReAct/web tools.

    This is much faster than using the tool-enabled agent. Set fast_mode=False in
    generate_text_data() when you want reference schema lookup and web search.
    """
    llm = _make_direct_llm()
    source = source.strip().lower()
    header = _fast_text_header(source)
    prompt = (
        "Generate synthetic text observations about an incident. Do not call tools or browse. "
        "Generate 1 to 3 rows. Use exactly the required CSV header and no extra columns. "
        "Text sources such as citizen posts and Twitter do not need physical sensor latitude/longitude.\n\n"
        f"Source: {source}\n"
        f"Time: {time_str}\n"
        f"Incident context: {event_desc}\n"
        f"Required header exactly: {header}\n\n"
        f"Format the header between {header_token_s} and {header_token_e}. "
        f"Format each CSV row between {text_token_s} and {text_token_e}. "
        "Use CSV quoting for fields that contain commas."
    )
    response = llm.invoke(prompt)
    print("AI messages:", [response.content])
    return parse_model_output(
        response.content,
        source,
        step,
        [header_token_s, header_token_e],
        [text_token_s, text_token_e],
        save_folder=save_folder,
        filename_prefix=incident_id,
    )


async def generate_text_data(
    time_str: str,
    event_desc: str,
    source: str,
    step: int,
    save_folder: str = "simulator/generated",
    incident_id: str = "incident_0",
    fast_mode: bool = True,
):
    """Generate text-like simulated data.

    fast_mode=True avoids MCP/ReAct/web lookup and is much faster. It also means
    news/citizen/twitter schemas are simple fixed schemas. Set fast_mode=False to
    use the older tool-enabled path.
    """
    if fast_mode:
        return await generate_text_data_fast_direct(
            time_str=time_str,
            event_desc=event_desc,
            source=source,
            step=step,
            save_folder=save_folder,
            incident_id=incident_id,
        )

    agent = await _ensure_text_agent()

    source = source.strip().lower()
    if source == "news":
        prompt = PromptTemplate(
            template=(
                "You are helping simulate an incident by producing text data. There are tools "
                "which you can use as reference to produce data of a similar schema. You may be "
                "producing data at a particular time step, in which case I will also provide the "
                "time and previous simulated context. Here is the time: {time}. Here is the event "
                "description: {event_description}. Here is the data source you should try to generate "
                "data for: {source}. Since this is news data, you should do a bit of web search to "
                "identify whether the writing style matches your created article. Return simulated "
                "data as text output. News may mention the incident location in the article text, but "
                "do not invent a sensor latitude/longitude for news."
            ),
            input_variables=["time", "event_description", "source"],
        )

        formatting = (
            " Please format your output as a comma-separated list of strings between "
            + text_token_s
            + " and "
            + text_token_e
            + " tags. You may use multiple tags if you wish to generate data from multiple "
            "articles, but generate only up to 5 at any one time. Please also use the "
            + header_token_s
            + header_token_e
            + " tags to generate a comma-separated list of column headers. For news, use "
            "time, headline, article_text, and mentioned_location as headers."
        )
    else:
        prompt = PromptTemplate(
            template=(
                "You are helping simulate an incident by producing text data. There are tools which "
                "you can use as reference to produce data of a similar schema (for example, twitter_data "
                "has a particular format, whereas citizen_data is also social media data with a particular "
                "format). You may be producing data at a particular time step, in which case I will also "
                "provide the time and previous simulated context. Here is the time: {time}. Here is the "
                "event description: {event_description}. Here is the data source you should try to generate "
                "data for: {source}. Try to use the data from the tools to match the schema and format. "
                "Return simulated data as text output and verify that the output format matches the "
                "reference data format. Do not force citizen/twitter-style text data to have physical "
                "sensor latitude/longitude; the location can appear naturally in the text if the source "
                "format normally includes it."
            ),
            input_variables=["time", "event_description", "source"],
        )

        formatting = (
            " Please format your output as comma-separated rows between "
            + text_token_s
            + " and "
            + text_token_e
            + " tags. You may use multiple tags if you wish to generate data from multiple "
            "text sources, but generate only up to 5 at any one time. Please also use the "
            + header_token_s
            + header_token_e
            + " tags to generate a comma-separated list of column headers. Use the reference "
            "schema when possible, but do not add latitude/longitude solely for citizen/twitter "
            "posts unless the reference data already has those fields."
        )

    prompt_text = prompt.format(time=time_str, event_description=event_desc, source=source) + formatting
    response = await agent.ainvoke({"messages": [{"role": "user", "content": prompt_text}]})

    tool_messages, ai_messages = get_relevant_messages(response)
    print("AI messages:", ai_messages)

    return parse_model_output(
        ai_messages[-1],
        source,
        step,
        [header_token_s, header_token_e],
        [text_token_s, text_token_e],
        save_folder=save_folder,
        filename_prefix=incident_id,
    )


if __name__ == "__main__":
    asyncio.run(
        generate_text_data(
            "May 20, 2025 at 3pm",
            "The start of a wildfire in Culver City, Los Angeles, with several buildings burnt down.",
            "citizen",
            1,
        )
    )
