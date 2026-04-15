# main.py

import logging
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langchain_mcp_adapters").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

import asyncio
import os
import functools
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent 
from langchain.agents import create_agent

from lib.jql_execution_pipeline import execute_jql_query
from lib.tool_filtering import initialize_tool_db, get_filtered_tools
from prompts.agent_prompt import JIRA_SYSTEM_PROMPT, CLASSIFIER_PROMPT

load_dotenv()

BOOLEAN_FIELDS = {
    "jira_get_all_projects": ["include_archived"],
    "jira_get_issue": ["update_history"],
    "jira_search_fields": ["refresh"],
    "jira_get_field_options": ["values_only"],
    "jira_get_issue_dates": ["include_status_changes", "include_status_summary"],
    "jira_get_issue_sla": ["working_hours_only", "include_raw_dates"],
    "jira_batch_create_issues": ["validate_only"],
}

def coerce_booleans(tool_name: str, kwargs: dict) -> dict:
    """Coerce any string 'true'/'false' values to actual booleans."""
    for key, value in kwargs.items():
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            kwargs[key] = value.strip().lower() == "true"
    return kwargs

def _clean_description(desc: str, max_chars: int = 150) -> str:
    """Remove ctx arg lines and truncate description to save tokens."""
    lines = desc.splitlines()
    filtered = []
    skip_ctx = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("ctx:"):
            skip_ctx = True
            continue
        if skip_ctx and stripped and not stripped.startswith("ctx"):
            skip_ctx = False
        if not skip_ctx:
            filtered.append(line)

    cleaned = "\n".join(filtered).strip()

    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(".", 1)[0] + "."

    return cleaned


def _strip_schema_descriptions(tool):
    """Remove verbose 'description' strings from args_schema properties."""
    schema = tool.args_schema
    if not isinstance(schema, dict):
        return

    props = schema.get("properties", {})
    for field_name, field_def in props.items():
        if "description" in field_def and isinstance(field_def["description"], str):
            # Keep only first sentence, max 60 chars
            short = field_def["description"].split(".")[0].strip()
            if len(short) > 60:
                short = short[:60]
            field_def["description"] = short


def _fix_boolean_schema(tool):
    """Change boolean fields to string type in schema so LLM string values pass validation."""
    schema = tool.args_schema
    if not isinstance(schema, dict):
        return

    props = schema.get("properties", {})
    for field_name, field_def in props.items():
        if field_def.get("type") == "boolean":
            field_def["type"] = "string"
            field_def["description"] = field_def.get("description", "") + " (true/false)"


def patch_mcp_tools(tools):
    """Strip ctx kwarg, fix boolean schemas, coerce booleans, clean descriptions."""
    patched = []
    for tool in tools:
        if not isinstance(tool, StructuredTool) or tool.coroutine is None:
            patched.append(tool)
            continue

        original = tool.coroutine
        tool_name = tool.name

        @functools.wraps(original)
        async def safe_coroutine(*args, _orig=original, _name=tool_name, **kwargs):
            kwargs.pop("ctx", None)
            kwargs = coerce_booleans(_name, kwargs)
            return await _orig(*args, **kwargs)

        object.__setattr__(tool, "coroutine", safe_coroutine)

        if tool.description:
            object.__setattr__(tool, "description", _clean_description(tool.description))

        _strip_schema_descriptions(tool)
        _fix_boolean_schema(tool) 
        patched.append(tool)

    return patched


def can_tools_handle(query, tools):
    if not tools:
        return False

    tool_info = "\n".join([tool.name for tool in tools])
    chain = PromptTemplate.from_template(CLASSIFIER_PROMPT) | ChatGroq(
        model="allam-2-7b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )
    result = chain.invoke({"query": query, "tools": tool_info}).content.strip().upper()
    return result == "JIRA"


async def main():
    jira_config = {
        "transport": "stdio",
        "command": "uvx",
        "args": ["mcp-atlassian"],
        "env": {
            "JIRA_URL": os.getenv("JIRA_URL"),
            "JIRA_USERNAME": os.getenv("JIRA_USERNAME"),
            "JIRA_API_TOKEN": os.getenv("JIRA_API_TOKEN"),
        }
    }

    client = MultiServerMCPClient({"jira": jira_config})
    print("client:", client)

    raw_jira_tools = await client.get_tools()
    print("Raw Jira Tools:", raw_jira_tools)
    jira_tools = patch_mcp_tools(raw_jira_tools) 
    print("Patched Jira Tools:", jira_tools)
    stores = initialize_tool_db(jira_tools)
    jira_vs = stores["jira"]

    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    chat_history = InMemoryChatMessageHistory()
    print("\nHybrid Jira Agent Ready (MCP + JQL). Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("\nExiting agent...\n")
            break

        chat_history.add_user_message(user_input)

        filtered_tools = get_filtered_tools(user_input, jira_vs, jira_tools)
        print("🔧 Filtered Tools:", [t.name for t in filtered_tools])

        if not filtered_tools:
            print("No relevant tools found → using JQL")
            decision = False
        else:
            decision = can_tools_handle(user_input, filtered_tools)

        print(f"Decision: {'JIRA' if decision else 'JQL'}")

        if not decision:
            print("Using JQL Pipeline...\n")
            result = await execute_jql_query(user_input)
            response = result["response"]

        else:
            print("Using MCP Agent...\n")
            try:
                agent = create_agent(     
                    model,
                    tools=filtered_tools,
                    system_prompt=JIRA_SYSTEM_PROMPT, 
                )
                result = await agent.ainvoke({
                    "messages": chat_history.messages[-8:]
                })
                response = result["messages"][-1].content

            except Exception as e:
                print(f"MCP Agent failed: {e}")
                print("Falling back to JQL...\n")
                fallback = await execute_jql_query(user_input)
                response = fallback["response"]

        print(f"\nAgent: {response}\n")
        chat_history.add_ai_message(response)


asyncio.run(main())