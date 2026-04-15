import logging
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langchain_mcp_adapters").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

import asyncio
import os
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_groq import ChatGroq

load_dotenv()


SLACK_ENV_VARS = [
    "SLACK_MCP_XOXP_TOKEN",
    "SLACK_MCP_XOXB_TOKEN",
    "SLACK_MCP_XOXC_TOKEN",
    "SLACK_MCP_XOXD_TOKEN",
    "SLACK_MCP_PORT",
    "SLACK_MCP_HOST",
    "SLACK_MCP_API_KEY",
    "SLACK_MCP_PROXY",
    "SLACK_MCP_USER_AGENT",
    "SLACK_MCP_CUSTOM_TLS",
    "SLACK_MCP_SERVER_CA",
    "SLACK_MCP_SERVER_CA_TOOLKIT",
    "SLACK_MCP_SERVER_CA_INSECURE",
    "SLACK_MCP_ADD_MESSAGE_TOOL",
    "SLACK_MCP_ADD_MESSAGE_MARK",
    "SLACK_MCP_ADD_MESSAGE_UNFURLING",
    "SLACK_MCP_USERS_CACHE",
    "SLACK_MCP_CHANNELS_CACHE",
    "SLACK_MCP_LOG_LEVEL",
]

def validate_slack_auth():
    has_xoxp = bool(os.getenv("SLACK_MCP_XOXP_TOKEN"))
    has_xoxb = bool(os.getenv("SLACK_MCP_XOXB_TOKEN"))
    has_xoxc = bool(os.getenv("SLACK_MCP_XOXC_TOKEN"))
    has_xoxd = bool(os.getenv("SLACK_MCP_XOXD_TOKEN"))

    if not has_xoxp and not has_xoxb and not (has_xoxc and has_xoxd):
        raise ValueError(
            "Slack authentication missing.\n"
            "Set one of:\n"
            "- SLACK_MCP_XOXP_TOKEN\n"
            "- SLACK_MCP_XOXB_TOKEN\n"
            "- OR (SLACK_MCP_XOXC_TOKEN + SLACK_MCP_XOXD_TOKEN)"
        )


def build_slack_env():
    env = {
        "PATH": os.environ.get("PATH", "")
    }

    for var in SLACK_ENV_VARS:
        value = os.getenv(var)
        if value:
            env[var] = value

    return env


def get_slack_config():
    transport = os.getenv("SLACK_MCP_TRANSPORT", "stdio").lower()

    validate_slack_auth()
    env = build_slack_env()

    if transport == "stdio":
        return {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "slack-mcp-server@latest", "--transport", "stdio"],
            "env": env
        }

    elif transport in ["sse", "streamable_http"]:
        host = os.getenv("SLACK_MCP_HOST", "127.0.0.1")
        port = os.getenv("SLACK_MCP_PORT", "13080")

        return {
            "transport": transport,
            "url": f"http://{host}:{port}/sse",
            "headers": {
                "Authorization": f"Bearer {os.getenv('SLACK_MCP_API_KEY', '')}"
            }
        }

    else:
        raise ValueError(f"Unsupported transport: {transport}")


async def create_slack_agent():

    slack_config = get_slack_config()

    client = MultiServerMCPClient({
        "slack": slack_config
    })

    tools = await client.get_tools()

    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    system_prompt = """
You are a Slack assistant.

STRICT RULES:
- ALWAYS use Slack tools
- NEVER hallucinate Slack data
- NEVER answer without tool usage

You help with:
- Reading messages
- Sending messages
- Searching channels
- Fetching users

Use conversation context when needed.
"""

    agent = create_agent(
        model,
        tools=tools,
        system_prompt=system_prompt
    )

    return agent


# ✅ OPTIONAL CLI RUNNER (like your GitHub script)
async def main():

    agent = await create_slack_agent()

    chat_history = InMemoryChatMessageHistory()

    print("\n🚀 Slack Agent Ready. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        chat_history.add_user_message(user_input)

        result = await agent.ainvoke({
            "messages": chat_history.messages[-8:]
        })

        response = result["messages"][-1].content

        print("\nAgent:", response, "\n")

        chat_history.add_ai_message(response)


if __name__ == "__main__":
    asyncio.run(main())