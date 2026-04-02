import logging
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langchain_mcp_adapters").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

import asyncio
import os
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()


async def main():

    github_config = {
        "transport": "stdio",
        "command": "npx",
        "args": ["@modelcontextprotocol/server-github"],
        "env": {
            "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        }
    }

    # Create MCP Client
    client = MultiServerMCPClient({
        "github": github_config
    })

    tools = await client.get_tools()

    print("\nAvailable GitHub Tools:\n")
    for tool in tools:
        print(f"{tool.name} → {tool.description}")


    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )


    system_prompt = """
You are a GitHub assistant.

You have access to GitHub tools.

When the user asks about repositories, issues, pull requests, or code:
- ALWAYS use the available tools
- NEVER guess GitHub information
- Retrieve real data using tools

Use previous conversation context if needed.
"""

    agent = create_agent(
        model,
        tools=tools,
        system_prompt=system_prompt
    )

    
    chat_history = InMemoryChatMessageHistory()

    print("\nGitHub Agent Ready. Type 'exit' to quit.\n")

   
    while True:

        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("\nExiting agent...\n")
            break

        chat_history.add_user_message(user_input)

        messages = chat_history.messages[-8:]

        result = await agent.ainvoke({
            "messages": messages
        })

        response = result["messages"][-1].content

        print("\nAgent:", response, "\n")

        chat_history.add_ai_message(response)


asyncio.run(main())