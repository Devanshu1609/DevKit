import os
import re
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper
from langchain.agents import create_agent

load_dotenv()

required_vars = [
    "GOOGLE_API_KEY",
    "GITHUB_APP_ID",
    "GITHUB_REPOSITORY",
    "GITHUB_APP_PRIVATE_KEY_PATH",
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

key_path = os.getenv("GITHUB_APP_PRIVATE_KEY_PATH")
with open(key_path, "r") as f:
    private_key = f.read()

os.environ["GITHUB_APP_PRIVATE_KEY"] = private_key

def sanitize_tool_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    return name.strip("_")

github = GitHubAPIWrapper()
toolkit = GitHubToolkit.from_github_api_wrapper(github)
tools = toolkit.get_tools()

for tool in tools:
    tool.name = sanitize_tool_name(tool.name)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
)

system_prompt = """
You are a GitHub AI assistant.

You help users interact with a GitHub repository using the available tools.
Use tools whenever repository actions are required.
Explain actions briefly and clearly.
"""

agent = create_agent(
    llm,
    tools=tools,
    system_prompt=system_prompt,
)

print("GitHub Agent Ready Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in {"exit", "quit"}:
        break

    result = agent.invoke({
    "messages": [
        {"role": "user", "content": user_input}
    ]
    })
    print(result.get("output", result))
