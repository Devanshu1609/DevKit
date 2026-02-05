import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from agno.agent import Agent
from agno.tools.slack import SlackTools
from agno.models.google import Gemini

from utils.slack_tools import get_channel_info

load_dotenv()

slack_client = WebClient(token=os.getenv("SLACK_TOKEN"))

def list_channels():
    response = slack_client.conversations_list(types="public_channel,private_channel")
    channels = [{"name": ch["name"], "id": ch["id"]} for ch in response["channels"]]

    # Split channels into two halves for parallel tables
    mid = (len(channels) + 1) // 2
    left = channels[:mid]
    right = channels[mid:]

    # Table headers
    header = "| Channel Name           | Channel ID              |"
    boundary = "|-----------------------|-------------------------|"
    print()
    print(f"{header}  {header}")
    print(f"{boundary}  {boundary}")

    # Print rows side by side
    for i in range(max(len(left), len(right))):
        left_row = f"| {left[i]['name']:<21} | {left[i]['id']:<23} |" if i < len(left) else " " * len(boundary)
        right_row = f"| {right[i]['name']:<21} | {right[i]['id']:<23} |" if i < len(right) else " " * len(boundary)
        print(f"{left_row}  {right_row}")
    print()


slack_tools = SlackTools()

agent = Agent(
    model=Gemini(id="gemini-2.5-flash-lite"),
    markdown=True,
    tools=[slack_tools, get_channel_info],
    instructions="""
You are a Slack AI assistant. 
Use the available tools to read and send messages in the channel. 
Always explain your result briefly, properly and in detail.
Do NOT answer any questions from your own knowledge.
You MUST use the provided tools to answer questions about slack.
"""
)

def ask_agent(channel_id: str, user_query: str):
    agent.print_response(
    f"""
    In the Slack channel with ID {channel_id}, {user_query}
    """,
    markdown=True
)

class QueryRequest(BaseModel):
    channel_id: str
    query: str


def ask_question(data: QueryRequest):
    channel_id = data.channel_id
    user_query = data.query

    # Join channel if not already in it
    try:
        slack_client.conversations_join(channel=channel_id)
        print(f"Joined channel {channel_id}")
    except SlackApiError as e:
        if e.response["error"] != "already_in_channel":
            raise HTTPException(status_code=400, detail=f"Slack error: {e.response['error']}")

    ask_agent(channel_id, user_query)


list_channels()
while True:
    channel_id = input("Enter the Slack channel ID to join (or 'exit' to quit): ")
    if channel_id.lower() == 'exit':
        break
    user_query = input("Enter your question about this channel: ")
    ask_question(QueryRequest(channel_id=channel_id, query=user_query))