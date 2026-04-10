import os
import json
import re
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bs4 import BeautifulSoup
from atlassian import Jira
from dotenv import load_dotenv

load_dotenv()

jira_client: Optional[Jira] = None

#Pydantic Models
class JiraFetchRequest(BaseModel):
    jql_query: str     
    limit: int = 50
    save_to_file: bool = False
    filename: str = "jira_data.jsonl"
    
class JiraIssue(BaseModel):
    id: Optional[str]
    key: Optional[str]
    summary: Optional[str]
    description: Optional[str]
    status: Optional[str]
    priority: Optional[str]
    created: Optional[str]
    assignee: Optional[str]
    project: Optional[str]
    
class JiraFetchResponse(BaseModel):
    count: int
    issues: List[JiraIssue]
    success: bool
    
# # Lifespan (Startup / Shutdown)
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     initialize_jira_client()
#     yield

# app = FastAPI(
#     title="Jira Data Pipeline API",
#     version="1.0.0",
#     lifespan=lifespan,
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

#Initialisation
def initialize_jira_client():
    """Initialize Jira client using environment variables"""
    global jira_client

    url = os.getenv("JIRA_INSTANCE_URL") or os.getenv("JIRA_URL")
    username = os.getenv("JIRA_USERNAME")
    token = os.getenv("JIRA_API_TOKEN")

    if not all([url, username, token]):
        raise RuntimeError("Missing required Jira environment variables")

    jira_client = Jira(
        url=url,
        username=username,
        password=token,
        cloud=True
    )

    
    
#Utility Functions

def clean_text(text) -> str:
    """Strip HTML, normalize whitespace, redact emails"""
    if not text:
        return ""

    if isinstance(text, dict):
        return json.dumps(text)

    try:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")
    except Exception:
        text = re.sub(r"<[^<]+?>", "", text)

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[REDACTED]", text)

    return text

def normalize_issues(issues: list) -> List[JiraIssue]:
    """Convert raw Jira issues into a flat schema"""
    normalized = []

    for issue in issues:
        fields = issue.get("fields", {})

        normalized.append(
            JiraIssue(
                id=issue.get("id"),
                key=issue.get("key"),
                summary=fields.get("summary"),
                description=clean_text(fields.get("description")),
                status=fields.get("status", {}).get("name"),
                priority=fields.get("priority", {}).get("name"),
                created=fields.get("created"),
                assignee=fields.get("assignee", {}).get("displayName")
                if fields.get("assignee") else None,
                project=fields.get("project", {}).get("name")
                if fields.get("project") else None,
            )
        )

    return normalized

def save_to_jsonl(data: List[JiraIssue], filename: str):
    """Persist normalized Jira data to JSONL"""
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(item.model_dump_json() + "\n")


# Core Jira Logic
def fetch_jira_issues(jql: str, limit: int) -> list:
    global jira_client

    if jira_client is None:
        print("⚡ Initializing Jira client...")
        initialize_jira_client()

    if jira_client is None:
        raise RuntimeError("Jira client STILL not initialized")

    results = jira_client.jql(jql, limit=limit)
    return results.get("issues", [])


async def fetch_jira_tokens(request: JiraFetchRequest):
    print(request.jql_query)
    """
    Fetch, normalize, and optionally persist Jira issues
    """
    try:
        raw_issues = fetch_jira_issues(request.jql_query, request.limit)
        normalized = normalize_issues(raw_issues)
        if request.save_to_file:
            save_to_jsonl(normalized, request.filename)

        return JiraFetchResponse(
            count=len(normalized),
            issues=normalized,
            success=True,
        )

    except Exception as e:
        print("Jira Fetch Error:", e)
        return JiraFetchResponse(
            count=0,
            issues=[],
            success=False
        )
