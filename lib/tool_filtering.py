# tool_filtering.py

import os
import time
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    batch_size=32,
    google_api_key=GOOGLE_API_KEY
)

DB_PATH = "./chroma_db"

def get_vectorstore(collection_name: str):
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)


def tool_to_documents(tools, source):
    docs = []
    for tool in tools:
        content = f"""
        Tool Name: {tool.name}
        Description: {tool.description or ''}
        """
        chunks = text_splitter.split_text(content)
        for chunk in chunks:
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "tool_name": tool.name,
                    "source": source
                }
            ))
    return docs


def safe_add_documents(vectorstore, docs, batch_size=10):
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        retries = 3
        for attempt in range(retries):
            try:
                vectorstore.add_documents(batch)
                break
            except Exception as e:
                if "429" in str(e):
                    wait_time = 50
                    print(f"Rate limit hit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise e


def store_tools_if_not_exists(vectorstore, tools, source):
    existing = vectorstore.get()
    existing_names = set()

    if existing and "metadatas" in existing:
        existing_names = set([
            meta.get("tool_name")
            for meta in existing["metadatas"]
            if meta.get("source") == source
        ])

    new_tools = [t for t in tools if t.name not in existing_names]

    if not new_tools:
        print(f" {source} tools already stored. Skipping embedding.")
        return

    docs = tool_to_documents(new_tools, source)
    safe_add_documents(vectorstore, docs, batch_size=5)
    print(f" Stored {len(new_tools)} new {source} tools.")


def initialize_tool_db(jira_tools=None, github_tools=None, slack_tools=None):
    stores = {}

    if jira_tools:
        jira_vs = get_vectorstore("Devpit-Tools-jira")
        try:
            if jira_vs._collection.count() > 0:
                print(" Jira tools already stored. Skipping embedding.")
            else:
                store_tools_if_not_exists(jira_vs, jira_tools, "jira")
        except Exception:
            store_tools_if_not_exists(jira_vs, jira_tools, "jira")
        stores["jira"] = jira_vs

    if github_tools:
        github_vs = get_vectorstore("Devpit-Tools-github")
        try:
            if github_vs._collection.count() > 0:
                print(" GitHub tools already stored. Skipping embedding.")
            else:
                store_tools_if_not_exists(github_vs, github_tools, "github")
        except Exception:
            store_tools_if_not_exists(github_vs, github_tools, "github")
        stores["github"] = github_vs

    if slack_tools:
        slack_vs = get_vectorstore("Devpit-Tools-slack")
        try:
            if slack_vs._collection.count() > 0:
                print(" Slack tools already stored. Skipping embedding.")
            else:
                store_tools_if_not_exists(slack_vs, slack_tools, "slack")
        except Exception:
            store_tools_if_not_exists(slack_vs, slack_tools, "slack")
        stores["slack"] = slack_vs

    return stores


def dense_mmr_retrieve(vectorstore, query, k=5, fetch_k=20):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k}
    )
    print(f"MMR retrieval (k={k}, fetch_k={fetch_k})")
    return retriever.invoke(query)


def map_docs_to_tools(docs, all_tools):
    tool_map = {tool.name: tool for tool in all_tools}
    selected = []
    seen = set()
    for doc in docs:
        name = doc.metadata.get("tool_name")
        if name in tool_map and name not in seen:
            selected.append(tool_map[name])
            seen.add(name)
    return selected


def get_filtered_tools(query, vectorstore, all_tools):
    docs = dense_mmr_retrieve(vectorstore, query)
    return map_docs_to_tools(docs, all_tools)