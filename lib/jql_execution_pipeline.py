from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import re
from lib.jira_ticket_pipeline import fetch_jira_tokens , JiraFetchRequest
from langchain_groq import ChatGroq
import os

jql_generation_prompt = PromptTemplate.from_template(
            """You are an expert in Jira Query Language (JQL). Your sole task is to convert a user's natural language request into a valid JQL query.
        You must only respond with the JQL query string and nothing else.

        --- Important JQL Syntax Rules ---
        1.  **Quoting:** Any string value containing spaces or special characters MUST be enclosed in single ('') or double ("") quotes.
            -   Example: `assignee = 'Aru Sharma'`
            -   Example: `summary ~ '"Detailed new feature"'`
        2.  **Usernames:** When searching for an assignee, it is best to use their name in quotes.
        3.  **Linked Issues:** Use `issue in linkedIssues('KEY')`, not `issueLink = KEY`
        4.  **Operators:** Use `=` for single values, `IN` for multiple.
        5.  **Dates:** Example: `created < '2019-01-01'`
        5. **Case:** Fields lowercase, keywords uppercase
        - Example: status = 'Open' ORDER BY created DESC
        6. **Allowed fields:** project, status, assignee, reporter, issuetype,
        priority, created, updated, resolution, labels, summary, description

        --- Examples ---
        User Request: "find all tickets in the 'PROJ' project"
        JQL Query: project = 'PROJ'

        User Request: "show me all open bugs in the 'Mobile' project assigned to Aru Sharma"
        JQL Query: project = 'Mobile' AND issuetype = 'Bug' AND status = 'Open' AND assignee = 'Aru Sharma'

        User Request: "what were the top 5 highest priority issues created last week?"
        JQL Query: created >= -7d ORDER BY priority DESC

        Now, convert the following user request into a JQL query.

        User Request: "{user_query}"
        JQL Query:"""
)

summarization_prompt = PromptTemplate.from_template(
            """You are a helpful assistant. The user asked the following question:

        "{user_query}"

        An AI agent attempted to answer this but failed. As a fallback, we ran a JQL query and got the following raw Jira issue data.
        Please analyze this data and provide a clear, concise, and helpful answer to the user's original question. If the data seems irrelevant or empty, state that you couldn't find relevant information.

        JSON Data:
        {json_data}

        Based on the data, answer the user's question.
        """
    )

llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

def validate_and_fix_jql(jql: str) -> str:
    """
    Validates and fixes common JQL mistakes:
    - Ensures string values with spaces are quoted
    - Fixes linked issue syntax
    - Cleans up extra quotes
    - Normalizes ORDER BY
    """
    fixed_jql = jql.strip()

    # Fix linkedIssues syntax
    fixed_jql = re.sub(r"issueLink\s*=\s*([A-Z]+-\d+)", r"issue in linkedIssues('\1')", fixed_jql)

    # Quote assignee with spaces
    fixed_jql = re.sub(
        r"assignee\s*=\s*([A-Za-z]+\s+[A-Za-z]+)",
        lambda m: f"assignee = '{m.group(1)}'",
        fixed_jql
    )

    # Quote project with spaces
    fixed_jql = re.sub(
        r"project\s*=\s*([A-Za-z]+\s+[A-Za-z]+)",
        lambda m: f"project = '{m.group(1)}'",
        fixed_jql
    )

    # Quote summary search with spaces
    fixed_jql = re.sub(
        r"summary\s*~\s*([^'\"]\S+)",
        lambda m: f"summary ~ '\"{m.group(1)}\"'",
        fixed_jql
    )

    # Normalize IN clauses
    fixed_jql = re.sub(r"\(\s*", "(", fixed_jql)
    fixed_jql = re.sub(r"\s*\)", ")", fixed_jql)
    fixed_jql = re.sub(r",\s*", ", ", fixed_jql)

    # Uppercase operators
    keywords = ["order by", "and", "or", "not", "in", "is", "empty"]
    for kw in keywords:
        fixed_jql = re.sub(rf"\b{kw}\b", kw.upper(), fixed_jql, flags=re.IGNORECASE)


    # Remove double quotes errors
    fixed_jql = fixed_jql.replace("''", "'").replace('""', '"')
    fixed_jql = re.sub(r"[;.,]+$", "", fixed_jql)

    # Normalize ORDER BY
    fixed_jql = re.sub(r"order by", "ORDER BY", fixed_jql, flags=re.IGNORECASE)

    return fixed_jql

async def execute_jql_query(query: str):
    jql_generation_chain = jql_generation_prompt | llm
    generated_jql = jql_generation_chain.invoke({"user_query": query}).content
    validated_jql = validate_and_fix_jql(generated_jql) 
    try:
        request = JiraFetchRequest(
            jql_query=validated_jql,
            limit=50,
            save_to_file=False
        )

        fallback_data = await fetch_jira_tokens(request)
        if not fallback_data:
                return {
                    "response": "The generated JQL query ran successfully but returned no issues. Please try rephrasing your request or be more specific.",
                    "method_used": "fallback",
                    "query_used": validated_jql,
                    "success": True
                }
        json_data = fallback_data.model_dump()
        summarization_chain = summarization_prompt | llm
        final_response = summarization_chain.invoke({
                "user_query": query,
                "json_data": json_data
        }).content

        return {
                "response": final_response,
                "method_used": "fallback",
                "query_used": validated_jql,
                "success": True
            }
    
    except Exception as fallback_e:
            print(f"Fallback JQL execution also failed: {fallback_e}")
            return {
                "response": f"I'm sorry, I couldn't process your request. Both the primary agent and the fallback query failed. The last error was: {fallback_e}",
                "method_used": "failed",
                "query_used": query,
                "success": False
            }
