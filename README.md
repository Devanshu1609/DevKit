# DevKit

An advanced, extensible chatbot platform designed for seamless integration with Jira, Slack, and GitHub. It leverages LangChain , Google-Gemini and Agno to provide intelligent, conversational automation and agent-based workflows. The platform features is easily customizable for a variety of community and project management
 
---

## Features

- **Jira Agent**: Query Jira using natural language, generate JQL, and summarize results with LLM fallback.
- **Slack Agent**: Manage conversations and interact with Slack using agent tools to summarize the conversations inside our channel.
- **Github Agent**: Interact with your repo to ask questions related to the project in natural language.

---

## Getting Started

### Prerequisites
- **Python** 3.8+

### Backend Setup (FastAPI)

1. **Create and activate a virtual environment**  

   On Unix/macOS:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   On Windows:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Install Python dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**  
   Create a `.env` file in the project root:

   ```ini
   # Backend

   # GROQ
   GROQ_API_KEY=<YOUR_GROQ_API_KEY>


   # Jira
   JIRA_API_TOKEN=<YOUR_JIRA_API_TOKEN>
   JIRA_USERNAME=<YOUR_JIRA_USERNAME>
   JIRA_INSTANCE_URL=<JIRA_URL>
   JIRA_CLOUD=True

   # GitHub
   GITHUB_APP_ID=<YOUR_GITHUB_APP_ID>
   GITHUB_REPOSITORY=staru09/Github_analyser
   GITHUB_BRANCH=main
   GITHUB_BASE_BRANCH=main
   GITHUB_APP_PRIVATE_KEY=<YOUR_GITHUB_APP_PRIVATE_KEY>
   SLACK_MCP_XOXP_TOK  # User OAuth token (xoxp-...)
   SLACK_MCP_XOXB_TOKEN  # Bot token (xoxb-...) - limited
   SLACK_MCP_XOXC_TOKEN  # Browser token (xoxc-...)
   SLACK_MCP_XOXD_TOKEN # Browser cookie d (xoxd-...)
   # Server configuration
   SLACK_MCP_PORT  # Port for SSE/HTTP (default: 13080)
   SLACK_MCP_HOST  # Host for SSE/HTTP (default: 127.0.0.1)
   SLACK_MCP_API_KEY  # Bearer token for SSE/HTTP transports
   # Proxy and network settings
   SLACK_MCP_PROXY # Proxy URL for outgoing requests
   SLACK_MCP_USER_AGENT  # Custom User-Agent (Enterprise)
   SLACK_MCP_CUSTOM_TLS  # Custom TLS (Enterprise Slack)
   # TLS/SSL settings
   SLACK_MCP_SERVER_CA  # Path to CA certificate
   SLACK_MCP_SERVER_CA_TOOLKIT  # HTTPToolkit CA
   SLACK_MCP_SERVER_CA_INSECURE  # Trust insecure (NOT RECOMMENDED)
   # Message posting settings
   SLACK_MCP_ADD_MESSAGE_TOOL  # Enable posting
   SLACK_MCP_ADD_MESSAGE_MARK  # Auto-mark as read
   SLACK_MCP_ADD_MESSAGE_UNFURLING  # Enable link unfurling
   #Cache configuration
   SLACK_MCP_USERS_CACHE  # Path to users cache file
   SLACK_MCP_CHANNELS_CACHE  # Path to channels cache
   # Logging
   SLACK_MCP_LOG_LEVEL  # debug, info, warn, error

   ```

4. **Run server**  
   Run this bash command to start the backend server.
   For Github_Bot :-
   ```bash
   python github_agent.py
   ```

   For slack_Bot :-
   ```bash
   python slack_agent.py
   ```

   For jira_Bot :-
   ```bash
   python jira_agent.py
   ```

---
