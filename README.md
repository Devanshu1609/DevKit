# Community Chatbot for Mifos Projects

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

   # OpenAI
   GOOGLE_API_KEY=<YOUR_OPENAI_API_KEY>

   # Slack
   SLACK_BOT_TOKEN=<YOUR_SLACK_BOT_TOKEN>

   # Jira
   JIRA_API_TOKEN=<YOUR_JIRA_API_TOKEN>
   JIRA_USERNAME=<YOUR_JIRA_USERNAME>
   JIRA_INSTANCE_URL=https://mifosforge.jira.com
   JIRA_CLOUD=True

   # GitHub
   GITHUB_APP_ID=<YOUR_GITHUB_APP_ID>
   GITHUB_REPOSITORY=staru09/Github_analyser
   GITHUB_BRANCH=main
   GITHUB_BASE_BRANCH=main
   GITHUB_APP_PRIVATE_KEY=<YOUR_GITHUB_APP_PRIVATE_KEY>

   ```

4. **Set up github-app-key**
   create a folder `/keys` in root directory and create a file `github-app-key.pem` inside it and add following in it :- 
   ```bash
   paste you github secret key here
   ```



5. **Run server**  
   Run this bash command to start the backend server.
   For Github_Bot :-
   ```bash
   github_agent.py
   ```

   For slack_Bot :-
   ```bash
   python slack_bot.py
   ```

   For jira_Bot :-
   ```bash
   python jira_agent.py
   ```

---