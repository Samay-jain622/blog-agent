# Blog Agent (uv + LangGraph)

AI-powered blog generation system with:
- LangGraph orchestration
- Groq + Mistral LLMs
- Tavily research
- Streamlit frontend
- Docker support

## Run locally

uv sync
uv run streamlit run frontend.py

## Docker

docker build -t blog-agent .
docker run -p 8501:8501 --env-file .env blog-agent

https://blog-agent-app-latest.onrender.com
