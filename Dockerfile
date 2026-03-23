# -------- Base Image --------
FROM python:3.12-slim-trixie

# Install uv (best method)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# -------- Dependency Layer (cache optimized) --------
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

# -------- App Layer --------
COPY . .

RUN uv sync --frozen

# Create required dirs
RUN mkdir -p images /root/.streamlit

# Streamlit config
RUN echo "[server]\n\
headless = true\n\
port = 8501\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml

# Expose port
EXPOSE 8501

# Run app
CMD ["uv", "run", "streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]