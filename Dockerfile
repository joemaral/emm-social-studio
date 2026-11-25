# Standalone Cloud Run Microservice for Automated Social Media Creation & Distribution
FROM python:3.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.9.24 /uv /uvx /bin/

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-dejavu-core \
    fonts-liberation \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

WORKDIR /service
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8080

COPY pyproject.toml ./pyproject.toml
RUN uv pip install --system --no-cache \
    fastapi uvicorn httpx pillow "psycopg[binary]" psycopg2-binary pydantic requests sqlalchemy google-cloud-storage

COPY app/ ./app/

USER appuser
EXPOSE 8080

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
