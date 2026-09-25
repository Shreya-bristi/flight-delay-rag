# API image used by both the API service and indexing job
# Dependencies are installed from uv.lock for reproducible builds

FROM ghcr.io/astral-sh/uv:0.12.3 AS uv

FROM python:3.12.14-slim-bookworm AS builder
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /build
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra ml --no-install-project --python /usr/local/bin/python3.12

FROM python:3.12.14-slim-bookworm
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY src ./src
COPY scripts ./scripts
COPY data ./data
COPY system_prompt.md ./system_prompt.md

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    SYSTEM_PROMPT_PATH=/app/system_prompt.md \
    HF_HOME=/models

# Run the application as a non-root user
RUN useradd -m -u 1000 app && mkdir -p /models && chown -R app:app /app /models
USER app

EXPOSE 8000

# Container liveness check, Kubernetes uses /ready separately for readiness
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health',timeout=3).status_code==200 else 1)"
CMD ["python", "-m", "uvicorn", "flight_delay.api:app", "--host", "0.0.0.0", "--port", "8000"]
