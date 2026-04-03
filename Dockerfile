FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Copy dependency files first (layer cache)
COPY pyproject.toml uv.lock* ./
COPY openenv.yaml .

RUN uv sync --no-dev

# Now copy the rest of the code
COPY server/ ./server/
COPY client/ ./client/

# Create non-root user and set permissions
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV ENABLE_WEB_INTERFACE=true
ENV PORT=8000
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/.venv/bin/uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
