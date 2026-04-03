FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

# Copy dependency files first (layer cache)
COPY pyproject.toml .
COPY uv.lock* .

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

CMD ["/app/.venv/bin/python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
