FROM python:3.11-slim-bookworm AS base

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files
COPY pyproject.toml /app/
COPY uv.lock* /app/

# Install dependencies
RUN uv sync --no-dev

# Copy source code
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY configs/ /app/configs/

# Create non-root user
RUN addgroup --system --gid 1001 appuser && \
    adduser --system --uid 1001 --gid 1001 appuser && \
    chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["uv", "run", "bike-demand-forecast"]
CMD ["--help"]
