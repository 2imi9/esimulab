# Web viewer container (CPU only)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock .python-version ./

# Install dependencies (without the project itself)
RUN uv sync --frozen --no-install-project

# Copy source code
COPY src/ src/
COPY README.md ./

# Install the project
RUN uv sync --frozen

# Shared data volume mount point
VOLUME /app/data

EXPOSE 8000

ENTRYPOINT ["uv", "run"]
CMD ["uvicorn", "esimulab.web.server:app", "--host", "0.0.0.0", "--port", "8000"]
