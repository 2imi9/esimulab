# Genesis simulation container with NVIDIA GPU support
# Requires: docker run --gpus all
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies for rasterio, GDAL, and Genesis rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    gdal-bin \
    libgdal-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

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

ENTRYPOINT ["uv", "run"]
CMD ["python", "-m", "esimulab"]
