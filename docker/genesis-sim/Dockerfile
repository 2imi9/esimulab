# Genesis simulation container with NVIDIA GPU support
# Requires: docker run --gpus all
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# System dependencies for rasterio, GDAL, pygrib, Genesis rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    gdal-bin \
    libgdal-dev \
    libgrib-api-dev \
    libeccodes-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libegl1-mesa \
    libgl1-mesa-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock .python-version ./

# Install core dependencies
RUN uv sync --frozen --no-install-project

# Install GPU extras: genesis-world, earth2studio, and their deps
RUN uv pip install \
    torch --index-url https://download.pytorch.org/whl/cu124 \
    && uv pip install \
    genesis-world \
    earth2studio \
    nvidia-physicsnemo

# Copy source code
COPY src/ src/
COPY README.md ./

# Install the project itself
RUN uv sync --frozen

# Shared data volume mount point
VOLUME /app/data

ENTRYPOINT ["uv", "run"]
CMD ["python", "-m", "esimulab"]
