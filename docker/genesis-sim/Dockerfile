# Genesis simulation container with NVIDIA GPU support
# Requires: docker run --gpus all
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# System dependencies for rasterio, GDAL, pygrib, Genesis rendering
# Add deadsnakes PPA for Python 3.11 on Ubuntu 22.04
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    gdal-bin \
    libgdal-dev \
    libeccodes-dev \
    libeccodes-tools \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libegl1-mesa \
    libgl1-mesa-dev \
    libxrender1 \
    libxext6 \
    libx11-6 \
    xvfb \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock .python-version ./

# Install core dependencies (without project)
RUN uv sync --frozen --no-install-project

# Copy source code
COPY src/ src/
COPY README.md ./

# Install the project itself
RUN uv sync --frozen

# Install GPU extras AFTER uv sync so they don't get overwritten
RUN uv pip install \
    torch --index-url https://download.pytorch.org/whl/cu124 \
    && uv pip install \
    genesis-world \
    earth2studio \
    nvidia-physicsnemo

# Shared data volume mount point
VOLUME /app/data

# Headless rendering: Xvfb provides a virtual X display for Genesis/pyglet
ENV DISPLAY=:99

# Entrypoint starts Xvfb then runs the command
COPY docker/genesis-sim/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uv", "run", "python", "-m", "esimulab"]
