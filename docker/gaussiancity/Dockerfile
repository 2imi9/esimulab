# GaussianCity inference container
# Requires CUDA 11.8 for compiled extensions (incompatible with Blackwell cu130)
# Based on: https://github.com/hzxie/GaussianCity
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    git curl libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install --upgrade pip

# PyTorch for CUDA 11.8
RUN pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118

# GaussianCity dependencies
RUN pip install easydict lxml numpy open3d opencv-python pillow plyfile \
    pynvml scipy tensorboard tqdm

WORKDIR /app

# Clone GaussianCity
RUN git clone --depth 1 https://github.com/hzxie/GaussianCity.git /app

# Compile CUDA extensions
RUN cd /app/extensions/footprint_extruder && pip install .
RUN cd /app/extensions/diff_gaussian_rasterization && pip install .
RUN cd /app/extensions/grid_encoder && pip install .
RUN cd /app/extensions/voxlib && pip install .

# Optional: flash attention (may fail on some GPUs)
RUN pip install flash_attn || true

ENTRYPOINT ["python3.11"]
CMD ["scripts/inference.py"]
