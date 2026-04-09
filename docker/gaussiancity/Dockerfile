# GaussianCity inference container
# CUDA 11.8 required — incompatible with Blackwell cu130, hence isolated container
# Paper: https://arxiv.org/abs/2406.06526
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    git curl wget \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN python3 -m ensurepip && python3 -m pip install --upgrade pip

# PyTorch for CUDA 11.8
RUN pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118

# GaussianCity Python deps
RUN pip install easydict lxml numpy open3d opencv-python-headless pillow \
    plyfile pynvml scipy tqdm

WORKDIR /app

# Clone GaussianCity
RUN git clone --depth 1 https://github.com/hzxie/GaussianCity.git /app

# Compile C++ extension (no CUDA needed)
RUN cd /app/extensions/footprint_extruder && pip install .

# Compile CUDA extensions (needs nvcc from CUDA 11.8)
RUN cd /app/extensions/diff_gaussian_rasterization && pip install . || \
    echo "WARNING: diff_gaussian_rasterization build failed — will retry at runtime"
RUN cd /app/extensions/grid_encoder && pip install . || \
    echo "WARNING: grid_encoder build failed — will retry at runtime"
RUN cd /app/extensions/voxlib && pip install . || \
    echo "WARNING: voxlib build failed — will retry at runtime"

# Shared data volume
VOLUME /app/data
VOLUME /models

ENTRYPOINT ["python3"]
CMD ["scripts/inference.py", "--rest_ckpt", "/models/rest.pth", "--bldg_ckpt", "/models/bldg.pth", "--output_file", "/app/data/gaussiancity/rendering.mp4"]
