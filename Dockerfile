# ComfyUI Pipeline Service
#
# Features:
#   - Custom nodes pre-installed for consistency across fleet
#   - TTF custom nodes for multi-person detection workflows
#   - Models mounted from host (not baked in)
#   - API wrapper for simplified workflow execution
#   - Health checks for fleet orchestration
#
# Build:
#   docker build -t comfyui-pipeline:latest .
#
# Run:
#   docker run --gpus all -p 8188:8188 -p 8189:8189 \
#     -v H:/models:/models \
#     -v ./workflows:/app/workflows \
#     comfyui-pipeline:latest

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ============================================================
# SYSTEM DEPENDENCIES
# ============================================================
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    build-essential \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && pip install --no-cache-dir --upgrade pip

WORKDIR /app

# ============================================================
# PYTORCH + COMFYUI
# ============================================================
# PyTorch 2.4+ required for ComfyUI (torch.library.custom_op)
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Pin numpy to 1.x (numpy 2.x breaks many packages)
RUN pip install --no-cache-dir "numpy<2"

RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI

WORKDIR /app/ComfyUI

RUN pip install --no-cache-dir -r requirements.txt

# Re-pin numpy after requirements.txt may have changed it
RUN pip install --no-cache-dir "numpy<2"

# ============================================================
# CUSTOM NODES - Community (baked in for fleet consistency)
# ============================================================
WORKDIR /app/ComfyUI/custom_nodes

# ComfyUI Manager - node management
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git

# KJNodes - Utility nodes (resize, batch, etc.)
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git

# WanVideoWrapper - Wan 2.1/2.2 video generation
RUN git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git

# Segment Anything 2 - SAM2 segmentation
RUN git clone https://github.com/kijai/ComfyUI-segment-anything-2.git

# VideoHelperSuite - Video loading/saving
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git

# WanAnimatePreprocess - Pose detection for animation
RUN git clone https://github.com/kijai/ComfyUI-WanAnimatePreprocess.git

# SAM 3D Objects - Image+mask to 3D GLB/STL/PLY
RUN git clone https://github.com/PozzettiAndrea/ComfyUI-SAM3DObjects.git

# Frame Interpolation - RIFE/FILM for FPS upscaling (Pipeline C)
RUN git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git

# ProPainter - Flow-based video inpainting/outpainting (Pipeline D/E)
RUN git clone https://github.com/daniabib/ComfyUI_ProPainter_Nodes.git

# Motion Capture - GVHMR-based mocap extraction + SMPL export (Pipeline B)
RUN git clone https://github.com/PozzettiAndrea/ComfyUI-MotionCapture.git

# MotionCapture dependencies - required node packs
RUN git clone https://github.com/PozzettiAndrea/ComfyUI-CameraPack.git
RUN git clone https://github.com/PozzettiAndrea/ComfyUI-HyMotion.git
RUN git clone https://github.com/PozzettiAndrea/ComfyUI-Env-Manager.git

# ============================================================
# CUSTOM NODES - TTF (internal nodes from repo)
# ============================================================

# Multi-Person Detector - bbox detection & tracking for character workflows
# Provides: MultiPersonBboxDetector, SaveBboxesJSON, LoadBboxesJSON,
#           FilterBboxesByPerson, LoadTrackedBboxesForPerson, LoadTrackedBboxesInfo
RUN mkdir -p /app/ComfyUI/custom_nodes/ComfyUI-MultiPersonDetector
COPY custom_nodes/ComfyUI-MultiPersonDetector/__init__.py \
     /app/ComfyUI/custom_nodes/ComfyUI-MultiPersonDetector/__init__.py

# ============================================================
# PYTHON DEPENDENCIES
# ============================================================
WORKDIR /app/ComfyUI

# API wrapper dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    httpx \
    requests

# Common dependencies for custom nodes
RUN pip install --no-cache-dir \
    opencv-python-headless \
    scikit-image \
    scipy \
    onnxruntime-gpu \
    accelerate \
    matplotlib \
    onnx \
    pyyaml \
    huggingface_hub \
    tqdm

# Install each custom node's requirements (|| true = continue if missing)
RUN cd /app/ComfyUI/custom_nodes/ComfyUI-Manager && \
    pip install --no-cache-dir -r requirements.txt || true

RUN cd /app/ComfyUI/custom_nodes/ComfyUI-KJNodes && \
    pip install --no-cache-dir -r requirements.txt || true

RUN cd /app/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper && \
    pip install --no-cache-dir -r requirements.txt || true

RUN cd /app/ComfyUI/custom_nodes/ComfyUI-segment-anything-2 && \
    pip install --no-cache-dir -r requirements.txt || true

RUN cd /app/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && \
    pip install --no-cache-dir -r requirements.txt || true

RUN cd /app/ComfyUI/custom_nodes/ComfyUI-WanAnimatePreprocess && \
    pip install --no-cache-dir -r requirements.txt || true

RUN cd /app/ComfyUI/custom_nodes/ComfyUI-SAM3DObjects && \
    pip install --no-cache-dir -r requirements.txt || true

RUN cd /app/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation && \
    pip install --no-cache-dir -r requirements.txt || true

RUN cd /app/ComfyUI/custom_nodes/ComfyUI_ProPainter_Nodes && \
    pip install --no-cache-dir -r requirements.txt || true

RUN cd /app/ComfyUI/custom_nodes/ComfyUI-MotionCapture && \
    pip install --no-cache-dir -r requirements.txt || true

RUN cd /app/ComfyUI/custom_nodes/ComfyUI-CameraPack && \
    pip install --no-cache-dir -r requirements.txt || true

RUN cd /app/ComfyUI/custom_nodes/ComfyUI-HyMotion && \
    pip install --no-cache-dir -r requirements.txt || true

RUN cd /app/ComfyUI/custom_nodes/ComfyUI-Env-Manager && \
    pip install --no-cache-dir -r requirements.txt || true

# pytorch3d - required by SAM3DObjects + MotionCapture
# Try prebuilt wheel first (fast), fall back to source build (slow but reliable)
# MAX_JOBS=2 prevents OOM during CUDA kernel compilation
ENV MAX_JOBS=2
RUN pip install --no-cache-dir pytorch3d \
    -f https://miropsota.github.io/torch_packages_builder/pytorch3d.html \
    || pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git"
ENV MAX_JOBS=

# Final numpy pin (in case any custom node pulled numpy 2.x)
RUN pip install --no-cache-dir "numpy<2"

# ============================================================
# DIRECTORY STRUCTURE
# ============================================================

# Create model directories (will be mounted from host)
RUN mkdir -p \
    /models/checkpoints \
    /models/vae \
    /models/controlnet \
    /models/clip \
    /models/upscale_models \
    /models/loras \
    /models/ipadapter \
    /models/clip_vision \
    /models/animatediff_models \
    /models/sam2 \
    /models/dwpose \
    /models/wan \
    /models/diffusion_models \
    /models/sam3d \
    /models/frame_interpolation \
    /models/propainter \
    /models/vace \
    /models/mocap

# Symlink models directory
RUN rm -rf /app/ComfyUI/models && \
    ln -s /models /app/ComfyUI/models

# Create working directories
RUN mkdir -p \
    /app/ComfyUI/output \
    /app/ComfyUI/input \
    /app/ComfyUI/user/default/workflows \
    /app/workflows

# ============================================================
# SCRIPTS
# ============================================================
COPY scripts/start.sh /app/start.sh
COPY scripts/api_wrapper.py /app/api_wrapper.py
COPY scripts/check_models.py /app/check_models.py

# Fix Windows line endings and make executable
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

# ============================================================
# ENVIRONMENT
# ============================================================
ENV COMFYUI_MODELS_PATH=/models
ENV COMFYUI_LISTEN=0.0.0.0
ENV COMFYUI_PORT=8188
ENV WORKFLOWS_DIR=/app/workflows
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
# RUNTIME
# ============================================================
EXPOSE 8188 8189

# Health check via wrapper API (has ComfyUI connectivity check)
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8189/health || exit 1

# Start both ComfyUI and API wrapper
CMD ["/app/start.sh"]