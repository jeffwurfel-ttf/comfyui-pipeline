# ComfyUI Pipeline Service
#
# Active workflows: ESRGAN upscale, Frame Interpolation, WAN video,
#                   Character Swap, SAM3D Objects, GVHMR Motion Capture
#
# Build:
#   docker build -t comfyui-pipeline:latest .
#   docker compose up -d --build

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
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && pip install --no-cache-dir --upgrade pip

# UV for fast venv creation (used by SAM3D installer)
RUN pip install --no-cache-dir uv

WORKDIR /app

# ============================================================
# PYTORCH + COMFYUI
# ============================================================
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir "numpy<2"

RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI

WORKDIR /app/ComfyUI
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "numpy<2"

# ============================================================
# CUSTOM NODES — Core / Existing
# ============================================================
WORKDIR /app/ComfyUI/custom_nodes

# Core utilities
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git

# Video generation (WAN)
RUN git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git

# Segmentation (SAM2 — used by character swap)
RUN git clone https://github.com/kijai/ComfyUI-segment-anything-2.git

# Video I/O
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git

# Pose detection
RUN git clone https://github.com/kijai/ComfyUI-WanAnimatePreprocess.git

# Frame Interpolation — RIFE/FILM for FPS upscaling
RUN git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git

# SAM 3D Objects — image to 3D mesh
RUN wget -q "https://cdn.comfy.org/pznodes/comfyui-sam3dobjects/0.0.11/node.zip" -O /tmp/sam3d.zip && \
    unzip -o /tmp/sam3d.zip -d /app/ComfyUI/custom_nodes/comfyui-sam3dobjects && \
    rm /tmp/sam3d.zip

# ============================================================
# CUSTOM NODES — Motion Capture / GVHMR (PozzettiAndrea)
# ============================================================
# All five packages are from the same author and designed to coexist.
# Install in one layer so their shared dependencies resolve together.

# Core motion capture: GVHMRInference, LoadGVHMRModels, LoadSMPL,
# SMPLtoBVH, BVHViewer, SMPLViewer, SMPLCameraViewer, LoadCameraTrajectory
RUN git clone https://github.com/PozzettiAndrea/ComfyUI-MotionCapture.git comfyui-motioncapture

# SMPL parameter retargeting → FBX (HYMotionNPZToSMPLParams, HYMotionSMPLToData,
# HYMotionRetargetFBX)
RUN git clone https://github.com/PozzettiAndrea/ComfyUI-HyMotion.git ComfyUI-HyMotion

# Camera intrinsics for moving-camera GVHMR variant (CameraIntrinsics node)
RUN git clone https://github.com/PozzettiAndrea/ComfyUI-CameraPack.git comfyui-camerapack

# Multiband I/O for scene_generation pipeline (MultibandLoad, MultibandToMasks)
RUN git clone https://github.com/PozzettiAndrea/ComfyUI-Multiband.git comfyui-multiband

# Geometry pack for scene_generation (GeomPackLoadMeshBatch,
# GeomPackCombineMeshesBatch, GeomPackPreviewMeshVTK)
RUN git clone https://github.com/PozzettiAndrea/ComfyUI-GeometryPack.git comfyui-geometrypack

# ============================================================
# CUSTOM NODES — Video Inpainting
# ============================================================
# ProPainter — video inpainting with mask (ProPainterInpaint node)
# Strip opencv from requirements.txt — conflicts with our headless install (line ~209).
# --no-deps prevents transitive pulls too.
RUN git clone https://github.com/daniabib/ComfyUI_ProPainter_Nodes \
    /app/ComfyUI/custom_nodes/ComfyUI_ProPainter_Nodes && \
    sed -i '/opencv-python/d' /app/ComfyUI/custom_nodes/ComfyUI_ProPainter_Nodes/requirements.txt && \
    pip install --no-cache-dir --no-deps \
        -r /app/ComfyUI/custom_nodes/ComfyUI_ProPainter_Nodes/requirements.txt

# ============================================================
# CUSTOM NODES — TTF
# ============================================================
RUN mkdir -p /app/ComfyUI/custom_nodes/ComfyUI-MultiPersonDetector
COPY custom_nodes/ComfyUI-MultiPersonDetector/__init__.py \
     /app/ComfyUI/custom_nodes/ComfyUI-MultiPersonDetector/__init__.py

RUN mkdir -p /app/ComfyUI/custom_nodes/ComfyUI-VRAMPurge
COPY custom_nodes/ComfyUI-VRAMPurge/__init__.py \
     /app/ComfyUI/custom_nodes/ComfyUI-VRAMPurge/__init__.py
# ============================================================
# PYTHON DEPENDENCIES — Main environment
# ============================================================
WORKDIR /app/ComfyUI

# API wrapper
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    httpx \
    requests

# Common deps
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
    tqdm \
    piexif \
    loguru

# Existing custom node requirements (main env)
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
RUN cd /app/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation && \
    pip install --no-cache-dir -r requirements.txt || true

# New custom node requirements (PozzettiAndrea packages)
RUN cd /app/ComfyUI/custom_nodes/comfyui-motioncapture && \
    pip install --no-cache-dir -r requirements.txt || true
RUN cd /app/ComfyUI/custom_nodes/ComfyUI-HyMotion && \
    pip install --no-cache-dir -r requirements.txt || true
RUN cd /app/ComfyUI/custom_nodes/comfyui-camerapack && \
    pip install --no-cache-dir -r requirements.txt || true
RUN cd /app/ComfyUI/custom_nodes/comfyui-multiband && \
    pip install --no-cache-dir -r requirements.txt || true
RUN cd /app/ComfyUI/custom_nodes/comfyui-geometrypack && \
    pip install --no-cache-dir -r requirements.txt || true

# comfy-env bootstrap for comfyui-motioncapture
# The node uses comfy-env to manage an isolated pixi/uv environment with
# CUDA-specific wheels (dpvo-cuda, torch-scatter) that can't go in the main env.
# comfy-env install must run at build time so the isolated env is baked in.
# pycolmap and ffmpeg-python are direct deps that comfy-env needs available.
#
# NOTE: comfy-env's uv fails on dpvo-cuda's non-PEP-440 version string.
# We let comfy-env set up the pixi env (which succeeds), then manually
# install the two CUDA wheels with pip (which tolerates the version).
RUN pip install --no-cache-dir "comfy-env>=0.2.14" pycolmap ffmpeg-python
RUN cd /app/ComfyUI/custom_nodes/comfyui-motioncapture && \
    comfy-env install || true

# Manually install the CUDA wheels that comfy-env/uv couldn't handle
RUN ENV_DIR=$(find /app/ComfyUI/custom_nodes/comfyui-motioncapture/nodes -maxdepth 1 -name "_env_*" -type d 2>/dev/null | head -1) && \
    if [ -n "$ENV_DIR" ] && [ -f "$ENV_DIR/.pixi/envs/default/bin/pip" ]; then \
        echo "Installing CUDA wheels into $ENV_DIR..." && \
        "$ENV_DIR/.pixi/envs/default/bin/pip" install --no-cache-dir --no-deps \
            "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/dpvo_cuda-latest/dpvo_cuda-0.0.0%2Bcu124torch2.4-cp311-cp311-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl" \
            "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/torch_scatter-latest/torch_scatter-2.1.2%2Bcu124torch2.4-cp311-cp311-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl" \
        && echo "✓ CUDA wheels installed" || echo "⚠ CUDA wheel install failed — mocap may need manual fix"; \
    else \
        echo "⚠ Pixi env not found — comfy-env install may have failed entirely"; \
    fi

# ============================================================
# OPENCV CONFLICT FIX
# ============================================================
# Several custom node requirements.txt files pull in the full opencv-python
# (with GUI/display dependencies) after we install opencv-python-headless above.
# Having both installed simultaneously causes cv2 import failures inside the
# container (no display server). This single uninstall+reinstall ensures only
# the headless variant is present, regardless of what requirements.txt files
# request.
#
# MUST run AFTER all requirements.txt installs — any earlier placement is
# overwritten by the next pip install -r.
RUN pip uninstall -y opencv-python || true && \
    pip install --no-cache-dir "opencv-python-headless==4.9.0.80"

# ============================================================
# SAM3D Objects — Isolated Python 3.10 venv
# ============================================================
# SAM3D uses its own venv because it requires Python 3.10 + PyTorch3D.
# The install.py script creates _env/ with PyTorch + PyTorch3D + gsplat.
# We then manually install nvdiffrast (not on PyPI) and pyvista (missing
# from install.py but required at runtime).

RUN cd /app/ComfyUI/custom_nodes/comfyui-sam3dobjects && \
    python install.py || true

# nvdiffrast — must compile from source with --no-build-isolation
ENV TORCH_CUDA_ARCH_LIST="8.9"
RUN if [ -f /app/ComfyUI/custom_nodes/comfyui-sam3dobjects/_env/bin/pip ]; then \
    /app/ComfyUI/custom_nodes/comfyui-sam3dobjects/_env/bin/pip install \
        --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git; \
    fi
ENV TORCH_CUDA_ARCH_LIST=

# pyvista — required by postprocessing_utils, not in install.py
RUN if [ -f /app/ComfyUI/custom_nodes/comfyui-sam3dobjects/_env/bin/pip ]; then \
    TORCH_CUDA_ARCH_LIST="8.9" \
    /app/ComfyUI/custom_nodes/comfyui-sam3dobjects/_env/bin/pip install \
        --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git; \
    fi
RUN if [ -f /app/ComfyUI/custom_nodes/comfyui-sam3dobjects/_env/bin/pip ]; then \
    /app/ComfyUI/custom_nodes/comfyui-sam3dobjects/_env/bin/pip install \
        --no-cache-dir pyvista; \
    fi

# Disable sam3d vendor cv2 shim — it shadows the real opencv-python-headless,
# breaking any node that imports cv2 after sam3dobjects (__init__.py loads vendor
# at module level). Renaming the dir removes it from the import lookup path.
RUN mv /app/ComfyUI/custom_nodes/comfyui-sam3dobjects/vendor/cv2 \
       /app/ComfyUI/custom_nodes/comfyui-sam3dobjects/vendor/cv2_disabled || true

# Final numpy pin — after ALL installs to prevent any dep from upgrading it
RUN pip install --no-cache-dir "numpy<2"

# ============================================================
# DIRECTORY STRUCTURE
# ============================================================
RUN mkdir -p \
    /models/checkpoints \
    /models/vae \
    /models/controlnet \
    /models/clip \
    /models/clip_vision \
    /models/upscale_models \
    /models/loras \
    /models/ipadapter \
    /models/sam2 \
    /models/sams \
    /models/sam3d \
    /models/dwpose \
    /models/wan \
    /models/diffusion_models \
    /models/onnx \
    /models/.hf_cache \
    /models/gvhmr \
    /models/gvhmr/checkpoints \
    /models/hymotion \
    /models/hymotion/fbx

# CRITICAL: folder_paths.models_dir -> /models -> host volume
RUN rm -rf /app/ComfyUI/models && \
    ln -s /models /app/ComfyUI/models

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
RUN sed -i 's/\r$//' /app/start.sh && \
    chmod +x /app/start.sh

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

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8189/health || exit 1
CMD ["/app/start.sh"]
