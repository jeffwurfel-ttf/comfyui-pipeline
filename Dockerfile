# ComfyUI Pipeline Service
#
# Active workflows: ESRGAN upscale, Frame Interpolation, WAN video,
#                   Character Swap, SAM3D Objects, GVHMR Motion Capture,
#                   SeedVR2 Restoration Upscale (Stage 1)
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

# ComfyUI core. Pinned to the commit the running gpu02 image was built from,
# recovered from the container (`git -C /app/ComfyUI rev-parse HEAD`), NOT from
# upstream HEAD. Tag context: v0.21.0-3-g428c3237.
ARG COMFYUI_COMMIT=428c323780a7549a4da03b8d282d0064c8e24180
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI && \
    git -C /app/ComfyUI checkout ${COMFYUI_COMMIT}

WORKDIR /app/ComfyUI
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "numpy<2"

# ============================================================
# CUSTOM NODES — Core / Existing
# ============================================================
WORKDIR /app/ComfyUI/custom_nodes

# Every SHA below was recovered from the RUNNING gpu02 container
# (`git -C /app/ComfyUI/custom_nodes/<pack> rev-parse HEAD`) — it is the commit
# that was ACTUALLY BUILT, not upstream HEAD. Five of these are behind upstream
# and that is the correct state: we pin what we validated. Moving any pack
# forward is a deliberate commit, never a side effect of a rebuild.
#
# All six of these packs are also VENDORED into ./custom_nodes/<pack>/ at these
# exact SHAs, byte-verified against the container (see .dev/VENDORED_NODES.md).
# The clone is what the image runs; the vendored tree is the reviewable copy
# and the only correct place to author a patch. They must move together.

# Core utilities
ARG COMFYUI_MANAGER_COMMIT=c2a33d2efcf4597aa29d9dcc87f111751e6ad587
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git && \
    git -C ComfyUI-Manager checkout ${COMFYUI_MANAGER_COMMIT}

ARG KJNODES_COMMIT=fca78c93f034c6e36080d64da83afe00bd5dbba6
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git && \
    git -C ComfyUI-KJNodes checkout ${KJNODES_COMMIT}

# Video generation (WAN)
ARG WANVIDEOWRAPPER_COMMIT=d18cdb18597f525ef8d613a0cb447080fbab8fce
RUN git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git && \
    git -C ComfyUI-WanVideoWrapper checkout ${WANVIDEOWRAPPER_COMMIT}

# Segmentation (SAM2 — used by character swap)
ARG SEGMENT_ANYTHING_2_COMMIT=0c35fff5f382803e2310103357b5e985f5437f32
RUN git clone https://github.com/kijai/ComfyUI-segment-anything-2.git && \
    git -C ComfyUI-segment-anything-2 checkout ${SEGMENT_ANYTHING_2_COMMIT}

# Video I/O
ARG VIDEOHELPERSUITE_COMMIT=2984ec4c4b93292421888f38db74a5e8802a8ff8
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    git -C ComfyUI-VideoHelperSuite checkout ${VIDEOHELPERSUITE_COMMIT}

# Pose detection
ARG WANANIMATEPREPROCESS_COMMIT=1a35b81a418bbba093356ad19b19bf2a76a24f4e
RUN git clone https://github.com/kijai/ComfyUI-WanAnimatePreprocess.git && \
    git -C ComfyUI-WanAnimatePreprocess checkout ${WANANIMATEPREPROCESS_COMMIT}

# Frame Interpolation — RIFE/FILM for FPS upscaling
# (cloned here, NOT vendored — nothing in-repo patches it)
ARG FRAME_INTERPOLATION_COMMIT=26545cc2dd95bc3d27f056016300673bdeee78f5
RUN git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git && \
    git -C ComfyUI-Frame-Interpolation checkout ${FRAME_INTERPOLATION_COMMIT}

# ============================================================
# CUSTOM NODE — SeedVR2 Video Upscaler (Stage 1 of restoration pipeline)
# ============================================================
# Pinned to commit 4490bd1 (Dec 24 2025) — known-good with our wrapper.
# We REPLACE upstream's __init__.py with a thin wrapper that registers a
# single ComfyUI node (SeedVR2RestorationUpscale) which delegates heavy
# work to inference_cli.py running in an isolated _env/ venv. This keeps
# SeedVR2's diffusers import chain isolated from the main ComfyUI process,
# preventing it from breaking Wan / SAM3D / other workflows on import.
#
# Source: ./seedvr2_wrapper/ in the build context.
ARG SEEDVR2_COMMIT=4490bd1f482e026674543386bb2a4d176da245b9
RUN git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git && \
    cd ComfyUI-SeedVR2_VideoUpscaler && \
    git checkout ${SEEDVR2_COMMIT} && \
    mv __init__.py __init__.py.original

COPY seedvr2_wrapper/__init__.py \
     /app/ComfyUI/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler/__init__.py
COPY seedvr2_wrapper/seedvr2_subprocess_node.py \
     /app/ComfyUI/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler/seedvr2_subprocess_node.py
COPY seedvr2_wrapper/setup_env.sh \
     /app/ComfyUI/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler/setup_env.sh

RUN chmod +x /app/ComfyUI/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler/setup_env.sh

# SAM 3D Objects — image to 3D mesh
#
# NOT a git clone: a version-pinned registry zip. There is no upstream commit
# SHA for this pack and none is invented here. The version in the URL (0.0.11)
# IS the pin, and it is corroborated by the box — `pyproject.toml` inside the
# unpacked tree reads version = "0.0.11".
# The unpacked source is vendored at ./custom_nodes/comfyui-sam3dobjects/ with a
# per-file sha256 manifest, which is this pack's reproducibility anchor in place
# of a SHA. See .dev/VENDORED_NODES.md.
#
# Directory name: comfyui-sam3dobjects (lowercase). This is what the container
# runs and what the zip unpacks to. The old git index entry spelled it
# ComfyUI-SAM3DObjects; that name was wrong and is gone.
ARG SAM3DOBJECTS_VERSION=0.0.11
RUN wget -q "https://cdn.comfy.org/pznodes/comfyui-sam3dobjects/${SAM3DOBJECTS_VERSION}/node.zip" -O /tmp/sam3d.zip && \
    unzip -o /tmp/sam3d.zip -d /app/ComfyUI/custom_nodes/comfyui-sam3dobjects && \
    rm /tmp/sam3d.zip

# ============================================================
# CUSTOM NODES — Motion Capture / GVHMR (PozzettiAndrea)
# ============================================================
# All five packages are from the same author and designed to coexist.
# Install in one layer so their shared dependencies resolve together.

# Core motion capture: GVHMRInference, LoadGVHMRModels, LoadSMPL,
# SMPLtoBVH, BVHViewer, SMPLViewer, SMPLCameraViewer, LoadCameraTrajectory
# SHAs recovered from the running container, same rule as the block above:
# the commit that was built, not upstream HEAD. Not vendored — no in-repo patch
# targets these.
ARG MOTIONCAPTURE_COMMIT=e93d9cbaa98c6fe580c87dd82e50a39722df0d8e
RUN git clone https://github.com/PozzettiAndrea/ComfyUI-MotionCapture.git comfyui-motioncapture && \
    git -C comfyui-motioncapture checkout ${MOTIONCAPTURE_COMMIT}

# SMPL parameter retargeting → FBX (HYMotionNPZToSMPLParams, HYMotionSMPLToData,
# HYMotionRetargetFBX)
ARG HYMOTION_COMMIT=4de4c2844b0f71f124cc19c5612b74b671820609
RUN git clone https://github.com/PozzettiAndrea/ComfyUI-HyMotion.git ComfyUI-HyMotion && \
    git -C ComfyUI-HyMotion checkout ${HYMOTION_COMMIT}

# Camera intrinsics for moving-camera GVHMR variant (CameraIntrinsics node)
ARG CAMERAPACK_COMMIT=60729ceb4e37db8135a4fa720c75cf664a1aeae6
RUN git clone https://github.com/PozzettiAndrea/ComfyUI-CameraPack.git comfyui-camerapack && \
    git -C comfyui-camerapack checkout ${CAMERAPACK_COMMIT}

# Multiband I/O for scene_generation pipeline (MultibandLoad, MultibandToMasks)
ARG MULTIBAND_COMMIT=121606fa1f36c11467f205f348aabc1395bd3de4
RUN git clone https://github.com/PozzettiAndrea/ComfyUI-Multiband.git comfyui-multiband && \
    git -C comfyui-multiband checkout ${MULTIBAND_COMMIT}

# Geometry pack for scene_generation (GeomPackLoadMeshBatch,
# GeomPackCombineMeshesBatch, GeomPackPreviewMeshVTK)
ARG GEOMETRYPACK_COMMIT=7aaaeb7e95d5e10d853e5c55ff1095495c0fc5df
RUN git clone https://github.com/PozzettiAndrea/ComfyUI-GeometryPack.git comfyui-geometrypack && \
    git -C comfyui-geometrypack checkout ${GEOMETRYPACK_COMMIT}

# ============================================================
# CUSTOM NODES — Video Inpainting
# ============================================================
# ProPainter — video inpainting with mask (ProPainterInpaint node)
# Strip opencv from requirements.txt — conflicts with our headless install (line ~209).
# --no-deps prevents transitive pulls too.
ARG PROPAINTER_COMMIT=9c27d5a0a508bae3296a1886ad026d8d4139d66c
RUN git clone https://github.com/daniabib/ComfyUI_ProPainter_Nodes \
    /app/ComfyUI/custom_nodes/ComfyUI_ProPainter_Nodes && \
    git -C /app/ComfyUI/custom_nodes/ComfyUI_ProPainter_Nodes checkout ${PROPAINTER_COMMIT} && \
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

# SaveEXRCompressed — TTF output node for 32-bit float EXR with configurable
# compression (DWAA/ZIP/PIZ/etc). Required by restoration-upscale-2x workflow.
# Depends on OpenEXR Python bindings (installed in common deps below).
RUN mkdir -p /app/ComfyUI/custom_nodes/ComfyUI-SaveEXRCompressed
COPY custom_nodes/ComfyUI-SaveEXRCompressed/__init__.py \
     /app/ComfyUI/custom_nodes/ComfyUI-SaveEXRCompressed/__init__.py

# NOTE: ComfyUI-DINOv3Embed + ComfyUI-VectorOut (Phase 6b/7) are added LATE in
# this file (just before their venv setup) to preserve build cache for the
# expensive main-deps + SAM3D/SeedVR2 venv layers below. See §DINOv3Embed.

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
    loguru \
    OpenEXR==3.4.11

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
#
# THIS IS THE ONLY LOCAL PATCH TO AN UPSTREAM PACK, AND IT MUST STAY HERE.
# It is NOT redundant with vendoring. The build fetches sam3dobjects from the
# comfy.org registry zip (line ~114) — it does NOT copy the vendored tree in —
# so the zip always unpacks a live `vendor/cv2/` and this mv is what removes it.
# Delete this line and the rebuilt image breaks cv2 for every node that imports
# it after sam3dobjects.
#
# The vendored tree records the patch's RESULT, not its mechanism: it was staged
# from the running container and therefore already contains
#   custom_nodes/comfyui-sam3dobjects/vendor/cv2_disabled/
# and no vendor/cv2/. That is the post-mv state, so the repo shows what the box
# actually runs. `|| true` keeps the rebuild green if upstream ever drops the
# shim; if that happens, drop this line and the vendored dir together.
RUN mv /app/ComfyUI/custom_nodes/comfyui-sam3dobjects/vendor/cv2 \
       /app/ComfyUI/custom_nodes/comfyui-sam3dobjects/vendor/cv2_disabled || true

# ============================================================
# SeedVR2 — Isolated venv for diffusers + transformers
# ============================================================
# Mirror's SAM3D's pattern: isolated venv prevents SeedVR2's diffusers
# import chain from breaking the main ComfyUI process. The venv inherits
# torch/numpy from the system via --system-site-packages, then installs
# its own diffusers==0.34.0 + transformers<5.0 + peft 0.17.x.
#
# Build-time verification asserts the import chain works — we'd rather
# the build fail here than discover the issue at first runtime.

RUN bash /app/ComfyUI/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler/setup_env.sh

# Verify the venv's import chain. This catches dependency drift if SeedVR2
# upstream changes their pyproject.toml without us re-pinning. Specifically
# guards against the HybridCache and attention_dispatch crashes we hit during
# initial integration.
RUN /app/ComfyUI/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler/_env/bin/python -c "\
import sys; \
import torch; \
import diffusers; assert diffusers.__version__ == '0.34.0', f'venv: wrong diffusers {diffusers.__version__}'; \
import transformers; assert transformers.__version__.startswith('4.'), f'venv: wrong transformers {transformers.__version__}'; \
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution; \
print(f'[SeedVR2-build] venv OK: torch={torch.__version__} diffusers={diffusers.__version__} transformers={transformers.__version__}')"

# Verify the wrapper module itself imports cleanly in the MAIN python (which
# is what ComfyUI uses to load custom nodes). Catches syntax errors and
# missing imports without waiting for ComfyUI to attempt the load at runtime.
RUN python -c "\
import sys; \
sys.path.insert(0, '/app/ComfyUI/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler'); \
import seedvr2_subprocess_node as m; \
assert 'SeedVR2RestorationUpscale' in m.NODE_CLASS_MAPPINGS, 'wrapper: node not registered'; \
print('[SeedVR2-build] wrapper module OK')"

# ============================================================
# DINOv3Embed (recognition embedder) + VectorOut (STRING-JSON sink) — Phase 6b/7.
# Placed here (late) so the COPYs don't invalidate cache for the main-deps and
# SAM3D/SeedVR2 venv layers above. DINOv3Embed's node runs in the MAIN env
# (subprocess wrapper only); its heavy transformers/model work runs in the
# isolated _env/ venv built just below, because the main env's transformers
# 5.8.0 cannot load under torch 2.4.1 (D032). Weights come from the /models
# bind-mount at runtime, not baked.
# ============================================================
RUN mkdir -p /app/ComfyUI/custom_nodes/ComfyUI-DINOv3Embed
COPY custom_nodes/ComfyUI-DINOv3Embed/__init__.py \
     /app/ComfyUI/custom_nodes/ComfyUI-DINOv3Embed/__init__.py
COPY custom_nodes/ComfyUI-DINOv3Embed/dinov3_worker.py \
     /app/ComfyUI/custom_nodes/ComfyUI-DINOv3Embed/dinov3_worker.py
COPY custom_nodes/ComfyUI-DINOv3Embed/setup_env.sh \
     /app/ComfyUI/custom_nodes/ComfyUI-DINOv3Embed/setup_env.sh
RUN chmod +x /app/ComfyUI/custom_nodes/ComfyUI-DINOv3Embed/setup_env.sh

RUN mkdir -p /app/ComfyUI/custom_nodes/ComfyUI-VectorOut
COPY custom_nodes/ComfyUI-VectorOut/__init__.py \
     /app/ComfyUI/custom_nodes/ComfyUI-VectorOut/__init__.py

# setup_env.sh uses --system-site-packages so no torch/CUDA wheel is pulled.
# Build-time verify is IMPORT-ONLY: no GPU and no /models bind-mount exist during
# build, so a model load/forward would fail for reasons unrelated to correctness.
RUN bash /app/ComfyUI/custom_nodes/ComfyUI-DINOv3Embed/setup_env.sh

RUN /app/ComfyUI/custom_nodes/ComfyUI-DINOv3Embed/_env/bin/python -c "\
import transformers; assert transformers.__version__.startswith('4.57'), f'venv: wrong transformers {transformers.__version__}'; \
from transformers import AutoModel; \
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES; \
assert MODEL_MAPPING_NAMES.get('dinov3_vit') == 'DINOv3ViTModel', 'dinov3_vit missing from auto registry'; \
from transformers.models.dinov3_vit import DINOv3ViTModel; \
print(f'[DINOv3-build] venv OK: transformers={transformers.__version__} (import-only, no weights/GPU at build)')"

# Verify the DINOv3 wrapper + worker import cleanly. Wrapper in MAIN python (how
# ComfyUI loads it); worker in the VENV python (where it actually runs). Neither
# import loads a model.
RUN python -c "\
import sys; \
sys.path.insert(0, '/app/ComfyUI/custom_nodes/ComfyUI-DINOv3Embed'); \
import __init__ as m; \
assert 'DINOv3Embed' in m.NODE_CLASS_MAPPINGS, 'DINOv3Embed not registered'; \
print('[DINOv3-build] wrapper module OK (main env)')"
RUN /app/ComfyUI/custom_nodes/ComfyUI-DINOv3Embed/_env/bin/python -c "\
import sys; \
sys.path.insert(0, '/app/ComfyUI/custom_nodes/ComfyUI-DINOv3Embed'); \
import dinov3_worker; \
print('[DINOv3-build] worker module OK (venv)')"

# Verify the VectorOut node registers in the MAIN python.
RUN python -c "\
import sys; \
sys.path.insert(0, '/app/ComfyUI/custom_nodes/ComfyUI-VectorOut'); \
import __init__ as m; \
assert 'SaveVectorJSON' in m.NODE_CLASS_MAPPINGS, 'SaveVectorJSON not registered'; \
print('[VectorOut-build] module OK')"

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

HEALTHCHECK --interval=30s --timeout=15s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:8189/health | \
        python3 -c "import sys,json; d=json.load(sys.stdin); \
        sys.exit(0 if d.get('comfyui_responsive') is True else 1)" || exit 1
CMD ["/app/start.sh"]
