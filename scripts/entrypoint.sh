#!/bin/bash
# ComfyUI Container Entrypoint
#
# Handles:
#   - WanVideoWrapper symlinks (detection/, text_encoders/)
#   - Model symlink creation (bridges host layout to ComfyUI expectations)
#   - Model symlink verification
#   - Optional model download on startup
#   - Custom node installation
#   - OpenCV health check
#   - Starting ComfyUI via CMD

set -e

echo "=========================================="
echo "ComfyUI Container Starting"
echo "=========================================="

# Verify models directory is mounted
if [ ! -d "/models" ]; then
    echo "ERROR: /models directory not mounted!"
    echo "Make sure your models volume is mounted."
    exit 1
fi

# =============================================================================
# WANVIDEOWRAPPER SYMLINKS
# =============================================================================
# WanVideoWrapper nodes hardcode specific paths that differ from the host layout.
# These symlinks are container-internal and must be created on every boot.
# Previously required manual `comfy_doctor.sh --fix` after every rebuild.

echo ""
echo "Setting up WanVideoWrapper symlinks..."
echo "---------------------------------------"

# --- detection/ flat paths (MultiPersonDetection) ---
mkdir -p /models/detection
if [ -f "/models/onnx/wholebody/vitpose-l-wholebody.onnx" ]; then
    ln -sf /models/onnx/wholebody/vitpose-l-wholebody.onnx /models/detection/vitpose-l-wholebody.onnx 2>/dev/null && \
        echo "  ✓ detection/vitpose-l-wholebody.onnx (flat)" || true
fi
if [ -f "/models/onnx/process_checkpoint/det/yolov10m.onnx" ]; then
    ln -sf /models/onnx/process_checkpoint/det/yolov10m.onnx /models/detection/yolov10m.onnx 2>/dev/null && \
        echo "  ✓ detection/yolov10m.onnx (flat)" || true
fi

# --- detection/ subdirectory paths (TTFCharacterSwap) ---
mkdir -p /models/detection/onnx/wholebody
mkdir -p /models/detection/process_checkpoint/det
if [ -f "/models/onnx/wholebody/vitpose-l-wholebody.onnx" ]; then
    ln -sf /models/onnx/wholebody/vitpose-l-wholebody.onnx /models/detection/onnx/wholebody/vitpose-l-wholebody.onnx 2>/dev/null && \
        echo "  ✓ detection/onnx/wholebody/vitpose-l-wholebody.onnx (subdir)" || true
fi
if [ -f "/models/onnx/process_checkpoint/det/yolov10m.onnx" ]; then
    ln -sf /models/onnx/process_checkpoint/det/yolov10m.onnx /models/detection/process_checkpoint/det/yolov10m.onnx 2>/dev/null && \
        echo "  ✓ detection/process_checkpoint/det/yolov10m.onnx (subdir)" || true
fi

# --- text_encoders/ (LoadWanVideoT5TextEncoder) ---
mkdir -p /models/text_encoders
for encoder in umt5-xxl-enc-bf16.safetensors clip_l.safetensors t5xxl_fp8_e4m3fn.safetensors; do
    if [ -f "/models/clip/${encoder}" ] && [ ! -e "/models/text_encoders/${encoder}" ]; then
        ln -sf "/models/clip/${encoder}" "/models/text_encoders/${encoder}" 2>/dev/null && \
            echo "  ✓ text_encoders/${encoder}" || true
    fi
done

echo "  WanVideoWrapper symlinks complete."

# =============================================================================
# MODEL SYMLINKS (host layout → ComfyUI expectations)
# =============================================================================
echo ""
echo "Setting up model symlinks..."
echo "----------------------------"

# --- checkpoints/ (for CheckpointLoaderSimple) ---
mkdir -p /models/checkpoints
if [ -d "/models/sdxl" ]; then
    for f in /models/sdxl/*.safetensors; do
        [ -f "$f" ] || continue
        base=$(basename "$f")
        if [ ! -e "/models/checkpoints/$base" ]; then
            ln -sf "$f" "/models/checkpoints/$base"
            echo "  checkpoints/$base -> sdxl/$base"
        fi
    done
fi

# --- diffusion_models/ (for UNETLoader) ---
mkdir -p /models/diffusion_models
if [ -d "/models/flux-dev" ]; then
    for f in /models/flux-dev/flux*.safetensors; do
        [ -f "$f" ] || continue
        base=$(basename "$f")
        if [ ! -e "/models/diffusion_models/$base" ]; then
            ln -sf "$f" "/models/diffusion_models/$base"
            echo "  diffusion_models/$base -> flux-dev/$base"
        fi
    done
fi

# --- vae/ (for VAELoader) ---
mkdir -p /models/vae
if [ -f "/models/flux-dev/ae.safetensors" ] && [ ! -e "/models/vae/ae.safetensors" ]; then
    ln -sf /models/flux-dev/ae.safetensors /models/vae/ae.safetensors
    echo "  vae/ae.safetensors -> flux-dev/ae.safetensors"
fi

# --- clip/ ---
mkdir -p /models/clip

echo "  Symlink setup complete."

# =============================================================================
# MODEL STATUS
# =============================================================================
echo ""
echo "Models Directory Contents:"
echo "--------------------------"
for dir in /models/*/; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -maxdepth 1 \( -type f -o -type l \) | wc -l)
        if [ "$count" -gt 0 ]; then
            echo "  $(basename $dir): $count files"
        fi
    fi
done

# =============================================================================
# OPENCV HEALTH CHECK
# =============================================================================
echo ""
echo "OpenCV Check:"
echo "-------------"
if python -c "import cv2; print(f'  ✓ cv2 {cv2.__version__}')" 2>/dev/null; then
    if python -c "import cv2; assert hasattr(cv2, 'VideoCapture'); print('  ✓ VideoCapture available')" 2>/dev/null; then
        true
    else
        echo "  ⚠ cv2 imported but VideoCapture missing — reinstalling headless"
        pip install --quiet --force-reinstall opencv-python-headless 2>/dev/null || true
    fi
else
    echo "  ⚠ cv2 import failed — reinstalling headless"
    pip install --quiet --force-reinstall opencv-python-headless 2>/dev/null || true
fi

# Auto-download models if AUTODOWNLOAD_MODELS=true
if [ "${AUTODOWNLOAD_MODELS:-false}" = "true" ]; then
    echo ""
    echo "Auto-downloading missing models..."
    python /app/models/manager.py download || true
fi

# Install/update custom nodes if requested
if [ "${UPDATE_CUSTOM_NODES:-false}" = "true" ]; then
    echo ""
    echo "Updating custom nodes..."
    cd /app/ComfyUI/custom_nodes
    for node_dir in */; do
        if [ -d "$node_dir/.git" ]; then
            echo "  Updating: $node_dir"
            (cd "$node_dir" && git pull) || true
        fi
    done
    for req_file in */requirements.txt; do
        if [ -f "$req_file" ]; then
            echo "  Installing requirements for: $(dirname $req_file)"
            pip install -q -r "$req_file" || true
        fi
    done
    # Re-fix opencv after any requirements.txt installs
    pip uninstall -y opencv-python 2>/dev/null; \
    pip install --quiet --force-reinstall opencv-python-headless 2>/dev/null || true
fi

# Show GPU info
echo ""
echo "GPU Information:"
echo "----------------"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  GPU info not available"

echo ""
echo "=========================================="
echo "Starting ComfyUI on port ${COMFYUI_PORT:-8188}"
echo "=========================================="
echo ""

# Execute the main command (CMD = /app/start.sh)
exec "$@"