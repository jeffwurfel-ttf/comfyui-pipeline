#!/bin/bash
# ComfyUI Container Entrypoint
#
# Handles:
#   - Model symlink creation (bridges host layout to ComfyUI expectations)
#   - Model symlink verification
#   - Optional model download on startup
#   - Custom node installation
#   - Starting ComfyUI

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
# MODEL SYMLINKS
# =============================================================================
# ComfyUI expects models in specific directories (checkpoints/, diffusion_models/,
# clip/, vae/, etc). Our host may store models in different locations.
# Create symlinks so ComfyUI can find everything.

echo ""
echo "Setting up model symlinks..."
echo "----------------------------"

# --- checkpoints/ (for CheckpointLoaderSimple) ---
# SDXL checkpoints live in /models/sdxl/ on host
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
# Flux model lives in /models/flux-dev/ on host
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
# Flux VAE (ae.safetensors) lives in /models/flux-dev/
mkdir -p /models/vae
if [ -f "/models/flux-dev/ae.safetensors" ] && [ ! -e "/models/vae/ae.safetensors" ]; then
    ln -sf /models/flux-dev/ae.safetensors /models/vae/ae.safetensors
    echo "  vae/ae.safetensors -> flux-dev/ae.safetensors"
fi

# --- clip/ (for CLIPLoader, DualCLIPLoader) ---
# Flux CLIP models - symlink if downloaded to flux-dev/ or clip/
mkdir -p /models/clip
if [ -d "/models/flux-dev/text_encoder" ]; then
    # Diffusers format T5 - ComfyUI can't use this directly
    # Users need to download the safetensors versions separately
    echo "  Note: flux-dev/text_encoder/ found (diffusers format)"
    echo "  For Flux profiles, download the ComfyUI-compatible versions:"
    echo "    clip/t5xxl_fp8_e4m3fn.safetensors"
    echo "    clip/clip_l.safetensors"
fi

echo "  Symlink setup complete."

# =============================================================================
# MODEL STATUS
# =============================================================================
echo ""
echo "Models Directory Contents:"
echo "--------------------------"
for dir in /models/*/; do
    if [ -d "$dir" ]; then
        # Count real files + symlinks
        count=$(find "$dir" -maxdepth 1 \( -type f -o -type l \) | wc -l)
        echo "  $(basename $dir): $count files"
    fi
done

# Show what ComfyUI will see in key directories
echo ""
echo "ComfyUI Model Paths:"
echo "--------------------"
echo "  checkpoints/:"
ls -1 /models/checkpoints/*.safetensors 2>/dev/null | while read f; do
    echo "    $(basename $f)"
done || echo "    (empty)"

echo "  diffusion_models/:"
ls -1 /models/diffusion_models/*.safetensors 2>/dev/null | while read f; do
    echo "    $(basename $f)"
done || echo "    (empty)"

echo "  vae/:"
ls -1 /models/vae/*.safetensors 2>/dev/null | while read f; do
    echo "    $(basename $f)"
done || echo "    (empty)"

echo "  clip/:"
ls -1 /models/clip/*.safetensors 2>/dev/null | while read f; do
    echo "    $(basename $f)"
done || echo "    (empty)"

echo "  upscale_models/:"
ls -1 /models/upscale_models/*.pth 2>/dev/null | while read f; do
    echo "    $(basename $f)"
done || echo "    (empty)"

# Check if model manager script exists
if [ -f "/app/models/manager.py" ]; then
    echo ""
    echo "Model Manager Available"
    echo "-----------------------"
    echo "  Status:   python /app/models/manager.py status"
    echo "  Download: python /app/models/manager.py download"
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
    
    # Install requirements for custom nodes
    for req_file in */requirements.txt; do
        if [ -f "$req_file" ]; then
            echo "  Installing requirements for: $(dirname $req_file)"
            pip install -q -r "$req_file" || true
        fi
    done
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

# Execute the main command
exec "$@"