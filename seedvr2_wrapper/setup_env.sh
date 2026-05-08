#!/usr/bin/env bash
# setup_env.sh — Create the SeedVR2 isolated venv inside the custom node directory.
#
# Run this INSIDE the comfyui-pipeline container:
#   docker exec -it comfyui-pipeline bash
#   cd /app/ComfyUI/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler
#   bash setup_env.sh
#
# Or from the host:
#   docker exec comfyui-pipeline bash /app/ComfyUI/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler/setup_env.sh
#
# Creates: ./_env/ (venv with pinned deps that don't conflict with main container)
# Idempotent: safe to re-run; will skip if _env/ already exists and is healthy.
#
# To force a clean rebuild:
#   rm -rf /app/ComfyUI/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler/_env

set -euo pipefail

NODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${NODE_DIR}/_env"
PYTHON="${PYTHON:-python3}"

echo "[setup_env] SeedVR2 isolated environment setup"
echo "[setup_env] Node dir: ${NODE_DIR}"
echo "[setup_env] Env dir:  ${ENV_DIR}"

# --- Sanity checks ---
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "[setup_env] ERROR: ${PYTHON} not found in PATH"
    exit 1
fi

PY_VERSION=$("${PYTHON}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[setup_env] Python: ${PYTHON} (${PY_VERSION})"

if [[ "${PY_VERSION}" != "3.10" ]] && [[ "${PY_VERSION}" != "3.11" ]] && [[ "${PY_VERSION}" != "3.12" ]]; then
    echo "[setup_env] WARNING: SeedVR2 expects Python 3.10+; got ${PY_VERSION}"
fi

# --- Idempotency check ---
if [[ -f "${ENV_DIR}/bin/python" ]]; then
    if "${ENV_DIR}/bin/python" -c "import diffusers, torch; print('OK')" >/dev/null 2>&1; then
        DIFFUSERS_VER=$("${ENV_DIR}/bin/python" -c "import diffusers; print(diffusers.__version__)")
        TORCH_VER=$("${ENV_DIR}/bin/python" -c "import torch; print(torch.__version__)")
        echo "[setup_env] _env/ already exists and is healthy"
        echo "[setup_env]   diffusers=${DIFFUSERS_VER}  torch=${TORCH_VER}"
        echo "[setup_env] To force rebuild: rm -rf ${ENV_DIR}"
        exit 0
    else
        echo "[setup_env] _env/ exists but is broken; removing and rebuilding"
        rm -rf "${ENV_DIR}"
    fi
fi

# --- Create venv ---
echo "[setup_env] Creating venv with --system-site-packages (inherits torch from main container)"
"${PYTHON}" -m venv --system-site-packages "${ENV_DIR}"

# Activate
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"

# Upgrade pip
echo "[setup_env] Upgrading pip"
pip install --quiet --upgrade pip

# --- Install pinned deps in venv ---
# Why these pins:
#   diffusers==0.34.0 — last pip release before attention_dispatch.py module
#                       (which crashes with torch 2.4.1's custom_op schema parser)
#   peft>=0.17.0      — SeedVR2 requires this floor
#   transformers<5.0  — diffusers 0.34.0 imports HybridCache which was removed
#                       in transformers 5.0; pin to 4.45+ for safety
# We rely on torch/torchvision/numpy from system site-packages (--system-site-packages)
# to avoid duplicating the 6GB+ torch+CUDA install.
#
# If diffusers 0.34.0 ends up requiring a newer torch we don't have, we'll
# install torch into the venv too (not ideal — disk cost — but works).

echo "[setup_env] Installing pinned deps"
pip install --quiet \
    "diffusers==0.34.0" \
    "transformers>=4.45.0,<5.0.0" \
    "peft>=0.17.0,<0.18" \
    "rotary_embedding_torch>=0.5.3" \
    "omegaconf>=2.3.0" \
    "safetensors" \
    "einops" \
    "tqdm" \
    "psutil" \
    "opencv-python" \
    "gguf" \
    "matplotlib"

# --- Verification ---
echo ""
echo "[setup_env] Verifying installation:"
"${ENV_DIR}/bin/python" - <<'PYEOF'
import sys
print(f"  Python:     {sys.version.split()[0]}")
import torch
print(f"  Torch:      {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
import diffusers
print(f"  Diffusers:  {diffusers.__version__}")
# The critical test: importing the VAE classes SeedVR2 needs without crashing
try:
    from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
    print(f"  VAE import: OK")
except Exception as e:
    print(f"  VAE import: FAILED — {e}")
    sys.exit(1)
import peft
print(f"  PEFT:       {peft.__version__}")
print("")
print("  All deps loaded cleanly. SeedVR2 should now run in this venv.")
PYEOF

echo ""
echo "[setup_env] DONE. Venv ready at: ${ENV_DIR}"
echo "[setup_env] Next: restart ComfyUI so the wrapper __init__.py loads"