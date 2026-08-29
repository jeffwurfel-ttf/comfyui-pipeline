#!/usr/bin/env bash
# setup_env.sh — Create the DINOv3 isolated venv inside the custom node directory.
#
# Structural precedent: ComfyUI-SeedVR2_VideoUpscaler/_env (NOT shared — its own
# pins). The main container env has transformers 5.8.0, whose modeling_utils ->
# integrations.moe registers a torch.library.custom_op that torch 2.4.1's
# infer_schema cannot parse (D032). This venv shadows transformers with 4.57.6
# (>= DINOv3's documented 4.56, predates the 5.x moe break) while INHERITING
# torch 2.4.1 from system site-packages — the exact pair proven to load DINOv3
# in SeedVR2's _env. transformers is not a torch dependency, so no torch/CUDA
# wheel is pulled; --system-site-packages avoids duplicating the ~6GB stack.
#
# Run from the host:
#   docker exec comfyui-pipeline bash /app/ComfyUI/custom_nodes/ComfyUI-DINOv3Embed/setup_env.sh
# Force a clean rebuild:
#   rm -rf /app/ComfyUI/custom_nodes/ComfyUI-DINOv3Embed/_env

set -euo pipefail

NODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${NODE_DIR}/_env"
PYTHON="${PYTHON:-python3}"
TRANSFORMERS_PIN="transformers==4.57.6"

echo "[setup_env] DINOv3 isolated environment setup"
echo "[setup_env] Node dir: ${NODE_DIR}"
echo "[setup_env] Env dir:  ${ENV_DIR}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "[setup_env] ERROR: ${PYTHON} not found in PATH"
    exit 1
fi

PY_VERSION=$("${PYTHON}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[setup_env] Python: ${PYTHON} (${PY_VERSION})"

# --- Idempotency: skip if the venv already shadows transformers 4.57.x on top of torch ---
if [[ -f "${ENV_DIR}/bin/python" ]]; then
    if "${ENV_DIR}/bin/python" -c "import transformers,torch; assert transformers.__version__.startswith('4.57')" >/dev/null 2>&1; then
        TVER=$("${ENV_DIR}/bin/python" -c "import transformers; print(transformers.__version__)")
        TORCHVER=$("${ENV_DIR}/bin/python" -c "import torch; print(torch.__version__)")
        echo "[setup_env] _env/ already healthy: transformers=${TVER} torch=${TORCHVER}"
        echo "[setup_env] To force rebuild: rm -rf ${ENV_DIR}"
        exit 0
    else
        echo "[setup_env] _env/ exists but is not healthy; removing and rebuilding"
        rm -rf "${ENV_DIR}"
    fi
fi

# --- Create venv inheriting the system torch/torchvision/numpy/PIL ---
echo "[setup_env] Creating venv with --system-site-packages (inherits torch ${PYTHON} main env)"
"${PYTHON}" -m venv --system-site-packages "${ENV_DIR}"
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"

echo "[setup_env] Upgrading pip"
pip install --quiet --upgrade pip

# Only transformers is pinned into the venv. It is NOT a torch dependency, so
# no torch/CUDA wheel is fetched; it pulls its own huggingface_hub<1.0 +
# tokenizers + safetensors (which correctly shadow the main env's newer ones).
echo "[setup_env] Installing ${TRANSFORMERS_PIN} (torch inherited from system site-packages)"
pip install --quiet "${TRANSFORMERS_PIN}"

# --- Verify (IMPORT-ONLY; no GPU and no /models bind-mount exist at build time) ---
echo "[setup_env] Verifying (import-only):"
"${ENV_DIR}/bin/python" - <<'PYEOF'
import sys
import torch
print(f"  torch:        {torch.__version__}")
import transformers
print(f"  transformers: {transformers.__version__}")
assert transformers.__version__.startswith("4.57"), transformers.__version__
# modeling_utils must import under the inherited torch 2.4.1 (the whole point)
from transformers import AutoModel
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
assert MODEL_MAPPING_NAMES.get("dinov3_vit") == "DINOv3ViTModel", "dinov3_vit not in auto registry"
from transformers.models.dinov3_vit import DINOv3ViTModel  # class resolvable
print("  DINOv3ViTModel resolvable in auto registry: OK")
print("  (no model load / forward — weights live on the /models bind-mount, absent at build)")
PYEOF

echo "[setup_env] DONE. Venv ready at: ${ENV_DIR}"
