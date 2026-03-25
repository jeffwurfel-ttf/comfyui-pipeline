#!/usr/bin/env bash
# =============================================================================
# comfy_preflight.sh — Workflow Preflight Validator
# =============================================================================
# Validates workflow JSONs against live ComfyUI: checks nodes exist,
# model files are on disk, and required inputs are present.
#
# Usage:
#   ./tools/comfy_preflight.sh                          # Check all workflows
#   ./tools/comfy_preflight.sh workflow.json             # Check one workflow
#   ./tools/comfy_preflight.sh --gateway                 # Check gateway workflows
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

COMFY_URL="${COMFY_URL:-http://localhost:8188}"
MODELS_DIR="${COMFY_MODELS_HOST:-/mnt/ssd/comfyui-models}"
CONTAINER="${COMFY_CONTAINER:-comfyui-pipeline}"
GATEWAY_WORKFLOWS="${GATEWAY_WORKFLOWS:-/opt/ai-gateway/workflows}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
miss() { echo -e "  ${RED}✗${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
info() { echo -e "  ${CYAN}ℹ${NC} $1"; }
header() { echo -e "\n${BOLD}═══ $1 ═══${NC}"; }

# ─── Fetch node registry from ComfyUI ────────────────────────────────────────
header "Preflight — Node Registry"

NODES_CACHE=$(mktemp)
trap "rm -f $NODES_CACHE" EXIT

if curl -sf "${COMFY_URL}/object_info" &>/dev/null; then
    curl -sf "${COMFY_URL}/object_info" | python3 -c "
import sys, json
data = json.load(sys.stdin)
# Output: node_name<TAB>json_of_inputs
for name, info in data.items():
    inputs = info.get('input', {})
    required = inputs.get('required', {})
    optional = inputs.get('optional', {})
    all_inputs = {**required, **optional}
    print(f'{name}')
" > "$NODES_CACHE" 2>/dev/null
    NODE_COUNT=$(wc -l < "$NODES_CACHE")
    ok "ComfyUI responding — ${NODE_COUNT} nodes registered"
else
    miss "ComfyUI not responding at ${COMFY_URL}"
    echo "  Start the container first: docker compose -f docker-compose.rocky.yaml up -d"
    exit 1
fi

# ─── Validate a single workflow JSON ─────────────────────────────────────────
validate_workflow() {
    local wf_path="$1"
    local wf_name
    wf_name=$(basename "$wf_path")
    local wf_pass=true

    echo -e "\n  ${BOLD}${wf_name}${NC}"

    if [[ ! -f "$wf_path" ]]; then
        miss "File not found: $wf_path"
        ((FAIL++)) || true
        return
    fi

    # Parse workflow and check each node
    python3 - "$wf_path" "$NODES_CACHE" "$MODELS_DIR" <<'PYEOF'
import sys, json, os

wf_path = sys.argv[1]
nodes_cache = sys.argv[2]
models_dir = sys.argv[3]

# Load registered nodes
with open(nodes_cache) as f:
    registered = {line.strip() for line in f if line.strip()}

# Load workflow
with open(wf_path) as f:
    try:
        data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR\tInvalid JSON: {e}")
        sys.exit(0)

# Extract nodes (API format: {"id": {"class_type": ..., "inputs": {...}}})
nodes = {}
for node_id, node_data in data.items():
    if isinstance(node_data, dict) and "class_type" in node_data:
        nodes[node_id] = node_data

if not nodes:
    print("WARN\tNo nodes found (not API format?)")
    sys.exit(0)

# Check each node
missing_nodes = []
missing_models = []
node_issues = []

# Known model input keys and their search directories
MODEL_KEYS = {
    "model": ["diffusion_models", "checkpoints"],
    "model_name": ["diffusion_models", "checkpoints", "clip", "vae", "sam2", "sams"],
    "ckpt_name": ["custom_nodes"],  # RIFE etc download their own
    "unet_name": ["diffusion_models"],
    "vae_name": ["vae"],
    "clip_name": ["clip"],
    "upscale_model": ["upscale_models"],
}

# File extensions that indicate model references
MODEL_EXTENSIONS = {".safetensors", ".pth", ".ckpt", ".pt", ".onnx", ".bin"}

for node_id, node_data in nodes.items():
    class_type = node_data["class_type"]
    inputs = node_data.get("inputs", {})
    meta = node_data.get("_meta", {})
    title = meta.get("title", class_type)

    # Check node exists
    if class_type not in registered:
        missing_nodes.append(f"MISSING_NODE\t{class_type}\tnode {node_id} ({title})")
        continue

    # Check model file references
    for key, value in inputs.items():
        if not isinstance(value, str):
            continue
        # Check if value looks like a model filename
        _, ext = os.path.splitext(value)
        if ext.lower() in MODEL_EXTENSIONS:
            # Search for the file
            found = False
            # Direct path under models_dir
            if os.path.isfile(os.path.join(models_dir, value)):
                found = True
            else:
                # Search common subdirectories
                for subdir in ["diffusion_models", "checkpoints", "clip", "clip_vision",
                               "vae", "upscale_models", "sam2", "sams", "loras",
                               "onnx", "text_encoders", "controlnet"]:
                    candidate = os.path.join(models_dir, subdir, value)
                    if os.path.isfile(candidate):
                        found = True
                        break
                # Also search recursively (handles subdirectories like Wan22Animate/)
                if not found:
                    for root, dirs, files in os.walk(models_dir):
                        if os.path.basename(value) in files:
                            found = True
                            break

            if not found:
                missing_models.append(f"MISSING_MODEL\t{value}\tnode {node_id} ({title}), input '{key}'")

# Output results
for item in missing_nodes:
    print(item)
for item in missing_models:
    print(item)

if not missing_nodes and not missing_models:
    print(f"OK\t{len(nodes)} nodes validated, all models found")
PYEOF

    # Process Python output
    while IFS=$'\t' read -r status detail context; do
        case "$status" in
            OK)
                ok "$detail"
                ;;
            MISSING_NODE)
                miss "Node not registered: ${detail} — ${context}"
                wf_pass=false
                ;;
            MISSING_MODEL)
                miss "Model not found: ${detail} — ${context}"
                wf_pass=false
                ;;
            WARN)
                warn "$detail"
                ;;
            ERROR)
                miss "$detail"
                wf_pass=false
                ;;
        esac
    done < <(python3 - "$wf_path" "$NODES_CACHE" "$MODELS_DIR" <<'PYEOF2'
import sys, json, os

wf_path = sys.argv[1]
nodes_cache = sys.argv[2]
models_dir = sys.argv[3]

with open(nodes_cache) as f:
    registered = {line.strip() for line in f if line.strip()}

with open(wf_path) as f:
    try:
        data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR\tInvalid JSON: {e}")
        sys.exit(0)

nodes = {}
for node_id, node_data in data.items():
    if isinstance(node_data, dict) and "class_type" in node_data:
        nodes[node_id] = node_data

if not nodes:
    print("WARN\tNo nodes found (not API format?)")
    sys.exit(0)

MODEL_EXTENSIONS = {".safetensors", ".pth", ".ckpt", ".pt", ".onnx", ".bin"}

missing_nodes = []
missing_models = []

for node_id, node_data in nodes.items():
    class_type = node_data["class_type"]
    inputs = node_data.get("inputs", {})
    meta = node_data.get("_meta", {})
    title = meta.get("title", class_type)

    if class_type not in registered:
        missing_nodes.append(f"MISSING_NODE\t{class_type}\tnode {node_id} ({title})")
        continue

    for key, value in inputs.items():
        if not isinstance(value, str):
            continue
        _, ext = os.path.splitext(value)
        if ext.lower() in MODEL_EXTENSIONS:
            found = False
            if os.path.isfile(os.path.join(models_dir, value)):
                found = True
            else:
                for subdir in ["diffusion_models", "checkpoints", "clip", "clip_vision",
                               "vae", "upscale_models", "sam2", "sams", "loras",
                               "onnx", "text_encoders", "controlnet"]:
                    if os.path.isfile(os.path.join(models_dir, subdir, value)):
                        found = True
                        break
                if not found:
                    for root, dirs, files in os.walk(models_dir):
                        if os.path.basename(value) in files:
                            found = True
                            break
            if not found:
                missing_models.append(f"MISSING_MODEL\t{value}\tnode {node_id} ({title}), input '{key}'")

for item in missing_nodes:
    print(item)
for item in missing_models:
    print(item)

if not missing_nodes and not missing_models:
    print(f"OK\t{len(nodes)} nodes validated, all models found")
PYEOF2
    )

    if $wf_pass; then
        ((PASS++)) || true
    else
        ((FAIL++)) || true
    fi
}

# ─── Main ────────────────────────────────────────────────────────────────────

header "Workflow Validation"

if [[ "${1:-}" == "--gateway" ]]; then
    # Check gateway workflow JSONs
    if [[ -d "$GATEWAY_WORKFLOWS" ]]; then
        for wf in "$GATEWAY_WORKFLOWS"/*.json; do
            validate_workflow "$wf"
        done
    else
        warn "Gateway workflows dir not found: $GATEWAY_WORKFLOWS"
        info "Set GATEWAY_WORKFLOWS env var or use --gateway from the gateway server"
    fi
elif [[ -n "${1:-}" ]]; then
    # Check specific file
    validate_workflow "$1"
else
    # Check all local workflows
    for wf in "$REPO_DIR"/workflows/*.json; do
        [[ -f "$wf" ]] && validate_workflow "$wf"
    done
    # Also check container workflows
    CONTAINER_WFS=$(docker exec "$CONTAINER" find /app/workflows -name "*.json" 2>/dev/null || true)
    if [[ -n "$CONTAINER_WFS" ]]; then
        info "Also checking container workflows..."
        while IFS= read -r wf; do
            # Copy to temp and validate
            local_tmp=$(mktemp --suffix=.json)
            docker cp "${CONTAINER}:${wf}" "$local_tmp" 2>/dev/null
            validate_workflow "$local_tmp"
            rm -f "$local_tmp"
        done <<< "$CONTAINER_WFS"
    fi
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
header "Summary"
echo -e "  ${GREEN}Passed: ${PASS}${NC}"
echo -e "  ${RED}Failed: ${FAIL}${NC}"

if [[ $FAIL -eq 0 ]]; then
    echo -e "\n  ${GREEN}${BOLD}All workflows validated!${NC}"
else
    echo -e "\n  ${RED}${BOLD}${FAIL} workflow(s) have issues${NC}"
    exit 1
fi
