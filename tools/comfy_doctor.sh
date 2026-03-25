#!/usr/bin/env bash
# =============================================================================
# comfy_doctor.sh — ComfyUI Workflow Readiness Diagnostic
# =============================================================================
# Run on the ComfyUI host (not inside the container) to diagnose and fix
# common issues before standing up new workflows.
#
# Usage:
#   ./comfy_doctor.sh                    # Full diagnostic
#   ./comfy_doctor.sh --check-workflow FILE.json   # Check specific workflow
#   ./comfy_doctor.sh --fix              # Auto-fix what we can
#   ./comfy_doctor.sh --check-nodes      # Just check custom node health
#
# Requires: docker, jq
# =============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------
CONTAINER="${COMFY_CONTAINER:-comfyui-pipeline}"
MODELS_HOST="${COMFY_MODELS_HOST:-/mnt/ssd/comfyui-models}"
MODELS_CONTAINER="${COMFY_MODELS_CONTAINER:-/models}"
COMFY_URL="${COMFY_URL:-http://localhost:8188}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

ERRORS=0
WARNINGS=0
FIXES=0
FIX_MODE=false
CHECK_WORKFLOW=""
CHECK_NODES_ONLY=false

# --- Argument parsing --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --fix) FIX_MODE=true; shift ;;
        --check-workflow) CHECK_WORKFLOW="$2"; shift 2 ;;
        --check-nodes) CHECK_NODES_ONLY=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--fix] [--check-workflow FILE.json] [--check-nodes]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Helpers -----------------------------------------------------------------
ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; ((WARNINGS++)) || true; }
fail() { echo -e "  ${RED}✗${NC} $1"; ((ERRORS++)) || true; }
info() { echo -e "  ${CYAN}ℹ${NC} $1"; }
header() { echo -e "\n${BOLD}═══ $1 ═══${NC}"; }
fixed() { echo -e "  ${GREEN}🔧${NC} $1"; ((FIXES++)) || true; }

# --- Preflight ---------------------------------------------------------------
header "Preflight Checks"

if ! command -v docker &>/dev/null; then
    fail "docker not found"; exit 1
fi
ok "docker available"

if ! command -v jq &>/dev/null; then
    warn "jq not found — workflow checks will be limited"
    HAS_JQ=false
else
    ok "jq available"
    HAS_JQ=true
fi

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    fail "Container '${CONTAINER}' not running"
    exit 1
fi
ok "Container '${CONTAINER}' is running"

HEALTH=$(docker inspect "$CONTAINER" --format '{{.State.Health.Status}}' 2>/dev/null || echo "none")
if [[ "$HEALTH" == "healthy" ]]; then
    ok "Container health: ${HEALTH}"
elif [[ "$HEALTH" == "none" ]]; then
    info "No health check configured"
else
    warn "Container health: ${HEALTH}"
fi

# Check ComfyUI is responding
if curl -sf "${COMFY_URL}/system_stats" &>/dev/null; then
    ok "ComfyUI API responding at ${COMFY_URL}"
    GPU_NAME=$(curl -sf "${COMFY_URL}/system_stats" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['devices'][0]['name'].split(':')[0].strip())" 2>/dev/null || echo "unknown")
    VRAM_GB=$(curl -sf "${COMFY_URL}/system_stats" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['devices'][0]['vram_total']/1073741824:.1f}\")" 2>/dev/null || echo "?")
    info "GPU: ${GPU_NAME} (${VRAM_GB} GB VRAM)"
else
    fail "ComfyUI API not responding at ${COMFY_URL}"
fi

if $CHECK_NODES_ONLY; then
    # Skip to node checks
    :
else

# --- Model Health Check ------------------------------------------------------
header "Model Health Check"

# Check for broken downloads (suspiciously small files)
echo -e "  Scanning for broken model files..."
BROKEN_FILES=$(find "$MODELS_HOST" \( -name "*.safetensors" -o -name "*.pth" -o -name "*.ckpt" -o -name "*.pt" -o -name "*.bin" \) -size -1k 2>/dev/null)
if [[ -n "$BROKEN_FILES" ]]; then
    while IFS= read -r f; do
        SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
        fail "Broken model (${SIZE} bytes): $f"
        if $FIX_MODE; then
            rm -f "$f"
            fixed "Removed broken file: $f"
        fi
    done <<< "$BROKEN_FILES"
else
    ok "No broken model files detected"
fi

# Check for broken symlinks
BROKEN_LINKS=$(find "$MODELS_HOST" -xtype l 2>/dev/null)
if [[ -n "$BROKEN_LINKS" ]]; then
    while IFS= read -r f; do
        TARGET=$(readlink "$f")
        fail "Broken symlink: $f -> $TARGET"
        if $FIX_MODE; then
            rm -f "$f"
            fixed "Removed broken symlink: $f"
        fi
    done <<< "$BROKEN_LINKS"
else
    ok "No broken symlinks"
fi

# Verify container can read models
CONTAINER_MODEL_COUNT=$(docker exec "$CONTAINER" find "$MODELS_CONTAINER" -name "*.safetensors" -o -name "*.pth" -o -name "*.ckpt" 2>/dev/null | wc -l)
HOST_MODEL_COUNT=$(find "$MODELS_HOST" -name "*.safetensors" -o -name "*.pth" -o -name "*.ckpt" 2>/dev/null | wc -l)
if [[ "$CONTAINER_MODEL_COUNT" -eq "$HOST_MODEL_COUNT" ]]; then
    ok "Container sees all ${HOST_MODEL_COUNT} model files"
else
    warn "Model count mismatch: host=${HOST_MODEL_COUNT}, container=${CONTAINER_MODEL_COUNT}"
    info "Possible cause: absolute symlinks that don't resolve inside container"
    info "Check symlinks with: find $MODELS_HOST -type l -exec ls -la {} \\;"
fi

# Check container can actually READ files (not just see them via ls)
SAMPLE_FILE=$(docker exec "$CONTAINER" find "$MODELS_CONTAINER" -name "*.yaml" -type f 2>/dev/null | head -1)
if [[ -n "$SAMPLE_FILE" ]]; then
    if docker exec "$CONTAINER" cat "$SAMPLE_FILE" &>/dev/null; then
        ok "Container can read model files (verified: $(basename $SAMPLE_FILE))"
    else
        fail "Container can list but NOT read model files (symlink resolution issue)"
        info "Fix: use 'cp' instead of 'ln -s' for cross-mount files"
    fi
fi

# Model inventory
echo -e "\n  ${BOLD}Model Inventory:${NC}"
for dir in "$MODELS_HOST"/*/; do
    dirname=$(basename "$dir")
    count=$(find "$dir" \( -name "*.safetensors" -o -name "*.pth" -o -name "*.ckpt" -o -name "*.pt" \) 2>/dev/null | wc -l)
    size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    if [[ "$count" -gt 0 ]]; then
        echo -e "    ${dirname}: ${count} model(s), ${size}"
    fi
done

# --- Docker Mount Check ------------------------------------------------------
header "Docker Mount Verification"

MOUNTS=$(docker inspect "$CONTAINER" --format '{{json .Mounts}}' 2>/dev/null)
if [[ -n "$MOUNTS" ]] && $HAS_JQ; then
    echo "$MOUNTS" | jq -r '.[] | "    \(.Source) → \(.Destination) [\(.Mode // "rw")\(.RW | if . then "" else " READ-ONLY" end)]"' 2>/dev/null
    ok "Mount configuration listed above"
else
    info "Install jq for detailed mount analysis"
fi

fi # end of non-CHECK_NODES_ONLY block

# --- WanVideoWrapper Symlink Requirements -------------------------------------
# WanVideoWrapper nodes hardcode specific container paths that don't exist by
# default. Two different path styles are required simultaneously:
#   - Flat:    /models/detection/yolov10m.onnx            (MultiPersonDetection)
#   - Subdir:  /models/detection/onnx/wholebody/...        (TTFCharacterSwap)
# text_encoders/ symlinks are required by LoadWanVideoT5TextEncoder.
# All are ephemeral — recreated by this script after every container rebuild.
header "WanVideoWrapper Symlink Requirements"

_check_container_symlink() {
    local link="$1"
    local target="$2"
    local desc="$3"

    if ! docker exec "$CONTAINER" test -f "$target" 2>/dev/null; then
        fail "${desc} — canonical target missing: ${target}"
        return
    fi
    if docker exec "$CONTAINER" test -e "$link" 2>/dev/null; then
        ok "${desc}"
    else
        fail "${desc} — symlink missing"
        if $FIX_MODE; then
            docker exec "$CONTAINER" bash -c "mkdir -p '$(dirname "$link")' && ln -sf '$target' '$link'" 2>/dev/null
            if docker exec "$CONTAINER" test -e "$link" 2>/dev/null; then
                fixed "Created: ${link}"
            else
                fail "  Could not create: ${link}"
            fi
        else
            info "  Run with --fix to create automatically"
        fi
    fi
}

echo -e "\n  ${BOLD}detection/ flat paths (MultiPersonDetection workflow):${NC}"
_check_container_symlink \
    "${MODELS_CONTAINER}/detection/vitpose-l-wholebody.onnx" \
    "${MODELS_CONTAINER}/onnx/wholebody/vitpose-l-wholebody.onnx" \
    "detection/vitpose-l-wholebody.onnx"
_check_container_symlink \
    "${MODELS_CONTAINER}/detection/yolov10m.onnx" \
    "${MODELS_CONTAINER}/onnx/process_checkpoint/det/yolov10m.onnx" \
    "detection/yolov10m.onnx"

echo -e "\n  ${BOLD}detection/ subdirectory paths (TTFCharacterSwap workflow):${NC}"
_check_container_symlink \
    "${MODELS_CONTAINER}/detection/onnx/wholebody/vitpose-l-wholebody.onnx" \
    "${MODELS_CONTAINER}/onnx/wholebody/vitpose-l-wholebody.onnx" \
    "detection/onnx/wholebody/vitpose-l-wholebody.onnx"
_check_container_symlink \
    "${MODELS_CONTAINER}/detection/process_checkpoint/det/yolov10m.onnx" \
    "${MODELS_CONTAINER}/onnx/process_checkpoint/det/yolov10m.onnx" \
    "detection/process_checkpoint/det/yolov10m.onnx"

echo -e "\n  ${BOLD}text_encoders/ (LoadWanVideoT5TextEncoder node):${NC}"
docker exec "$CONTAINER" mkdir -p "${MODELS_CONTAINER}/text_encoders" 2>/dev/null || true
_check_container_symlink \
    "${MODELS_CONTAINER}/text_encoders/umt5-xxl-enc-bf16.safetensors" \
    "${MODELS_CONTAINER}/clip/umt5-xxl-enc-bf16.safetensors" \
    "text_encoders/umt5-xxl-enc-bf16.safetensors"
_check_container_symlink \
    "${MODELS_CONTAINER}/text_encoders/clip_l.safetensors" \
    "${MODELS_CONTAINER}/clip/clip_l.safetensors" \
    "text_encoders/clip_l.safetensors"
_check_container_symlink \
    "${MODELS_CONTAINER}/text_encoders/t5xxl_fp8_e4m3fn.safetensors" \
    "${MODELS_CONTAINER}/clip/t5xxl_fp8_e4m3fn.safetensors" \
    "text_encoders/t5xxl_fp8_e4m3fn.safetensors"

# --- Custom Node Health -------------------------------------------------------
header "Custom Node Health"

NODES_DIR="/app/ComfyUI/custom_nodes"
CUSTOM_NODES=$(docker exec "$CONTAINER" ls "$NODES_DIR" 2>/dev/null | grep -v "__pycache__" | grep -v "\.py")
echo -e "  ${BOLD}Installed custom nodes:${NC}"
for node in $CUSTOM_NODES; do
    NODE_PATH="${NODES_DIR}/${node}"
    
    # Check if it has a venv (isolated environment)
    HAS_VENV=$(docker exec "$CONTAINER" test -d "${NODE_PATH}/_env" && echo "yes" || echo "no")
    
    # Check for install errors in logs
    HAS_LOG=$(docker exec "$CONTAINER" test -f "${NODE_PATH}/install.log" && echo "yes" || echo "no")
    if [[ "$HAS_LOG" == "yes" ]]; then
        INSTALL_ERRORS=$(docker exec "$CONTAINER" grep -c "ERROR\|FAILED" "${NODE_PATH}/install.log" 2>/dev/null || echo "0")
    else
        INSTALL_ERRORS=0
    fi
    
    VENV_TAG=""
    [[ "$HAS_VENV" == "yes" ]] && VENV_TAG=" [isolated venv]"
    
    if [[ "$INSTALL_ERRORS" -gt 0 ]]; then
        warn "${node}${VENV_TAG} — ${INSTALL_ERRORS} install errors in log"
    else
        ok "${node}${VENV_TAG}"
    fi
    
    # For nodes with venvs, check critical imports
    if [[ "$HAS_VENV" == "yes" ]]; then
        VENV_PYTHON="${NODE_PATH}/_env/bin/python"
        
        # Check for common missing deps
        for dep in pyvista nvdiffrast torch; do
            if ! docker exec "$CONTAINER" "$VENV_PYTHON" -c "import $dep" 2>/dev/null; then
                fail "  └─ Missing in venv: ${dep}"
                if $FIX_MODE && [[ "$dep" == "pyvista" ]]; then
                    docker exec "$CONTAINER" "$VENV_PYTHON" -m pip install pyvista -q 2>/dev/null && fixed "Installed pyvista in ${node} venv"
                fi
                if $FIX_MODE && [[ "$dep" == "nvdiffrast" ]]; then
                    info "  └─ nvdiffrast requires manual install: docker exec $CONTAINER $VENV_PYTHON -m pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git"
                fi
            fi
        done
    fi
done

# --- Get registered node types ------------------------------------------------
header "Node Registry"

# Get all registered node class names from ComfyUI
REGISTERED_NODES=$(docker exec "$CONTAINER" python3 -c "
import sys
sys.path.insert(0, '/app/ComfyUI')
import importlib
import os

# Collect NODE_CLASS_MAPPINGS from all custom nodes
all_nodes = {}
nodes_dir = '/app/ComfyUI/custom_nodes'
for d in os.listdir(nodes_dir):
    init_path = os.path.join(nodes_dir, d, '__init__.py')
    if os.path.isfile(init_path):
        try:
            spec = importlib.util.spec_from_file_location(d, init_path)
            mod = importlib.util.module_from_spec(spec)
            # Don't actually execute - just grep for the mapping
            pass
        except:
            pass

# Simpler: just grep for class_type registrations
import subprocess
result = subprocess.run(
    ['grep', '-roh', '\"[A-Za-z0-9_]*\"', '--include=__init__.py'],
    capture_output=True, text=True, cwd=nodes_dir
)
" 2>/dev/null || true)

# Simpler approach: extract from ComfyUI's object_info endpoint
if curl -sf "${COMFY_URL}/object_info" &>/dev/null; then
    NODE_COUNT=$(curl -sf "${COMFY_URL}/object_info" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "?")
    ok "ComfyUI reports ${NODE_COUNT} registered node types"
    
    # Save node list for workflow checking
    NODES_CACHE=$(mktemp)
    curl -sf "${COMFY_URL}/object_info" | python3 -c "import sys,json; [print(k) for k in json.load(sys.stdin)]" > "$NODES_CACHE" 2>/dev/null
else
    warn "Cannot fetch node registry from ComfyUI API"
    NODES_CACHE=""
fi

# --- Workflow Compatibility Check --------------------------------------------
if [[ -n "$CHECK_WORKFLOW" ]]; then
    header "Workflow Compatibility: $(basename $CHECK_WORKFLOW)"
    
    if [[ ! -f "$CHECK_WORKFLOW" ]]; then
        fail "Workflow file not found: $CHECK_WORKFLOW"
    elif ! $HAS_JQ; then
        fail "jq required for workflow analysis"
    else
        # Extract all class_types from workflow
        # Handle both API format {"1": {"class_type": ...}} and UI format {"nodes": [{"type": ...}]}
        WORKFLOW_NODES=$(jq -r '
            if .nodes then
                # UI format
                .nodes[].type
            else
                # API format
                .[].class_type
            end
        ' "$CHECK_WORKFLOW" 2>/dev/null | sort -u)
        
        if [[ -z "$WORKFLOW_NODES" ]]; then
            fail "Could not parse node types from workflow"
        else
            echo -e "  ${BOLD}Required nodes:${NC}"
            MISSING_NODES=()
            while IFS= read -r node_type; do
                if [[ -n "$NODES_CACHE" ]] && grep -qx "$node_type" "$NODES_CACHE" 2>/dev/null; then
                    ok "$node_type"
                elif [[ -n "$NODES_CACHE" ]]; then
                    fail "$node_type — NOT REGISTERED"
                    
                    # Fuzzy match suggestions
                    SUGGESTIONS=$(grep -i "$(echo $node_type | sed 's/[_-]/ /g' | awk '{print $1}')" "$NODES_CACHE" 2>/dev/null | head -3)
                    if [[ -n "$SUGGESTIONS" ]]; then
                        info "  └─ Did you mean: $(echo $SUGGESTIONS | tr '\n' ', ')"
                    fi
                    MISSING_NODES+=("$node_type")
                else
                    info "$node_type (cannot verify — API unavailable)"
                fi
            done <<< "$WORKFLOW_NODES"
            
            if [[ ${#MISSING_NODES[@]} -gt 0 ]]; then
                echo ""
                warn "${#MISSING_NODES[@]} node(s) missing — workflow will fail"
                info "These nodes may have been renamed in newer versions"
                info "Check installed versions vs workflow version"
            fi
        fi
        
        # Check referenced models
        echo -e "\n  ${BOLD}Referenced models/files:${NC}"
        REFERENCED_FILES=$(jq -r '
            if .nodes then
                .nodes[].widgets_values[]?
            else
                .. | strings
            end
        ' "$CHECK_WORKFLOW" 2>/dev/null | grep -E "\.(safetensors|pth|ckpt|pt|png|mp4|jpg)$" | sort -u)
        
        if [[ -n "$REFERENCED_FILES" ]]; then
            while IFS= read -r ref; do
                # Check in common locations
                FOUND=false
                for search_dir in "$MODELS_HOST" /mnt/ssd/comfyui-input; do
                    if find "$search_dir" -name "$(basename $ref)" 2>/dev/null | head -1 | grep -q .; then
                        ok "$ref"
                        FOUND=true
                        break
                    fi
                done
                if ! $FOUND; then
                    warn "Referenced file not found locally: $ref"
                fi
            done <<< "$REFERENCED_FILES"
        else
            info "No model/file references detected in workflow"
        fi
    fi
fi

# --- Bulk Workflow Check (scan a directory) -----------------------------------
if [[ -z "$CHECK_WORKFLOW" ]] && [[ -n "$NODES_CACHE" ]]; then
    # Check all workflows in the container's workflow dir
    WORKFLOW_FILES=$(docker exec "$CONTAINER" find /app/workflows -name "*.json" 2>/dev/null)
    if [[ -n "$WORKFLOW_FILES" ]]; then
        header "Workflow Scan (container workflows)"
        while IFS= read -r wf; do
            WF_NAME=$(basename "$wf")
            WF_CONTENT=$(docker exec "$CONTAINER" cat "$wf" 2>/dev/null)
            MISSING=$(echo "$WF_CONTENT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    nodes = set()
    if 'nodes' in data:
        nodes = {n['type'] for n in data['nodes']}
    else:
        nodes = {v['class_type'] for v in data.values() if isinstance(v, dict) and 'class_type' in v}
    
    with open('$NODES_CACHE') as f:
        registered = {line.strip() for line in f}
    
    missing = nodes - registered
    if missing:
        print(' '.join(missing))
except:
    pass
" 2>/dev/null)
            if [[ -n "$MISSING" ]]; then
                warn "${WF_NAME}: missing nodes: ${MISSING}"
            else
                ok "${WF_NAME}"
            fi
        done <<< "$WORKFLOW_FILES"
    fi
fi

# --- Disk Space ---------------------------------------------------------------
header "Disk Space"

MODELS_USAGE=$(df -h "$MODELS_HOST" 2>/dev/null | tail -1 | awk '{print $3 "/" $2 " (" $5 " used)"}')
HOME_USAGE=$(df -h /home 2>/dev/null | tail -1 | awk '{print $3 "/" $2 " (" $5 " used)"}')
echo -e "  Models disk: ${MODELS_USAGE}"
echo -e "  Home disk:   ${HOME_USAGE}"

MODELS_PCT=$(df "$MODELS_HOST" 2>/dev/null | tail -1 | awk '{gsub(/%/,""); print $5}')
if [[ "$MODELS_PCT" -gt 90 ]]; then
    fail "Models disk >90% full!"
elif [[ "$MODELS_PCT" -gt 75 ]]; then
    warn "Models disk >75% full"
else
    ok "Disk space healthy"
fi

# --- Empty model directories (may need population) ---------------------------
header "Empty Model Directories"
for dir in "$MODELS_HOST"/*/; do
    dirname=$(basename "$dir")
    count=$(find "$dir" \( -name "*.safetensors" -o -name "*.pth" -o -name "*.ckpt" -o -name "*.pt" \) 2>/dev/null | wc -l)
    if [[ "$count" -eq 0 ]]; then
        info "${dirname}/ — empty (may need models for some workflows)"
    fi
done

# --- Summary ------------------------------------------------------------------
header "Summary"
echo -e "  Errors:   ${RED}${ERRORS}${NC}"
echo -e "  Warnings: ${YELLOW}${WARNINGS}${NC}"
if $FIX_MODE; then
    echo -e "  Fixed:    ${GREEN}${FIXES}${NC}"
fi
echo ""

if [[ "$ERRORS" -eq 0 && "$WARNINGS" -eq 0 ]]; then
    echo -e "  ${GREEN}${BOLD}All checks passed!${NC}"
elif [[ "$ERRORS" -eq 0 ]]; then
    echo -e "  ${YELLOW}${BOLD}Passed with warnings${NC}"
else
    echo -e "  ${RED}${BOLD}Issues found — fix before running workflows${NC}"
    if ! $FIX_MODE; then
        echo -e "  Run with ${BOLD}--fix${NC} to auto-fix what we can"
    fi
fi

# Cleanup
[[ -n "${NODES_CACHE:-}" ]] && rm -f "$NODES_CACHE"

exit $ERRORS