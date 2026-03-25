#!/usr/bin/env bash
# =============================================================================
# bootstrap.sh — ComfyUI Pipeline First-Time Setup & Model Management
# =============================================================================
#
# Run on the ComfyUI host (10.10.210.80) as jeffw.
#
# Usage:
#   ./bootstrap.sh              # Full setup (dirs + models + validate)
#   ./bootstrap.sh --dirs-only  # Just create directory structure
#   ./bootstrap.sh --sam3d      # Download SAM3D models only
#   ./bootstrap.sh --validate   # Validate existing setup
#   ./bootstrap.sh --cleanup    # Remove duplicate/orphaned model files
#
# Prerequisites:
#   - .env file with HF_TOKEN set
#   - HuggingFace account with access to facebook/sam-3d-objects
#   - /mnt/ssd mounted and writable
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSD_BASE="/mnt/ssd"
MODELS_DIR="${SSD_BASE}/comfyui-models"
INPUT_DIR="${SSD_BASE}/comfyui-input"
OUTPUT_DIR="${SSD_BASE}/comfyui-output"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# --- Load .env ---------------------------------------------------------------
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    set -a
    source "${SCRIPT_DIR}/.env"
    set +a
    echo -e "${GREEN}✓${NC} Loaded .env"
else
    echo -e "${YELLOW}⚠${NC} No .env file found — HF downloads will fail for gated repos"
    echo "  Create .env with: HF_TOKEN=hf_your_token_here"
fi

# --- Argument parsing --------------------------------------------------------
ACTION="full"
while [[ $# -gt 0 ]]; do
    case $1 in
        --dirs-only) ACTION="dirs"; shift ;;
        --sam3d)     ACTION="sam3d"; shift ;;
        --gvhmr)     ACTION="gvhmr"; shift ;;
        --validate)  ACTION="validate"; shift ;;
        --cleanup)   ACTION="cleanup"; shift ;;
        -h|--help)
            echo "Usage: $0 [--dirs-only|--sam3d|--gvhmr|--validate|--cleanup]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Helpers -----------------------------------------------------------------
ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }
info() { echo -e "  ${CYAN}ℹ${NC} $1"; }
header() { echo -e "\n${BOLD}═══ $1 ═══${NC}"; }

# =============================================================================
# DIRECTORY STRUCTURE
# =============================================================================
setup_dirs() {
    header "Directory Structure"

    # Verify SSD is mounted
    if ! mountpoint -q "$SSD_BASE" 2>/dev/null; then
        fail "/mnt/ssd is not a mount point — is the SSD mounted?"
        echo "  Run: sudo mount /dev/sdb1 /mnt/ssd"
        exit 1
    fi
    ok "SSD mounted at ${SSD_BASE}"

    # Create model subdirectories
    # These match what ComfyUI's folder_paths expects + custom node requirements
    MODEL_DIRS=(
        checkpoints
        clip
        clip_vision
        controlnet
        diffusion_models
        dwpose
        ipadapter
        loras
        onnx
        sam2
        sams
        sam3d
        upscale_models
        vae
        wan
        .hf_cache
    )

    for dir in "${MODEL_DIRS[@]}"; do
        mkdir -p "${MODELS_DIR}/${dir}"
    done
    ok "Model directories created"

    # Create input/output
    mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"
    ok "Input/Output directories created"

    # Ensure symlinks from home
    for name in comfyui-models comfyui-input comfyui-output; do
        target="${SSD_BASE}/${name}"
        link="${HOME}/${name}"
        if [[ -L "$link" ]]; then
            current=$(readlink "$link")
            if [[ "$current" == "$target" ]]; then
                ok "Symlink OK: ${link} -> ${target}"
            else
                warn "Symlink points elsewhere: ${link} -> ${current}"
                info "Expected: ${target}"
            fi
        elif [[ -d "$link" ]]; then
            warn "${link} is a real directory, not a symlink"
            info "Consider: rm -rf ${link} && ln -s ${target} ${link}"
        else
            ln -s "$target" "$link"
            ok "Created symlink: ${link} -> ${target}"
        fi
    done

    # Ownership
    chown -R "$(whoami)":"$(whoami)" "$MODELS_DIR" "$INPUT_DIR" "$OUTPUT_DIR" 2>/dev/null || true
    ok "Ownership set"

    # Disk space
    AVAIL=$(df -h "$SSD_BASE" | tail -1 | awk '{print $4}')
    info "SSD available: ${AVAIL}"
}

# =============================================================================
# SAM3D MODEL DOWNLOAD
# =============================================================================
download_sam3d() {
    header "SAM3D Models"

    SAM3D_DIR="${MODELS_DIR}/sam3d/hf/checkpoints"
    mkdir -p "$SAM3D_DIR"

    # Check HF token
    if [[ -z "${HF_TOKEN:-}" || "${HF_TOKEN}" == "hf_REPLACE_WITH_YOUR_TOKEN" ]]; then
        fail "HF_TOKEN not set or still placeholder"
        echo ""
        echo "  SAM3D requires access to a gated HuggingFace repo."
        echo "  Steps:"
        echo "    1. Create account at https://huggingface.co"
        echo "    2. Request access at https://huggingface.co/facebook/sam-3d-objects"
        echo "    3. Create token at https://huggingface.co/settings/tokens"
        echo "    4. Set HF_TOKEN=hf_... in ${SCRIPT_DIR}/.env"
        return 1
    fi

    # Check if already downloaded (look for pipeline.yaml + a large ckpt)
    if [[ -f "${SAM3D_DIR}/pipeline.yaml" ]] && \
       find "$SAM3D_DIR" -name "*.ckpt" -size +100M 2>/dev/null | head -1 | grep -q .; then
        ok "SAM3D models already downloaded"
        info "Files in ${SAM3D_DIR}:"
        ls -lh "$SAM3D_DIR" | grep -E "\.(ckpt|yaml|safetensors|pt)" | while read line; do
            echo "    $line"
        done
        return 0
    fi

    info "Downloading SAM3D models from HuggingFace..."
    info "This downloads ~15GB and may take 10-30 minutes"
    echo ""

    # Use huggingface-cli for reliable download with resume support
    if command -v huggingface-cli &>/dev/null; then
        HF_TOKEN="$HF_TOKEN" huggingface-cli download \
            facebook/sam-3d-objects \
            --local-dir "${MODELS_DIR}/sam3d/hf" \
            --token "$HF_TOKEN" \
            --local-dir-use-symlinks False
    else
        # Fallback: use Python
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'facebook/sam-3d-objects',
    local_dir='${MODELS_DIR}/sam3d/hf',
    token='${HF_TOKEN}',
    local_dir_use_symlinks=False
)
print('Download complete!')
"
    fi

    # Validate
    if [[ -f "${SAM3D_DIR}/pipeline.yaml" ]]; then
        ok "SAM3D download complete"
        TOTAL=$(du -sh "${MODELS_DIR}/sam3d" | cut -f1)
        info "Total size: ${TOTAL}"
    else
        fail "Download completed but pipeline.yaml not found"
        info "Check ${SAM3D_DIR}/ for partial downloads"
    fi
}

# =============================================================================
# GVHMR MODEL DOWNLOAD
# =============================================================================
download_gvhmr() {
    header "GVHMR Models"

    GVHMR_DIR="${MODELS_DIR}/gvhmr"
    mkdir -p "${GVHMR_DIR}"

    # HF token optional for this repo (camenduru/GVHMR is public)
    if [[ -z "${HF_TOKEN:-}" || "${HF_TOKEN}" == "hf_REPLACE_WITH_YOUR_TOKEN" ]]; then
        info "HF_TOKEN not set — camenduru/GVHMR is public so this should still work"
    fi

    # --- Helper: download a single file from HuggingFace ---
    hf_download_file() {
        local repo="$1"
        local remote_file="$2"
        local dest="$3"

        if [[ -f "$dest" ]]; then
            ok "Already present: $(basename "$dest")"
            return 0
        fi

        info "Downloading ${repo}/${remote_file}..."
        mkdir -p "$(dirname "$dest")"
        python3 -c "
from huggingface_hub import hf_hub_download
import os, shutil
token = os.environ.get('HF_TOKEN') or None
cached = hf_hub_download(
    repo_id='${repo}',
    filename='${remote_file}',
    token=token,
    cache_dir='${MODELS_DIR}/.hf_cache',
)
shutil.copy2(cached, '${dest}')
print('  saved to ${dest}')
"
        if [[ -f "$dest" ]]; then
            SIZE=$(du -sh "$dest" | cut -f1)
            ok "$(basename "$dest") (${SIZE})"
        else
            fail "Download failed: $(basename "$dest")"
            return 1
        fi
    }

    # --- GVHMR main checkpoint ---
    hf_download_file \
        "camenduru/GVHMR" \
        "gvhmr/gvhmr_siga24_release.ckpt" \
        "${GVHMR_DIR}/gvhmr/gvhmr_siga24_release.ckpt"

    # --- HMR2 backbone ---
    hf_download_file \
        "camenduru/GVHMR" \
        "hmr2/epoch=10-step=25000.ckpt" \
        "${GVHMR_DIR}/hmr2/epoch=10-step=25000.ckpt"

    # --- ViTPose detector ---
    hf_download_file \
        "camenduru/GVHMR" \
        "vitpose/vitpose-h-multi-coco.pth" \
        "${GVHMR_DIR}/vitpose/vitpose-h-multi-coco.pth"

    # --- YOLO detector ---
    hf_download_file \
        "camenduru/GVHMR" \
        "yolo/yolov8x.pt" \
        "${GVHMR_DIR}/yolo/yolov8x.pt"

    # --- DPVO camera trajectory estimator (moving camera variant only) ---
    info "Downloading DPVO (moving camera variant)..."
    hf_download_file \
        "camenduru/GVHMR" \
        "dpvo/dpvo.pth" \
        "${GVHMR_DIR}/dpvo/dpvo.pth" || warn "DPVO download failed — only needed for gvhmr-pose-moving workflow"

    # --- Validate ---
    echo ""
    REQUIRED=(
        "gvhmr/gvhmr_siga24_release.ckpt"
        "hmr2/epoch=10-step=25000.ckpt"
        "vitpose/vitpose-h-multi-coco.pth"
        "yolo/yolov8x.pt"
    )
    ERRORS=0
    for f in "${REQUIRED[@]}"; do
        if [[ -f "${GVHMR_DIR}/${f}" ]]; then
            SIZE=$(du -sh "${GVHMR_DIR}/${f}" | cut -f1)
            ok "${f} (${SIZE})"
        else
            fail "Missing: ${f}"
            ((ERRORS++))
        fi
    done

    echo ""
    TOTAL=$(du -sh "${GVHMR_DIR}" | cut -f1)
    info "GVHMR total: ${TOTAL}"

    if [[ "$ERRORS" -eq 0 ]]; then
        ok "GVHMR models ready"
    else
        fail "${ERRORS} required file(s) missing"
    fi
}

# =============================================================================
# VALIDATE
# =============================================================================
validate() {
    header "Validation"

    ERRORS=0

    # Check for broken model files
    BROKEN=$(find "$MODELS_DIR" \( -name "*.safetensors" -o -name "*.pth" -o -name "*.ckpt" \) -size -1k 2>/dev/null)
    if [[ -n "$BROKEN" ]]; then
        while IFS= read -r f; do
            SIZE=$(stat -c%s "$f" 2>/dev/null || echo "?")
            fail "Broken model (${SIZE}b): $(basename $f)"
            ((ERRORS++))
        done <<< "$BROKEN"
    else
        ok "No broken model files"
    fi

    # Check for broken symlinks
    BROKEN_LINKS=$(find "$MODELS_DIR" -xtype l 2>/dev/null)
    if [[ -n "$BROKEN_LINKS" ]]; then
        while IFS= read -r f; do
            fail "Broken symlink: $f"
            ((ERRORS++))
        done <<< "$BROKEN_LINKS"
    else
        ok "No broken symlinks"
    fi

    # Verify SAM3D structure
    SAM3D_DIR="${MODELS_DIR}/sam3d/hf/checkpoints"
    if [[ -f "${SAM3D_DIR}/pipeline.yaml" ]]; then
        # Check it references files that exist
        while IFS=': ' read -r key value; do
            if [[ "$key" == *_ckpt_path ]]; then
                if [[ -f "${SAM3D_DIR}/${value}" ]]; then
                    ok "SAM3D ${value} exists"
                else
                    fail "SAM3D missing: ${value} (referenced in pipeline.yaml)"
                    ((ERRORS++))
                fi
            fi
        done < <(grep "_ckpt_path:" "${SAM3D_DIR}/pipeline.yaml")
    else
        warn "SAM3D not downloaded yet (run with --sam3d)"
    fi

    # Model inventory
    echo ""
    info "Model inventory:"
    TOTAL_SIZE=0
    for dir in "$MODELS_DIR"/*/; do
        dirname=$(basename "$dir")
        [[ "$dirname" == .* ]] && continue
        count=$(find "$dir" \( -name "*.safetensors" -o -name "*.pth" -o -name "*.ckpt" -o -name "*.pt" \) 2>/dev/null | wc -l)
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        [[ "$count" -gt 0 ]] && echo "    ${dirname}: ${count} file(s), ${size}"
    done
    TOTAL=$(du -sh "$MODELS_DIR" 2>/dev/null | cut -f1)
    echo "    ─────────────────────"
    echo "    Total: ${TOTAL}"

    echo ""
    if [[ "$ERRORS" -eq 0 ]]; then
        ok "All checks passed"
    else
        fail "${ERRORS} issue(s) found"
    fi
}

# =============================================================================
# CLEANUP
# =============================================================================
cleanup() {
    header "Cleanup"

    # Remove orphaned sam3dobjects if sam3d/hf/checkpoints has the real files
    SAM3D_LEGACY="${MODELS_DIR}/sam3dobjects"
    SAM3D_PROPER="${MODELS_DIR}/sam3d/hf/checkpoints"

    if [[ -d "$SAM3D_LEGACY" ]] && [[ -f "${SAM3D_PROPER}/pipeline.yaml" ]]; then
        # Check that proper dir has actual model files (not just our safetensors copies)
        PROPER_CKPTS=$(find "$SAM3D_PROPER" -name "*.ckpt" -size +100M 2>/dev/null | wc -l)
        if [[ "$PROPER_CKPTS" -gt 0 ]]; then
            LEGACY_SIZE=$(du -sh "$SAM3D_LEGACY" | cut -f1)
            info "sam3dobjects/ (${LEGACY_SIZE}) is superseded by sam3d/hf/checkpoints/"
            read -p "  Delete ${SAM3D_LEGACY}? [y/N] " confirm
            if [[ "$confirm" == [yY] ]]; then
                rm -rf "$SAM3D_LEGACY"
                ok "Removed legacy sam3dobjects/ (freed ${LEGACY_SIZE})"
            fi
        else
            warn "sam3d/hf/checkpoints/ doesn't have .ckpt files yet — keeping sam3dobjects/"
            info "Run --sam3d first to download proper models, then --cleanup"
        fi
    elif [[ -d "$SAM3D_LEGACY" ]]; then
        info "sam3dobjects/ exists but sam3d/hf/checkpoints/ not set up"
        info "Run --sam3d first, then --cleanup"
    fi

    # Remove broken files
    BROKEN=$(find "$MODELS_DIR" \( -name "*.safetensors" -o -name "*.pth" -o -name "*.ckpt" \) -size -1k 2>/dev/null)
    if [[ -n "$BROKEN" ]]; then
        echo ""
        info "Found broken model files:"
        echo "$BROKEN" | while read f; do echo "    $f"; done
        read -p "  Delete all broken files? [y/N] " confirm
        if [[ "$confirm" == [yY] ]]; then
            echo "$BROKEN" | while read f; do rm -f "$f"; done
            ok "Removed broken files"
        fi
    fi

    # Remove broken symlinks
    BROKEN_LINKS=$(find "$MODELS_DIR" -xtype l 2>/dev/null)
    if [[ -n "$BROKEN_LINKS" ]]; then
        echo ""
        info "Found broken symlinks:"
        echo "$BROKEN_LINKS" | while read f; do echo "    $f -> $(readlink $f)"; done
        read -p "  Delete all broken symlinks? [y/N] " confirm
        if [[ "$confirm" == [yY] ]]; then
            echo "$BROKEN_LINKS" | while read f; do rm -f "$f"; done
            ok "Removed broken symlinks"
        fi
    fi

    ok "Cleanup complete"
    AVAIL=$(df -h "$SSD_BASE" | tail -1 | awk '{print $4}')
    info "SSD available: ${AVAIL}"
}

# =============================================================================
# MAIN
# =============================================================================
echo -e "${BOLD}ComfyUI Pipeline Bootstrap${NC}"
echo "Server: $(hostname) | SSD: ${SSD_BASE} | Models: ${MODELS_DIR}"
echo ""

case "$ACTION" in
    full)
        setup_dirs
        download_sam3d
        download_gvhmr
        validate
        echo ""
        header "Next Steps"
        echo "  1. Review any warnings above"
        echo "  2. Build and start: docker compose up -d --build"
        echo "  3. Verify: ./comfy_doctor.sh"
        ;;
    dirs)
        setup_dirs
        ;;
    sam3d)
        download_sam3d
        ;;
    gvhmr)
        download_gvhmr
        ;;
    validate)
        validate
        ;;
    cleanup)
        cleanup
        ;;
esac