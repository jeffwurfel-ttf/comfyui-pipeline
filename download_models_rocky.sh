#!/bin/bash
# =============================================================================
# ComfyUI Model Downloader — Rocky Linux Deployment
# =============================================================================
#
# Single script to download ALL models needed for the ComfyUI pipeline.
# Profiles: SDXL, Flux Dev (fp8), Wan 2.2 Animate, Wan 2.1 T2V, ESRGAN
#
# Usage:
#   chmod +x download_models_rocky.sh
#   ./download_models_rocky.sh                       # Download everything
#   ./download_models_rocky.sh --profile sdxl        # Just SDXL
#   ./download_models_rocky.sh --profile flux        # Just Flux Dev
#   ./download_models_rocky.sh --profile wan         # Wan 2.2 Animate + deps
#   ./download_models_rocky.sh --profile wan-t2v     # Wan 2.1 T2V + deps
#   ./download_models_rocky.sh --profile esrgan      # ESRGAN upscalers
#   ./download_models_rocky.sh --check               # Show what's missing
#
# Environment:
#   MODELS_DIR   — Where models live (default: /data/comfyui/models)
#   HF_TOKEN     — HuggingFace token for gated repos (Flux VAE)
#
# Total download (all profiles): ~65 GB
#   SDXL:          ~7 GB
#   Flux Dev fp8: ~17 GB
#   Wan 2.2 I2V:  ~20 GB  (includes shared Wan deps: VAE + UMT5)
#   Wan 2.1 T2V:  ~15 GB  (model only; shares VAE/CLIP with I2V)
#   ESRGAN:       ~0.2 GB
# =============================================================================

set -e

# Configuration
MODELS_DIR="${MODELS_DIR:-/data/comfyui/models}"
HF_TOKEN="${HF_TOKEN:-}"
PROFILE="${1:-all}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse args
CHECK_ONLY=false
if [ "$1" = "--check" ]; then
    CHECK_ONLY=true
    PROFILE="all"
elif [ "$1" = "--profile" ]; then
    PROFILE="${2:-all}"
fi

# =============================================================================
# Helpers
# =============================================================================

log()  { echo -e "${GREEN}[+]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[x]${NC} $1"; }
info() { echo -e "${BLUE}[i]${NC} $1"; }

DOWNLOAD_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

download_hf() {
    local repo="$1"
    local filename="$2"
    local dest="$3"
    local subfolder="${4:-}"

    if [ -f "$dest" ]; then
        local size_mb=$(du -m "$dest" | cut -f1)
        log "Exists: $(basename $dest) (${size_mb}MB)"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        return 0
    fi

    mkdir -p "$(dirname $dest)"

    local url="https://huggingface.co/${repo}/resolve/main"
    if [ -n "$subfolder" ]; then
        url="${url}/${subfolder}/${filename}"
    else
        url="${url}/${filename}"
    fi

    info "Downloading: ${repo} / ${filename}"
    info "  -> ${dest}"

    local auth_header=""
    if [ -n "$HF_TOKEN" ]; then
        auth_header="--header=Authorization: Bearer ${HF_TOKEN}"
    fi

    if command -v wget &> /dev/null; then
        wget -c --show-progress -q \
            ${auth_header} \
            -O "${dest}.tmp" \
            "$url" && mv "${dest}.tmp" "$dest"
    elif command -v curl &> /dev/null; then
        local curl_opts=(-L -C - --progress-bar -o "${dest}.tmp")
        if [ -n "$HF_TOKEN" ]; then
            curl_opts+=(-H "Authorization: Bearer ${HF_TOKEN}")
        fi
        curl "${curl_opts[@]}" "$url" && mv "${dest}.tmp" "$dest"
    else
        err "Neither wget nor curl found. Install: dnf install wget"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi

    if [ -f "$dest" ]; then
        local size_mb=$(du -m "$dest" | cut -f1)
        log "Downloaded: $(basename $dest) (${size_mb}MB)"
        DOWNLOAD_COUNT=$((DOWNLOAD_COUNT + 1))
    else
        err "Download FAILED: $dest"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi
}

download_url() {
    local url="$1"
    local dest="$2"

    if [ -f "$dest" ]; then
        local size_mb=$(du -m "$dest" | cut -f1)
        log "Exists: $(basename $dest) (${size_mb}MB)"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        return 0
    fi

    mkdir -p "$(dirname $dest)"
    info "Downloading: $(basename $dest)"
    info "  -> ${dest}"

    if command -v wget &> /dev/null; then
        wget -c --show-progress -q -O "${dest}.tmp" "$url" && mv "${dest}.tmp" "$dest"
    else
        curl -L -C - --progress-bar -o "${dest}.tmp" "$url" && mv "${dest}.tmp" "$dest"
    fi

    if [ -f "$dest" ]; then
        local size_mb=$(du -m "$dest" | cut -f1)
        log "Downloaded: $(basename $dest) (${size_mb}MB)"
        DOWNLOAD_COUNT=$((DOWNLOAD_COUNT + 1))
    else
        err "Download FAILED: $dest"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi
}

check_file() {
    local path="$1"
    local desc="$2"
    if [ -f "$path" ]; then
        local size_mb=$(du -m "$path" | cut -f1)
        echo -e "  ${GREEN}✓${NC} ${desc} (${size_mb}MB)"
    else
        echo -e "  ${RED}✗${NC} ${desc} — MISSING"
    fi
}

# =============================================================================
# Profile: SDXL (~7 GB)
# =============================================================================

download_sdxl() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  SDXL Profile (~7 GB)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    download_hf \
        "stabilityai/stable-diffusion-xl-base-1.0" \
        "sd_xl_base_1.0.safetensors" \
        "${MODELS_DIR}/checkpoints/sd_xl_base_1.0.safetensors"

    download_hf \
        "stabilityai/sdxl-vae" \
        "sdxl_vae.safetensors" \
        "${MODELS_DIR}/vae/sdxl_vae.safetensors"
}

# =============================================================================
# Profile: Flux Dev fp8 (~17 GB)
# =============================================================================

download_flux() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Flux Dev fp8 Profile (~17 GB)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Flux UNET fp8 (~11.9 GB)
    download_hf \
        "Comfy-Org/flux1-dev" \
        "flux1-dev-fp8.safetensors" \
        "${MODELS_DIR}/diffusion_models/flux1-dev-fp8.safetensors"

    # T5-XXL fp8 text encoder (~4.89 GB)
    download_hf \
        "comfyanonymous/flux_text_encoders" \
        "t5xxl_fp8_e4m3fn.safetensors" \
        "${MODELS_DIR}/clip/t5xxl_fp8_e4m3fn.safetensors"

    # CLIP-L text encoder (~246 MB)
    download_hf \
        "comfyanonymous/flux_text_encoders" \
        "clip_l.safetensors" \
        "${MODELS_DIR}/clip/clip_l.safetensors"

    # Flux VAE (~335 MB) — GATED REPO, needs HF_TOKEN
    # Accept license: https://huggingface.co/black-forest-labs/FLUX.1-dev
    download_hf \
        "black-forest-labs/FLUX.1-dev" \
        "ae.safetensors" \
        "${MODELS_DIR}/vae/ae.safetensors"
}

# =============================================================================
# Shared Wan Dependencies (~6 GB)
# Used by both Wan 2.2 Animate and Wan 2.1 T2V
# =============================================================================

download_wan_shared() {
    echo ""
    info "Downloading shared Wan dependencies..."

    # Wan 2.1 VAE (~335 MB)
    download_hf \
        "Kijai/WanVideo_comfy" \
        "wan_2.1_vae.safetensors" \
        "${MODELS_DIR}/vae/wan_2.1_vae.safetensors"

    # UMT5-XXL text encoder bf16 (~5 GB)
    download_hf \
        "Kijai/WanVideo_comfy" \
        "umt5-xxl-enc-bf16.safetensors" \
        "${MODELS_DIR}/clip/umt5-xxl-enc-bf16.safetensors"
}

# =============================================================================
# Profile: Wan 2.2 Animate / I2V (~21 GB with shared deps)
# =============================================================================

download_wan() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Wan 2.2 Animate (I2V) Profile (~21 GB)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Shared Wan deps (VAE + UMT5)
    download_wan_shared

    # Wan 2.2 Animate model fp8 (~14 GB)
    mkdir -p "${MODELS_DIR}/diffusion_models/Wan22Animate"
    download_hf \
        "Kijai/WanVideo_comfy" \
        "Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors" \
        "${MODELS_DIR}/diffusion_models/Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors"

    # CLIP Vision H (~3.9 GB) — used for reference image conditioning in I2V
    download_hf \
        "Kijai/WanVideo_comfy" \
        "clip_vision_h.safetensors" \
        "${MODELS_DIR}/clip_vision/clip_vision_h.safetensors"

    # LoRA: Relight (~200 MB)
    download_hf \
        "Kijai/WanVideo_comfy" \
        "WanAnimate_relight_lora_fp16.safetensors" \
        "${MODELS_DIR}/loras/WanAnimate_relight_lora_fp16.safetensors"

    # LoRA: LightX2V CFG step distill (~400 MB)
    download_hf \
        "Kijai/WanVideo_comfy" \
        "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" \
        "${MODELS_DIR}/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"

    # SAM2 — segmentation/masking (~900 MB)
    download_hf \
        "Kijai/sam2-safetensors" \
        "sam2.1_hiera_large.safetensors" \
        "${MODELS_DIR}/sam2/sam2.1_hiera_large.safetensors"

    info "Note: Pose ONNX models (vitpose, yolo) auto-download on first use."
}

# =============================================================================
# Profile: Wan 2.1 T2V (~15 GB with shared deps)
# =============================================================================

download_wan_t2v() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Wan 2.1 Text-to-Video Profile (~15 GB)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Shared Wan deps (VAE + UMT5) — skips if already downloaded
    download_wan_shared

    # Wan 2.1 T2V 14B fp8 (~14.9 GB)
    # Text-to-video model (no reference image needed)
    download_hf \
        "Kijai/WanVideo_comfy" \
        "Wan2_1-T2V-14B_fp8_e4m3fn.safetensors" \
        "${MODELS_DIR}/diffusion_models/Wan2_1-T2V-14B_fp8_e4m3fn.safetensors"
}

# =============================================================================
# Profile: ESRGAN Upscalers (~0.2 GB)
# =============================================================================

download_esrgan() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  ESRGAN Profile (~0.2 GB)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    download_url \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
        "${MODELS_DIR}/upscale_models/RealESRGAN_x4plus.pth"

    download_url \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth" \
        "${MODELS_DIR}/upscale_models/RealESRGAN_x4plus_anime_6B.pth"

    download_url \
        "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth" \
        "${MODELS_DIR}/upscale_models/4x-UltraSharp.pth"
}

# =============================================================================
# Check Mode
# =============================================================================

check_models() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Model Status — ${MODELS_DIR}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    echo "  SDXL Profile:"
    check_file "${MODELS_DIR}/checkpoints/sd_xl_base_1.0.safetensors" "SDXL Base 1.0 checkpoint"
    check_file "${MODELS_DIR}/vae/sdxl_vae.safetensors" "SDXL VAE"
    echo ""

    echo "  Flux Dev fp8 Profile:"
    check_file "${MODELS_DIR}/diffusion_models/flux1-dev-fp8.safetensors" "Flux UNET fp8"
    check_file "${MODELS_DIR}/clip/t5xxl_fp8_e4m3fn.safetensors" "T5-XXL fp8 encoder"
    check_file "${MODELS_DIR}/clip/clip_l.safetensors" "CLIP-L encoder"
    check_file "${MODELS_DIR}/vae/ae.safetensors" "Flux VAE (ae)"
    echo ""

    echo "  Wan 2.2 Animate (I2V) Profile:"
    check_file "${MODELS_DIR}/diffusion_models/Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors" "Wan 2.2 Animate 14B fp8"
    check_file "${MODELS_DIR}/clip_vision/clip_vision_h.safetensors" "CLIP Vision H"
    check_file "${MODELS_DIR}/loras/WanAnimate_relight_lora_fp16.safetensors" "Relight LoRA"
    check_file "${MODELS_DIR}/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" "LightX2V LoRA"
    check_file "${MODELS_DIR}/sam2/sam2.1_hiera_large.safetensors" "SAM2 Large"
    echo ""

    echo "  Wan 2.1 T2V Profile:"
    check_file "${MODELS_DIR}/diffusion_models/Wan2_1-T2V-14B_fp8_e4m3fn.safetensors" "Wan 2.1 T2V 14B fp8"
    echo ""

    echo "  Shared Wan Dependencies:"
    check_file "${MODELS_DIR}/vae/wan_2.1_vae.safetensors" "Wan 2.1 VAE"
    check_file "${MODELS_DIR}/clip/umt5-xxl-enc-bf16.safetensors" "UMT5-XXL encoder"
    echo ""

    echo "  ESRGAN Profile:"
    check_file "${MODELS_DIR}/upscale_models/RealESRGAN_x4plus.pth" "RealESRGAN x4plus"
    check_file "${MODELS_DIR}/upscale_models/RealESRGAN_x4plus_anime_6B.pth" "RealESRGAN Anime"
    check_file "${MODELS_DIR}/upscale_models/4x-UltraSharp.pth" "4x-UltraSharp"
    echo ""

    # Disk usage
    if [ -d "${MODELS_DIR}" ]; then
        local total=$(du -sh "${MODELS_DIR}" 2>/dev/null | cut -f1)
        echo -e "  Total disk usage: ${total}"
    fi
}

# =============================================================================
# Main
# =============================================================================

echo "=========================================="
echo "  ComfyUI Model Downloader — Rocky Linux"
echo "=========================================="
echo "  Models dir: ${MODELS_DIR}"
echo "  Profile:    ${PROFILE}"
if [ -n "$HF_TOKEN" ]; then
    echo "  HF Token:   set (${#HF_TOKEN} chars)"
else
    echo "  HF Token:   not set (needed for Flux VAE gated repo)"
fi
echo "=========================================="

# Check mode
if [ "$CHECK_ONLY" = true ]; then
    check_models
    exit 0
fi

# Create base directory structure
log "Creating directory structure..."
mkdir -p "${MODELS_DIR}"/{checkpoints,diffusion_models,vae,clip,clip_vision,loras,controlnet,upscale_models,sam2,ipadapter,wan,dwpose}

# Download based on profile
case "$PROFILE" in
    sdxl)
        download_sdxl
        ;;
    flux|flux-dev)
        download_flux
        ;;
    wan|wan-video|wan22|wan-i2v)
        download_wan
        ;;
    wan-t2v|wan21-t2v)
        download_wan_t2v
        ;;
    wan-all)
        download_wan
        download_wan_t2v
        ;;
    esrgan|upscale)
        download_esrgan
        ;;
    all)
        download_sdxl
        download_flux
        download_wan
        download_wan_t2v
        download_esrgan
        ;;
    *)
        err "Unknown profile: $PROFILE"
        echo "  Available: sdxl, flux, wan, wan-t2v, wan-all, esrgan, all"
        exit 1
        ;;
esac

# Final report
echo ""
echo "=========================================="
echo "  Download Complete"
echo "=========================================="
echo -e "  Downloaded: ${GREEN}${DOWNLOAD_COUNT}${NC} files"
echo -e "  Skipped:    ${BLUE}${SKIP_COUNT}${NC} (already exist)"
if [ $FAIL_COUNT -gt 0 ]; then
    echo -e "  Failed:     ${RED}${FAIL_COUNT}${NC}"
fi
echo ""
check_models

echo ""
echo "Next steps:"
echo "  1. Start ComfyUI container:"
echo "     cd /opt/comfyui-pipeline"
echo "     docker-compose -f docker-compose.rocky.yaml up -d"
echo ""
echo "  2. Watch startup logs:"
echo "     docker-compose -f docker-compose.rocky.yaml logs -f"
echo ""
echo "  3. Test lifecycle:"
echo "     python test_lifecycle.py --url http://localhost:8189 --profile sdxl"
echo ""
echo "  4. Update gateway .env:"
echo "     COMFYUI_URL=http://host.docker.internal:8188"
echo "     COMFYUI_WRAPPER_URL=http://host.docker.internal:8189"
echo ""
