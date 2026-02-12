#!/bin/bash
# =============================================================================
# ComfyUI Model Downloader for TTF GPU Nodes
# =============================================================================
#
# Downloads all models needed for: SDXL, Flux Dev (fp8), Wan 2.2, ESRGAN
#
# Usage:
#   chmod +x download_models.sh
#   ./download_models.sh                    # Download everything
#   ./download_models.sh --profile sdxl     # Just SDXL models
#   ./download_models.sh --profile flux     # Just Flux models
#   ./download_models.sh --profile wan      # Just Wan 2.2 models
#   ./download_models.sh --profile esrgan   # Just ESRGAN models
#   ./download_models.sh --check            # Show what's missing
#
# Models directory: /data/comfyui/models (configurable via MODELS_DIR env)
#
# Some models (Flux VAE, SDXL) are on gated HuggingFace repos.
# Set HF_TOKEN if you get 401 errors:
#   export HF_TOKEN="hf_your_token_here"
#
# Total download size (all profiles): ~45 GB
#   SDXL:    ~7 GB
#   Flux:   ~17 GB
#   Wan 2.2: ~20 GB
#   ESRGAN:  ~0.2 GB
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

download_hf() {
    # download_hf <repo> <filename> <dest_path> [subfolder]
    local repo="$1"
    local filename="$2"
    local dest="$3"
    local subfolder="${4:-}"

    if [ -f "$dest" ]; then
        local size_mb=$(du -m "$dest" | cut -f1)
        log "Already exists: $(basename $dest) (${size_mb}MB)"
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
        local curl_auth=""
        if [ -n "$HF_TOKEN" ]; then
            curl_auth="-H \"Authorization: Bearer ${HF_TOKEN}\""
        fi
        curl -L -C - --progress-bar \
            ${curl_auth} \
            -o "${dest}.tmp" \
            "$url" && mv "${dest}.tmp" "$dest"
    else
        err "Neither wget nor curl found. Install one: dnf install wget"
        return 1
    fi

    if [ -f "$dest" ]; then
        local size_mb=$(du -m "$dest" | cut -f1)
        log "Downloaded: $(basename $dest) (${size_mb}MB)"
    else
        err "Download failed: $dest"
        return 1
    fi
}

download_url() {
    # download_url <url> <dest_path>
    local url="$1"
    local dest="$2"

    if [ -f "$dest" ]; then
        local size_mb=$(du -m "$dest" | cut -f1)
        log "Already exists: $(basename $dest) (${size_mb}MB)"
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
    else
        err "Download failed: $dest"
        return 1
    fi
}

check_file() {
    # check_file <path> <description>
    local path="$1"
    local desc="$2"
    if [ -f "$path" ]; then
        local size_mb=$(du -m "$path" | cut -f1)
        echo -e "  ${GREEN}✓${NC} ${desc} (${size_mb}MB)"
    else
        echo -e "  ${RED}✗${NC} ${desc} - MISSING"
        echo -e "      ${path}"
    fi
}

# =============================================================================
# Profile Downloads
# =============================================================================

download_sdxl() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  SDXL Profile (~7 GB)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # SDXL Base checkpoint (6.94 GB)
    download_hf \
        "stabilityai/stable-diffusion-xl-base-1.0" \
        "sd_xl_base_1.0.safetensors" \
        "${MODELS_DIR}/checkpoints/sd_xl_base_1.0.safetensors"

    # SDXL VAE (optional - checkpoint includes VAE, but explicit is better)
    download_hf \
        "stabilityai/sdxl-vae" \
        "sdxl_vae.safetensors" \
        "${MODELS_DIR}/vae/sdxl_vae.safetensors"
}

download_flux() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Flux Dev fp8 Profile (~17 GB)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Flux UNET - fp8 quantized (~11.9 GB)
    # This is the fp8 version that fits on 24GB cards
    download_hf \
        "Comfy-Org/flux1-dev" \
        "flux1-dev-fp8.safetensors" \
        "${MODELS_DIR}/diffusion_models/flux1-dev-fp8.safetensors"

    # T5-XXL text encoder - fp8 (~4.89 GB)
    download_hf \
        "comfyanonymous/flux_text_encoders" \
        "t5xxl_fp8_e4m3fn.safetensors" \
        "${MODELS_DIR}/clip/t5xxl_fp8_e4m3fn.safetensors"

    # CLIP-L text encoder (~246 MB)
    download_hf \
        "comfyanonymous/flux_text_encoders" \
        "clip_l.safetensors" \
        "${MODELS_DIR}/clip/clip_l.safetensors"

    # Flux VAE - ae.safetensors (~335 MB)
    # NOTE: This is from the gated FLUX.1-dev repo. If you get a 401, either:
    #   1. Set HF_TOKEN and accept the license at https://huggingface.co/black-forest-labs/FLUX.1-dev
    #   2. Or copy ae.safetensors from your Windows box (H:\dev\AI_dev\models\flux-dev\ae.safetensors)
    download_hf \
        "black-forest-labs/FLUX.1-dev" \
        "ae.safetensors" \
        "${MODELS_DIR}/vae/ae.safetensors"
}

download_wan() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Wan 2.2 Animate Profile (~20 GB)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Wan 2.2 Animate model - fp8 (~14 GB)
    mkdir -p "${MODELS_DIR}/diffusion_models/Wan22Animate"
    download_hf \
        "Kijai/WanVideo_comfy" \
        "Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors" \
        "${MODELS_DIR}/diffusion_models/Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors"

    # Wan 2.1 VAE (~335 MB)
    download_hf \
        "Kijai/WanVideo_comfy" \
        "wan_2.1_vae.safetensors" \
        "${MODELS_DIR}/vae/wan_2.1_vae.safetensors"

    # UMT5-XXL text encoder - bf16 (~5 GB)
    download_hf \
        "Kijai/WanVideo_comfy" \
        "umt5-xxl-enc-bf16.safetensors" \
        "${MODELS_DIR}/clip/umt5-xxl-enc-bf16.safetensors"

    # CLIP Vision H (~3.9 GB)
    # Used by WanVideoClipVisionEncode for reference image conditioning
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

    # SAM2 - for segmentation/masking (~900 MB)
    download_hf \
        "Kijai/sam2-safetensors" \
        "sam2.1_hiera_large.safetensors" \
        "${MODELS_DIR}/sam2/sam2.1_hiera_large.safetensors"

    # Pose detection models (for WanAnimatePreprocess)
    # These are ONNX models used by the OnnxDetectionModelLoader node
    # They go in the custom_nodes data directory, but ComfyUI-WanAnimatePreprocess
    # auto-downloads them on first use. We pre-download for reliability.
    info "Note: Pose detection ONNX models (vitpose, yolo) are auto-downloaded"
    info "      by ComfyUI-WanAnimatePreprocess on first use."
}

download_esrgan() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  ESRGAN Profile (~0.2 GB)${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # RealESRGAN x4plus (64 MB)
    download_url \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
        "${MODELS_DIR}/upscale_models/RealESRGAN_x4plus.pth"

    # RealESRGAN x4plus Anime (17 MB)
    download_url \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth" \
        "${MODELS_DIR}/upscale_models/RealESRGAN_x4plus_anime_6B.pth"

    # 4x-UltraSharp (64 MB)
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
    echo -e "${BLUE}  Model Status Check${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  Models dir: ${MODELS_DIR}"
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

    echo "  Wan 2.2 Animate Profile:"
    check_file "${MODELS_DIR}/diffusion_models/Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors" "Wan 2.2 Animate 14B fp8"
    check_file "${MODELS_DIR}/vae/wan_2.1_vae.safetensors" "Wan 2.1 VAE"
    check_file "${MODELS_DIR}/clip/umt5-xxl-enc-bf16.safetensors" "UMT5-XXL encoder"
    check_file "${MODELS_DIR}/clip_vision/clip_vision_h.safetensors" "CLIP Vision H"
    check_file "${MODELS_DIR}/loras/WanAnimate_relight_lora_fp16.safetensors" "Relight LoRA"
    check_file "${MODELS_DIR}/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" "LightX2V LoRA"
    check_file "${MODELS_DIR}/sam2/sam2.1_hiera_large.safetensors" "SAM2 Large"
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
echo "  ComfyUI Model Downloader"
echo "=========================================="
echo "  Models dir: ${MODELS_DIR}"
echo "  Profile:    ${PROFILE}"
if [ -n "$HF_TOKEN" ]; then
    echo "  HF Token:   set (${#HF_TOKEN} chars)"
else
    echo "  HF Token:   not set (may need for gated repos)"
fi
echo "=========================================="

# Check mode
if [ "$CHECK_ONLY" = true ]; then
    check_models
    exit 0
fi

# Create base directory structure
log "Creating directory structure..."
mkdir -p "${MODELS_DIR}"/{checkpoints,diffusion_models,vae,clip,clip_vision,loras,controlnet,upscale_models,sam2,ipadapter}

# Download based on profile
case "$PROFILE" in
    sdxl)
        download_sdxl
        ;;
    flux|flux-dev)
        download_flux
        ;;
    wan|wan-video|wan22)
        download_wan
        ;;
    esrgan|upscale)
        download_esrgan
        ;;
    all)
        download_sdxl
        download_flux
        download_wan
        download_esrgan
        ;;
    *)
        err "Unknown profile: $PROFILE"
        echo "  Available: sdxl, flux, wan, esrgan, all"
        exit 1
        ;;
esac

# Final check
echo ""
echo "=========================================="
echo "  Download Complete"
echo "=========================================="
check_models

echo ""
echo "Next steps:"
echo "  1. Update docker-compose.yml volume mount:"
echo "     volumes:"
echo "       - ${MODELS_DIR}:/models"
echo ""
echo "  2. Start ComfyUI:"
echo "     docker-compose up -d"
echo ""
echo "  3. Test lifecycle:"
echo "     python test_lifecycle.py --profile sdxl"
echo ""