# =============================================================================
# ComfyUI Model Downloader (PowerShell)
# =============================================================================
#
# Downloads models for: SDXL, Flux Dev (fp8), Wan 2.2, ESRGAN
#
# Usage:
#   .\download_models.ps1                    # Download everything
#   .\download_models.ps1 -Profile sdxl      # Just SDXL
#   .\download_models.ps1 -Profile flux      # Just Flux fp8
#   .\download_models.ps1 -Profile wan       # Just Wan 2.2
#   .\download_models.ps1 -Profile esrgan    # Just ESRGAN
#   .\download_models.ps1 -Check             # Show what's missing
#
# Total: ~45 GB  (SDXL ~7GB, Flux ~17GB, Wan ~20GB, ESRGAN ~0.2GB)
# =============================================================================

param(
    [string]$Profile = "all",
    [string]$ModelsDir = "H:\dev\AI_dev\models",
    [string]$HfToken = $env:HF_TOKEN,
    [switch]$Check
)

$ErrorActionPreference = "Continue"

# =============================================================================
# Helpers
# =============================================================================

function Download-HF {
    param(
        [string]$Repo,
        [string]$Filename,
        [string]$Dest,
        [string]$Subfolder = ""
    )

    if (Test-Path $Dest) {
        $size = [math]::Round((Get-Item $Dest).Length / 1MB)
        Write-Host "  [OK] Already exists: $(Split-Path $Dest -Leaf) (${size}MB)" -ForegroundColor Green
        return
    }

    $dir = Split-Path $Dest -Parent
    if (!(Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }

    $url = "https://huggingface.co/$Repo/resolve/main"
    if ($Subfolder) { $url = "$url/$Subfolder/$Filename" }
    else { $url = "$url/$Filename" }

    Write-Host "  [DL] $Repo / $Filename" -ForegroundColor Yellow
    Write-Host "       -> $Dest" -ForegroundColor DarkGray

    $headers = @{}
    if ($HfToken) { $headers["Authorization"] = "Bearer $HfToken" }

    try {
        $tmp = "$Dest.tmp"
        $ProgressPreference = 'SilentlyContinue'  # Massively speeds up downloads
        Invoke-WebRequest -Uri $url -OutFile $tmp -Headers $headers -UseBasicParsing
        Move-Item $tmp $Dest -Force
        $size = [math]::Round((Get-Item $Dest).Length / 1MB)
        Write-Host "  [OK] Downloaded: $(Split-Path $Dest -Leaf) (${size}MB)" -ForegroundColor Green
    } catch {
        Write-Host "  [!!] FAILED: $($_.Exception.Message)" -ForegroundColor Red
        if (Test-Path $tmp) { Remove-Item $tmp -Force }

        # If file is large, suggest alternative
        if ($_.Exception.Message -match "401") {
            Write-Host "       Gated repo - set `$env:HF_TOKEN or download manually" -ForegroundColor Yellow
        }
    }
}

function Download-Url {
    param(
        [string]$Url,
        [string]$Dest
    )

    if (Test-Path $Dest) {
        $size = [math]::Round((Get-Item $Dest).Length / 1MB)
        Write-Host "  [OK] Already exists: $(Split-Path $Dest -Leaf) (${size}MB)" -ForegroundColor Green
        return
    }

    $dir = Split-Path $Dest -Parent
    if (!(Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }

    Write-Host "  [DL] $(Split-Path $Dest -Leaf)" -ForegroundColor Yellow

    try {
        $tmp = "$Dest.tmp"
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $Url -OutFile $tmp -UseBasicParsing
        Move-Item $tmp $Dest -Force
        $size = [math]::Round((Get-Item $Dest).Length / 1MB)
        Write-Host "  [OK] Downloaded: $(Split-Path $Dest -Leaf) (${size}MB)" -ForegroundColor Green
    } catch {
        Write-Host "  [!!] FAILED: $($_.Exception.Message)" -ForegroundColor Red
        if (Test-Path $tmp) { Remove-Item $tmp -Force }
    }
}

function Check-File {
    param([string]$Path, [string]$Desc)
    if (Test-Path $Path) {
        $size = [math]::Round((Get-Item $Path).Length / 1MB)
        Write-Host "  [OK] $Desc (${size}MB)" -ForegroundColor Green
    } else {
        Write-Host "  [--] $Desc - MISSING" -ForegroundColor Red
    }
}

# =============================================================================
# Profiles
# =============================================================================

function Download-SDXL {
    Write-Host "`n=== SDXL Profile (~7 GB) ===" -ForegroundColor Cyan

    Download-HF -Repo "stabilityai/stable-diffusion-xl-base-1.0" `
        -Filename "sd_xl_base_1.0.safetensors" `
        -Dest "$ModelsDir\checkpoints\sd_xl_base_1.0.safetensors"

    Download-HF -Repo "stabilityai/sdxl-vae" `
        -Filename "sdxl_vae.safetensors" `
        -Dest "$ModelsDir\vae\sdxl_vae.safetensors"
}

function Download-Flux {
    Write-Host "`n=== Flux Dev fp8 Profile (~17 GB) ===" -ForegroundColor Cyan

    # Flux UNET fp8 (~12 GB)
    Download-HF -Repo "Comfy-Org/flux1-dev" `
        -Filename "flux1-dev-fp8.safetensors" `
        -Dest "$ModelsDir\diffusion_models\flux1-dev-fp8.safetensors"

    # T5-XXL fp8 (~5 GB)
    Download-HF -Repo "comfyanonymous/flux_text_encoders" `
        -Filename "t5xxl_fp8_e4m3fn.safetensors" `
        -Dest "$ModelsDir\clip\t5xxl_fp8_e4m3fn.safetensors"

    # CLIP-L (~246 MB)
    Download-HF -Repo "comfyanonymous/flux_text_encoders" `
        -Filename "clip_l.safetensors" `
        -Dest "$ModelsDir\clip\clip_l.safetensors"

    # Flux VAE (~335 MB)
    # You already have this at flux-dev\ae.safetensors - copy if download fails
    if (Test-Path "$ModelsDir\vae\ae.safetensors") {
        $size = [math]::Round((Get-Item "$ModelsDir\vae\ae.safetensors").Length / 1MB)
        Write-Host "  [OK] Already exists: ae.safetensors (${size}MB)" -ForegroundColor Green
    } elseif (Test-Path "$ModelsDir\flux-dev\ae.safetensors") {
        Write-Host "  [CP] Copying ae.safetensors from flux-dev\" -ForegroundColor Yellow
        Copy-Item "$ModelsDir\flux-dev\ae.safetensors" "$ModelsDir\vae\ae.safetensors"
        Write-Host "  [OK] Copied ae.safetensors to vae\" -ForegroundColor Green
    } else {
        Download-HF -Repo "black-forest-labs/FLUX.1-dev" `
            -Filename "ae.safetensors" `
            -Dest "$ModelsDir\vae\ae.safetensors"
    }
}

function Download-Wan {
    Write-Host "`n=== Wan 2.2 Animate Profile (~20 GB) ===" -ForegroundColor Cyan

    # Wan 2.2 Animate 14B fp8 (~14 GB)
    Download-HF -Repo "Kijai/WanVideo_comfy" `
        -Filename "Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors" `
        -Dest "$ModelsDir\diffusion_models\Wan22Animate\Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors"

    # Wan 2.1 VAE
    Download-HF -Repo "Kijai/WanVideo_comfy" `
        -Filename "wan_2.1_vae.safetensors" `
        -Dest "$ModelsDir\vae\wan_2.1_vae.safetensors"

    # UMT5-XXL encoder
    Download-HF -Repo "Kijai/WanVideo_comfy" `
        -Filename "umt5-xxl-enc-bf16.safetensors" `
        -Dest "$ModelsDir\clip\umt5-xxl-enc-bf16.safetensors"

    # CLIP Vision H
    Download-HF -Repo "Kijai/WanVideo_comfy" `
        -Filename "clip_vision_h.safetensors" `
        -Dest "$ModelsDir\clip_vision\clip_vision_h.safetensors"

    # LoRAs
    Download-HF -Repo "Kijai/WanVideo_comfy" `
        -Filename "WanAnimate_relight_lora_fp16.safetensors" `
        -Dest "$ModelsDir\loras\WanAnimate_relight_lora_fp16.safetensors"

    Download-HF -Repo "Kijai/WanVideo_comfy" `
        -Filename "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" `
        -Dest "$ModelsDir\loras\lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"

    # SAM2
    Download-HF -Repo "Kijai/sam2-safetensors" `
        -Filename "sam2.1_hiera_large.safetensors" `
        -Dest "$ModelsDir\sam2\sam2.1_hiera_large.safetensors"
}

function Download-ESRGAN {
    Write-Host "`n=== ESRGAN Profile (~0.2 GB) ===" -ForegroundColor Cyan

    Download-Url -Url "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" `
        -Dest "$ModelsDir\upscale_models\RealESRGAN_x4plus.pth"

    Download-Url -Url "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth" `
        -Dest "$ModelsDir\upscale_models\RealESRGAN_x4plus_anime_6B.pth"

    Download-Url -Url "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth" `
        -Dest "$ModelsDir\upscale_models\4x-UltraSharp.pth"
}

function Show-Check {
    Write-Host "`n=== Model Status ===" -ForegroundColor Cyan
    Write-Host "  Models dir: $ModelsDir`n"

    Write-Host "  SDXL:" -ForegroundColor White
    Check-File "$ModelsDir\checkpoints\sd_xl_base_1.0.safetensors" "SDXL Base 1.0"
    Check-File "$ModelsDir\vae\sdxl_vae.safetensors" "SDXL VAE"

    Write-Host "`n  Flux Dev fp8:" -ForegroundColor White
    Check-File "$ModelsDir\diffusion_models\flux1-dev-fp8.safetensors" "Flux UNET fp8"
    Check-File "$ModelsDir\clip\t5xxl_fp8_e4m3fn.safetensors" "T5-XXL fp8"
    Check-File "$ModelsDir\clip\clip_l.safetensors" "CLIP-L"
    Check-File "$ModelsDir\vae\ae.safetensors" "Flux VAE (ae)"

    Write-Host "`n  Wan 2.2 Animate:" -ForegroundColor White
    Check-File "$ModelsDir\diffusion_models\Wan22Animate\Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors" "Wan 2.2 14B fp8"
    Check-File "$ModelsDir\vae\wan_2.1_vae.safetensors" "Wan VAE"
    Check-File "$ModelsDir\clip\umt5-xxl-enc-bf16.safetensors" "UMT5-XXL"
    Check-File "$ModelsDir\clip_vision\clip_vision_h.safetensors" "CLIP Vision H"
    Check-File "$ModelsDir\loras\WanAnimate_relight_lora_fp16.safetensors" "Relight LoRA"
    Check-File "$ModelsDir\loras\lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" "LightX2V LoRA"
    Check-File "$ModelsDir\sam2\sam2.1_hiera_large.safetensors" "SAM2 Large"

    Write-Host "`n  ESRGAN:" -ForegroundColor White
    Check-File "$ModelsDir\upscale_models\RealESRGAN_x4plus.pth" "RealESRGAN x4plus"
    Check-File "$ModelsDir\upscale_models\RealESRGAN_x4plus_anime_6B.pth" "RealESRGAN Anime"
    Check-File "$ModelsDir\upscale_models\4x-UltraSharp.pth" "4x-UltraSharp"
}

# =============================================================================
# Main
# =============================================================================

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  ComfyUI Model Downloader" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Models dir: $ModelsDir"
Write-Host "  Profile:    $Profile"
if ($HfToken) {
    Write-Host "  HF Token:   set ($($HfToken.Length) chars)"
} else {
    Write-Host "  HF Token:   not set (may need for gated repos)"
}
Write-Host "==========================================" -ForegroundColor Cyan

if ($Check) {
    Show-Check
    exit 0
}

# Create directories
$dirs = @("checkpoints","diffusion_models","vae","clip","clip_vision","loras",
          "controlnet","upscale_models","sam2","ipadapter")
foreach ($d in $dirs) {
    $p = Join-Path $ModelsDir $d
    if (!(Test-Path $p)) { New-Item -ItemType Directory -Path $p -Force | Out-Null }
}

switch ($Profile) {
    "sdxl"    { Download-SDXL }
    "flux"    { Download-Flux }
    "flux-dev"{ Download-Flux }
    "wan"     { Download-Wan }
    "wan-video" { Download-Wan }
    "esrgan"  { Download-ESRGAN }
    "upscale" { Download-ESRGAN }
    "all" {
        Download-SDXL
        Download-Flux
        Download-Wan
        Download-ESRGAN
    }
    default {
        Write-Host "Unknown profile: $Profile" -ForegroundColor Red
        Write-Host "Available: sdxl, flux, wan, esrgan, all"
        exit 1
    }
}

Write-Host ""
Show-Check

Write-Host "`nDone! Rebuild container to pick up new models:" -ForegroundColor Green
Write-Host "  docker-compose build && docker-compose up -d"