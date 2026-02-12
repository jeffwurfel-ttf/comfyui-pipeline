# Download Models for TTF Character Swap Workflow
# 
# Usage:
#   .\download_models.ps1                    # Download all required models
#   .\download_models.ps1 -ModelsPath D:\models  # Custom models path
#   .\download_models.ps1 -SkipLarge         # Skip large models (>10GB)
#
# Requirements:
#   - huggingface-cli (pip install huggingface_hub)
#   - Or manual download from URLs below

param(
    [string]$ModelsPath = "H:\models",
    [switch]$SkipLarge,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

Write-Host "TTF Character Swap - Model Downloader" -ForegroundColor Cyan
Write-Host "=" * 50
Write-Host "Models path: $ModelsPath"
Write-Host ""

# Create directories
$dirs = @(
    "$ModelsPath\wan",
    "$ModelsPath\clip",
    "$ModelsPath\vae",
    "$ModelsPath\loras",
    "$ModelsPath\sam2",
    "$ModelsPath\dwpose",
    "$ModelsPath\ultralytics"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        Write-Host "Creating: $dir"
        if (-not $DryRun) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
}

# Model definitions
$models = @(
    @{
        Name = "Wan 2.1 I2V Model (FP8 - smaller)"
        Url = "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2.1-I2V-14B-480P_fp8_e4m3fn.safetensors"
        Path = "$ModelsPath\wan\Wan2.1-I2V-14B-480P_fp8_e4m3fn.safetensors"
        Size = "14GB"
        Required = $true
        Large = $true
    },
    @{
        Name = "Wan VAE"
        Url = "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/wan_2.1_vae.safetensors"
        Path = "$ModelsPath\vae\wan_2.1_vae.safetensors"
        Size = "330MB"
        Required = $true
        Large = $false
    },
    @{
        Name = "Wan CLIP (UMT5-XXL FP8)"
        Url = "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
        Path = "$ModelsPath\clip\umt5_xxl_fp8_e4m3fn_scaled.safetensors"
        Size = "4.9GB"
        Required = $true
        Large = $false
    },
    @{
        Name = "SAM2 Hiera Large"
        Url = "https://huggingface.co/Kijai/sam2-safetensors/resolve/main/sam2_hiera_large.safetensors"
        Path = "$ModelsPath\sam2\sam2_hiera_large.safetensors"
        Size = "900MB"
        Required = $true
        Large = $false
    },
    @{
        Name = "DWPose Model"
        Url = "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx"
        Path = "$ModelsPath\dwpose\dw-ll_ucoco_384.onnx"
        Size = "300MB"
        Required = $true
        Large = $false
    },
    @{
        Name = "YOLO Detection (for DWPose)"
        Url = "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx"
        Path = "$ModelsPath\dwpose\yolox_l.onnx"
        Size = "200MB"
        Required = $true
        Large = $false
    },
    @{
        Name = "Relight LoRA"
        Url = "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/WanAnimate_relight_lora_fp16.safetensors"
        Path = "$ModelsPath\loras\WanAnimate_relight_lora_fp16.safetensors"
        Size = "200MB"
        Required = $false
        Large = $false
    },
    @{
        Name = "LightX2V LoRA (CFG Distill)"
        Url = "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
        Path = "$ModelsPath\loras\lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
        Size = "400MB"
        Required = $false
        Large = $false
    }
)

Write-Host ""
Write-Host "Models to download:" -ForegroundColor Yellow
Write-Host "-" * 50

$toDownload = @()
foreach ($model in $models) {
    $exists = Test-Path $model.Path
    $skip = $SkipLarge -and $model.Large
    
    if ($exists) {
        Write-Host "  [EXISTS] $($model.Name)" -ForegroundColor Green
    } elseif ($skip) {
        Write-Host "  [SKIP]   $($model.Name) ($($model.Size) - use -SkipLarge:$false to include)" -ForegroundColor Yellow
    } else {
        $req = if ($model.Required) { "REQUIRED" } else { "optional" }
        Write-Host "  [NEED]   $($model.Name) ($($model.Size), $req)" -ForegroundColor $(if ($model.Required) { "Red" } else { "Gray" })
        $toDownload += $model
    }
}

if ($toDownload.Count -eq 0) {
    Write-Host ""
    Write-Host "All models already downloaded!" -ForegroundColor Green
    exit 0
}

Write-Host ""
Write-Host "Downloading $($toDownload.Count) models..." -ForegroundColor Cyan
Write-Host ""

if ($DryRun) {
    Write-Host "[DRY RUN] Would download:"
    foreach ($model in $toDownload) {
        Write-Host "  $($model.Url)"
        Write-Host "  -> $($model.Path)"
    }
    exit 0
}

# Check for curl or wget
$downloader = $null
if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
    $downloader = "curl"
} elseif (Get-Command wget -ErrorAction SilentlyContinue) {
    $downloader = "wget"
} else {
    Write-Host "No downloader found. Using Invoke-WebRequest (slower)..." -ForegroundColor Yellow
    $downloader = "iwr"
}

$success = 0
$failed = 0

foreach ($model in $toDownload) {
    Write-Host "Downloading: $($model.Name) ($($model.Size))..." -ForegroundColor Cyan
    
    try {
        switch ($downloader) {
            "curl" {
                & curl.exe -L -o $model.Path $model.Url --progress-bar
            }
            "wget" {
                & wget -O $model.Path $model.Url
            }
            "iwr" {
                Invoke-WebRequest -Uri $model.Url -OutFile $model.Path
            }
        }
        
        if (Test-Path $model.Path) {
            $fileSize = (Get-Item $model.Path).Length / 1MB
            Write-Host "  Done: $([math]::Round($fileSize, 1)) MB" -ForegroundColor Green
            $success++
        } else {
            throw "File not created"
        }
    } catch {
        Write-Host "  FAILED: $_" -ForegroundColor Red
        $failed++
    }
    
    Write-Host ""
}

Write-Host "=" * 50
Write-Host "Download complete!" -ForegroundColor Cyan
Write-Host "  Success: $success"
Write-Host "  Failed:  $failed"

if ($failed -gt 0) {
    Write-Host ""
    Write-Host "Some downloads failed. Try manually from:" -ForegroundColor Yellow
    Write-Host "  https://huggingface.co/Kijai/WanVideo_comfy"
    Write-Host "  https://huggingface.co/Kijai/sam2-safetensors"
    Write-Host "  https://huggingface.co/yzd-v/DWPose"
}