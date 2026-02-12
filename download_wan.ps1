<#
.SYNOPSIS
    Download Wan 2.1 T2V model for text-to-video generation.
.DESCRIPTION
    Downloads Wan2_1-T2V-14B_fp8_scaled (~14.5 GB) to diffusion_models/
.EXAMPLE
    .\download_wan_t2v.ps1
#>

param(
    [string]$ModelsDir = "H:\dev\AI_dev\models"
)

$dest = "$ModelsDir\diffusion_models\Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors"

if (Test-Path $dest) {
    $size = [math]::Round((Get-Item $dest).Length / 1GB, 1)
    Write-Host "Already exists: $(Split-Path $dest -Leaf) (${size}GB)" -ForegroundColor Green
    exit 0
}

$url = "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/T2V/Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors"

Write-Host "Downloading Wan 2.1 T2V model (~14.5 GB)..." -ForegroundColor Yellow
Write-Host "  From: Kijai/WanVideo_comfy_fp8_scaled"
Write-Host "  To:   $dest"
Write-Host ""

$dir = Split-Path $dest -Parent
if (!(Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }

$tmp = "$dest.tmp"
$ProgressPreference = 'SilentlyContinue'

try {
    Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing
    Move-Item $tmp $dest -Force
    $size = [math]::Round((Get-Item $dest).Length / 1GB, 1)
    Write-Host "Downloaded: $(Split-Path $dest -Leaf) (${size}GB)" -ForegroundColor Green
} catch {
    Write-Host "FAILED: $($_.Exception.Message)" -ForegroundColor Red
    if (Test-Path $tmp) { Remove-Item $tmp -Force }
}