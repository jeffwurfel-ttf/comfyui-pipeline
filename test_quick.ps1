# ComfyUI Lifecycle Quick Test (PowerShell)
#
# Usage:
#   .\test_quick.ps1                      # Test SDXL
#   .\test_quick.ps1 -Profile "flux-dev"  # Test Flux
#   .\test_quick.ps1 -Url "http://10.0.1.5:8189"
#
# For the full test suite, use: python test_lifecycle.py

param(
    [string]$Url = "http://localhost:8189",
    [string]$Profile = "sdxl"
)

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  ComfyUI Lifecycle Quick Test" -ForegroundColor Cyan
Write-Host "  Target:  $Url" -ForegroundColor Cyan
Write-Host "  Profile: $Profile" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Health check
Write-Host "`n[1] Health check..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "$Url/health" -TimeoutSec 10
    Write-Host "  Status:  $($health.status)" -ForegroundColor Green
    Write-Host "  GPU:     $($health.gpu_name)"
    Write-Host "  VRAM:    $($health.vram_used_mb)MB / $($health.vram_total_mb)MB"
    Write-Host "  Model:   $($health.model_name ?? 'none')"
    Write-Host "  ComfyUI: $($health.comfyui)"
} catch {
    Write-Host "  FAILED: Cannot connect to $Url" -ForegroundColor Red
    Write-Host "  Is the container running? docker-compose up -d" -ForegroundColor Red
    exit 1
}

# List profiles
Write-Host "`n[2] Available profiles..." -ForegroundColor Yellow
$profiles = Invoke-RestMethod -Uri "$Url/model/profiles" -TimeoutSec 10
foreach ($p in $profiles.profiles) {
    $marker = if ($p.name -eq $Profile) { " <-- testing" } else { "" }
    Write-Host "  $($p.name): $($p.description) (~$($p.estimated_vram_mb)MB)$marker"
}

# Load model
Write-Host "`n[3] Loading '$Profile'..." -ForegroundColor Yellow
Write-Host "  (this may take 30-120 seconds)" -ForegroundColor DarkGray
$sw = [System.Diagnostics.Stopwatch]::StartNew()
try {
    $load = Invoke-RestMethod -Uri "$Url/model/load" -Method Post `
        -ContentType "application/json" `
        -Body (@{model=$Profile} | ConvertTo-Json) `
        -TimeoutSec 360
    $sw.Stop()
    Write-Host "  Success:   $($load.success)" -ForegroundColor Green
    Write-Host "  VRAM used: $($load.vram_used_mb) MB"
    Write-Host "  Load time: $($load.load_time_ms) ms"
    Write-Host "  Wall time: $($sw.Elapsed.TotalSeconds.ToString('F1'))s"
} catch {
    $sw.Stop()
    Write-Host "  FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "  Check logs: docker-compose logs -f comfyui" -ForegroundColor Red
    exit 1
}

# Status
Write-Host "`n[4] Model status..." -ForegroundColor Yellow
$status = Invoke-RestMethod -Uri "$Url/model/status" -TimeoutSec 10
Write-Host "  State:   $($status.state)"
Write-Host "  Profile: $($status.profile)"
Write-Host "  VRAM:    $($status.vram_used_mb)MB used / $($status.vram_available_mb)MB free"

# Generate
Write-Host "`n[5] Generating test image (512x512, 8 steps)..." -ForegroundColor Yellow
$sw = [System.Diagnostics.Stopwatch]::StartNew()
try {
    $gen = Invoke-RestMethod -Uri "$Url/generate" -Method Post `
        -ContentType "application/json" `
        -Body (@{
            prompt = "a red cube on white background, 3d render"
            negative_prompt = "blurry, text"
            width = 512
            height = 512
            num_steps = 8
            seed = 42
        } | ConvertTo-Json) `
        -TimeoutSec 300
    $sw.Stop()

    if ($gen.success) {
        Write-Host "  Success!" -ForegroundColor Green
        Write-Host "  Latency: $($gen.latency_ms) ms"
        Write-Host "  Seed:    $($gen.images[0].seed)"

        # Save image
        $bytes = [Convert]::FromBase64String($gen.images[0].base64)
        $outPath = Join-Path $PWD "test_output.png"
        [IO.File]::WriteAllBytes($outPath, $bytes)
        Write-Host "  Saved:   $outPath ($($bytes.Length) bytes)" -ForegroundColor Green
    } else {
        Write-Host "  Generation failed: $($gen.error)" -ForegroundColor Red
    }
} catch {
    $sw.Stop()
    Write-Host "  FAILED: $($_.Exception.Message)" -ForegroundColor Red
}

# Unload
Write-Host "`n[6] Unloading model..." -ForegroundColor Yellow
$unload = Invoke-RestMethod -Uri "$Url/model/unload" -Method Post -TimeoutSec 30
Write-Host "  Success:    $($unload.success)" -ForegroundColor Green
Write-Host "  VRAM freed: $($unload.vram_freed_mb) MB"

# Verify unloaded
Write-Host "`n[7] Verifying unloaded state..." -ForegroundColor Yellow
$status = Invoke-RestMethod -Uri "$Url/model/status" -TimeoutSec 10
if ($status.state -eq "unloaded") {
    Write-Host "  State: unloaded" -ForegroundColor Green
} else {
    Write-Host "  State: $($status.state) (expected: unloaded)" -ForegroundColor Red
}

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  LIFECYCLE TEST COMPLETE" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan