# ComfyUI Pipeline v1 — Server Audit & Architecture Plan

## Executive Summary

After 17 sessions of iterative development, the comfyui-pipeline repo has accumulated significant cruft: 5 overlapping download scripts, a stale model manifest, a 14GB orphaned model file in the repo root, stale HuggingFace cache directories, and no single-command way to validate that a workflow JSON has everything it needs to run. This document audits the current state, proposes a clean v1 architecture, and provides the concrete cleanup steps.

---

## 1. Current State Audit

### 1.1 Model Directory (`/mnt/ssd/comfyui-models`)

**Actual models on disk (the real inventory):**

| Directory | File | Size (approx) | Used By |
|-----------|------|---------------|---------|
| checkpoints/ | sd_xl_base_1.0.safetensors | 6.9 GB | SDXL txt2img |
| clip/ | clip_l.safetensors | 246 MB | Flux |
| clip/ | t5xxl_fp8_e4m3fn.safetensors | 4.9 GB | Flux |
| clip/ | umt5-xxl-enc-bf16.safetensors | 5.0 GB | All Wan workflows |
| clip_vision/ | clip_vision_h.safetensors | 3.9 GB | Wan I2V |
| diffusion_models/ | flux1-dev-fp8.safetensors | 11.9 GB | Flux txt2img |
| diffusion_models/ | Wan2_1-T2V-14B_fp8_e4m3fn.safetensors | 14.9 GB | Wan T2V |
| diffusion_models/ | Wan2_1-I2V-14B-720P_fp8_e4m3fn.safetensors | ~14 GB | Wan I2V |
| diffusion_models/ | Wan2_1-FLF2V-14B-720P_fp8_e4m3fn.safetensors | ~14 GB | Wan FLF2V (first/last frame) |
| diffusion_models/ | wan2.2_fun_inpaint_5B_bf16.safetensors | ~10 GB | Wan Fun Inpaint (5B) |
| diffusion_models/ | wan2.2_fun_inpaint_low_noise_14B_fp8_scaled.safetensors | ~14 GB | Wan Fun Inpaint (14B) |
| diffusion_models/ | Wan2_2-I2V-A14B-LOW_bf16.safetensors | ~28 GB | Wan 2.2 Animate (char swap) |
| loras/ | wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors | ~400 MB | Wan speed LoRA |
| sam2/ | sam2.1_hiera_large.safetensors | 900 MB | SAM2 segmentation |
| sam3d/hf/checkpoints/ | (11 files: slat_*, ss_*, pipeline.yaml) | ~15 GB | SAM3D mesh gen |
| sams/ | sam_vit_b_01ec64.pth | ~375 MB | Legacy SAM |
| text_encoders/ | umt5_xxl_fp8_e4m3fn_scaled.safetensors | ~5 GB | Alternate T5 (unused?) |
| upscale_models/ | RealESRGAN_x4plus.pth | 64 MB | ESRGAN |
| upscale_models/ | RealESRGAN_x4plus_anime_6B.pth | 17 MB | ESRGAN anime |
| upscale_models/ | 4x-UltraSharp.pth | 64 MB | ESRGAN sharp |
| vae/ | ae.safetensors | 335 MB | Flux |
| vae/ | sdxl_vae.safetensors | 320 MB | SDXL |
| vae/ | wan_2.1_vae.safetensors | 335 MB | Wan 2.1 |
| vae/ | wan2.2_vae.safetensors | ~335 MB | Wan 2.2 |

**Problems found:**

1. **`_hf_staging/`** — HuggingFace download cache with .lock/.metadata files. Wasted space, safe to delete.

2. **`sam3d/hf/_models_cache/`** — Contains entire dinov2 repo clone (~200+ files). This was auto-downloaded by SAM3D at runtime. Massive cruft but needed by the SAM3D node at inference time (it loads dinov2 from torch hub cache). Must be preserved but could be relocated.

3. **`sam3d/hf/` root files** — CODE_OF_CONDUCT.md, CONTRIBUTING.md, doc/ folder, LICENSE, .gitattributes, README.md. These came from the HuggingFace `snapshot_download` and are unnecessary. Safe to delete (keep only `checkpoints/` and `_models_cache/`).

4. **`text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`** — This is an fp8 quantized variant of the UMT5 encoder. The bf16 version in `clip/` is what workflows actually reference. Unless a workflow specifically uses this, it's ~5GB of dead weight. Verify before deleting.

5. **`onnx/`** — Directory exists but is empty. Detection workflows need `process_checkpoint/det/yolov10m.onnx` here (~30MB download).

6. **`Wan22Animate/`** — Subdirectory under diffusion_models/ that was created for the old `_scaled_KJ` model that doesn't exist on HuggingFace. Now empty since `Wan2_2-I2V-A14B-LOW_bf16.safetensors` was downloaded to the parent directory instead. Delete empty dir.

7. **`vae/wan2.2_vae.safetensors`** — Need to verify this is actually different from `wan_2.1_vae.safetensors`. The Wan 2.2 48ch VAE (1.41GB) should be significantly larger than the 2.1 VAE. If they're the same size, one may be mislabeled.

### 1.2 Scripts & Tools Inventory

**The mess — 5 overlapping download mechanisms:**

| File | Platform | MODELS_DIR default | Status |
|------|----------|-------------------|--------|
| `download_models.sh` | Linux (generic) | `/data/comfyui/models` ❌ | Stale — wrong path, references non-existent `Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ` |
| `download_models_rocky.sh` | Rocky Linux | `/data/comfyui/models` ❌ | Better — has wan-t2v profile, but wrong default path, same stale model ref |
| `download_wan.ps1` | Windows | unknown | PowerShell only — irrelevant for server |
| `download_ttf_models.ps1` | Windows | unknown | PowerShell only — irrelevant for server |
| `download_models.ps1` | Windows | unknown | PowerShell only — irrelevant for server |
| `models/manifest.yaml` | Cross-platform | `/models` (container) | Incomplete — missing Wan, RIFE, SAM2, SAM3D, detection models |
| `models/manager.py` | Cross-platform | `/models` (container) | Good framework but doesn't know about diffusion_models category |

**Other scripts:**

| File | Purpose | Status |
|------|---------|--------|
| `bootstrap.sh` | First-time setup (dirs, SAM3D download, validate) | Good foundation, needs updating |
| `comfy_doctor.sh` | Runtime diagnostic (nodes, models, mounts, disk) | Good — already does node + workflow checks |
| `scripts/api_wrapper.py` | 37KB wrapper service (port 8189) for GPU profiles | Core infrastructure, don't touch |
| `scripts/check_models.py` | Model verification | Superseded by manager.py |
| `scripts/entrypoint.sh` | Docker container entrypoint | Core infrastructure |
| `scripts/start.sh` | ComfyUI startup | Core infrastructure |
| `comfyui.sh` | Unknown | Need to inspect |
| `comfyui.ps1` | PowerShell launcher | Windows only |
| `test_lifecycle.py` | Profile lifecycle tests | Used for smoke testing |
| `test_quick.ps1` | PowerShell quick test | Windows only |

**Stray file:**
- `~/services/comfyui-pipeline/Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors` — **14GB model file in the repo root!** This is the model that doesn't exist on HuggingFace anymore. Either it was downloaded before Kijai removed it, or it's a renamed copy. It's not in the models volume, so ComfyUI can't see it. **Delete after confirming wan_t2v.json is updated to use the existing model.**

**Also suspicious:**
- `comfyui-inputworkflows` — Listed as a regular file (not a directory). Probably stale.

### 1.3 Gateway Workflow JSONs (in the AI Gateway project)

These are the workflow JSONs the gateway sends to ComfyUI:

| File | Workflow | Model Referenced | Model Present? |
|------|----------|-----------------|----------------|
| wan_t2v.json | Wan Text-to-Video | `Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors` | ❌ (have `_e4m3fn` without `_scaled_KJ`) |
| wan_i2v.json | Wan Image-to-Video | Need to check | Need to check |
| flux_txt2img.json | Flux txt2img | `flux1-dev-fp8.safetensors` | ✅ |
| sdxl_txt2img.json | SDXL txt2img | `sd_xl_base_1.0.safetensors` | ✅ |
| esrgan_upscale.json | ESRGAN upscale | `RealESRGAN_x4plus.pth` | ✅ |
| TTFCharacterSwap_api.json | Character swap | Multiple | Need to verify |
| TTFCharacterSwapMasked_api.json | Masked char swap | Multiple | Need to verify |
| MultiPersonDetection_api.json | Person detection | `yolov10m.onnx` | ❌ (onnx dir empty) |
| MaskGeneration_api.json | Mask generation | Depends on detection | ❌ (depends on detection) |

### 1.4 ComfyUI Pipeline Workflows (in the pipeline repo)

Only 3 workflows in `~/services/comfyui-pipeline/workflows/`:
- `Multiperson_Detection_api.json`
- `TTFCharacterSwap_api.json`
- `TTFCharacterSwapWithFaceAnimationTransfer.json` (66KB — the monolith)

These are old — the gateway has its own copies. Unclear which is authoritative.

---

## 2. v1 Architecture: The Clean System

### 2.1 Guiding Principles

1. **Single source of truth** for model requirements: one `models.yaml` manifest
2. **One download script** that reads from the manifest: `comfy_models.sh`
3. **Workflow → model mapping** built into the manifest so `comfy_doctor.sh` can validate
4. **No model files outside `/mnt/ssd/comfyui-models/`** — ever
5. **No Windows PowerShell scripts on the server** — archive or delete
6. **Gateway workflow JSONs are authoritative** — pipeline repo workflows are copies

### 2.2 Proposed File Structure

```
~/services/comfyui-pipeline/
├── Dockerfile                    # Main container build
├── Dockerfile.full-build         # Full build variant (archive)
├── docker-compose.yml            # Dev compose
├── docker-compose.rocky.yaml     # Production compose
├── .env                          # Environment vars (HF_TOKEN, etc.)
├── .env.example                  # Template
├── .gitignore
├── README.md                     # ← UPDATED comprehensive guide
│
├── models/
│   ├── manifest.yaml             # ← UPDATED single source of truth
│   └── manager.py                # ← UPDATED to cover all categories
│
├── scripts/
│   ├── entrypoint.sh             # Container entrypoint
│   ├── start.sh                  # ComfyUI startup
│   ├── api_wrapper.py            # GPU profile wrapper (port 8189)
│   └── check_models.py           # ← RETIRE (replaced by comfy_models.sh)
│
├── tools/                        # ← NEW directory for operator tools
│   ├── comfy_doctor.sh           # ← MOVED from root, updated
│   ├── comfy_models.sh           # ← NEW unified model manager (replaces 5 scripts)
│   └── comfy_discovery.py        # ← Node schema discovery tool
│
├── custom_nodes/
│   ├── ComfyUI-MultiPersonDetector/
│   └── websocket_image_save.py
│
├── workflows/                    # Reference copies (gateway is authoritative)
│   └── .gitkeep
│
└── archive/                      # ← NEW for retired scripts
    ├── download_models.sh
    ├── download_models_rocky.sh
    ├── download_models.ps1
    ├── download_wan.ps1
    ├── download_ttf_models.ps1
    ├── comfyui.ps1
    ├── test_quick.ps1
    └── bootstrap.sh              # Functionality absorbed into comfy_models.sh
```

### 2.3 The Unified Model Manifest (`models/manifest.yaml`)

This is the single source of truth. Every model the system needs is listed here, tagged with which workflows require it.

```yaml
version: "2.0"
models_dir: "/mnt/ssd/comfyui-models"  # Host path (overridden by env)

# ─────────────────────────────────────────────────
# GENERATION MODELS (loaded via UNETLoader / WanVideoModelLoader)
# ─────────────────────────────────────────────────
diffusion_models:
  - name: "Flux.1 Dev fp8"
    file: "diffusion_models/flux1-dev-fp8.safetensors"
    source: { repo: "Comfy-Org/flux1-dev", file: "flux1-dev-fp8.safetensors" }
    size_gb: 11.9
    workflows: [flux-txt2img]

  - name: "Wan 2.1 T2V 14B fp8"
    file: "diffusion_models/Wan2_1-T2V-14B_fp8_e4m3fn.safetensors"
    source: { repo: "Kijai/WanVideo_comfy", file: "Wan2_1-T2V-14B_fp8_e4m3fn.safetensors" }
    size_gb: 14.9
    workflows: [wan-t2v]

  - name: "Wan 2.1 I2V 14B 720P fp8"
    file: "diffusion_models/Wan2_1-I2V-14B-720P_fp8_e4m3fn.safetensors"
    source: { repo: "Kijai/WanVideo_comfy", file: "Wan2_1-I2V-14B-720P_fp8_e4m3fn.safetensors" }
    size_gb: 14.0
    workflows: [wan-i2v]

  - name: "Wan 2.1 FLF2V 14B 720P fp8"
    file: "diffusion_models/Wan2_1-FLF2V-14B-720P_fp8_e4m3fn.safetensors"
    source: { repo: "Kijai/WanVideo_comfy", file: "Wan2_1-FLF2V-14B-720P_fp8_e4m3fn.safetensors" }
    size_gb: 14.0
    workflows: [wan-flf2v]

  - name: "Wan 2.2 Fun Inpaint 5B bf16"
    file: "diffusion_models/wan2.2_fun_inpaint_5B_bf16.safetensors"
    source: { repo: "Kijai/WanVideo_comfy", file: "wan2.2_fun_inpaint_5B_bf16.safetensors" }
    size_gb: 10.0
    workflows: [wan-fun-inpaint]

  - name: "Wan 2.2 Animate I2V A14B LOW bf16"
    file: "diffusion_models/Wan2_2-I2V-A14B-LOW_bf16.safetensors"
    source: { repo: "Kijai/WanVideo_comfy", file: "Wan2_2-I2V-A14B-LOW_bf16.safetensors" }
    size_gb: 28.0
    workflows: [wan-charswap]

# ─────────────────────────────────────────────────
# CHECKPOINTS
# ─────────────────────────────────────────────────
checkpoints:
  - name: "SDXL 1.0 Base"
    file: "checkpoints/sd_xl_base_1.0.safetensors"
    source: { repo: "stabilityai/stable-diffusion-xl-base-1.0", file: "sd_xl_base_1.0.safetensors" }
    size_gb: 6.94
    workflows: [sdxl-txt2img]

# ─────────────────────────────────────────────────
# TEXT ENCODERS / CLIP
# ─────────────────────────────────────────────────
clip:
  - name: "CLIP-L (Flux)"
    file: "clip/clip_l.safetensors"
    source: { repo: "comfyanonymous/flux_text_encoders", file: "clip_l.safetensors" }
    size_gb: 0.24
    workflows: [flux-txt2img]

  - name: "T5-XXL fp8 (Flux)"
    file: "clip/t5xxl_fp8_e4m3fn.safetensors"
    source: { repo: "comfyanonymous/flux_text_encoders", file: "t5xxl_fp8_e4m3fn.safetensors" }
    size_gb: 4.89
    workflows: [flux-txt2img]

  - name: "UMT5-XXL bf16 (Wan)"
    file: "clip/umt5-xxl-enc-bf16.safetensors"
    source: { repo: "Kijai/WanVideo_comfy", file: "umt5-xxl-enc-bf16.safetensors" }
    size_gb: 5.0
    workflows: [wan-t2v, wan-i2v, wan-fun-inpaint, wan-charswap]

# ─────────────────────────────────────────────────
# CLIP VISION
# ─────────────────────────────────────────────────
clip_vision:
  - name: "CLIP Vision H"
    file: "clip_vision/clip_vision_h.safetensors"
    source: { repo: "Kijai/WanVideo_comfy", file: "clip_vision_h.safetensors" }
    size_gb: 3.9
    workflows: [wan-i2v, wan-charswap]

# ─────────────────────────────────────────────────
# VAE
# ─────────────────────────────────────────────────
vae:
  - name: "Flux VAE (ae)"
    file: "vae/ae.safetensors"
    source: { repo: "black-forest-labs/FLUX.1-dev", file: "ae.safetensors", gated: true }
    size_gb: 0.34
    workflows: [flux-txt2img]

  - name: "SDXL VAE"
    file: "vae/sdxl_vae.safetensors"
    source: { repo: "stabilityai/sdxl-vae", file: "sdxl_vae.safetensors" }
    size_gb: 0.32
    workflows: [sdxl-txt2img]

  - name: "Wan 2.1 VAE"
    file: "vae/wan_2.1_vae.safetensors"
    source: { repo: "Kijai/WanVideo_comfy", file: "wan_2.1_vae.safetensors" }
    size_gb: 0.34
    workflows: [wan-t2v, wan-i2v]

  - name: "Wan 2.2 VAE (48ch)"
    file: "vae/wan2.2_vae.safetensors"
    source: { repo: "Kijai/WanVideo_comfy", file: "wan2.2_vae.safetensors" }
    size_gb: 1.41
    workflows: [wan-fun-inpaint, wan-charswap]

# ─────────────────────────────────────────────────
# UPSCALE
# ─────────────────────────────────────────────────
upscale_models:
  - name: "RealESRGAN x4plus"
    file: "upscale_models/RealESRGAN_x4plus.pth"
    source: { url: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" }
    size_gb: 0.064
    workflows: [esrgan]

  - name: "RealESRGAN x4plus Anime"
    file: "upscale_models/RealESRGAN_x4plus_anime_6B.pth"
    source: { url: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth" }
    size_gb: 0.017
    workflows: [esrgan]

  - name: "4x-UltraSharp"
    file: "upscale_models/4x-UltraSharp.pth"
    source: { url: "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth" }
    size_gb: 0.064
    workflows: [esrgan]

# ─────────────────────────────────────────────────
# SEGMENTATION / DETECTION
# ─────────────────────────────────────────────────
sam2:
  - name: "SAM2 Large"
    file: "sam2/sam2.1_hiera_large.safetensors"
    source: { repo: "Kijai/sam2-safetensors", file: "sam2.1_hiera_large.safetensors" }
    size_gb: 0.9
    workflows: [detection, mask-gen, wan-charswap]

sams:
  - name: "SAM ViT-B"
    file: "sams/sam_vit_b_01ec64.pth"
    source: { url: "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" }
    size_gb: 0.375
    workflows: [mask-gen]

onnx:
  - name: "YOLOv10m Detection"
    file: "onnx/process_checkpoint/det/yolov10m.onnx"
    source: { url: "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.onnx" }
    size_gb: 0.03
    workflows: [detection]

# ─────────────────────────────────────────────────
# LORAS
# ─────────────────────────────────────────────────
loras:
  - name: "LightX2V 4-step I2V LoRA"
    file: "loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
    source: { repo: "Kijai/WanVideo_comfy", file: "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors" }
    size_gb: 0.4
    workflows: [wan-charswap]

# ─────────────────────────────────────────────────
# SAM3D (special — HuggingFace snapshot)
# ─────────────────────────────────────────────────
sam3d:
  - name: "SAM3D TRELLIS checkpoints"
    file: "sam3d/hf/checkpoints/pipeline.yaml"
    source: { repo: "JeffreyXiang/TRELLIS-image-large", snapshot: true, gated: false }
    size_gb: 15.0
    workflows: [sam3d-objects]
    note: "Downloaded via huggingface-cli snapshot_download"
```

### 2.4 The Unified Model Manager (`tools/comfy_models.sh`)

Replaces: `download_models.sh`, `download_models_rocky.sh`, `download_wan.ps1`, `download_ttf_models.ps1`, `download_models.ps1`, `bootstrap.sh`, `scripts/check_models.py`

```
Usage:
  ./tools/comfy_models.sh status                    # What's installed vs missing
  ./tools/comfy_models.sh status --workflow wan-t2v  # What does wan-t2v need?
  ./tools/comfy_models.sh download                   # Download all missing models
  ./tools/comfy_models.sh download --workflow wan-t2v # Download only what wan-t2v needs
  ./tools/comfy_models.sh download --profile wan     # Download all wan-related
  ./tools/comfy_models.sh verify                     # Check file sizes, no corruption
  ./tools/comfy_models.sh cleanup                    # Remove orphaned/broken files
```

Key behaviors:
- Reads from `models/manifest.yaml` — single source of truth
- Uses `wget -c` for resumable downloads
- Checks file size before downloading (skip if already present and correct size)
- Supports `--workflow` flag to download only models needed for a specific workflow
- Supports `--profile` flag for groups: sdxl, flux, wan, esrgan, detection, sam3d
- Outputs colored status table showing installed/missing/broken

### 2.5 Updated `comfy_doctor.sh`

Add a `--check-models` mode that cross-references workflow JSONs against the manifest:

```
./tools/comfy_doctor.sh                              # Full diagnostic (existing)
./tools/comfy_doctor.sh --check-workflow FILE.json   # Validate specific workflow (existing)
./tools/comfy_doctor.sh --check-models               # NEW: manifest vs disk audit
./tools/comfy_doctor.sh --preflight                  # NEW: nodes + models + workflows all-in-one
```

The `--preflight` mode is the "start the server and make sure everything works" command. It:
1. Checks container is running
2. Fetches node registry from ComfyUI API
3. For each workflow JSON in the gateway, validates:
   - All class_types exist in node registry
   - All model files referenced exist on disk
   - All required node inputs have values
4. Produces a pass/fail report

---

## 3. Immediate Cleanup Steps

Run these on the server in order:

### Phase 1: Delete cruft (saves ~19GB+)

```bash
# 1. Remove stray 14GB model from repo root
rm ~/services/comfyui-pipeline/Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors

# 2. Remove HuggingFace staging cache
rm -rf /mnt/ssd/comfyui-models/_hf_staging

# 3. Remove sam3d repo docs (keep checkpoints + _models_cache)
rm -f /mnt/ssd/comfyui-models/sam3d/hf/CODE_OF_CONDUCT.md
rm -f /mnt/ssd/comfyui-models/sam3d/hf/CONTRIBUTING.md
rm -f /mnt/ssd/comfyui-models/sam3d/hf/LICENSE
rm -f /mnt/ssd/comfyui-models/sam3d/hf/.gitattributes
rm -f /mnt/ssd/comfyui-models/sam3d/hf/README.md
rm -rf /mnt/ssd/comfyui-models/sam3d/hf/doc
rm -rf /mnt/ssd/comfyui-models/sam3d/hf/.cache  # download metadata, not needed

# 4. Remove empty Wan22Animate directory
rmdir /mnt/ssd/comfyui-models/diffusion_models/Wan22Animate 2>/dev/null

# 5. Check if text_encoders is referenced by anything
# If not, this is a 5GB duplicate of clip/umt5-xxl-enc-bf16.safetensors
ls -la /mnt/ssd/comfyui-models/text_encoders/

# 6. Remove stray file in repo root
rm -f ~/services/comfyui-pipeline/comfyui-inputworkflows
```

### Phase 2: Fix workflow JSONs

```bash
# Fix wan_t2v.json model reference (on gateway server)
# In wan_t2v.json node "1", change:
#   "model": "Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors"
# To:
#   "model": "Wan2_1-T2V-14B_fp8_e4m3fn.safetensors"
```

### Phase 3: Download missing models

```bash
# YOLO detection model (~30MB)
mkdir -p /mnt/ssd/comfyui-models/onnx/process_checkpoint/det
wget -c -O /mnt/ssd/comfyui-models/onnx/process_checkpoint/det/yolov10m.onnx \
  "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.onnx"
```

### Phase 4: Reorganize scripts

```bash
cd ~/services/comfyui-pipeline

# Create new directories
mkdir -p tools archive

# Move doctor to tools
mv comfy_doctor.sh tools/

# Archive old download scripts
mv download_models.sh archive/
mv download_models_rocky.sh archive/
mv download_models.ps1 archive/
mv download_ttf_models.ps1 archive/
mv download_wan.ps1 archive/
mv comfyui.ps1 archive/
mv test_quick.ps1 archive/

# bootstrap.sh functionality goes into comfy_models.sh
mv bootstrap.sh archive/
```

### Phase 5: Rebuild container

```bash
cd ~/services/comfyui-pipeline
docker compose -f docker-compose.rocky.yaml build --no-cache
docker compose -f docker-compose.rocky.yaml up -d
```

---

## 4. Updated README

See next section — this replaces the current dashboard-only README with a comprehensive operational guide.

---

## 5. Workflow-to-Model Quick Reference

For any workflow, here's exactly what you need on disk:

| Workflow | Diffusion Model | VAE | Text Encoder | Other |
|----------|----------------|-----|--------------|-------|
| **sdxl-txt2img** | checkpoints/sd_xl_base_1.0 | vae/sdxl_vae | (built-in) | — |
| **flux-txt2img** | diffusion_models/flux1-dev-fp8 | vae/ae | clip/t5xxl_fp8 + clip/clip_l | — |
| **wan-t2v** | diffusion_models/Wan2_1-T2V-14B_fp8 | vae/wan_2.1_vae | clip/umt5-xxl-enc-bf16 | — |
| **wan-i2v** | diffusion_models/Wan2_1-I2V-14B-720P_fp8 | vae/wan_2.1_vae | clip/umt5-xxl-enc-bf16 | clip_vision/clip_vision_h |
| **wan-fun-inpaint** | diffusion_models/wan2.2_fun_inpaint_5B_bf16 | vae/wan2.2_vae | clip/umt5-xxl-enc-bf16 | — |
| **wan-charswap** | diffusion_models/Wan2_2-I2V-A14B-LOW_bf16 | vae/wan2.2_vae | clip/umt5-xxl-enc-bf16 | clip_vision/clip_vision_h, sam2, loras |
| **esrgan** | — | — | — | upscale_models/* |
| **rife** | — | — | — | (built into RIFE VFI node) |
| **detection** | — | — | — | onnx/process_checkpoint/det/yolov10m.onnx |
| **mask-gen** | — | — | — | sams/sam_vit_b, depends on detection output |
| **sam3d-objects** | — | — | — | sam3d/hf/checkpoints/*, sam3d/hf/_models_cache/* |
