# ComfyUI Pipeline — TTF GPU Compute Node

Docker-based ComfyUI deployment for TTF's AI Gateway, providing GPU-accelerated image/video generation, character replacement, segmentation, and 3D mesh creation.

**Server:** 10.10.210.80 (TTF-LAX-GPU01)
**Container:** `comfyui-pipeline` (ComfyUI on port 8188, API wrapper on port 8189)
**Models:** `/mnt/ssd/comfyui-models` (host) → `/models` (container)
**Repo:** `~/services/comfyui-pipeline`

---

## Quick Start

```bash
# 1. Clone and configure
cd ~/services/comfyui-pipeline
cp .env.example .env
# Edit .env: set HF_TOKEN for gated HuggingFace repos

# 2. Download models
./tools/comfy_models.sh status          # See what's installed/missing
./tools/comfy_models.sh download        # Download everything

# 3. Build and start
docker compose -f docker-compose.rocky.yaml up -d --build

# 4. Validate (also fixes container symlinks)
./tools/comfy_doctor.sh --fix
```

ComfyUI UI: http://10.10.210.80:8188
API Wrapper: http://10.10.210.80:8189

---

## Repository Structure

```
comfyui-pipeline/
│
├── Dockerfile                    # Main container build (Rocky Linux base)
├── Dockerfile.full-build         # Full build with all dependencies baked in
├── docker-compose.yml            # Dev compose (Windows/WSL)
├── docker-compose.rocky.yaml     # Production compose (Rocky Linux server)
├── .env / .env.example           # HF_TOKEN, model paths, GPU config
│
├── models/
│   ├── manifest.yaml             # Model registry — every model listed here
│   └── manager.py                # Python model manager (container-side)
│
├── scripts/                      # Container-internal scripts
│   ├── entrypoint.sh             # Docker ENTRYPOINT — runs on container start
│   ├── start.sh                  # Starts ComfyUI with correct args
│   ├── api_wrapper.py            # GPU profile wrapper service (port 8189)
│   │                             #   Manages model loading/unloading per workflow
│   │                             #   Profiles: sdxl, flux-dev, wan-video, esrgan
│   └── check_models.py           # Model verification (container-side)
│
├── tools/                        # Operator tools — run on host, not in container
│   ├── comfy_doctor.sh           # System diagnostic, symlink fixer & workflow validator
│   ├── comfy_models.sh           # Unified model downloader (reads manifest.yaml)
│   ├── comfy_preflight.sh        # Workflow JSON validator (nodes + models)
│   ├── comfy_onboard.sh          # New workflow onboarding assistant
│   └── comfy_discovery.py        # Node schema extractor (queries ComfyUI API)
│
├── custom_nodes/                 # Custom ComfyUI nodes (mounted into container)
│   ├── ComfyUI-MultiPersonDetector/  # Multi-person bbox detection
│   └── websocket_image_save.py       # WebSocket output node
│
├── workflows/                    # Reference workflow JSONs (gateway is authoritative)
│   ├── TTFCharacterSwap_api.json
│   └── Multiperson_Detection_api.json
│
└── archive/                      # Retired scripts (kept for reference)
    ├── download_models.sh        # Replaced by tools/comfy_models.sh
    ├── download_models_rocky.sh  # Replaced by tools/comfy_models.sh
    └── ...
```

---

## Key Files Explained

### Docker Files

| File | Purpose |
|------|---------|
| **Dockerfile** | Main build. Installs ComfyUI, custom nodes (Kijai WanVideo, VHS, RIFE, SAM3D, WanAnimatePreprocess, ProPainter), pip dependencies. |
| **docker-compose.rocky.yaml** | Production config for Rocky Linux server. Maps `/mnt/ssd/comfyui-models` → `/models`, exposes ports 8188 and 8189. |

### Operator Tools (`tools/`)

| Tool | What It Does |
|------|-------------|
| **comfy_doctor.sh** | The go-to diagnostic. Checks container health, GPU/VRAM, model file integrity, broken symlinks, **WanVideoWrapper container symlinks** (auto-creates with `--fix`), node registration, workflow compatibility, disk space. Run after every rebuild. |
| **comfy_models.sh** | Unified model manager. Reads `models/manifest.yaml`, downloads missing models. Supports `--workflow <name>` to download only what a specific workflow needs. |
| **comfy_preflight.sh** | Validates workflow JSONs against live ComfyUI: checks all nodes exist and all model files are on disk. |
| **comfy_onboard.sh** | Give it a workflow JSON and it checks nodes, finds model references, attempts HuggingFace downloads, and reports what to add to manifest.yaml. |

### Container Scripts (`scripts/`)

| Script | When It Runs |
|--------|-------------|
| **entrypoint.sh** | On container start. Sets up environment, checks model mounts. |
| **start.sh** | Launches ComfyUI + API wrapper. |
| **api_wrapper.py** | GPU profile management service on port 8189. Handles model loading/unloading to fit workflows within 24GB VRAM. |

---

## Model Directory Layout

```
/mnt/ssd/comfyui-models/          ← host path, mounted as /models in container
├── checkpoints/                  # SDXL checkpoints
├── clip/                         # Text encoders (T5-XXL, CLIP-L, UMT5-XXL bf16)
├── clip_vision/                  # Vision encoders (CLIP Vision H)
├── diffusion_models/             # UNet/DiT models
│   └── Wan22Animate/             # WanAnimate-specific models (36-channel)
├── loras/                        # LoRA adapters
├── onnx/                         # ONNX models — CANONICAL storage location
│   ├── wholebody/vitpose-l-wholebody.onnx
│   └── process_checkpoint/det/yolov10m.onnx
├── sam2/                         # SAM2 segmentation
├── sam3d/                        # TRELLIS 3D mesh generation
├── sams/                         # Legacy SAM (ViT-B)
├── upscale_models/               # ESRGAN upscalers
└── vae/                          # VAE decoders
```

**Container-internal symlinks** (NOT on host — created by `comfy_doctor.sh --fix`):
```
/models/detection/                ← required by OnnxDetectionModelLoader
  vitpose-l-wholebody.onnx        → /models/onnx/wholebody/vitpose-l-wholebody.onnx
  yolov10m.onnx                   → /models/onnx/process_checkpoint/det/yolov10m.onnx
  onnx/wholebody/...              → same targets (TTFCharacterSwap uses subdir paths)
  process_checkpoint/det/...      → same targets

/models/text_encoders/            ← required by LoadWanVideoT5TextEncoder
  umt5-xxl-enc-bf16.safetensors   → /models/clip/umt5-xxl-enc-bf16.safetensors
  clip_l.safetensors              → /models/clip/clip_l.safetensors
  t5xxl_fp8_e4m3fn.safetensors    → /models/clip/t5xxl_fp8_e4m3fn.safetensors
```

---

## Supported Workflows

| Workflow ID | Description | Key Models | GPU Profile |
|-------------|-------------|------------|-------------|
| **sdxl-txt2img** | SDXL text-to-image | sd_xl_base_1.0, sdxl_vae | sdxl |
| **flux-txt2img** | Flux text-to-image | flux1-dev-fp8, t5xxl, clip_l, ae | flux-dev |
| **wan-t2v** | Wan 2.1 text-to-video | Wan2_1-T2V-14B_fp8, umt5-xxl, wan_2.1_vae | wan-video |
| **wan-i2v** | Wan 2.1 image-to-video | Wan2_1-I2V-14B-720P_fp8, umt5-xxl, clip_vision_h | wan-video |
| **wan-i2v-22** | Wan 2.2 image-to-video | Wan2_2-I2V-A14B-LOW_bf16, umt5-xxl, wan2.2_vae | wan-video |
| **wan-fun-inpaint** | Wan 2.2 diffusion inpainting | wan2.2_fun_inpaint_5B, umt5-xxl, wan2.2_vae | wan-video |
| **wan-charswap** | Character replacement | Wan2_2-Animate-14B_fp8 (36ch), umt5-xxl, clip_vision_h, sam2, vitpose, yolo | wan-video |
| **esrgan** | 4x image upscaling | RealESRGAN_x4plus, UltraSharp | esrgan |
| **rife** | Frame interpolation | Built into node | — |
| **detection** | Multi-person bbox detection | yolov10m.onnx, vitpose | — |
| **mask-gen** | Mask generation from bboxes | sam_vit_b | — |
| **sam3d-objects** | Image to 3D mesh | TRELLIS checkpoints | — |
| **gvhmr-pose-static** | Human mesh recovery (static cam) | gvhmr, hmr2a, SMPL | — |
| **gvhmr-pose-moving** | Human mesh recovery (moving cam) | gvhmr, hmr2a, SMPL, dpvo | — |

---

## Common Operations

### After Any Container Rebuild

```bash
# Always run this first — recreates ephemeral container symlinks
./tools/comfy_doctor.sh --fix
```

### Adding a New Workflow

1. Build in ComfyUI UI at http://10.10.210.80:8188
2. Export API format: Settings → Enable Dev Mode → Save (API Format)
3. Validate: `./tools/comfy_preflight.sh my_workflow.json`
4. Onboard (checks nodes, finds missing models): `./tools/comfy_onboard.sh my_workflow.json`
5. Add any new models to `models/manifest.yaml`
6. Download: `./tools/comfy_models.sh download --workflow <name>`
7. Add to AI Gateway — create YAML manifest in gateway's workflows directory

### Debugging a Failing Workflow

```bash
# 1. Fix and check everything
./tools/comfy_doctor.sh --fix

# 2. Validate specific workflow
./tools/comfy_preflight.sh /path/to/workflow.json

# 3. Check ComfyUI logs
docker logs comfyui-pipeline --tail 100

# 4. Get exact node schema
curl -s http://10.10.210.80:8188/object_info/NodeName | python3 -m json.tool

# 5. Check job error
curl -s http://10.10.210.80:8188/history/<prompt_id> | python3 -m json.tool | grep -A5 "exception_message"

# 6. Check wrapper service
curl http://10.10.210.80:8189/model/status
```

### Rebuilding the Container

```bash
cd ~/services/comfyui-pipeline
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
docker compose -f docker-compose.rocky.yaml build --no-cache
docker compose -f docker-compose.rocky.yaml up -d
./tools/comfy_doctor.sh --fix   # ← always run after rebuild
```

### Downloading the WanAnimate Model (for character swap)

```bash
# Confirm disk space first (~18.4GB needed)
df -h /mnt/ssd

./tools/comfy_models.sh download --workflow wan-charswap
# This will download Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors
# into /mnt/ssd/comfyui-models/diffusion_models/Wan22Animate/
```

---

## Architecture Notes

### How the AI Gateway Connects

```
Artist Workstation
    ↓
AI Gateway (10.10.210.9:8100)
    ↓ (HTTP POST with workflow JSON + params)
API Wrapper (10.10.210.80:8189)
    ↓ (loads GPU profile, submits to ComfyUI)
ComfyUI (10.10.210.80:8188)
    ↓ (executes workflow on RTX 4090)
Output → /mnt/ssd/comfyui-output/
    ↓
AI Gateway serves result back to artist
```

### Key Lessons (from production battle-testing)

1. **After every container rebuild, run `comfy_doctor.sh --fix`** — recreates ephemeral container symlinks, validates node registration, and catches broken models. This is the single most important operational habit.

2. **WanVideoWrapper requires container-internal symlinks** — `OnnxDetectionModelLoader` looks for models at `/models/detection/` (two path styles) and `LoadWanVideoT5TextEncoder` looks at `/models/text_encoders/`. Neither directory exists by default. `comfy_doctor.sh --fix` creates them. They are lost on rebuild.

3. **WanVideoAnimateEmbeds requires a specific fine-tuned model** — `Wan2_2-I2V-A14B-LOW` will fail with `expected 36 channels, got 68`. The TTFCharacterSwap workflow needs `Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors` from `Kijai/WanVideo_comfy_fp8_scaled`.

4. **Wan 2.2 workflows** — ALWAYS use WanVideoWrapper nodes, not native KSampler/VAEDecode. Wan 2.2 needs 48ch VAE (1.41GB), resolutions divisible by 32, umt5-xxl-enc-bf16 T5 (not scaled fp8), frames must be 4n+1.

5. **nginx stale upstream** — after any Docker container rebuild on the gateway server, restart nginx to refresh upstream IP resolution.

6. **Google SDK routing conflict** — if both `GOOGLE_API_KEY` and `GOOGLE_APPLICATION_CREDENTIALS` are set, the SDK routes to the rate-limited Developer API instead of Vertex AI.

7. **ComfyUI race condition** — when ComfyUI marks a job "completed", outputs may not yet be written. Treat empty outputs on "completed" as still-processing; re-verify from history.

8. **Python bytecode** — clear `__pycache__` during deployments to prevent stale `.pyc` causing silent failures in multi-worker deployments.

9. **Never patch container source files** — bake dependencies into Dockerfiles, use bootstrap scripts for model downloads.

10. **Polling timeouts must match actual generation times** — Wan2.2 on 200+ frames needs 30+ minutes; Luma Ray 3+ needs 20+ minutes. Mismatched timeouts are a common failure mode.

11. **YOLOv10m ONNX source** — the correct source is `Wan-AI/Wan2.2-Animate-14B` repo, not THU-MIG. The files must be at `onnx/process_checkpoint/det/yolov10m.onnx` on the host.

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `HF_TOKEN` | — | HuggingFace token (required for Flux VAE, SAM3D, gated models) |
| `COMFYUI_URL` | `http://localhost:8188` | ComfyUI API endpoint |
| `COMFYUI_WRAPPER_URL` | `http://localhost:8189` | API wrapper endpoint |
| `COMFY_MODELS_HOST` | `/mnt/ssd/comfyui-models` | Host model directory |
| `COMFY_MODELS_CONTAINER` | `/models` | Container model directory |
| `COMFY_CONTAINER` | `comfyui-pipeline` | Docker container name |