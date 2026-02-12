#!/usr/bin/env python3
"""
ComfyUI API Wrapper Service v2.0

Model lifecycle management + workflow execution for AI Gateway integration.

Endpoints:
    GET  /health           - Health + VRAM + model state
    POST /model/load       - Load a model profile
    POST /model/unload     - Free VRAM
    GET  /model/status     - Current state (node manager polls this)
    GET  /model/profiles   - List available profiles
    POST /generate         - Profile-aware txt2img
    POST /run              - Run arbitrary workflow JSON
    POST /run/{workflow}   - Run named workflow
    GET  /workflows        - List workflow files
"""

import base64
import io
import json
import logging
import os
import random
import time
import uuid
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://localhost:8188")
WORKFLOWS_DIR = Path(os.environ.get("WORKFLOWS_DIR", "/app/workflows"))
TIMEOUT = float(os.environ.get("WORKFLOW_TIMEOUT", "900"))
WARMUP_TIMEOUT = float(os.environ.get("WARMUP_TIMEOUT", "300"))

# Output cleanup settings
COMFYUI_OUTPUT_DIR = Path(os.environ.get("COMFYUI_OUTPUT_DIR", "/app/ComfyUI/output"))
COMFYUI_INPUT_DIR = Path(os.environ.get("COMFYUI_INPUT_DIR", "/app/ComfyUI/input"))
OUTPUT_MAX_AGE_HOURS = float(os.environ.get("OUTPUT_MAX_AGE_HOURS", "24"))
OUTPUT_MAX_SIZE_GB = float(os.environ.get("OUTPUT_MAX_SIZE_GB", "10"))


# =============================================================================
# Model Profiles
# =============================================================================
# Warmup workflows are minimal (64x64, 1 step) to force ComfyUI to load
# model weights into VRAM without wasting compute.
#
# File paths must match what ComfyUI sees via its model search directories.
# The entrypoint.sh creates symlinks to bridge the host layout.

PROFILES = {

    # -------------------------------------------------------------------------
    # SDXL - Stable Diffusion XL 1.0
    # -------------------------------------------------------------------------
    # Uses: checkpoints/sd_xl_base_1.0.safetensors (symlinked from sdxl/)
    # VRAM: ~8 GB
    "sdxl": {
        "description": "Stable Diffusion XL 1.0",
        "capabilities": ["txt2img", "img2img"],
        "estimated_vram_mb": 8000,
        "default_resolution": [1024, 1024],
        "default_steps": 25,
        "default_cfg": 7.5,
        "warmup_workflow": {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "positive"},
                "inputs": {"text": "warmup", "clip": ["1", 1]}
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "negative"},
                "inputs": {"text": "", "clip": ["1", 1]}
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 64, "height": 64, "batch_size": 1}
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0], "positive": ["2", 0],
                    "negative": ["3", 0], "latent_image": ["4", 0],
                    "seed": 0, "steps": 1, "cfg": 1.0,
                    "sampler_name": "euler", "scheduler": "normal",
                    "denoise": 1.0
                }
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]}
            },
            "7": {
                "class_type": "PreviewImage",
                "inputs": {"images": ["6", 0]}
            }
        },
    },

    # -------------------------------------------------------------------------
    # Flux Dev (fp8 quantized for 24GB cards)
    # -------------------------------------------------------------------------
    # Uses:
    #   diffusion_models/flux1-dev-fp8.safetensors  (~12GB, DOWNLOAD NEEDED)
    #   clip/t5xxl_fp8_e4m3fn.safetensors           (~5GB, DOWNLOAD NEEDED)
    #   clip/clip_l.safetensors                      (~240MB, DOWNLOAD NEEDED)
    #   vae/ae.safetensors                           (symlinked from flux-dev/)
    # VRAM: ~16 GB
    #
    # NOTE: You have flux1-dev.safetensors (23GB fp16) which is too large for
    # 24GB cards. Download the fp8 version from:
    #   https://huggingface.co/Comfy-Org/flux1-dev
    #   File: flux1-dev-fp8.safetensors -> diffusion_models/
    #
    # For CLIP encoders:
    #   https://huggingface.co/comfyanonymous/flux_text_encoders
    #   Files: t5xxl_fp8_e4m3fn.safetensors -> clip/
    #          clip_l.safetensors -> clip/
    "flux-dev": {
        "description": "Flux.1 Dev (fp8 quantized)",
        "capabilities": ["txt2img"],
        "estimated_vram_mb": 16000,
        "default_resolution": [1024, 1024],
        "default_steps": 20,
        "default_cfg": 1.0,  # Flux uses guidance embedding, not CFG
        "warmup_workflow": {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": "flux1-dev-fp8.safetensors",
                    "weight_dtype": "fp8_e4m3fn"
                }
            },
            "2": {
                "class_type": "DualCLIPLoader",
                "inputs": {
                    "clip_name1": "t5xxl_fp8_e4m3fn.safetensors",
                    "clip_name2": "clip_l.safetensors",
                    "type": "flux"
                }
            },
            "3": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "ae.safetensors"}
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "positive"},
                "inputs": {"text": "warmup", "clip": ["2", 0]}
            },
            "5": {
                "class_type": "EmptySD3LatentImage",
                "inputs": {"width": 64, "height": 64, "batch_size": 1}
            },
            "6": {
                "class_type": "BasicGuider",
                "inputs": {"model": ["1", 0], "conditioning": ["4", 0]}
            },
            "7": {
                "class_type": "KSamplerSelect",
                "inputs": {"sampler_name": "euler"}
            },
            "8": {
                "class_type": "BasicScheduler",
                "inputs": {
                    "model": ["1", 0], "scheduler": "simple",
                    "steps": 1, "denoise": 1.0
                }
            },
            "9": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["10", 0], "guider": ["6", 0],
                    "sampler": ["7", 0], "sigmas": ["8", 0],
                    "latent_image": ["5", 0]
                }
            },
            "10": {
                "class_type": "RandomNoise",
                "inputs": {"noise_seed": 0}
            },
            "11": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["9", 0], "vae": ["3", 0]}
            },
            "12": {
                "class_type": "PreviewImage",
                "inputs": {"images": ["11", 0]}
            }
        },
    },

    # -------------------------------------------------------------------------
    # Wan 2.2 Animate (character swap / video generation)
    # -------------------------------------------------------------------------
    # Uses (already in your models dir):
    #   diffusion_models/Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors
    #   vae/wan_2.1_vae.safetensors (or loaded via WanVideoVAELoader)
    #   clip/umt5-xxl-enc-bf16.safetensors
    #   clip_vision/clip_vision_h.safetensors
    #   loras/WanAnimate_relight_lora_fp16.safetensors
    #   loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors
    # VRAM: ~18 GB with block swap
    #
    # No warmup workflow - Wan uses custom nodes (WanVideoModelLoader, etc)
    # with block swap + offloading, so it streams in/out of VRAM on demand.
    # The first real job triggers model loading.
    "wan-video": {
        "description": "Wan 2.2 Animate (14B fp8, character swap + video gen)",
        "capabilities": ["video", "character_swap"],
        "estimated_vram_mb": 18000,
        "default_resolution": [960, 544],
        "default_steps": 4,
        "default_cfg": 1.0,
        "warmup_workflow": None,  # Loads on first job via custom nodes
    },

    # -------------------------------------------------------------------------
    # ESRGAN Upscaling
    # -------------------------------------------------------------------------
    # Uses: upscale_models/RealESRGAN_x4plus.pth
    # VRAM: ~1 GB, loads instantly
    "esrgan": {
        "description": "Real-ESRGAN x4 upscaling",
        "capabilities": ["upscale"],
        "estimated_vram_mb": 1000,
        "default_resolution": None,
        "default_steps": None,
        "default_cfg": None,
        "warmup_workflow": None,  # Tiny model, loads on first use
    },
}


# =============================================================================
# Profile State Manager
# =============================================================================

class ProfileManager:
    """Tracks loaded profile, manages load/unload via ComfyUI API."""

    def __init__(self, comfyui_url: str):
        self.comfyui_url = comfyui_url.rstrip("/")
        self.current_profile: Optional[str] = None
        self.loaded_at: Optional[float] = None
        self.last_used: Optional[float] = None
        self.load_time_ms: Optional[float] = None
        self._loading = False
        self._lock = threading.Lock()
        self._client_id = str(uuid.uuid4())

    @property
    def state(self) -> str:
        if self._loading:
            return "loading"
        if self.current_profile:
            return "loaded"
        return "unloaded"

    def get_vram_info(self) -> dict:
        try:
            r = requests.get(f"{self.comfyui_url}/system_stats", timeout=5)
            if r.status_code == 200:
                devices = r.json().get("devices", [])
                if devices:
                    gpu = devices[0]
                    vram_total = gpu.get("vram_total", 0) / (1024 * 1024)
                    vram_free = gpu.get("vram_free", 0) / (1024 * 1024)
                    return {
                        "gpu_name": gpu.get("name", "unknown"),
                        "vram_total_mb": round(vram_total, 1),
                        "vram_used_mb": round(vram_total - vram_free, 1),
                        "vram_available_mb": round(vram_free, 1),
                    }
        except Exception:
            pass
        return {"gpu_name": "unknown", "vram_total_mb": 0,
                "vram_used_mb": 0, "vram_available_mb": 0}

    def _queue_prompt(self, workflow: dict) -> str:
        payload = {"prompt": workflow, "client_id": self._client_id}
        r = requests.post(f"{self.comfyui_url}/prompt", json=payload, timeout=30)
        r.raise_for_status()
        result = r.json()
        if "error" in result:
            raise RuntimeError(f"ComfyUI queue error: {result['error']}")
        if "node_errors" in result and result["node_errors"]:
            errors = result["node_errors"]
            msgs = []
            for nid, err in errors.items():
                if isinstance(err, dict):
                    msgs.append(f"Node {nid}: {err.get('class_type','?')}: {err.get('errors', err)}")
                else:
                    msgs.append(f"Node {nid}: {err}")
            raise RuntimeError(f"Node errors: {'; '.join(msgs)}")
        return result["prompt_id"]

    def _wait_for_prompt(self, prompt_id: str, timeout: float) -> dict:
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = requests.get(f"{self.comfyui_url}/history/{prompt_id}", timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    if prompt_id in data:
                        history = data[prompt_id]
                        status = history.get("status", {})
                        if status.get("completed"):
                            return history
                        if status.get("status_str") == "error":
                            raise RuntimeError(f"Workflow failed: {status.get('messages', [])}")
            except requests.exceptions.RequestException:
                pass
            time.sleep(1.0)
        raise TimeoutError(f"Warmup timed out after {timeout}s")

    def free_vram(self):
        try:
            requests.post(f"{self.comfyui_url}/free",
                          json={"unload_models": True, "free_memory": True}, timeout=30)
        except Exception as e:
            print(f"[ProfileManager] Warning: /free failed: {e}")

    def load(self, profile_name: str) -> dict:
        if profile_name not in PROFILES:
            return {"success": False,
                    "error": f"Unknown profile: {profile_name}. Available: {list(PROFILES.keys())}"}

        if self.current_profile == profile_name and not self._loading:
            vram = self.get_vram_info()
            return {"success": True, "model": profile_name,
                    "already_loaded": True, "vram_used_mb": vram["vram_used_mb"],
                    "load_time_ms": 0}

        if not self._lock.acquire(blocking=False):
            return {"success": False, "error": "Load already in progress"}

        self._loading = True
        load_start = time.time()

        try:
            profile = PROFILES[profile_name]

            print(f"[ProfileManager] Freeing existing models...")
            self.free_vram()
            time.sleep(1)

            warmup = profile.get("warmup_workflow")
            if not warmup:
                # No warmup needed - mark loaded, models load on first use
                self.current_profile = profile_name
                self.loaded_at = time.time()
                self.last_used = time.time()
                self.load_time_ms = 0
                self._loading = False
                vram = self.get_vram_info()
                return {"success": True, "model": profile_name,
                        "description": profile["description"],
                        "vram_used_mb": vram["vram_used_mb"], "load_time_ms": 0,
                        "note": "Models load on first job"}

            print(f"[ProfileManager] Warming up '{profile_name}'...")
            prompt_id = self._queue_prompt(warmup)
            self._wait_for_prompt(prompt_id, WARMUP_TIMEOUT)

            load_time_ms = (time.time() - load_start) * 1000
            vram = self.get_vram_info()

            self.current_profile = profile_name
            self.loaded_at = time.time()
            self.last_used = time.time()
            self.load_time_ms = load_time_ms
            self._loading = False

            print(f"[ProfileManager] '{profile_name}' loaded in "
                  f"{load_time_ms:.0f}ms, VRAM: {vram['vram_used_mb']}MB")

            return {"success": True, "model": profile_name,
                    "description": profile["description"],
                    "vram_used_mb": vram["vram_used_mb"],
                    "load_time_ms": round(load_time_ms, 1)}

        except Exception as e:
            self._loading = False
            self.current_profile = None
            print(f"[ProfileManager] Load failed: {e}")
            return {"success": False, "error": str(e)}
        finally:
            self._lock.release()

    def unload(self) -> dict:
        vram_before = self.get_vram_info()
        self.free_vram()
        time.sleep(1)
        vram_after = self.get_vram_info()
        freed = vram_before["vram_used_mb"] - vram_after["vram_used_mb"]
        prev = self.current_profile
        self.current_profile = None
        self.loaded_at = None
        self.load_time_ms = None
        print(f"[ProfileManager] Unloaded '{prev}', freed {max(0, freed):.0f}MB")
        return {"success": True, "vram_freed_mb": round(max(0, freed), 1)}

    def touch(self):
        self.last_used = time.time()

    def get_status(self) -> dict:
        vram = self.get_vram_info()
        return {
            "state": self.state,
            "model_name": self.current_profile,
            "profile": self.current_profile,
            "vram_used_mb": vram["vram_used_mb"],
            "vram_available_mb": vram["vram_available_mb"],
            "vram_total_mb": vram["vram_total_mb"],
            "gpu_name": vram["gpu_name"],
            "last_used": self.last_used,
            "loaded_at": self.loaded_at,
            "load_time_ms": self.load_time_ms,
        }


# =============================================================================
# ComfyUI Client
# =============================================================================

class ComfyClient:
    IMAGE_NODES = ["LoadImage"]
    PROMPT_NODES = ["CLIPTextEncode"]
    SEED_NODES = ["KSampler", "KSamplerAdvanced"]

    def __init__(self, base_url: str = COMFYUI_URL):
        self.base_url = base_url.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def health_check(self) -> dict:
        r = requests.get(f"{self.base_url}/system_stats", timeout=5)
        r.raise_for_status()
        return r.json()

    def upload_image(self, image_bytes: bytes, filename: str) -> str:
        files = {"image": (filename, io.BytesIO(image_bytes), "image/png")}
        r = requests.post(f"{self.base_url}/upload/image",
                          files=files, data={"overwrite": "true"}, timeout=30)
        r.raise_for_status()
        return r.json().get("name", filename)

    def inject_inputs(self, workflow, image_name=None, prompt=None,
                      negative=None, seed=None, **kwargs):
        wf = json.loads(json.dumps(workflow))
        for nid, node in wf.items():
            ct = node.get("class_type", "")
            inp = node.get("inputs", {})
            if image_name and ct in self.IMAGE_NODES:
                inp["image"] = image_name
            if prompt and ct in self.PROMPT_NODES:
                title = node.get("_meta", {}).get("title", "").lower()
                if "negative" not in title:
                    inp["text"] = prompt
            if negative and ct in self.PROMPT_NODES:
                title = node.get("_meta", {}).get("title", "").lower()
                if "negative" in title:
                    inp["text"] = negative
            if seed is not None and ct in self.SEED_NODES:
                if "seed" in inp:
                    inp["seed"] = seed
        for key, value in kwargs.items():
            if "." in key:
                nid, param = key.split(".", 1)
                if nid in wf:
                    wf[nid]["inputs"][param] = value
        return wf

    def queue(self, workflow: dict) -> str:
        payload = {"prompt": workflow, "client_id": self.client_id}
        r = requests.post(f"{self.base_url}/prompt", json=payload, timeout=30)
        r.raise_for_status()
        result = r.json()
        if "error" in result:
            raise RuntimeError(result["error"])
        if "node_errors" in result and result["node_errors"]:
            raise RuntimeError(f"Node errors: {result['node_errors']}")
        return result["prompt_id"]

    def wait(self, prompt_id: str, timeout: float = TIMEOUT) -> dict:
        start = time.time()
        while time.time() - start < timeout:
            r = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=10)
            if r.status_code == 200:
                data = r.json()
                if prompt_id in data:
                    history = data[prompt_id]
                    status = history.get("status", {})
                    if status.get("completed"):
                        return history
                    if status.get("status_str") == "error":
                        raise RuntimeError("Workflow execution failed")
            time.sleep(1)
        raise TimeoutError(f"Workflow timeout after {timeout}s")

    def get_outputs(self, history: dict) -> List[bytes]:
        outputs = []
        for nid, node_out in history.get("outputs", {}).items():
            for img in node_out.get("images", []):
                params = {"filename": img["filename"],
                          "type": img.get("type", "output"),
                          "subfolder": img.get("subfolder", "")}
                r = requests.get(f"{self.base_url}/view", params=params, timeout=30)
                if r.status_code == 200:
                    outputs.append(r.content)
        return outputs


# =============================================================================
# Globals
# =============================================================================

profile_mgr = ProfileManager(COMFYUI_URL)
comfy = ComfyClient()


# =============================================================================
# FastAPI
# =============================================================================

app = FastAPI(title="ComfyUI Pipeline API",
              description="Model lifecycle + workflows for AI Gateway", version="2.0.0")


class LoadRequest(BaseModel):
    model: str = "sdxl"

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    num_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: int = -1
    num_images: int = 1

class WorkflowRequest(BaseModel):
    workflow: Dict[str, Any]
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    inputs: Optional[Dict[str, Any]] = None
    timeout: Optional[float] = None  # Override default WORKFLOW_TIMEOUT

class WorkflowResponse(BaseModel):
    success: bool
    images: List[str] = []
    latency_ms: float = 0.0
    error: Optional[str] = None


# -- Lifecycle ----------------------------------------------------------------

@app.get("/health")
async def health():
    try:
        stats = comfy.health_check()
        devices = stats.get("devices", [{}])
        gpu = devices[0] if devices else {}
        vt = gpu.get("vram_total", 0) / (1024 * 1024)
        vf = gpu.get("vram_free", 0) / (1024 * 1024)
        return {
            "status": "healthy",
            "device": "cuda" if gpu.get("type") == "cuda" else gpu.get("type", "unknown"),
            "gpu_name": gpu.get("name", "unknown"),
            "model_loaded": profile_mgr.current_profile is not None,
            "model_name": profile_mgr.current_profile,
            "vram_used_mb": round(vt - vf, 1),
            "vram_total_mb": round(vt, 1),
            "vram_available_mb": round(vf, 1),
            "comfyui": "connected",
        }
    except Exception as e:
        return {"status": "degraded", "device": "unknown", "gpu_name": "unknown",
                "model_loaded": False, "model_name": None,
                "vram_used_mb": 0, "vram_total_mb": 0, "vram_available_mb": 0,
                "comfyui": "disconnected", "error": str(e)}

@app.post("/model/load")
async def load_model(request: LoadRequest):
    result = profile_mgr.load(request.model)
    if not result.get("success"):
        raise HTTPException(500, detail=result.get("error", "Load failed"))
    return result

@app.post("/model/unload")
async def unload_model():
    return profile_mgr.unload()

@app.get("/model/status")
async def model_status():
    return profile_mgr.get_status()

@app.get("/model/profiles")
async def list_profiles():
    profiles = [{"name": n, "description": p["description"],
                 "capabilities": p["capabilities"],
                 "estimated_vram_mb": p["estimated_vram_mb"],
                 "default_resolution": p.get("default_resolution")}
                for n, p in PROFILES.items()]
    return {"profiles": profiles, "current": profile_mgr.current_profile}


# -- Upload -------------------------------------------------------------------

@app.post("/upload/image")
async def upload_image(image: UploadFile = File(...)):
    """Upload an image to ComfyUI's input directory."""
    try:
        image_name = comfy.upload_image(await image.read(), image.filename)
        return {"name": image_name, "success": True}
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {e}")


# -- Generate -----------------------------------------------------------------

@app.post("/generate")
async def generate(request: GenerateRequest):
    current = profile_mgr.current_profile
    if not current:
        raise HTTPException(503, "No model loaded. Call POST /model/load first.")

    profile = PROFILES.get(current)
    if not profile:
        raise HTTPException(500, f"Unknown profile: {current}")
    if "txt2img" not in profile.get("capabilities", []):
        raise HTTPException(400, f"Profile '{current}' doesn't support txt2img")

    steps = request.num_steps or profile.get("default_steps", 25)
    cfg = request.guidance_scale if request.guidance_scale is not None else profile.get("default_cfg", 7.5)
    seed = request.seed if request.seed >= 0 else random.randint(0, 2**32 - 1)

    # Try workflow file first, fall back to building from warmup template
    workflow = _load_workflow_file(current)
    if workflow is None:
        workflow = _build_generate_workflow(
            current, request.prompt, request.negative_prompt,
            request.width, request.height, steps, cfg, seed, request.num_images)

    if workflow is None:
        raise HTTPException(500, f"No txt2img workflow for '{current}'")

    workflow = comfy.inject_inputs(workflow, prompt=request.prompt,
        negative=request.negative_prompt or None, seed=seed)

    start_time = time.time()
    try:
        prompt_id = comfy.queue(workflow)
        history = comfy.wait(prompt_id)
        outputs = comfy.get_outputs(history)
        profile_mgr.touch()
        images_b64 = [base64.b64encode(img).decode() for img in outputs]
        return {"success": True, "model": current,
                "images": [{"base64": b, "seed": seed, "format": "png"} for b in images_b64],
                "latency_ms": round((time.time() - start_time) * 1000, 1)}
    except Exception as e:
        return {"success": False, "error": str(e), "model": current,
                "latency_ms": round((time.time() - start_time) * 1000, 1)}


def _load_workflow_file(profile_name):
    for name in [f"{profile_name}_txt2img.json",
                 f"{profile_name.replace('-', '_')}_txt2img.json"]:
        path = WORKFLOWS_DIR / name
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
    return None


def _build_generate_workflow(profile_name, prompt, negative,
                             width, height, steps, cfg, seed, num_images):
    profile = PROFILES.get(profile_name)
    if not profile or not profile.get("warmup_workflow"):
        return None

    wf = json.loads(json.dumps(profile["warmup_workflow"]))

    if profile_name == "sdxl":
        for nid, node in wf.items():
            ct = node["class_type"]
            inp = node["inputs"]
            if ct == "CLIPTextEncode":
                title = node.get("_meta", {}).get("title", "").lower()
                inp["text"] = (negative or "") if "negative" in title else prompt
            elif ct == "EmptyLatentImage":
                inp.update({"width": width, "height": height, "batch_size": num_images})
            elif ct == "KSampler":
                inp.update({"steps": steps, "cfg": cfg, "seed": seed})
            elif ct == "PreviewImage":
                node["class_type"] = "SaveImage"
                node["inputs"] = {"images": inp["images"], "filename_prefix": "gateway"}
        return wf

    elif profile_name == "flux-dev":
        for nid, node in wf.items():
            ct = node["class_type"]
            inp = node["inputs"]
            if ct == "CLIPTextEncode":
                inp["text"] = prompt
            elif ct == "EmptySD3LatentImage":
                inp.update({"width": width, "height": height, "batch_size": num_images})
            elif ct == "BasicScheduler":
                inp["steps"] = steps
            elif ct == "RandomNoise":
                inp["noise_seed"] = seed
            elif ct == "PreviewImage":
                node["class_type"] = "SaveImage"
                node["inputs"] = {"images": inp["images"], "filename_prefix": "gateway"}
        return wf

    return None


# -- Workflows ----------------------------------------------------------------

@app.get("/workflows")
async def list_workflows():
    if not WORKFLOWS_DIR.exists():
        return {"workflows": [], "path": str(WORKFLOWS_DIR)}
    workflows = []
    for f in WORKFLOWS_DIR.glob("*.json"):
        try:
            wf = json.loads(f.read_text())
            workflows.append({"name": f.stem, "file": f.name, "nodes": len(wf)})
        except Exception:
            pass
    return {"workflows": workflows, "path": str(WORKFLOWS_DIR)}

@app.post("/run", response_model=WorkflowResponse)
async def run_workflow(request: WorkflowRequest):
    start_time = time.time()
    try:
        workflow = comfy.inject_inputs(request.workflow, prompt=request.prompt,
            negative=request.negative_prompt, seed=request.seed, **(request.inputs or {}))
        prompt_id = comfy.queue(workflow)
        history = comfy.wait(prompt_id, timeout=request.timeout or TIMEOUT)
        outputs = comfy.get_outputs(history)
        profile_mgr.touch()
        _auto_cleanup()
        return WorkflowResponse(success=True,
            images=[base64.b64encode(img).decode() for img in outputs],
            latency_ms=(time.time() - start_time) * 1000)
    except Exception as e:
        return WorkflowResponse(success=False, error=str(e),
            latency_ms=(time.time() - start_time) * 1000)

@app.post("/run/{workflow_name}", response_model=WorkflowResponse)
async def run_named_workflow(workflow_name: str,
    image: Optional[UploadFile] = File(None), prompt: Optional[str] = Form(None),
    negative_prompt: Optional[str] = Form(None), seed: Optional[int] = Form(None)):
    start_time = time.time()
    wf_path = WORKFLOWS_DIR / f"{workflow_name}.json"
    if not wf_path.exists():
        raise HTTPException(404, f"Workflow not found: {workflow_name}")
    try:
        workflow = json.loads(wf_path.read_text())
    except Exception as e:
        raise HTTPException(400, f"Invalid workflow: {e}")
    try:
        image_name = None
        if image:
            image_name = comfy.upload_image(await image.read(), image.filename)
        workflow = comfy.inject_inputs(workflow, image_name=image_name,
            prompt=prompt, negative=negative_prompt, seed=seed)
        prompt_id = comfy.queue(workflow)
        history = comfy.wait(prompt_id)
        outputs = comfy.get_outputs(history)
        profile_mgr.touch()
        return WorkflowResponse(success=True,
            images=[base64.b64encode(img).decode() for img in outputs],
            latency_ms=(time.time() - start_time) * 1000)
    except Exception as e:
        return WorkflowResponse(success=False, error=str(e),
            latency_ms=(time.time() - start_time) * 1000)

@app.post("/run/{workflow_name}/raw")
async def run_named_workflow_raw(workflow_name: str,
    image: Optional[UploadFile] = File(None), prompt: Optional[str] = Form(None),
    negative_prompt: Optional[str] = Form(None), seed: Optional[int] = Form(None)):
    wf_path = WORKFLOWS_DIR / f"{workflow_name}.json"
    if not wf_path.exists():
        raise HTTPException(404, f"Workflow not found: {workflow_name}")
    try:
        workflow = json.loads(wf_path.read_text())
    except Exception as e:
        raise HTTPException(400, f"Invalid workflow: {e}")
    try:
        image_name = None
        if image:
            image_name = comfy.upload_image(await image.read(), image.filename)
        workflow = comfy.inject_inputs(workflow, image_name=image_name,
            prompt=prompt, negative=negative_prompt, seed=seed)
        prompt_id = comfy.queue(workflow)
        history = comfy.wait(prompt_id)
        outputs = comfy.get_outputs(history)
        profile_mgr.touch()
        if not outputs:
            raise HTTPException(500, "No output images")
        return Response(content=outputs[0], media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# -- Output Cleanup -----------------------------------------------------------

def _cleanup_outputs(max_age_hours: float = None, dry_run: bool = False) -> dict:
    """Remove old files from ComfyUI output and input directories.
    
    Returns summary of what was (or would be) deleted.
    """
    if max_age_hours is None:
        max_age_hours = OUTPUT_MAX_AGE_HOURS
    
    cutoff = time.time() - (max_age_hours * 3600)
    results = {"deleted": [], "freed_mb": 0, "errors": [], "dry_run": dry_run}
    
    for dir_path in [COMFYUI_OUTPUT_DIR, COMFYUI_INPUT_DIR]:
        if not dir_path.exists():
            continue
        for f in dir_path.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                size_mb = f.stat().st_size / (1024 * 1024)
                if dry_run:
                    results["deleted"].append({"file": f.name, "dir": dir_path.name,
                                               "size_mb": round(size_mb, 1)})
                    results["freed_mb"] += size_mb
                else:
                    try:
                        f.unlink()
                        results["deleted"].append({"file": f.name, "dir": dir_path.name,
                                                   "size_mb": round(size_mb, 1)})
                        results["freed_mb"] += size_mb
                    except Exception as e:
                        results["errors"].append({"file": f.name, "error": str(e)})
    
    results["freed_mb"] = round(results["freed_mb"], 1)
    return results


def _get_output_stats() -> dict:
    """Get current output directory stats."""
    stats = {"directories": {}}
    total_size = 0
    total_files = 0
    
    for dir_path in [COMFYUI_OUTPUT_DIR, COMFYUI_INPUT_DIR]:
        if not dir_path.exists():
            continue
        files = list(dir_path.iterdir())
        file_list = []
        dir_size = 0
        for f in files:
            if f.is_file():
                size = f.stat().st_size
                dir_size += size
                age_hours = (time.time() - f.stat().st_mtime) / 3600
                file_list.append({
                    "name": f.name,
                    "size_mb": round(size / (1024 * 1024), 1),
                    "age_hours": round(age_hours, 1),
                })
        file_list.sort(key=lambda x: x["age_hours"])
        stats["directories"][dir_path.name] = {
            "path": str(dir_path),
            "file_count": len(file_list),
            "total_mb": round(dir_size / (1024 * 1024), 1),
            "files": file_list,
        }
        total_size += dir_size
        total_files += len(file_list)
    
    stats["total_files"] = total_files
    stats["total_mb"] = round(total_size / (1024 * 1024), 1)
    stats["total_gb"] = round(total_size / (1024 * 1024 * 1024), 2)
    stats["max_gb"] = OUTPUT_MAX_SIZE_GB
    stats["max_age_hours"] = OUTPUT_MAX_AGE_HOURS
    return stats


def _auto_cleanup():
    """Run cleanup if output folder exceeds size threshold."""
    total = 0
    for dir_path in [COMFYUI_OUTPUT_DIR, COMFYUI_INPUT_DIR]:
        if dir_path.exists():
            total += sum(f.stat().st_size for f in dir_path.iterdir() if f.is_file())
    
    total_gb = total / (1024 * 1024 * 1024)
    if total_gb > OUTPUT_MAX_SIZE_GB:
        logger.warning(f"Output dirs at {total_gb:.1f}GB (limit {OUTPUT_MAX_SIZE_GB}GB), cleaning up")
        result = _cleanup_outputs()
        if result["deleted"]:
            logger.info(f"Auto-cleanup: removed {len(result['deleted'])} files, freed {result['freed_mb']}MB")


@app.get("/outputs/stats")
async def output_stats():
    """Get output directory size and file listing."""
    return _get_output_stats()


@app.post("/outputs/cleanup")
async def cleanup_outputs(max_age_hours: Optional[float] = None, dry_run: bool = False):
    """Clean up old output and input files.
    
    Args:
        max_age_hours: Delete files older than this (default: OUTPUT_MAX_AGE_HOURS env)
        dry_run: If true, show what would be deleted without deleting
    """
    return _cleanup_outputs(max_age_hours=max_age_hours, dry_run=dry_run)


@app.post("/outputs/purge")
async def purge_outputs():
    """Delete ALL output and input files immediately."""
    return _cleanup_outputs(max_age_hours=0)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("WRAPPER_PORT", "8189"))
    host = os.environ.get("WRAPPER_HOST", "0.0.0.0")
    print(f"ComfyUI Pipeline API v2.0")
    print(f"  Wrapper:   http://{host}:{port}")
    print(f"  ComfyUI:   {COMFYUI_URL}")
    print(f"  Workflows: {WORKFLOWS_DIR}")
    print(f"  Profiles:  {', '.join(PROFILES.keys())}")
    uvicorn.run(app, host=host, port=port)