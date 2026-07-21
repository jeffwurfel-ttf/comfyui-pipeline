"""
ComfyUI-DINOv3Embed — TTF subprocess wrapper.

Registers ONE node, DINOv3Embed, which delegates model work to dinov3_worker.py
running in the isolated _env/ venv (transformers 4.57.x). The main ComfyUI env
CANNOT load DINOv3 — transformers 5.8.0's modeling_utils crashes under torch
2.4.1 (D032) — so all transformers/model code is confined to the subprocess.
This module imports only main-env-safe things (torch tensor ops, numpy, PIL,
subprocess); it never imports transformers.

Contract (D037/D038):
  IN : IMAGE [B,H,W,3] + optional MASK [B,H,W]  (MASK -> masked pool; else global pool)
  OUT: STRING (JSON) — both readouts per instance: cls + pool. Raw vectors.
       Feeds SaveVectorJSON (ComfyUI-VectorOut), which agrees on STRING.
"""
import os
import json
import subprocess
import tempfile
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

NODE_DIR = Path(__file__).parent.resolve()
VENV_PYTHON = NODE_DIR / "_env" / "bin" / "python"
WORKER = NODE_DIR / "dinov3_worker.py"
MODELS_ROOT = Path("/models/dinov3")
DEFAULT_MODEL = "dinov3-vitl16"          # D034: vitl16 confirmed
DEFAULT_TIMEOUT = 300


def _list_models():
    """Model dirs under /models/dinov3 that carry a config.json. Falls back to
    the confirmed default so INPUT_TYPES never fails at startup."""
    found = []
    try:
        for d in sorted(MODELS_ROOT.iterdir()):
            if (d / "config.json").is_file():
                found.append(d.name)
    except Exception:
        pass
    if DEFAULT_MODEL in found:
        found.remove(DEFAULT_MODEL)
    return [DEFAULT_MODEL] + found


class DINOv3Embed:
    """Embed instance crops with DINOv3 (subprocess venv). Emits CLS + pooled
    vectors as JSON. Optional MASK drives masked patch pooling."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (_list_models(), {"default": DEFAULT_MODEL}),
            },
            "optional": {
                "mask": ("MASK",),
                "timeout_seconds": ("INT", {"default": DEFAULT_TIMEOUT, "min": 30, "max": 3600}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("embeddings_json",)
    FUNCTION = "embed"
    CATEGORY = "TTF/recognition"
    DESCRIPTION = "DINOv3 CLS + masked-pool embeddings per instance crop (runs in isolated venv)."

    def embed(self, image: torch.Tensor, model: str, mask=None, timeout_seconds: int = DEFAULT_TIMEOUT):
        if not VENV_PYTHON.exists():
            raise RuntimeError(f"DINOv3 venv missing at {VENV_PYTHON}. Run: bash {NODE_DIR}/setup_env.sh")
        if not WORKER.exists():
            raise RuntimeError(f"worker not found at {WORKER}")
        model_dir = MODELS_ROOT / model
        if not (model_dir / "config.json").is_file():
            raise RuntimeError(f"model dir not found or incomplete: {model_dir}")

        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError(f"Expected IMAGE [B,H,W,3]; got {tuple(image.shape)}")
        B = image.shape[0]

        # normalize optional mask to a [B',H,W] stack
        mask_stack = None
        if mask is not None:
            mstk = mask if mask.ndim == 3 else mask.unsqueeze(0)
            mask_stack = mstk

        work_dir = Path(tempfile.mkdtemp(prefix="dinov3_job_"))
        try:
            items = []
            img_u8 = (image.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
            for i in range(B):
                ip = work_dir / f"img_{i:04d}.png"
                Image.fromarray(img_u8[i], mode="RGB").save(ip, compress_level=1)
                item = {"image": str(ip), "mask": None}
                if mask_stack is not None:
                    mi = mask_stack[i] if i < mask_stack.shape[0] else mask_stack[0]
                    m_u8 = (mi.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
                    mp = work_dir / f"mask_{i:04d}.png"
                    Image.fromarray(m_u8, mode="L").save(mp)
                    item["mask"] = str(mp)
                items.append(item)

            manifest = work_dir / "manifest.json"
            with open(manifest, "w", encoding="utf-8") as f:
                json.dump({"model_dir": str(model_dir), "items": items}, f)

            cmd = [str(VENV_PYTHON), str(WORKER), "--manifest", str(manifest)]
            print(f"[DINOv3Embed] subprocess: {' '.join(cmd)}")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True,
                                        timeout=timeout_seconds, cwd=str(NODE_DIR), check=False)
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(f"DINOv3 worker timed out after {timeout_seconds}s. "
                                   f"stderr tail: {(e.stderr or '')[-2000:]}") from e
            if result.returncode != 0:
                raise RuntimeError(
                    f"DINOv3 worker exit {result.returncode}\n"
                    f"--- stderr (tail) ---\n{result.stderr[-2000:]}")

            try:
                payload = json.loads(result.stdout.strip())
            except Exception as e:
                raise RuntimeError(
                    f"DINOv3 worker did not return valid JSON: {e}\n"
                    f"--- stdout (tail) ---\n{result.stdout[-1000:]}\n"
                    f"--- stderr (tail) ---\n{result.stderr[-1000:]}") from e

            n = len(payload.get("instances", []))
            print(f"[DINOv3Embed] {n} instance(s), model={payload.get('model')} dim={payload.get('dim')}")
            return (json.dumps(payload),)
        finally:
            try:
                shutil.rmtree(work_dir)
            except Exception as e:
                print(f"[DINOv3Embed] Warning: failed to clean up {work_dir}: {e}")


NODE_CLASS_MAPPINGS = {"DINOv3Embed": DINOv3Embed}
NODE_DISPLAY_NAME_MAPPINGS = {"DINOv3Embed": "DINOv3 Embed (subprocess)"}
WEB_DIRECTORY = None
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "DINOv3Embed"]

print("[DINOv3Embed] Wrapper loaded. Node: DINOv3Embed. Heavy work in isolated venv: ./_env/")
