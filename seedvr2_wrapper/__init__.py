"""
ComfyUI-SeedVR2_VideoUpscaler — TTF subprocess wrapper.

This __init__.py REPLACES the original SeedVR2 entrypoint. It registers
ONE node, SeedVR2RestorationUpscale, which delegates all heavy work to
inference_cli.py running in the isolated _env/ venv via subprocess.

Why this exists:
  SeedVR2 imports diffusers.models.autoencoders.vae, which transitively
  loads diffusers.models.attention_dispatch, which crashes at module-load
  time on torch 2.4.1 (PEP 604 union syntax in torch.library.custom_op
  schema). Running SeedVR2 in an isolated venv with diffusers==0.34.0
  (which does not have attention_dispatch.py) sidesteps the bug entirely
  and prevents SeedVR2's import-time global torch patches (issue #468)
  from affecting Wan and other workflows.

The original SeedVR2 nodes (4-node modular: DiT loader, VAE loader,
torch.compile, upscaler) are NOT registered. We expose a single
opinionated node sized for the restoration pipeline.

To restore original behavior: rename __init__.py.original back to
__init__.py and delete this file.
"""

# CRITICAL: do not import anything from .src — that triggers the diffusers chain.
# Only import our wrapper module which is pure Python, no SeedVR2 deps.

from .seedvr2_subprocess_node import (
    SeedVR2RestorationUpscale,
    NODE_CLASS_MAPPINGS as _MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _DISPLAY,
)

NODE_CLASS_MAPPINGS = _MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = _DISPLAY

WEB_DIRECTORY = None  # no JS extensions

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "SeedVR2RestorationUpscale",
]

print("[SeedVR2-TTF] Wrapper loaded. Registered node: SeedVR2RestorationUpscale")
print("[SeedVR2-TTF] Heavy work runs in isolated venv: ./_env/")
