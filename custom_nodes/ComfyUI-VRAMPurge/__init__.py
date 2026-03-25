"""
ComfyUI-VRAMPurge — Nuclear VRAM cleanup node.

Runs inside ComfyUI's process so it has direct access to all model
references, torch CUDA state, and Python garbage collector.

When executed as a workflow, it:
  1. Clears ComfyUI's internal model cache
  2. Clears WanVideoWrapper's cached models (module-level refs)
  3. Forces Python garbage collection
  4. Empties PyTorch CUDA cache
  5. Reports VRAM before/after

Install:
  Copy this file to ComfyUI/custom_nodes/ComfyUI-VRAMPurge/__init__.py

The API wrapper queues this as a single-node workflow via POST /vram/purge.
"""

import gc
import importlib
import sys
from typing import Any, Dict

import torch


class VRAMPurge:
    """Execute to force-clear all GPU memory."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "confirm": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "purge"
    CATEGORY = "TTF/Utils"
    OUTPUT_NODE = True

    def purge(self, confirm: bool = True):
        if not confirm:
            return ("Purge skipped (confirm=False)",)

        report_lines = []

        # --- Before stats ---
        if torch.cuda.is_available():
            before_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            before_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            report_lines.append(
                f"Before: {before_allocated:.2f}GB allocated, "
                f"{before_reserved:.2f}GB reserved"
            )

        # --- Step 1: ComfyUI model management ---
        try:
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.soft_empty_cache()
            report_lines.append("ComfyUI model cache cleared")
        except Exception as e:
            report_lines.append(f"ComfyUI cache clear failed: {e}")

        # --- Step 2: WanVideoWrapper cleanup ---
        # WanVideoWrapper holds model references in various module-level
        # caches and node instance variables. Walk all loaded modules and
        # clear known cache patterns.
        wan_modules_cleared = 0
        for mod_name, mod in list(sys.modules.items()):
            if mod is None:
                continue
            # WanVideoWrapper modules
            if "WanVideoWrapper" in mod_name or "wanvideo" in mod_name.lower():
                for attr_name in list(vars(mod).keys()):
                    obj = getattr(mod, attr_name, None)
                    # Clear model-like objects (nn.Module, large tensors)
                    if isinstance(obj, torch.nn.Module):
                        try:
                            obj.cpu()
                            setattr(mod, attr_name, None)
                            wan_modules_cleared += 1
                        except Exception:
                            pass
                    elif isinstance(obj, torch.Tensor) and obj.is_cuda:
                        try:
                            setattr(mod, attr_name, None)
                            wan_modules_cleared += 1
                        except Exception:
                            pass

        # Also clear any cached models in WanVideoWrapper nodes
        try:
            nodes_dir = "/app/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper"
            for mod_name in list(sys.modules.keys()):
                if nodes_dir in str(getattr(sys.modules.get(mod_name), "__file__", "")):
                    mod = sys.modules[mod_name]
                    for attr in dir(mod):
                        obj = getattr(mod, attr, None)
                        if isinstance(obj, dict):
                            # Clear dict caches that might hold models
                            for k, v in list(obj.items()):
                                if isinstance(v, torch.nn.Module):
                                    obj[k] = None
                                    wan_modules_cleared += 1
        except Exception as e:
            report_lines.append(f"WanVideoWrapper deep clean warning: {e}")

        if wan_modules_cleared:
            report_lines.append(f"WanVideoWrapper: cleared {wan_modules_cleared} cached objects")

        # --- Step 3: Clear other custom node caches ---
        # SAM2, ONNX runtime, etc.
        other_cleared = 0
        for mod_name in list(sys.modules.keys()):
            if mod_name is None:
                continue
            mod = sys.modules.get(mod_name)
            if mod is None:
                continue
            # Look for common patterns: loaded_model, cached_model, _model, etc.
            for attr_name in ["loaded_model", "cached_model", "_model", "_loaded",
                              "current_model", "model_cache"]:
                obj = getattr(mod, attr_name, None)
                if isinstance(obj, torch.nn.Module):
                    try:
                        obj.cpu()
                        setattr(mod, attr_name, None)
                        other_cleared += 1
                    except Exception:
                        pass

        if other_cleared:
            report_lines.append(f"Other caches: cleared {other_cleared} objects")

        # --- Step 4: ONNX Runtime cleanup ---
        try:
            import onnxruntime
            # ORT sessions hold GPU memory — can't easily free them,
            # but gc will collect unreferenced sessions
        except ImportError:
            pass

        # --- Step 5: Aggressive garbage collection ---
        gc.collect()
        gc.collect()  # Second pass catches reference cycles

        # --- Step 6: PyTorch CUDA cache ---
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            after_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            after_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            freed = before_reserved - after_reserved

            report_lines.append(
                f"After: {after_allocated:.2f}GB allocated, "
                f"{after_reserved:.2f}GB reserved"
            )
            report_lines.append(f"Freed: {freed:.2f}GB")

        report = "\n".join(report_lines)
        print(f"[VRAMPurge] {report}")
        return (report,)


# =============================================================================
# ComfyUI Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "VRAMPurge": VRAMPurge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VRAMPurge": "VRAM Purge (Nuclear)",
}