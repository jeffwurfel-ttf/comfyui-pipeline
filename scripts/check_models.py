#!/usr/bin/env python3
"""
Check required models for TTF Character Swap workflow.

Usage:
    python check_models.py
    python check_models.py --models-path H:/models
"""

import argparse
import os
from pathlib import Path

# Default models path
MODELS_PATH = os.environ.get("MODELS_PATH", "H:/dev/AI_dev/models")

# Required models for TTF Character Swap workflow
REQUIRED_MODELS = {
    "Wan 2.1 I2V Model": {
        "paths": [
            "wan/wan2.1_i2v_480p_14B_bf16.safetensors",
            "wan/Wan2.1-I2V-14B-480P_fp8_e4m3fn.safetensors",
            "diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors",
        ],
        "required": True,
        "size": "~14-28GB",
        "url": "https://huggingface.co/Kijai/WanVideo_comfy",
    },
    "Wan VAE": {
        "paths": [
            "wan/wan_2.1_vae.safetensors",
            "vae/wan_2.1_vae.safetensors",
        ],
        "required": True,
        "size": "~330MB",
        "url": "https://huggingface.co/Kijai/WanVideo_comfy",
    },
    "Wan CLIP": {
        "paths": [
            "clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "clip/umt5-xxl-enc-bf16.safetensors",
        ],
        "required": True,
        "size": "~5-10GB",
        "url": "https://huggingface.co/Kijai/WanVideo_comfy",
    },
    "Relight LoRA": {
        "paths": [
            "loras/WanAnimate_relight_lora_fp16.safetensors",
        ],
        "required": False,
        "size": "~200MB",
        "url": "https://huggingface.co/Kijai/WanVideo_comfy",
    },
    "LightX2V LoRA (CFG Distill)": {
        "paths": [
            "loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        ],
        "required": False,
        "size": "~400MB",
        "url": "https://huggingface.co/Kijai/WanVideo_comfy",
    },
    "SAM2 Model": {
        "paths": [
            "sam2/sam2_hiera_large.safetensors",
            "sam2/sam2.1_hiera_large.safetensors",
        ],
        "required": True,
        "size": "~900MB",
        "url": "https://huggingface.co/Kijai/sam2-safetensors",
    },
    "DWPose Model": {
        "paths": [
            "dwpose/dw-ll_ucoco_384.onnx",
            "pose/dw-ll_ucoco_384.onnx",
        ],
        "required": True,
        "size": "~300MB",
        "url": "https://huggingface.co/yzd-v/DWPose",
    },
    "YOLO Detection": {
        "paths": [
            "yolo/yolov8x.pt",
            "ultralytics/yolov8x.pt",
        ],
        "required": False,
        "size": "~130MB",
        "url": "https://github.com/ultralytics/assets/releases",
    },
}


def check_model(models_path: Path, model_info: dict) -> tuple[bool, str]:
    """Check if any of the model paths exist."""
    for rel_path in model_info["paths"]:
        full_path = models_path / rel_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            if size_mb > 1024:
                size_str = f"{size_mb / 1024:.1f}GB"
            else:
                size_str = f"{size_mb:.0f}MB"
            return True, f"{rel_path} ({size_str})"
    return False, model_info["paths"][0]


def main():
    parser = argparse.ArgumentParser(description="Check required models for TTF workflow")
    parser.add_argument("--models-path", "-m", default=MODELS_PATH, help=f"Models path (default: {MODELS_PATH})")
    args = parser.parse_args()
    
    models_path = Path(args.models_path)
    
    print("TTF Character Swap - Model Checker")
    print("=" * 50)
    print(f"Models path: {models_path}")
    print()
    
    if not models_path.exists():
        print(f"ERROR: Models path does not exist: {models_path}")
        return 1
    
    # Check each model
    found_required = 0
    missing_required = 0
    found_optional = 0
    missing_optional = 0
    
    print("Required Models:")
    print("-" * 50)
    for name, info in REQUIRED_MODELS.items():
        if not info["required"]:
            continue
        
        found, path = check_model(models_path, info)
        status = "✓" if found else "✗"
        
        if found:
            print(f"  {status} {name}")
            print(f"      {path}")
            found_required += 1
        else:
            print(f"  {status} {name} (MISSING)")
            print(f"      Expected: {path}")
            print(f"      Download: {info['url']}")
            print(f"      Size: {info['size']}")
            missing_required += 1
    
    print()
    print("Optional Models (for better quality):")
    print("-" * 50)
    for name, info in REQUIRED_MODELS.items():
        if info["required"]:
            continue
        
        found, path = check_model(models_path, info)
        status = "✓" if found else "○"
        
        if found:
            print(f"  {status} {name}")
            print(f"      {path}")
            found_optional += 1
        else:
            print(f"  {status} {name} (not found)")
            print(f"      Download: {info['url']}")
            missing_optional += 1
    
    # Summary
    print()
    print("=" * 50)
    print("Summary:")
    print(f"  Required: {found_required}/{found_required + missing_required}")
    print(f"  Optional: {found_optional}/{found_optional + missing_optional}")
    
    if missing_required > 0:
        print()
        print("⚠ Missing required models! Download from:")
        print("  https://huggingface.co/Kijai/WanVideo_comfy")
        print("  https://huggingface.co/Kijai/sam2-safetensors")
        return 1
    else:
        print()
        print("✓ All required models present!")
        return 0


if __name__ == "__main__":
    exit(main())