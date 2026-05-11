"""
SaveEXRCompressed — ComfyUI output node for float32 EXR with compression.

Saves IMAGE tensors as 32-bit float EXR files with configurable compression.
Uses the OpenEXR library already installed in the container.

Install: Drop this file into ComfyUI/custom_nodes/
         No pip install needed — uses OpenEXR + numpy already present.

Compression options:
  DWAA  — lossy, ~10x smaller, visually lossless, best for VFX delivery (default)
  DWAB  — same as DWAA but 256 scanlines (better for full-frame reads)
  ZIP   — lossless, ~2x smaller, safe for masters
  ZIPS  — lossless, single scanline (better for Nuke line-by-line reads)
  PIZ   — lossless, good for grainy images
  NONE  — uncompressed (not recommended)
"""

import os
import numpy as np

try:
    import OpenEXR
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False
    print("[SaveEXRCompressed] WARNING: OpenEXR not installed. pip install OpenEXR")

import folder_paths


# Map friendly names to OpenEXR compression constants
COMPRESSION_MAP = {
    "dwaa": "DWAA_COMPRESSION",
    "dwab": "DWAB_COMPRESSION",
    "zip": "ZIP_COMPRESSION",
    "zips": "ZIPS_COMPRESSION",
    "piz": "PIZ_COMPRESSION",
    "pxr24": "PXR24_COMPRESSION",
    "none": "NO_COMPRESSION",
}


class SaveEXRCompressed:
    """Save images as 32-bit float EXR with compression."""

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "exr_output"}),
                "compression": (
                    list(COMPRESSION_MAP.keys()),
                    {"default": "dwaa"},
                ),
                "colorspace": (
                    ["linear", "sRGB"],
                    {"default": "linear"},
                ),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_exr"
    OUTPUT_NODE = True
    CATEGORY = "image/save"

    def save_exr(self, images, filename_prefix, compression, colorspace):
        if not HAS_OPENEXR:
            raise RuntimeError(
                "OpenEXR not installed. Run: pip install OpenEXR"
            )

        # Get compression constant
        comp_name = COMPRESSION_MAP.get(compression, "DWAA_COMPRESSION")
        comp_value = getattr(OpenEXR, comp_name, OpenEXR.DWAA_COMPRESSION)

        results = []
        batch_size = images.shape[0]

        for i in range(batch_size):
            # IMAGE tensor is [B, H, W, C] float32, range 0-1
            img = images[i].cpu().numpy().astype(np.float32)

            # Convert sRGB to linear if requested
            if colorspace == "linear":
                # sRGB to linear conversion
                img = np.where(
                    img <= 0.04045,
                    img / 12.92,
                    ((img + 0.055) / 1.055) ** 2.4,
                )
                img = img.astype(np.float32)

            height, width, channels = img.shape

            # Build filename with frame number
            frame_num = i + 1
            filename = f"{filename_prefix}_{frame_num:04d}.exr"
            filepath = os.path.join(self.output_dir, filename)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Write EXR using OpenEXR 3.x API — separate R,G,B channels
            # (not "RGB" group which AE/Nuke may not read as standard color)
            if channels >= 3:
                exr_channels = {
                    "R": np.ascontiguousarray(img[:, :, 0]),
                    "G": np.ascontiguousarray(img[:, :, 1]),
                    "B": np.ascontiguousarray(img[:, :, 2]),
                }
            else:
                grey = np.ascontiguousarray(img[:, :, 0])
                exr_channels = {"R": grey, "G": grey, "B": grey}
            exr_header = {
                "compression": comp_value,
                "type": OpenEXR.scanlineimage,
            }

            with OpenEXR.File(exr_header, exr_channels) as outfile:
                outfile.write(filepath)

            file_size = os.path.getsize(filepath)
            size_mb = file_size / (1024 * 1024)
            results.append(filename)
            print(
                f"[SaveEXRCompressed] Saved: {filename} "
                f"({width}x{height}, {compression}, {size_mb:.1f}MB)"
            )

        # Return UI info so ComfyUI knows files were saved
        return {
            "ui": {
                "images": [
                    {
                        "filename": fn,
                        "subfolder": "",
                        "type": "output",
                    }
                    for fn in results
                ]
            }
        }


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "SaveEXRCompressed": SaveEXRCompressed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveEXRCompressed": "Save EXR (Compressed)",
}
