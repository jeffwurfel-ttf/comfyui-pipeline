"""
ComfyUI-VectorOut — generic float-vector / JSON sink.

SaveVectorJSON clones the proven SaveBboxesJSON shape
(ComfyUI-MultiPersonDetector/__init__.py:126): RETURN_TYPES=(),
OUTPUT_NODE=True, writes a file to the output dir and returns ui.files so the
path is recoverable from /history. Input is a STRING (JSON) — it agrees with
DINOv3Embed's STRING output (D038), so there is no custom type string to drift
across packages, and any STRING-JSON producer can feed it.
"""
import os
import json

try:
    import folder_paths
except Exception:
    folder_paths = None


class SaveVectorJSON:
    """Saves a JSON string (e.g. DINOv3 embeddings) to the output folder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vectors_json": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "embeddings"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "TTF/recognition"
    OUTPUT_NODE = True
    DESCRIPTION = "Saves a JSON string of floats/vectors to the output folder."

    def save(self, vectors_json, filename_prefix="embeddings"):
        if folder_paths:
            output_dir = folder_paths.get_output_directory()
        else:
            output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "output")
            os.makedirs(output_dir, exist_ok=True)

        counter = 1
        while True:
            filename = f"{filename_prefix}_{counter:04d}.json"
            filepath = os.path.join(output_dir, filename)
            if not os.path.exists(filepath):
                break
            counter += 1

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(vectors_json)

        print(f"[SaveVectorJSON] Saved to: {filepath}")
        return {"ui": {"files": [{"filename": filename, "subfolder": "", "type": "output"}]}}


NODE_CLASS_MAPPINGS = {"SaveVectorJSON": SaveVectorJSON}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveVectorJSON": "Save Vector JSON"}
WEB_DIRECTORY = None
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "SaveVectorJSON"]
