"""
SeedVR2 subprocess wrapper node for ComfyUI.

Contract:
  Input:  IMAGE batch (torch tensor [N,H,W,3], float in [0,1]) + parameters
  Output: IMAGE batch (torch tensor [N,H',W',3]) — upscaled frames

Heavy work runs in subprocess via ./_env/bin/python ./inference_cli.py
which keeps SeedVR2's diffusers import chain isolated from the main
ComfyUI process.

Why mp4 round-trip:
  SeedVR2's inference_cli.py treats a directory of PNGs as N independent
  images (each gets padded to batch_size and processed solo) — useless
  for video upscaling and OOMs at 1080p. Passing a single mp4 makes the
  CLI process it as ONE clip with batch_size applied to internal chunks.

Why VAE tiling defaults ON:
  At 1080p output (854x480 → 1920x1080), the VAE encode pads input to
  1936x1088 internally. Without tiling, this OOMs on a 24GB 4090 even
  with batch_size=5. Tiling adds ~10% overhead but is non-negotiable
  for our hardware target.
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
from PIL import Image


# --- Constants ---

NODE_DIR = Path(__file__).parent.resolve()
VENV_PYTHON = NODE_DIR / "_env" / "bin" / "python"
INFERENCE_CLI = NODE_DIR / "inference_cli.py"
FFMPEG_BIN = "/usr/bin/ffmpeg"

DEFAULT_TIMEOUT = 1800
INTERMEDIATE_FPS = 24


def _coerce_batch_size(n: int) -> int:
    """SeedVR2 expects 4n+1 batch sizes. Coerce to nearest valid."""
    if n < 5:
        return 5
    return ((n - 1) // 4) * 4 + 1


# --- The node ---

class SeedVR2RestorationUpscale:
    """
    Stage 1 of the restoration pipeline. Upscales an image batch by 2x (or
    to a configurable target resolution) using SeedVR2 7B FP16.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "target_resolution": ("INT", {
                    "default": 1080, "min": 256, "max": 2160, "step": 8,
                }),
                "batch_size": ("INT", {
                    "default": 5, "min": 5, "max": 200, "step": 4,
                    "tooltip": "Frames per batch (4n+1; coerced if not). 5 is safest at 1080p on 24GB."
                }),
                "color_correction": (["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"], {
                    "default": "lab",
                }),
                "uniform_batch_size": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                # VAE tiling — required for 1080p output on 24GB GPUs
                "vae_encode_tiled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Tile VAE encode (required for 1080p+ on 24GB GPUs)."
                }),
                "vae_encode_tile_size": ("INT", {
                    "default": 1024, "min": 256, "max": 2048, "step": 64,
                }),
                "vae_encode_tile_overlap": ("INT", {
                    "default": 128, "min": 32, "max": 512, "step": 32,
                }),
                "vae_decode_tiled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Tile VAE decode (required for 1080p+ on 24GB GPUs)."
                }),
                "vae_decode_tile_size": ("INT", {
                    "default": 1024, "min": 256, "max": 2048, "step": 64,
                }),
                "vae_decode_tile_overlap": ("INT", {
                    "default": 128, "min": 32, "max": 512, "step": 32,
                }),
                # Memory optimization
                "blocks_to_swap": ("INT", {
                    "default": 0, "min": 0, "max": 36,
                    "tooltip": "Transformer blocks offloaded to CPU. 0 = no offload (24GB+ GPUs)."
                }),
                "temporal_overlap": ("INT", {
                    "default": 0, "min": 0, "max": 10,
                    "tooltip": "Overlap frames between batches for smoothing. 0 ok at small batch."
                }),
                "max_resolution": ("INT", {
                    "default": 0, "min": 0, "max": 4320,
                }),
                "dit_model": (
                    [
                        "seedvr2_ema_7b_fp16.safetensors",
                        "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
                        "seedvr2_ema_3b_fp16.safetensors",
                    ],
                    {"default": "seedvr2_ema_7b_fp16.safetensors"}
                ),
                "timeout_seconds": ("INT", {"default": DEFAULT_TIMEOUT, "min": 60, "max": 14400}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_frames",)
    FUNCTION = "upscale"
    CATEGORY = "TTF/restoration"

    def upscale(
        self,
        images: torch.Tensor,
        target_resolution: int,
        batch_size: int,
        color_correction: str,
        uniform_batch_size: bool,
        seed: int,
        vae_encode_tiled: bool = True,
        vae_encode_tile_size: int = 1024,
        vae_encode_tile_overlap: int = 128,
        vae_decode_tiled: bool = True,
        vae_decode_tile_size: int = 1024,
        vae_decode_tile_overlap: int = 128,
        blocks_to_swap: int = 0,
        temporal_overlap: int = 0,
        max_resolution: int = 0,
        dit_model: str = "seedvr2_ema_7b_fp16.safetensors",
        timeout_seconds: int = DEFAULT_TIMEOUT,
    ) -> Tuple[torch.Tensor]:

        # Pre-flight
        if not VENV_PYTHON.exists():
            raise RuntimeError(f"SeedVR2 venv missing at {VENV_PYTHON}. Run: bash {NODE_DIR}/setup_env.sh")
        if not INFERENCE_CLI.exists():
            raise RuntimeError(f"inference_cli.py not found at {INFERENCE_CLI}")
        if not Path(FFMPEG_BIN).exists():
            raise RuntimeError(f"ffmpeg not found at {FFMPEG_BIN}")

        n_frames, height, width, channels = images.shape
        if channels != 3:
            raise ValueError(f"Expected 3-channel IMAGE input; got {channels}")
        if n_frames < 1:
            raise ValueError("No frames in input batch")

        coerced_batch = _coerce_batch_size(batch_size)
        if coerced_batch != batch_size:
            print(f"[SeedVR2-TTF] Coerced batch_size {batch_size} -> {coerced_batch} (4n+1)")
        effective_batch = min(coerced_batch, n_frames if n_frames >= 5 else 5)

        # Working directories
        base_tmp = NODE_DIR / "_tmp"
        base_tmp.mkdir(exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix="seedvr2_job_", dir=base_tmp))
        in_png_dir = work_dir / "in_png"
        out_extracted_dir = work_dir / "out_png"
        in_png_dir.mkdir()
        out_extracted_dir.mkdir()
        in_mp4_path = work_dir / "input.mp4"
        out_mp4_path = work_dir / "seedvr2_output.mp4"

        try:
            # 1. Tensor -> PNG sequence
            print(f"[SeedVR2-TTF] Staging {n_frames} frames as PNGs in {in_png_dir}")
            self._write_frames_as_png(images, in_png_dir)

            # 2. PNG sequence -> lossless mp4
            print(f"[SeedVR2-TTF] Encoding PNGs to lossless mp4 at {in_mp4_path}")
            self._encode_pngs_to_mp4(in_png_dir, in_mp4_path)

            # 3. Build CLI args
            cli_args = [
                str(VENV_PYTHON),
                str(INFERENCE_CLI),
                str(in_mp4_path),
                "--output", str(out_mp4_path),
                "--output_format", "mp4",
                "--video_backend", "ffmpeg",
                "--dit_model", dit_model,
                "--resolution", str(target_resolution),
                "--batch_size", str(effective_batch),
                "--color_correction", color_correction,
                "--seed", str(seed),
                "--temporal_overlap", str(temporal_overlap),
            ]
            if uniform_batch_size:
                cli_args.append("--uniform_batch_size")

            # VAE encode tiling
            if vae_encode_tiled:
                cli_args += [
                    "--vae_encode_tiled",
                    "--vae_encode_tile_size", str(vae_encode_tile_size),
                    "--vae_encode_tile_overlap", str(vae_encode_tile_overlap),
                ]
            # VAE decode tiling
            if vae_decode_tiled:
                cli_args += [
                    "--vae_decode_tiled",
                    "--vae_decode_tile_size", str(vae_decode_tile_size),
                    "--vae_decode_tile_overlap", str(vae_decode_tile_overlap),
                ]

            if blocks_to_swap > 0:
                cli_args += ["--blocks_to_swap", str(blocks_to_swap),
                             "--dit_offload_device", "cpu"]
            if max_resolution > 0:
                cli_args += ["--max_resolution", str(max_resolution)]

            print(f"[SeedVR2-TTF] Subprocess: {' '.join(cli_args)}")

            # 4. Run subprocess
            try:
                result = subprocess.run(
                    cli_args,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    cwd=str(NODE_DIR),
                    check=False,
                )
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(
                    f"SeedVR2 subprocess timed out after {timeout_seconds}s. "
                    f"stderr tail: {(e.stderr or b'').decode()[-2000:]}"
                ) from e

            if result.returncode != 0:
                raise RuntimeError(
                    f"SeedVR2 subprocess exit code {result.returncode}\n"
                    f"--- stdout (tail) ---\n{result.stdout[-2000:]}\n"
                    f"--- stderr (tail) ---\n{result.stderr[-2000:]}"
                )

            # 5. Resolve actual output path
            actual_out = self._resolve_output_mp4(out_mp4_path, work_dir)
            if actual_out is None:
                raise RuntimeError(
                    f"SeedVR2 succeeded but no output mp4 found.\n"
                    f"Looked for: {out_mp4_path}\n"
                    f"work_dir contents: {list(work_dir.rglob('*'))}\n"
                    f"--- stdout (tail) ---\n{result.stdout[-2000:]}"
                )
            print(f"[SeedVR2-TTF] SeedVR2 wrote output to {actual_out}")

            # 6. mp4 -> PNG sequence
            print(f"[SeedVR2-TTF] Extracting frames to {out_extracted_dir}")
            self._decode_mp4_to_pngs(actual_out, out_extracted_dir)

            # 7. PNG sequence -> tensor
            output_pngs = sorted(out_extracted_dir.glob("frame_*.png"))
            if not output_pngs:
                raise RuntimeError(f"No PNGs extracted from {actual_out}")
            print(f"[SeedVR2-TTF] Reading {len(output_pngs)} output frames")
            tensor_out = self._read_frames_from_png(output_pngs)

            print(f"[SeedVR2-TTF] Done: {tensor_out.shape}")
            return (tensor_out,)

        finally:
            try:
                shutil.rmtree(work_dir)
            except Exception as e:
                print(f"[SeedVR2-TTF] Warning: failed to clean up {work_dir}: {e}")

    # --- I/O helpers ---

    @staticmethod
    def _write_frames_as_png(images: torch.Tensor, out_dir: Path) -> None:
        arr = (images.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
        for i in range(arr.shape[0]):
            Image.fromarray(arr[i], mode="RGB").save(
                out_dir / f"frame_{i:06d}.png",
                compress_level=1,
            )

    @staticmethod
    def _read_frames_from_png(paths: List[Path]) -> torch.Tensor:
        frames = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            frames.append(np.asarray(img, dtype=np.float32) / 255.0)
        return torch.from_numpy(np.stack(frames, axis=0))

    @staticmethod
    def _encode_pngs_to_mp4(png_dir: Path, mp4_path: Path) -> None:
        cmd = [
            FFMPEG_BIN, "-y", "-loglevel", "error",
            "-framerate", str(INTERMEDIATE_FPS),
            "-i", str(png_dir / "frame_%06d.png"),
            "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
            "-pix_fmt", "yuv444p",
            str(mp4_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg PNG->mp4 failed: {result.stderr}")
        if not mp4_path.exists() or mp4_path.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg produced no output at {mp4_path}")

    @staticmethod
    def _decode_mp4_to_pngs(mp4_path: Path, out_dir: Path) -> None:
        cmd = [
            FFMPEG_BIN, "-y", "-loglevel", "error",
            "-i", str(mp4_path),
            "-start_number", "0",
            str(out_dir / "frame_%06d.png"),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg mp4->PNG failed: {result.stderr}")

    @staticmethod
    def _resolve_output_mp4(expected: Path, work_dir: Path) -> Optional[Path]:
        if expected.exists() and expected.stat().st_size > 0:
            return expected
        candidates = [
            p for p in work_dir.rglob("*.mp4")
            if p.name != "input.mp4" and p.stat().st_size > 0
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)


# --- ComfyUI registration ---

NODE_CLASS_MAPPINGS = {
    "SeedVR2RestorationUpscale": SeedVR2RestorationUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2RestorationUpscale": "SeedVR2 Restoration Upscale (subprocess)",
}