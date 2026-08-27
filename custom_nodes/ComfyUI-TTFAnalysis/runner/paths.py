"""
Where the vendored model source and weights live.

The runner must stay runnable standalone — no ComfyUI import anywhere at module
level — so locations come from the environment with a repo-relative default,
never from folder_paths. P3's node wrapper sets these; nothing here depends on it.

  TTF_ANALYSIS_REPOS   dir holding Video-Depth-Anything/ and SEA-RAFT/
  TTF_ANALYSIS_CKPT    dir holding video_depth_anything_vits.pth
"""
import os
from pathlib import Path

# Default to the .dev probe locations so a checkout runs without setup. P3
# repoints these at the vendored copies baked into the image.
_DEFAULT_ROOT = Path(__file__).resolve().parents[3] / ".dev" / "analysis-probe"

REPOS = Path(os.environ.get("TTF_ANALYSIS_REPOS", _DEFAULT_ROOT / "repos"))
CKPT = Path(os.environ.get("TTF_ANALYSIS_CKPT", _DEFAULT_ROOT / "checkpoints"))

VDA_CHECKPOINT = CKPT / "video_depth_anything_vits.pth"
SEARAFT_HF = "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"
SEARAFT_CFG = "config/eval/spring-M.json"

# NOTE: VDA's 32-frame window used to live here as VDA_INFER_LEN and was read
# directly by the chunker. It now lives on the provider as Cost.window, because
# it is a property of that model rather than a global of the runner.


def check():
    missing = []
    for p in (REPOS / "Video-Depth-Anything", REPOS / "SEA-RAFT", VDA_CHECKPOINT):
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            "missing analysis assets:\n  " + "\n  ".join(missing) +
            "\nSet TTF_ANALYSIS_REPOS / TTF_ANALYSIS_CKPT.")
