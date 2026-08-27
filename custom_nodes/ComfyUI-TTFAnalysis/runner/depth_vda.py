"""
Depth — Video-Depth-Anything, SMALL ONLY.

The encoder is hardcoded 'vits'. Base and Large weights are CC-BY-NC-4.0 while
the code is Apache-2.0 (see .dev/LICENSE_AUDIT.md); making the size a parameter
would put a non-commercial checkpoint one argument away in a pipeline that ships
in client deliverables. If a larger encoder is ever licensed, change it here
deliberately.

Streamed: one 32-frame window resident at a time, never a whole shot.
"""
import json

import cv2
import numpy as np
import torch

from . import paths
from .isolation import repo

DEV = torch.device("cuda")


def load(device=DEV):
    paths.check()
    with repo(paths.REPOS / "Video-Depth-Anything"):
        from video_depth_anything.video_depth import VideoDepthAnything
        from video_depth_anything.util.transform import (
            NormalizeImage, PrepareForNet, Resize)
        from utils.util import compute_scale_and_shift
        m = VideoDepthAnything(encoder="vits", features=64,
                               out_channels=[48, 96, 192, 384])
        m.load_state_dict(torch.load(paths.VDA_CHECKPOINT, map_location="cpu"),
                          strict=True)
        m = m.to(device).eval()
        helpers = dict(Resize=Resize, NormalizeImage=NormalizeImage,
                       PrepareForNet=PrepareForNet,
                       compute_scale_and_shift=compute_scale_and_shift)
    return m, helpers


def transform(helpers, input_size=518):
    from torchvision.transforms import Compose
    return Compose([
        helpers["Resize"](width=input_size, height=input_size,
                          resize_target=False, keep_aspect_ratio=True,
                          ensure_multiple_of=14, resize_method="lower_bound",
                          image_interpolation_method=cv2.INTER_CUBIC),
        helpers["NormalizeImage"](mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        helpers["PrepareForNet"](),
    ])


def plan_windows(n, win=paths.VDA_INFER_LEN, overlap=8):
    """Windows of exactly `win`, striding by win-overlap.

    The last window SLIDES BACK to end at n rather than being a short window:
    the packaged path pads anything under win up to a full window anyway, so a
    short tail costs the same either way. Sliding back turns those frames into
    already-computed overlap instead of a second full-price window.
    """
    if n <= win:
        return [(0, n, n)]
    stride = win - overlap
    starts = list(range(0, max(n - win, 0) + 1, stride))
    if starts[-1] != n - win:
        starts.append(n - win)
    out, covered = [], 0
    for s in starts:
        stop = s + win
        out.append((s, stop, max(0, stop - max(s, covered))))
        covered = max(covered, stop)
    return out


def stream(model, helpers, video, s0, s1, H, W, max_side, sink, tmp,
           input_size=518, overlap=8, on_block=None, device=DEV):
    """Per-shot depth into `sink` (float16) and `tmp` (float32, for normals).

    Windows are re-fitted to the previous one with the upstream least-squares
    scale+shift over the overlap. Chunking ourselves loses the packaged path's
    internal keyframe alignment, and without re-fitting, depth drifts between
    windows inside a single shot — which P2's shot-scoped normalization would
    then bake in permanently.
    """
    from .streaming import WindowFrameReader
    T = s1 - s0
    tf = transform(helpers, input_size)
    css = helpers["compute_scale_and_shift"]
    reader = WindowFrameReader(video, max_side)
    tail = {}
    dmin, dmax = np.inf, -np.inf
    try:
        for (a, b, _) in plan_windows(T, paths.VDA_INFER_LEN, overlap):
            frames = reader.window(s0 + a, s0 + b)
            assert frames is not None and len(frames), f"decode gap at {s0 + a}"
            proc = np.stack([tf({"image": f.astype(np.float32) / 255.0})["image"]
                             for f in frames])
            pad = paths.VDA_INFER_LEN - len(proc)
            if pad > 0:
                proc = np.concatenate([proc, np.repeat(proc[-1:], pad, 0)])
            x = torch.from_numpy(proc).unsqueeze(0).to(device)
            with torch.no_grad(), torch.autocast("cuda"):
                d = model(x)
            d = torch.nn.functional.interpolate(
                d.float().flatten(0, 1).unsqueeze(1), size=(H, W),
                mode="bilinear", align_corners=True)[:, 0].cpu().numpy()
            del x
            if pad > 0:
                d = d[:paths.VDA_INFER_LEN - pad]

            # Align against EVERY already-written frame this window re-covers,
            # not just the nominal overlap: the slid-back tail window can cover
            # up to win-1 of them, and fitting on fewer changes the scale.
            ov = [i for i in range(a, min(b, T)) if i in tail]
            if ov:
                pred = np.stack([d[i - a] for i in ov]).reshape(-1)
                tgt = np.stack([tail[i] for i in ov]).reshape(-1)
                s, sh = css(pred, tgt, np.ones_like(tgt, bool))
                d = d * s + sh
                d[d < 0] = 0          # upstream video_depth.py:147,152,158

            new = [i for i in range(a, min(b, T)) if i not in tail]
            if new:
                assert new == list(range(new[0], new[0] + len(new)))
                block = np.stack([d[i - a] for i in new])
                sink.write("depth", new[0], block.astype(np.float16))
                tmp.write("depth32", new[0], block)
                if on_block:
                    on_block(block)
                dmin = min(dmin, float(block.min()))
                dmax = max(dmax, float(block.max()))
                # only WRITTEN values enter history; a re-estimate of an
                # already-stored frame must never replace the stored one
                for i in new:
                    tail[i] = d[i - a]
            for i in [i for i in tail if i < min(b, T) - paths.VDA_INFER_LEN]:
                del tail[i]
            del d
    finally:
        reader.close()
    return dmin, dmax
