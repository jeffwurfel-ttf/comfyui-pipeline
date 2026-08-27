"""
Optical flow — SEA-RAFT (BSD-3-Clause), spring-M config.

Streamed: two frames resident at a time. Nothing accumulates.
"""
import json
from types import SimpleNamespace

import numpy as np
import torch

from . import paths
from .isolation import repo

DEV = torch.device("cuda")


def load(device=DEV):
    paths.check()
    root = paths.REPOS / "SEA-RAFT"
    with repo(root, root / "core"):
        from raft import RAFT
        cfg = json.loads((root / paths.SEARAFT_CFG).read_text(encoding="utf-8"))
        args = SimpleNamespace(**cfg)
        m = RAFT.from_pretrained(paths.SEARAFT_HF, args=args).to(device).eval()
    return m, args


def stream(model, args, video, s0, s1, max_side, sink, on_frame=None, device=DEV):
    """Consecutive pairs within a shot -> sink['flow'], one pair at a time.

    T source frames give T-1 fields. RAFT pads AND unpads internally
    (core/raft.py:92,131) and returns a dict, so it must not be wrapped in
    another InputPadder and the field to take is r['final'].
    """
    from .streaming import WindowFrameReader
    reader = WindowFrameReader(video, max_side)
    try:
        prev = None
        for i in range(s0, s1):
            f = reader.window(i, i + 1)
            if f is None or not len(f):
                break
            cur = torch.from_numpy(f).permute(0, 3, 1, 2).float().to(device)
            if prev is not None:
                with torch.no_grad():
                    r = model(prev, cur, iters=args.iters, test_mode=True)
                fl = r["final"][0].cpu().numpy()
                sink.write("flow", i - s0 - 1, fl[None])
                if on_frame:
                    on_frame(fl[None])
                del r, fl
            del prev
            prev = cur
        del prev
    finally:
        reader.close()
