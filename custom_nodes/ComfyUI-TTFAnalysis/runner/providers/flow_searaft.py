"""
Optical flow — SEA-RAFT (BSD-3-Clause), spring-M.

Three display proxies, because they answer different questions:
  flow_mag     magnitude heatmap — reads best while SCRUBBING
  flow_wheel   direction wheel   — reads best on a PAUSED frame
  flow_arrows  sparse arrows     — reads direction at a glance without hue
"""
import numpy as np

from .. import flow_searaft as impl
from ..normalize import (
    flow_clamp, flow_to_arrows, flow_to_magnitude, flow_to_wheel,
)
from ..registry import (
    Cost, Dataset, Display, LicenseRow, Provider, ProxySpec, Schema, register,
)

# Measured peak at 1920x1340: 13304 MiB. This is the number the scheduler uses
# to refuse co-scheduling. Two of these on a 24 GB card is an OOM, and the
# point of putting it here is that the refusal happens in the planner rather
# than in the CUDA allocator halfway through a shot.
VRAM_MB = 13304


def _mag(frame, ctx):
    return flow_to_magnitude(frame, ctx.lo, ctx.hi,
                             ctx.get("depth_ramp", "gray"),
                             ctx.get("flow_map", "sqrt"))


def _wheel(frame, ctx):
    return flow_to_wheel(frame, ctx.lo, ctx.hi, ctx.get("flow_map", "sqrt"))


def _arrows(frame, ctx):
    return flow_to_arrows(frame, ctx.lo, ctx.hi,
                          grid=ctx.get("arrow_grid", 32),
                          mapping=ctx.get("flow_map", "sqrt"))


def _run(ctx):
    model, args = ctx.models["raft"]
    impl.stream(model, args, ctx.video, ctx.s0, ctx.s1, ctx.max_side, ctx.sink,
                on_frame=lambda f: ctx.record("flow", f, "vector2"))


register(Provider(
    name="flow",
    env="_env",
    depends_on=(),
    doc="Dense optical flow between consecutive frames, pixels/frame.",
    schema=Schema((Dataset("flow", ("T-1", 2, "H", "W"), "float16",
                           chunk_t=4, frames="T-1"),)),
    cost=Cost(vram_mb=VRAM_MB, window=2, gpu=True,
              note="one pair resident at a time; flat in clip length"),
    display=Display(
        dataset="flow",
        readout_kind="vector2",
        # chunks arrive as raw (t,2,H,W) flow; the clamp is a percentile of
        # MAGNITUDE, not of the signed components — percentiling x and y
        # separately would be a different and meaningless number.
        range_fn=lambda chunks: flow_clamp(
            (np.sqrt(c[:, 0].astype(np.float32) ** 2
                     + c[:, 1].astype(np.float32) ** 2) for c in chunks),
            pct=99.0),
        proxies=(
            ProxySpec("flow_mag", _mag,
                      "magnitude heatmap; brightness change over time is what "
                      "the eye tracks while scrubbing"),
            ProxySpec("flow_wheel", _wheel,
                      "hue = direction, value = magnitude; best on a paused "
                      "frame where hues can be compared side by side"),
            ProxySpec("flow_arrows", _arrows,
                      "sparse arrows on a fixed pixel grid; direction without "
                      "needing to decode a hue wheel"),
        ),
    ),
    license=LicenseRow(
        model="SEA-RAFT (Tartan-C-T-TSKH-spring540x960-M)",
        code_license="BSD-3-Clause",
        weight_license="BSD-3-Clause",
        gated=False,
        verdict="CLEAN",
        source_url="https://github.com/princeton-vl/SEA-RAFT/blob/9137517ba24e628442aec097d3afe71d03503b75/LICENSE",
        date_checked="2026-08-26",
        note="Retain copyright notice + disclaimer; do not use Princeton's "
             "name to endorse.",
    ),
    run=_run,
))
