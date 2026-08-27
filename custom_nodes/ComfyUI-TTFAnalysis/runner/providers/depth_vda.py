"""
Depth — Video-Depth-Anything, SMALL ONLY.

Note: `runner.providers.depth_vda` (this declaration) is distinct from
`runner.depth_vda` (the model implementation it wraps). The provider is
metadata plus a thin run(); the numerics stay where they were so this refactor
cannot move them.
"""
import numpy as np

from .. import depth_vda as impl
from ..normalize import apply_range, ramp, shot_range
from ..registry import (
    Cost, Dataset, Display, LicenseRow, Provider, ProxySpec, Schema, register,
)

# The 32 is VDA's INFER_LEN (video_depth.py:29) and it lives HERE, not in the
# chunker: the packaged path pads any clip shorter than 32 up to 32, so a
# 16-frame chunk costs exactly what a 32-frame one does. That is a property of
# this model, not of chunking in general.
WINDOW = 32
OVERLAP = 8

# Measured on the 398-frame 1920x1340 shot. Aspect-ratio driven, not
# resolution driven: VDA normalises the SHORT side to 518, so 1280x720
# (-> 518x921) actually peaks higher than 1920x1340 (-> 518x742).
VRAM_MB = 7887


def _render_depth(frame, ctx):
    return ramp(apply_range(frame, ctx.lo, ctx.hi), ctx.get("depth_ramp", "gray"))


def _run(ctx):
    model, helpers = ctx.models["vda"]
    dmin, dmax = impl.stream(
        model, helpers, ctx.video, ctx.s0, ctx.s1, ctx.H, ctx.W, ctx.max_side,
        ctx.sink, ctx.state["depth32_sink"],
        ctx.param("input_size", 518), ctx.param("overlap", OVERLAP),
        window=WINDOW,
        on_block=lambda b: ctx.record("depth", b, "scalar"))
    ctx.sink.attrs("depth", min=dmin, max=dmax)
    ctx.state["depth_minmax"] = (dmin, dmax)


register(Provider(
    name="depth",
    env="_env",
    depends_on=(),
    doc="Temporally consistent relative inverse depth (larger = nearer).",
    schema=Schema((Dataset("depth", ("T", "H", "W"), "float16", chunk_t=8),)),
    cost=Cost(vram_mb=VRAM_MB, window=WINDOW, overlap=OVERLAP, gpu=True,
              note="peak tracks aspect ratio, not source resolution"),
    display=Display(
        dataset="depth",
        readout_kind="scalar",
        range_fn=lambda chunks: shot_range(chunks, mode="percentile"),
        proxies=(ProxySpec("depth", _render_depth,
                           "grayscale by default; inferno as a toggle. Never "
                           "turbo/jet — rainbow ramps destroy the gradient "
                           "legibility that is the point of a depth proxy."),),
    ),
    license=LicenseRow(
        model="Video-Depth-Anything (vits / Small)",
        code_license="Apache-2.0",
        weight_license="Apache-2.0 (Small ONLY — Base/Large are CC-BY-NC-4.0)",
        gated=False,
        verdict="CLEAN",
        source_url="https://github.com/DepthAnything/Video-Depth-Anything/blob/4f5ae23172ba60fd7bc11ef671cca678842c7072/LICENSE",
        date_checked="2026-08-26",
        note="Encoder is hardcoded 'vits'. Making the size a parameter would "
             "put a non-commercial checkpoint one argument away.",
    ),
    run=_run,
))
