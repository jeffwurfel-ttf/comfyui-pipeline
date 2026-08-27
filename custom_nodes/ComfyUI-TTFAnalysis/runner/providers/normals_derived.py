"""
Surface normals — DERIVED from depth by gradient cross-product. Our code, no
model, no licence obligation. DSINE was rejected on licence in P0.

This is the provider that makes `depends_on` load-bearing. The dependency on
depth was previously implicit: proxies.py just happened to read the depth
dataset when rendering normals, and cli.py just happened to run depth first.
Declaring it means the scheduler enforces the ordering and a normals-only
request pulls depth in automatically instead of failing on a missing dataset.
"""
import numpy as np

from ..normalize import bilateral_depth, normals_to_rgb, shot_range
from ..registry import (
    Cost, Dataset, Display, LicenseRow, Provider, ProxySpec, Schema, register,
)
from ..streaming import exact_quantiles, h5_chunks

CHUNK = 16

# Elementwise math on a depth chunk; no model weights, no correlation volume.
VRAM_MB = 900


def _render_normals(frame, ctx):
    return normals_to_rgb(frame)


def _derive(chunk, ctx):
    """Display-time re-derivation from FILTERED depth.

    The bilateral filter acts on depth BEFORE differentiation — that is the
    edge-preserving use it is designed for, and measurement confirmed the
    post-gradient placement is a near no-op (5.7% vs 51% background noise
    reduction). Keeping it here means the STORED normals stay unfiltered (data)
    while the filter stays a display parameter (presentation).
    """
    from ..normals_derived import normals_from_depth_cpu
    s = ctx.get("bilateral", 1.0)
    d = bilateral_depth(chunk.astype(np.float32), s) if s > 0 else chunk.astype(np.float32)
    n, _ = normals_from_depth_cpu(d, ctx.lo, ctx.hi, ctx.get("fov", 60.0))
    return n


def _run(ctx):
    from ..normals_derived import normals_from_depth_gpu
    tmp = ctx.state["depth32_path"]
    n = ctx.n_frames
    q = exact_quantiles(
        lambda: (c.reshape(-1) for _, c in h5_chunks(tmp, "depth32", 8)),
        n * ctx.H * ctx.W, (1, 99))
    lo, hi = q[1], q[99]
    ctx.state["normals_percentiles"] = {"p1": lo, "p99": hi}
    vs, vn = 0.0, 0
    for start, chunk in h5_chunks(tmp, "depth32", CHUNK):
        nr, m = normals_from_depth_gpu(chunk, lo, hi, ctx.param("fov", 60.0))
        ctx.sink.write("normals", start, nr.astype(np.float16))
        ctx.sink.write("normals_valid", start, m)
        ctx.record("normals", nr, "unit3")
        vs += float(m.sum()); vn += m.size
        del nr, m
    ctx.state["normals_valid_frac"] = vs / max(vn, 1)


register(Provider(
    name="normals",
    env="_env",
    depends_on=("depth",),
    doc="Unit surface normals from the depth gradient, plus a validity mask "
        "marking depth discontinuities.",
    schema=Schema((
        Dataset("normals", ("T", 3, "H", "W"), "float16", chunk_t=4),
        Dataset("normals_valid", ("T", "H", "W"), "uint8", chunk_t=8),
    )),
    cost=Cost(vram_mb=VRAM_MB, window=CHUNK, gpu=True),
    display=Display(
        dataset="normals",
        readout_kind="unit3",
        derive_from="depth",
        derive=_derive,
        range_fn=lambda chunks: shot_range(chunks, mode="percentile"),
        proxies=(ProxySpec("normals", _render_normals,
                           "standard tangent-space (n+1)/2 RGB"),),
    ),
    license=LicenseRow(
        model="derived (gradient cross-product)",
        code_license="ours — no third-party code",
        weight_license="n/a — no weights",
        gated=False,
        verdict="CLEAN",
        source_url="internal: runner/normals_derived.py",
        date_checked="2026-08-26",
        note="Replaces DSINE, which is BLOCKED for commercial use (Imperial "
             "College bespoke licence). See .dev/LICENSE_AUDIT.md.",
    ),
    run=_run,
))
