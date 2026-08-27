"""
P2 — normalization and colorization. Pure CPU, no GPU, no model.

This is where flicker is either solved or introduced. The single rule that
matters: **normalize across the SHOT, never per frame.** VDA spends its whole
temporal-attention budget making depth consistent frame to frame; per-frame
min/max normalization throws that away in the last step and hands back exactly
the flicker the model removed. Same for flow — a per-frame scale makes a static
frame look as energetic as a whip-pan.

Ranges are returned alongside the pixels and belong in the manifest. Without the
range a proxy cannot be inverted back to data and the display is a dead end.

Nothing here writes colour without being told a range computed over the shot.
"""
import subprocess
from pathlib import Path

import cv2
import numpy as np

EPS = 1e-6
BANNED_RAMPS = {"turbo", "jet", "rainbow", "hsv", "nipy_spectral", "gist_rainbow"}


# ────────────────────────────────────────────────────────────── ranges
def shot_range(chunks, mode="percentile", pct=(1.0, 99.0)):
    """Shot-scoped (lo, hi) for a scalar field.

    `chunks` is an iterable of arrays — the whole shot need never be resident.
    Percentile is the default rather than min/max because a single corrupt or
    flashed frame otherwise sets the range for every other frame in the shot.
    That is not hypothetical: frame 1 of the mountaindemo clip is a dark frame
    whose depth max is 14.85 against a shot-typical 9.5, so plain min/max
    compresses all 397 good frames to accommodate one bad one.

    The default is (1, 99), not something tighter like (0.1, 99.9). One bad
    frame in a shot of N is 1/N of the samples, so the upper percentile has to
    sit BELOW 100 - 100/N to exclude it at all: at N=398 that is 99.75, and
    p99.9 would still land inside the bad frame. p99 rejects a single outlier
    frame for any shot longer than ~100 frames, which is the regime that
    matters, at the cost of clipping 1% at each end.
    """
    if mode == "minmax":
        lo, hi = np.inf, -np.inf
        for c in chunks:
            if c.size:
                lo = min(lo, float(np.nanmin(c))); hi = max(hi, float(np.nanmax(c)))
        if not np.isfinite(lo):
            return 0.0, 1.0
        return lo, hi
    from .streaming import exact_quantiles
    mats, n = [], 0
    for c in chunks:                       # materialise refs, not copies
        mats.append(np.asarray(c).reshape(-1)); n += mats[-1].size
    q = exact_quantiles(lambda: iter(mats), n, pct)
    return float(q[pct[0]]), float(q[pct[1]])


def flow_clamp(chunks, pct=99.0):
    """Upper clamp for flow MAGNITUDE, shot-scoped.

    p99 not max: one whip-pan frame with 250 px displacement otherwise crushes
    every ordinary frame in the shot to near-black. Lower bound is fixed at 0 —
    zero motion is a real, meaningful value and must map to the bottom of the
    ramp, not to whatever the quietest frame happens to be.
    """
    mats, n = [], 0
    for c in chunks:
        mats.append(np.asarray(c).reshape(-1)); n += mats[-1].size
    if not n:
        return 0.0, 1.0
    from .streaming import exact_quantiles
    q = exact_quantiles(lambda: iter(mats), n, (pct,))
    return 0.0, max(float(q[pct]), EPS)


def apply_range(x, lo, hi):
    """Map to [0,1] with the shot range. A constant field (hi==lo) maps to 0,
    never NaN — the degenerate case the unit tests pin down."""
    span = hi - lo
    if not np.isfinite(span) or span <= EPS:
        return np.zeros_like(x, np.float32)
    return np.clip((x.astype(np.float32) - lo) / span, 0.0, 1.0)


# ───────────────────────────────────────────────────────────── colorize
def ramp(x01, name="gray"):
    """x01 in [0,1] -> uint8 RGB. Rainbow ramps are refused, not just
    discouraged: they destroy gradient legibility, which is the entire point of
    a depth proxy, and they read as banding on smooth surfaces."""
    if name in BANNED_RAMPS:
        raise ValueError(
            f"ramp {name!r} is banned for analysis proxies — rainbow ramps "
            f"break gradient legibility. Use 'gray' (default) or 'inferno'.")
    x = np.clip(x01, 0, 1)
    if name == "gray":
        g = (x * 255).astype(np.uint8)
        return np.stack([g, g, g], -1)
    import matplotlib
    cm = matplotlib.colormaps[name]
    return (cm(x)[..., :3] * 255).astype(np.uint8)


def normals_to_rgb(n):
    """Standard tangent-space normal-map encoding: (n+1)/2 per component.
    n is (3,H,W) or (H,W,3) unit vectors."""
    if n.ndim == 3 and n.shape[0] == 3:
        n = n.transpose(1, 2, 0)
    # rint, not truncation: (n+1)/2*255 puts a zero component at 127.5, and
    # casting straight to uint8 floors every component, biasing the whole map
    # down by up to 1 LSB and putting "flat" at 127 instead of the conventional
    # 128 that normal-map tooling expects.
    v = np.rint((n.astype(np.float32) + 1.0) * 0.5 * 255.0)
    return np.clip(v, 0, 255).astype(np.uint8)


FLOW_MAPS = ("linear", "sqrt", "log")
LOG_K = 32.0


def magnitude_curve(x01, mapping="sqrt", k=LOG_K):
    """Tone curve for flow magnitude, applied AFTER the shot-scoped clamp.

    The p99 clamp fixes the outlier problem but not the distribution problem:
    flow magnitude is heavy-tailed, so under a linear map a shot containing a
    real 74 px/frame move pushes all ordinary motion into the bottom few percent
    of the ramp and it reads as black. The clamp is not at fault — the linear
    map is.

    All three curves are monotone and pin both ends (0->0, 1->1), so the big
    move still saturates and nothing is reordered; only the mid-tones lift.
      linear  x
      sqrt    x**0.5          moderate lift, the default
      log     log1p(kx)/log1p(k)   aggressive lift for very heavy tails
    """
    x = np.clip(x01, 0.0, 1.0)
    if mapping == "linear":
        return x
    if mapping == "sqrt":
        return np.sqrt(x)
    if mapping == "log":
        return np.log1p(x * k) / np.log1p(k)
    raise ValueError(f"unknown flow mapping {mapping!r}; expected {FLOW_MAPS}")


def flow_magnitude(flow):
    f = flow.transpose(1, 2, 0) if flow.shape[0] == 2 else flow
    return np.sqrt(f[..., 0].astype(np.float32) ** 2
                   + f[..., 1].astype(np.float32) ** 2)


def flow_to_magnitude(flow, lo, hi, name="gray", mapping="sqrt", k=LOG_K):
    """Magnitude heatmap. Reads better than the wheel while scrubbing, because
    the eye tracks brightness change over time far better than hue change."""
    mag = flow_magnitude(flow)
    return ramp(magnitude_curve(apply_range(mag, lo, hi), mapping, k), name)


def flow_to_wheel(flow, lo, hi, mapping="sqrt", k=LOG_K):
    """Direction wheel: hue = direction, value = clamped magnitude.
    Reads better than the heatmap on a PAUSED frame, where the eye can compare
    hues side by side. Both are emitted; they answer different questions.

    The value channel takes the same tone curve as the heatmap — otherwise the
    two proxies disagree about how energetic the same frame is."""
    f = flow.transpose(1, 2, 0) if flow.shape[0] == 2 else flow
    fx, fy = f[..., 0].astype(np.float32), f[..., 1].astype(np.float32)
    mag = np.sqrt(fx ** 2 + fy ** 2)
    ang = np.arctan2(-fy, -fx) + np.pi                     # 0..2pi
    hsv = np.zeros(mag.shape + (3,), np.uint8)
    hsv[..., 0] = (ang * 90.0 / np.pi).astype(np.uint8)    # OpenCV hue 0..179
    hsv[..., 1] = 255
    hsv[..., 2] = (magnitude_curve(apply_range(mag, lo, hi), mapping, k)
                   * 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def bilateral_normals(n, strength=1.0, d_px=15, sigma_color=0.15):
    """POST-gradient filter: smooths the normal VECTORS, then renormalizes.

    Kept for comparison against the pre-gradient path, not because it is
    expected to win. By the time normals exist, depth noise has already been
    amplified by differentiation and is no longer separable from real surface
    detail — a filter here cannot tell them apart, whereas the same filter on
    depth still can, because there the noise is small and the edges are large.
    n is (t,3,H,W); returns the same shape, unit length.
    """
    if strength <= 0:
        return n
    dd = int(max(3, round(float(d_px) * strength)))
    sc = float(sigma_color) * strength
    out = np.empty_like(n, np.float32)
    for i in range(len(n)):
        img = np.ascontiguousarray(n[i].transpose(1, 2, 0), np.float32)
        f = cv2.bilateralFilter(img, dd, sc, max(dd, 1.0))
        ln = np.linalg.norm(f, axis=-1, keepdims=True)
        out[i] = (f / np.maximum(ln, 1e-8)).transpose(2, 0, 1)
    return out


def bilateral_depth(d, strength=1.0, d_px=15, sigma_color=0.08):
    """Edge-preserving pre-filter applied to DEPTH before differentiating.

    Measured in P1: background normal roughness 0.119 -> 0.061 while subject
    detail survives, because the filter smooths within surfaces and stops at
    depth edges. strength scales both the spatial and range sigmas; 0 disables.
    """
    if strength <= 0:
        return d
    dd = float(d_px) * strength
    sc = float(sigma_color) * strength
    out = np.empty_like(d, np.float32)
    for i, f in enumerate(np.atleast_3d(d.astype(np.float32)) if d.ndim == 2 else d):
        out[i] = cv2.bilateralFilter(np.ascontiguousarray(f, np.float32),
                                     int(max(3, round(dd))), sc, max(dd, 1.0))
    return out


# ─────────────────────────────────────────────────────────── mp4 writer
class Mp4Writer:
    """H.264 via ffmpeg stdin. Frame count and fps are asserted on close —
    Coyote scrubs these against the source in lockstep and blends them as
    variable-opacity overlays, so an off-by-one silently misattributes every
    signal to the wrong frame."""

    def __init__(self, path, w, h, fps, crf=18):
        self.path, self.n, self.expect = Path(path), 0, None
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", f"{fps}", "-i", "-", "-an",
             "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
             "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE)

    def write(self, rgb):
        self.proc.stdin.write(np.ascontiguousarray(rgb, np.uint8).tobytes())
        self.n += 1

    def close(self, expect=None):
        self.proc.stdin.close()
        rc = self.proc.wait()
        err = self.proc.stderr.read().decode()[-800:]
        assert rc == 0, f"ffmpeg failed ({rc}) for {self.path}: {err}"
        if expect is not None:
            assert self.n == expect, (
                f"{self.path.name}: wrote {self.n} frames, source has {expect}. "
                f"Proxies must match the source frame count exactly.")
        return self.n


def probe_frames(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,r_frame_rate",
         "-of", "csv=p=0", str(path)], text=True).strip().split(",")
    fr = [x for x in out if "/" in x]
    nb = [x for x in out if x.isdigit()]
    rate = eval(fr[0]) if fr else None                      # noqa: S307 (a/b)
    return (int(nb[0]) if nb else None), rate
