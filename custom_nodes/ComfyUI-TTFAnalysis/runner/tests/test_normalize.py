#!/usr/bin/env python
"""
P2 unit tests on synthetic fields. No GPU, no model, no real footage.

Run:  python -m runner.tests.test_normalize

The four required fields, and what each is actually pinning down:
  ramp        monotone geometry survives normalization
  step        a discontinuity stays a discontinuity, and the edge does not bleed
  spike       ONE bad frame must not re-scale the other frames  <- the flicker bug
  constant    hi == lo must not divide by zero
"""
import sys

import numpy as np

from ..normalize import (
    BANNED_RAMPS, apply_range, bilateral_depth, flow_clamp, flow_to_magnitude,
    flow_to_wheel, normals_to_rgb, ramp, shot_range,
)

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'ok ' if cond else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    if not cond:
        FAIL.append(name)


# ───────────────────────────────────────────────────────────── fields
def f_ramp(T=8, H=32, W=64):
    x = np.linspace(0, 1, W, dtype=np.float32)[None, None, :]
    return np.repeat(np.repeat(x, H, 1), T, 0)


def f_step(T=8, H=32, W=64):
    a = np.zeros((T, H, W), np.float32); a[:, :, W // 2:] = 1.0
    return a


def f_spike(T=128, H=32, W=64):
    # 128 frames, ONE blown. A single bad frame is 1/T of the samples, so a
    # percentile can only reject it if 100-pct > 100/T. At T=8 (the first cut of
    # this test) one frame is 12.5% of the data and NO sane percentile excludes
    # it — the test failed for want of a realistic shot length, not a bug.
    a = f_ramp(T, H, W).copy(); a[3] *= 50.0
    return a


def f_const(T=8, H=32, W=64, v=0.7):
    return np.full((T, H, W), v, np.float32)


def main():
    print("P2 normalize — synthetic field tests\n")

    # ── ramp ────────────────────────────────────────────────────────────
    print("ramp:")
    r = f_ramp()
    lo, hi = shot_range(r, mode="minmax")
    n = apply_range(r, lo, hi)
    check("range is shot-scoped", (lo, hi) == (0.0, 1.0), f"{lo},{hi}")
    check("maps to [0,1]", n.min() == 0.0 and n.max() == 1.0)
    check("monotone preserved", bool(np.all(np.diff(n[0, 0]) >= 0)))
    check("all frames identical after norm", bool(np.allclose(n[0], n[-1])),
          "no per-frame drift on a static field")

    # ── step ────────────────────────────────────────────────────────────
    print("\nstep discontinuity:")
    s = f_step()
    lo, hi = shot_range(s, mode="minmax")
    n = apply_range(s, lo, hi)
    W = s.shape[2]
    check("step preserved", n[0, 0, W // 2 - 1] == 0.0 and n[0, 0, W // 2] == 1.0)
    check("no intermediate values", set(np.unique(n).tolist()) == {0.0, 1.0},
          "normalization must not blur an edge")
    bf = bilateral_depth(s, strength=1.0)
    edge_jump = float(bf[0, 16, W // 2] - bf[0, 16, W // 2 - 1])
    check("bilateral keeps the edge", edge_jump > 0.5, f"jump={edge_jump:.3f}")

    # ── spike — the flicker test ────────────────────────────────────────
    print("\nsingle-frame spike (the flicker case):")
    sp = f_spike()
    lo_m, hi_m = shot_range(sp, mode="minmax")
    lo_p, hi_p = shot_range(sp, mode="percentile", pct=(1.0, 99.0))
    n_m = apply_range(sp, lo_m, hi_m)
    n_p = apply_range(sp, lo_p, hi_p)
    good_m = float(n_m[0].max())
    good_p = float(n_p[0].max())
    check("minmax IS corrupted by the spike", good_m < 0.05,
          f"clean frame peaks at {good_m:.4f} of the ramp — crushed")
    check("percentile is robust", good_p > 0.9,
          f"clean frame peaks at {good_p:.4f}")
    # the real invariant: normalizing PER FRAME would make every frame identical
    per_frame = np.stack([apply_range(f, f.min(), f.max()) for f in sp])
    check("per-frame norm erases the spike (why it is banned)",
          bool(np.allclose(per_frame[3], per_frame[0])),
          "frame 3 is 50x hotter yet looks identical — flicker source")
    check("shot-scoped norm keeps frames distinguishable",
          not np.allclose(n_p[3], n_p[0]))

    # ── constant ────────────────────────────────────────────────────────
    print("\nconstant field (divide-by-zero):")
    c = f_const()
    lo, hi = shot_range(c, mode="minmax")
    n = apply_range(c, lo, hi)
    check("no NaN/Inf", bool(np.isfinite(n).all()), f"lo={lo} hi={hi}")
    check("maps to 0.0", bool((n == 0).all()))
    check("percentile path also safe",
          bool(np.isfinite(apply_range(c, *shot_range(c, "percentile"))).all()))
    z = np.zeros((4, 2, 8, 8), np.float32)
    check("flow clamp on all-zero motion is safe",
          bool(np.isfinite(flow_to_magnitude(z[0], *flow_clamp(
              [np.sqrt(z[:, 0] ** 2 + z[:, 1] ** 2)])).astype(np.float32)).all()))

    # ── flow ────────────────────────────────────────────────────────────
    print("\nflow:")
    T, H, W = 128, 16, 32            # realistic shot length, see f_spike
    fl = np.zeros((T, 2, H, W), np.float32)
    fl[:, 0] = 1.0
    fl[2, 0] = 200.0                                  # one whip-pan frame
    mag = np.sqrt(fl[:, 0] ** 2 + fl[:, 1] ** 2)
    lo_c, hi_c = flow_clamp([mag], pct=99.0)
    lo_x, hi_x = 0.0, float(mag.max())
    quiet_p = float(apply_range(mag[0], lo_c, hi_c).mean())
    quiet_x = float(apply_range(mag[0], lo_x, hi_x).mean())
    check("p99 clamp keeps quiet frames visible", quiet_p > 0.5, f"{quiet_p:.3f}")
    check("max-scaling crushes them", quiet_x < 0.02,
          f"{quiet_x:.4f} — one whip-pan blacks out the shot")
    check("flow lower bound pinned at 0", lo_c == 0.0,
          "zero motion must map to the bottom of the ramp")
    wheel = flow_to_wheel(fl[0], lo_c, hi_c)
    heat = flow_to_magnitude(fl[0], lo_c, hi_c)
    check("wheel is RGB uint8", wheel.shape == (H, W, 3) and wheel.dtype == np.uint8)
    check("heatmap is RGB uint8", heat.shape == (H, W, 3) and heat.dtype == np.uint8)
    r_fl = np.zeros((2, H, W), np.float32); r_fl[0] = 5.0
    l_fl = np.zeros((2, H, W), np.float32); l_fl[0] = -5.0
    check("wheel encodes direction (L != R hue)",
          flow_to_wheel(r_fl, 0, 5)[0, 0, 0] != flow_to_wheel(l_fl, 0, 5)[0, 0, 0])

    # ── ramps ───────────────────────────────────────────────────────────
    print("\nramps:")
    g = ramp(np.linspace(0, 1, 16, dtype=np.float32), "gray")
    check("gray is neutral", bool((g[..., 0] == g[..., 1]).all() and (g[..., 1] == g[..., 2]).all()))
    check("inferno available", ramp(np.linspace(0, 1, 16, dtype=np.float32), "inferno").shape == (16, 3))
    for bad in ("turbo", "jet"):
        try:
            ramp(np.zeros(4, np.float32), bad); ok = False
        except ValueError:
            ok = True
        check(f"{bad} refused", ok)
    check("banned list covers rainbow family", {"turbo", "jet"} <= BANNED_RAMPS)

    # ── normals ─────────────────────────────────────────────────────────
    print("\nnormals:")
    n3 = np.zeros((3, 8, 8), np.float32); n3[2] = -1.0
    rgb = normals_to_rgb(n3)
    check("tangent-space mapping (0,0,-1) -> (128,128,0)",
          tuple(rgb[0, 0]) == (128, 128, 0), str(tuple(rgb[0, 0])))
    n3b = np.zeros((3, 8, 8), np.float32); n3b[0] = 1.0
    check("(+1,0,0) -> R=255", normals_to_rgb(n3b)[0, 0, 0] == 255)
    check("output is uint8 HWC", rgb.dtype == np.uint8 and rgb.shape == (8, 8, 3))

    print(f"\n{'ALL PASSED' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
