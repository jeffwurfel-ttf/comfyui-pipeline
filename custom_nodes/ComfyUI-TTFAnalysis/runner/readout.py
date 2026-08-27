"""
Per-frame min/max/mean plus anomaly flags — the correctness gate.

Statistics accumulate as frames are written, so the readout costs no extra pass.
Checks are signal-specific on purpose: a jump in the frame mean is the flicker
symptom for depth, whose scale must stay put inside a shot, and is meaningless
for flow, where magnitude legitimately swings by hundreds of percent as things
start and stop moving. Applying the flicker test to flow produced 73 false flags
on the first P1 run and buried the two real ones.
"""
import numpy as np


class FrameStats:
    def __init__(self):
        self.rows = []
        self.raw = []      # UNROUNDED (min,max). `rows` is rounded for the
                           # manifest and must never be used for tests.

    def add(self, arr, kind):
        for fr in arr:
            if kind == "flow":
                v = np.sqrt(fr[0].astype(np.float64) ** 2
                            + fr[1].astype(np.float64) ** 2)
            elif kind == "normals":
                v = np.linalg.norm(fr.astype(np.float64), axis=0)
            else:
                v = fr.astype(np.float64)
            self.raw.append((float(v.min()), float(v.max())))
            self.rows.append({"frame": len(self.rows),
                              "min": round(float(v.min()), 5),
                              "max": round(float(v.max()), 5),
                              "mean": round(float(v.mean()), 5)})


def flags_for(rows, kind, shape_hw=None, raw=None):
    fl = []
    means = np.array([r["mean"] for r in rows]) if rows else np.array([])
    mins = np.array([r["min"] for r in rows]) if rows else np.array([])
    maxs = np.array([r["max"] for r in rows]) if rows else np.array([])
    for i, r in enumerate(rows):
        if not np.isfinite([r["min"], r["max"], r["mean"]]).all():
            fl.append(f"frame {i}: non-finite values")
        # Degeneracy is tested on UNROUNDED extrema. Testing the 5dp manifest
        # values instead flagged every normals frame, since |n| is 1.0 by
        # construction and rounds to a zero spread. The check is meaningless
        # for normals anyway: constant |n| is the desired property there.
        if kind != "normals":
            lo_i, hi_i = (raw[i] if raw else (r["min"], r["max"]))
            if hi_i - lo_i < 1e-8:
                fl.append(f"frame {i}: constant field (degenerate)")
    if kind == "depth":
        if len(means) > 1:
            d = np.abs(np.diff(means)) / np.maximum(np.abs(means[:-1]), 1e-6)
            for j in np.where(d > 0.25)[0]:
                fl.append(f"frame {j}->{j+1}: depth mean jumped {d[j]*100:.0f}% "
                          f"({means[j]:.4f} -> {means[j+1]:.4f})")
        neg = int((mins < -1e-3).sum())
        if neg:
            fl.append(f"{neg} frame(s) with negative inverse-depth")
    elif kind == "normals" and len(means):
        if np.abs(means - 1.0).max() > 0.02:
            fl.append(f"unit-length violation: |n| {means.min():.4f}..{means.max():.4f}")
    elif kind == "flow" and shape_hw and len(maxs):
        diag = float(np.hypot(*shape_hw))
        for j in np.where(maxs > 0.5 * diag)[0][:5]:
            fl.append(f"frame {j}: max |flow| {maxs[j]:.1f}px exceeds half the "
                      f"frame diagonal ({0.5*diag:.0f}px)")
    return fl
