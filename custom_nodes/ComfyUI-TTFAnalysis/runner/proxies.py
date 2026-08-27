"""
P2 proxy rendering — generic. Pure CPU.

There are no per-signal branches here. This module knows how to: resolve a
shot-scoped range, walk a dataset in chunks, optionally run a display-time
derive hook, and hand frames to renderers. WHICH range, WHICH derive and WHICH
renderers all come off `provider.display`.

Adding a signal must not require editing this file. If it does, the Display
contract is missing something — extend that, not this.

Every range used is recorded into proxies.json: without it a proxy cannot be
inverted back to the underlying data and the display is a dead end.
"""
import json
from pathlib import Path

import numpy as np

from .normalize import Mp4Writer, probe_frames

CH = 16


def log(m):
    print(f"[p2] {m}", flush=True)


class DisplayCtx:
    """What a renderer or derive hook is allowed to see: the shot-scoped range
    and the display parameters. Deliberately NOT the frame index — a renderer
    that could see time could compute a per-frame range, which is the flicker
    bug this whole layer exists to prevent."""

    __slots__ = ("lo", "hi", "params")

    def __init__(self, lo, hi, params):
        self.lo, self.hi, self.params = lo, hi, dict(params or {})

    def get(self, k, default=None):
        return self.params.get(k, default)


class ShotData:
    """Uniform lazy access over HDF5 or P1's npz."""

    def __init__(self, shot_dir):
        self.dir = Path(shot_dir)
        self.h5 = self.dir / "signals.h5"
        self.mode = "h5" if self.h5.exists() else "npz"

    def has(self, name):
        if self.mode == "h5":
            import h5py
            with h5py.File(self.h5, "r") as f:
                return name in f
        return (self.dir / f"{name}.npz").exists()

    def shape(self, name):
        if self.mode == "h5":
            import h5py
            with h5py.File(self.h5, "r") as f:
                return f[name].shape
        z = np.load(self.dir / f"{name}.npz")
        return z[z.files[0]].shape

    def chunks(self, name, ch=CH):
        if self.mode == "h5":
            import h5py
            with h5py.File(self.h5, "r") as f:
                d = f[name]
                for i in range(0, d.shape[0], ch):
                    yield i, np.asarray(d[i:i + ch])
        else:
            z = np.load(self.dir / f"{name}.npz")
            a = z[z.files[0]]
            for i in range(0, a.shape[0], ch):
                yield i, np.asarray(a[i:i + ch])


def _dump(d, name, img):
    import cv2
    if d:
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def render_signal(sd, provider, out, si, T, fps, params, dump_n, dump_dir):
    """Render every proxy a provider declares. Returns a manifest fragment."""
    disp = provider.display
    if not sd.has(disp.dataset) and not (disp.derive_from and sd.has(disp.derive_from)):
        return None
    src = disp.derive_from or disp.dataset
    shape = sd.shape(disp.dataset if sd.has(disp.dataset) else src)
    H, W = shape[-2], shape[-1]

    lo, hi = (0.0, 1.0)
    if disp.range_fn is not None:
        lo, hi = disp.range_fn(c for _, c in sd.chunks(src))
    ctx = DisplayCtx(lo, hi, params)

    writers = {}
    for spec in disp.proxies:
        p = out / f"shot_{si:03d}_{spec.suffix}.mp4"
        writers[spec.suffix] = (p, Mp4Writer(p, W, H, fps))
    picks = set(np.linspace(0, T - 1, dump_n).astype(int)) if dump_n else set()
    last = {}
    written = 0

    for start, chunk in sd.chunks(src):
        frames = disp.derive(chunk, ctx) if disp.derive is not None else chunk
        for j, fr in enumerate(np.asarray(frames)):
            fr = fr.astype(np.float32)
            for spec in disp.proxies:
                img = spec.render(fr, ctx)
                writers[spec.suffix][1].write(img)
                last[spec.suffix] = img
                if start + j in picks:
                    _dump(dump_dir,
                          f"{spec.suffix}_shot{si:03d}_f{start+j:05d}.png", img)
            written += 1

    # A T-1 dataset (flow) must still yield a T-frame proxy, or Coyote's
    # lockstep scrub drifts by one from the last frame onward. Hold the final
    # field for the extra frame rather than emitting a blank one.
    ds = next((d for d in provider.schema.datasets if d.name == disp.dataset), None)
    tail = None
    if ds is not None and ds.frames == "T-1" and written == T - 1:
        for spec in disp.proxies:
            writers[spec.suffix][1].write(last[spec.suffix])
        tail = "last field held for frame T-1"
        written += 1

    rec = {"ranges": {"lo": float(lo), "hi": float(hi)},
           "readout_kind": disp.readout_kind, "proxies": {}}
    if tail:
        rec["ranges"]["tail_policy"] = tail
    if disp.derive is not None:
        rec["ranges"]["derived_from"] = src
    for spec in disp.proxies:
        p, w = writers[spec.suffix]
        rec["proxies"][spec.suffix] = {"path": str(p), "frames": w.close(expect=T),
                                       "doc": spec.doc}
    log(f"  {provider.name:<8}-> {', '.join(s.suffix for s in disp.proxies)}  "
        f"{T} frames  range [{lo:.4f}, {hi:.4f}]")
    return rec


def run_colorize(a, providers=None):
    from . import registry
    from .scheduler import order
    if providers is None:
        registry.load_all()
        providers = order(registry.all_providers())

    run = Path(a.run)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dump_dir = out / "png_dump"
    man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    fps = man.get("fps", 24.0)
    params = {"depth_ramp": a.depth_ramp, "bilateral": a.bilateral,
              "flow_map": getattr(a, "flow_map", "sqrt"),
              "arrow_grid": getattr(a, "arrow_grid", 32),
              "fov": a.fov, "flow_pct": a.flow_pct}
    log(f"{run.name}: {man.get('frames')} frames @ {man.get('res')} {fps} fps, "
        f"{len(man.get('shots', []))} shot(s), " +
        ", ".join(f"{k}={v}" for k, v in params.items()))

    recs = []
    for si, shot in enumerate(man.get("shots", [])):
        d = run / f"shot_{si:03d}"
        if not d.exists():
            continue
        log(f"--- shot {si} (source frames {shot['start']}-{shot['end']}) ---")
        sd = ShotData(d)
        T = shot["n_frames"]
        rec = {"shot": si, "source_start": shot["start"],
               "source_end": shot["end"], "source_frames": T, "signals": {}}
        for p in providers:
            r = render_signal(sd, p, out, si, T, fps, params, a.dump_png, dump_dir)
            if r is not None:
                rec["signals"][p.name] = r
        for name, r in rec["signals"].items():
            for k, v in r["proxies"].items():
                assert v["frames"] == T, (
                    f"shot {si} {name}/{k}: {v['frames']} proxy frames vs "
                    f"{T} source frames")
        recs.append(rec)

    log("verifying written files with ffprobe…")
    total = 0
    for r in recs:
        for name, sig in r["signals"].items():
            for k, v in sig["proxies"].items():
                nb, rate = probe_frames(v["path"])
                v["ffprobe_frames"], v["ffprobe_fps"] = nb, rate
                assert nb == r["source_frames"], (
                    f"shot {r['shot']} {name}/{k}: ffprobe counts {nb}, source "
                    f"has {r['source_frames']}")
                assert abs(float(rate) - fps) < 1e-6, f"{k}: fps {rate} != {fps}"
                total += 1
    log(f"  {total} proxies verified: frame count and fps match source exactly")

    (out / "proxies.json").write_text(
        json.dumps({"run": str(run), "source": man.get("video"), "fps": fps,
                    "params": params,
                    "providers": [p.name for p in providers],
                    "shots": recs}, indent=2), encoding="utf-8")
    log(f"proxies.json -> {out/'proxies.json'}")
    if dump_dir.exists():
        log(f"png dumps -> {dump_dir}  ({len(list(dump_dir.glob('*.png')))} files)")
