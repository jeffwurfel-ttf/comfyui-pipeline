"""
P2 proxy rendering — raw analysis data to scrubbable H.264. Pure CPU.

Every range used is recorded into proxies.json. Without it a proxy cannot be
inverted back to the underlying data and the display is a dead end.

Normals are RE-DERIVED here rather than read from the stored array, because the
bilateral filter acts on depth BEFORE differentiation. Keeping it here means the
stored normals stay unfiltered (that is data) while the filter stays a display
parameter (that is presentation). --bilateral 0 uses the stored array.
"""
import json
from pathlib import Path

import numpy as np

from .normalize import (
    Mp4Writer, apply_range, bilateral_depth, flow_clamp, flow_to_magnitude,
    flow_to_wheel, normals_to_rgb, probe_frames, ramp, shot_range,
)

CH = 16


def log(m):
    print(f"[p2] {m}", flush=True)


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


def do_shot(sd, out, si, fps, ramp_name, bilateral, flow_pct, fov,
            dump_n, dump_dir):
    rec = {"shot": si, "ranges": {}, "proxies": {}}
    depth_shape = sd.shape("depth") if sd.has("depth") else None
    T = depth_shape[0] if depth_shape else None

    if depth_shape:
        H, W = depth_shape[1:]
        lo, hi = shot_range((c.astype(np.float32) for _, c in sd.chunks("depth")),
                            mode="percentile")
        rec["ranges"]["depth"] = {"lo": lo, "hi": hi, "mode": "percentile",
                                  "pct": [1.0, 99.0], "ramp": ramp_name}
        p = out / f"shot_{si:03d}_depth.mp4"
        w = Mp4Writer(p, W, H, fps)
        picks = set(np.linspace(0, T - 1, dump_n).astype(int)) if dump_n else set()
        for start, c in sd.chunks("depth"):
            for j, fr in enumerate(c.astype(np.float32)):
                img = ramp(apply_range(fr, lo, hi), ramp_name)
                w.write(img)
                if start + j in picks:
                    _dump(dump_dir, f"depth_shot{si:03d}_f{start+j:05d}.png", img)
        rec["proxies"]["depth"] = {"path": str(p), "frames": w.close(expect=T)}
        log(f"  depth   -> {p.name}  {T} frames  range [{lo:.4f}, {hi:.4f}]")

    if depth_shape:
        H, W = depth_shape[1:]
        p = out / f"shot_{si:03d}_normals.mp4"
        w = Mp4Writer(p, W, H, fps)
        picks = set(np.linspace(0, T - 1, dump_n).astype(int)) if dump_n else set()
        src = "re-derived from filtered depth" if bilateral > 0 else "stored array"
        if bilateral > 0:
            from .normals_derived import normals_from_depth_cpu
            dlo, dhi = shot_range(
                (c.astype(np.float32) for _, c in sd.chunks("depth")),
                mode="percentile")
            for start, c in sd.chunks("depth"):
                filt = bilateral_depth(c.astype(np.float32), bilateral)
                n, _ = normals_from_depth_cpu(filt, dlo, dhi, fov)
                for j, fr in enumerate(n):
                    img = normals_to_rgb(fr)
                    w.write(img)
                    if start + j in picks:
                        _dump(dump_dir, f"normals_shot{si:03d}_f{start+j:05d}.png", img)
            rec["ranges"]["normals"] = {"encoding": "(n+1)/2", "bilateral": bilateral,
                                        "derived_from_depth_pct": [dlo, dhi]}
        else:
            for start, c in sd.chunks("normals"):
                for j, fr in enumerate(c.astype(np.float32)):
                    img = normals_to_rgb(fr)
                    w.write(img)
                    if start + j in picks:
                        _dump(dump_dir, f"normals_shot{si:03d}_f{start+j:05d}.png", img)
            rec["ranges"]["normals"] = {"encoding": "(n+1)/2", "bilateral": 0}
        rec["proxies"]["normals"] = {"path": str(p), "frames": w.close(expect=T),
                                     "source": src}
        log(f"  normals -> {p.name}  {T} frames  ({src})")

    if sd.has("flow"):
        fs = sd.shape("flow")
        Tf, H, W = fs[0], fs[2], fs[3]
        lo, hi = flow_clamp(
            (np.sqrt(c[:, 0].astype(np.float32) ** 2 + c[:, 1].astype(np.float32) ** 2)
             for _, c in sd.chunks("flow")), pct=flow_pct)
        rec["ranges"]["flow"] = {"lo": lo, "hi": hi, "mode": f"p{flow_pct} clamp",
                                 "units": "pixels/frame"}
        pm = out / f"shot_{si:03d}_flow_mag.mp4"
        pw = out / f"shot_{si:03d}_flow_wheel.mp4"
        wm, ww = Mp4Writer(pm, W, H, fps), Mp4Writer(pw, W, H, fps)
        picks = set(np.linspace(0, T - 1, dump_n).astype(int)) if dump_n else set()
        last_m = last_w = None
        for start, c in sd.chunks("flow"):
            for j, fr in enumerate(c.astype(np.float32)):
                im_m = flow_to_magnitude(fr, lo, hi, ramp_name)
                im_w = flow_to_wheel(fr, lo, hi)
                wm.write(im_m); ww.write(im_w)
                last_m, last_w = im_m, im_w
                if start + j in picks:
                    _dump(dump_dir, f"flowmag_shot{si:03d}_f{start+j:05d}.png", im_m)
                    _dump(dump_dir, f"flowwheel_shot{si:03d}_f{start+j:05d}.png", im_w)
        # T source frames yield T-1 fields. The proxy must still be T frames or
        # Coyote's lockstep scrub drifts by one from the last frame onward.
        if T and Tf == T - 1:
            wm.write(last_m); ww.write(last_w)
            rec["ranges"]["flow"]["tail_policy"] = "last field held for frame T-1"
        rec["proxies"]["flow_mag"] = {"path": str(pm), "frames": wm.close(expect=T)}
        rec["proxies"]["flow_wheel"] = {"path": str(pw), "frames": ww.close(expect=T)}
        log(f"  flow    -> {pm.name} + {pw.name}  {T} frames  "
            f"clamp [0, {hi:.3f}] px/frame (p{flow_pct})")
    return rec


def run_colorize(a):
    run = Path(a.run)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dump_dir = out / "png_dump"
    man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    fps = man.get("fps", 24.0)
    log(f"{run.name}: {man.get('frames')} frames @ {man.get('res')} {fps} fps, "
        f"{len(man.get('shots', []))} shot(s), ramp={a.depth_ramp}, "
        f"bilateral={a.bilateral}")

    recs = []
    for si, shot in enumerate(man.get("shots", [])):
        d = run / f"shot_{si:03d}"
        if not d.exists():
            continue
        log(f"--- shot {si} (source frames {shot['start']}-{shot['end']}) ---")
        r = do_shot(ShotData(d), out, si, fps, a.depth_ramp, a.bilateral,
                    a.flow_pct, a.fov, a.dump_png, dump_dir)
        r["source_start"], r["source_end"] = shot["start"], shot["end"]
        r["source_frames"] = shot["n_frames"]
        for k, v in r["proxies"].items():
            assert v["frames"] == shot["n_frames"], (
                f"shot {si} {k}: {v['frames']} proxy frames vs "
                f"{shot['n_frames']} source frames")
        recs.append(r)

    log("verifying written files with ffprobe…")
    total = 0
    for r in recs:
        for k, v in r["proxies"].items():
            nb, rate = probe_frames(v["path"])
            v["ffprobe_frames"], v["ffprobe_fps"] = nb, rate
            assert nb == r["source_frames"], (
                f"shot {r['shot']} {k}: ffprobe counts {nb}, source has "
                f"{r['source_frames']}")
            assert abs(float(rate) - fps) < 1e-6, f"{k}: fps {rate} != {fps}"
            total += 1
    log(f"  {total} proxies verified: frame count and fps match source exactly")

    (out / "proxies.json").write_text(
        json.dumps({"run": str(run), "source": man.get("video"), "fps": fps,
                    "params": {"depth_ramp": a.depth_ramp,
                               "bilateral": a.bilateral,
                               "flow_pct": a.flow_pct, "fov": a.fov},
                    "shots": recs}, indent=2), encoding="utf-8")
    log(f"proxies.json -> {out/'proxies.json'}")
    if dump_dir.exists():
        log(f"png dumps -> {dump_dir}  ({len(list(dump_dir.glob('*.png')))} files)")
