#!/usr/bin/env python
"""
Standalone CLI for the analysis runner. No ComfyUI import anywhere — P3's node
wrapper sits on top of this and does not require rewriting it.

  python -m runner.cli analyze  --video CLIP --signals depth,flow,normals --out DIR
  python -m runner.cli colorize --run DIR --out DIR

analyze  writes raw data only (HDF5, float16) plus a readout. No colorization.
colorize turns that into scrubbable H.264 proxies. Pure CPU.

Nothing accumulates a whole shot in memory in either direction: P1 peaked at
41.5 GB RSS of 46 GB on one 398-frame 1920x1340 shot, which put a 90-second shot
past the box.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import depth_vda, flow_searaft, paths, shots
from .isolation import no_lazy_colliding_import
from .readout import FrameStats, flags_for
from .streaming import H5Sink, exact_quantiles, h5_chunks, probe_video

NORMALS_CHUNK = 16


def log(tag, m):
    print(f"[{tag}] {m}", flush=True)


def mib(x):
    return round(x / 2**20, 1)


def rss_gb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return round(int(line.split()[1]) / 1048576, 2)
    except Exception:
        pass
    return None


def smi_mib():
    try:
        o = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], text=True)
        for line in o.strip().splitlines():
            pid, used = [s.strip() for s in line.split(",")]
            if int(pid) == os.getpid():
                return int(used)
    except Exception:
        pass
    return None


# ────────────────────────────────────────────────────────────── analyze
def stream_normals(tmp_path, sink, fov, lo, hi, stats):
    from .normals_derived import normals_from_depth_gpu
    valid_sum, valid_n = 0.0, 0
    for start, chunk in h5_chunks(tmp_path, "depth32", NORMALS_CHUNK):
        n, m = normals_from_depth_gpu(chunk, lo, hi, fov)
        sink.write("normals", start, n.astype(np.float16))
        sink.write("normals_valid", start, m)
        stats.add(n, "normals")
        valid_sum += float(m.sum())
        valid_n += m.size
        del n, m
    return valid_sum / max(valid_n, 1)


def cmd_analyze(a):
    import torch
    signals = [s.strip() for s in a.signals.split(",") if s.strip()]
    assert set(signals) <= {"depth", "flow", "normals"}, signals
    if "normals" in signals and "depth" not in signals:
        signals.insert(0, "depth")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.skip_invariant_test:
        from .tests import test_module_invariant as inv
        log("run", "checking module invariant…")
        inv.main()

    T, H, W, fps = probe_video(a.video, a.max_side)
    log("run", f"{Path(a.video).name}: {T} frames, {W}x{H}, {fps:.2f} fps")
    sl = shots.detect(a.video, a.shot_threshold, T)
    log("run", f"shots: {len(sl)} -> {sl}")

    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    vda = helpers = raft = rargs = None
    if "depth" in signals:
        vda, helpers = depth_vda.load()
    if "flow" in signals:
        raft, rargs = flow_searaft.load()
    log("run", f"weights resident: {mib(torch.cuda.memory_allocated())} MiB")

    man = {
        "video": str(Path(a.video).resolve()), "frames": T, "res": f"{W}x{H}",
        "fps": round(fps, 3), "shots": [], "signals": signals,
        "params": {"input_size": a.input_size, "overlap": a.overlap,
                   "fov_deg": a.fov, "max_side": a.max_side,
                   "shot_threshold": a.shot_threshold, "vda_encoder": "vits",
                   "searaft_cfg": "spring-M",
                   },
        "timings": {}, "peak_vram_MiB": {}, "flags": [], "storage": "hdf5",
    }
    times = {s: 0.0 for s in signals}

    for si, (s0, s1) in enumerate(sl):
        sd = out / f"shot_{si:03d}"
        sd.mkdir(exist_ok=True)
        n = s1 - s0
        entry = {"index": si, "start": s0, "end": s1, "n_frames": n,
                 "readout": {}, "flags": {}}
        log("run", f"--- shot {si}: frames {s0}-{s1} ({n}) ---")
        sink = H5Sink(sd / "signals.h5")
        tmp_path = sd / "_depth32.tmp.h5"
        try:
            if "depth" in signals:
                sink.create("depth", (n, H, W), np.float16)
                tmp = H5Sink(tmp_path, compression=None)
                tmp.create("depth32", (n, H, W), np.float32)
                st = FrameStats()
                torch.cuda.reset_peak_memory_stats()
                t = time.time()
                with no_lazy_colliding_import("VDA forward"):
                    dmn, dmx = depth_vda.stream(
                        vda, helpers, a.video, s0, s1, H, W, a.max_side, sink,
                        tmp, a.input_size, a.overlap,
                        on_block=lambda b: st.add(b, "depth"))
                tmp.close()
                times["depth"] += time.time() - t
                man["peak_vram_MiB"]["depth"] = max(
                    man["peak_vram_MiB"].get("depth", 0),
                    mib(torch.cuda.max_memory_allocated()))
                entry["readout"]["depth"] = st.rows
                entry["flags"]["depth"] = flags_for(st.rows, "depth", raw=st.raw)
                sink.attrs("depth", min=dmn, max=dmx)
                log("run", f"  depth  {time.time()-t:6.2f}s  peak "
                           f"{man['peak_vram_MiB']['depth']} MiB  RSS {rss_gb()} GB")

            if "normals" in signals:
                sink.create("normals", (n, 3, H, W), np.float16, chunk_t=4)
                sink.create("normals_valid", (n, H, W), np.uint8)
                st = FrameStats()
                t = time.time()
                q = exact_quantiles(
                    lambda: (c.reshape(-1) for _, c in h5_chunks(tmp_path, "depth32", 8)),
                    n * H * W, (1, 99))
                lo, hi = q[1], q[99]
                vfrac = stream_normals(tmp_path, sink, a.fov, lo, hi, st)
                times["normals"] += time.time() - t
                entry["readout"]["normals"] = st.rows
                entry["flags"]["normals"] = flags_for(st.rows, "normals", raw=st.raw)
                entry["normals_valid_frac"] = round(vfrac, 4)
                entry["normals_percentiles"] = {"p1": lo, "p99": hi}
                log("run", f"  normals{time.time()-t:6.2f}s  "
                           f"valid {vfrac*100:.1f}%"
                           f"  RSS {rss_gb()} GB")

            if "flow" in signals and n > 1:
                sink.create("flow", (n - 1, 2, H, W), np.float16, chunk_t=4)
                st = FrameStats()
                torch.cuda.reset_peak_memory_stats()
                t = time.time()
                with no_lazy_colliding_import("SEA-RAFT forward"):
                    flow_searaft.stream(raft, rargs, a.video, s0, s1, a.max_side,
                                        sink, on_frame=lambda f: st.add(f, "flow"))
                times["flow"] += time.time() - t
                man["peak_vram_MiB"]["flow"] = max(
                    man["peak_vram_MiB"].get("flow", 0),
                    mib(torch.cuda.max_memory_allocated()))
                entry["readout"]["flow"] = st.rows
                entry["flags"]["flow"] = flags_for(st.rows, "flow", (H, W), raw=st.raw)
                log("run", f"  flow   {time.time()-t:6.2f}s  peak "
                           f"{man['peak_vram_MiB']['flow']} MiB  RSS {rss_gb()} GB")
        finally:
            sink.close()
            if tmp_path.exists():
                tmp_path.unlink()

        man["shots"].append(entry)
        for k, v in entry["flags"].items():
            man["flags"] += [f"shot{si} {k}: {x}" for x in v]

    man["timings"] = {k: round(v, 2) for k, v in times.items()}
    man["peak_vram_MiB"]["overall_allocated"] = mib(torch.cuda.max_memory_allocated())
    man["peak_vram_MiB"]["nvidia_smi_process"] = smi_mib()
    man["peak_rss_GB"] = rss_gb()
    # manifest LAST: ComfyUI reports "completed" before outputs land, so the
    # manifest's presence is the only valid completion signal.
    (out / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    log("run", f"manifest -> {out/'manifest.json'}   peak RSS {man['peak_rss_GB']} GB")
    if man["flags"]:
        log("run", f"FLAGS ({len(man['flags'])}):")
        for f in man["flags"][:10]:
            log("run", f"  ! {f}")
    else:
        log("run", "no anomaly flags")


# ───────────────────────────────────────────────────────────── colorize
def cmd_colorize(a):
    from .proxies import run_colorize
    run_colorize(a)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="runner.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)

    an = sub.add_parser("analyze")
    an.add_argument("--video", required=True)
    an.add_argument("--signals", default="depth,flow,normals")
    an.add_argument("--out", required=True)
    an.add_argument("--max-side", type=int, default=None)
    an.add_argument("--shot-threshold", type=float, default=27.0)
    an.add_argument("--input-size", type=int, default=518)
    an.add_argument("--overlap", type=int, default=8)
    an.add_argument("--fov", type=float, default=60.0)
    an.add_argument("--skip-invariant-test", action="store_true")
    an.set_defaults(fn=cmd_analyze)

    co = sub.add_parser("colorize")
    co.add_argument("--run", required=True)
    co.add_argument("--out", required=True)
    co.add_argument("--depth-ramp", default="gray")
    co.add_argument("--bilateral", type=float, default=1.0)
    co.add_argument("--flow-pct", type=float, default=99.0)
    co.add_argument("--fov", type=float, default=60.0)
    co.add_argument("--dump-png", type=int, default=6)
    co.set_defaults(fn=cmd_colorize)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
