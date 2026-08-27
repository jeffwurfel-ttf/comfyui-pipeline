#!/usr/bin/env python
"""
Standalone CLI. No ComfyUI import anywhere — P3's node wrapper sits on top and
does not require rewriting this.

  python -m runner.cli analyze  --video CLIP --signals depth,flow,normals --out DIR
  python -m runner.cli colorize --run DIR --out DIR
  python -m runner.cli plan     --signals depth,flow,normals
  python -m runner.cli providers

analyze writes raw data only (HDF5, float16) plus a readout. No colorization.
There are no per-signal branches: the registry supplies providers, the
scheduler orders them, and each provider's run() writes its own datasets.

Nothing accumulates a whole shot in memory. P1 peaked at 41.5 GB RSS of 46 GB
on one 398-frame 1920x1340 shot, which put a 90-second shot past the box.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import registry, shots
from .context import RunContext
from .isolation import no_lazy_colliding_import
from .readout import FrameStats, flags_for
from .scheduler import DEFAULT_BUDGET_MB, explain, order, plan
from .streaming import H5Sink, probe_video

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


def _resolve_shape(dataset, T, H, W):
    out = []
    for ax in dataset.shape:
        if ax == "T":
            out.append(T)
        elif ax == "T-1":
            out.append(T - 1)
        elif ax == "H":
            out.append(H)
        elif ax == "W":
            out.append(W)
        else:
            out.append(int(ax))
    return tuple(out)


def cmd_analyze(a):
    import torch
    registry.load_all()
    want = [s.strip() for s in a.signals.split(",") if s.strip()] or None
    ordered, batches = plan(registry.all_providers(), want, a.vram_budget)
    log("run", f"providers: {' -> '.join(p.name for p in ordered)}")
    for i, b in enumerate(batches):
        log("run", f"  batch {i}: {', '.join(b.names)}  ({b.vram_mb} MiB)")

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
    # Model loading stays keyed off which providers were actually selected.
    models = {}
    names = {p.name for p in ordered}
    if "depth" in names:
        from . import depth_vda
        models["vda"] = depth_vda.load()
    if "flow" in names:
        from . import flow_searaft
        models["raft"] = flow_searaft.load()
    log("run", f"weights resident: {mib(torch.cuda.memory_allocated())} MiB")

    man = {
        "video": str(Path(a.video).resolve()), "frames": T, "res": f"{W}x{H}",
        "fps": round(fps, 3), "shots": [], "signals": [p.name for p in ordered],
        "params": {"input_size": a.input_size, "overlap": a.overlap,
                   "fov_deg": a.fov, "max_side": a.max_side,
                   "shot_threshold": a.shot_threshold,
                   "vram_budget_mb": a.vram_budget},
        "providers": {p.name: {"env": p.env, "depends_on": list(p.depends_on),
                               "vram_mb": p.cost.vram_mb,
                               "window": p.cost.window,
                               "license_verdict": p.license.verdict}
                      for p in ordered},
        "timings": {}, "peak_vram_MiB": {}, "flags": [], "storage": "hdf5",
    }
    times = {p.name: 0.0 for p in ordered}

    for si, (s0, s1) in enumerate(sl):
        sd = out / f"shot_{si:03d}"
        sd.mkdir(exist_ok=True)
        n = s1 - s0
        entry = {"index": si, "start": s0, "end": s1, "n_frames": n,
                 "readout": {}, "flags": {}}
        log("run", f"--- shot {si}: frames {s0}-{s1} ({n}) ---")
        sink = H5Sink(sd / "signals.h5")
        tmp_path = sd / "_depth32.tmp.h5"
        tmp = None
        try:
            ctx = RunContext(
                video=a.video, s0=s0, s1=s1, H=H, W=W, max_side=a.max_side,
                sink=sink, workdir=sd, models=models,
                params={"input_size": a.input_size, "overlap": a.overlap,
                        "fov": a.fov})
            # depth's float32 side-channel: normals needs unquantised depth and
            # the shot-scoped percentiles, which is exactly what depends_on
            # encodes. Created here because it is scratch, not a signal.
            if any(p.name == "depth" for p in ordered):
                tmp = H5Sink(tmp_path, compression=None)
                tmp.create("depth32", (n, H, W), np.float32)
                ctx.state["depth32_sink"] = tmp
                ctx.state["depth32_path"] = tmp_path

            for p in ordered:
                for ds in p.schema.datasets:
                    shape = _resolve_shape(ds, n, H, W)
                    if shape[0] <= 0:
                        continue
                    sink.create(ds.name, shape, np.dtype(ds.dtype),
                                chunk_t=ds.chunk_t)
                st = FrameStats()
                ctx.stats[p.name] = st
                torch.cuda.reset_peak_memory_stats()
                t = time.time()
                with no_lazy_colliding_import(f"{p.name} forward"):
                    p.run(ctx)
                if p.name == "depth" and tmp is not None:
                    tmp.close(); tmp = None
                times[p.name] += time.time() - t
                man["peak_vram_MiB"][p.name] = max(
                    man["peak_vram_MiB"].get(p.name, 0),
                    mib(torch.cuda.max_memory_allocated()))
                kind = p.display.readout_kind
                entry["readout"][p.name] = st.rows
                entry["flags"][p.name] = flags_for(
                    st.rows, kind,
                    (H, W) if kind == "vector2" else None, raw=st.raw)
                log("run", f"  {p.name:<8}{time.time()-t:6.2f}s  peak "
                           f"{man['peak_vram_MiB'][p.name]} MiB  RSS {rss_gb()} GB")
            if "normals_valid_frac" in ctx.state:
                entry["normals_valid_frac"] = round(ctx.state["normals_valid_frac"], 4)
            if "normals_percentiles" in ctx.state:
                entry["normals_percentiles"] = ctx.state["normals_percentiles"]
        finally:
            if tmp is not None:
                tmp.close()
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
    # manifest LAST: ComfyUI reports "completed" before outputs land, so its
    # presence is the only valid completion signal.
    (out / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    log("run", f"manifest -> {out/'manifest.json'}   peak RSS {man['peak_rss_GB']} GB")
    if man["flags"]:
        log("run", f"FLAGS ({len(man['flags'])}):")
        for f in man["flags"][:10]:
            log("run", f"  ! {f}")
    else:
        log("run", "no anomaly flags")


def cmd_colorize(a):
    from .proxies import run_colorize
    run_colorize(a)


def cmd_plan(a):
    registry.load_all()
    want = [s.strip() for s in a.signals.split(",") if s.strip()] or None
    print(explain(registry.all_providers(), want, a.vram_budget))


def cmd_providers(a):
    from . import dispatch
    registry.load_all()
    for p in registry.all_providers():
        print(f"{p.name}")
        print(f"  env          {p.env}  (interpreter: "
              f"{dispatch.interpreter_for(p.env)}, "
              f"in-process: {dispatch.is_in_process(p.env)})")
        print(f"  depends_on   {p.depends_on or '()'}")
        print(f"  cost         vram {p.cost.vram_mb} MiB, window {p.cost.window}, "
              f"overlap {p.cost.overlap}")
        print(f"  datasets     {', '.join(d.name for d in p.schema.datasets)}")
        print(f"  proxies      {', '.join(s.suffix for s in p.display.proxies)}")
        print(f"  licence      {p.license.verdict} — code {p.license.code_license} "
              f"/ weights {p.license.weight_license} (checked "
              f"{p.license.date_checked})")


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
    an.add_argument("--vram-budget", type=int, default=DEFAULT_BUDGET_MB)
    an.add_argument("--skip-invariant-test", action="store_true")
    an.set_defaults(fn=cmd_analyze)

    co = sub.add_parser("colorize")
    co.add_argument("--run", required=True)
    co.add_argument("--out", required=True)
    co.add_argument("--depth-ramp", default="gray")
    co.add_argument("--bilateral", type=float, default=1.0)
    co.add_argument("--flow-pct", type=float, default=99.0)
    co.add_argument("--flow-map", default="sqrt", choices=["linear", "sqrt", "log"])
    co.add_argument("--arrow-grid", type=int, default=32)
    co.add_argument("--fov", type=float, default=60.0)
    co.add_argument("--dump-png", type=int, default=6)
    co.set_defaults(fn=cmd_colorize)

    pl = sub.add_parser("plan")
    pl.add_argument("--signals", default="")
    pl.add_argument("--vram-budget", type=int, default=DEFAULT_BUDGET_MB)
    pl.set_defaults(fn=cmd_plan)

    pv = sub.add_parser("providers")
    pv.set_defaults(fn=cmd_providers)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
