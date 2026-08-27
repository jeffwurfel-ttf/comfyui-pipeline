"""
TTF analysis runner — depth, optical flow, derived normals, normalization.

Standalone by construction: nothing here imports ComfyUI at module level, so the
package runs from a plain checkout and P3's node wrapper sits on top without
requiring a rewrite. Submodules are NOT imported eagerly — importing this
package must not pull torch, h5py or the vendored repos into a process that only
wants runner.normalize (pure CPU).
"""
__all__ = [
    "paths", "isolation", "streaming", "readout", "shots",
    "depth_vda", "flow_searaft", "normals_derived", "normalize", "proxies",
]
