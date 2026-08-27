"""
Bounded-memory primitives for the analysis harness.

P1 accumulated a whole shot per signal in RAM: 41.5 GB peak RSS of 46 GB total
for one 398-frame 1920x1340 shot. A 90-second shot would OOM the box and take
ComfyUI with it, so P4 cannot run on that shape.

Three pieces remove the accumulation:

  H5Sink              incremental chunked writes; npz cannot append.
  exact_quantiles     shot-scoped percentiles WITHOUT holding the shot. Exact,
                      not approximate — see below.
  WindowFrameReader   sequential decode with a sliding buffer, so the source
                      video is never fully resident either.

On exactness: the naive fix for percentiles is to subsample. That would change
the normals output, and "a fix that changes the numbers is a bug." This uses a
histogram-then-refine order statistic that reproduces np.percentile's linear
interpolation bit-for-bit while holding only an 8 MB histogram.
"""
import numpy as np

try:
    import h5py
except ImportError:                                    # pragma: no cover
    h5py = None


# ───────────────────────────────────────────────────────────── sink
class H5Sink:
    """Incremental writer. One dataset per signal, chunked along time.

    gzip level 4 lands within a few percent of npz's whole-array deflate while
    staying seekable, which is what lets P2 stream these back a chunk at a time
    instead of loading a shot to colorize it.
    """

    def __init__(self, path, compression="gzip", level=4):
        assert h5py is not None, "h5py required for streaming writes"
        self.path = str(path)
        self.f = h5py.File(self.path, "w")
        self.compression, self.level = compression, level
        self.ds = {}

    def create(self, name, shape, dtype, chunk_t=8):
        chunks = (min(chunk_t, shape[0]),) + tuple(shape[1:])
        kw = {}
        if self.compression:      # h5py rejects compression_opts when unset
            kw = dict(compression=self.compression, compression_opts=self.level)
        self.ds[name] = self.f.create_dataset(
            name, shape=shape, dtype=dtype, chunks=chunks, **kw)
        return self.ds[name]

    def write(self, name, start, arr):
        # Cast through numpy, never let HDF5 do the narrowing conversion.
        # HDF5's float32->float16 path truncates toward zero while numpy rounds
        # to nearest-even, so handing it a float32 array silently produces
        # values one ULP off from `arr.astype(np.float16)`. depth and normals
        # happened to be cast at the call site and matched; flow was not, and
        # that alone broke byte-identity against the P1 artifacts.
        d = self.ds[name]
        a = np.asarray(arr)
        if a.dtype != d.dtype:
            a = a.astype(d.dtype)
        d[start:start + len(a)] = a

    def attrs(self, name, **kw):
        for k, v in kw.items():
            self.ds[name].attrs[k] = v

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


def h5_chunks(path, name, chunk=16):
    """Yield (start, array) slices of an on-disk dataset without loading it."""
    with h5py.File(str(path), "r") as f:
        d = f[name]
        for i in range(0, d.shape[0], chunk):
            yield i, d[i:i + chunk]


def h5_read(path, name):
    with h5py.File(str(path), "r") as f:
        return f[name][...]


# ──────────────────────────────────────────────────── exact quantiles
def exact_quantiles(chunk_factory, total_n, qs, bins=1 << 20):
    """Exact np.percentile(..., method='linear') over data too large to hold.

    chunk_factory() must return a FRESH iterator of 1-D float arrays each call;
    the data is traversed three times (min/max, histogram, refine).

    Reproduces numpy exactly: for percentile q the virtual index is
    (N-1)*q/100, and the result interpolates linearly between the two
    bracketing order statistics. Those two order statistics are recovered
    exactly by locating their histogram bin, then re-scanning only that bin.
    """
    lo_v, hi_v = np.inf, -np.inf
    for c in chunk_factory():
        if c.size:
            lo_v = min(lo_v, float(np.nanmin(c)))
            hi_v = max(hi_v, float(np.nanmax(c)))
    if not np.isfinite(lo_v) or not np.isfinite(hi_v):
        return {q: float("nan") for q in qs}
    if hi_v <= lo_v:                                   # constant field
        return {q: lo_v for q in qs}

    span = hi_v - lo_v
    hist = np.zeros(bins, np.int64)
    for c in chunk_factory():
        idx = ((c.astype(np.float64) - lo_v) / span * (bins - 1)).astype(np.int64)
        np.clip(idx, 0, bins - 1, out=idx)
        hist += np.bincount(idx, minlength=bins)
    cum = np.cumsum(hist)

    need = {}
    for q in qs:
        vi = (total_n - 1) * (q / 100.0)
        k0 = int(np.floor(vi)); k1 = int(np.ceil(vi))
        need[q] = (k0, k1, vi - k0)
    ks = sorted({k for v in need.values() for k in v[:2]})
    kbin = {k: int(np.searchsorted(cum, k + 1, side="left")) for k in ks}
    wanted = sorted(set(kbin.values()))

    buckets = {b: [] for b in wanted}
    for c in chunk_factory():
        idx = ((c.astype(np.float64) - lo_v) / span * (bins - 1)).astype(np.int64)
        np.clip(idx, 0, bins - 1, out=idx)
        for b in wanted:
            m = idx == b
            if m.any():
                buckets[b].append(c[m])

    order = {}
    for b in wanted:
        vals = np.sort(np.concatenate(buckets[b])) if buckets[b] else np.array([lo_v])
        base = int(cum[b - 1]) if b > 0 else 0
        for k in ks:
            if kbin[k] == b:
                order[k] = float(vals[min(max(k - base, 0), len(vals) - 1)])
    return {q: order[k0] + frac * (order[k1] - order[k0])
            for q, (k0, k1, frac) in need.items()}


# ──────────────────────────────────────────────────── frame reader
class WindowFrameReader:
    """Sequential decode serving monotonically-increasing absolute windows.

    plan_windows() always emits non-decreasing starts (the tail window slides
    back but never before the previous start), so a forward-only decode with a
    sliding buffer is sufficient and avoids H.264 seeking, which is both slow
    and unreliable on long GOPs.
    """

    def __init__(self, path, max_side=None, to_rgb=True):
        import cv2
        self.cv2, self.path = cv2, str(path)
        self.max_side, self.to_rgb = max_side, to_rgb
        self.cap = cv2.VideoCapture(self.path)
        self.pos = 0
        self.buf = {}

    def _prep(self, f):
        if self.max_side:
            h, w = f.shape[:2]
            s = self.max_side / max(h, w)
            if s < 1.0:
                f = self.cv2.resize(f, (int(round(w * s)), int(round(h * s))),
                                    interpolation=self.cv2.INTER_AREA)
        return self.cv2.cvtColor(f, self.cv2.COLOR_BGR2RGB) if self.to_rgb else f

    def window(self, a, b):
        for k in [k for k in self.buf if k < a]:
            del self.buf[k]
        while self.pos < b:
            ok, f = self.cap.read()
            if not ok:
                break
            if self.pos >= a:
                self.buf[self.pos] = self._prep(f)
            self.pos += 1
        idx = [i for i in range(a, b) if i in self.buf]
        return np.stack([self.buf[i] for i in idx]) if idx else None

    def close(self):
        self.cap.release()
        self.buf.clear()


def probe_video(path, max_side=None):
    import cv2
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    ok, f = cap.read()
    cap.release()
    assert ok, f"cannot decode {path}"
    h, w = f.shape[:2]
    if max_side:
        s = max_side / max(h, w)
        if s < 1.0:
            h, w = int(round(h * s)), int(round(w * s))
    return n, h, w, fps
