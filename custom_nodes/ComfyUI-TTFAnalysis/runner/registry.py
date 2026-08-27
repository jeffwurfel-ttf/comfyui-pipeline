"""
Provider registry — the contract every analysis signal declares itself through.

A Provider is metadata plus two callables. Nothing about a signal is special
cased anywhere else in the runner: the scheduler reads `depends_on` and `cost`,
dispatch reads `env`, and the proxy renderer reads `display`. Adding a signal
means adding one file under providers/; it must require editing nothing else.

LICENSE IS PART OF THE CONTRACT. Registration ASSERTS a complete LicenseRow —
it does not warn. Every model in this pipeline ships in commercial client
deliverables, and the failure mode we have already hit twice (DSINE's Imperial
licence, Video-Depth-Anything's Base/Large weights being CC-BY-NC-4.0 while its
code is Apache-2.0) is a signal arriving with its licence unexamined. A warning
scrolls past. An assert cannot.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

VERDICTS = ("CLEAN", "CONDITIONAL", "BLOCKED", "UNRESOLVED")


# ────────────────────────────────────────────────────────────── licence
@dataclass(frozen=True)
class LicenseRow:
    """One row of .dev/LICENSE_AUDIT.md, attached to the code it describes.

    `weight_license` is separate from `code_license` on purpose: VDA is
    Apache-2.0 in code and CC-BY-NC-4.0 for its Base/Large weights, and
    collapsing the two is exactly how that trap gets missed.
    """
    model: str
    code_license: str
    weight_license: str
    gated: bool
    verdict: str
    source_url: str
    date_checked: str
    note: str = ""

    def validate(self, who):
        missing = [f for f in ("model", "code_license", "weight_license",
                               "verdict", "source_url", "date_checked")
                   if not str(getattr(self, f) or "").strip()]
        assert not missing, (
            f"provider {who!r}: LicenseRow is incomplete, missing {missing}. "
            f"A signal may not register without a complete licence row — see "
            f".dev/LICENSE_AUDIT.md.")
        assert self.verdict in VERDICTS, (
            f"provider {who!r}: verdict {self.verdict!r} not one of {VERDICTS}")
        assert isinstance(self.gated, bool), (
            f"provider {who!r}: `gated` must be an explicit bool, not "
            f"{self.gated!r} — 'unknown' is not an answer here.")
        assert len(self.date_checked) == 10 and self.date_checked[4] == "-", (
            f"provider {who!r}: date_checked must be YYYY-MM-DD, got "
            f"{self.date_checked!r}")


# ─────────────────────────────────────────────────────────────── schema
@dataclass(frozen=True)
class Dataset:
    """One array a provider writes. `shape` uses symbolic axes: T frames, H/W
    the analysis resolution, ints for fixed axes."""
    name: str
    shape: Tuple
    dtype: str
    chunk_t: int = 8
    frames: str = "T"          # "T" or "T-1" (flow yields one fewer)


@dataclass(frozen=True)
class Schema:
    datasets: Tuple[Dataset, ...]

    def primary(self):
        return self.datasets[0]

    def validate(self, who):
        assert self.datasets, f"provider {who!r}: schema declares no datasets"
        seen = set()
        for d in self.datasets:
            assert d.name not in seen, f"provider {who!r}: duplicate dataset {d.name}"
            seen.add(d.name)
            assert d.frames in ("T", "T-1"), (
                f"provider {who!r}: dataset {d.name} frames={d.frames!r}")


# ───────────────────────────────────────────────────────────────── cost
@dataclass(frozen=True)
class Cost:
    """What the scheduler needs to not OOM the box.

    `vram_mb` is the MEASURED peak, not a guess. flow is 13304 MiB at 1920x1340;
    two such providers co-scheduled on a 24 GB card is an OOM, and the scheduler
    is expected to refuse rather than discover that at runtime.

    `window` is the number of frames a single forward consumes. It lives here
    rather than in the chunker because it is a property of the model: VDA's
    packaged path pads any clip shorter than 32 up to 32, so a 16-frame chunk
    costs exactly what a 32-frame one does. The chunker must not know that.
    """
    vram_mb: int
    window: int = 1
    overlap: int = 0
    gpu: bool = True
    note: str = ""

    def validate(self, who):
        assert self.vram_mb >= 0, f"provider {who!r}: negative vram_mb"
        assert self.window >= 1, f"provider {who!r}: window must be >= 1"
        assert 0 <= self.overlap < self.window or self.window == 1, (
            f"provider {who!r}: overlap {self.overlap} invalid for window "
            f"{self.window}")


# ────────────────────────────────────────────────────────────── display
@dataclass(frozen=True)
class ProxySpec:
    """One rendered output. `render(frame, ctx) -> HxWx3 uint8`.

    ctx carries the shot-scoped range and the display params, so a renderer
    never computes a range itself — that is what would reintroduce per-frame
    normalization and with it the flicker VDA exists to remove.
    """
    suffix: str
    render: Callable
    doc: str = ""


@dataclass(frozen=True)
class Display:
    """How P2 turns this signal into proxies. Replaces the per-signal if/elif
    chains that used to live in proxies.py."""
    dataset: str
    proxies: Tuple[ProxySpec, ...]
    range_fn: Optional[Callable] = None      # (chunks) -> (lo, hi)
    derive: Optional[Callable] = None        # (chunk, ctx) -> frames to render
    derive_from: Optional[str] = None        # dataset the derive hook reads
    readout_kind: str = "scalar"             # scalar | vector2 | unit3

    def validate(self, who):
        assert self.proxies, f"provider {who!r}: display declares no proxies"
        for p in self.proxies:
            assert callable(p.render), f"provider {who!r}: proxy {p.suffix} render"


# ─────────────────────────────────────────────────────────────── provider
@dataclass(frozen=True)
class Provider:
    name: str
    env: str                       # "_env" | "main"
    schema: Schema
    cost: Cost
    display: Display
    license: LicenseRow
    run: Optional[Callable] = None   # (ctx) -> None; writes via ctx.sink
    depends_on: Tuple[str, ...] = ()
    doc: str = ""

    def validate(self):
        assert self.name and self.name.isidentifier(), (
            f"provider name {self.name!r} must be a valid identifier")
        assert self.env in ENVS, (
            f"provider {self.name!r}: env {self.env!r} not one of {tuple(ENVS)}")
        self.license.validate(self.name)
        self.schema.validate(self.name)
        self.cost.validate(self.name)
        self.display.validate(self.name)
        assert self.run is None or callable(self.run), (
            f"provider {self.name!r}: run must be callable or None")


# environments dispatch knows how to reach. See dispatch.py.
ENVS = ("_env", "main")

_REGISTRY = {}


def register(p: Provider):
    p.validate()
    assert p.name not in _REGISTRY, f"provider {p.name!r} already registered"
    _REGISTRY[p.name] = p
    return p


def unregister(name):
    """Only for tests and the throwaway dispatch proof."""
    _REGISTRY.pop(name, None)


def get(name):
    assert name in _REGISTRY, (
        f"unknown provider {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def all_providers():
    """Registration order is preserved and is load-bearing: the scheduler uses
    it as the tie-break in its topological sort, which keeps execution order
    stable across runs. Changing it changes GPU call order, which is part of
    the byte-identity contract."""
    return tuple(_REGISTRY.values())


def names():
    return tuple(_REGISTRY)


def load_all():
    """Import every module in providers/ so each self-registers.

    This is what makes 'adding a provider edits zero existing files' true —
    there is no list of signals anywhere to append to.
    """
    import importlib
    import pkgutil
    from . import providers
    for m in pkgutil.iter_modules(providers.__path__):
        if not m.name.startswith("_"):
            importlib.import_module(f"{providers.__name__}.{m.name}")
    return all_providers()
