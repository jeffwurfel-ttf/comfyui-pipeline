"""
What a provider's `run(ctx)` receives.

Deliberately narrow. A provider gets the shot it is working on, somewhere to
write, and a `state` dict for handing artifacts to its declared dependents —
nothing else. It does not get the list of other providers, the manifest, or the
scheduler, so it cannot reach sideways into a signal it did not declare a
dependency on.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RunContext:
    video: str
    s0: int
    s1: int
    H: int
    W: int
    max_side: Optional[int]
    sink: Any                       # streaming.H5Sink
    workdir: Path                   # per-shot scratch, deleted after the shot
    params: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    models: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self):
        return self.s1 - self.s0

    def param(self, k, default=None):
        return self.params.get(k, default)

    def record(self, name, arr, kind):
        """Feed written frames to that signal's readout accumulator."""
        st = self.stats.get(name)
        if st is not None:
            st.add(arr, kind)
