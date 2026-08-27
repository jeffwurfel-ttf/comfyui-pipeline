"""
sys.path / sys.modules containment for the vendored analysis repos.

Video-Depth-Anything, SEA-RAFT and DSINE are script-layout repos that each
expect their own root on sys.path, and each ships a DIFFERENT top-level `utils`
package — SEA-RAFT and DSINE both resolve `utils.utils`. Whichever imports first
wins sys.modules and silently hands the wrong module to the second.

`repo()` opens a sys.path window and purges every colliding top-level name on
both edges. Instantiated classes keep working after the purge because a
function's __globals__ holds its module dict alive independently of sys.modules.

This is only sound while every model-level import resolves AT IMPORT TIME.
`no_lazy_colliding_import` wraps forward passes and fails loudly if a future
dependency bump introduces a lazy `import utils` inside one — which the purge
cannot protect and which would otherwise return silently wrong tensors.
"""
import sys

COLLIDING = {
    "utils", "models", "projects", "config", "datasets", "core", "benchmark",
    "loss", "losses", "raft", "update", "corr", "extractor", "layer",
    "video_depth_anything",
}


def purge():
    for name in list(sys.modules):
        if name.split(".")[0] in COLLIDING:
            del sys.modules[name]


class repo:
    def __init__(self, *paths):
        self.paths = [str(p) for p in paths]

    def __enter__(self):
        self.saved = list(sys.path)
        purge()
        for p in reversed(self.paths):
            sys.path.insert(0, p)
        return self

    def __exit__(self, *exc):
        sys.path[:] = self.saved
        purge()
        return False


class no_lazy_colliding_import:
    """Tripwire for forward passes. See module docstring."""

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.before = set(sys.modules)
        return self

    def __exit__(self, *exc):
        if exc[0] is not None:
            return False
        new = {n for n in set(sys.modules) - self.before
               if n.split(".")[0] in COLLIDING}
        assert not new, (
            f"[{self.label}] a colliding module was imported DURING the forward "
            f"pass: {sorted(new)}. The sys.modules purge cannot protect a lazy "
            f"import — this forward may be using another repo's `utils`. "
            f"Vendor the repos under distinct package names before shipping.")
        return False
