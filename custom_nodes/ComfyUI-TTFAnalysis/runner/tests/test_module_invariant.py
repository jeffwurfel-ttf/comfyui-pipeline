#!/usr/bin/env python
"""
Guards the invariant the whole single-venv architecture rests on.

All three analysis repos ship a DIFFERENT top-level `utils` package, and
SEA-RAFT and DSINE both resolve the same dotted name `utils.utils`. The
sys.modules purge in runner.isolation contains that, but only because every
model-level import currently resolves AT IMPORT TIME.

The failure mode this exists to catch: a future dependency bump adds a lazy
`import utils` inside a forward pass. The purge would then hand that forward the
WRONG repo's utils, and the model would return silently wrong tensors instead of
raising. That must fail loudly here first.

  A. Each repo's `utils` resolves to a file INSIDE that repo while loaded.
  B. After the context manager exits, every colliding name is gone.
  C. (runner.isolation.no_lazy_colliding_import, used around forward passes)

Run:  python -m runner.tests.test_module_invariant

DSINE is exercised for the invariant only. It is NOT in the model set — rejected
on license, see .dev/LICENSE_AUDIT.md — but its repo is the worst offender, so
the guard keeps covering it when present.
"""
import sys
from pathlib import Path

from .. import paths
from ..isolation import COLLIDING, repo


def _mod_file(name):
    m = sys.modules.get(name)
    return Path(getattr(m, "__file__", "") or "").resolve()


def assert_inside(modname, repo_dir, label):
    f = _mod_file(modname)
    rd = (paths.REPOS / repo_dir).resolve()
    assert f and rd in f.parents, (
        f"[{label}] sys.modules['{modname}'] resolves to {f or '<missing>'}, "
        f"which is NOT inside {rd}. Cross-repo contamination.")
    print(f"  ok  {label}: {modname} -> {f.relative_to(paths.REPOS)}")


def assert_purged(label):
    leaked = sorted(n for n in sys.modules if n.split(".")[0] in COLLIDING)
    assert not leaked, f"[{label}] colliding names leaked past the purge: {leaked}"
    print(f"  ok  {label}: sys.modules clean ({len(COLLIDING)} names purged)")


def main():
    assert paths.REPOS.exists(), f"repos not found at {paths.REPOS}"
    print("module-invariant test")

    with repo(paths.REPOS / "Video-Depth-Anything"):
        from video_depth_anything.video_depth import VideoDepthAnything  # noqa
        import utils.util  # noqa
        assert_inside("utils.util", "Video-Depth-Anything", "VDA")
    assert_purged("VDA")

    with repo(paths.REPOS / "SEA-RAFT", paths.REPOS / "SEA-RAFT" / "core"):
        from raft import RAFT  # noqa
        import utils.utils  # noqa
        assert_inside("utils.utils", "SEA-RAFT", "SEA-RAFT")
    assert_purged("SEA-RAFT")

    dsine = paths.REPOS / "DSINE"
    if dsine.exists():
        with repo(dsine):
            import utils.utils  # noqa
            assert_inside("utils.utils", "DSINE", "DSINE")
        assert_purged("DSINE")
    else:
        print("  --  DSINE repo absent; skipped (invariant coverage only)")

    # ordering proof: the same dotted name must resolve differently per repo
    with repo(paths.REPOS / "SEA-RAFT", paths.REPOS / "SEA-RAFT" / "core"):
        import utils.utils  # noqa
        first = _mod_file("utils.utils")
    if dsine.exists():
        with repo(dsine):
            import utils.utils  # noqa
            second = _mod_file("utils.utils")
        assert first != second, (
            f"utils.utils resolved to the SAME file for SEA-RAFT and DSINE "
            f"({first}). The purge is not working; the second import reused the "
            f"first repo's cached module.")
        print("  ok  ordering: utils.utils resolves distinctly per repo")

    print("module-invariant test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
