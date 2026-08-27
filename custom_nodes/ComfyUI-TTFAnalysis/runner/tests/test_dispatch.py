#!/usr/bin/env python
"""
Proves env="main" dispatch actually crosses a process boundary.

No production provider uses env="main" yet — 2D pose will, and it is blocked on
the AGPL detector (see providers/pose_vitpose.py). An unexercised branch is a
lie, so this test registers a throwaway main-env provider, dispatches real work
to a DIFFERENT interpreter, checks the result, and unregisters it.

The proof that matters is `pid` and `executable`: if the callee reports the same
pid as the caller, no boundary was crossed and the test is worthless.

Run:  python -m runner.tests.test_dispatch
"""
import os
import sys

from .. import dispatch
from ..registry import (
    Cost, Dataset, Display, LicenseRow, Provider, ProxySpec, Schema,
    register, unregister,
)

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'ok ' if cond else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    if not cond:
        FAIL.append(name)


def throwaway_work(payload):
    """Runs in the MAIN env. Deliberately stdlib-only — the whole point of a
    separate env is that it may not have torch, h5py or the vendored repos."""
    xs = payload["values"]
    return {"n": len(xs), "mean": sum(xs) / len(xs),
            "pid": os.getpid(), "executable": sys.executable,
            "has_torch": "torch" in sys.modules}


def main():
    print("dispatch test\n")

    print("interpreters:")
    for env in ("_env", "main"):
        interp = dispatch.interpreter_for(env)
        print(f"  {env:<6} -> {interp}  (in-process: {dispatch.is_in_process(env)})")

    # ── the boundary itself ─────────────────────────────────────────────
    print("\nenv='main' crosses a process boundary:")
    me = os.getpid()
    rep = dispatch.call("main", "runner.tests.test_dispatch:throwaway_work",
                        {"values": [1, 2, 3, 4]})
    check("callee ran in a DIFFERENT process", rep["pid"] != me,
          f"caller pid {me}, callee pid {rep['pid']}")
    # abspath, not realpath — the venv's bin/python is a SYMLINK to the system
    # interpreter, so realpath reports the two environments as identical. That
    # is the exact bug this test caught in dispatch.py itself.
    check("callee used a different interpreter",
          os.path.abspath(rep["executable"]) != os.path.abspath(sys.executable),
          f"{sys.executable} -> {rep['executable']}")
    check("payload crossed intact", rep["n"] == 4 and abs(rep["mean"] - 2.5) < 1e-9,
          f"n={rep['n']} mean={rep['mean']}")
    check("main env is genuinely separate (no torch there)",
          rep["has_torch"] is False)

    # ── in-process fast path is semantically identical ──────────────────
    print("\nenv='_env' in-process fast path:")
    rep2 = dispatch.call("_env", "runner.tests.test_dispatch:throwaway_work",
                         {"values": [1, 2, 3, 4]})
    check("same result shape as the subprocess path",
          rep2["n"] == rep["n"] and abs(rep2["mean"] - rep["mean"]) < 1e-9)
    check("_env ran in-process (this interpreter)", rep2["pid"] == me)

    # ── errors surface, they do not vanish ──────────────────────────────
    print("\nfailure propagation:")
    try:
        dispatch.call("main", "runner.tests.test_dispatch:does_not_exist", {})
        ok = False
    except dispatch.DispatchError:
        ok = True
    check("missing target raises DispatchError", ok)
    try:
        dispatch.call("nope", "x:y", {})
        ok = False
    except dispatch.DispatchError:
        ok = True
    check("unknown env raises DispatchError", ok)

    # ── a real Provider declaring env='main' registers and dispatches ───
    print("\nthrowaway main-env PROVIDER:")
    p = Provider(
        name="throwaway_main",
        env="main",
        depends_on=(),
        schema=Schema((Dataset("throwaway", ("T",), "float16"),)),
        cost=Cost(vram_mb=0, window=1, gpu=False),
        display=Display(dataset="throwaway",
                        proxies=(ProxySpec("throwaway", lambda f, c: f),)),
        license=LicenseRow(
            model="throwaway", code_license="n/a", weight_license="n/a",
            gated=False, verdict="CLEAN", source_url="internal",
            date_checked="2026-08-26"),
        run=lambda ctx: None,
    )
    register(p)
    try:
        check("provider with env='main' registers", "throwaway_main" in
              [q.name for q in __import__("runner.registry", fromlist=["x"]).all_providers()])
        out = dispatch.call(p.env, "runner.tests.test_dispatch:throwaway_work",
                            {"values": [10, 20]})
        check("its work dispatches out-of-process", out["pid"] != me,
              f"pid {out['pid']}")
    finally:
        unregister("throwaway_main")
    check("throwaway unregistered afterwards",
          "throwaway_main" not in
          [q.name for q in __import__("runner.registry", fromlist=["x"]).all_providers()])

    print(f"\n{'ALL PASSED' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
