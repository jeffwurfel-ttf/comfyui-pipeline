"""
Environment dispatch — maps `provider.env` to an interpreter and crosses the
process boundary when it has to.

This is the whole point of the provider model. The pipeline already runs two
incompatible Python environments: the main ComfyUI env (torch 2.4.1, the node
runtime) and isolated `_env` venvs for anything with conflicting pins. Today
every analysis provider lives in `_env`, so nothing crosses. The moment a
provider lives in `main` — 2D pose is the obvious one, ViTPose is already wired
there — the boundary is real.

`env="main"` is therefore implemented and TESTED even with no production
provider using it. An untested branch is a lie; see
runner/tests/test_dispatch.py, which exercises the real subprocess path.

Interpreters:
  _env   TTF_ANALYSIS_ENV_PYTHON, else the current interpreter
  main   TTF_ANALYSIS_MAIN_PYTHON, else the first system python3 found

If the resolved interpreter IS the current one, the call runs in-process. That
is an optimisation, not a semantic difference — `call()` behaves identically
either way, which is what makes the in-process fast path safe.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

RUNNER_DIR = Path(__file__).resolve().parent
PKG_ROOT = RUNNER_DIR.parent


class DispatchError(RuntimeError):
    pass


def _first(*cands):
    for c in cands:
        if not c:
            continue
        p = shutil.which(c) if not os.path.isabs(c) else (c if os.path.exists(c) else None)
        if p:
            return p
    return None


def interpreter_for(env):
    if env == "_env":
        return os.environ.get("TTF_ANALYSIS_ENV_PYTHON") or sys.executable
    if env == "main":
        p = _first(os.environ.get("TTF_ANALYSIS_MAIN_PYTHON"),
                   "/usr/bin/python3.11", "/usr/bin/python3", "python3")
        if not p:
            raise DispatchError(
                "env='main' requested but no main interpreter found. Set "
                "TTF_ANALYSIS_MAIN_PYTHON.")
        return p
    raise DispatchError(f"unknown env {env!r}")


def _same_interpreter(a, b):
    """Identity test for environments — abspath, NOT realpath.

    A venv's bin/python is usually a symlink to the system interpreter, so
    realpath() reports `_env/bin/python` and `/usr/bin/python3.11` as the same
    thing. They are not: they have different sys.prefix and different
    site-packages, which is the entire reason the two environments exist. The
    first cut of this used realpath and silently ran every env='main' call
    in-process — the dispatch test caught it because the callee reported the
    caller's own pid.
    """
    return os.path.abspath(a) == os.path.abspath(b)


def is_in_process(env):
    try:
        return _same_interpreter(interpreter_for(env), sys.executable)
    except DispatchError:
        return False


def call(env, target, payload, timeout=1800):
    """Invoke `target` ("pkg.mod:func") with `payload` in `env`, return its
    JSON-able result.

    Same semantics in-process and out-of-process, deliberately: the callee sees
    a dict and returns a dict either way, so a provider cannot accidentally
    depend on sharing memory with the caller.
    """
    interp = interpreter_for(env)
    if _same_interpreter(interp, sys.executable):
        return _invoke(target, payload)

    req = json.dumps({"target": target, "payload": payload})
    proc = subprocess.run(
        [interp, "-c", _WORKER, req],
        capture_output=True, text=True, timeout=timeout,
        cwd=str(PKG_ROOT),
        env={**os.environ, "PYTHONPATH": str(PKG_ROOT) + os.pathsep
             + os.environ.get("PYTHONPATH", "")})
    if proc.returncode != 0:
        raise DispatchError(
            f"env={env!r} ({interp}) target={target!r} exited "
            f"{proc.returncode}\nstderr:\n{proc.stderr[-2000:]}")
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        res = json.loads(line)
    except Exception as e:
        raise DispatchError(
            f"env={env!r} target={target!r} did not return JSON: {e}\n"
            f"stdout tail:\n{proc.stdout[-2000:]}")
    if not res.get("ok"):
        raise DispatchError(
            f"env={env!r} target={target!r} raised: {res.get('error')}")
    return res["result"]


def _invoke(target, payload):
    """In-process path. Errors are wrapped in DispatchError so a caller sees
    the SAME exception type whichever side of the boundary the callee ran on —
    the subprocess path already did this, and the mismatch meant a target typo
    surfaced as AttributeError in-process and DispatchError out-of-process."""
    import importlib
    mod, _, fn = target.partition(":")
    if not fn:
        raise DispatchError(f"target {target!r} must be 'module:function'")
    try:
        obj = getattr(importlib.import_module(mod), fn)
    except (ImportError, AttributeError) as e:
        raise DispatchError(f"target {target!r} not resolvable: {e}") from e
    try:
        return obj(payload)
    except Exception as e:
        raise DispatchError(f"target {target!r} raised: {e!r}") from e


# Kept as source rather than a file so the worker cannot drift from the caller.
_WORKER = r"""
import json, sys, importlib, traceback
req = json.loads(sys.argv[1])
try:
    mod, _, fn = req["target"].partition(":")
    out = getattr(importlib.import_module(mod), fn)(req["payload"])
    print(json.dumps({"ok": True, "result": out}))
except Exception:
    print(json.dumps({"ok": False, "error": traceback.format_exc()}))
    sys.exit(0)
"""


def probe(env):
    """What interpreter would this env use, and does it work? Used by the
    dispatch test and worth printing in a manifest."""
    interp = interpreter_for(env)
    ver = call(env, "runner.dispatch:_self_report", {})
    return {"env": env, "interpreter": interp,
            "in_process": is_in_process(env), **ver}


def _self_report(payload):
    return {"executable": sys.executable,
            "version": sys.version.split()[0],
            "pid": os.getpid(),
            "echo": payload.get("echo")}
