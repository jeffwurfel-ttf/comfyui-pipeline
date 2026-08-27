#!/usr/bin/env python
"""
Registry and scheduler contract.

The load-bearing one is licence completeness: a provider missing any LicenseRow
field must FAIL registration, not warn. Two licence traps have already been hit
in this project (DSINE's Imperial licence, VDA's Base/Large weights being
CC-BY-NC-4.0 while its code is Apache-2.0), and both were the kind of thing a
warning scrolls past.

Run:  python -m runner.tests.test_registry
"""
import sys

from .. import registry
from ..registry import (
    Cost, Dataset, Display, LicenseRow, Provider, ProxySpec, Schema,
    register, unregister,
)
from ..scheduler import BudgetError, DependencyError, batches, order

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'ok ' if cond else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    if not cond:
        FAIL.append(name)


def raises(fn, exc=AssertionError):
    try:
        fn()
        return False
    except exc:
        return True


GOOD_LIC = dict(model="m", code_license="Apache-2.0", weight_license="Apache-2.0",
                gated=False, verdict="CLEAN", source_url="https://x",
                date_checked="2026-08-26")


def mk(name, deps=(), vram=100, lic=None, env="_env"):
    return Provider(
        name=name, env=env, depends_on=deps,
        schema=Schema((Dataset(name, ("T",), "float16"),)),
        cost=Cost(vram_mb=vram),
        display=Display(dataset=name, proxies=(ProxySpec(name, lambda f, c: f),)),
        license=LicenseRow(**(lic or GOOD_LIC)),
        run=lambda ctx: None)


def main():
    print("registry / scheduler contract\n")

    print("licence completeness is an ASSERT, not a warning:")
    for field in ("model", "code_license", "weight_license", "source_url",
                  "date_checked"):
        bad = dict(GOOD_LIC); bad[field] = ""
        check(f"missing {field} refuses registration",
              raises(lambda b=bad: mk("tmp_lic", lic=b).validate()))
    bad = dict(GOOD_LIC); bad["verdict"] = "PROBABLY FINE"
    check("bogus verdict refused", raises(lambda: mk("t", lic=bad).validate()))
    bad = dict(GOOD_LIC); bad["gated"] = "unknown"
    check("non-bool `gated` refused ('unknown' is not an answer)",
          raises(lambda: mk("t", lic=bad).validate()))
    bad = dict(GOOD_LIC); bad["date_checked"] = "26/08/2026"
    check("malformed date refused", raises(lambda: mk("t", lic=bad).validate()))
    check("complete row validates", mk("t").validate() is None)

    print("\nprovider validation:")
    check("unknown env refused",
          raises(lambda: mk("t", env="somewhere").validate()))
    check("empty schema refused", raises(
        lambda: Provider(name="t", env="_env", schema=Schema(()),
                         cost=Cost(vram_mb=1),
                         display=Display(dataset="t",
                                         proxies=(ProxySpec("t", lambda f, c: f),)),
                         license=LicenseRow(**GOOD_LIC)).validate()))
    check("display with no proxies refused", raises(
        lambda: Provider(name="t", env="_env",
                         schema=Schema((Dataset("t", ("T",), "float16"),)),
                         cost=Cost(vram_mb=1),
                         display=Display(dataset="t", proxies=()),
                         license=LicenseRow(**GOOD_LIC)).validate()))

    print("\nscheduler ordering:")
    a, b, c = mk("a"), mk("b"), mk("c", deps=("a",))
    o = order([a, b, c])
    check("topological: dependency precedes dependent",
          o.index(c) > o.index(a), " -> ".join(p.name for p in o))
    check("stable: registration order is the tie-break",
          [p.name for p in o] == ["a", "b", "c"])
    check("repeatable across calls",
          [p.name for p in order([a, b, c])] == [p.name for p in o])
    check("missing dependency raises",
          raises(lambda: order([mk("x", deps=("nope",))]), DependencyError))
    d1 = mk("d1", deps=("d2",)); d2 = mk("d2", deps=("d1",))
    check("cycle raises", raises(lambda: order([d1, d2]), DependencyError))
    sub = order([a, b, c], want=["c"])
    check("requesting a dependent pulls its dependency in",
          [p.name for p in sub] == ["a", "c"], " -> ".join(p.name for p in sub))

    print("\nscheduler VRAM budgeting:")
    big = mk("big", vram=13304); small = mk("small", vram=900)
    bs = batches(order([small, big]), budget_mb=20000)
    check("fits in one batch under budget", len(bs) == 1, f"{[x.names for x in bs]}")
    bs = batches(order([big, mk("big2", vram=13304)]), budget_mb=20000)
    check("two 13.3 GB providers are NOT co-scheduled", len(bs) == 2,
          f"{[x.names for x in bs]} — would have OOMed a 24 GB card")
    check("a provider over budget alone REFUSES rather than OOMing",
          raises(lambda: batches(order([big]), budget_mb=1000), BudgetError))
    dep = order([mk("p"), mk("q", deps=("p",))])
    check("a dependent is never batched with its dependency",
          len(batches(dep, budget_mb=999999)) == 2,
          f"{[x.names for x in batches(dep, budget_mb=999999)]}")

    print("\nreal providers:")
    registry.load_all()
    names = [p.name for p in registry.all_providers()]
    check("depth, flow, normals registered",
          set(names) == {"depth", "flow", "normals"}, str(names))
    check("pose_vitpose is a spec-only stub and does NOT register",
          "pose2d" not in names and "pose_vitpose" not in names)
    nm = registry.get("normals")
    check("normals declares depends_on=('depth',)", nm.depends_on == ("depth",))
    check("depth carries window=32 (moved out of the chunker)",
          registry.get("depth").cost.window == 32)
    check("flow carries its measured 13304 MiB peak",
          registry.get("flow").cost.vram_mb == 13304)
    check("flow declares three proxies incl. sparse arrows",
          [s.suffix for s in registry.get("flow").display.proxies]
          == ["flow_mag", "flow_wheel", "flow_arrows"])
    for p in registry.all_providers():
        check(f"{p.name}: licence row complete", p.license.validate(p.name) is None)
    o = order(registry.all_providers())
    check("execution order", [p.name for p in o] == ["depth", "flow", "normals"],
          " -> ".join(p.name for p in o))

    print(f"\n{'ALL PASSED' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
