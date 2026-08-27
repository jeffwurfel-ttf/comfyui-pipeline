"""
Ordering and co-scheduling. Reads only `depends_on` and `cost` off providers.

Two jobs:

  order()   topological sort on depends_on. STABLE: registration order is the
            tie-break, so the emitted order does not wobble between runs. That
            matters more than it looks — execution order fixes the sequence of
            CUDA calls, and the byte-identity contract is stated against a
            specific order.

  batches() groups the order into co-schedulable batches under a VRAM budget.
            A provider whose measured peak exceeds the budget REFUSES rather
            than being scheduled and OOMing: flow peaks at 13304 MiB at
            1920x1340, so two of it on a 24 GB card is not a thing to discover
            at runtime.
"""
from dataclasses import dataclass
from typing import Tuple

DEFAULT_BUDGET_MB = 20000        # 24 GB card, leaving CUDA context + headroom


class DependencyError(RuntimeError):
    pass


class BudgetError(RuntimeError):
    pass


def order(providers, want=None):
    """Stable topological sort. `want` optionally restricts to a subset, with
    dependencies pulled in automatically."""
    by_name = {p.name: p for p in providers}
    if want is not None:
        need, stack = set(), list(want)
        while stack:
            n = stack.pop()
            if n in need:
                continue
            if n not in by_name:
                raise DependencyError(
                    f"requested provider {n!r} is not registered; have "
                    f"{sorted(by_name)}")
            need.add(n)
            stack.extend(by_name[n].depends_on)
        pool = [p for p in providers if p.name in need]
    else:
        pool = list(providers)

    for p in pool:
        for d in p.depends_on:
            if d not in by_name:
                raise DependencyError(
                    f"provider {p.name!r} depends on {d!r}, which is not "
                    f"registered")

    out, done = [], set()
    remaining = list(pool)                    # registration order preserved
    while remaining:
        progressed = False
        for p in list(remaining):
            if all(d in done for d in p.depends_on):
                out.append(p)
                done.add(p.name)
                remaining.remove(p)
                progressed = True
        if not progressed:
            cyc = sorted(p.name for p in remaining)
            raise DependencyError(f"dependency cycle among {cyc}")
    return tuple(out)


@dataclass(frozen=True)
class Batch:
    providers: Tuple
    vram_mb: int

    @property
    def names(self):
        return tuple(p.name for p in self.providers)


def batches(ordered, budget_mb=DEFAULT_BUDGET_MB):
    """Group into co-schedulable batches without violating the budget or a
    dependency. A provider is never batched with something it depends on."""
    out, cur, cur_mb, placed = [], [], 0, set()
    for p in ordered:
        if p.cost.vram_mb > budget_mb:
            raise BudgetError(
                f"provider {p.name!r} needs {p.cost.vram_mb} MiB, over the "
                f"{budget_mb} MiB budget on its own. Lower the analysis "
                f"resolution or raise the budget deliberately; do not let it "
                f"reach the allocator.")
        dep_in_batch = any(d in {q.name for q in cur} for d in p.depends_on)
        if cur and (dep_in_batch or cur_mb + p.cost.vram_mb > budget_mb):
            out.append(Batch(tuple(cur), cur_mb))
            cur, cur_mb = [], 0
        cur.append(p)
        cur_mb += p.cost.vram_mb
        placed.add(p.name)
    if cur:
        out.append(Batch(tuple(cur), cur_mb))
    return tuple(out)


def plan(providers, want=None, budget_mb=DEFAULT_BUDGET_MB):
    o = order(providers, want)
    return o, batches(o, budget_mb)


def explain(providers, want=None, budget_mb=DEFAULT_BUDGET_MB):
    o, bs = plan(providers, want, budget_mb)
    lines = [f"order: {' -> '.join(p.name for p in o)}",
             f"budget: {budget_mb} MiB"]
    for i, b in enumerate(bs):
        lines.append(f"  batch {i}: {', '.join(b.names)}  ({b.vram_mb} MiB)")
    return "\n".join(lines)
