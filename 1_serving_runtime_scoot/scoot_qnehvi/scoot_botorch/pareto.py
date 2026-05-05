from __future__ import annotations

from typing import Iterable, List


def dominates(a, b) -> bool:
    # Objectives are stored as [-throughput, ttft, tpot], all minimized.
    better_or_equal = all(x <= y for x, y in zip(a, b))
    strictly_better = any(x < y for x, y in zip(a, b))
    return better_or_equal and strictly_better


def pareto_indices(objectives: Iterable[Iterable[float]]) -> List[int]:
    objs = [list(obj) for obj in objectives]
    frontier = []
    for i, obj in enumerate(objs):
        if not any(i != j and dominates(other, obj) for j, other in enumerate(objs)):
            frontier.append(i)
    return frontier
