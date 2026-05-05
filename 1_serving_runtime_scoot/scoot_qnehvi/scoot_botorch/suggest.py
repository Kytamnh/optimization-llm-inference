from __future__ import annotations

import time
from typing import Callable, Iterable, List, Tuple

import torch

from .space import Config, ScootSearchSpace


def suggest_sobol(space: ScootSearchSpace, seen: Iterable[Tuple[object, ...]], n: int = 1) -> List[Config]:
    return space.sobol_configs(n, skip=len(set(seen)) + 17, seen=seen)


def suggest_qnehvi(
    space: ScootSearchSpace,
    acq_func,
    seen: Iterable[Tuple[object, ...]],
    *,
    accept_config: Callable[[Config], bool] | None = None,
    pool_size: int = 2048,
    seed: int | None = None,
) -> Config:
    if acq_func is None:
        return suggest_sobol(space, seen, 1)[0]

    seen_set = set(seen)
    seed = int(seed if seed is not None else time.time())
    candidates = space.candidate_pool(pool_size, seen=seen_set, seed=seed)
    if accept_config is not None:
        candidates = [cfg for cfg in candidates if accept_config(cfg)]
    if not candidates:
        return suggest_sobol(space, seen_set, 1)[0]

    x = torch.stack([space.encode(cfg) for cfg in candidates]).double()
    scores = []
    try:
        with torch.no_grad():
            for start in range(0, x.shape[0], 256):
                batch = x[start : start + 256].unsqueeze(1)
                scores.append(acq_func(batch).detach().cpu())
        score = torch.cat(scores)
        best_idx = int(torch.argmax(score).item())
        return candidates[best_idx]
    except Exception:
        return suggest_sobol(space, seen_set, 1)[0]
