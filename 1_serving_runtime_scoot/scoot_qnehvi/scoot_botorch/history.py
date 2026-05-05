from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from .space import Config, ScootSearchSpace


HistoryItem = Dict[str, object]


def load_history(path: str | Path) -> List[HistoryItem]:
    path = Path(path)
    if not path.exists():
        return []
    return json.loads(path.read_text())


def save_history(path: str | Path, history: Iterable[HistoryItem]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(history), indent=2))


def seen_keys(space: ScootSearchSpace, history: Iterable[HistoryItem]) -> set[Tuple[object, ...]]:
    keys = set()
    for item in history:
        rec = item.get("rec") or []
        if rec:
            keys.add(space.key(rec[0]))
    return keys


def successful_items(history: Iterable[HistoryItem]) -> List[HistoryItem]:
    return [item for item in history if item.get("obj") is not None]


def build_training_tensors(space: ScootSearchSpace, history: Iterable[HistoryItem]):
    xs = []
    ys = []
    for item in successful_items(history):
        rec = item["rec"][0]
        obj = np.asarray(item["obj"], dtype=float).reshape(-1)
        if obj.size < 3:
            continue
        xs.append(space.encode(rec))
        # Stored objective is [-throughput, ttft, tpot]. BoTorch maximizes.
        ys.append([-obj[0], -obj[1], -obj[2]])
    if not xs:
        return None, None
    return torch.stack(xs).double(), torch.tensor(ys, dtype=torch.double)


def objective_from_benchmark_result(result: Dict[str, object]) -> List[List[float]]:
    return [
        [
            -1.0 * float(result["request_throughput"]),
            float(result["mean_ttft_ms"]),
            float(result["mean_tpot_ms"]),
        ]
    ]


def history_item(config: Config, obj, rec_time: float, run_time: float) -> HistoryItem:
    return {
        "rec": [dict(config)],
        "obj": None if obj is None else np.asarray(obj, dtype=float).tolist(),
        "rec_time": rec_time,
        "run_time": run_time,
    }
