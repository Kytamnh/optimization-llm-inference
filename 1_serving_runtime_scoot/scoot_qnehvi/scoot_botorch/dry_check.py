from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .constraints import hard_constraint_violations, is_valid_config
from .pareto import pareto_indices
from .space import ScootSearchSpace
from .suggest import suggest_sobol


def run_checks(all_trials_csv: Path | None = None):
    space = ScootSearchSpace(gpu_nums=4, min_world_size=1, max_sequence_length=4096)

    ref = space.reference_config()
    assert is_valid_config(space, ref)
    encoded = space.encode(ref)
    decoded = space.decode(encoded.tolist())
    assert space.key(ref) == space.key(decoded), (ref, decoded)

    invalid = {
        "tp": 3,
        "max_num_seqs": 4096,
        "max_num_batched_tokens": 64,
        "block_size": 8,
        "enable_chunked_prefill": True,
        "scheduler_delay_factor": 2.0,
        "enable_prefix_caching": True,
        "disable_custom_all_reduce": False,
        "use_v2_block_manager": True,
    }
    assert hard_constraint_violations(space, invalid)

    repaired = space.repair(
        {
            "tp": 4,
            "max_num_seqs": 4096,
            "max_num_batched_tokens": 64,
            "block_size": 8,
            "enable_chunked_prefill": False,
            "scheduler_delay_factor": 2.0,
            "enable_prefix_caching": False,
            "disable_custom_all_reduce": False,
            "use_v2_block_manager": True,
        }
    )
    assert not hard_constraint_violations(space, repaired)

    seen = set()
    configs = suggest_sobol(space, seen, n=16)
    keys = [space.key(cfg) for cfg in configs]
    assert len(keys) == len(set(keys))

    frontier_count = None
    if all_trials_csv and all_trials_csv.exists():
        rows = list(csv.DictReader(all_trials_csv.open()))
        objectives = [
            [
                -float(row["request_throughput"]),
                float(row["mean_ttft_ms"]),
                float(row["mean_tpot_ms"]),
            ]
            for row in rows
        ]
        frontier = pareto_indices(objectives)
        frontier_count = len(frontier)
        assert frontier_count == 6, frontier

    return {
        "reference": ref,
        "sobol_unique": len(keys),
        "frontier_count": frontier_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Dry checks for the qNEHVI SCOOT variant")
    parser.add_argument("--all_trials_csv", type=Path, default=None)
    args = parser.parse_args()
    result = run_checks(args.all_trials_csv)
    print(result)


if __name__ == "__main__":
    main()
