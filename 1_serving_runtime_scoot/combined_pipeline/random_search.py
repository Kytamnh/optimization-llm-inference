"""Random-search baseline over the SCOOT 9-parameter search space.

Run from the qNEHVI repo so `benchmark_pipeline.sh`, `tuner_conf`,
`scoot_botorch`, and `utils` are all on the path.
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench_runner import cleanup_servers, evaluate  # noqa: E402

REPO_DIR = Path(os.getcwd())
sys.path.insert(0, str(REPO_DIR))

from scoot_botorch.space import ScootSearchSpace  # noqa: E402
from scoot_botorch.history import history_item, load_history, save_history  # noqa: E402
from utils import gen_res_dir_path, get_ref_config  # noqa: E402

RES_DIR_PREFIX = os.environ.get("SCOOT_RES_DIR_PREFIX", "random")
RES_DIR = os.environ.get("SCOOT_RES_DIR", "./tune_res")
LOG_DIR = os.path.join(RES_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="Random-search baseline over the SCOOT search space"
    )
    p.add_argument("--model_path", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--dataset_name", default="sharegpt")
    p.add_argument("--model", required=True)
    p.add_argument("--total_resource", required=True)
    p.add_argument("--request_rate", type=int, default=5)
    p.add_argument("--num_requests", type=int, default=1000)
    p.add_argument("--num_trials", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pressure_test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    gpu_nums = torch.cuda.device_count()
    assert gpu_nums >= 1, "Random search needs at least one CUDA GPU"

    logging.basicConfig(
        filename=os.path.join(
            LOG_DIR, f"random_{args.model}_{args.total_resource}.log"
        ),
        level=logging.INFO,
    )

    min_world_size = get_ref_config("min_world_size")
    max_seq_len = get_ref_config("max_sequence_length")
    space = ScootSearchSpace(
        gpu_nums=gpu_nums,
        min_world_size=min_world_size,
        max_sequence_length=max_seq_len,
    )

    res_dir_path = gen_res_dir_path(
        args.model,
        args.request_rate,
        args.num_requests,
        args.total_resource,
        args.dataset_name,
        RES_DIR,
        exp=0,
        bo=True,
        dir_prefix=RES_DIR_PREFIX,
    )
    os.environ["RES_DIR_PATH"] = res_dir_path
    history_path = Path(res_dir_path) / "rec_history_random.json"
    history = load_history(history_path)

    seen = set()
    for item in history:
        rec = item.get("rec") or []
        if rec:
            seen.add(space.key(rec[0]))

    print(f"Random search: target {args.num_trials} trials, resuming at {len(history)}")
    print(f"Search space tp values: {space.tp_values}")

    while len(history) < args.num_trials:
        cfg = None
        for _ in range(256):
            candidate = space.random_config()
            k = space.key(candidate)
            if k not in seen:
                seen.add(k)
                cfg = candidate
                break
        if cfg is None:
            print("Search space appears exhausted; stopping early")
            break

        iteration = len(history) + 1
        print(f"Iteration {iteration}: {cfg}")
        start = time.time()
        result = evaluate(cfg, gpu_nums, res_dir_path, args, min_world_size)
        run_time = time.time() - start

        y = None
        if result is not None:
            y = [
                -1.0 * float(result["request_throughput"]),
                float(result["mean_ttft_ms"]),
                float(result["mean_tpot_ms"]),
            ]
        history.append(history_item(cfg, y, 0.0, run_time))
        save_history(history_path, history)

        failed = sum(1 for h in history if h.get("obj") is None)
        succeed = len(history) - failed
        print(f"iteration={iteration} obj={y} succeed={succeed} failed={failed}")
        logging.info(
            f"iteration={iteration} cfg={cfg} obj={y} succeed={succeed} failed={failed}"
        )

    cleanup_servers(gpu_nums, min_world_size)
    print(f"Random search complete: {len(history)} trials")


if __name__ == "__main__":
    main()
