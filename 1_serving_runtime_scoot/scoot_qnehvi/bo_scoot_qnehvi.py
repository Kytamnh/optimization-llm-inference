import argparse
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor

from scoot_botorch.acquisition import build_qnehvi
from scoot_botorch.constraints import is_valid_config
from scoot_botorch.history import (
    build_training_tensors,
    history_item,
    load_history,
    objective_from_benchmark_result,
    save_history,
    seen_keys,
    successful_items,
)
from scoot_botorch.models import fit_objective_model
from scoot_botorch.pareto import pareto_indices
from scoot_botorch.space import Config, ScootSearchSpace
from scoot_botorch.suggest import suggest_qnehvi, suggest_sobol
from utils import check_port, gen_res_dir_path, get_ref_config


RES_DIR_PREFIX = os.environ.get("SCOOT_RES_DIR_PREFIX", "qnehvi")
RES_DIR = os.environ.get("SCOOT_RES_DIR", "./tune_res")
# Base TCP port used to host vLLM api_servers. Overridable via SCOOT_BASE_PORT
# so concurrent jobs on shared multi-tenant nodes (e.g. tron rtxa5000 nodes)
# don't collide on the default 8000.
BASE_PORT = int(os.environ.get("SCOOT_BASE_PORT", 8000))
LOG_DIR = os.path.join(RES_DIR, "logs")
RAW_DIR = os.path.join(RES_DIR, "raw")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--request_rate", type=int, default=20)
    parser.add_argument("--num_requests", type=int, default=1000)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--total_resource", type=str, required=True)
    parser.add_argument("--bo_loop", type=int, default=50)
    parser.add_argument("--bo_batch_size", type=int, default=1)
    parser.add_argument("--exp_num", type=int, default=1)
    parser.add_argument("--dataset_name", type=str, default="sharegpt")
    parser.add_argument("--num_obj", type=int, default=3)
    parser.add_argument("--pressure_test", action="store_true")
    parser.add_argument("--sobol_init", type=int, default=10)
    parser.add_argument("--candidate_pool_size", type=int, default=2048)
    return parser


def config_to_combination(config: Config) -> Tuple:
    return (
        int(config["tp"]),
        int(config["max_num_seqs"]),
        int(config["max_num_batched_tokens"]),
        int(config["block_size"]),
        str(config["enable_chunked_prefill"]),
        float(config["scheduler_delay_factor"]),
        str(config["enable_prefix_caching"]),
        str(config["disable_custom_all_reduce"]),
        str(config["use_v2_block_manager"]),
    )


def run_benchmark_pipeline(config: Config, gpu_nums: int, args):
    combination = config_to_combination(config)
    tp_size = combination[0]
    ports = ",".join([str(BASE_PORT + i) for i in range(int(gpu_nums / tp_size))])
    gpus = [str(i) for i in range(gpu_nums)]
    grouped_gpus = [",".join(gpus[i : i + tp_size]) for i in range(0, gpu_nums, tp_size)]
    grouped_gpus_string = "#".join(grouped_gpus)
    raw_file_path = os.path.join(
        RAW_DIR,
        f"benchmark_tp_{combination[0]}_mns_{combination[1]}_mnbt_{combination[2]}_bs_{combination[3]}.txt",
    )

    command = (
        f"bash benchmark_pipeline.sh {args.model_path} {args.dataset_path} {args.request_rate} "
        f"{args.num_requests} {args.pressure_test} {0} {combination[0]} {1} "
        f"{combination[1]} {combination[2]} {combination[5]} {combination[3]} "
        f"{ports} {grouped_gpus_string} {args.model} {combination[4]} {args.dataset_name} "
        f"{combination[6]} {combination[7]} {combination[8]} 2>&1 | tee {raw_file_path}"
    )
    logging.info(command)
    os.system(command)


def result_matches_config(result: Dict[str, object], config: Config) -> bool:
    combination = config_to_combination(config)
    return combination == (
        int(result["tp"]),
        int(result["max_num_seqs"]),
        int(result["max_num_batched_tokens"]),
        int(result["block_size"]),
        str(result["enable_chunked_prefill"]),
        float(result["scheduler_delay_factor"]),
        str(result["enable_prefix_caching"]),
        str(result["disable_custom_all_reduce"]),
        str(result["use_v2_block_manager"]),
    )


def find_matching_result(res_dir_path: str, config: Config, started_at: float) -> Optional[Dict[str, object]]:
    result_files = []
    for root, _, files in os.walk(res_dir_path):
        for name in files:
            if not (name.startswith("vllm") and name.endswith(".json")):
                continue
            path = os.path.join(root, name)
            if os.path.getmtime(path) >= started_at:
                result_files.append(path)

    result_files.sort(key=os.path.getmtime, reverse=True)
    for path in result_files:
        try:
            with open(path, "r") as f:
                result = json.load(f)
        except Exception:
            logging.warning(f"Failed to read benchmark result {path}: {traceback.format_exc()}")
            continue
        if result_matches_config(result, config):
            return result

    logging.warning(f"No matching benchmark result found for {config}; checked {len(result_files)} files")
    return None


def cleanup_servers(gpu_nums: int, min_world_size: int):
    os.system('pgrep -f "vllm.entrypoints.api_server" | xargs -r kill -9')
    ports = ",".join([str(BASE_PORT + i) for i in range(int(gpu_nums / min_world_size))])
    for port in ports.split(","):
        for _ in range(3):
            if check_port(int(port)):
                logging.info(f"port {int(port)} is not cleared. Closing it.")
                os.system(f"lsof -t -i:{int(port)} | xargs -r kill -9")
            else:
                break
        assert not check_port(int(port)), "A port is still occupied; cannot continue safely"


def evaluate_config(config: Config, gpu_nums: int, res_dir_path: str, args, min_world_size: int):
    cleanup_servers(gpu_nums, min_world_size)
    time.sleep(5)
    started_at = time.time()
    run_benchmark_pipeline(config, gpu_nums, args)
    result = find_matching_result(res_dir_path, config, started_at)
    if result is None:
        return None
    return objective_from_benchmark_result(result)


def compute_delta_and_continuous_right(history):
    delta = 0.5
    continuous_right = 0
    for idx, item in enumerate(history):
        if idx < 9:
            continue
        if item.get("obj") is None:
            delta = round(min(0.75, max(delta + 0.05, 0.5)), 3)
            continuous_right = 0
        else:
            continuous_right += 1
            if continuous_right >= 5:
                continuous_right -= 5
                delta = round(max(0.25, delta - 0.05), 3)
    return delta, continuous_right


def make_rf_gate(space: ScootSearchSpace, history, delta: float):
    if not any(item.get("obj") is None for item in history):
        return None
    xs = []
    ys = []
    for item in history:
        rec = item.get("rec") or []
        if not rec:
            continue
        xs.append(space.encode(rec[0]).tolist())
        ys.append(0 if item.get("obj") is None else 1)
    if len(xs) < 4 or len(set(ys)) < 2:
        return None
    model = RandomForestRegressor(random_state=42)
    model.fit(xs, ys)

    def accept(config: Config) -> bool:
        return float(model.predict([space.encode(config).tolist()])[0]) >= delta

    return accept


def write_pareto_summary(path: Path, history):
    rows = [item for item in successful_items(history) if item.get("obj") is not None]
    objectives = [np.asarray(item["obj"], dtype=float).reshape(-1).tolist() for item in rows]
    frontier = [rows[i] for i in pareto_indices(objectives)]
    path.write_text(json.dumps(frontier, indent=2))


def select_next_config(space, history, args, iteration_idx: int):
    seen = seen_keys(space, history)
    if not history:
        return space.reference_config(), "reference"

    sobol_target = min(args.bo_loop, max(1, args.sobol_init))
    if len(history) < sobol_target:
        return suggest_sobol(space, seen, 1)[0], "sobol"

    train_x, train_y = build_training_tensors(space, history)
    if train_x is None or train_y is None or train_x.shape[0] < 2:
        return suggest_sobol(space, seen, 1)[0], "sobol_fallback_no_success"
    model = fit_objective_model(train_x, train_y, space.categorical_dims)
    acq = build_qnehvi(model, train_x, train_y)
    delta, _ = compute_delta_and_continuous_right(history)
    rf_gate = make_rf_gate(space, history, delta)
    cfg = suggest_qnehvi(
        space,
        acq,
        seen,
        accept_config=rf_gate,
        pool_size=args.candidate_pool_size,
        seed=1000 + iteration_idx,
    )
    return cfg, "qnehvi" if acq is not None else "sobol_fallback"


def main(args):
    if args.num_obj != 3:
        raise ValueError("bo_scoot_qnehvi.py currently implements the 3-objective SCOOT variant only")
    if args.bo_batch_size != 1:
        raise ValueError("bo_scoot_qnehvi.py v1 supports bo_batch_size=1")

    gpu_nums = torch.cuda.device_count()
    assert gpu_nums % 2 == 0 or gpu_nums == 1

    logging.basicConfig(
        filename=os.path.join(
            LOG_DIR,
            f"qnehvi_{args.bo_batch_size}_{args.model}_{args.total_resource}_"
            f"num_requests{args.num_requests}_request_rate{args.request_rate}_{args.dataset_name}.log",
        ),
        level=logging.INFO,
    )

    min_world_size = get_ref_config("min_world_size")
    max_sequence_length = get_ref_config("max_sequence_length")
    space = ScootSearchSpace(
        gpu_nums=gpu_nums,
        min_world_size=min_world_size,
        max_sequence_length=max_sequence_length,
    )
    logging.info(f"Input tuning arguments: {args}")
    logging.info(f"Search space tp values: {space.tp_values}")

    for exp in range(args.exp_num):
        res_dir_path = gen_res_dir_path(
            args.model,
            args.request_rate,
            args.num_requests,
            args.total_resource,
            args.dataset_name,
            RES_DIR,
            exp=exp,
            bo=True,
            dir_prefix=RES_DIR_PREFIX,
        )
        os.environ["RES_DIR_PATH"] = res_dir_path
        history_path = Path(res_dir_path) / "rec_history_qnehvi.json"
        pareto_path = Path(res_dir_path) / "pareto_frontier_qnehvi.json"
        history = load_history(history_path)

        print(res_dir_path)
        print(f"Starting qNEHVI loop with {len(history)} existing observations")

        while len(history) < args.bo_loop:
            iteration = len(history) + 1
            suggest_start = time.time()
            config, source = select_next_config(space, history, args, iteration)
            rec_time = time.time() - suggest_start

            if not is_valid_config(space, config):
                raise ValueError(f"Internal error: invalid config selected: {config}")

            logging.info(f"Iteration {iteration}: source={source}, config={config}")
            run_start = time.time()
            y = evaluate_config(config, gpu_nums, res_dir_path, args, min_world_size)
            run_time = time.time() - run_start

            history.append(history_item(config, y, rec_time, run_time))
            save_history(history_path, history)
            write_pareto_summary(pareto_path, history)

            failed_num = len([item for item in history if item.get("obj") is None])
            succeed_num = len(history) - failed_num
            delta, continuous_right = compute_delta_and_continuous_right(history)
            logging.info(
                f"total_tune_num: {len(history)}, succeed_num: {succeed_num}, failed_num: {failed_num}"
            )
            logging.info(f"For iteration {iteration}, BO time cost is {rec_time}")
            logging.info(f"For iteration {iteration}, rec config is {config}")
            logging.info(f"For iteration {iteration}, rec obj is {y}")
            logging.info(f"After {iteration} iterations, threshold delta is {delta}")
            logging.info(f"After {iteration} iterations, continuous right number is {continuous_right}")
            print(
                f"iteration={iteration} source={source} obj={y} "
                f"succeed={succeed_num} failed={failed_num}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct BoTorch qNEHVI SCOOT variant")
    parser = add_args(parser)
    main(parser.parse_args())
