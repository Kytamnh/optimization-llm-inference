"""Shared benchmark helpers for combined-pipeline stages.

Must be imported from inside the qNEHVI repo
(scoot_run/qnehvi/SCOOT-SLO-Oriented-Performance-Tuning) so that
`benchmark_pipeline.sh`, `tuner_conf/conf.json`, `scoot_botorch`, and
`utils` are reachable in the working directory.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

REPO_DIR = Path(os.getcwd())
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scoot_botorch.space import Config  # noqa: E402
from utils import check_port  # noqa: E402

RES_DIR = os.environ.get("SCOOT_RES_DIR", "./tune_res")
RES_DIR_PREFIX = os.environ.get("SCOOT_RES_DIR_PREFIX", "combined")
RAW_DIR = os.path.join(RES_DIR, RES_DIR_PREFIX, "raw")
os.makedirs(RAW_DIR, exist_ok=True)


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


def run_benchmark_pipeline(config: Config, gpu_nums: int, args) -> None:
    combo = config_to_combination(config)
    tp_size = combo[0]
    ports = ",".join([str(8000 + i) for i in range(int(gpu_nums / tp_size))])
    gpus = [str(i) for i in range(gpu_nums)]
    grouped_gpus = [",".join(gpus[i : i + tp_size]) for i in range(0, gpu_nums, tp_size)]
    grouped_gpus_string = "#".join(grouped_gpus)
    raw_file_path = os.path.join(
        RAW_DIR,
        f"benchmark_tp_{combo[0]}_mns_{combo[1]}_mnbt_{combo[2]}_bs_{combo[3]}.txt",
    )
    cmd = (
        f"bash benchmark_pipeline.sh {args.model_path} {args.dataset_path} {args.request_rate} "
        f"{args.num_requests} {args.pressure_test} 0 {combo[0]} 1 "
        f"{combo[1]} {combo[2]} {combo[5]} {combo[3]} "
        f"{ports} {grouped_gpus_string} {args.model} {combo[4]} {args.dataset_name} "
        f"{combo[6]} {combo[7]} {combo[8]} 2>&1 | tee {raw_file_path}"
    )
    logging.info(cmd)
    os.system(cmd)


def result_matches(result: Dict[str, object], config: Config) -> bool:
    return config_to_combination(config) == (
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


def find_matching_result(
    res_dir_path: str, config: Config, started_at: float
) -> Optional[Dict[str, object]]:
    paths = []
    for root, _, files in os.walk(res_dir_path):
        for name in files:
            if name.startswith("vllm") and name.endswith(".json"):
                p = os.path.join(root, name)
                if os.path.getmtime(p) >= started_at:
                    paths.append(p)
    paths.sort(key=os.path.getmtime, reverse=True)
    for p in paths:
        try:
            with open(p) as f:
                r = json.load(f)
        except Exception:
            continue
        if result_matches(r, config):
            return r
    logging.warning(f"No matching benchmark result found for {config}")
    return None


def cleanup_servers(gpu_nums: int, min_world_size: int) -> None:
    os.system('pgrep -f "vllm.entrypoints.api_server" | xargs -r kill -9')
    ports = [8000 + i for i in range(int(gpu_nums / min_world_size))]
    for port in ports:
        for _ in range(3):
            if check_port(port):
                logging.info(f"port {port} not cleared; closing")
                os.system(f"lsof -t -i:{port} | xargs -r kill -9")
            else:
                break
        assert not check_port(port), f"Port {port} still occupied"


def evaluate(
    config: Config, gpu_nums: int, res_dir_path: str, args, min_world_size: int
):
    cleanup_servers(gpu_nums, min_world_size)
    time.sleep(5)
    started = time.time()
    run_benchmark_pipeline(config, gpu_nums, args)
    return find_matching_result(res_dir_path, config, started)
