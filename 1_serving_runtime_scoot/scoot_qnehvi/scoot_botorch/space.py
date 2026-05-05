from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


Config = Dict[str, object]


@dataclass(frozen=True)
class ScootSearchSpace:
    gpu_nums: int
    min_world_size: int
    max_sequence_length: int

    def __post_init__(self):
        if self.gpu_nums < 1:
            raise ValueError("gpu_nums must be positive")
        if self.min_world_size < 1:
            raise ValueError("min_world_size must be positive")

    @property
    def tp_values(self) -> List[int]:
        values = []
        value = 1
        while value <= self.gpu_nums:
            if value >= self.min_world_size and self.gpu_nums % value == 0:
                values.append(value)
            value *= 2
        return values or [self.gpu_nums]

    @property
    def max_num_seqs_values(self) -> List[int]:
        return [2**p for p in range(6, 14)]

    @property
    def block_size_values(self) -> List[int]:
        return [8, 16, 32]

    @property
    def scheduler_values(self) -> List[float]:
        return [round(i / 10.0, 1) for i in range(0, 21, 2)]

    @property
    def max_num_batched_tokens_upper(self) -> int:
        return max(8192, self.max_sequence_length * 2)

    @property
    def dim(self) -> int:
        return 9

    @property
    def categorical_dims(self) -> List[int]:
        return [0, 3, 4, 6, 7, 8]

    def reference_config(self) -> Config:
        return self.repair(
            {
                "tp": self.min_world_size,
                "max_num_seqs": 256,
                "max_num_batched_tokens": max(4096, self.max_sequence_length),
                "block_size": 16,
                "enable_chunked_prefill": False,
                "scheduler_delay_factor": 0.0,
                "enable_prefix_caching": False,
                "disable_custom_all_reduce": False,
                "use_v2_block_manager": False,
            }
        )

    def key(self, config: Config) -> Tuple[object, ...]:
        cfg = self.repair(config)
        return (
            int(cfg["tp"]),
            int(cfg["max_num_seqs"]),
            int(cfg["max_num_batched_tokens"]),
            int(cfg["block_size"]),
            bool(cfg["enable_chunked_prefill"]),
            round(float(cfg["scheduler_delay_factor"]), 3),
            bool(cfg["enable_prefix_caching"]),
            bool(cfg["disable_custom_all_reduce"]),
            bool(cfg["use_v2_block_manager"]),
        )

    def encode(self, config: Config, *, dtype=torch.double) -> torch.Tensor:
        cfg = self.repair(config)
        tp_idx = self.tp_values.index(int(cfg["tp"]))
        block_idx = self.block_size_values.index(int(cfg["block_size"]))
        sched_idx = self.scheduler_values.index(round(float(cfg["scheduler_delay_factor"]), 1))

        seq_log = math.log2(int(cfg["max_num_seqs"]))
        seq_norm = (seq_log - 6.0) / (13.0 - 6.0)

        mbt_log = math.log2(int(cfg["max_num_batched_tokens"]))
        upper_log = math.log2(self.max_num_batched_tokens_upper)
        mbt_norm = (mbt_log - 6.0) / (upper_log - 6.0)

        values = [
            float(tp_idx),
            float(seq_norm),
            float(mbt_norm),
            float(block_idx),
            1.0 if cfg["enable_chunked_prefill"] else 0.0,
            float(sched_idx) / (len(self.scheduler_values) - 1),
            1.0 if cfg["enable_prefix_caching"] else 0.0,
            1.0 if cfg["disable_custom_all_reduce"] else 0.0,
            1.0 if cfg["use_v2_block_manager"] else 0.0,
        ]
        return torch.tensor(values, dtype=dtype)

    def decode(self, x: Sequence[float]) -> Config:
        vals = [float(v) for v in x]
        tp = self.tp_values[_round_index(vals[0], len(self.tp_values))]
        seq_power = round(_scale(vals[1], 0.0, 1.0, 6.0, 13.0))
        max_num_seqs = 2 ** int(_clip(seq_power, 6, 13))

        upper_log = math.log2(self.max_num_batched_tokens_upper)
        mbt_log = _scale(vals[2], 0.0, 1.0, 6.0, upper_log)
        max_num_batched_tokens = int(round(2**mbt_log))

        block_size = self.block_size_values[_round_index(vals[3], len(self.block_size_values))]
        enable_chunked_prefill = vals[4] >= 0.5
        scheduler_delay_factor = self.scheduler_values[
            _round_index(vals[5] * (len(self.scheduler_values) - 1), len(self.scheduler_values))
        ]
        enable_prefix_caching = vals[6] >= 0.5
        disable_custom_all_reduce = vals[7] >= 0.5
        use_v2_block_manager = vals[8] >= 0.5

        return self.repair(
            {
                "tp": tp,
                "max_num_seqs": max_num_seqs,
                "max_num_batched_tokens": max_num_batched_tokens,
                "block_size": block_size,
                "enable_chunked_prefill": enable_chunked_prefill,
                "scheduler_delay_factor": scheduler_delay_factor,
                "enable_prefix_caching": enable_prefix_caching,
                "disable_custom_all_reduce": disable_custom_all_reduce,
                "use_v2_block_manager": use_v2_block_manager,
            }
        )

    def repair(self, config: Config) -> Config:
        tp = _nearest(int(config["tp"]), self.tp_values)
        max_num_seqs = _nearest_power_of_two(int(config["max_num_seqs"]), 64, 8192)
        max_num_batched_tokens = int(
            _clip(int(config["max_num_batched_tokens"]), 64, self.max_num_batched_tokens_upper)
        )
        block_size = _nearest(int(config["block_size"]), self.block_size_values)
        enable_chunked_prefill = bool(config["enable_chunked_prefill"])
        scheduler_delay_factor = _nearest(float(config["scheduler_delay_factor"]), self.scheduler_values)
        enable_prefix_caching = bool(config["enable_prefix_caching"])
        disable_custom_all_reduce = bool(config["disable_custom_all_reduce"])
        use_v2_block_manager = bool(config["use_v2_block_manager"])

        if not enable_chunked_prefill:
            max_num_batched_tokens = max(max_num_batched_tokens, self.max_sequence_length)
        if max_num_seqs > max_num_batched_tokens:
            max_num_seqs = _nearest_power_of_two(max_num_batched_tokens, 64, 8192, floor=True)
        if enable_chunked_prefill and enable_prefix_caching:
            enable_prefix_caching = False

        return {
            "tp": tp,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "block_size": block_size,
            "enable_chunked_prefill": enable_chunked_prefill,
            "scheduler_delay_factor": scheduler_delay_factor,
            "enable_prefix_caching": enable_prefix_caching,
            "disable_custom_all_reduce": disable_custom_all_reduce,
            "use_v2_block_manager": use_v2_block_manager,
        }

    def sobol_configs(
        self,
        n: int,
        *,
        skip: int = 0,
        seen: Iterable[Tuple[object, ...]] | None = None,
    ) -> List[Config]:
        seen_keys = set(seen or [])
        engine = torch.quasirandom.SobolEngine(dimension=self.dim, scramble=True, seed=42 + skip)
        configs: List[Config] = []
        attempts = 0
        while len(configs) < n and attempts < max(256, n * 128):
            point = engine.draw(1).squeeze(0).tolist()
            cfg = self.decode(point)
            key = self.key(cfg)
            attempts += 1
            if key in seen_keys:
                continue
            seen_keys.add(key)
            configs.append(cfg)
        while len(configs) < n:
            cfg = self.random_config()
            key = self.key(cfg)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            configs.append(cfg)
        return configs

    def random_config(self) -> Config:
        return self.repair(
            {
                "tp": random.choice(self.tp_values),
                "max_num_seqs": random.choice(self.max_num_seqs_values),
                "max_num_batched_tokens": int(
                    round(2 ** random.uniform(6, math.log2(self.max_num_batched_tokens_upper)))
                ),
                "block_size": random.choice(self.block_size_values),
                "enable_chunked_prefill": bool(random.getrandbits(1)),
                "scheduler_delay_factor": random.choice(self.scheduler_values),
                "enable_prefix_caching": bool(random.getrandbits(1)),
                "disable_custom_all_reduce": bool(random.getrandbits(1)),
                "use_v2_block_manager": bool(random.getrandbits(1)),
            }
        )

    def candidate_pool(
        self,
        n: int,
        *,
        seen: Iterable[Tuple[object, ...]] | None = None,
        seed: int = 1234,
    ) -> List[Config]:
        random.seed(seed)
        configs = self.sobol_configs(n, skip=seed, seen=seen)
        return configs


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _nearest(value, choices):
    return min(choices, key=lambda choice: abs(float(choice) - float(value)))


def _round_index(value: float, length: int) -> int:
    return int(_clip(round(value), 0, length - 1))


def _scale(value: float, in_low: float, in_high: float, out_low: float, out_high: float) -> float:
    value = _clip(value, in_low, in_high)
    if in_high == in_low:
        return out_low
    return out_low + (value - in_low) * (out_high - out_low) / (in_high - in_low)


def _nearest_power_of_two(value: int, low: int, high: int, *, floor: bool = False) -> int:
    value = int(_clip(value, low, high))
    powers = [2**p for p in range(int(math.log2(low)), int(math.log2(high)) + 1)]
    if floor:
        valid = [p for p in powers if p <= value]
        return valid[-1] if valid else low
    return _nearest(value, powers)
