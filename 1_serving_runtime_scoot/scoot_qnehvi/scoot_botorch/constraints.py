from __future__ import annotations

from typing import Tuple

from .space import Config, ScootSearchSpace


def hard_constraint_violations(space: ScootSearchSpace, config: Config) -> Tuple[str, ...]:
    cfg = config
    violations = []
    if int(cfg.get("max_num_batched_tokens", 0)) < int(cfg.get("max_num_seqs", 0)):
        violations.append("max_num_batched_tokens < max_num_seqs")
    if (
        not bool(cfg.get("enable_chunked_prefill", False))
        and int(cfg.get("max_num_batched_tokens", 0)) < space.max_sequence_length
    ):
        violations.append("chunked prefill disabled with too-small max_num_batched_tokens")
    if bool(cfg.get("enable_chunked_prefill", False)) and bool(cfg.get("enable_prefix_caching", False)):
        violations.append("chunked prefill and prefix caching both enabled")
    tp = int(cfg.get("tp", 0))
    if tp < 1 or space.gpu_nums % tp != 0 or tp not in space.tp_values:
        violations.append("tensor parallel size does not divide visible GPU count")
    return tuple(violations)


def is_valid_config(space: ScootSearchSpace, config: Config) -> bool:
    return not hard_constraint_violations(space, config)
