"""LLM-Agent search over the SCOOT 9-parameter space (Stage 5 of combined wrapper).

Two-phase method: (1) **Sobol warm-start** of N=10 quasi-random configs
matching qNEHVI's `--sobol_init=10` default — closes the init-phase
fairness gap vs qNEHVI; (2) **LLM phase** of `--num_trials - 10`
single-shot Claude Opus 4.7 calls with adaptive thinking at `xhigh`
effort. Each LLM call sees the full sorted history (Sobol results
included, marked as such in the prompt) and proposes one new config.

Benchmarks run via the same `_bench_runner.evaluate()` interface that
random_search.py uses, and history is persisted in the same
`rec_history_*.json` schema for apples-to-apples comparison with SCOOT,
qNEHVI, and random.

Method label in published comparisons: "Agent (Sobol-10 + Opus 4.7
adaptive xhigh)". This is intentionally NOT a pure-cold-start agent —
matching qNEHVI's init makes the optimization-phase comparison fair.

Run from the qNEHVI repo so `benchmark_pipeline.sh`, `tuner_conf`,
`scoot_botorch`, and `utils` are all on the path.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
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

RES_DIR_PREFIX = os.environ.get("SCOOT_RES_DIR_PREFIX", "agent")
RES_DIR = os.environ.get("SCOOT_RES_DIR", "./tune_res")
LOG_DIR = os.path.join(RES_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

PARAM_KEYS = [
    "tp",
    "max_num_seqs",
    "max_num_batched_tokens",
    "block_size",
    "enable_chunked_prefill",
    "scheduler_delay_factor",
    "enable_prefix_caching",
    "disable_custom_all_reduce",
    "use_v2_block_manager",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM-Agent (Claude) search over the SCOOT 9-parameter space"
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
    p.add_argument("--llm_model", default="claude-opus-4-7",
                   help="Anthropic model id (default: claude-opus-4-7). "
                        "Use claude-sonnet-4-6 if Opus 4.7 is not enabled on your account; "
                        "with Sonnet 4.6, also change effort='xhigh' below to effort='max' "
                        "(xhigh is Opus 4.7-only).")
    p.add_argument("--api_key", default=None,
                   help="Anthropic API key; falls back to ANTHROPIC_API_KEY env var")
    p.add_argument("--program_md", default=None,
                   help="Path to the agent prompt template (default: agent_program.md sibling)")
    return p.parse_args()


def load_program_template(path: str | None) -> str:
    if path is None:
        path = str(Path(__file__).resolve().parent / "agent_program.md")
    with open(path) as f:
        return f.read()


def render_program(template: str, *, gpu_name: str, gpu_nums: int,
                   tp_values: list, max_seq_len: int,
                   request_rate: int, num_requests: int) -> str:
    return (
        template
        .replace("{{GPU_NAME}}", gpu_name)
        .replace("{{GPU_NUMS}}", str(gpu_nums))
        .replace("{{TP_VALUES}}", "{" + ", ".join(str(v) for v in tp_values) + "}")
        .replace("{{MAX_SEQ_LEN}}", str(max_seq_len))
        .replace("{{REQUEST_RATE}}", str(request_rate))
        .replace("{{NUM_REQUESTS}}", str(num_requests))
    )


def detect_gpu_name() -> str:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.cuda.get_device_name(0)
    return "unknown"


def extract_config_json(response_text: str) -> dict | None:
    """Parse the CONFIG: {...} JSON block from the agent's response."""
    # Preferred: explicit CONFIG: marker followed by a single JSON object.
    m = re.search(r"CONFIG:\s*\n?\s*(\{.*?\})\s*(?:\nLEARNED:|\Z)",
                  response_text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Fallback: any JSON object containing "tp" or "tensor_parallel_size".
    m = re.search(r"\{[^{}]*\"tp\"[^{}]*\}", response_text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # Fallback: fenced code block.
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


def extract_reasoning(response_text: str) -> str:
    patterns = re.search(r"PATTERNS:\s*(.+?)(?=\n\s*HYPOTHESIS:|\n\s*CONFIG:|\Z)",
                         response_text, re.DOTALL)
    hypothesis = re.search(r"HYPOTHESIS:\s*(.+?)(?=\n\s*CONFIG:|\Z)",
                           response_text, re.DOTALL)
    if patterns and hypothesis:
        return f"PATTERNS: {patterns.group(1).strip()} | HYPOTHESIS: {hypothesis.group(1).strip()}"
    return response_text.strip().splitlines()[0][:200] if response_text.strip() else ""


def coerce_config(raw: dict) -> dict:
    """Coerce LLM-produced types into what ScootSearchSpace.repair() expects.

    The agent may emit string booleans ("true"/"True") or float ints. repair()
    is strict about types, so normalize here before handing it off.
    """
    out = dict(raw)

    # Drop any vLLM-0.11.2-flavored keys the LLM might hallucinate from
    # transfer-learning on Rayhan's old prompt.
    for stale in ("gpu_memory_utilization", "enforce_eager", "kv_cache_dtype",
                  "swap_space", "tensor_parallel_size"):
        if stale in out and stale != "tp":
            # tensor_parallel_size is the long-form alias for tp.
            if stale == "tensor_parallel_size" and "tp" not in out:
                out["tp"] = out.pop(stale)
            else:
                out.pop(stale, None)

    def _as_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
        return bool(v)

    bool_keys = ("enable_chunked_prefill", "enable_prefix_caching",
                 "disable_custom_all_reduce", "use_v2_block_manager")
    for k in bool_keys:
        if k in out:
            out[k] = _as_bool(out[k])

    int_keys = ("tp", "max_num_seqs", "max_num_batched_tokens", "block_size")
    for k in int_keys:
        if k in out:
            try:
                out[k] = int(out[k])
            except (TypeError, ValueError):
                pass

    if "scheduler_delay_factor" in out:
        try:
            out["scheduler_delay_factor"] = float(out["scheduler_delay_factor"])
        except (TypeError, ValueError):
            pass

    return out


def is_complete_config(cfg: dict) -> bool:
    return all(k in cfg for k in PARAM_KEYS)


def pareto_frontier_indices(history: list) -> list[int]:
    """Indices of trials whose obj=[-thr, ttft, tpot] is non-dominated.

    Lower is better on all 3 components (throughput is already negated).
    """
    objs = [item.get("obj") for item in history]
    valid = [i for i, o in enumerate(objs) if o is not None]
    frontier = []
    for i in valid:
        a = objs[i]
        dominated = False
        for j in valid:
            if i == j:
                continue
            b = objs[j]
            if (b[0] <= a[0] and b[1] <= a[1] and b[2] <= a[2]
                    and (b[0] < a[0] or b[1] < a[1] or b[2] < a[2])):
                dominated = True
                break
        if not dominated:
            frontier.append(i)
    return frontier


def format_history_for_prompt(history: list, space: ScootSearchSpace) -> str:
    """Render past trials as an OPRO-sorted table with a Pareto callout."""
    if not history:
        return "No experiments run yet."

    successes = [(i, item) for i, item in enumerate(history)
                 if item.get("obj") is not None]
    failures = [(i, item) for i, item in enumerate(history)
                if item.get("obj") is None]

    lines = []
    if successes:
        # Pareto frontier callout (over successful trials).
        frontier_idx = set(pareto_frontier_indices(history))
        frontier = [(i, item) for i, item in successes if i in frontier_idx]
        if frontier:
            lines.append("### Pareto frontier (non-dominated trials)")
            lines.append("trial | thr (req/s) | TTFT ms | TPOT ms | config")
            lines.append("------|-------------|---------|---------|-------")
            for i, item in frontier:
                obj = item["obj"]
                cfg = item["rec"][0]
                lines.append(
                    f"{i+1:5d} | {-obj[0]:11.2f} | {obj[1]:7.1f} | {obj[2]:7.1f} | "
                    f"{json.dumps(_compact_cfg(cfg))}"
                )
            lines.append("")

        # Full table sorted ascending by throughput (best throughput last).
        # obj[0] is -throughput, so largest obj[0] = worst throughput goes first.
        successes_sorted = sorted(successes, key=lambda p: p[1]["obj"][0], reverse=True)
        lines.append("### All successful trials (sorted by throughput, best last)")
        lines.append("trial | thr (req/s) | TTFT ms | TPOT ms | tp | mns | mbt | bs | chunked | sched | prefix | noAR | v2bm")
        lines.append("------|-------------|---------|---------|----|----|-----|----|---------|-------|--------|------|-----")
        for i, item in successes_sorted:
            obj = item["obj"]
            cfg = item["rec"][0]
            lines.append(
                f"{i+1:5d} | {-obj[0]:11.2f} | {obj[1]:7.1f} | {obj[2]:7.1f} | "
                f"{int(cfg['tp']):2d} | {int(cfg['max_num_seqs']):4d} | "
                f"{int(cfg['max_num_batched_tokens']):4d} | {int(cfg['block_size']):2d} | "
                f"{str(bool(cfg['enable_chunked_prefill']))[0]} | "
                f"{float(cfg['scheduler_delay_factor']):4.1f} | "
                f"{str(bool(cfg['enable_prefix_caching']))[0]} | "
                f"{str(bool(cfg['disable_custom_all_reduce']))[0]} | "
                f"{str(bool(cfg['use_v2_block_manager']))[0]}"
            )

    if failures:
        lines.append("")
        lines.append("### Failed trials (benchmark crashed or timed out)")
        for i, item in failures:
            cfg = item["rec"][0]
            lines.append(f"{i+1}: {json.dumps(_compact_cfg(cfg))}")

    return "\n".join(lines) if lines else "No experiments run yet."


def _compact_cfg(cfg: dict) -> dict:
    """Re-key a config to a stable ordering for compact display."""
    return {k: cfg[k] for k in PARAM_KEYS if k in cfg}


def format_seen_keys(seen: set) -> str:
    if not seen:
        return "None yet"
    rows = [str(k) for k in sorted(seen, key=str)]
    return "\n".join(rows)


def main():
    args = parse_args()
    random.seed(args.seed)

    gpu_nums = torch.cuda.device_count()
    assert gpu_nums >= 1, "Agent search needs at least one CUDA GPU"

    logging.basicConfig(
        filename=os.path.join(
            LOG_DIR, f"agent_{args.model}_{args.total_resource}.log"
        ),
        level=logging.INFO,
    )

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set and --api_key not provided.",
              file=sys.stderr)
        print("Export it before sbatch:  export ANTHROPIC_API_KEY=sk-ant-...",
              file=sys.stderr)
        sys.exit(2)

    try:
        import anthropic  # imported lazily so a missing dep gives a clear error
    except ImportError:
        print("ERROR: anthropic SDK not installed in this conda env.", file=sys.stderr)
        print("Install with:  pip install 'anthropic>=0.96'  (≥0.96 needed for Opus 4.7 + adaptive thinking)", file=sys.stderr)
        sys.exit(2)
    client = anthropic.Anthropic(api_key=api_key)

    min_world_size = get_ref_config("min_world_size")
    max_seq_len = get_ref_config("max_sequence_length")
    space = ScootSearchSpace(
        gpu_nums=gpu_nums,
        min_world_size=min_world_size,
        max_sequence_length=max_seq_len,
    )

    gpu_name = detect_gpu_name()
    program_template = load_program_template(args.program_md)
    system_prompt = render_program(
        program_template,
        gpu_name=gpu_name,
        gpu_nums=gpu_nums,
        tp_values=space.tp_values,
        max_seq_len=max_seq_len,
        request_rate=args.request_rate,
        num_requests=args.num_requests,
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
    history_path = Path(res_dir_path) / "rec_history_agent.json"
    reasoning_dir = Path(res_dir_path) / "logs"
    reasoning_dir.mkdir(parents=True, exist_ok=True)

    history = load_history(history_path)
    seen = set()
    for item in history:
        rec = item.get("rec") or []
        if rec:
            seen.add(space.key(rec[0]))

    print(f"Agent search: target {args.num_trials} trials, resuming at {len(history)}")
    print(f"GPU: {gpu_name} x {gpu_nums} | tp_values={space.tp_values} | max_seq_len={max_seq_len}")
    print(f"LLM model: {args.llm_model}")
    print(f"Results dir: {res_dir_path}")

    total_input_tokens = 0
    total_output_tokens = 0

    # Sobol warm-start: 10 quasi-random initial points for parity with qNEHVI's
    # --sobol_init=10 default in bo_scoot_qnehvi.py. The remaining 20 trials
    # (assuming --num_trials=30) are LLM-proposed. The agent's chosen method
    # name in published comparisons is "Agent (Sobol-10 + LLM-N)".
    WARM_START_N = 10
    if len(history) == 0 and args.num_trials > WARM_START_N:
        print(f"\nWarm-start phase: {WARM_START_N} Sobol points "
              f"(parity with qNEHVI --sobol_init=10)")
        sobol_configs = space.sobol_configs(n=WARM_START_N, seen=seen)
        for cfg in sobol_configs:
            if space.key(cfg) in seen:
                continue
            seen.add(space.key(cfg))
            iter_n = len(history) + 1
            print(f"  Sobol-init {iter_n}/{WARM_START_N}: {cfg}")
            t_start = time.time()
            result = evaluate(cfg, gpu_nums, res_dir_path, args, min_world_size)
            run_time = time.time() - t_start
            y = None
            if result is not None:
                y = [-1.0 * float(result["request_throughput"]),
                     float(result["mean_ttft_ms"]),
                     float(result["mean_tpot_ms"])]
                print(f"    obj=[thr={-y[0]:.2f}, ttft={y[1]:.1f}, tpot={y[2]:.1f}] "
                      f"({run_time:.0f}s)")
            else:
                print(f"    obj=None (benchmark failed, {run_time:.0f}s)")
            history.append(history_item(cfg, y, 0.0, run_time))
            save_history(history_path, history)
            logging.info(f"sobol_init {iter_n}/{WARM_START_N} cfg={cfg} obj={y}")
        print(f"\nWarm-start complete: {len(history)} trials. "
              f"Now switching to LLM-proposed configs.")

    while len(history) < args.num_trials:
        iteration = len(history) + 1
        print(f"\n===== Agent iteration {iteration}/{args.num_trials} =====")

        history_block = format_history_for_prompt(history, space)
        seen_block = format_seen_keys(seen)

        # Tell the LLM about the Sobol warm-start phase so it correctly
        # interprets the early history rows as random samples (not
        # cherry-picked) and frames itself as starting at LLM-call N
        # rather than iteration 1.
        sobol_aware_note = ""
        if WARM_START_N > 0 and iteration > WARM_START_N:
            llm_call_n = iteration - WARM_START_N
            llm_total = args.num_trials - WARM_START_N
            sobol_aware_note = (
                f"\n## Note on past trials\n"
                f"Trials 1-{WARM_START_N} in the history table were Sobol "
                f"quasi-random initialization (matching qNEHVI's "
                f"--sobol_init={WARM_START_N}). They are NOT cherry-picked "
                f"and do NOT reflect any optimization signal — treat them "
                f"as random samples that map out the space. From your "
                f"perspective, this is **LLM proposal {llm_call_n} of "
                f"{llm_total}**.\n"
            )

        user_message = (
            f"## Status\n"
            f"Iteration {iteration} of {args.num_trials}. "
            f"You have {args.num_trials - iteration} evaluations after this one."
            f"{sobol_aware_note}\n"
            f"## Past Results\n{history_block}\n\n"
            f"## Already Tested 9-tuples (do NOT repeat any of these)\n"
            f"`(tp, max_num_seqs, max_num_batched_tokens, block_size, "
            f"enable_chunked_prefill, scheduler_delay_factor, enable_prefix_caching, "
            f"disable_custom_all_reduce, use_v2_block_manager)`\n"
            f"{seen_block}\n\n"
            f"## Your Task\n"
            f"Propose ONE new configuration in the exact PATTERNS / HYPOTHESIS / CONFIG / "
            f"LEARNED format. The CONFIG must be a JSON object with all 9 keys. "
            f"`tp` must be one of {space.tp_values}."
        )

        # OPRO temperature ramp: explore high, exploit low.
        temperature = 1.0 if iteration <= args.num_trials * 0.5 else 0.5

        cfg = None
        response_text = ""
        api_latency = 0.0
        api_input_tokens = 0
        api_output_tokens = 0
        try:
            t0 = time.time()
            # Adaptive thinking + xhigh effort = strongest reasoning available on Opus 4.7.
            # Manual thinking (type=enabled, budget_tokens=N) is rejected on Opus 4.7 and
            # deprecated on Sonnet 4.6. xhigh effort is Opus 4.7 only; for Sonnet 4.6
            # change "xhigh" to "max" via --llm_model claude-sonnet-4-6.
            response = client.messages.create(
                model=args.llm_model,
                max_tokens=64000,                       # was 2000; xhigh thinking needs headroom
                thinking={"type": "adaptive"},
                output_config={"effort": "xhigh"},
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            api_latency = time.time() - t0
            # When thinking is enabled, response.content has multiple blocks;
            # iterate to find the text block (order is not guaranteed).
            response_text = ""
            for block in response.content:
                if block.type == "text":
                    response_text = block.text
                    break
            api_input_tokens = response.usage.input_tokens
            api_output_tokens = response.usage.output_tokens  # includes thinking tokens
            total_input_tokens += api_input_tokens
            total_output_tokens += api_output_tokens
            print(f"  LLM: {api_input_tokens} in / {api_output_tokens} out tokens "
                  f"(includes thinking), {api_latency:.1f}s @ T={temperature}")

            raw = extract_config_json(response_text)
            if raw is not None:
                raw = coerce_config(raw)
                if is_complete_config(raw):
                    cfg = space.repair(raw)
                else:
                    missing = [k for k in PARAM_KEYS if k not in raw]
                    print(f"  Parse OK but missing keys {missing}; "
                          f"falling back to random_config()")
            else:
                print("  Could not parse CONFIG block; falling back to random_config()")
        except Exception as exc:
            print(f"  LLM API error: {exc} -- falling back to random_config()")
            response_text = f"[API error: {exc}]"

        if cfg is None:
            cfg = space.random_config()

        # Anti-repeat: if the LLM duplicated, switch to random_config until unique.
        if space.key(cfg) in seen:
            print(f"  Duplicate of a tested config; resampling randomly")
            for _ in range(256):
                candidate = space.random_config()
                if space.key(candidate) not in seen:
                    cfg = candidate
                    break

        if space.key(cfg) in seen:
            print("  Search space appears exhausted; stopping early")
            break

        seen.add(space.key(cfg))

        # Persist per-iteration reasoning to its own file.
        reasoning = extract_reasoning(response_text)
        reasoning_file = reasoning_dir / f"agent_iter_{iteration}.txt"
        with open(reasoning_file, "w") as f:
            f.write(f"=== Iteration {iteration}/{args.num_trials} ===\n")
            f.write(f"Temperature: {temperature}\n")
            f.write(f"API tokens: {api_input_tokens} in / {api_output_tokens} out, "
                    f"{api_latency:.1f}s\n")
            f.write(f"Final config (after repair): {json.dumps(cfg)}\n\n")
            f.write("=== Full LLM response ===\n")
            f.write(response_text)
            f.write("\n")

        print(f"  Config: {cfg}")
        print(f"  Reasoning: {reasoning[:160]}")

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
            print(f"  Result: thr={-y[0]:.2f} req/s ttft={y[1]:.1f} ms tpot={y[2]:.1f} ms "
                  f"({run_time:.0f}s)")
        else:
            print(f"  Result: FAILED (no matching benchmark output, {run_time:.0f}s)")

        history.append(history_item(cfg, y, 0.0, run_time))
        save_history(history_path, history)

        with open(reasoning_file, "a") as f:
            f.write("\n=== Benchmark result ===\n")
            f.write(f"obj={y} run_time={run_time:.1f}s\n")

        succeed = sum(1 for h in history if h.get("obj") is not None)
        failed = len(history) - succeed
        logging.info(
            f"iteration={iteration} cfg={cfg} obj={y} succeed={succeed} failed={failed}"
        )
        print(f"  Progress: {succeed} succeeded, {failed} failed of {len(history)} total. "
              f"Cumulative tokens: {total_input_tokens} in / {total_output_tokens} out")

    cleanup_servers(gpu_nums, min_world_size)
    print(f"\nAgent search complete: {len(history)} trials")
    print(f"Total LLM tokens: {total_input_tokens} in / {total_output_tokens} out")


if __name__ == "__main__":
    main()
