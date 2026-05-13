"""Aggregate every combined-pipeline run dir into a single report.md.

Auto-discovers all run directories under ``combined/`` and (optionally) any
LLM-Agent runs under ``../../llm_agent/results/llm_agent/``. For each run we
emit per-method summaries: trial counts, per-objective bests, Pareto frontier,
and the hypervolume-best config relative to the per-run vLLM-default
benchmark. Finally we print an overall ranking across runs.

Usage:
    conda activate scoot-botorch
    python 1_serving_runtime_scoot/combined_pipeline/results/_make_report.py

Optional flags:
    --runs-dir PATH         alternate combined/ root (default: alongside this file)
    --agent-runs-dir PATH   alternate llm_agent/results/llm_agent root
    --output PATH           where to write report.md (default: alongside this file)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent
DEFAULT_COMBINED_DIR = DEFAULT_RESULTS_DIR / "combined"
DEFAULT_AGENT_DIR = (
    Path(__file__).resolve().parents[2] / "llm_agent" / "results" / "llm_agent"
)
DEFAULT_REPORT_PATH = DEFAULT_RESULTS_DIR / "report.md"

REC_HISTORY_FILES = {
    "scoot": "rec_history.json",
    "qnehvi": "rec_history_qnehvi.json",
    "random": "rec_history_random.json",
    "agent": "rec_history_agent.json",
}

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
METRIC_KEYS = ["request_throughput", "mean_ttft_ms", "mean_tpot_ms"]
METHODS = ["scoot", "qnehvi", "default", "random", "agent"]


def collect_trials(base: Path) -> List[Dict]:
    """Read every vllm-*.json under base; return list of {params, metrics}."""
    trials: List[Dict] = []
    if not base or not base.exists():
        return trials
    for path in sorted(glob.glob(str(base / "**" / "vllm-*.json"), recursive=True)):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        if any(d.get(k) is None for k in METRIC_KEYS):
            continue
        params = {k: d.get(k) for k in PARAM_KEYS}
        metrics = {k: float(d[k]) for k in METRIC_KEYS}
        trials.append({"params": params, "metrics": metrics, "_file": os.path.basename(path)})
    return trials


def pareto_frontier(trials: List[Dict]) -> List[Dict]:
    frontier = []
    for i, t in enumerate(trials):
        ti = t["metrics"]
        dominated = False
        for j, o in enumerate(trials):
            if i == j:
                continue
            oj = o["metrics"]
            ge = (
                oj["request_throughput"] >= ti["request_throughput"]
                and oj["mean_ttft_ms"] <= ti["mean_ttft_ms"]
                and oj["mean_tpot_ms"] <= ti["mean_tpot_ms"]
            )
            strict = (
                oj["request_throughput"] > ti["request_throughput"]
                or oj["mean_ttft_ms"] < ti["mean_ttft_ms"]
                or oj["mean_tpot_ms"] < ti["mean_tpot_ms"]
            )
            if ge and strict:
                dominated = True
                break
        if not dominated:
            frontier.append(t)
    return frontier


def hv_score(trial: Dict, ref: Dict) -> Optional[float]:
    if ref is None:
        return None
    t = trial["metrics"]
    r = ref["metrics"]
    dt = (t["request_throughput"] - r["request_throughput"]) / max(r["request_throughput"], 1e-9)
    df = (r["mean_ttft_ms"] - t["mean_ttft_ms"]) / max(r["mean_ttft_ms"], 1e-9)
    do = (r["mean_tpot_ms"] - t["mean_tpot_ms"]) / max(r["mean_tpot_ms"], 1e-9)
    return dt * df * do if (dt > 0 and df > 0 and do > 0) else (dt + df + do)


def best_per_objective(trials: List[Dict]) -> Dict[str, Dict]:
    return {
        "throughput": max(trials, key=lambda t: t["metrics"]["request_throughput"]),
        "ttft": min(trials, key=lambda t: t["metrics"]["mean_ttft_ms"]),
        "tpot": min(trials, key=lambda t: t["metrics"]["mean_tpot_ms"]),
    }


def fmt_params(p: Dict) -> str:
    return ", ".join(f"{k}={p[k]}" for k in PARAM_KEYS)


def fmt_metrics(m: Dict) -> str:
    return (
        f"throughput={m['request_throughput']:.3f} req/s, "
        f"TTFT={m['mean_ttft_ms']:.2f} ms, "
        f"TPOT={m['mean_tpot_ms']:.2f} ms"
    )


def collect_runtime(method: str, method_root: Optional[Path]) -> Optional[Dict]:
    if method_root is None or not method_root.exists():
        return None
    if method == "default":
        files = sorted(glob.glob(str(method_root / "**/vllm-*.json"), recursive=True))
        if not files:
            return None
        d = json.load(open(files[0]))
        dur = d.get("duration")
        if dur is None:
            return None
        return {
            "trials_logged": 1, "fails": 0,
            "run_time_total_s": float(dur), "run_time_mean_s": float(dur),
            "rec_time_total_s": 0.0, "rec_time_mean_s": 0.0,
        }
    fname = REC_HISTORY_FILES.get(method)
    if not fname:
        return None
    matches = glob.glob(str(method_root / "**" / fname), recursive=True)
    if not matches:
        return None
    history = json.load(open(matches[0]))
    if not history:
        return None
    rts = [float(h.get("run_time", 0) or 0) for h in history]
    rcs = [float(h.get("rec_time", 0) or 0) for h in history]
    fails = sum(1 for h in history if h.get("obj") is None)
    return {
        "trials_logged": len(history),
        "fails": fails,
        "run_time_total_s": sum(rts),
        "run_time_mean_s": sum(rts) / len(rts) if rts else 0.0,
        "rec_time_total_s": sum(rcs),
        "rec_time_mean_s": sum(rcs) / len(rcs) if rcs else 0.0,
    }


def fmt_runtime(rt: Optional[Dict], target_trials: int) -> str:
    if rt is None:
        return "Wall-clock: TBD / pending — no rec_history yet."
    total_s = rt["run_time_total_s"] + rt["rec_time_total_s"]
    incomplete = ""
    if rt["trials_logged"] < target_trials - 1:
        incomplete = f" (incomplete: {rt['trials_logged']}/{target_trials} BO trials logged)"
    return (
        f"Wall-clock: total **{total_s/60:.1f} min** ({total_s/3600:.2f} h){incomplete}; "
        f"mean per trial: run={rt['run_time_mean_s']:.1f}s, "
        f"BO suggest={rt['rec_time_mean_s']:.1f}s; "
        f"failed configs: {rt['fails']}"
    )


def render_method_section(method: str, trials: List[Dict],
                          default_ref: Optional[Dict],
                          method_root: Optional[Path] = None,
                          target_trials: int = 30) -> str:
    if not trials:
        rt = collect_runtime(method, method_root)
        out = [f"### {method}", "", "_TBD / pending — no successful trials yet._", ""]
        if rt:
            out.append("- " + fmt_runtime(rt, target_trials))
        return "\n".join(out) + "\n"

    n = len(trials)
    rt = collect_runtime(method, method_root)
    bp = best_per_objective(trials)
    front = pareto_frontier(trials)

    hv_best = None
    if default_ref is not None:
        scored = [(t, hv_score(t, default_ref)) for t in trials]
        scored = [(t, s) for t, s in scored if s is not None]
        if scored:
            hv_best = max(scored, key=lambda x: x[1])[0]

    out = [f"### {method}", "",
           f"- Successful trials: **{n}**",
           f"- Pareto frontier size: **{len(front)}**"]
    if rt:
        out.append("- " + fmt_runtime(rt, target_trials))
    out.append("")

    if hv_best is not None:
        out += [
            "**Recommended config (max hypervolume vs vLLM default):**",
            "",
            f"- Params: {fmt_params(hv_best['params'])}",
            f"- Metrics: {fmt_metrics(hv_best['metrics'])}",
            "",
        ]
    else:
        out += ["_(No vLLM-default reference for this run; HV-best omitted.)_", ""]

    out += ["**Per-objective best:**", "",
            f"- Best throughput: `{fmt_metrics(bp['throughput']['metrics'])}`",
            f"  - Params: {fmt_params(bp['throughput']['params'])}",
            f"- Best TTFT: `{fmt_metrics(bp['ttft']['metrics'])}`",
            f"  - Params: {fmt_params(bp['ttft']['params'])}",
            f"- Best TPOT: `{fmt_metrics(bp['tpot']['metrics'])}`",
            f"  - Params: {fmt_params(bp['tpot']['params'])}", ""]

    if 0 < len(front) <= 8:
        out += ["**Pareto frontier (all non-dominated configs):**", ""]
        for t in sorted(front, key=lambda x: -x["metrics"]["request_throughput"]):
            out += [f"- {fmt_metrics(t['metrics'])}",
                    f"  - {fmt_params(t['params'])}"]
        out.append("")
    return "\n".join(out)


def discover_runs(combined_dir: Path, agent_dir: Path) -> List[Dict]:
    """Find every run directory under combined/ and pair with an agent run if any.

    Pairing is by SLURM_JOB_ID suffix: a combined run named ``rtxa6000_12345``
    is paired with an agent run named ``rtxa6000_12346`` if and only if the
    agent run has the same prefix and was submitted within a few hours. To
    keep this generic and side-effect-free, we just match on the trailing
    digit suffix; users can override by symlinking their agent dir under
    ``combined/<run>/tune_res/agent/``.
    """
    runs = []
    if combined_dir.exists():
        for d in sorted(combined_dir.iterdir()):
            if not d.is_dir():
                continue
            tune_res = d / "tune_res"
            if not tune_res.exists():
                continue
            method_root = {m: tune_res / m for m in METHODS}
            # If agent results not under combined run dir, look for a
            # parallel run in the llm_agent results tree by exact name match.
            if not (tune_res / "agent").exists() and agent_dir.exists():
                candidate = agent_dir / d.name / "tune_res" / "agent"
                if candidate.exists():
                    method_root["agent"] = candidate
            runs.append({"label": d.name, "run_dir": d, "method_root": method_root})
    return runs


def render_run(run: Dict) -> str:
    method_root = run["method_root"]
    method_trials = {m: collect_trials(method_root.get(m)) for m in METHODS}
    default_ref = method_trials["default"][0] if method_trials.get("default") else None

    out = [f"## {run['label']}", "",
           f"- Run dir: `{run['run_dir']}`", ""]

    for m in METHODS:
        target = 1 if m == "default" else 30
        out.append(render_method_section(
            m, method_trials[m], default_ref,
            method_root=method_root.get(m), target_trials=target,
        ))
    return "\n".join(out)


def render_ranking(runs: List[Dict]) -> str:
    method_wins = {m: 0 for m in METHODS}
    contests = 0
    pareto_credit = {m: 0 for m in METHODS}
    pareto_total = 0
    method_present = {m: 0 for m in METHODS}

    for run in runs:
        method_root = run["method_root"]
        method_trials = {m: collect_trials(method_root.get(m)) for m in METHODS}
        present = [m for m in METHODS if method_trials[m]]
        for m in present:
            method_present[m] += 1
        if len(present) < 2:
            continue
        bests = {}
        for m in present:
            bests[m] = (
                max(t["metrics"]["request_throughput"] for t in method_trials[m]),
                min(t["metrics"]["mean_ttft_ms"] for t in method_trials[m]),
                min(t["metrics"]["mean_tpot_ms"] for t in method_trials[m]),
            )
        for i, big in enumerate([True, False, False]):
            w = (max if big else min)(present, key=lambda m: bests[m][i])
            method_wins[w] += 1
            contests += 1

        pool = []
        for m in present:
            for t in method_trials[m]:
                pool.append((m, t["metrics"]))
        for i, (m, t) in enumerate(pool):
            dom = False
            for j, (_, o) in enumerate(pool):
                if i == j:
                    continue
                if (o["request_throughput"] >= t["request_throughput"]
                    and o["mean_ttft_ms"] <= t["mean_ttft_ms"]
                    and o["mean_tpot_ms"] <= t["mean_tpot_ms"]
                    and (o["request_throughput"] > t["request_throughput"]
                         or o["mean_ttft_ms"] < t["mean_ttft_ms"]
                         or o["mean_tpot_ms"] < t["mean_tpot_ms"])):
                    dom = True
                    break
            if not dom:
                pareto_credit[m] += 1
                pareto_total += 1

    out = ["## Overall method ranking", "",
           "Aggregated across all runs containing the relevant method.", "",
           "1. **Per-objective wins**: per (run × objective), count which method's",
           "   best trial wins (3 metrics × N runs = up to 3N contests).",
           "2. **Pooled Pareto contributions**: pool all trials across all methods",
           "   per run, compute the union Pareto frontier, count points per method.",
           "", "### Per-objective wins", "",
           "| Method | Wins | Win rate | Runs covered |",
           "|---|---|---|---|"]
    for m in sorted(METHODS, key=lambda m: -method_wins[m]):
        wr = (method_wins[m] / contests * 100) if contests else 0
        out.append(f"| {m} | {method_wins[m]} | {wr:.0f}% | {method_present[m]}/{len(runs)} |")
    out += ["", f"_Total contests: {contests}._", "",
            "### Pooled Pareto contributions", "",
            "| Method | Frontier points | Share | Runs covered |",
            "|---|---|---|---|"]
    for m in sorted(METHODS, key=lambda m: -pareto_credit[m]):
        share = (pareto_credit[m] / pareto_total * 100) if pareto_total else 0
        out.append(f"| {m} | {pareto_credit[m]} | {share:.0f}% | {method_present[m]}/{len(runs)} |")
    out += ["", f"_Total frontier points: {pareto_total}._", ""]

    combined_score = {m: method_wins[m] + pareto_credit[m] for m in METHODS}
    ranking = sorted(METHODS, key=lambda m: -combined_score[m])
    out += ["### Combined ranking", "",
            "| Rank | Method | Combined score |", "|---|---|---|"]
    for i, m in enumerate(ranking, 1):
        out.append(f"| {i} | **{m}** | {combined_score[m]} |")
    out += [""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_COMBINED_DIR)
    ap.add_argument("--agent-runs-dir", type=Path, default=DEFAULT_AGENT_DIR)
    ap.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    args = ap.parse_args()

    runs = discover_runs(args.runs_dir, args.agent_runs_dir)
    if not runs:
        print(f"No runs found under {args.runs_dir}. "
              f"Submit a sbatch from combined_pipeline/run_configs/ first.")
        return

    parts = [
        "# Combined SCOOT Pipeline — Results Report", "",
        "Comparison of up to **five** configuration-tuning methods (SCOOT, qNEHVI,",
        "vLLM defaults, random search, LLM-Agent) on the same vLLM serving workload.",
        "Three objectives: maximize **request_throughput** (req/s), minimize",
        "**mean_ttft_ms** and **mean_tpot_ms**.", "",
        "Methods:",
        "- **scoot** — HEBO + EHVI (multi-objective Bayesian optimization).",
        "- **qnehvi** — BoTorch mixed-space GP + qNoisyExpectedHypervolumeImprovement.",
        "- **default** — vLLM 0.5.5 stock defaults (only `--gpu-memory-utilization 0.9 --trust-remote-code`).",
        "- **random** — N random configs from the same 9-parameter space.",
        "- **agent** — LLM-Agent: 10 Sobol-init + 20 LLM-proposed configs.",
        "",
        f"_Discovered {len(runs)} run(s) under `{args.runs_dir}`._",
        "", "---", "",
    ]
    for run in runs:
        parts.append(render_run(run))
        parts.append("\n---\n")
    parts.append(render_ranking(runs))
    parts.append("")
    parts += [
        "## Notes", "",
        "- **Recommended** = trial maximizing the hypervolume product of relative",
        "  improvements (+throughput, -TTFT, -TPOT) over the per-run vLLM-default.",
        "  When all three improve, score is positive; if any worsens, score collapses",
        "  to the sum so two-out-of-three winners still rank above three-out-of-three",
        "  losers. If the default benchmark didn't run for a run, HV-best is omitted.",
        "- **Pareto frontier** lists configs no other observed config strictly",
        "  dominates across all three objectives.",
        "- **Per-objective best** picks the trial that wins each metric in isolation.",
        "- **TBD / pending** indicates the corresponding stage hasn't produced any",
        "  valid trials yet.", "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
