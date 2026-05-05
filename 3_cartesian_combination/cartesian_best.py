"""Score the Cartesian product of FlexGen × qNEHVI Pareto sets.

Inputs (by run-root, e.g. results/run_<jobid>):
  flexgen/flexgen_*_pareto.json   produced by flexgen_pareto.py
  qnehvi/<bo_*>/exp0/pareto_frontier_qnehvi.json  produced by bo_scoot_qnehvi.py
  qnehvi/<bo_*>/exp0/vllm-*.json                  per-trial measured metrics

Pipeline:
  1. Load both Pareto sets.
  2. Build Cartesian product: N_flexgen × M_qnehvi pairs.
  3. For each pair, compute a single combined score by min-max normalizing the
     four objectives across the cross-product and summing them. The four
     objectives are FlexGen's predicted per-token latency (min) and qNEHVI's
     measured throughput (max), TTFT (min), TPOT (min).
  4. Pick the lowest-score pair as the "best combined configuration".
  5. Report the best pair's parameters + its objective values, and compare
     against the vLLM/SCOOT reference (the default configuration the qNEHVI
     paper uses as the hypervolume reference).

LIMITATION: FlexGen and qNEHVI tune disjoint parameter spaces (memory
placement + batching for FlexGen vs vLLM serving runtime args for qNEHVI). A
"combined" configuration is not directly deployable in either engine — this
script reports the best joint Pareto point as an analytical comparison, not a
runnable config.

Usage:
  python cartesian_best.py <run_root>
  python cartesian_best.py <run_root> --weights 1,1,1,1
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Default vLLM/SCOOT reference (matches bo_scoot_qnehvi.py's reference_config).
# tp = min_world_size, which depends on the GPU; we read the actual value used
# in this run from tuner_conf_qnehvi.json.
DEFAULT_QNEHVI_PARAMS_TEMPLATE = {
    "tp": None,                              # filled from tuner_conf
    "max_num_seqs": 256,
    "max_num_batched_tokens": 4096,          # max(4096, max_seq_len)
    "block_size": 16,
    "enable_chunked_prefill": False,
    "scheduler_delay_factor": 0.0,
    "enable_prefix_caching": False,
    "disable_custom_all_reduce": False,
    "use_v2_block_manager": False,
}

# Default FlexGen policy (engine defaults, no offloading, no compression).
DEFAULT_FLEXGEN_PARAMS = {
    "gpu_batch_size": 1,
    "num_gpu_batches": 1,
    "block_size": 1,
    "compression": "fp16",
    "cpu_compute_delegate": False,
    "overlap_io_compute": False,
    "weights":     {"gpu": 1.0, "cpu": 0.0, "disk": 0.0},
    "kv_cache":    {"gpu": 1.0, "cpu": 0.0, "disk": 0.0},
    "activations": {"gpu": 1.0, "cpu": 0.0, "disk": 0.0},
}


def load_flexgen_pareto(run_root: Path) -> List[Dict]:
    paths = sorted(glob.glob(str(run_root / "flexgen" / "flexgen_*_pareto.json")))
    if not paths:
        raise SystemExit(f"No flexgen_*_pareto.json under {run_root}/flexgen — "
                         "run flexgen_pareto.py first")
    d = json.loads(Path(paths[-1]).read_text())
    # Use Pareto front A (latency vs block_size); tag source for later display
    front = d.get("pareto_latency_vs_block_size", [])
    if not front:
        raise SystemExit(f"Empty Pareto in {paths[-1]}")
    return front


def load_qnehvi_pareto(run_root: Path) -> Tuple[List[Dict], Path]:
    """Returns (frontier_with_measured_metrics, qnehvi_run_dir)."""
    front_paths = sorted(glob.glob(str(run_root / "qnehvi" / "**" / "pareto_frontier_qnehvi.json"),
                                   recursive=True))
    if not front_paths:
        raise SystemExit(f"No pareto_frontier_qnehvi.json under {run_root}/qnehvi")
    front = json.loads(Path(front_paths[0]).read_text())
    vllm_dir = Path(front_paths[0]).parent

    # Each entry: {rec: [{...9 params...}], obj: [[-throughput, TTFT, TPOT]], ...}
    enriched = []
    for item in front:
        rec = item["rec"][0]
        obj = item["obj"][0]
        params = {k: rec.get(k) for k in (
            "tp", "max_num_seqs", "max_num_batched_tokens", "block_size",
            "enable_chunked_prefill", "scheduler_delay_factor",
            "enable_prefix_caching", "disable_custom_all_reduce", "use_v2_block_manager",
        )}
        metrics = {
            "request_throughput": -float(obj[0]),  # stored negated for maximization
            "mean_ttft_ms":        float(obj[1]),
            "mean_tpot_ms":        float(obj[2]),
        }
        enriched.append({"params": params, "metrics": metrics,
                         "rec_time_s": item.get("rec_time"), "run_time_s": item.get("run_time")})
    return enriched, vllm_dir


def find_default_in_qnehvi(run_root: Path, default_params: Dict) -> Dict | None:
    """If qNEHVI happened to evaluate the default config, return its measured
    metrics. Otherwise None."""
    matches = []
    for f in sorted(glob.glob(str(run_root / "qnehvi" / "**" / "vllm-*.json"), recursive=True)):
        v = json.loads(Path(f).read_text())
        ok = True
        for k, want in default_params.items():
            if want is None:
                continue
            got = v.get(k)
            # Booleans round-trip as strings in vLLM JSON, normalize.
            if isinstance(want, bool):
                got = str(got) if got is not None else None
                want = str(want)
            if got != want:
                ok = False
                break
        if ok:
            matches.append(v)
    return matches[0] if matches else None


def normalize(values: List[float], lower_is_better: bool) -> List[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0 for _ in values]
    if lower_is_better:
        return [(v - lo) / (hi - lo) for v in values]
    return [(hi - v) / (hi - lo) for v in values]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root")
    ap.add_argument("--weights", default="1,1,1,1",
                    help="comma-separated weights for [flexgen_latency, qnehvi_throughput, "
                         "qnehvi_TTFT, qnehvi_TPOT] (lower combined score = better)")
    args = ap.parse_args()
    run_root = Path(args.run_root).resolve()

    weights = [float(w) for w in args.weights.split(",")]
    if len(weights) != 4:
        raise SystemExit("--weights expects 4 comma-separated numbers")

    # Load Pareto sets
    f_pareto = load_flexgen_pareto(run_root)
    q_pareto, vllm_dir = load_qnehvi_pareto(run_root)
    print(f"FlexGen Pareto front: {len(f_pareto)} configs (from latency-vs-block-size)")
    print(f"qNEHVI  Pareto front: {len(q_pareto)} configs (3-objective measured)")
    print(f"Cartesian product:    {len(f_pareto) * len(q_pareto)} pairs\n")

    # Build cross-product with raw metrics
    pairs = []
    for f in f_pareto:
        for q in q_pareto:
            pairs.append({
                "f": f,
                "q": q,
                "f_lat": float(f["per_token_latency_ms"]),
                "q_thr": float(q["metrics"]["request_throughput"]),
                "q_ttft": float(q["metrics"]["mean_ttft_ms"]),
                "q_tpot": float(q["metrics"]["mean_tpot_ms"]),
            })

    # Min-max normalize each metric, combine
    norms = {
        "f_lat":  normalize([p["f_lat"]  for p in pairs], lower_is_better=True),
        "q_thr":  normalize([p["q_thr"]  for p in pairs], lower_is_better=False),
        "q_ttft": normalize([p["q_ttft"] for p in pairs], lower_is_better=True),
        "q_tpot": normalize([p["q_tpot"] for p in pairs], lower_is_better=True),
    }
    for i, p in enumerate(pairs):
        p["score"] = (
            weights[0] * norms["f_lat"][i]
            + weights[1] * norms["q_thr"][i]
            + weights[2] * norms["q_ttft"][i]
            + weights[3] * norms["q_tpot"][i]
        )

    pairs.sort(key=lambda p: p["score"])
    best = pairs[0]

    # Find the qNEHVI vLLM default measurement, if it exists
    default_params = dict(DEFAULT_QNEHVI_PARAMS_TEMPLATE)
    tuner_conf_path = run_root / "tuner_conf_qnehvi.json"
    if tuner_conf_path.exists():
        tc = json.loads(tuner_conf_path.read_text())
        default_params["tp"] = tc.get("min_world_size")
        default_params["max_num_batched_tokens"] = max(4096, tc.get("max_sequence_length", 4096))

    default_measured = find_default_in_qnehvi(run_root, default_params)

    # Render report
    out_md = run_root / "cartesian_best.md"
    out_json = run_root / "cartesian_best.json"

    out_json.write_text(json.dumps({
        "run_root": str(run_root),
        "weights": {"f_lat": weights[0], "q_thr": weights[1], "q_ttft": weights[2], "q_tpot": weights[3]},
        "n_flexgen_pareto": len(f_pareto),
        "n_qnehvi_pareto": len(q_pareto),
        "n_pairs": len(pairs),
        "best_pair": {
            "score": best["score"],
            "flexgen": best["f"],
            "qnehvi": best["q"],
            "metrics": {
                "flexgen_predicted_latency_ms": best["f_lat"],
                "qnehvi_throughput_req_s":     best["q_thr"],
                "qnehvi_mean_ttft_ms":         best["q_ttft"],
                "qnehvi_mean_tpot_ms":         best["q_tpot"],
            },
        },
        "default_qnehvi_params": default_params,
        "default_qnehvi_measured": (
            None if default_measured is None
            else {k: default_measured.get(k) for k in
                  ["request_throughput", "mean_ttft_ms", "mean_tpot_ms"]}
        ),
    }, indent=2))

    md = []
    md.append("# Cartesian-product best of (FlexGen × qNEHVI)")
    md.append("")
    md.append(f"- Run root: `{run_root}`")
    md.append(f"- FlexGen Pareto: {len(f_pareto)} configs (latency-vs-block_size front)")
    md.append(f"- qNEHVI Pareto:  {len(q_pareto)} configs (3-objective measured)")
    md.append(f"- Cartesian pairs: {len(pairs)}")
    md.append(f"- Score weights [f_lat, q_thr, q_ttft, q_tpot]: {weights}")
    md.append("")
    md.append("## Best combined pair (lowest combined-normalized score)")
    md.append("")
    md.append(f"- Combined score: **{best['score']:.4f}**")
    md.append("")
    md.append("**FlexGen side:**")
    f = best["f"]
    md.append(f"- predicted per-token latency: **{best['f_lat']:.2f} ms/token**")
    md.append(f"- gpu_batch_size={f['gpu_batch_size']}, num_gpu_batches={f['num_gpu_batches']}, block_size={f['block_size']}")
    md.append(f"- compression={f['compression']}, cpu_compute_delegate={f['cpu_compute_delegate']}, overlap_io_compute={f['overlap_io_compute']}")
    md.append(f"- weights:     {f['weights']}")
    md.append(f"- kv_cache:    {f['kv_cache']}")
    md.append(f"- activations: {f['activations']}")
    md.append("")
    md.append("**qNEHVI side (measured on real ShareGPT workload):**")
    q = best["q"]
    qm = q["metrics"]
    qp = q["params"]
    md.append(f"- throughput: **{qm['request_throughput']:.3f} req/s**")
    md.append(f"- mean TTFT:  **{qm['mean_ttft_ms']:.2f} ms**")
    md.append(f"- mean TPOT:  **{qm['mean_tpot_ms']:.2f} ms**")
    md.append(
        f"- tp={qp['tp']}, max_num_seqs={qp['max_num_seqs']}, "
        f"max_num_batched_tokens={qp['max_num_batched_tokens']}, block_size={qp['block_size']}, "
        f"enable_chunked_prefill={qp['enable_chunked_prefill']}, "
        f"scheduler_delay_factor={qp['scheduler_delay_factor']}, "
        f"enable_prefix_caching={qp['enable_prefix_caching']}, "
        f"disable_custom_all_reduce={qp['disable_custom_all_reduce']}, "
        f"use_v2_block_manager={qp['use_v2_block_manager']}"
    )
    md.append("")

    md.append("## Comparison with default configurations")
    md.append("")
    md.append("**FlexGen default policy** (no offloading, no compression, no overlap; "
              "what `experiments/run_flexgen.py` would have picked if all 240 enums were equally feasible):")
    md.append("")
    md.append(f"- {DEFAULT_FLEXGEN_PARAMS}")
    md.append("- predicted per-token latency: TBD (would need to re-run run_flexgen.py with this policy forced; "
              "the cost model can compute it but the orchestrator doesn't expose a single-policy mode).")
    md.append("")
    md.append("**vLLM/qNEHVI default** (the SCOOT paper's hypervolume reference, "
              "automatically derived from the GPU's `min_world_size` for this run):")
    md.append("")
    md.append(f"- {default_params}")
    if default_measured is not None:
        md.append(f"- measured throughput: **{default_measured['request_throughput']:.3f} req/s**")
        md.append(f"- measured TTFT:       **{default_measured['mean_ttft_ms']:.2f} ms**")
        md.append(f"- measured TPOT:       **{default_measured['mean_tpot_ms']:.2f} ms**")
        md.append("")
        md.append("**Improvement of the best Cartesian pair vs default:**")
        md.append("")
        thr_imp  = 100 * (qm["request_throughput"] - default_measured["request_throughput"]) / default_measured["request_throughput"]
        ttft_imp = 100 * (default_measured["mean_ttft_ms"] - qm["mean_ttft_ms"]) / default_measured["mean_ttft_ms"]
        tpot_imp = 100 * (default_measured["mean_tpot_ms"] - qm["mean_tpot_ms"]) / default_measured["mean_tpot_ms"]
        md.append(f"- throughput: {thr_imp:+.2f}%")
        md.append(f"- TTFT:       {ttft_imp:+.2f}% (negative = worse, positive = lower latency = better)")
        md.append(f"- TPOT:       {tpot_imp:+.2f}%")
    else:
        md.append("- _Default config was not evaluated by qNEHVI in this run._")
        md.append("  qNEHVI's BO suggest mechanism rarely picks the exact default config; the closest matched")
        md.append("  trial would still be a suggested point, not the default. To get a measured default,")
        md.append("  run `vLLM_default/scripts/benchmark_vllm_defaults.sh` once with the same model/dataset/qps.")

    md.append("")
    md.append("## Top 5 pairs by combined score")
    md.append("")
    md.append("| rank | score | f_latency | q_throughput | q_TTFT | q_TPOT | flexgen_gbs/num_gb/q | qnehvi_tp/mns/mnbt |")
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for i, p in enumerate(pairs[:5], start=1):
        f = p["f"]; q = p["q"]
        md.append(
            f"| {i} | {p['score']:.4f} | {p['f_lat']:.2f} | {p['q_thr']:.3f} | "
            f"{p['q_ttft']:.2f} | {p['q_tpot']:.2f} | "
            f"{f['gpu_batch_size']}/{f['num_gpu_batches']}/{f['compression']} | "
            f"{q['params']['tp']}/{q['params']['max_num_seqs']}/{q['params']['max_num_batched_tokens']} |"
        )
    md.append("")
    md.append("## Caveat")
    md.append("")
    md.append(
        "FlexGen and qNEHVI tune disjoint parameter spaces:\n"
        "- FlexGen tunes the FlexGen engine's batching/placement (8 params, including "
        "weights/KV/activations placement across GPU/CPU/disk and int4/fp16 compression).\n"
        "- qNEHVI tunes the vLLM serving engine's runtime args (9 params, including "
        "tensor-parallel size, max_num_seqs, scheduler_delay_factor, etc.).\n\n"
        "These are NOT the same engine. The \"best Cartesian pair\" reported above is an "
        "analytical Pareto-of-Paretos pick, not a deployable configuration. Concretely: you "
        "would deploy *either* FlexGen's chosen policy (in the FlexGen engine) *or* qNEHVI's "
        "chosen vLLM args (in vLLM 0.5.5), not both simultaneously."
    )
    out_md.write_text("\n".join(md))

    print(f"Best pair score: {best['score']:.4f}")
    print(f"  FlexGen side : lat={best['f_lat']:.2f} ms/token, gbs={best['f']['gpu_batch_size']}, "
          f"num_gb={best['f']['num_gpu_batches']}, q={best['f']['compression']}")
    print(f"  qNEHVI side  : thr={best['q_thr']:.3f} req/s, TTFT={best['q_ttft']:.2f} ms, "
          f"TPOT={best['q_tpot']:.2f} ms")
    if default_measured:
        print(f"  vs default   : measured (thr={default_measured['request_throughput']:.3f}, "
              f"TTFT={default_measured['mean_ttft_ms']:.2f}, TPOT={default_measured['mean_tpot_ms']:.2f})")
    print(f"\nWrote {out_md}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
