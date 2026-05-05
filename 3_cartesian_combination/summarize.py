"""Combine qNEHVI + FlexGen results from one flexgen_qnehvi run into a unified
report.md. Reads ${RUN_ROOT} from argv[1].

Layout expected under RUN_ROOT:
  qnehvi/<bo_*>/exp0/vllm-*.json + rec_history_qnehvi.json
  qnehvi/<bo_*>/exp0/pareto_frontier_qnehvi.json
  flexgen/flexgen_*.json   (per-token-latency optimization)
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

OBJ_KEYS = ["request_throughput", "mean_ttft_ms", "mean_tpot_ms"]
PARAM_KEYS = [
    "tp", "max_num_seqs", "max_num_batched_tokens", "block_size",
    "enable_chunked_prefill", "scheduler_delay_factor",
    "enable_prefix_caching", "disable_custom_all_reduce", "use_v2_block_manager",
]


def load_qnehvi(run_root: Path) -> dict:
    files = sorted(glob.glob(str(run_root / "qnehvi" / "**" / "vllm-*.json"), recursive=True))
    trials = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if any(d.get(k) is None for k in OBJ_KEYS):
            continue
        trials.append({
            "params": {k: d[k] for k in PARAM_KEYS},
            "metrics": {k: float(d[k]) for k in OBJ_KEYS},
        })
    if not trials:
        return {"trials": []}
    bt = max(trials, key=lambda t: t["metrics"]["request_throughput"])
    bf = min(trials, key=lambda t: t["metrics"]["mean_ttft_ms"])
    bo = min(trials, key=lambda t: t["metrics"]["mean_tpot_ms"])
    front_path = list(run_root.glob("qnehvi/**/pareto_frontier_qnehvi.json"))
    pareto_size = 0
    if front_path:
        try:
            pareto_size = len(json.load(open(front_path[0])))
        except Exception:
            pass
    rec_history = list(run_root.glob("qnehvi/**/rec_history_qnehvi.json"))
    rt_total_s = 0.0
    fails = 0
    if rec_history:
        h = json.load(open(rec_history[0]))
        rt_total_s = sum(float(x.get("run_time", 0) or 0) + float(x.get("rec_time", 0) or 0) for x in h)
        fails = sum(1 for x in h if x.get("obj") is None)
    return {
        "trials": trials, "best_throughput": bt, "best_ttft": bf, "best_tpot": bo,
        "pareto_size": pareto_size, "wall_clock_s": rt_total_s, "fails": fails,
    }


def load_flexgen(run_root: Path) -> dict | None:
    files = sorted(glob.glob(str(run_root / "flexgen" / "flexgen_*.json")))
    if not files:
        return None
    d = json.load(open(files[-1]))  # latest run
    return {
        "best_policy": d.get("best_policy", {}),
        "input_system": d.get("input", {}).get("system", {}),
        "input_model": d.get("input", {}).get("model", {}),
        "input_workload": d.get("input", {}).get("workload", {}),
        "machine_id": d.get("machine_id"),
        "top_k_count": len(d.get("top_k", [])),
        "search_time_s": d.get("search_time_s"),
    }


def fmt_qnehvi_config(t: dict) -> list[str]:
    p = t["params"]
    m = t["metrics"]
    return [
        f"  - throughput={m['request_throughput']:.3f} req/s, TTFT={m['mean_ttft_ms']:.2f} ms, TPOT={m['mean_tpot_ms']:.2f} ms",
        f"  - tp={p['tp']}, max_num_seqs={p['max_num_seqs']}, max_num_batched_tokens={p['max_num_batched_tokens']}, "
        f"block_size={p['block_size']}, enable_chunked_prefill={p['enable_chunked_prefill']}, "
        f"scheduler_delay_factor={p['scheduler_delay_factor']}, enable_prefix_caching={p['enable_prefix_caching']}, "
        f"disable_custom_all_reduce={p['disable_custom_all_reduce']}, use_v2_block_manager={p['use_v2_block_manager']}",
    ]


def write_report(run_root: Path) -> None:
    qn = load_qnehvi(run_root)
    fg = load_flexgen(run_root)
    out = ["# flexgen + qNEHVI run summary", "", f"Run dir: `{run_root}`", "", "---", ""]

    out += ["## qNEHVI (vLLM serving optimization)", ""]
    if not qn["trials"]:
        out += ["_TBD / pending — no successful trials yet._", ""]
    else:
        out += [
            f"- Successful trials: **{len(qn['trials'])}**",
            f"- Pareto frontier size: **{qn['pareto_size']}**",
            f"- Wall-clock: total **{qn['wall_clock_s']/60:.1f} min** ({qn['wall_clock_s']/3600:.2f} h); failed configs: {qn['fails']}",
            "",
            "**Best throughput:**",
            *fmt_qnehvi_config(qn["best_throughput"]),
            "",
            "**Best TTFT:**",
            *fmt_qnehvi_config(qn["best_ttft"]),
            "",
            "**Best TPOT:**",
            *fmt_qnehvi_config(qn["best_tpot"]),
            "",
        ]

    out += ["---", "", "## FlexGen (cost-model + LP placement optimization)", ""]
    if fg is None:
        out += ["_TBD / pending — no flexgen result yet._", ""]
    else:
        bp = fg["best_policy"]
        sys = fg["input_system"]
        m = fg["input_model"]
        wl = fg["input_workload"]
        out += [
            f"- Machine: `{fg['machine_id']}`",
            f"- Top-k candidates considered: {fg['top_k_count']}",
            f"- Live capacity: GPU VRAM={sys.get('gpu_vram_gb', 0):.1f} GB, "
            f"RAM={sys.get('ram_gb', 0):.1f} GB, disk={sys.get('disk_gb', 0):.1f} GB",
            f"- Calibration: PCIe={sys.get('pcie_bw_gbs', 0):.1f} GB/s, "
            f"disk={sys.get('disk_bw_gbs', 0):.1f} GB/s, fp16={sys.get('tflops_fp16', 0):.1f} TFLOPS, "
            f"int4={sys.get('tflops_int4', 0):.1f} TFLOPS",
            f"- Model spec: {m.get('hf_id')}; layers={m.get('num_layers')}, hidden={m.get('hidden_dim')}, "
            f"heads={m.get('num_heads')}/kv={m.get('num_kv_heads')}",
            f"- Workload: prompt_len={wl.get('prompt_len')}, decode_len={wl.get('decode_len')}",
            "",
            "**Best policy (predicted):**",
            "",
            f"- gpu_batch_size={bp.get('gpu_batch_size')}, num_gpu_batches={bp.get('num_gpu_batches')}, block_size={bp.get('block_size')}",
            f"- compression={bp.get('compression')}, cpu_compute_delegate={bp.get('cpu_compute_delegate')}, overlap_io_compute={bp.get('overlap_io_compute')}",
            f"- weights placement: {bp.get('weights')}",
            f"- kv_cache placement: {bp.get('kv_cache')}",
            f"- activations placement: {bp.get('activations')}",
            f"- Predicted per-token latency: **{bp.get('per_token_latency_ms', 0):.2f} ms/token**",
            "",
        ]

    out += [
        "---", "",
        "## Notes", "",
        "- qNEHVI optimizes 9 vLLM serving parameters using BoTorch's mixed-space GP + Noisy Expected Hypervolume Improvement; objective is the Pareto front of (throughput, TTFT, TPOT) measured on real ShareGPT workload.",
        "- FlexGen optimizes 8 inference-engine parameters (gbs, num_gb, compression, cpu_delegate, overlap, plus 9 continuous placement fractions for weights/KV/activations across GPU/CPU/disk) using a cost-model + LP. Output is a single best policy minimizing predicted per-token latency for the given (prompt_len, decode_len) workload.",
        "- The two methods optimize **different parameter spaces** and produce **different output formats**. Direct numeric comparison is not meaningful; the value here is seeing how each method profiles the same hardware + model.",
    ]

    out_path = run_root / "report.md"
    out_path.write_text("\n".join(out))
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root")
    args = ap.parse_args()
    write_report(Path(args.run_root))


if __name__ == "__main__":
    main()
