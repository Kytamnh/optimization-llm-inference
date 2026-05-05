"""Extract a multi-objective Pareto frontier from a FlexGen result JSON.

FlexGen's policy_search optimizes a single objective (predicted per-token
latency). It returns the top-20 candidates by latency in `top_k_candidates`.
This script promotes those into multi-objective views:

  Pareto front A — (latency MIN, block_size MAX)
      The natural latency vs throughput trade-off:
      - small batch: lower latency but less parallelism
      - large batch: higher latency but more concurrent work per "block"

  Pareto front B — (latency MIN, gpu_residency MAX)
      latency vs how much state lives on the (faster) GPU:
      - all on GPU: lowest latency but uses full VRAM
      - some on CPU/disk: higher latency but frees VRAM for other workloads
      - gpu_residency is the average of (w_g, c_g, h_g)

Outputs (next to the input JSON):
  <name>_pareto.json   — both fronts, full configs
  <name>_pareto.md     — markdown summary tables
  <name>_pareto.csv    — flat CSV of all 20 candidates with both objectives

Usage:
    python flexgen_pareto.py /path/to/flexgen_*.json [more JSONs...]
    python flexgen_pareto.py --run-root /path/to/results/<run_dir>
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Dict, List


def _gpu_residency(c: Dict) -> float:
    return (c["weights"]["gpu"] + c["kv_cache"]["gpu"] + c["activations"]["gpu"]) / 3.0


def _enrich(c: Dict) -> Dict:
    return {
        **c,
        "throughput_tok_s_per_replica": round(1000.0 / c["per_token_latency_ms"], 4),
        "gpu_residency": round(_gpu_residency(c), 4),
    }


def _pareto_indices(values: List[tuple[float, float]],
                    minimize: tuple[bool, bool]) -> List[int]:
    """Return indices of non-dominated points. Each value is a 2-tuple."""
    sign = (-1.0 if minimize[0] else 1.0, -1.0 if minimize[1] else 1.0)
    # higher-is-better view
    pts = [(v[0] * sign[0], v[1] * sign[1]) for v in values]
    out = []
    for i, p in enumerate(pts):
        dominated = False
        for j, q in enumerate(pts):
            if i == j:
                continue
            if q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1]):
                dominated = True
                break
        if not dominated:
            out.append(i)
    return out


def _fmt_placement(p: Dict) -> str:
    return f"GPU={p['gpu']:.2f}/CPU={p['cpu']:.2f}/DISK={p['disk']:.2f}"


def _fmt_row(c: Dict) -> str:
    return (
        f"  - latency=**{c['per_token_latency_ms']:.2f} ms/token**, "
        f"throughput={c['throughput_tok_s_per_replica']:.1f} tok/s/replica, "
        f"block_size={c['block_size']}, gpu_residency={c['gpu_residency']:.2f}\n"
        f"    - gbs={c['gpu_batch_size']}, num_gb={c['num_gpu_batches']}, "
        f"q={c['compression']}, delegate={c['cpu_compute_delegate']}, "
        f"overlap={c['overlap_io_compute']}\n"
        f"    - weights:     {_fmt_placement(c['weights'])}\n"
        f"    - kv_cache:    {_fmt_placement(c['kv_cache'])}\n"
        f"    - activations: {_fmt_placement(c['activations'])}"
    )


def process_one(json_path: Path) -> Dict:
    d = json.loads(json_path.read_text())
    cands = [_enrich(c) for c in d.get("top_k_candidates", [])]
    if not cands:
        raise RuntimeError(f"No top_k_candidates in {json_path}")

    # Front A: minimize latency, maximize block_size
    front_a_idx = _pareto_indices(
        [(c["per_token_latency_ms"], c["block_size"]) for c in cands],
        minimize=(True, False),
    )
    front_a = sorted(
        [cands[i] for i in front_a_idx],
        key=lambda c: c["per_token_latency_ms"],
    )

    # Front B: minimize latency, maximize gpu_residency
    front_b_idx = _pareto_indices(
        [(c["per_token_latency_ms"], c["gpu_residency"]) for c in cands],
        minimize=(True, False),
    )
    front_b = sorted(
        [cands[i] for i in front_b_idx],
        key=lambda c: c["per_token_latency_ms"],
    )

    payload = {
        "source_json": str(json_path),
        "machine_id": d.get("machine_id"),
        "model": d.get("input", {}).get("model", {}).get("hf_id"),
        "workload": d.get("input", {}).get("workload"),
        "system": d.get("input", {}).get("system"),
        "candidates_total": len(cands),
        "pareto_latency_vs_block_size": front_a,
        "pareto_latency_vs_gpu_residency": front_b,
        "all_candidates": cands,
    }

    base = json_path.with_suffix("")
    json_out = base.with_name(base.name + "_pareto.json")
    md_out = base.with_name(base.name + "_pareto.md")
    csv_out = base.with_name(base.name + "_pareto.csv")

    json_out.write_text(json.dumps(payload, indent=2))

    md = []
    md.append(f"# FlexGen Pareto fronts — `{json_path.name}`")
    md.append("")
    md.append(f"- Source: `{json_path}`")
    md.append(f"- Machine: `{payload['machine_id']}`")
    md.append(f"- Model: `{payload['model']}`")
    wl = payload["workload"] or {}
    md.append(f"- Workload: prompt_len={wl.get('prompt_len')}, decode_len={wl.get('decode_len')}")
    sys = payload["system"] or {}
    md.append(
        f"- GPU VRAM={sys.get('gpu_vram_gb', 0):.2f} GiB, "
        f"PCIe={sys.get('pcie_bw_gbs', 0):.1f} GB/s, "
        f"fp16={sys.get('tflops_fp16', 0):.1f} TFLOPS, "
        f"int4={sys.get('tflops_int4', 0):.1f} TFLOPS"
    )
    md.append(f"- Candidates considered (top-20 by predicted latency): **{len(cands)}**")
    md.append("")
    md.append("## Pareto front A: latency (min) vs block_size (max)")
    md.append("")
    md.append(f"Non-dominated set size: **{len(front_a)}**")
    md.append("")
    for c in front_a:
        md.append(_fmt_row(c))
        md.append("")
    md.append("## Pareto front B: latency (min) vs gpu_residency (max)")
    md.append("")
    md.append(f"Non-dominated set size: **{len(front_b)}**")
    md.append("")
    for c in front_b:
        md.append(_fmt_row(c))
        md.append("")
    md_out.write_text("\n".join(md))

    if cands:
        with csv_out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "per_token_latency_ms", "throughput_tok_s_per_replica",
                "gpu_batch_size", "num_gpu_batches", "block_size",
                "compression", "cpu_compute_delegate", "overlap_io_compute",
                "weights_gpu", "weights_cpu", "weights_disk",
                "kv_gpu", "kv_cpu", "kv_disk",
                "act_gpu", "act_cpu", "act_disk",
                "gpu_residency",
                "in_pareto_latency_vs_block",
                "in_pareto_latency_vs_residency",
            ])
            front_a_set = {id(c) for c in front_a}
            front_b_set = {id(c) for c in front_b}
            for c in cands:
                w.writerow([
                    c["per_token_latency_ms"], c["throughput_tok_s_per_replica"],
                    c["gpu_batch_size"], c["num_gpu_batches"], c["block_size"],
                    c["compression"], c["cpu_compute_delegate"], c["overlap_io_compute"],
                    c["weights"]["gpu"], c["weights"]["cpu"], c["weights"]["disk"],
                    c["kv_cache"]["gpu"], c["kv_cache"]["cpu"], c["kv_cache"]["disk"],
                    c["activations"]["gpu"], c["activations"]["cpu"], c["activations"]["disk"],
                    c["gpu_residency"],
                    int(id(c) in front_a_set),
                    int(id(c) in front_b_set),
                ])

    print(f"Wrote {json_out}")
    print(f"Wrote {md_out}")
    print(f"Wrote {csv_out}")
    print(f"  Pareto A (latency-vs-block): {len(front_a)} configs")
    print(f"  Pareto B (latency-vs-gpu_residency): {len(front_b)} configs")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Pareto fronts from FlexGen result JSON")
    ap.add_argument("inputs", nargs="*", help="One or more flexgen_*.json paths")
    ap.add_argument("--run-root", help="Process every flexgen/flexgen_*.json under this dir")
    args = ap.parse_args()

    paths: list[Path] = []
    if args.run_root:
        paths.extend(Path(p) for p in glob.glob(f"{args.run_root}/**/flexgen_*.json",
                                               recursive=True))
    paths.extend(Path(p) for p in args.inputs)
    paths = [p for p in paths if not p.name.endswith("_pareto.json")]
    if not paths:
        raise SystemExit("No input JSONs given (use positional args or --run-root)")

    for p in paths:
        if not p.exists():
            print(f"SKIP missing: {p}")
            continue
        try:
            process_one(p)
        except Exception as e:
            print(f"FAIL {p}: {e}")


if __name__ == "__main__":
    main()
