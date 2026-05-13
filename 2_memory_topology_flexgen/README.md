# Experiment 2 — FlexGen Memory-Topology Search ($\Omega_T$)

A head-to-head between FlexGen's two-level (outer enumeration + inner LP)
policy search and a uniform 3,375-point grid baseline. Both methods
optimize over the **memory topology** parameter space:

$$\Omega_T = \big\{b, n, q, \text{delegate}, \text{overlap},\ (w_g,w_c,w_d), (h_g,h_c,h_d), (c_g,c_c,c_d)\big\}$$

That is five enumerated knobs (GPU batch size $b$, number of GPU batches
$n$, compression $q$, CPU-compute delegation, I/O-compute overlap) plus
three continuous placement-fraction triples (weights, activations, KV cache)
that each must sum to one.

Unlike Experiment 1, the FlexGen search is **analytical**: it evaluates
configurations against an empirically-calibrated cost model rather than a
real serving benchmark. So the search itself completes in ~10 seconds on a
laptop and does not require a GPU job allocation.

> **Prerequisites:** complete the *Setup* section in the
> [top-level README](../README.md). The FlexGen search needs only the
> `scoot-botorch` environment.

## Methods

| Method | Inner search over the 9 placement fractions | Wall-clock |
| --- | --- | --- |
| **FlexGen LP** | exact LP solve per outer enum point (PuLP / CBC) | ~10 s (240 LP solves) |
| **Grid search** | enumerate 3,375 discrete grid points (step = 0.25) per outer enum | ~0.1 s (vectorized) |

Both methods enumerate the same 240 outer points
($6 \times 5 \times 2 \times 2 \times 2$). They differ only in how the
inner continuous placements are searched.

## Files

```
2_memory_topology_flexgen/
├── README.md                                   # this file
└── flexgen_solver/                             # the FlexGen project
    ├── README.md                               # upstream notes (kept for reference)
    ├── pipeline.py                             # one-command pipeline driver
    ├── experiments/
    │   ├── run_flexgen.py                      # FlexGen LP search (inner = LP)
    │   ├── run_grid_baseline.py                # grid baseline (inner = 3,375-point grid)
    │   └── compare_lp_vs_grid.py               # per-model comparison harness
    ├── src/flexgen/                            # core library
    │   ├── cost_model.py                       # FlexGen analytical cost model
    │   ├── lp_formulation.py                   # the inner LP (PuLP)
    │   ├── policy_search.py                    # outer enumeration loop
    │   ├── grid_search_baseline.py             # discrete grid baseline
    │   ├── system_probe.py                     # GPU/RAM/disk discovery
    │   └── calibration.py                      # PCIe/disk/TFLOPS bench
    ├── analysis/                               # plotting + report-building helpers
    └── config_flexgen*.yml                     # workload + system config templates
```

## Running the LP-vs-grid comparison

The FlexGen solver only needs each model's `config.json` (a few KB), plus a
calibration of host PCIe / disk / GPU TFLOPS that is auto-cached after the
first invocation. The comparison itself is pure arithmetic — it runs on
any machine with Python and `scoot-botorch` installed; a CUDA-capable GPU
is needed only to make the calibration numbers representative (without
CUDA the script falls back to a hard-coded 16 GB/s PCIe estimate and
prints a warning that the resulting latencies are not GPU-representative).

`compare_lp_vs_grid.py` accepts either pre-existing LP and grid result
JSONs (via `--lp` / `--grid`) **or** the `--run-both` flag, which runs
both methods inline. The first time you run a given model you want
`--run-both`:

```bash
cd /path/to/optimization-llm-inference
source setenv.sh
conda activate scoot-botorch
cd 2_memory_topology_flexgen/flexgen_solver

# Each invocation runs both methods on one model under simulated VRAM.
# (--sim-gpu-gb forces memory pressure even on a large GPU.)
python experiments/compare_lp_vs_grid.py --run-both \
    --model HuggingFaceTB/SmolLM2-135M-Instruct  --sim-gpu-gb 0.3
python experiments/compare_lp_vs_grid.py --run-both \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0   --sim-gpu-gb 0.3
python experiments/compare_lp_vs_grid.py --run-both \
    --model Qwen/Qwen2-7B                        --sim-gpu-gb 2.0
python experiments/compare_lp_vs_grid.py --run-both \
    --model mistralai/Mistral-7B-v0.1            --sim-gpu-gb 2.0

python analysis/generate_report.py
# -> writes report/cross_model_results.md
```

Each `compare_lp_vs_grid.py` call produces one
`experiments/results/<model>/comparison_*.json` plus per-model plots under
`analysis/plots/`. The aggregated report lives at
`report/cross_model_results.md`.

## Models used

| Model | HuggingFace id | Params | `--sim-gpu-gb` |
| --- | --- | --- | --- |
| SmoLLM2-135M | `HuggingFaceTB/SmolLM2-135M-Instruct` | 0.135 B | use real VRAM (model fits) |
| TinyLlama-1.1B | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1 B | 0.3 GiB |
| Qwen2-7B | `Qwen/Qwen2-7B` | 6.5 B | 2.0 GiB |
| Mistral-7B-v0.1 | `mistralai/Mistral-7B-v0.1` | 7.0 B | 2.0 GiB |

These are public on HuggingFace and downloaded on-demand the first time
each command runs (only `config.json` is needed, ~4 KB per model).

## What "wins" looks like

Indicative numbers from one CUDA-capable host (your absolute latencies
will vary with the calibrated PCIe / disk / TFLOPS — see "Calibration"
below); the **shape** of the LP-vs-grid gap should reproduce:

| Model | LP latency | Grid latency | LP advantage |
| --- | --- | --- | --- |
| SmoLLM2-135M | 1.45 ms | 1.45 ms | tie (no offloading) |
| TinyLlama-1.1B | 33.04 ms | 42.50 ms | $-22.3\%$ |
| Qwen2-7B | 120.63 ms | 158.19 ms | $-23.8\%$ |
| Mistral-7B | 166.56 ms | 190.69 ms | $-12.7\%$ |

When the model fits entirely in GPU, both methods find the trivial all-GPU
corner and tie. When offloading is required, the LP picks exact fractional
placements (e.g., $w_g = 0.658$) that the 0.25-step grid can only
approximate.

### Calibration

On the first run for a given hostname+GPU combination FlexGen benchmarks
PCIe bandwidth, disk bandwidth, and per-dtype TFLOPS, then caches the
result under `configs/system_calibration/<hostname>_<gpu>.json`. Subsequent
runs reuse the cache. To force re-benchmarking pass `--recalibrate`. On a
host without CUDA the script logs a clear warning and falls back to a
hard-coded 16 GB/s PCIe and a CPU-measured "tflops" — useful for testing
the plumbing but not for comparison numbers.

## Why no Slurm sbatch in this directory?

The FlexGen comparison runs in seconds on a laptop. There is no need for a
GPU job allocation -- the solver only reads model configs and does
arithmetic. If you want to integrate FlexGen with the real-benchmark
pipeline (qNEHVI), see
[../3_cartesian_combination/](../3_cartesian_combination/), which does
schedule a Slurm job for FlexGen's serving validation.

## Key formulation: the inner LP

For a fixed outer enumeration $e$, the inner search over the placement
fractions is the linear program

$$\min_{w, h, c}\ t_{\text{block}}(e, w, h, c)$$

subject to capacity and stochasticity constraints

$$w \cdot W + c \cdot KV(b, n) + h \cdot A \le \rho \cdot V_{\text{GPU}}$$
$$w_g + w_c + w_d = 1,\quad c_g + c_c + c_d = 1,\quad h_g + h_c + h_d = 1$$
$$0 \le w_*, c_*, h_* \le 1$$

where $W$, $A$, and $KV(b, n)$ are weight, activation, and KV-cache sizes
per layer; $V_{\text{GPU}}$ is GPU VRAM and $\rho \in (0, 1)$ a safety
factor (default $\rho = 0.9$). The objective $t_{\text{block}}$ is a
linear combination of compute time and four I/O terms (weight load, KV
transfer, activation transfer) -- linear in the placement variables, so the
inner problem is a true linear program and solved exactly by PuLP/CBC.

The grid baseline samples each placement vector on $\{0, 0.25, 0.5, 0.75,
1\}^3$, evaluates the same cost expression for each of the $5^3 = 125$
combinations per vector ($125^3$ total reduced to $3{,}375$ after enforcing
the simplex constraints), and reports the minimum. It is faster but
strictly worse on memory-pressured models.
