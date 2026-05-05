# Experiment 1 — vLLM Serving-Runtime Tuning ($\Omega_S$)

Five-method comparison on the **vLLM serving runtime** parameter space:

$$\Omega_S = \big\{\text{tp}, \text{max-num-seqs}, \text{max-num-batched-tokens}, \text{block-size},$$
$$\text{enable-chunked-prefill}, \text{scheduler-delay-factor},$$
$$\text{enable-prefix-caching}, \text{disable-custom-all-reduce}, \text{use-v2-block-manager}\big\}$$

That is nine knobs, mostly mixed-type (categorical, boolean, integer
power-of-two). Each evaluation is an end-to-end vLLM stress test on
Llama-2-7B-Chat-HF over a 1,000-prompt ShareGPT slice at $\lambda = 5$ req/s,
and we measure throughput, mean TTFT, and mean TPOT.

> **Prerequisites:** complete the *Setup* section in the
> [top-level README](../README.md) and `source ../setenv.sh` before
> submitting any sbatch.

## Methods

| Method | One-line idea | Code root |
| --- | --- | --- |
| **vLLM defaults** | a single benchmark with vLLM's stock arguments — no tuning | `vllm_default/` |
| **Random search** | uniformly sample $N=30$ configurations | `combined_pipeline/random_search.py` |
| **LLM Agent** | a frontier LLM proposes the next config given the metric history | `llm_agent/agent_search.py` |
| **SCOOT (HEBO + EHVI)** | original SCOOT: HEBO Gaussian process + EHVI acquisition | `scoot_baseline/bo_scoot.py` |
| **SCOOT-qNEHVI (ours)** | SCOOT with a BoTorch mixed-space GP and qNEHVI acquisition | `scoot_qnehvi/bo_scoot_qnehvi.py` |

## How they're orchestrated

`combined_pipeline/run_configs/combined_*.sbatch` runs **all five** methods
sequentially on a single GPU allocation. The five stages share the same
model, dataset, request rate and benchmark client, and write per-trial
results into a single tree per run:

```
combined_pipeline/results/combined/<run_dir>/
├── driver.out / driver.err                  # consolidated logs
├── submitted.sbatch
├── tuner_conf_scoot.json                    # min_world_size + max_seq_len
├── tuner_conf_qnehvi.json
└── tune_res/
    ├── scoot/   bo_*/exp0/{vllm-*.json (30), rec_history.json}
    ├── qnehvi/  bo_*/exp0/{vllm-*.json (30), rec_history_qnehvi.json,
    │                       pareto_frontier_qnehvi.json}
    ├── default/ vllm-*.json (1)
    └── random/  bo_*/exp0/{vllm-*.json (30), rec_history_random.json}
```

The LLM Agent baseline lives in a separate `llm_agent/` directory because
it needs a separate Slurm submission (it talks to an external LLM API).

## Files

```
1_serving_runtime_scoot/
├── README.md                                         # this file
├── combined_pipeline/                                # 5-method orchestrator
│   ├── _bench_runner.py                              # vLLM benchmark runner used by random search
│   ├── random_search.py                              # uniform-sampling baseline
│   ├── results/_make_report.py                       # cross-allocation aggregator
│   └── run_configs/
│       ├── combined_rtx6000ada.sbatch                # example: 2x rtx6000ada (Ada, 48 GiB)
│       ├── combined_rtxa6000.sbatch                  # example: 2x rtxa6000 (Ampere, 48 GiB)
│       ├── combined_rtxa4000.sbatch                  # example: 4x rtxa4000 (Ampere, 16 GiB)
│       └── combined_l40s_preemptible.sbatch          # example: 2x L40S (preemptible)
├── scoot_baseline/                                   # SCOOT (HEBO + EHVI) -- original code
│   ├── bo_scoot.py                                   # main BO loop
│   ├── benchmark_pipeline.sh                         # vLLM benchmark dispatcher
│   ├── clients/                                      # vLLM-derived benchmark client
│   ├── hebo/                                         # HEBO library bundled
│   └── tuner_conf/                                   # min_world_size + max_seq_len autoconf
├── scoot_qnehvi/                                     # SCOOT-qNEHVI -- modified code
│   ├── bo_scoot_qnehvi.py                            # ...replaces HEBO+EHVI with BoTorch+qNEHVI
│   └── scoot_botorch/                                # mixed-space GP + qNEHVI acquisition
├── llm_agent/                                        # LLM-as-optimizer baseline
│   ├── agent_search.py
│   ├── _bench_runner.py
│   └── run_configs/llm_agent_*.sbatch
└── vllm_default/                                     # single-config benchmark wrapper
    ├── scripts/{benchmark,run_server,run_client}_vllm_defaults.sh
    └── run_configs/vllm_default_llama2_sharegpt.sbatch
```

## Submitting a run

```bash
cd /path/to/final
source setenv.sh

# Pick the sbatch that names the GPU type your cluster has, e.g.:
sbatch 1_serving_runtime_scoot/combined_pipeline/run_configs/combined_rtxa6000.sbatch
```

The four sbatch variants under `combined_pipeline/run_configs/` differ only
in the `--gres` line and a few resource caps that we tuned for the GPU
type. Pick whichever matches your hardware, then **edit it** to use your
cluster's `--account` / `--partition` / `--qos` / exact `--gres` syntax (see
the top-level README §6 for guidance).

The `combined_l40s_preemptible.sbatch` is an example for clusters that
expose a preemptible queue (sometimes called a back-fill or low-priority
queue). Remove it if your cluster has no such queue.

## Quick smoke test (~5 minutes)

```bash
BO_LOOP=2 RANDOM_TRIALS=2 NUM_REQUESTS=50 \
    sbatch 1_serving_runtime_scoot/combined_pipeline/run_configs/combined_rtxa6000.sbatch
```

## Resuming a partial run

All BO methods write `rec_history*.json` after every trial. To re-launch
into the same run dir and skip already-completed trials:

```bash
sbatch --export=ALL,RESUME_DIR=/path/to/run/dir \
    1_serving_runtime_scoot/combined_pipeline/run_configs/combined_rtxa6000.sbatch
```

## Aggregating results into one Markdown table

After one or more runs complete, a small aggregator walks every result
directory it can find and emits a cross-allocation Markdown report:

```bash
conda activate scoot-botorch
python 1_serving_runtime_scoot/combined_pipeline/results/_make_report.py
# writes 1_serving_runtime_scoot/combined_pipeline/results/report.md
```

The aggregator pulls best-per-objective and Pareto-frontier statistics from
each method's `vllm-*.json` files and produces one row per (allocation,
method) pair.

## Wall-clock guidance

Per-trial benchmarks take 4-6 minutes on a 2-GPU 48-GiB allocation
(Llama-2-7B-Chat, 1,000 prompts at 5 req/s). The full five-stage pipeline
is therefore approximately:

| Stage | Trials | Per trial | Stage total |
| --- | --- | --- | --- |
| vLLM default | 1 | ~4 min | ~4 min |
| SCOOT (HEBO + EHVI) | 30 | ~5 min + 7 s suggest | ~2.0 h |
| SCOOT-qNEHVI | 30 | ~5 min + 12 s suggest | ~2.2 h |
| LLM Agent | 30 | ~5 min | ~2.5 h |
| Random | 30 | ~5 min | ~2.5 h |
| **Total** | **121** | -- | **~9-10 h** |

Default sbatch wall-time is `--time=1-00:00:00` (24 h). Raise it if your
allocation is significantly slower than 48-GiB-class GPUs.

## What you can do with the results

1. **Pareto-front extraction.** Each method's `rec_history*.json` plus its
   `vllm-*.json` files give you all $N$ measured triples (throughput,
   TTFT, TPOT). The aggregator computes per-method Pareto fronts.
2. **Per-objective best.** Pick the trial that wins on each individual
   metric in isolation (best throughput; best TTFT; best TPOT).
3. **Hypervolume comparison.** Use the per-allocation vLLM-default
   benchmark as the hypervolume reference point and rank methods by HV
   improvement -- this is the metric `_make_report.py` reports as
   "Recommended config (max hypervolume vs vLLM default)".

## Method differences worth knowing

**SCOOT-qNEHVI vs SCOOT.** Three changes:

1. *Noise-aware acquisition.* qNEHVI marginalizes over the GP posterior of
   past observations, where EHVI conditions on point estimates. On real
   noisy benchmarks this matters with a small ($N = 30$) budget.
2. *Native mixed-space kernel.* BoTorch's `MixedSingleTaskGP` uses a
   Hamming-style kernel for categorical dimensions; HEBO uses encoding
   tricks. Six of the nine $\Omega_S$ dimensions are categorical or boolean.
3. *Wider exploration.* SCOOT-qNEHVI scores 2,048 Sobol candidates per BO
   step under the acquisition; HEBO uses gradient-based optimization which
   can stall on the categorical components.
