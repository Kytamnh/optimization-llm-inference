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

`combined_pipeline/run_configs/combined_*.sbatch` runs **all five** stages
sequentially on a single GPU allocation. Stages 1-4 share the same model,
dataset, request rate, and benchmark client. Stage 5 (LLM Agent) is
**gated on `ANTHROPIC_API_KEY`**: if the key is exported (in `env.sh` or
the submit env) Stage 5 runs after Stage 4; if it is empty Stage 5 prints
a "skipping" message and the job ends with Stages 1-4 done. Per-trial
results land in a single tree per run:

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
    ├── random/  bo_*/exp0/{vllm-*.json (30), rec_history_random.json}
    └── agent/   bo_*/exp0/{vllm-*.json (30), rec_history_agent.json,    # if Stage 5 ran
                            logs/agent_iter_*.txt}
```

There is also a standalone wrapper at `llm_agent/run_configs/llm_agent_*.sbatch`
that runs **only** the LLM-Agent stage on a fresh allocation. Use it if
you want to add agent results to a previously completed Stages-1-4 run
without re-running the BO methods. The standalone sbatches hard-fail if
`ANTHROPIC_API_KEY` is not set.

## Files

```
1_serving_runtime_scoot/
├── README.md                                         # this file
├── combined_pipeline/                                # 5-stage orchestrator
│   ├── _bench_runner.py                              # vLLM benchmark runner used by random search
│   ├── random_search.py                              # uniform-sampling baseline
│   ├── results/_make_report.py                       # cross-run aggregator -> report.md
│   └── run_configs/
│       ├── combined_rtx6000ada.sbatch                # example: 2x rtx6000ada (Ada, 48 GiB)
│       ├── combined_rtxa6000.sbatch                  # example: 2x rtxa6000 (Ampere, 48 GiB)
│       ├── combined_rtxa4000.sbatch                  # example: 4x rtxa4000 (Ampere, 16 GiB)
│       └── combined_l40s_preemptible.sbatch          # example: 2x L40S (preemptible)
├── scoot_baseline/                                   # SCOOT (HEBO + EHVI)
│   ├── bo_scoot.py                                   # main BO loop
│   ├── requirements.txt                              # scoot-vllm env (vLLM + HEBO)
│   ├── benchmark_pipeline.sh                         # vLLM benchmark dispatcher
│   ├── clients/                                      # vLLM-derived benchmark client
│   ├── hebo/                                         # HEBO library bundled
│   └── tuner_conf/                                   # min_world_size + max_seq_len autoconf
├── scoot_qnehvi/                                     # SCOOT-qNEHVI
│   ├── bo_scoot_qnehvi.py                            # replaces HEBO+EHVI with BoTorch+qNEHVI
│   ├── requirements.txt                              # scoot-botorch env (vLLM + BoTorch + anthropic)
│   └── scoot_botorch/                                # mixed-space GP + qNEHVI acquisition
├── llm_agent/                                        # LLM-as-optimizer (Stage 5)
│   ├── agent_search.py                               # Sobol-10 + 20 LLM calls
│   ├── agent_program.md                              # system prompt template
│   ├── _bench_runner.py
│   ├── docs/agent-stage.md                           # design + flow diagrams
│   └── run_configs/llm_agent_*.sbatch                # standalone "run only Stage 5"
└── vllm_default/                                     # single-config benchmark wrapper
    ├── scripts/{benchmark,run_server,run_client}_vllm_defaults.sh
    └── run_configs/vllm_default_llama2_sharegpt.sbatch
```

## Submitting a run

```bash
cd /path/to/optimization-llm-inference
source setenv.sh

# Pick the sbatch that names the GPU type your cluster has, e.g.:
sbatch 1_serving_runtime_scoot/combined_pipeline/run_configs/combined_rtxa6000.sbatch
```

The four sbatch variants under `combined_pipeline/run_configs/` differ only
in the `--gres` line and a few resource caps tuned for the GPU type. Pick
whichever matches your hardware, then **edit it** to use your cluster's
`--account` / `--partition` / `--qos` / exact `--gres` syntax (see the
top-level README §7 for guidance, including the `SKIP_GPU_CHECK=1` escape
hatch and the `CONDA_INIT_CMD` override for non-`module`-based conda).

The `combined_l40s_preemptible.sbatch` is an example for clusters that
expose a preemptible queue (sometimes called a back-fill or low-priority
queue). Remove it if your cluster has no such queue.

To run **only** the LLM-Agent stage on its own allocation (e.g. to add
agent results to a Stages-1-4 run that completed earlier without an
ANTHROPIC_API_KEY), use the standalone wrapper:

```bash
sbatch 1_serving_runtime_scoot/llm_agent/run_configs/llm_agent_rtxa6000.sbatch
```

## Quick smoke test (~5 minutes)

```bash
BO_LOOP=2 RANDOM_TRIALS=2 NUM_REQUESTS=50 AGENT_TRIALS=2 \
    sbatch 1_serving_runtime_scoot/combined_pipeline/run_configs/combined_rtxa6000.sbatch
```

If `ANTHROPIC_API_KEY` is not set Stage 5 is skipped; the smoke test then
covers Stages 1-4. To smoke-test Stage 5 too, copy
`env.sh.example -> env.sh` and add a key first.

## Resuming a partial run

All BO methods write `rec_history*.json` after every trial. To re-launch
into the same run dir and skip already-completed trials:

```bash
sbatch --export=ALL,RESUME_DIR=/path/to/run/dir \
    1_serving_runtime_scoot/combined_pipeline/run_configs/combined_rtxa6000.sbatch
```

## Aggregating results into one Markdown table

After one or more runs complete, a small aggregator auto-discovers every
run directory under `combined_pipeline/results/combined/` and emits a
cross-run Markdown report:

```bash
conda activate scoot-botorch
python 1_serving_runtime_scoot/combined_pipeline/results/_make_report.py
# writes 1_serving_runtime_scoot/combined_pipeline/results/report.md
```

The aggregator pulls best-per-objective, Pareto-frontier, and
hypervolume-best (relative to vLLM-default) statistics from each method's
`vllm-*.json` files and produces a per-run section plus a cross-run
ranking. It also pairs combined runs with same-named standalone agent
runs under `llm_agent/results/llm_agent/` so per-allocation reports
include all five methods. Override the discovery roots with `--runs-dir`
and `--agent-runs-dir`; override the output path with `--output`.

## Wall-clock guidance

Per-trial benchmarks take 4-6 minutes on a 2-GPU 48-GiB allocation
(Llama-2-7B-Chat, 1,000 prompts at 5 req/s). The full five-stage pipeline
is therefore approximately:

| Stage | Trials | Per trial | Stage total |
| --- | --- | --- | --- |
| 1. SCOOT (HEBO + EHVI) | 30 | ~5 min + 7 s suggest | ~2.0 h |
| 2. SCOOT-qNEHVI | 30 | ~5 min + 12 s suggest | ~2.2 h |
| 3. vLLM default | 1 | ~4 min | ~4 min |
| 4. Random | 30 | ~5 min | ~2.5 h |
| 5. LLM Agent (skipped if no API key) | 30 (10 Sobol + 20 LLM) | ~5 min | ~2.5 h |
| **Total** | **121** | -- | **~9-10 h** with Stage 5; **~7-8 h** without |

Default sbatch wall-time is `--time=12:00:00`. Raise it if your allocation
is significantly slower than 48-GiB-class GPUs, or if you find Stage 5
pushes you past the wall.

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

## Stage 5 detail: LLM-Agent search

Two-phase: (1) **Sobol warm-start** of $N=10$ quasi-random configs (matches
qNEHVI's `--sobol_init=10` default — closes the init-phase fairness gap
vs qNEHVI). (2) **LLM phase** of `--num_trials - 10` single-shot LLM
calls. Each LLM call sees the full sorted history (Sobol included, marked
as such in the prompt) and proposes one new config. Configs are repaired
through `ScootSearchSpace.repair()` before benchmarking, so out-of-range
LLM proposals are clamped/snapped rather than crashing.

Default LLM is `claude-opus-4-7` with `thinking={"type": "adaptive"}` and
`output_config={"effort": "xhigh"}` (these need `anthropic>=0.96`). Switch
models via the sbatch env: `LLM_MODEL=claude-sonnet-4-6 sbatch ...`. The
prompt template is at `llm_agent/agent_program.md` and the design doc at
`llm_agent/docs/agent-stage.md`.

**Adapting to a different LLM provider.** `agent_search.py` only depends
on `import anthropic` for the LLM call — swap that import + the
`client.messages.create(...)` call for your provider's SDK and adjust
`scoot_qnehvi/requirements.txt` accordingly. The rest of the pipeline
(history schema, repair, benchmark dispatch) is provider-agnostic.

**Cost guidance.** ~$10-15 per 30-trial run with Opus 4.7 + adaptive
thinking + xhigh effort (thinking tokens billed at $25/MTok output rate).
A `claude-sonnet-4-6` run is ~5x cheaper and runs faster. The `--seed 42`
controls only the Sobol init; LLM proposals are not deterministic.
