# Experiment 3 — Cartesian Combination of FlexGen × SCOOT-qNEHVI

A joint analysis of the two complementary methods studied in Experiments 1
and 2: SCOOT-qNEHVI tunes the **vLLM serving runtime** ($\Omega_S$) by
real benchmarking, and FlexGen tunes the **memory topology** ($\Omega_T$)
analytically. Each method produces a Pareto front. We build the
**Cartesian product** of the two fronts and pick a joint best.

> **Prerequisites:** complete the *Setup* section in the
> [top-level README](../README.md) and `source ../setenv.sh`.

## Why a different model from Experiment 1?

Llama-2-7B-fp16 (~13 GiB) fits trivially on 48-GiB cards, so FlexGen's LP
collapses to the all-on-GPU corner ($w_g = c_g = h_g = 1$) at any
reasonable batch and there is no offloading trade-off to exploit. We use
**Llama-2-13B-Chat-HF** (~24 GiB fp16) so the LP makes meaningful
placement decisions while still letting vLLM serve the model on a single
48-GiB GPU or a pair of 24-GiB GPUs.

## What this experiment does

1. Runs **FlexGen** (~30 s analytical) → Pareto front
   $\mathcal{P}_T \subset \Omega_T$ in (per-token latency, block size).
2. Runs **SCOOT-qNEHVI** (~4 h on real GPUs) → Pareto front
   $\mathcal{P}_S \subset \Omega_S$ in (throughput, TTFT, TPOT).
3. Builds the cross-product $\mathcal{P}_S \times \mathcal{P}_T$, scores
   each pair under a normalized 4-objective sum, picks the lowest-score
   pair.
4. Compares the joint best against the vLLM-default measurement.

## Files

```
3_cartesian_combination/
├── README.md                                # this file
├── workload_llama2_sharegpt.yaml            # FlexGen workload (prompt 119, decode 320)
├── precompute_tuner_conf.py                 # mem-cheap conf.json builder
├── flexgen_pareto.py                        # 2D Pareto fronts from a FlexGen result
├── cartesian_best.py                        # cross-product scoring + comparison vs default
├── summarize.py                             # per-run unified report.md
└── run_configs/
    ├── flexgen_qnehvi_rtxa6000.sbatch       # example: 2x rtxa6000 (Ampere, 48 GiB) -- non-preemptible
    └── flexgen_qnehvi_rtxa5000.sbatch       # example: 2x rtxa5000 (Ampere, 24 GiB) -- non-preemptible
```

The two example sbatches cover two interesting VRAM regimes:
- **48 GiB cards (rtxa6000):** Llama-2-13B fits cleanly with KV
  pressure only at large batch; FlexGen LP makes a small placement
  decision.
- **24 GiB cards (rtxa5000):** Llama-2-13B is tight; the LP must trade
  off CPU/disk offloading more aggressively.

## Submitting a run

```bash
cd /path/to/optimization-llm-inference
source setenv.sh

# Pick the sbatch matching your hardware (edit --account / --partition /
# --qos / --gres in the sbatch first; see top-level README §7):
sbatch 3_cartesian_combination/run_configs/flexgen_qnehvi_rtxa6000.sbatch
# or:
sbatch 3_cartesian_combination/run_configs/flexgen_qnehvi_rtxa5000.sbatch
```

If your cluster reports a different GPU name string than the sbatch
expects (e.g. `NVIDIA RTX 6000 Ada Generation` instead of
`NVIDIA RTX A6000`), bypass the guard with `SKIP_GPU_CHECK=1 sbatch ...`.
For non-`module load conda` setups, set `CONDA_INIT_CMD` to your
cluster's conda-init incantation (see top-level README §7).

Each sbatch:
1. Runs **FlexGen** via `../2_memory_topology_flexgen/flexgen_solver/experiments/run_flexgen.py`
   with the SCOOT-aligned `workload_llama2_sharegpt.yaml`.
2. Runs **SCOOT-qNEHVI** (`../1_serving_runtime_scoot/scoot_qnehvi/bo_scoot_qnehvi.py`)
   for $N=30$ trials on Llama-2-13B-Chat.
3. On exit (success **or** failure), runs `summarize.py` to produce
   `report.md`.

After the run completes, build the FlexGen 2D Pareto fronts and the
cross-product table:

```bash
PY=$SCOOT_BOTORCH_ENV/bin/python
RUN=3_cartesian_combination/results/run_<jobid>

$PY 3_cartesian_combination/flexgen_pareto.py --run-root "$RUN"
$PY 3_cartesian_combination/cartesian_best.py  "$RUN"
```

`summarize.py` runs automatically inside the sbatch via an `EXIT` trap, so
you always get a `report.md` even if the qNEHVI stage hits the wall-time.

## Cartesian scoring

`cartesian_best.py` builds all $|\mathcal{P}_S| \times |\mathcal{P}_T|$
pairs. For each pair we form a 4-objective vector
$\big(\widetilde\Phi_F(t),\ -\widetilde T(s),\ \widetilde\Phi_S(s),\ \widetilde\Theta(s)\big)$
of FlexGen predicted latency, qNEHVI throughput (negated), qNEHVI TTFT,
qNEHVI TPOT. Each component is min-max normalized to $[0, 1]$ across the
cross product. The combined score is

$$\mathrm{score}(s, t) = \sum_{k=1}^{4} \alpha_k \cdot \widetilde f_k(s, t)$$

with default uniform weights $\alpha = (1, 1, 1, 1)$. The pair with the
lowest score wins.

You can re-weight to emphasize a specific objective, e.g. 3x weight on
TTFT:

```bash
$PY 3_cartesian_combination/cartesian_best.py "$RUN" --weights 1,1,3,1
```

## Sbatch headers and hardware tweaks

The two example sbatches need the same Slurm-header edits as Experiment 1
(`--account`, `--partition`, `--qos`, exact `--gres` GPU-type string).
Beyond those, the resource caps differ slightly:

| Sbatch | Memory | CPUs | Wall time | Notes |
| --- | --- | --- | --- | --- |
| `flexgen_qnehvi_rtxa6000.sbatch` | 32 GiB | 8 | 6 h | non-preemptible 48-GiB cards |
| `flexgen_qnehvi_rtxa5000.sbatch` | 32 GiB | 4 | 8 h | tighter 24-GiB cards, slower per-trial |

Memory must be at least ~30 GiB for Llama-2-13B serving. `precompute_tuner_conf.py`
avoids the upstream SCOOT helper that would otherwise materialize the full
24 GiB of Llama-2-13B weights in host RAM (see "Why precompute_tuner_conf.py"
below).

## Limitation (also discussed in §7 of the report)

The Cartesian product is an **analytical** Pareto-of-Paretos pick — not a
deployable single configuration. $\Omega_S$ and $\Omega_T$ are knobs of
two different inference engines (vLLM and FlexGen), and there is no
production runtime exposing both. The natural extension is a **nested
optimizer** in which SCOOT-qNEHVI is the outer loop and FlexGen LP is the
inner loop:

$$A^*(B) = \arg\min_{A \in \Omega_T} G(A, B)$$

where $G$ is FlexGen's predicted per-token cost given that vLLM is
configured by $B \in \Omega_S$. Implementing this requires a unified
inference engine that exposes both $\Omega_S$ and $\Omega_T$ knobs --
substantial engineering work because vLLM does not expose memory placement
and FlexGen does not implement PagedAttention.

## Why `precompute_tuner_conf.py`

The upstream SCOOT helper `tuner_conf/tuner_conf.sh` calls
`AutoModelForCausalLM.from_pretrained()` purely to count parameters; on
Llama-2-13B that materializes ~24 GiB of weights in host RAM and overshoots
a tight `--mem` request. `precompute_tuner_conf.py` produces the same
`{max_sequence_length, min_world_size}` JSON by reading
`config.json` + `model.safetensors.index.json` headers only -- no model
load, microseconds of compute. The sbatches in this directory call this
helper instead of the upstream script.

## Wall-clock guidance

| Stage | Per-trial cost | Total |
| --- | --- | --- |
| FlexGen LP search (240 inner LP solves) | ~30 s analytical, no GPU | ~30 s |
| FlexGen calibration (one-time PCIe + TFLOPS bench) | ~10 s on first run | ~10 s |
| SCOOT-qNEHVI on Llama-2-13B (2x 48-GiB GPUs, 1,000 prompts at 5 qps) | ~7-9 min/trial × 30 | ~3.5-4.5 h |
| Cleanup + `summarize.py` | <1 min | <1 min |
| **Total** | -- | **~3.5-4.5 h** |

On 24-GiB cards (rtxa5000-class) per-trial cost rises to ~8-12 min because
KV cache becomes the bottleneck at moderate batch sizes; budget ~5 h for
the qNEHVI stage there.
