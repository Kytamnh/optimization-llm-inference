# Optimization for LLM Inference Configuration

A reproducible study of how to automatically choose efficient runtime
configurations for serving large language models on real hardware. The
problem is multi-objective: serving systems trade off throughput,
time-to-first-token (TTFT), and time-per-output-token (TPOT) against each
other under coupled decisions about batching, tensor-parallelism, KV-cache
layout, scheduling, and CPU/GPU/disk memory placement.

This repository compares **five tuning methods** on the **vLLM serving
runtime** parameter space ($\Omega_S$), **two methods** on the **FlexGen
memory topology** parameter space ($\Omega_T$), and a **Cartesian
combination** of the two methods' Pareto fronts on a larger model.

## Repository layout

```
final/
├── README.md                            ← top-level guide (this file)
├── setenv.sh                            ← environment-variable template, source before submitting
│
├── 1_serving_runtime_scoot/             ← vLLM serving experiments (Ω_S, 5 methods)
│   ├── README.md
│   ├── combined_pipeline/               ← orchestrator that runs all 5 methods sequentially
│   ├── scoot_baseline/                  ← SCOOT (HEBO + EHVI) optimizer
│   ├── scoot_qnehvi/                    ← SCOOT-qNEHVI (BoTorch + qNEHVI) optimizer
│   ├── llm_agent/                       ← LLM-as-optimizer baseline
│   └── vllm_default/                    ← vLLM default-config benchmark
│
├── 2_memory_topology_flexgen/           ← FlexGen experiments (Ω_T, LP vs grid)
│   ├── README.md
│   └── flexgen_solver/                  ← cost-model + LP policy search
│
└── 3_cartesian_combination/             ← Cartesian product of Pareto fronts
    ├── README.md
    ├── flexgen_pareto.py                ← FlexGen 2D Pareto front extractor
    ├── cartesian_best.py                ← cross-product scoring
    ├── precompute_tuner_conf.py         ← memory-cheap conf.json builder
    ├── summarize.py                     ← per-run unified report
    ├── workload_llama2_sharegpt.yaml    ← FlexGen workload aligned to ShareGPT
    └── run_configs/                     ← Slurm sbatch scripts
```

**Each experiment has its own README** with the parameter space, methods,
math, run-time guidance, and result-aggregation steps:
- [1_serving_runtime_scoot/README.md](1_serving_runtime_scoot/README.md) - vLLM serving runtime tuning
- [2_memory_topology_flexgen/README.md](2_memory_topology_flexgen/README.md) - FlexGen offloading topology
- [3_cartesian_combination/README.md](3_cartesian_combination/README.md) - cross-method joint analysis

## What each experiment does

| Subdirectory | Parameter space | Methods compared | Wall-clock per full run |
| --- | --- | --- | --- |
| `1_serving_runtime_scoot/` | $\Omega_S$ - vLLM serving runtime args (9 mixed-type knobs) | vLLM defaults, random search, LLM agent, SCOOT (HEBO+EHVI), SCOOT-qNEHVI | ~9-10 hours (5 methods, $N=30$ trials each) |
| `2_memory_topology_flexgen/` | $\Omega_T$ - FlexGen offloading + batching (8 enum + 9 continuous) | FlexGen LP, 3,375-point grid baseline | seconds (analytical) |
| `3_cartesian_combination/` | Pareto sets from both above | Cartesian-product Pareto-of-Paretos | ~4-5 hours (qNEHVI dominates time; FlexGen is seconds) |

# Setup (one-time, ~30-60 minutes)

## 1. System prerequisites

- Linux + Slurm scheduler + CUDA-capable NVIDIA GPU(s) for $\Omega_S$ and
  $\Omega_T$ experiments. The pipelines target Compute Capability 7.5+
  (e.g., RTX 20-series and newer; Turing, Ampere, Ada, Hopper). FlexGen's
  analytical search runs on any machine with CUDA + Python.
- Conda / Miniconda 23+.
- ~30 GiB free disk for Llama-2-7B-Chat (required for $\Omega_S$).
  Add ~26 GiB if you also plan to run $\Omega_T$ + Cartesian on
  Llama-2-13B-Chat.

## 2. Install Python environments

We use two Conda environments. Both are based on Python 3.10.

```bash
# In a clean shell:
conda create -n scoot-vllm    python=3.10 -y
conda create -n scoot-botorch python=3.10 -y
```

### `scoot-vllm` (used for SCOOT baseline and vLLM-default benchmarking)

```bash
conda activate scoot-vllm
pip install --upgrade pip setuptools wheel
pip install -r 1_serving_runtime_scoot/scoot_baseline/requirements.txt

# Sanity check
python -c "import torch, transformers, vllm; \
           print('torch', torch.__version__); \
           print('vllm', vllm.__version__); \
           print('cuda', torch.cuda.is_available())"
# Expect vllm == 0.5.5 and cuda == True (when on a GPU node).
```

If `vllm==0.5.5` pulls a `torch` / CUDA combination that does not match your
driver, install a matching `torch` build first (see
<https://pytorch.org/get-started/locally/>) and then re-run
`pip install --no-deps -r ... requirements.txt`.

### `scoot-botorch` (used for SCOOT-qNEHVI, random search, LLM agent, FlexGen, all analysis)

`scoot-botorch` is a strict superset of `scoot-vllm` plus three extras:

```bash
conda activate scoot-botorch
pip install --upgrade pip setuptools wheel
pip install -r 1_serving_runtime_scoot/scoot_baseline/requirements.txt
pip install botorch gpytorch pulp

# Sanity check
python -c "import torch, transformers, vllm, botorch, gpytorch, pulp; \
           print('all imports ok')"
```

## 3. Download the model(s)

We use the publicly-mirrored NousResearch checkpoints (no HuggingFace gated
access required). Both come in `safetensors` form, which is what vLLM 0.5.5
loads.

```bash
# Pick any directory with ~30 GiB free for the 7B (and ~26 GiB more for the 13B):
mkdir -p $HOME/models && cd $HOME/models

# Llama-2-7B-Chat (~13 GiB) -- needed for 1_serving_runtime_scoot
huggingface-cli download NousResearch/Llama-2-7b-chat-hf \
    --local-dir Llama-2-7b-chat-hf

# Llama-2-13B-Chat (~26 GiB) -- needed for 3_cartesian_combination
huggingface-cli download NousResearch/Llama-2-13b-chat-hf \
    --local-dir Llama-2-13b-chat-hf
```

The mirrors include both `safetensors` and legacy `pytorch_model-*.bin`
shards. vLLM uses `safetensors` only, so you can free disk by deleting the
`.bin` files:

```bash
cd $HOME/models/Llama-2-7b-chat-hf  && rm -f pytorch_model-*.bin pytorch_model.bin.index.json
cd $HOME/models/Llama-2-13b-chat-hf && rm -f pytorch_model-*.bin pytorch_model.bin.index.json
```

## 4. Download the dataset

The 5-method comparison and the Cartesian-product run consume a 1,000-prompt
slice of the public ShareGPT trace.

```bash
mkdir -p $HOME/datasets && cd $HOME/datasets
huggingface-cli download Aeala/ShareGPT_Vicuna_unfiltered \
    ShareGPT_V3_unfiltered_cleaned_split.json --repo-type dataset \
    --local-dir .
mv ShareGPT_V3_unfiltered_cleaned_split.json sharegpt_llama2_2k_filtered.json
```

The benchmark client filters internally
(prompt $\le 1024$, output $\ge 4$, prompt + output $\le 2048$ tokens), so
any ShareGPT-style JSON with a `conversations: [{from, value}, ...]` schema
and $\ge 1{,}000$ valid entries works.

## 5. Configure paths via `setenv.sh`

`setenv.sh` at the repository root holds environment variables consumed by
every sbatch script. Open it and adjust the defaults to match your setup:

```bash
$EDITOR setenv.sh
```

The variables to set are:

| Variable | What it points at |
| --- | --- |
| `PROJECT_ROOT` | This `final/` directory (auto-detected, usually no edit needed) |
| `SCOOT_VLLM_ENV` | Conda env path for `scoot-vllm` (e.g. `$HOME/.conda/envs/scoot-vllm`) |
| `SCOOT_BOTORCH_ENV` | Conda env path for `scoot-botorch` |
| `MODEL_ROOT_7B` | Local Llama-2-7B-Chat directory |
| `MODEL_ROOT_13B` | Local Llama-2-13B-Chat directory |
| `DATASET_PATH` | Local ShareGPT JSON file |
| `HF_HOME` | Any large directory for HuggingFace cache |
| `SLURM_LOG_DIR` | Where Slurm `--output` / `--error` files land |

After editing, source it once per shell:

```bash
cd /path/to/final
source setenv.sh
```

You should see all variables echoed back. Subsequent `sbatch` invocations
inherit these via `--export=ALL` (Slurm's default).

## 6. Edit Slurm headers in each `*.sbatch`

The example sbatch files use placeholder Slurm queue names. Adjust **all of
the following** to your cluster's naming:

```bash
#SBATCH --account=YOUR_ACCOUNT          # set to your billing/allocation account
#SBATCH --partition=YOUR_PARTITION      # set to a GPU-bearing partition
#SBATCH --qos=high                      # set to a QoS that allows your --gres
#SBATCH --gres=gpu:rtx6000ada:2         # GPU-type:count -- exact name depends on cluster
#SBATCH --cpus-per-task=4               # raise/lower per your queue limits
#SBATCH --mem=32G                       # raise if vLLM OOMs while loading the model
#SBATCH --time=12:00:00                 # raise if your run hits the wall
```

The `--gres` GPU-type string (e.g. `rtx6000ada`, `rtxa6000`, `rtxa4000`,
`l40s`) varies between clusters. Run `sinfo -p YOUR_PARTITION -N -o "%N %t %G"`
or your cluster's equivalent to see the exact GRES tags it advertises, and
update the sbatch line accordingly.

The sbatch files also have an internal guard that checks the GPU type via
`nvidia-smi --query-gpu=name --format=csv,noheader` against an exact string
(e.g. `'NVIDIA RTX A6000'`). If the runtime check fails on first launch with
"Expected …" you'll need to update the `grep -c '^NVIDIA …$'` line to match
the exact name your `nvidia-smi` reports.

## 7. Verify with a 5-minute smoke test

Once setup is complete, override the trial budget to make one full
end-to-end run finish in ~5 minutes:

```bash
cd /path/to/final
source setenv.sh

BO_LOOP=2 RANDOM_TRIALS=2 NUM_REQUESTS=50 \
    sbatch 1_serving_runtime_scoot/combined_pipeline/run_configs/combined_rtx6000ada.sbatch
```

If this produces a directory under
`1_serving_runtime_scoot/combined_pipeline/results/combined/<jobid>/` with a
`tune_res/scoot/.../vllm-*.json` file, the end-to-end stack is functioning.
You can now read each experiment's README and submit a real run.

# Hardware sizing guidance

| Model | Per-GPU VRAM needed | Suggested allocation |
| --- | --- | --- |
| Llama-2-7B-Chat fp16 | $\ge 24$ GiB single-card *or* $\ge 16$ GiB with `tp=2` | 2x 24 GiB GPUs (e.g. RTX A5000) -- minimum that allows `tp=1` configs |
| Llama-2-13B-Chat fp16 | $\ge 32$ GiB single-card *or* $\ge 24$ GiB with `tp=2` | 2x 48 GiB GPUs (e.g. RTX A6000 / L40S / RTX 6000 Ada) for the Cartesian study |

The pipelines also handle smaller GPUs (e.g. 16 GiB RTX A4000) by forcing
$\mathrm{tp} \ge 2$; the search space adapts via the `tuner_conf` step (see
[1_serving_runtime_scoot/README.md](1_serving_runtime_scoot/README.md)).

# Resuming a partially-completed run

All Bayesian-optimization methods write `rec_history*.json` after every
trial. If a job is killed, re-launch it into the same run directory and the
optimizer will skip already-completed trials:

```bash
sbatch --export=ALL,RESUME_DIR=/path/to/run/dir \
    1_serving_runtime_scoot/combined_pipeline/run_configs/combined_rtx6000ada.sbatch
```

This pattern works for every sbatch in the repository.
