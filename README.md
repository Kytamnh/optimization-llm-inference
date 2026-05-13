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
optimization-llm-inference/
├── README.md                            ← top-level guide (this file)
├── setenv.sh                            ← environment-variable template, source before submitting
├── env.sh.example                       ← template for site-local secrets (ANTHROPIC_API_KEY)
│
├── 1_serving_runtime_scoot/             ← vLLM serving experiments (Ω_S, 5 methods)
│   ├── README.md
│   ├── combined_pipeline/               ← 5-stage orchestrator (SCOOT, qNEHVI, default, random, agent)
│   ├── scoot_baseline/                  ← SCOOT (HEBO + EHVI) optimizer + requirements.txt
│   ├── scoot_qnehvi/                    ← SCOOT-qNEHVI (BoTorch + qNEHVI) optimizer + requirements.txt
│   ├── llm_agent/                       ← LLM-as-optimizer baseline (Sobol-10 + Claude calls)
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
- [1_serving_runtime_scoot/README.md](1_serving_runtime_scoot/README.md) - vLLM serving runtime tuning (5 methods, Ω_S)
- [2_memory_topology_flexgen/README.md](2_memory_topology_flexgen/README.md) - FlexGen offloading topology (Ω_T)
- [3_cartesian_combination/README.md](3_cartesian_combination/README.md) - cross-method joint analysis

## What each experiment does

| Subdirectory | Parameter space | Methods compared | Wall-clock per full run |
| --- | --- | --- | --- |
| `1_serving_runtime_scoot/` | $\Omega_S$ - vLLM serving runtime args (9 mixed-type knobs) | vLLM defaults, random search, LLM agent, SCOOT (HEBO+EHVI), SCOOT-qNEHVI | ~9-10 hours (5 methods, $N=30$ trials each); ~7-8 h if you skip the LLM-agent stage |
| `2_memory_topology_flexgen/` | $\Omega_T$ - FlexGen offloading + batching (8 enum + 9 continuous) | FlexGen LP, 3,375-point grid baseline | seconds (analytical) |
| `3_cartesian_combination/` | Pareto sets from both above | Cartesian-product Pareto-of-Paretos | ~4-5 hours (qNEHVI dominates time; FlexGen is seconds) |

# Setup (one-time, ~30-60 minutes)

## 1. System prerequisites

You provide the hardware and the cluster scheduler. The pipeline runs
unmodified on any system that satisfies:

- **OS**: Linux (RHEL/Ubuntu/Debian, etc.). macOS works for FlexGen
  analytical search only.
- **Job scheduler**: Slurm — every full-experiment sbatch in this repo is
  a Slurm script. If you don't have Slurm, see
  [§8 Running without Slurm](#8-running-without-slurm) for the equivalent
  shell-only invocations.
- **GPU**: CUDA-capable NVIDIA GPU(s) with Compute Capability 7.5+
  (Turing / Ampere / Ada / Hopper, e.g. RTX 20-series and newer).
  FlexGen's analytical search (Experiment 2) runs on any machine with
  Python — no GPU required.
- **Conda / Miniconda 23+**. Install from
  <https://docs.conda.io/projects/miniconda/en/latest/> if needed.
- **Disk**: ~30 GiB free for Llama-2-7B-Chat (Experiment 1). Add ~26 GiB
  for Llama-2-13B-Chat if you plan to run Experiment 3.
- **Network**: reachable HuggingFace Hub for the **first** model/dataset
  download; subsequent runs use `$HF_HOME` cache.
- **(Optional) Anthropic API key** for the LLM-Agent stage (Stage 5 of
  Experiment 1, ~$10-15 per 30-trial run with Claude Opus 4.7 + adaptive
  thinking at xhigh effort). Stage 5 is **silently skipped** if no key is
  set; the other four methods still produce results.

VRAM sizing per GPU type:

| Model | Per-GPU VRAM needed | Suggested allocation |
| --- | --- | --- |
| Llama-2-7B-Chat fp16 | $\ge 24$ GiB single-card *or* $\ge 16$ GiB with `tp=2` | 2x 24 GiB GPUs (e.g. RTX A5000) -- minimum that allows `tp=1` configs |
| Llama-2-13B-Chat fp16 | $\ge 32$ GiB single-card *or* $\ge 24$ GiB with `tp=2` | 2x 48 GiB GPUs (e.g. RTX A6000 / L40S / RTX 6000 Ada) for the Cartesian study |

The pipelines also handle smaller GPUs (e.g. 16 GiB RTX A4000) by forcing
$\mathrm{tp} \ge 2$; the search space adapts via the `tuner_conf` step
(see [1_serving_runtime_scoot/README.md](1_serving_runtime_scoot/README.md)).

## 2. Install Python environments

We use two Conda environments based on Python 3.10. Each has a dedicated
`requirements.txt`.

```bash
# In a clean shell:
conda create -n scoot-vllm    python=3.10 -y
conda create -n scoot-botorch python=3.10 -y
```

### `scoot-vllm` (Stage 1 SCOOT + Stage 3 vLLM-default)

```bash
conda activate scoot-vllm
pip install --upgrade pip setuptools wheel
pip install -r 1_serving_runtime_scoot/scoot_baseline/requirements.txt

# Sanity check (run on a CUDA host for cuda==True):
python -c "import torch, transformers, vllm; \
           print('torch', torch.__version__); \
           print('vllm', vllm.__version__); \
           print('cuda', torch.cuda.is_available())"
# Expect vllm == 0.5.5.
```

If `vllm==0.5.5` pulls a `torch` / CUDA combination that does not match your
driver, install a matching `torch` build first (see
<https://pytorch.org/get-started/locally/>) and then re-run
`pip install --no-deps -r ... requirements.txt`.

### `scoot-botorch` (Stage 2 qNEHVI, Stage 4 random, Stage 5 LLM agent, FlexGen, all analysis)

`scoot-botorch` is a strict superset of `scoot-vllm` plus BoTorch / GPyTorch
/ PuLP / the Anthropic SDK (for Stage 5). Use the dedicated requirements
file under `scoot_qnehvi/`:

```bash
conda activate scoot-botorch
pip install --upgrade pip setuptools wheel
pip install -r 1_serving_runtime_scoot/scoot_qnehvi/requirements.txt

# Sanity check
python -c "import torch, transformers, vllm, botorch, gpytorch, pulp, anthropic; \
           print('all imports ok')"
```

If you do not plan to run Stage 5 (the LLM agent) you can omit `anthropic`
by editing `1_serving_runtime_scoot/scoot_qnehvi/requirements.txt` before
the `pip install` — the rest of the pipeline does not need it. The combined
sbatches detect a missing `ANTHROPIC_API_KEY` and skip Stage 5
automatically.

## 3. Download the model(s)

We use the publicly-mirrored NousResearch checkpoints (no HuggingFace gated
access required). Both come in `safetensors` form, which is what vLLM 0.5.5
loads. The original SCOOT paper used `meta-llama/Llama-2-7b-chat-hf`; the
NousResearch mirror is bit-for-bit equivalent.

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

For exact reproducibility you can pin a HuggingFace commit:
`huggingface-cli download <repo> --revision <sha> --local-dir ...`. Find
SHAs at the model page on huggingface.co.

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
| `PROJECT_ROOT` | This repo's root (auto-detected, usually no edit needed) |
| `SCOOT_VLLM_ENV` | Conda env path for `scoot-vllm` (e.g. `$HOME/.conda/envs/scoot-vllm`) |
| `SCOOT_BOTORCH_ENV` | Conda env path for `scoot-botorch` |
| `MODEL_ROOT_7B` | Local Llama-2-7B-Chat directory |
| `MODEL_ROOT_13B` | Local Llama-2-13B-Chat directory |
| `DATASET_PATH` | Local ShareGPT JSON file |
| `HF_HOME` | Any large directory for HuggingFace cache |
| `SLURM_LOG_DIR` | Where Slurm `--output` / `--error` files land |

After editing, source it once per shell:

```bash
cd /path/to/optimization-llm-inference
source setenv.sh
```

You should see all variables echoed back. Subsequent `sbatch` invocations
inherit these via `--export=ALL` (Slurm's default). `setenv.sh` also
auto-sources `env.sh` if you create one (see §6).

## 6. (Optional) Site-local secrets via `env.sh`

The LLM-Agent stage (Stage 5 of Experiment 1) calls the Anthropic API and
needs `ANTHROPIC_API_KEY` exported. The recommended way is a gitignored
`env.sh` at the repo root that the sbatches auto-source:

```bash
cp env.sh.example env.sh
chmod 600 env.sh                # readable only by you
$EDITOR env.sh                  # paste sk-ant-... into ANTHROPIC_API_KEY
```

Get your key at <https://console.anthropic.com/>. Cost guidance for one
30-trial Stage 5 run with Claude Opus 4.7 + adaptive thinking + xhigh
effort is ~$10-15 (thinking tokens are billed at the $25/MTok output
rate). Switch to a cheaper model with
`LLM_MODEL=claude-sonnet-4-6 sbatch ...` if you want to spend less.

If `env.sh` does not exist or `ANTHROPIC_API_KEY` is empty, **the combined
sbatch silently skips Stage 5** and Stages 1-4 still produce results. The
standalone `llm_agent/run_configs/*.sbatch` scripts hard-fail with a clear
error if the key is missing, since their only job is the agent.

The same `env.sh` is also a convenient place for `HF_TOKEN` if you ever
swap NousResearch/* for the gated meta-llama/* originals. The file is
listed in `.gitignore` and never committed.

## 7. Edit Slurm headers in each `*.sbatch`

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

Each sbatch also has an internal guard that checks the GPU type via
`nvidia-smi --query-gpu=name --format=csv,noheader` against an exact string
(e.g. `'NVIDIA RTX A6000'`). If your cluster reports a different name,
**either** edit the `grep -c '^NVIDIA …$'` line to match what
`nvidia-smi --query-gpu=name --format=csv,noheader` prints on your node,
**or** bypass the guard entirely with `SKIP_GPU_CHECK=1`:

```bash
SKIP_GPU_CHECK=1 sbatch 1_serving_runtime_scoot/combined_pipeline/run_configs/combined_rtxa6000.sbatch
```

`SKIP_GPU_CHECK=1` trusts whatever GPUs Slurm allocated and uses
`nvidia-smi`'s reported count as `GPU_TARGET_COUNT`.

### Conda activation in the sbatch

The sbatches all need `conda` on the PATH inside the job. They try in
order:

1. If you set `CONDA_INIT_CMD` in the submit env, they `eval` that
   (e.g. `CONDA_INIT_CMD='module load miniconda3'` or
   `CONDA_INIT_CMD='source ~/miniconda3/etc/profile.d/conda.sh'`).
2. Otherwise they try `module load conda` if `module` is available.
3. Otherwise they assume `conda` is already on PATH (Miniconda installed
   in `$HOME` etc.).

If none of these work, the sbatch exits with a clear error message —
adjust your `CONDA_INIT_CMD` accordingly.

## 8. Running without Slurm

If you don't have a Slurm scheduler, you can still run every method
manually. The sbatches are thin wrappers around the python entry points;
all the env vars they set up are documented in
[`1_serving_runtime_scoot/README.md`](1_serving_runtime_scoot/README.md).
The minimal recipe per method is:

```bash
cd /path/to/optimization-llm-inference
source setenv.sh

# Stage 1: SCOOT (HEBO + EHVI)
conda activate scoot-vllm
cd 1_serving_runtime_scoot/scoot_baseline
( cd tuner_conf && bash tuner_conf.sh "$MODEL_ROOT_7B" )
SCOOT_RES_DIR=/tmp/run_scoot SCOOT_RES_DIR_PREFIX=scoot \
  python bo_scoot.py --model_path "$MODEL_ROOT_7B" --dataset_path "$DATASET_PATH" \
  --dataset_name sharegpt --model llama2_7b_chat_scoot --total_resource 2GPUs_full \
  --request_rate 5 --bo_loop 30 --exp_num 1 --num_requests 1000 --num_obj 3

# Stage 2: qNEHVI
conda activate scoot-botorch
cd ../scoot_qnehvi
( cd tuner_conf && bash tuner_conf.sh "$MODEL_ROOT_7B" )
SCOOT_RES_DIR=/tmp/run_qnehvi SCOOT_RES_DIR_PREFIX=qnehvi \
  python bo_scoot_qnehvi.py [same args]

# Stage 3: vLLM default benchmark
cd ../vllm_default
bash scripts/benchmark_vllm_defaults.sh "$MODEL_ROOT_7B" "$DATASET_PATH" 5 1000 2 llama2_7b_chat_scoot sharegpt

# Stage 4: random search
cd ../scoot_qnehvi   # script imports from this dir
SCOOT_RES_DIR=/tmp/run_random SCOOT_RES_DIR_PREFIX=random \
  python ../combined_pipeline/random_search.py [same args + --num_trials 30 --seed 42]

# Stage 5: LLM agent
export ANTHROPIC_API_KEY=sk-ant-...
SCOOT_RES_DIR=/tmp/run_agent SCOOT_RES_DIR_PREFIX=agent \
  python ../llm_agent/agent_search.py [same args + --num_trials 30 --seed 42 \
                                       --llm_model claude-opus-4-7 \
                                       --program_md ../llm_agent/agent_program.md]
```

`TOTAL_RESOURCE` is a free-form label written into result paths; pick
something descriptive for your hardware (e.g. `2A6000_full`).
`GPU_TARGET_COUNT` for Stage 3 should equal the number of GPUs vLLM will
use for tensor parallelism.

## 9. Verify with a 5-minute smoke test

Once setup is complete, override the trial budget to make one full
end-to-end run finish in ~5 minutes:

```bash
cd /path/to/optimization-llm-inference
source setenv.sh

BO_LOOP=2 RANDOM_TRIALS=2 NUM_REQUESTS=50 \
    sbatch 1_serving_runtime_scoot/combined_pipeline/run_configs/combined_rtx6000ada.sbatch
```

If this produces a directory under
`1_serving_runtime_scoot/combined_pipeline/results/combined/<jobid>/` with a
`tune_res/scoot/.../vllm-*.json` file, the end-to-end stack is functioning.
You can now read each experiment's README and submit a real run.

For the FlexGen experiment (Experiment 2) the smoke test runs in seconds
on any host with `scoot-botorch` and Python:

```bash
conda activate scoot-botorch
cd 2_memory_topology_flexgen/flexgen_solver
python experiments/compare_lp_vs_grid.py \
    --model HuggingFaceTB/SmolLM2-135M-Instruct --sim-gpu-gb 0.3 --run-both
```

You should see `Best policy: gbs=… num_gb=…` and a comparison summary.
On a host without CUDA you'll see a "no CUDA detected" warning and the
latency numbers won't reflect any GPU — useful only as a plumbing check.

# Aggregating results across runs

After one or more combined runs complete, aggregate every method's per-trial
JSONs into one Markdown report:

```bash
conda activate scoot-botorch
python 1_serving_runtime_scoot/combined_pipeline/results/_make_report.py
# writes 1_serving_runtime_scoot/combined_pipeline/results/report.md
```

The aggregator auto-discovers any run directory under
`1_serving_runtime_scoot/combined_pipeline/results/combined/` and pairs it
(by name) with any matching standalone agent run under
`1_serving_runtime_scoot/llm_agent/results/llm_agent/`. The report contains
per-run per-method best-per-objective configs, Pareto frontiers, and a
cross-run ranking. Pass `--runs-dir` / `--agent-runs-dir` / `--output` to
override defaults.

# Resuming a partially-completed run

All Bayesian-optimization methods write `rec_history*.json` after every
trial. If a job is killed, re-launch it into the same run directory and the
optimizer will skip already-completed trials:

```bash
sbatch --export=ALL,RESUME_DIR=/path/to/run/dir \
    1_serving_runtime_scoot/combined_pipeline/run_configs/combined_rtx6000ada.sbatch
```

This pattern works for every sbatch in the repository.
