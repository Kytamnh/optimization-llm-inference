#!/usr/bin/env bash
# Source this file BEFORE submitting any sbatch (or running any python
# entry-point) in this project.
#   $ cd /path/to/optimization-llm-inference
#   $ source setenv.sh
#
# Edit the values below to match YOUR system. See README.md §2-§5 for how to
# create the conda envs, download the models, and download the dataset.

# ---------------------------------------------------------------------------
# 1. Project root  --  computed from this file's location; usually no edits
# ---------------------------------------------------------------------------
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# ---------------------------------------------------------------------------
# 2. Conda environments  --  see top-level README for how to create them
# ---------------------------------------------------------------------------
# Holds vLLM 0.5.5, transformers, torch, plus the SCOOT helpers (HEBO + EHVI).
export SCOOT_VLLM_ENV="${SCOOT_VLLM_ENV:-${HOME}/.conda/envs/scoot-vllm}"

# Holds vLLM 0.5.5 + BoTorch + GPyTorch + PuLP (for SCOOT-qNEHVI and FlexGen).
export SCOOT_BOTORCH_ENV="${SCOOT_BOTORCH_ENV:-${HOME}/.conda/envs/scoot-botorch}"

# ---------------------------------------------------------------------------
# 3. Models  --  see README.md §3 for download instructions
# ---------------------------------------------------------------------------
# Llama-2-7B-Chat-HF (~13 GiB, used in 1_serving_runtime_scoot)
export MODEL_ROOT_7B="${MODEL_ROOT_7B:-${HOME}/models/Llama-2-7b-chat-hf}"

# Llama-2-13B-Chat-HF (~26 GiB, used in 3_cartesian_combination only)
export MODEL_ROOT_13B="${MODEL_ROOT_13B:-${HOME}/models/Llama-2-13b-chat-hf}"

# ---------------------------------------------------------------------------
# 4. Dataset  --  ShareGPT (filtered)
# ---------------------------------------------------------------------------
# Provide a JSON file matching the ShareGPT vicuna-unfiltered schema. The
# benchmark client filters internally, but the file should contain >=1000
# valid conversations. README.md §4 has download + filter instructions.
export DATASET_PATH="${DATASET_PATH:-${HOME}/datasets/sharegpt_llama2_2k_filtered.json}"

# ---------------------------------------------------------------------------
# 5. Caches  --  any large filesystem
# ---------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

# ---------------------------------------------------------------------------
# 6. Slurm log directory  --  created on first submission if missing
# ---------------------------------------------------------------------------
export SLURM_LOG_DIR="${SLURM_LOG_DIR:-${PROJECT_ROOT}/slurm_logs}"
mkdir -p "$SLURM_LOG_DIR"

# ---------------------------------------------------------------------------
# 7. Site-local secrets  --  optional, for the LLM-Agent stage
# ---------------------------------------------------------------------------
# If env.sh exists at the repo root it is sourced automatically (and also
# auto-sourced by combined / llm_agent sbatches). Use it to export
# ANTHROPIC_API_KEY (and HF_TOKEN if you fetch gated models). The file is
# .gitignored. Template at env.sh.example.
if [ -f "${PROJECT_ROOT}/env.sh" ]; then
  set +u; source "${PROJECT_ROOT}/env.sh"; set -u
fi

# ---------------------------------------------------------------------------
# Quick sanity print
# ---------------------------------------------------------------------------
echo "PROJECT_ROOT      = $PROJECT_ROOT"
echo "SCOOT_VLLM_ENV    = $SCOOT_VLLM_ENV"
echo "SCOOT_BOTORCH_ENV = $SCOOT_BOTORCH_ENV"
echo "MODEL_ROOT_7B     = $MODEL_ROOT_7B"
echo "MODEL_ROOT_13B    = $MODEL_ROOT_13B"
echo "DATASET_PATH      = $DATASET_PATH"
echo "HF_HOME           = $HF_HOME"
echo "SLURM_LOG_DIR     = $SLURM_LOG_DIR"
