#!/usr/bin/env bash
#SBATCH --job-name=scoot-combined-rtxa4000      # 4x rtxa4000 variant
#SBATCH --account=YOUR_ACCOUNT       # TODO: set to your Slurm account
#SBATCH --partition=YOUR_PARTITION   # TODO: set to your Slurm partition
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa4000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# --chdir omitted: submit with `cd ${PROJECT_ROOT} && sbatch ...`
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# 4x rtxa4000 (Ampere, 16 GiB VRAM). Non-preemptible.
# A4000 has only 16 GiB VRAM, so Llama-2-7B fp16 (~13 GiB) requires tp >= 2.
# tuner_conf will compute min_world_size = 2, restricting the SCOOT search
# space to tp in {2, 4} for this run -- different from the 48-GiB cards but
# still a valid 9-parameter optimization with 30 trials per stage.

set -euo pipefail

PACKAGE_ROOT=${PROJECT_ROOT}/1_serving_runtime_scoot/combined_pipeline
ASSET_ROOT=${PROJECT_ROOT}
SCOOT_REPO=${PROJECT_ROOT}/1_serving_runtime_scoot/scoot_baseline
QNEHVI_REPO=${PROJECT_ROOT}/1_serving_runtime_scoot/scoot_qnehvi
VLLM_DEFAULT_PKG=${PROJECT_ROOT}/1_serving_runtime_scoot/vllm_default
LLM_AGENT_PKG=${PROJECT_ROOT}/1_serving_runtime_scoot/llm_agent
MODEL_PATH=${MODEL_ROOT_7B}
DATASET_PATH=${DATASET_PATH}
RUN_ROOT=${PACKAGE_ROOT}/results/combined/rtxa4000_${SLURM_JOB_ID}
SCOOT_ENV=${SCOOT_VLLM_ENV}
QNEHVI_ENV=${SCOOT_BOTORCH_ENV}

# Resume mode: if RESUME_DIR is exported by the caller, reuse that RUN_ROOT
# so the python optimizers (bo_scoot.py, bo_scoot_qnehvi.py, random_search.py)
# auto-skip already-completed trials via their rec_history*.json files.
if [ -n "${RESUME_DIR:-}" ]; then
  RUN_ROOT="${RESUME_DIR}"
  echo "RESUME mode: RUN_ROOT=${RUN_ROOT}"
  if [ ! -d "${RUN_ROOT}" ]; then
    echo "ERROR: RESUME_DIR does not exist: ${RUN_ROOT}"
    exit 2
  fi
fi

mkdir -p "${PACKAGE_ROOT}/results/slurm" "${RUN_ROOT}"

exec > >(tee -a "${RUN_ROOT}/driver.out")
exec 2> >(tee -a "${RUN_ROOT}/driver.err" >&2)

echo "Job started: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
hostname
nvidia-smi -L

GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader)
GPU_TARGET_COUNT=$(printf '%s\n' "${GPU_NAMES}" | grep -c '^NVIDIA RTX A4000$' || true)
EXPECTED_GPU_COUNT=4
# Set SKIP_GPU_CHECK=1 to bypass the exact-name match (cluster GPU strings vary).
if [ "${GPU_TARGET_COUNT}" -ne "${EXPECTED_GPU_COUNT}" ] && [ -z "${SKIP_GPU_CHECK:-}" ]; then
  echo "Expected ${EXPECTED_GPU_COUNT} NVIDIA RTX A4000 GPUs, but allocated:"
  printf '%s\n' "${GPU_NAMES}"
  echo "Set SKIP_GPU_CHECK=1 in the submit env to bypass this check."
  exit 2
fi
if [ "${GPU_TARGET_COUNT}" -ne "${EXPECTED_GPU_COUNT}" ]; then
  GPU_TARGET_COUNT=$(printf '%s\n' "${GPU_NAMES}" | wc -l)
  echo "SKIP_GPU_CHECK set; using ${GPU_TARGET_COUNT} allocated GPU(s)."
fi

GPU_LABEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr ' ' '_' | tr -cd '[:alnum:]_')
TOTAL_RESOURCE="4${GPU_LABEL}_full"

REQUEST_RATE=${REQUEST_RATE:-5}
NUM_REQUESTS=${NUM_REQUESTS:-1000}
BO_LOOP=${BO_LOOP:-30}
RANDOM_TRIALS=${RANDOM_TRIALS:-30}
MODEL_NAME=${MODEL_NAME:-llama2_7b_chat_scoot}

echo "REQUEST_RATE=${REQUEST_RATE} NUM_REQUESTS=${NUM_REQUESTS} BO_LOOP=${BO_LOOP} RANDOM_TRIALS=${RANDOM_TRIALS}"
echo "TOTAL_RESOURCE=${TOTAL_RESOURCE}"

export HF_HOME=${HF_HOME}
export NUMBA_CACHE_DIR=/tmp/${USER}/numba-cache-combined-rtxa4000
export OUTLINES_CACHE_DIR=/tmp/${USER}/outlines-cache-combined-rtxa4000
mkdir -p "${HF_HOME}" "${NUMBA_CACHE_DIR}" "${OUTLINES_CACHE_DIR}"

# Conda init: prefer CONDA_INIT_CMD if set; otherwise try `module load conda`;
# otherwise assume conda is already on PATH.
if [ -n "${CONDA_INIT_CMD:-}" ]; then
  eval "${CONDA_INIT_CMD}"
elif command -v module >/dev/null 2>&1 && module avail conda 2>&1 | grep -q conda; then
  module load conda 2>/dev/null || true
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not on PATH. Either load it before sbatch or set CONDA_INIT_CMD."
  exit 2
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

# Optionally source site-local secrets (ANTHROPIC_API_KEY for Stage 5, etc.).
if [ -f "${ASSET_ROOT}/env.sh" ]; then
  echo "Sourcing site-local env: ${ASSET_ROOT}/env.sh"
  set +u; source "${ASSET_ROOT}/env.sh"; set -u
fi

conda_activate()   { set +u; conda activate "$1";  local rc=$?; set -u; return $rc; }
conda_deactivate() { set +u; conda deactivate;     local rc=$?; set -u; return $rc; }

cleanup_between_stages() {
  echo "Cleaning up vLLM servers and ports between stages"
  pgrep -f "vllm.entrypoints.api_server" 2>/dev/null | xargs -r kill -9 || true
  for port in 8000 8001 8002 8003; do
    lsof -t -i:${port} 2>/dev/null | xargs -r kill -9 || true
  done
  sleep 5
}

if [ -z "${RESUME_DIR:-}" ]; then
  cp "${PACKAGE_ROOT}/run_configs/combined_rtxa4000.sbatch" "${RUN_ROOT}/submitted.sbatch" 2>/dev/null || true
  cp "${PACKAGE_ROOT}/README.md" "${RUN_ROOT}/README.md" 2>/dev/null || true
else
  cp "${PACKAGE_ROOT}/run_configs/combined_rtxa4000.sbatch" "${RUN_ROOT}/resume_${SLURM_JOB_ID}.sbatch" 2>/dev/null || true
fi

############################################################
# Stage 1: Standard SCOOT (HEBO + EHVI)
############################################################
echo
echo "===== Stage 1: Standard SCOOT ====="
date
conda_activate "${SCOOT_ENV}"
export PATH="${SCOOT_ENV}/bin:${PATH}"
export SCOOT_RES_DIR=${RUN_ROOT}/tune_res
export SCOOT_RES_DIR_PREFIX=scoot
mkdir -p "${SCOOT_RES_DIR}"
cd "${SCOOT_REPO}"
echo "Refreshing tuner_conf for SCOOT"
( cd tuner_conf && bash tuner_conf.sh "${MODEL_PATH}" )
cp tuner_conf/conf.json "${RUN_ROOT}/tuner_conf_scoot.json"
python bo_scoot.py \
  --model_path "${MODEL_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --dataset_name sharegpt \
  --model "${MODEL_NAME}" \
  --total_resource "${TOTAL_RESOURCE}" \
  --request_rate "${REQUEST_RATE}" \
  --bo_loop "${BO_LOOP}" \
  --exp_num 1 \
  --num_requests "${NUM_REQUESTS}" \
  --num_obj 3
cleanup_between_stages
conda_deactivate

############################################################
# Stage 2: qNEHVI (BoTorch mixed-space GP)
############################################################
echo
echo "===== Stage 2: qNEHVI ====="
date
conda_activate "${QNEHVI_ENV}"
export PATH="${QNEHVI_ENV}/bin:${PATH}"
export SCOOT_RES_DIR=${RUN_ROOT}/tune_res
export SCOOT_RES_DIR_PREFIX=qnehvi
cd "${QNEHVI_REPO}"
echo "Refreshing tuner_conf for qNEHVI"
( cd tuner_conf && bash tuner_conf.sh "${MODEL_PATH}" )
cp tuner_conf/conf.json "${RUN_ROOT}/tuner_conf_qnehvi.json"
python bo_scoot_qnehvi.py \
  --model_path "${MODEL_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --dataset_name sharegpt \
  --model "${MODEL_NAME}" \
  --total_resource "${TOTAL_RESOURCE}" \
  --request_rate "${REQUEST_RATE}" \
  --bo_loop "${BO_LOOP}" \
  --exp_num 1 \
  --num_requests "${NUM_REQUESTS}" \
  --num_obj 3
cleanup_between_stages
conda_deactivate

############################################################
# Stage 3: vLLM default (single benchmark, no tuning args)
############################################################
echo
echo "===== Stage 3: vLLM default ====="
date
conda_activate "${SCOOT_ENV}"
export PATH="${SCOOT_ENV}/bin:${PATH}"
DEFAULT_RES_DIR=${RUN_ROOT}/tune_res/default
DEFAULT_RESULT_SUBDIR="${MODEL_NAME}_qps${REQUEST_RATE}_prompts${NUM_REQUESTS}_${TOTAL_RESOURCE}_sharegpt"
export RES_DIR_PATH=${DEFAULT_RES_DIR}/${DEFAULT_RESULT_SUBDIR}
mkdir -p "${RES_DIR_PATH}"
DEFAULT_VLLM_COUNT=$(find "${RES_DIR_PATH}" -name 'vllm-*.json' -type f 2>/dev/null | wc -l)
if [ "${DEFAULT_VLLM_COUNT}" -ge 1 ]; then
  echo "Stage 3: ${DEFAULT_VLLM_COUNT} default vllm-*.json already in ${RES_DIR_PATH}, skipping"
else
  cd "${SCOOT_REPO}"
  # benchmark_vllm_defaults.sh has a `trap cleanup EXIT` whose lsof|xargs
  # returns non-zero when ports are already empty, which under pipefail
  # bubbles up as a bogus exit-1. Absorb it so stage 4 still runs.
  bash "${VLLM_DEFAULT_PKG}/scripts/benchmark_vllm_defaults.sh" \
    "${MODEL_PATH}" \
    "${DATASET_PATH}" \
    "${REQUEST_RATE}" \
    "${NUM_REQUESTS}" \
    "${GPU_TARGET_COUNT}" \
    "${MODEL_NAME}" \
    sharegpt \
    || echo "Stage 3 returned $?, continuing to stage 4"
fi
unset RES_DIR_PATH
cleanup_between_stages
conda_deactivate

############################################################
# Stage 4: Random search over the same 9-parameter space
############################################################
echo
echo "===== Stage 4: Random search ====="
date
conda_activate "${QNEHVI_ENV}"
export PATH="${QNEHVI_ENV}/bin:${PATH}"
export SCOOT_RES_DIR=${RUN_ROOT}/tune_res
export SCOOT_RES_DIR_PREFIX=random
cd "${QNEHVI_REPO}"
python "${PACKAGE_ROOT}/random_search.py" \
  --model_path "${MODEL_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --dataset_name sharegpt \
  --model "${MODEL_NAME}" \
  --total_resource "${TOTAL_RESOURCE}" \
  --request_rate "${REQUEST_RATE}" \
  --num_requests "${NUM_REQUESTS}" \
  --num_trials "${RANDOM_TRIALS}" \
  --seed 42
cleanup_between_stages
conda_deactivate

############################################################
# Stage 5: LLM-Agent (Sobol-10 + 20 LLM calls). Skipped if
# ANTHROPIC_API_KEY is not set (in env.sh or the submit env).
############################################################
echo
echo "===== Stage 5: LLM-Agent ====="
date
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "Skipping Stage 5: ANTHROPIC_API_KEY not set."
  echo "  Enable: copy env.sh.example to env.sh and add ANTHROPIC_API_KEY=sk-ant-..."
  echo "  See top-level README §6 for full instructions."
else
  AGENT_TRIALS=${AGENT_TRIALS:-${BO_LOOP}}
  LLM_MODEL=${LLM_MODEL:-claude-opus-4-7}
  conda_activate "${QNEHVI_ENV}"
  export PATH="${QNEHVI_ENV}/bin:${PATH}"
  if ! python -c "import anthropic" 2>/dev/null; then
    echo "ERROR: anthropic SDK missing in ${QNEHVI_ENV}."
    echo "Install with: ${QNEHVI_ENV}/bin/pip install 'anthropic>=0.96'"
    conda_deactivate
  else
    export SCOOT_RES_DIR=${RUN_ROOT}/tune_res
    export SCOOT_RES_DIR_PREFIX=agent
    cd "${QNEHVI_REPO}"
    python "${LLM_AGENT_PKG}/agent_search.py" \
      --model_path "${MODEL_PATH}" \
      --dataset_path "${DATASET_PATH}" \
      --dataset_name sharegpt \
      --model "${MODEL_NAME}" \
      --total_resource "${TOTAL_RESOURCE}" \
      --request_rate "${REQUEST_RATE}" \
      --num_requests "${NUM_REQUESTS}" \
      --num_trials "${AGENT_TRIALS}" \
      --seed 42 \
      --llm_model "${LLM_MODEL}" \
      --program_md "${LLM_AGENT_PKG}/agent_program.md" \
      || echo "Stage 5 returned $?, continuing"
    cleanup_between_stages
    conda_deactivate
  fi
fi

echo
echo "All stages complete: $(date)"
find "${RUN_ROOT}/tune_res" -name 'vllm*.json' -type f 2>/dev/null | sort > "${RUN_ROOT}/benchmark_json_files.txt" || true
find "${RUN_ROOT}/tune_res" -type f 2>/dev/null | sort > "${RUN_ROOT}/result_files.txt" || true
echo "Wrote benchmark file list to ${RUN_ROOT}/benchmark_json_files.txt"
