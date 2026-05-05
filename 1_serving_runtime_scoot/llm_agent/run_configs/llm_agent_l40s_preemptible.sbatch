#!/usr/bin/env bash
#SBATCH --job-name=llm-agent-l40s
#SBATCH --account=YOUR_PREEMPTIBLE_ACCOUNT     # TODO: set to your preemptible-queue account, if any
#SBATCH --partition=YOUR_PREEMPTIBLE_PARTITION # TODO: set to your preemptible partition, if any
#SBATCH --qos=YOUR_PREEMPTIBLE_QOS             # TODO
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
# --chdir omitted: submit with `cd ${PROJECT_ROOT} && sbatch ...`
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# 2x L40S (Ada, 48 GiB VRAM, datacenter SKU). Targets a preemptible queue.
# PREEMPTIBLE: this job can be killed mid-run. agent_search.py saves
# rec_history_agent.json after every trial, so resume via RESUME_DIR works.
# Hardware/workload exactly matches scoot_run/combined/run_configs/combined_l40s_preemptible.sbatch
# so the LLM-agent results are directly comparable to that allocation's
# scoot/qnehvi/default/random results.
#
# Stage: ONLY LLM-Agent (Sobol-10 init + 20 Claude Opus 4.7 calls).
# The script self-resumes from rec_history_agent.json if RUN_ROOT is reused.

set -euo pipefail

PACKAGE_ROOT=${PROJECT_ROOT}/1_serving_runtime_scoot/llm_agent
ASSET_ROOT=${PROJECT_ROOT}
QNEHVI_REPO=${PROJECT_ROOT}/1_serving_runtime_scoot/scoot_qnehvi
MODEL_PATH=${MODEL_ROOT_7B}
DATASET_PATH=${DATASET_PATH}
RUN_ROOT=${PACKAGE_ROOT}/results/llm_agent/l40s_${SLURM_JOB_ID}
QNEHVI_ENV=${SCOOT_BOTORCH_ENV}

# Resume mode: if RESUME_DIR is exported by the caller, reuse that RUN_ROOT
# so agent_search.py auto-skips already-completed trials.
if [ -n "${RESUME_DIR:-}" ]; then
  RUN_ROOT="${RESUME_DIR}"
  echo "RESUME mode: RUN_ROOT=${RUN_ROOT}"
  if [ ! -d "${RUN_ROOT}" ]; then
    echo "ERROR: RESUME_DIR does not exist: ${RUN_ROOT}"
    exit 2
  fi
fi

mkdir -p "${PACKAGE_ROOT}/results/slurm" "${RUN_ROOT}"

# Optional site-local env file (gitignored). Sourced if present so
# ANTHROPIC_API_KEY can stay out of the repo. Template at project root.
if [ -f "${ASSET_ROOT}/env.sh" ]; then
  echo "Sourcing site-local env: ${ASSET_ROOT}/env.sh"
  set +u; source "${ASSET_ROOT}/env.sh"; set -u
fi

exec > >(tee -a "${RUN_ROOT}/driver.out")
exec 2> >(tee -a "${RUN_ROOT}/driver.err" >&2)

echo "Job started: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
hostname
nvidia-smi -L

# Hard requirement: agent must have an API key. Fail fast with a clear
# message rather than silently degrading to random search (which would
# duplicate stage 4 of combined).
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "ERROR: ANTHROPIC_API_KEY is not set."
  echo "Either export it before sbatch:    export ANTHROPIC_API_KEY=sk-ant-..."
  echo "Or copy env.sh.example -> env.sh:  cp ${ASSET_ROOT}/env.sh.example ${ASSET_ROOT}/env.sh"
  exit 2
fi

GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader)
GPU_TARGET_COUNT=$(printf '%s\n' "${GPU_NAMES}" | grep -c '^NVIDIA L40S$' || true)
if [ "${GPU_TARGET_COUNT}" -ne 2 ]; then
  echo "Expected 2 NVIDIA L40S GPUs, but allocated:"
  printf '%s\n' "${GPU_NAMES}"
  exit 2
fi

GPU_LABEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr ' ' '_' | tr -cd '[:alnum:]_')
TOTAL_RESOURCE="2${GPU_LABEL}_full"

REQUEST_RATE=${REQUEST_RATE:-5}
NUM_REQUESTS=${NUM_REQUESTS:-1000}
AGENT_TRIALS=${AGENT_TRIALS:-30}
MODEL_NAME=${MODEL_NAME:-llama2_7b_chat_scoot}
LLM_MODEL=${LLM_MODEL:-claude-opus-4-7}

echo "REQUEST_RATE=${REQUEST_RATE} NUM_REQUESTS=${NUM_REQUESTS} AGENT_TRIALS=${AGENT_TRIALS}"
echo "TOTAL_RESOURCE=${TOTAL_RESOURCE} LLM_MODEL=${LLM_MODEL}"

export HF_HOME=${HF_HOME}
export NUMBA_CACHE_DIR=/tmp/${USER}/numba-cache-llm-agent
export OUTLINES_CACHE_DIR=/tmp/${USER}/outlines-cache-llm-agent
mkdir -p "${HF_HOME}" "${NUMBA_CACHE_DIR}" "${OUTLINES_CACHE_DIR}"

module load conda/202211
source "$(conda info --base)/etc/profile.d/conda.sh"

# Conda activate/deactivate hooks reference unset shell vars (e.g. ZSH_VERSION
# inside ruby_deactivate.sh in the base env), which trip `set -u`.
conda_activate()   { set +u; conda activate "$1";  local rc=$?; set -u; return $rc; }
conda_deactivate() { set +u; conda deactivate;     local rc=$?; set -u; return $rc; }

cleanup_servers() {
  echo "Cleaning up vLLM servers and ports"
  pgrep -f "vllm.entrypoints.api_server" 2>/dev/null | xargs -r kill -9 || true
  for port in 8000 8001 8002 8003; do
    lsof -t -i:${port} 2>/dev/null | xargs -r kill -9 || true
  done
  sleep 5
}

if [ -z "${RESUME_DIR:-}" ]; then
  cp "${PACKAGE_ROOT}/run_configs/llm_agent_l40s_preemptible.sbatch" "${RUN_ROOT}/submitted.sbatch" 2>/dev/null || true
  cp "${PACKAGE_ROOT}/README.md" "${RUN_ROOT}/README.md" 2>/dev/null || true
else
  cp "${PACKAGE_ROOT}/run_configs/llm_agent_l40s_preemptible.sbatch" "${RUN_ROOT}/resume_${SLURM_JOB_ID}.sbatch" 2>/dev/null || true
fi

############################################################
# LLM-Agent search (Sobol-10 init + 20 Claude Opus 4.7 calls)
############################################################
echo
echo "===== LLM-Agent search ====="
date
conda_activate "${QNEHVI_ENV}"
export PATH="${QNEHVI_ENV}/bin:${PATH}"

# Sanity-check that anthropic SDK is installed (the agent fatally exits
# if missing; surface it here with a clearer message before benchmarking
# starts).
if ! python -c "import anthropic" 2>/dev/null; then
  echo "ERROR: anthropic SDK missing in ${QNEHVI_ENV}."
  echo "Install with:  ${QNEHVI_ENV}/bin/pip install 'anthropic>=0.96'"
  exit 2
fi

export SCOOT_RES_DIR=${RUN_ROOT}/tune_res
export SCOOT_RES_DIR_PREFIX=agent
mkdir -p "${SCOOT_RES_DIR}"
cd "${QNEHVI_REPO}"
echo "Refreshing tuner_conf"
( cd tuner_conf && bash tuner_conf.sh "${MODEL_PATH}" )
cp tuner_conf/conf.json "${RUN_ROOT}/tuner_conf.json"

python "${PACKAGE_ROOT}/agent_search.py" \
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
  --program_md "${PACKAGE_ROOT}/agent_program.md"

cleanup_servers
conda_deactivate

echo
echo "Run complete: $(date)"
find "${RUN_ROOT}/tune_res" -name 'vllm*.json' -type f 2>/dev/null | sort > "${RUN_ROOT}/benchmark_json_files.txt" || true
find "${RUN_ROOT}/tune_res" -type f 2>/dev/null | sort > "${RUN_ROOT}/result_files.txt" || true
echo "Wrote benchmark file list to ${RUN_ROOT}/benchmark_json_files.txt"
