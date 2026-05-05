#!/usr/bin/env bash
#SBATCH --job-name=scoot-combined-l40s          # 2x L40S preemptible variant
#SBATCH --account=YOUR_PREEMPTIBLE_ACCOUNT     # TODO: set to your preemptible-queue account, if any
#SBATCH --partition=YOUR_PREEMPTIBLE_PARTITION # TODO: set to your preemptible partition, if any
#SBATCH --qos=YOUR_PREEMPTIBLE_QOS             # TODO
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# --chdir omitted: submit with `cd ${PROJECT_ROOT} && sbatch ...`
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# 2x L40S (Ada, 48 GiB VRAM, datacenter SKU). Targets a preemptible queue.
# PREEMPTIBLE: this job can be killed mid-run when a higher-priority job
# wants the node back. Each stage's `rec_history*.json` is written after
# every trial, so SCOOT/qNEHVI/random will resume on resubmit if you point
# the new job at the same RUN_ROOT (currently it uses ${SLURM_JOB_ID} so a
# resubmit gets a fresh dir; copy/rename old vllm-*.json into the new
# scoot/qnehvi/random subdirs to resume).
#

set -euo pipefail

PACKAGE_ROOT=${PROJECT_ROOT}/1_serving_runtime_scoot/combined_pipeline
ASSET_ROOT=${PROJECT_ROOT}
SCOOT_REPO=${PROJECT_ROOT}/1_serving_runtime_scoot/scoot_baseline
QNEHVI_REPO=${PROJECT_ROOT}/1_serving_runtime_scoot/scoot_qnehvi
VLLM_DEFAULT_PKG=${PROJECT_ROOT}/1_serving_runtime_scoot/vllm_default
MODEL_PATH=${MODEL_ROOT_7B}
DATASET_PATH=${DATASET_PATH}
RUN_ROOT=${PACKAGE_ROOT}/results/combined/l40s_${SLURM_JOB_ID}
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
BO_LOOP=${BO_LOOP:-30}
RANDOM_TRIALS=${RANDOM_TRIALS:-30}
MODEL_NAME=${MODEL_NAME:-llama2_7b_chat_scoot}

echo "REQUEST_RATE=${REQUEST_RATE} NUM_REQUESTS=${NUM_REQUESTS} BO_LOOP=${BO_LOOP} RANDOM_TRIALS=${RANDOM_TRIALS}"
echo "TOTAL_RESOURCE=${TOTAL_RESOURCE}"

export HF_HOME=${HF_HOME}
export NUMBA_CACHE_DIR=/tmp/${USER}/numba-cache-combined-l40s
export OUTLINES_CACHE_DIR=/tmp/${USER}/outlines-cache-combined-l40s
mkdir -p "${HF_HOME}" "${NUMBA_CACHE_DIR}" "${OUTLINES_CACHE_DIR}"

module load conda/202211
source "$(conda info --base)/etc/profile.d/conda.sh"

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
  cp "${PACKAGE_ROOT}/run_configs/combined_l40s_preemptible.sbatch" "${RUN_ROOT}/submitted.sbatch" 2>/dev/null || true
  cp "${PACKAGE_ROOT}/README.md" "${RUN_ROOT}/README.md" 2>/dev/null || true
else
  cp "${PACKAGE_ROOT}/run_configs/combined_l40s_preemptible.sbatch" "${RUN_ROOT}/resume_${SLURM_JOB_ID}.sbatch" 2>/dev/null || true
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

echo
echo "All stages complete: $(date)"
find "${RUN_ROOT}/tune_res" -name 'vllm*.json' -type f 2>/dev/null | sort > "${RUN_ROOT}/benchmark_json_files.txt" || true
find "${RUN_ROOT}/tune_res" -type f 2>/dev/null | sort > "${RUN_ROOT}/result_files.txt" || true
echo "Wrote benchmark file list to ${RUN_ROOT}/benchmark_json_files.txt"
