#!/usr/bin/env bash
#SBATCH --job-name=flexgen-qnehvi-rtxa5000
#SBATCH --account=YOUR_ACCOUNT       # TODO: set to your Slurm account
#SBATCH --partition=YOUR_PARTITION   # TODO: set to your Slurm partition
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa5000:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
# --chdir omitted: submit with `cd ${PROJECT_ROOT} && sbatch ...`
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# Sequentially runs:
#   Stage 1: qNEHVI (vLLM serving optimization, 30 trials, ~4-5 h on rtxa5000)
#   Stage 2: FlexGen (cost-model + LP placement search, ~30 s)
# on Llama-2-13B-Chat using ShareGPT for qNEHVI and a SCOOT-aligned synthetic
# workload (prompt_len=119, decode_len=320 mean tokens) for FlexGen.
#
# Target: any 2x rtxa5000 (24 GiB) allocation -- non-preemptible.
# rtxa5000 = 24 GiB VRAM each, so Llama-2-13B fp16 (~24 GiB) requires tp >= 2.
# tuner_conf will compute min_world_size=2, restricting tp to {2}.
# Distinct from any combined/ allocation: combined uses rtxa6000/rtx6000ada/
# rtxa4000/l40s; this sbatch is the only rtxa5000 path in the project.
# Memory request kept tight (16 G) so the job fits in mem-pressured nodes.

set -euo pipefail

PACKAGE_ROOT=${PROJECT_ROOT}/3_cartesian_combination
ASSET_ROOT=${PROJECT_ROOT}
QNEHVI_REPO=${PROJECT_ROOT}/1_serving_runtime_scoot/scoot_qnehvi
FLEXGEN_REPO=${PROJECT_ROOT}/2_memory_topology_flexgen/flexgen_solver
MODEL_PATH=${MODEL_ROOT_13B}
DATASET_PATH=${DATASET_PATH}
WORKLOAD_PATH=${PACKAGE_ROOT}/workload_llama2_sharegpt.yaml
RUN_ROOT=${PACKAGE_ROOT}/results/rtxa5000_${SLURM_JOB_ID}
ENV_PATH=${SCOOT_BOTORCH_ENV}

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
GPU_TARGET_COUNT=$(printf '%s\n' "${GPU_NAMES}" | grep -c '^NVIDIA RTX A5000$' || true)
# Set SKIP_GPU_CHECK=1 to bypass the exact-name match (cluster GPU strings vary).
if [ "${GPU_TARGET_COUNT}" -ne 2 ] && [ -z "${SKIP_GPU_CHECK:-}" ]; then
  echo "Expected 2 NVIDIA RTX A5000 GPUs, but allocated:"
  printf '%s\n' "${GPU_NAMES}"
  echo "Set SKIP_GPU_CHECK=1 in the submit env to bypass this check."
  exit 2
fi
if [ "${GPU_TARGET_COUNT}" -ne 2 ]; then
  GPU_TARGET_COUNT=$(printf '%s\n' "${GPU_NAMES}" | wc -l)
  echo "SKIP_GPU_CHECK set; using ${GPU_TARGET_COUNT} allocated GPU(s)."
fi

GPU_LABEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr ' ' '_' | tr -cd '[:alnum:]_')
TOTAL_RESOURCE="2${GPU_LABEL}_full"

REQUEST_RATE=${REQUEST_RATE:-5}
NUM_REQUESTS=${NUM_REQUESTS:-1000}
BO_LOOP=${BO_LOOP:-30}
MODEL_NAME=${MODEL_NAME:-llama2_13b_chat_scoot}

echo "REQUEST_RATE=${REQUEST_RATE} NUM_REQUESTS=${NUM_REQUESTS} BO_LOOP=${BO_LOOP}"
echo "TOTAL_RESOURCE=${TOTAL_RESOURCE}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "WORKLOAD_PATH=${WORKLOAD_PATH}"

export HF_HOME=${HF_HOME}
export NUMBA_CACHE_DIR=/tmp/${USER}/numba-cache-flexgen-qnehvi-rtxa5000
export OUTLINES_CACHE_DIR=/tmp/${USER}/outlines-cache-flexgen-qnehvi-rtxa5000
mkdir -p "${HF_HOME}" "${NUMBA_CACHE_DIR}" "${OUTLINES_CACHE_DIR}"

# Pick a TCP base port from the JOB_ID hash so concurrent jobs on the same
# multi-tenant node don't collide on the default 8000.
# Range [10000, 40000); two consecutive ports needed for tp=1 + 2 replicas.
export SCOOT_BASE_PORT=$((10000 + (SLURM_JOB_ID % 30000)))
echo "SCOOT_BASE_PORT=${SCOOT_BASE_PORT}"

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
  cp "${PACKAGE_ROOT}/run_configs/flexgen_qnehvi_rtxa5000.sbatch" "${RUN_ROOT}/submitted.sbatch" 2>/dev/null || true
  cp "${PACKAGE_ROOT}/workload_llama2_sharegpt.yaml" "${RUN_ROOT}/workload.yaml" 2>/dev/null || true
else
  cp "${PACKAGE_ROOT}/run_configs/flexgen_qnehvi_rtxa5000.sbatch" "${RUN_ROOT}/resume_${SLURM_JOB_ID}.sbatch" 2>/dev/null || true
fi

conda_activate "${ENV_PATH}"
export PATH="${ENV_PATH}/bin:${PATH}"

# Always regenerate the unified report on exit so partial runs (e.g. qNEHVI
# wall-clock kill) still produce a report.md with whatever results exist.
on_exit() {
  local rc=$?
  echo
  echo "===== on_exit (rc=${rc}): building unified report ====="
  "${ENV_PATH}/bin/python" "${PACKAGE_ROOT}/summarize.py" "${RUN_ROOT}" || echo "summarize returned $?"
  find "${RUN_ROOT}" -type f -name '*.json' 2>/dev/null | sort > "${RUN_ROOT}/result_files.txt" || true
  echo "Job ended: $(date)"
}
trap on_exit EXIT

############################################################
# Stage 1: FlexGen (cost-model + LP, fast — runs first so its
# results are durable even if qNEHVI later fails or hits walltime)
############################################################
echo
echo "===== Stage 1: FlexGen ====="
date
FLEXGEN_OUT=${RUN_ROOT}/flexgen
mkdir -p "${FLEXGEN_OUT}"
cd "${FLEXGEN_REPO}"
PYTHONPATH=. python experiments/run_flexgen.py \
  --model "${MODEL_PATH}" \
  --workload "${WORKLOAD_PATH}" \
  --output-dir "${FLEXGEN_OUT}" \
  --log-dir "${FLEXGEN_OUT}" \
  --cache-dir "${FLEXGEN_OUT}" \
  --recalibrate \
  || echo "FlexGen returned $?, continuing to stage 2"

############################################################
# Stage 2: qNEHVI (BoTorch mixed-space GP, vLLM serving)
############################################################
echo
echo "===== Stage 2: qNEHVI ====="
date
export SCOOT_RES_DIR=${RUN_ROOT}
export SCOOT_RES_DIR_PREFIX=qnehvi
mkdir -p "${SCOOT_RES_DIR}"
cd "${QNEHVI_REPO}"
# Skip the upstream tuner_conf.sh because its calc_min_world_size() loads the
# entire model via AutoModelForCausalLM.from_pretrained, which OOMs on
# Llama-2-13B with our small mem allocation. Use the lightweight helper that
# reads safetensors index metadata and nvidia-smi instead.
echo "Pre-computing tuner_conf/conf.json without loading model weights"
"${ENV_PATH}/bin/python" "${PACKAGE_ROOT}/precompute_tuner_conf.py" \
  "${MODEL_PATH}" "tuner_conf/conf.json"
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
  --num_obj 3 \
  || echo "qNEHVI returned $?, on_exit trap will still summarize"
cleanup_between_stages
conda_deactivate
