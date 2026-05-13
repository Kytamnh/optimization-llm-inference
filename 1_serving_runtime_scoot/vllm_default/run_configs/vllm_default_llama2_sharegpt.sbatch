#!/usr/bin/env bash
#SBATCH --job-name=vllm-default-llama2-sharegpt
#SBATCH --account=YOUR_ACCOUNT       # TODO: set to your Slurm account
#SBATCH --partition=YOUR_PARTITION   # TODO: set to your Slurm partition
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
# --chdir omitted: submit with `cd ${PROJECT_ROOT} && sbatch ...`
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
set -euo pipefail

PACKAGE_ROOT=${PROJECT_ROOT}/1_serving_runtime_scoot/vllm_default
ASSET_ROOT=${PROJECT_ROOT}
BENCH_REPO=${PROJECT_ROOT}/1_serving_runtime_scoot/scoot_baseline
MODEL_PATH=${MODEL_ROOT_7B}
DATASET_PATH=${DATASET_PATH}
RUN_ROOT=${PACKAGE_ROOT}/results/vllm_default_llama2_sharegpt/${SLURM_JOB_ID}
CONDA_ENV=${SCOOT_VLLM_ENV:?Source setup/setenv.sh first}
PYTHON=${CONDA_ENV}/bin/python

mkdir -p "${PACKAGE_ROOT}/results/slurm" "${RUN_ROOT}"

exec > >(tee -a "${RUN_ROOT}/driver.out")
exec 2> >(tee -a "${RUN_ROOT}/driver.err" >&2)

echo "Job started: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "Node list: ${SLURM_JOB_NODELIST}"
echo "Benchmark repo: ${BENCH_REPO}"

module load conda/202211
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PATH="${CONDA_ENV}/bin:${PATH}"

export HF_HOME=${HF_HOME}
export NUMBA_CACHE_DIR=/tmp/${USER}/numba-cache-vllm-default
export OUTLINES_CACHE_DIR=/tmp/${USER}/outlines-cache-vllm-default
export BASELINE_RES_DIR=${RUN_ROOT}/baseline_res
export RES_DIR_PATH=${BASELINE_RES_DIR}/vllm_default
mkdir -p "${HF_HOME}" "${NUMBA_CACHE_DIR}" "${OUTLINES_CACHE_DIR}" "${RES_DIR_PATH}"

echo "Environment snapshot"
hostname
nvidia-smi
GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader)
GPU_A6000_COUNT=$(printf '%s\n' "${GPU_NAMES}" | grep -c '^NVIDIA RTX A6000$' || true)
if [ "${GPU_A6000_COUNT}" -ne 4 ]; then
  echo "Expected 4 NVIDIA RTX A6000 GPUs, but allocated:"
  printf '%s\n' "${GPU_NAMES}"
  echo "Exiting before benchmarking. Resubmit when an A6000 allocation is available."
  exit 2
fi

"${PYTHON}" - <<'PY'
import torch, transformers, vllm, numpy
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"gpu_{i}", props.name, round(props.total_memory / 1024**3, 2), "GiB")
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("numpy", numpy.__version__)
PY

"${PYTHON}" -m pip freeze > "${RUN_ROOT}/pip-freeze.txt"
cp "${PACKAGE_ROOT}/run_configs/vllm_default_llama2_sharegpt.sbatch" "${RUN_ROOT}/submitted.sbatch"
find "${PACKAGE_ROOT}" -maxdepth 3 -type f | sort > "${RUN_ROOT}/vllm_default_file_snapshot.txt"

cd "${BENCH_REPO}"

echo "Starting vLLM built-in-default benchmark"
GPU_LABEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr ' ' '_' | tr -cd '[:alnum:]_')
RESULT_SUBDIR="llama2_7b_chat_scoot_qps5_prompts1000_4${GPU_LABEL}_sharegpt"
export RES_DIR_PATH=${BASELINE_RES_DIR}/vllm_default/${RESULT_SUBDIR}
mkdir -p "${RES_DIR_PATH}"

bash "${PACKAGE_ROOT}/scripts/benchmark_vllm_defaults.sh" \
  "${MODEL_PATH}" \
  "${DATASET_PATH}" \
  5 \
  1000 \
  4 \
  llama2_7b_chat_scoot \
  sharegpt

echo "Result files"
find "${BASELINE_RES_DIR}" -type f | sort > "${RUN_ROOT}/result_files.txt"
find "${BASELINE_RES_DIR}" -name 'vllm*.json' -type f | sort > "${RUN_ROOT}/benchmark_json_files.txt"

echo "Job finished: $(date)"
