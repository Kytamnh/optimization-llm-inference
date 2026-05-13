#!/usr/bin/env bash
set -euo pipefail

model_path=$1
dataset_path=$2
request_rate=$3
num_requests=$4
gpu_nums=$5
model=$6
dataset_name=$7

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ports=()
for ((i=0; i<gpu_nums; i++)); do
    ports+=("$((8000 + i))")
done

port_csv="$(IFS=,; echo "${ports[*]}")"

cleanup() {
    pgrep -f "vllm.entrypoints.api_server" | xargs -r kill -9
    for port in "${ports[@]}"; do
        echo "int_port=${port}"
        lsof -t -i:"${port}" | xargs -r kill -9
    done
}
trap cleanup EXIT

echo "benchmark_vllm_defaults.sh"
echo "model_path=${model_path}"
echo "dataset_path=${dataset_path}"
echo "request_rate=${request_rate}"
echo "num_requests=${num_requests}"
echo "gpu_nums=${gpu_nums}"
echo "ports=${port_csv}"
echo "model=${model}"
echo "dataset_name=${dataset_name}"

echo "server start"
for ((i=0; i<gpu_nums; i++)); do
    port="${ports[$i]}"
    echo "device:${i}"
    echo "addr:${port}"
    CUDA_VISIBLE_DEVICES="${i}" bash "${SCRIPT_DIR}/run_server_vllm_defaults.sh" "${model_path}" "${port}" &
done
echo "finish server start"

echo "client start"
bash "${SCRIPT_DIR}/run_client_vllm_defaults.sh" \
    "${model_path}" \
    "${dataset_path}" \
    "${request_rate}" \
    "${num_requests}" \
    "${port_csv}" \
    "${model}" \
    "${dataset_name}"
echo "finish client start"
