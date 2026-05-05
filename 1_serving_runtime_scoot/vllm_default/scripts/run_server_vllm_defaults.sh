#!/usr/bin/env bash
set -euo pipefail

model_path=$1
port=$2

int_port=$((port + 0))

echo "run_server_vllm_defaults.sh"
echo "port ${int_port}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo python -m vllm.entrypoints.api_server \
    --model "${model_path}" \
    --disable-log-requests \
    --port "${int_port}" \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code

python -m vllm.entrypoints.api_server \
    --model "${model_path}" \
    --disable-log-requests \
    --port "${int_port}" \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code

