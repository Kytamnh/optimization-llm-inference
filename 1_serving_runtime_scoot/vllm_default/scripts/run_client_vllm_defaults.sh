#!/usr/bin/env bash
set -euo pipefail

model_path=$1
dataset_path=$2
request_rate=$3
num_requests=$4
port=$5
model=$6
dataset_name=$7

echo "run_client_vllm_defaults.sh"
echo python -m clients.benchmark_serving \
    --backend vllm \
    --tokenizer "${model_path}" \
    --dataset-name "${dataset_name}" \
    --dataset-path "${dataset_path}" \
    --request-rate "${request_rate}" \
    --model "${model}" \
    --num-prompts "${num_requests}" \
    --save-result \
    --port "${port}" \
    --trust-remote-code \
    --disable-tqdm \
    --seed 42

python -m clients.benchmark_serving \
    --backend vllm \
    --tokenizer "${model_path}" \
    --dataset-name "${dataset_name}" \
    --dataset-path "${dataset_path}" \
    --request-rate "${request_rate}" \
    --model "${model}" \
    --num-prompts "${num_requests}" \
    --save-result \
    --port "${port}" \
    --trust-remote-code \
    --disable-tqdm \
    --seed 42

