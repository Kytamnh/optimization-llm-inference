#!/usr/bin/env bash
# Stage 3 (vLLM-default) benchmark client.
# We pass the vLLM 0.5.5 stock defaults explicitly here so the resulting
# vllm-*.json records them under the same keys the BO methods use, instead
# of leaving the 9 SCOOT-tuned fields as `null`. The actual server flags
# come from run_server_vllm_defaults.sh; these flags only annotate the
# result JSON to make cross-method comparison clean.
set -euo pipefail

model_path=$1
dataset_path=$2
request_rate=$3
num_requests=$4
port=$5
model=$6
dataset_name=$7

# vLLM 0.5.5 stock defaults that match `run_server_vllm_defaults.sh`
# (which only passes --gpu-memory-utilization 0.9 + --trust-remote-code).
DEFAULT_TP=1
DEFAULT_MAX_NUM_SEQS=256
DEFAULT_MAX_NUM_BATCHED_TOKENS=4096
DEFAULT_BLOCK_SIZE=16
DEFAULT_SCHEDULER_DELAY_FACTOR=0.0
DEFAULT_ENABLE_CHUNKED_PREFILL=False
DEFAULT_ENABLE_PREFIX_CACHING=False
DEFAULT_DISABLE_CUSTOM_ALL_REDUCE=False
DEFAULT_USE_V2_BLOCK_MANAGER=False

echo "run_client_vllm_defaults.sh"
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
    --seed 42 \
    --tensor-parallel-size "${DEFAULT_TP}" \
    --max-num-seqs "${DEFAULT_MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${DEFAULT_MAX_NUM_BATCHED_TOKENS}" \
    --block-size "${DEFAULT_BLOCK_SIZE}" \
    --scheduler-delay-factor "${DEFAULT_SCHEDULER_DELAY_FACTOR}" \
    --enable-chunked-prefill "${DEFAULT_ENABLE_CHUNKED_PREFILL}" \
    --enable-prefix-caching "${DEFAULT_ENABLE_PREFIX_CACHING}" \
    --disable-custom-all-reduce "${DEFAULT_DISABLE_CUSTOM_ALL_REDUCE}" \
    --use-v2-block-manager "${DEFAULT_USE_V2_BLOCK_MANAGER}"

