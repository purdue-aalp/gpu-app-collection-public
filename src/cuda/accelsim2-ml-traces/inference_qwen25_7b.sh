#!/bin/bash
# Qwen2.5-7B inference wrapper for Accel-Sim run_hw_trace.py
# Arguments: $1 = ctx_len, $2 = prompt_tokens, $3 = gen_tokens

set -e

CTX_LEN="${1:-256}"
PROMPT_TOKENS="${2:-255}"
GEN_TOKENS="${3:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACER_DIR="${SCRIPT_DIR}/../tracer_nvbit"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"


export PYTHONPATH="${TRACER_DIR}/others:${TRACER_DIR}/tracer_tool:${PYTHONPATH}"
export NVBIT_INSTRUMENTATION_ENABLED=0
export ALLOW_REG_VAL_TRACING=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

echo "=== Qwen2.5-7B Inference ==="
echo "  ctx_len: ${CTX_LEN}"
echo "  prompt_tokens: ${PROMPT_TOKENS}"
echo "  gen_tokens: ${GEN_TOKENS}"
echo "  TRACES_FOLDER: ${TRACES_FOLDER}"

python3 "${SCRIPT_DIR}/qwen25_inference_trace.py" \
    --ctx_len "${CTX_LEN}" \
    --prompt_tokens "${PROMPT_TOKENS}" \
    --gen_tokens "${GEN_TOKENS}"
