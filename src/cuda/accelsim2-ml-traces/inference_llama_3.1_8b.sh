#!/bin/bash
# Llama 3.1 8B inference wrapper for Accel-Sim run_hw_trace.py
# Usage: Called by run_hw_trace.py with CUDA_INJECTION64_PATH set
#
# Arguments (passed via args in define-all-apps.yml):
#   $1 = ctx_len (e.g., 256)
#   $2 = prompt_tokens (e.g., 255)
#   $3 = gen_tokens (e.g., 1)

set -e

CTX_LEN="${1:-256}"
PROMPT_TOKENS="${2:-255}"
GEN_TOKENS="${3:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACER_DIR="${SCRIPT_DIR}/../tracer_nvbit"

# Activate venv
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"


# Setup torch_hook
export PYTHONPATH="${TRACER_DIR}/others:${TRACER_DIR}/tracer_tool:${PYTHONPATH}"

# Disable instrumentation by default - torch_hook enables per-layer
export NVBIT_INSTRUMENTATION_ENABLED=0

# vLLM settings
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# Register value tracing
export ALLOW_REG_VAL_TRACING=1

echo "=== Llama 3.1 8B Inference ==="
echo "  ctx_len: ${CTX_LEN}"
echo "  prompt_tokens: ${PROMPT_TOKENS}"
echo "  gen_tokens: ${GEN_TOKENS}"
echo "  TRACES_FOLDER: ${TRACES_FOLDER}"

python3 "${SCRIPT_DIR}/llama_inference_trace.py" \
    --ctx_len "${CTX_LEN}" \
    --prompt_tokens "${PROMPT_TOKENS}" \
    --gen_tokens "${GEN_TOKENS}"
