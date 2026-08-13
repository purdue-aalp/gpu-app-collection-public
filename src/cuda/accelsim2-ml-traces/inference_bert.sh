#!/bin/bash
# BERT-large inference wrapper for Accel-Sim run_hw_trace.py
# Usage: Called by run_hw_trace.py with CUDA_INJECTION64_PATH set
#
# Arguments (passed via args in define-all-apps.yml):
#   $1 = batch_size (e.g., 32)
#   $2 = seq_len (e.g., 512)

set -e

BATCH_SIZE="${1:-32}"
SEQ_LEN="${2:-512}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACER_DIR="${SCRIPT_DIR}/../tracer_nvbit"

# Activate venv
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"


# Setup torch_hook
export PYTHONPATH="${TRACER_DIR}/others:${TRACER_DIR}/tracer_tool:${PYTHONPATH}"

# Disable instrumentation by default - torch_hook enables per-layer
export NVBIT_INSTRUMENTATION_ENABLED=0

# Register value tracing
export ALLOW_REG_VAL_TRACING=1

echo "=== BERT-large Inference ==="
echo "  batch_size: ${BATCH_SIZE}"
echo "  seq_len: ${SEQ_LEN}"
echo "  TRACES_FOLDER: ${TRACES_FOLDER}"

python3 "${SCRIPT_DIR}/bert_inference_trace.py" \
    --batch_size "${BATCH_SIZE}" \
    --seq_len "${SEQ_LEN}" \
    --num_batches 1
