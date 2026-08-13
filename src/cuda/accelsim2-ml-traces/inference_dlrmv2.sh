#!/bin/bash
# DLRMv2 inference wrapper for Accel-Sim run_hw_trace.py
# Arguments: $1 = batch_size (e.g., 256)

set -e

BATCH_SIZE="${1:-256}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACER_DIR="${SCRIPT_DIR}/../tracer_nvbit"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"


export PYTHONPATH="${TRACER_DIR}/others:${TRACER_DIR}/tracer_tool:${PYTHONPATH}"
export NVBIT_INSTRUMENTATION_ENABLED=0
export ALLOW_REG_VAL_TRACING=1

echo "=== DLRMv2 Inference ==="
echo "  batch_size: ${BATCH_SIZE}"
echo "  TRACES_FOLDER: ${TRACES_FOLDER}"

python3 "${SCRIPT_DIR}/dlrmv2_inference_trace.py" \
    --batch_size "${BATCH_SIZE}" \
    --num_batches 1
