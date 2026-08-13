#!/bin/bash
# 3D-UNet inference wrapper for Accel-Sim run_hw_trace.py
# Arguments: $1 = batch_size, $2 = input_size (volume dimension)

set -e

BATCH_SIZE="${1:-1}"
INPUT_SIZE="${2:-128}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACER_DIR="${SCRIPT_DIR}/../tracer_nvbit"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"


export PYTHONPATH="${TRACER_DIR}/others:${TRACER_DIR}/tracer_tool:${PYTHONPATH}"
export NVBIT_INSTRUMENTATION_ENABLED=0
export ALLOW_REG_VAL_TRACING=1

echo "=== 3D-UNet Inference ==="
echo "  batch_size: ${BATCH_SIZE}"
echo "  input_size: ${INPUT_SIZE}"
echo "  TRACES_FOLDER: ${TRACES_FOLDER}"

python3 "${SCRIPT_DIR}/unet3d_inference_trace.py" \
    --batch_size "${BATCH_SIZE}" \
    --input_size "${INPUT_SIZE}" \
    --num_batches 1
