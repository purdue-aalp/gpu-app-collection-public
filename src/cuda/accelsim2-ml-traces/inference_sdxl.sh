#!/bin/bash
# Stable Diffusion XL inference wrapper for Accel-Sim run_hw_trace.py
# Arguments: $1 = num_steps, $2 = image_size

set -e

NUM_STEPS="${1:-20}"
IMAGE_SIZE="${2:-1024}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACER_DIR="${SCRIPT_DIR}/../tracer_nvbit"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"


export PYTHONPATH="${TRACER_DIR}/others:${TRACER_DIR}/tracer_tool:${PYTHONPATH}"
export NVBIT_INSTRUMENTATION_ENABLED=0
export ALLOW_REG_VAL_TRACING=1

echo "=== Stable Diffusion XL Inference ==="
echo "  num_steps: ${NUM_STEPS}"
echo "  image_size: ${IMAGE_SIZE}"
echo "  TRACES_FOLDER: ${TRACES_FOLDER}"

python3 "${SCRIPT_DIR}/sdxl_inference_trace.py" \
    --num_steps "${NUM_STEPS}" \
    --image_size "${IMAGE_SIZE}" \
    --num_images 1
