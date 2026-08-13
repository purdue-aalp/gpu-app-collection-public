#!/bin/bash
# Llama 3.1 8B training wrapper for Accel-Sim run_hw_trace.py
# Arguments: $1 = batch_size, $2 = seq_len, $3 = num_steps

set -e

BATCH_SIZE="${1:-1}"
SEQ_LEN="${2:-256}"
NUM_STEPS="${3:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACER_DIR="${SCRIPT_DIR}/../tracer_nvbit"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"


export PYTHONPATH="${TRACER_DIR}/others:${TRACER_DIR}/tracer_tool:${PYTHONPATH}"
export NVBIT_INSTRUMENTATION_ENABLED=0
export ALLOW_REG_VAL_TRACING=1

echo "=== Llama 3.1 8B Training ==="
echo "  batch_size: ${BATCH_SIZE}"
echo "  seq_len: ${SEQ_LEN}"
echo "  num_steps: ${NUM_STEPS}"
echo "  TRACES_FOLDER: ${TRACES_FOLDER}"

python3 "${SCRIPT_DIR}/llama_training_trace.py" \
    --batch_size "${BATCH_SIZE}" \
    --seq_len "${SEQ_LEN}" \
    --num_steps "${NUM_STEPS}"
