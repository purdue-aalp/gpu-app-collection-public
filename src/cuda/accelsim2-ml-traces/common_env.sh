#!/bin/bash
# Shared environment for the Accel-Sim ML workload wrappers.
#
# Portable by design: contains NO machine-specific absolute paths. Everything is
# overridable via environment variables so a third party can reproduce the traces
# without editing any script. Source this from a wrapper:
#
#     source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"
#
# Environment variables (all optional):
#   WORKLOAD_VENV  Path to a Python venv to activate. If unset, the wrapper assumes
#                  the required packages (vllm, torch, transformers, nvtx, ...) are
#                  already importable in the active environment.
#   ACCELSIM_ROOT  Path to the accel-sim-framework checkout, used to locate the NVBit
#                  tracer's torch_hook. Required when these scripts live OUTSIDE the
#                  accel-sim tree (e.g. in gpu-app-collection-public). When the scripts
#                  sit in <accel-sim>/util/workloads it is auto-derived.
#   HF_HOME        HuggingFace cache dir (models auto-download on first run; gated
#                  repos such as Llama require `huggingface-cli login` or HF_TOKEN).
#   TORCH_HOME     Torch hub cache dir (used by torchvision/torch.hub models).

# 1. Optional Python environment ------------------------------------------------
if [ -n "${WORKLOAD_VENV:-}" ] && [ -f "${WORKLOAD_VENV}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${WORKLOAD_VENV}/bin/activate"
fi

# 2. Locate the NVBit tracer torch_hook (needed on PYTHONPATH) -------------------
#    Prefer an explicit ACCELSIM_ROOT; otherwise assume this file is in-tree at
#    <accel-sim>/util/workloads/ and derive the root two levels up.
_WL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${ACCELSIM_ROOT:=$(cd "${_WL_DIR}/../.." 2>/dev/null && pwd)}"
if [ -n "${ACCELSIM_ROOT:-}" ] && [ -d "${ACCELSIM_ROOT}/util/tracer_nvbit" ]; then
    _TRACER_DIR="${ACCELSIM_ROOT}/util/tracer_nvbit"
    export PYTHONPATH="${_TRACER_DIR}/others:${_TRACER_DIR}/tracer_tool:${PYTHONPATH}"
else
    echo "common_env.sh: WARNING: could not locate the NVBit tracer torch_hook." >&2
    echo "  Set ACCELSIM_ROOT to your accel-sim-framework checkout." >&2
fi

# 3. Model caches (auto-download; point at a large disk if desired) -------------
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

# 4. vLLM settings required for single-process NVBit tracing ---------------------
export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
