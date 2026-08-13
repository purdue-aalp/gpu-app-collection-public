#!/bin/bash
# FlashAttention-2 microbenchmark wrapper for Accel-Sim run_hw_trace.py / run_hw.py.
# Runs the SAME attention benchmark as the FA3 suite but with --fa-version 2, which
# is supported on Ampere (sm80+) where FA3 (Hopper-only, cc9.x) is not.
#
# IMPORTANT: this is NOT a torch_hook workload. There is no cudaProfilerStart/Stop
# region, so:
#   - tracing instruments ALL kernels in this single-shot benchmark
#     (NVBIT_INSTRUMENTATION_ENABLED is left at its default of 1 -- do NOT set it to 0)
#   - NCU must be run WITHOUT "--profile-from-start off" (otherwise it profiles nothing).
# Both trace and NCU therefore capture the same full set of kernels for the run.
set -e

# Benchmark lives next to this wrapper (in-repo); fall back to the legacy external path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${SCRIPT_DIR}/benchmark_hopper_forward_varlen_prefill.py"

# Activate venv (provides torch + vLLM's bundled flash_attn_varlen_func)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"

# Register value tracing (v6), matching the other suites
export ALLOW_REG_VAL_TRACING=1

echo "=== FlashAttention-2 benchmark (fa_version=2) ==="
echo "  args: $*"
echo "  TRACES_FOLDER: ${TRACES_FOLDER}"

python3 "${BENCH}" --fa-version 2 "$@"
