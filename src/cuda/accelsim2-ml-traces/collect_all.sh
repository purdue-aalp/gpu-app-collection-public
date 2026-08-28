#!/bin/bash
# Turnkey collection of the ML workload dataset: for every suite, collect BOTH the
# SASS trace (run_hw_trace.py) and the NCU counter file (run_hw.py), on one GPU.
#
# CRITICAL invariant (see the two recipes below): NCU must profile EXACTLY the same
# kernels that are traced.
#   - torch_hook model suites use a cudaProfilerStart/Stop region -> NCU is run WITH
#     "--profile-from-start off" so it honors that region (= the traced layers).
#   - the FA microbenchmark has no profiler region -> trace instruments all kernels and
#     NCU runs WITHOUT "--profile-from-start off".
#
# Usage:
#   ACCELSIM_ROOT=/path/to/accel-sim-framework  ./collect_all.sh [DEVICE]
#
# Environment:
#   ACCELSIM_ROOT   (required) accel-sim-framework checkout whose define-all-apps.yml
#                   defines these suites and provides run_hw_trace.py / run_hw.py.
#   DEVICE          GPU index (positional arg $1, default 0).
#   SKIP_LARGE=1    Drop mixtral_inference + llama_training (for <=32 GB GPUs, e.g. RTX 5090).
#   FA_SUITE        (DISABLED) FA microbenchmark suite. The FA suites are currently
#                   commented out -- see the note above run_suite_fa at the bottom.
#   LOG_DIR         Where per-suite logs go (default ./collect_logs).
#   WORKLOAD_VENV   Passed through to the wrappers (Python venv to activate); optional.
#   SUITE_VENV_<suite>
#                   Per-suite venv override, for suites whose Python deps conflict with
#                   the default env. Example: torchrec/fbgemm_gpu must match the exact
#                   libtorch it was built against, so dlrmv2 typically needs its own env:
#                     export SUITE_VENV_dlrmv2_inference=/path/to/torchrec_env
#                   Suites with no override fall back to WORKLOAD_VENV.
set -uo pipefail

: "${ACCELSIM_ROOT:?set ACCELSIM_ROOT to your accel-sim-framework checkout}"
: "${CUDA_INSTALL_PATH:=/usr/local/cuda}"
export ACCELSIM_ROOT CUDA_INSTALL_PATH
export PATH="${CUDA_INSTALL_PATH}/bin:${PATH}"      # run_hw.py needs ncu on PATH

# --- link the workload scripts into the accel-sim tree ------------------------
# define-all-apps.yml declares every suite's exec_dir as "$ACCELSIM_ROOT/util/workloads".
# These scripts now live in gpu-app-collection instead, so that path no longer exists
# in-tree. Point it at this directory automatically -- no manual setup, no YAML edit.
# A real directory there (the old in-tree layout) is left untouched.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKLOADS_LINK="${ACCELSIM_ROOT}/util/workloads"
if [ -L "$WORKLOADS_LINK" ]; then
    if [ "$(readlink -f "$WORKLOADS_LINK")" != "$SCRIPT_DIR" ]; then
        ln -sfn "$SCRIPT_DIR" "$WORKLOADS_LINK" \
            && echo "note: repointed $WORKLOADS_LINK -> $SCRIPT_DIR"
    fi
elif [ -d "$WORKLOADS_LINK" ]; then
    echo "note: using existing in-tree $WORKLOADS_LINK"
elif [ -e "$WORKLOADS_LINK" ]; then
    echo "ERROR: $WORKLOADS_LINK exists but is neither a directory nor a symlink" >&2
    exit 1
else
    mkdir -p "$(dirname "$WORKLOADS_LINK")"
    ln -s "$SCRIPT_DIR" "$WORKLOADS_LINK" \
        && echo "note: linked $WORKLOADS_LINK -> $SCRIPT_DIR"
fi

DEVICE="${1:-0}"
DEFAULT_WORKLOAD_VENV="${WORKLOAD_VENV:-}"
FAILED_SUITES=()

# Select the Python env for a suite: SUITE_VENV_<suite> if set, else the default.
select_venv () {
    local key="SUITE_VENV_$1" val
    val="$(eval printf '%s' "\${${key}:-}")"
    if [ -n "$val" ]; then
        export WORKLOAD_VENV="$val"
        echo "    [venv] $1 -> $val"
    elif [ -n "$DEFAULT_WORKLOAD_VENV" ]; then
        export WORKLOAD_VENV="$DEFAULT_WORKLOAD_VENV"
    else
        unset WORKLOAD_VENV 2>/dev/null || true
    fi
}
LOG="${LOG_DIR:-$PWD/collect_logs}"; mkdir -p "$LOG"
TRACE="${ACCELSIM_ROOT}/util/tracer_nvbit/run_hw_trace.py"
NCU="${ACCELSIM_ROOT}/util/hw_stats/run_hw.py"
cd "$ACCELSIM_ROOT"

for t in "$TRACE" "$NCU"; do
    [ -x "$t" ] || { echo "ERROR: not found/executable: $t" >&2; exit 1; }
done

# Recipe 1: torch_hook model suites (selective-layer). NCU honors the profiler region.
run_suite () {
    local S=$1
    select_venv "$S"
    echo "### [$(date)] TRACE $S on GPU$DEVICE"
    ALLOW_REG_VAL_TRACING=1 "$TRACE" -B "$S" -D "$DEVICE" --spinlock_handling mark_region \
        2>&1 | tee "$LOG/$S.trace.log"
    echo "### [$(date)] NCU   $S on GPU$DEVICE"
    "$NCU" -B "$S" -D "$DEVICE" --ncu-flags "--profile-from-start off" \
        2>&1 | tee "$LOG/$S.ncu.log"
    if grep -qE "Traceback \(most recent|Error invoking" "$LOG/$S.trace.log" 2>/dev/null; then
        FAILED_SUITES+=("$S"); echo "### [WARN] $S reported errors -- see $LOG/$S.trace.log"
    fi
}

# Recipe 2: FA microbenchmark (no profiler region). Instrument-all trace + default NCU.
run_suite_fa () {
    local S=$1
    select_venv "$S"
    echo "### [$(date)] TRACE(FA) $S on GPU$DEVICE"
    ALLOW_REG_VAL_TRACING=1 "$TRACE" -B "$S" -D "$DEVICE" --spinlock_handling none \
        2>&1 | tee "$LOG/$S.trace.log"
    echo "### [$(date)] NCU(FA)   $S on GPU$DEVICE"
    "$NCU" -B "$S" -D "$DEVICE" 2>&1 | tee "$LOG/$S.ncu.log"
}

SUITES=(qwen25_inference qwen25_inference_2048 llama_inference llama_inference_2048
        mixtral_inference deepseek_r1_inference llama_training resnet_inference
        bert_inference dlrmv2_inference sdxl_inference unet3d_inference
        retinanet_inference whisper_inference)

if [ "${SKIP_LARGE:-0}" = "1" ]; then
    echo "SKIP_LARGE=1: dropping mixtral_inference and llama_training (<=32 GB GPU)"
    tmp=(); for s in "${SUITES[@]}"; do
        [ "$s" = "mixtral_inference" ] && continue
        [ "$s" = "llama_training" ] && continue
        tmp+=("$s")
    done; SUITES=("${tmp[@]}")
fi

for S in "${SUITES[@]}"; do run_suite "$S"; done
# FA suites disabled: benchmark_hopper_forward_varlen_prefill.py was removed, and it
# was the shared driver for BOTH FA2 (--fa-version 2) and FA3 (--fa-version 3).
# Restore that file and the FA3 entry in define-all-apps.yml to re-enable.
#run_suite_fa "${FA_SUITE:-FA2}"

echo "### ALL DONE $(date)"
if [ "${#FAILED_SUITES[@]}" -gt 0 ]; then
    echo "### FAILED SUITES (${#FAILED_SUITES[@]}): ${FAILED_SUITES[*]}"
else
    echo "### no suite reported errors"
fi
echo "traces: $(find "${ACCELSIM_ROOT}"/hw_run/traces -name kernelslist.g 2>/dev/null | wc -l)" \
     "ncu: $(find "${ACCELSIM_ROOT}"/hw_run -name '*.ncu-rep' 2>/dev/null | wc -l)"
