# Accel-Sim 2.0 ML Trace Workloads

vLLM / HuggingFace workload wrappers used to collect the ML SASS traces and NCU
counter files for Accel-Sim 2.0 (LLM inference/training, vision, recsys, audio,
diffusion, and a FlashAttention microbenchmark). Each workload uses a PyTorch hook
to selectively trace a representative layer, so a single traced block extrapolates
to full-model performance.

These scripts are self-contained and **portable**: they contain no machine-specific
paths. Everything is configured through environment variables.

## Requirements
- An **accel-sim-framework** checkout (provides `run_hw_trace.py`, `run_hw.py`, the
  NVBit tracer + `torch_hook`, and the `define-all-apps.yml` suite definitions).
- A Python environment with `vllm`, `torch`, `transformers`, `nvtx` (see the model
  driver imports). Gated models (e.g. Llama) need `huggingface-cli login` / `HF_TOKEN`.
- CUDA toolkit with `ncu` on `PATH`; an NVBit-TMA core built for your GPU arch.

## Environment variables (all optional unless noted)
| Var | Meaning |
|---|---|
| `ACCELSIM_ROOT` | **Required** when these scripts live outside the accel-sim tree — path to the accel-sim-framework checkout (locates the tracer `torch_hook` and the run scripts). |
| `WORKLOAD_VENV` | Path to a Python venv to activate. If unset, the active environment is used as-is. |
| `HF_HOME` | HuggingFace cache dir (models auto-download; point at a large disk if desired). |
| `TORCH_HOME` | Torch hub cache dir. |

`common_env.sh` centralizes all of the above; every wrapper sources it.

## Turnkey collection
Collect BOTH the trace and the NCU counters for every suite on one GPU:
```bash
export ACCELSIM_ROOT=/path/to/accel-sim-framework
./collect_all.sh [DEVICE]          # DEVICE defaults to 0
```
Options: `SKIP_LARGE=1` drops `mixtral_inference` + `llama_training` on <=32 GB GPUs;
`FA_SUITE=FA3` selects the Hopper FlashAttention microbenchmark (default `FA2` for
Ampere/Blackwell); `LOG_DIR=...` sets the per-suite log directory.

### The two collection recipes (kept identical between trace and NCU)
- **torch_hook model suites** (selective-layer): trace with `--spinlock_handling
  mark_region`; NCU **with** `--profile-from-start off` so it honors the
  `cudaProfilerStart/Stop` region and profiles exactly the traced layers.
- **FA microbenchmark**: no profiler region — trace instruments all kernels
  (`--spinlock_handling none`); NCU runs **without** `--profile-from-start off`.

## Running a single suite manually
```bash
export ACCELSIM_ROOT=/path/to/accel-sim-framework
cd "$ACCELSIM_ROOT"
ALLOW_REG_VAL_TRACING=1 ./util/tracer_nvbit/run_hw_trace.py -B llama_inference -D 0 \
    --spinlock_handling mark_region
./util/hw_stats/run_hw.py -B llama_inference -D 0 --ncu-flags "--profile-from-start off"
```

## Files
- `common_env.sh` — shared, path-independent environment setup (sourced by all wrappers).
- `collect_all.sh` — turnkey trace + NCU collection across all suites.
- `inference_*.sh`, `training_*.sh`, `fa2_attention.sh` — per-workload launch wrappers
  (invoked by `run_hw_trace.py` / `run_hw.py` via `define-all-apps.yml`).
- `*_inference_trace.py`, `llama_training_trace.py`
  — the model driver scripts the wrappers call.
- `_llm_prompt_utils.py` — shared prompt-construction helpers.
