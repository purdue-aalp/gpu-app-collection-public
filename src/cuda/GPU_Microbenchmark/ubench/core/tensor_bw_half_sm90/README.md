# Tensor Bandwidth Benchmark - CUTLASS CuTE WGMMA Implementation (SM90)

This directory contains a reimplementation of the tensor core bandwidth benchmark using **CUTLASS CuTE library** with **WGMMA (Warp Group Matrix Multiply-Accumulate)** operations for NVIDIA Hopper (SM90+) GPUs.

## Overview

The original `tensor_bandwidth` function used WMMA (Warp Matrix Multiply-Accumulate) API, which is supported on Volta, Turing, Ampere, and later architectures. This version replaces the WMMA implementation with CUTLASS CuTE's WGMMA operations, which provide:

- Native support for Hopper's GMMA (Generalized Matrix Multiply-Accumulate) instructions
- Warp-group level operations (128 threads instead of 32)
- More efficient tensor core utilization on Hopper architecture

## Key Changes

### 1. **Kernel Implementation** ([tensor_bw_half_sm90.h](tensor_bw_half_sm90.h))
   - Replaced `wmma::fragment` with CUTLASS CuTE `TiledMma` and fragments
   - Uses `SM90_64x64x16_F16F16F16_SS` for 64x64x16 matrix tiles
   - Implements warpgroup synchronization (`warpgroup_arrive`, `warpgroup_wait`)
   - Uses `cute::gemm()` instead of `wmma::mma_sync()`

### 2. **Thread Configuration**
   - Changed from 32 threads (warp) to 128 threads (warpgroup)
   - THREADS_PER_BLOCK now determined by `size(tiled_mma)`

### 3. **Build System** ([Makefile](Makefile))
   - Updated to include CUTLASS library paths
   - Restricted to SM90+ architectures (Hopper and Blackwell)
   - Added CUTLASS-specific compiler flags (--expt-relaxed-constexpr, etc.)

## Requirements

- **GPU**: NVIDIA Hopper (H100, H200) or Blackwell (B100, B200) with compute capability 9.0+
- **CUDA Toolkit**: Version with SM90 support
- **CUTLASS Library**: Included in `/home/shen449/gpu-app-collection/include/`

## Build Instructions

```bash
make
```

## Usage

```bash
./tensor_bw_half_sm90
```

## Output

The benchmark measures and reports:
- **WGMMA bandwidth**: Throughput of WGMMA operations
- **GMMA bandwidth**: On Hopper, WGMMA directly maps to GMMA instructions
- **Total clock cycles**: Total time for all iterations

## Implementation Details

### GEMM Operation
```cpp
gemm(tiled_mma, tCrA, tCrB, tCrC);  // CuTE WGMMA
```

vs. original:
```cpp
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // WMMA API
```

### Synchronization
```cpp
warpgroup_fence_operand(tCrC);
warpgroup_arrive();
gemm(tiled_mma, tCrA, tCrB, tCrC);
warpgroup_commit_batch();
warpgroup_wait<0>();
```

## Source Reference

Based on CUTLASS example: `/home/shen449/gpu-app-collection/src/cuda/hopper_tensor_core/wgmma_sm90.cu`

## License

Inherits license from original GPU_Microbenchmark and CUTLASS (BSD-3-Clause)
