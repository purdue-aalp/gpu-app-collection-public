# Tensor Latency Benchmark - CUTLASS CuTE tcgen05.mma Implementation

This directory contains a reimplementation of the tensor core latency benchmark using **CUTLASS CuTE library** with **tcgen05.mma (UMMA - Unified Matrix Multiply-Accumulate)** operations for NVIDIA Blackwell (SM100) GPUs.

## Overview

This version builds upon the WGMMA implementation in `tensor_lat_half_2` and adapts it to use Blackwell's tcgen05.mma instructions. The tcgen05.mma operations provide:

- Native support for Blackwell's UMMA (Unified Matrix Multiply-Accumulate) instructions
- Larger matrix tile sizes (128x256x16 vs 64x64x16)
- Tensor Memory (TMEM) for accumulator storage instead of registers
- Enhanced performance on Blackwell architecture

## Key Changes from tensor_lat_half_2 (SM90 WGMMA)

### 1. **MMA Instructions**
   - **SM90 (WGMMA)**: `SM90_64x64x16_F16F16F16_SS`
   - **SM100 (tcgen05.mma)**: `SM100_MMA_F16BF16_SS<128, 256>`

### 2. **Accumulator Storage**
   - **SM90**: Accumulators stored in registers (RMEM)
   - **SM100**: Accumulators stored in Tensor Memory (TMEM) with explicit allocation

### 3. **TMEM Management** ([tensor_lat_half.h](tensor_lat_half.h))
   ```cpp
   using TmemAllocator = cute::TMEM::Allocator1Sm;
   TmemAllocator tmem_allocator{};
   tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &smem.tmem_base_ptr);
   tCtAcc.data() = smem.tmem_base_ptr;
   ```

### 4. **Shared Memory Layout**
   - **SM90**: `GMMA::Layout_MN_SW128_Atom` with `tile_to_shape`
   - **SM100**: `UMMA::Layout_K_SW128_Atom` with `UMMA::tile_to_mma_shape`

### 5. **Barrier Synchronization**
   - Added MMA barrier for tcgen05.mma completion tracking:
   ```cpp
   cutlass::arch::umma_arrive(&smem.mma_barrier);
   cute::wait_barrier(smem.mma_barrier, mma_barrier_phase_bit);
   ```

### 6. **Execution Model**
   - **SM90**: All warps in warpgroup participate in MMA
   - **SM100**: Only warp 0 executes tcgen05.mma (single-warp execution)
   ```cpp
   if (elect_one_warp) {
       gemm(tiled_mma, tCrA, tCrB, tCtAcc);
   }
   ```

### 7. **Accumulator Loading**
   - Added TMEM-to-RMEM copy operation:
   ```cpp
   TiledCopy tiled_t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
   copy(tiled_t2r_copy, tDtAcc, tDrAcc);
   ```

### 8. **Accumulator Scaling**
   - Uses `UMMA::ScaleOut` to control accumulator behavior:
   ```cpp
   tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;  // First iteration clears
   tiled_mma.accumulate_ = UMMA::ScaleOut::One;   // Subsequent iterations accumulate
   ```

## Requirements

- **GPU**: NVIDIA Blackwell (B100, B200, GB200) with compute capability 10.0a
- **CUDA Toolkit**: Version with SM100 support (CUDA 12.6+)
- **CUTLASS Library**: Included in `/home/shen449/cutlass_private/`

## Build Instructions

```bash
make
```

## Usage

```bash
./tensor_lat_half
```

## Output

The benchmark measures and reports:
- **tcgen05.mma latency**: Average clock cycles per tcgen05.mma operation
- **UMMA latency**: On Blackwell, tcgen05.mma directly maps to UMMA instructions
- **Total clock cycles**: Total time for all iterations

## Implementation Details

### GEMM Operation
```cpp
gemm(tiled_mma, tCrA, tCrB, tCtAcc);  // CuTE tcgen05.mma with TMEM accumulator
```

vs. SM90:
```cpp
gemm(tiled_mma, tCrA, tCrB, tCrC);  // CuTE WGMMA with register accumulator
```

vs. original:
```cpp
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // WMMA API
```

### Complete Latency Measurement Flow
```cpp
// 1. Allocate TMEM
tmem_allocator.allocate(...);

// 2. Initialize barrier
initialize_barrier(smem.mma_barrier, 1);

// 3. Measure latency
if (elect_one_warp) {
    for (int j = 0; j < REPEAT_ITERS; ++j) {
        gemm(tiled_mma, tCrA, tCrB, tCtAcc);
    }
    umma_arrive(&smem.mma_barrier);
}
wait_barrier(smem.mma_barrier, phase_bit);

// 4. Load from TMEM to RMEM
copy(tiled_t2r_copy, tDtAcc, tDrAcc);

// 5. Cleanup TMEM
tmem_allocator.release_allocation_lock();
tmem_allocator.free(...);
```

## Architecture Comparison

| Feature | SM90 (WGMMA) | SM100 (tcgen05.mma) |
|---------|--------------|---------------------|
| Instruction | `wgmma.mma` | `tcgen05.mma` |
| MMA Tile Size | 64x64x16 | 128x256x16 |
| Accumulator Location | Registers (RMEM) | Tensor Memory (TMEM) |
| Thread Execution | Warpgroup-level | Single-warp |
| Layout Type | MN-major | K-major |
| Memory Management | Automatic | Manual TMEM alloc/free |

## Source Reference

Based on CUTLASS example: `/home/shen449/cutlass_private/examples/cute/tutorial/blackwell/01_mma_sm100.cu`

Adapted from: `/home/shen449/gpu-app-collection-public/src/cuda/GPU_Microbenchmark/ubench/core/tensor_lat_half_2/`

## License

Inherits license from original GPU_Microbenchmark and CUTLASS (BSD-3-Clause)
