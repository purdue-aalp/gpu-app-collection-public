# CUTLASS Hopper Tensor Core Examples

This directory contains CUTLASS examples demonstrating NVIDIA Hopper (SM 90) tensor core operations.

## Applications

- **wgmma_sm90**: Demonstrates Warp Group Matrix Multiply-Accumulate (WGMMA) operations on Hopper GPUs
- **wgmma_tma_sm90**: Demonstrates WGMMA with Tensor Memory Accelerator (TMA) on Hopper GPUs

## Build Instructions

```bash
make all
```

Or build individual targets:
```bash
make wgmma_sm90
make wgmma_tma_sm90
```

## Requirements

- CUDA Toolkit with SM 90 (Hopper) support
- NVIDIA GPU with compute capability 9.0 or higher (H100, H200, etc.)
- C++17 compatible compiler

## Source

Ported from CUTLASS library: https://github.com/NVIDIA/cutlass
Original location: `examples/cute/tutorial/hopper/`

## License

BSD-3-Clause (inherited from CUTLASS)
