#ifndef BW_TENSOR_SM90_DEF_H
#define BW_TENSOR_SM90_DEF_H

#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <cute/tensor.hpp>
#include "cutlass/cluster_launch.hpp"

#include "../../../hw_def/hw_def.h"

#define REPEAT_ITERS 4096

#define M_SIZE 16 * 16

using namespace cute;

// Shared memory structure for CUTLASS CuTE WGMMA
template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage_Bandwidth
{
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
};

// CUTLASS CuTE-based tensor bandwidth kernel using WGMMA for Hopper
template <class TA, class TB, class TC, class TiledMma, class SmemLayoutA, class SmemLayoutB>
__global__ void tensor_bandwidth(uint64_t *startClk, uint64_t *stopClk,
                                 TA const* a, TB const* b, TC *res,
                                 TiledMma tiled_mma,
                                 SmemLayoutA sA_layout, SmemLayoutB sB_layout) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // Shared memory for WGMMA operands
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage_Bandwidth<TA, TB, SmemLayoutA, SmemLayoutB>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);

  // Create shared memory tensors
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);

  // Load data from global to shared memory (simple coalesced copy)
  for (int i = threadIdx.x; i < M_SIZE && i < size(sA); i += blockDim.x) {
    sA(i) = a[i];
  }
  for (int i = threadIdx.x; i < M_SIZE && i < size(sB); i += blockDim.x) {
    sB(i) = b[i];
  }

  __syncthreads();

  // Get this thread's slice of the MMA operation
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);

  // Partition shared memory tensors for this thread
  auto tCsA = thr_mma.partition_A(sA);
  auto tCsB = thr_mma.partition_B(sB);

  // Create dummy gmem tensor for C partitioning (just for shape, not used for actual memory access)
  auto gC = make_tensor(make_gmem_ptr(res), make_shape(Int<64>{}, Int<64>{}), Stride<_1, Int<64>>{});
  auto tCgC = thr_mma.partition_C(gC);

  // Create register fragments
  auto tCrA = thr_mma.make_fragment_A(tCsA);
  auto tCrB = thr_mma.make_fragment_B(tCsB);
  auto tCrC = thr_mma.make_fragment_C(tCgC);

  // Initialize accumulator
  clear(tCrC);

  // synchronize all threads
  asm volatile("bar.sync 0;");

  // Perform warpgroup matrix multiply-accumulate
  warpgroup_fence_operand(tCrC);
  warpgroup_arrive();

  // start timing
  uint64_t start = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");

  // Measure WGMMA bandwidth through repeated operations
  for (int j = 0; j < REPEAT_ITERS; ++j) {
    gemm(tiled_mma, tCrA, tCrB, tCrC);
  }

  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(tCrC);
  // synchronize all threads
  asm volatile("bar.sync 0;");

  // stop timing
  uint64_t stop = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");

  // Write result to prevent optimization
  if (threadIdx.x == 0 && size(tCrC) > 0) {
    res[0] = tCrC(0);
  }

  // write time and data back to memory
  startClk[gid] = start;
  stopClk[gid] = stop;
}

template <class T, class R> float tensor_bw() {
  // Create TiledMMA using CUTLASS CuTE WGMMA for Hopper
  // Using F16F16F16 with 64x64x16 tile size (matching wgmma_sm90.cu)
  using TiledMma = decltype(make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN,GMMA::Major::MN>{}));
  TiledMma tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN,GMMA::Major::MN>{});

  // Define shared memory layouts for 64x16 tiles
  auto bM = Int<64>{};
  auto bN = Int<64>{};
  auto bK = Int<16>{};
  auto sA_layout = tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{}, make_shape(bM, bK));
  auto sB_layout = tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{}, make_shape(bN, bK));

  // Configuration: WGMMA requires 128 threads (warpgroup size on Hopper)
  config.THREADS_PER_BLOCK = size(tiled_mma);
  config.THREADS_PER_SM = size(tiled_mma);
  config.BLOCKS_NUM = 1;
  config.TOTAL_THREADS = size(tiled_mma);

  uint64_t *startClk = (uint64_t *)malloc(config.TOTAL_THREADS * sizeof(uint64_t));
  uint64_t *stopClk = (uint64_t *)malloc(config.TOTAL_THREADS * sizeof(uint64_t));
  T *data1 = (T *)malloc(M_SIZE * sizeof(T));
  T *data2 = (T *)malloc(M_SIZE * sizeof(T));
  R *res = (R *)malloc(M_SIZE * sizeof(R));

  uint64_t *startClk_g;
  uint64_t *stopClk_g;
  T *data1_g;
  T *data2_g;
  R *res_g;

  for (uint32_t i = 0; i < M_SIZE; i++) {
    data1[i] = (T)1.0f;
    data2[i] = (T)1.0f;
  }

  gpuErrchk(cudaMalloc(&startClk_g, config.TOTAL_THREADS * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, config.TOTAL_THREADS * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&data1_g, M_SIZE * sizeof(T)));
  gpuErrchk(cudaMalloc(&data2_g, M_SIZE * sizeof(T)));
  gpuErrchk(cudaMalloc(&res_g, M_SIZE * sizeof(R)));

  gpuErrchk(
      cudaMemcpy(data1_g, data1, M_SIZE * sizeof(T), cudaMemcpyHostToDevice));
  gpuErrchk(
      cudaMemcpy(data2_g, data2, M_SIZE * sizeof(T), cudaMemcpyHostToDevice));

  // Calculate shared memory size
  int smem_size = sizeof(SharedStorage_Bandwidth<T, T, decltype(sA_layout), decltype(sB_layout)>);

  // Set shared memory configuration
  auto* kernel_ptr = &tensor_bandwidth<T, T, R, TiledMma, decltype(sA_layout), decltype(sB_layout)>;
  gpuErrchk(cudaFuncSetAttribute(kernel_ptr,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  smem_size));

  // Launch kernel with TiledMma and smem layouts
  kernel_ptr<<<config.BLOCKS_NUM, config.THREADS_PER_BLOCK, smem_size>>>(
      startClk_g, stopClk_g, data1_g, data2_g, res_g, tiled_mma, sA_layout, sB_layout);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(startClk, startClk_g, config.TOTAL_THREADS * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk, stopClk_g, config.TOTAL_THREADS * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));

  // Calculate bandwidth metrics
  float wgmma_cycles_per_op, gmma_cycles_per_op;
  uint64_t total_time = stopClk[0] - startClk[0];
  wgmma_cycles_per_op = ((float)(total_time)) / ((float)(REPEAT_ITERS));
  // Note: On Hopper, WGMMA is the actual instruction, not decomposed like WMMA->HMMA
  gmma_cycles_per_op = wgmma_cycles_per_op;  // WGMMA directly maps to GMMA instructions

  std::cout << "WGMMA cycles per op (CuTE) = " << wgmma_cycles_per_op << " (clk)\n";
  std::cout << "GMMA cycles per op = " << gmma_cycles_per_op << " (clk)\n";
  std::cout << "Total Clk number = " << total_time << "\n";

  // Cleanup
  cudaFree(startClk_g);
  cudaFree(stopClk_g);
  cudaFree(data1_g);
  cudaFree(data2_g);
  cudaFree(res_g);
  free(startClk);
  free(stopClk);
  free(data1);
  free(data2);
  free(res);

  return wgmma_cycles_per_op;
}

#endif
