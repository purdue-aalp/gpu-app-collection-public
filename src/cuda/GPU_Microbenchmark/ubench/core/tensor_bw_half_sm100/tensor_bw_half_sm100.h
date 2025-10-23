#ifndef BW_TENSOR_SM100_DEF_H
#define BW_TENSOR_SM100_DEF_H

#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/barrier.h>
#include <cute/arch/tmem_allocator_sm100.hpp>

#include "../../../hw_def/hw_def.h"

#define REPEAT_ITERS 4096

#define M_SIZE 128 * 256

using namespace cute;

// Shared memory structure for CUTLASS CuTE tcgen05.mma on SM100
template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage_Bandwidth
{
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;

  alignas(16) cute::uint64_t mma_barrier;   // Barrier to track MMA computation on SMEM
  alignas(16) cute::uint32_t tmem_base_ptr; // Base pointer for TMEM allocation

  CUTE_DEVICE constexpr auto tensor_sA() { return make_tensor(make_smem_ptr(A.begin()), SmemLayoutA{}); }
  CUTE_DEVICE constexpr auto tensor_sB() { return make_tensor(make_smem_ptr(B.begin()), SmemLayoutB{}); }
};

// CUTLASS CuTE-based tensor bandwidth kernel using tcgen05.mma for Blackwell (SM100)
template <class TA, class TB, class TC, class TiledMma, class SmemLayoutA, class SmemLayoutB, int Row=64, int Col=256>
__global__ void tensor_bandwidth(uint64_t *startClk, uint64_t *stopClk,
                                 TA const* a, TB const* b, TC *res,
                                 TiledMma tiled_mma,
                                 SmemLayoutA sA_layout, SmemLayoutB sB_layout) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int tbid = blockIdx.x;

  // Shared memory for tcgen05.mma operands
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage_Bandwidth<TA, TB, SmemLayoutA, SmemLayoutB>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);

  // Create shared memory tensors
  Tensor sA = smem.tensor_sA();
  Tensor sB = smem.tensor_sB();

  // Load data from global to shared memory (simple coalesced copy)
  for (int i = threadIdx.x; i < M_SIZE && i < size(sA); i += blockDim.x) {
    sA(i) = a[i];
  }
  for (int i = threadIdx.x; i < M_SIZE && i < size(sB); i += blockDim.x) {
    sB(i) = b[i];
  }

  __syncthreads();

  // Get this thread's slice of the MMA operation (SM100 uses peer CTA coordinate)
  int mma_v = 0;  // Single CTA, so peer coordinate is 0
  auto cta_mma = tiled_mma.get_slice(mma_v);

  // MMA Fragment Allocation
  // For tcgen05.mma operations:
  // - Matrices A and B are sourced from SMEM
  // - tCrA and tCrB provide descriptor views of sA and sB respectively
  auto tCrA = cta_mma.make_fragment_A(sA);
  auto tCrB = cta_mma.make_fragment_B(sB);

  // TMEM Allocation
  // On SM100 architecture, accumulators are stored exclusively in tensor memory (TMEM).
  // Create a dummy gmem tensor for C (just for shape)
  auto gC = make_tensor(make_gmem_ptr(res), make_shape(Int<Row>{}, Int<Col>{}), make_stride(Int<Col>{}, Int<1>{}));
  auto tCgC = cta_mma.partition_C(gC);
  auto tCtAcc = cta_mma.make_fragment_C(tCgC);

  uint32_t elect_one_thr  = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};

  if (elect_one_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &smem.tmem_base_ptr);
  }
  __syncthreads(); // Wait for all threads until warp0 allocates TMEM
  tCtAcc.data() = smem.tmem_base_ptr;

  // Barrier Initialization
  if (elect_one_warp && elect_one_thr) {
    cute::initialize_barrier(smem.mma_barrier, /* num_ctas */ 1);
  }
  int mma_barrier_phase_bit = 0;
  __syncthreads();

  // synchronize all threads
  asm volatile("bar.sync 0;");

  // start timing
  uint64_t start = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");

  // Set mma accumulate option to zero so that the first MMA instruction will clear the TMEM accumulator.
  //tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
  tiled_mma.accumulate_ = UMMA::ScaleOut::One;

  // tcgen05.mma instructions require single-warp execution
  if (elect_one_warp) {
    // Measure tcgen05.mma bandwidth through repeated operations
    for (int j = 0; j < REPEAT_ITERS; ++j) {
      // This line spawns 4 128x256x16 MMA ops via cute/algorithm/gemm.hpp:298
      gemm(tiled_mma, tCrA, tCrB, tCtAcc);
      
      // This line spawns one 128x256x16 MMA op
      //gemm(tiled_mma, tCrA(_,_,0), tCrB(_,_,1), tCtAcc);
      
      //tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
    // Ensure MMAs are completed
    cutlass::arch::umma_arrive(&smem.mma_barrier);
  }

  // Wait MMAs to complete
  cute::wait_barrier(smem.mma_barrier, mma_barrier_phase_bit);
  mma_barrier_phase_bit ^= 1;

  // synchronize all threads
  asm volatile("bar.sync 0;");

  // stop timing
  uint64_t stop = 0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");

  // Simple write to prevent optimization - we don't actually need to load from TMEM for bandwidth measurement
  if (threadIdx.x == 0) {
    res[0] = (TC)1.0f;
  }

  // write time and data back to memory
  if(threadIdx.x == 0){
    startClk[tbid] = start;
    stopClk[tbid] = stop;
  }
  
  // Release and deallocate TMEM
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(smem.tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }
}

template <class T, class R, int Row=64, int Col=256> float tensor_bw() {
  // Create TiledMMA using CUTLASS CuTE tcgen05.mma for Blackwell (SM100)
  // Using F16BF16 with 128x256x16 tile size (matching 01_mma_sm100.cu)
  using TiledMma = decltype(make_tiled_mma(SM100_MMA_F16BF16_SS<T, T, R,
                                                                 Row, Col,
                                                                 UMMA::Major::K, UMMA::Major::K>{}));
  TiledMma tiled_mma = make_tiled_mma(SM100_MMA_F16BF16_SS<T, T, R,
                                                            Row, Col,
                                                            UMMA::Major::K, UMMA::Major::K>{});

  // using TiledMma = decltype(make_tiled_mma(SM100_MMA_F8F6F4_SS{}));
  // TiledMma tiled_mma = make_tiled_mma(SM100_MMA_F8F6F4_SS{});

  // Define MMA tiler sizes (static) - following 01_mma_sm100.cu pattern
  auto bM = tile_size<0>(tiled_mma);             // MMA Tile M = 128
  auto bN = tile_size<1>(tiled_mma);             // MMA Tile N = 256
  auto bK = tile_size<2>(tiled_mma) * Int<8>{}; // MMA Tile K = 16 * 4 = 64 (need at least 4 for swizzle layout)
  auto mma_tiler = make_shape(bM, bN, bK);       // (128, 256, 64)

  // Determine the SMEM layouts using partition_shape and tile_to_mma_shape
  // Pre-partitioned Tile Shape (MmaTile_M, MmaTile_K) to post-partitioned (MmaA, NumMma_M, NumMma_K)
  auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  // Pre-partitioned Tile Shape (MmaTile_N, MmaTile_K) to post-partitioned (MmaB, NumMma_N, NumMma_K)
  auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));

  // Create swizzled SMEM layouts using UMMA helper functions
  auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW32_Atom<T>{}, mma_shape_A);
  auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW32_Atom<T>{}, mma_shape_B);

  // Print layout information
  // std::cout << "TiledMma:\t"; print(tiled_mma); std::cout << "\n";
  std::cout << "mma_tiler:\t"; print(mma_tiler); std::cout << "\n";
  // std::cout << "mma_tiler total size:\t" << int(size(mma_tiler)) << "\n";
  std::cout << "mma_shape_A:\t"; print(mma_shape_A); std::cout << "\n";
  std::cout << "mma_shape_B:\t"; print(mma_shape_B); std::cout << "\n";
  std::cout << "sA_layout:\t"; print(sA_layout); std::cout << "\n";
  std::cout << "sB_layout:\t"; print(sB_layout); std::cout << "\n";

  // Configuration: tcgen05.mma uses 128 threads on SM100
  config.THREADS_PER_BLOCK = 128;
  config.THREADS_PER_SM = 128;
  config.BLOCKS_NUM = 148;
  config.TOTAL_THREADS = 128;

  // Allocate buffers - res needs to be large enough for 128x256 output
  int OUT_SIZE = 128 * 256;
  uint64_t *startClk = (uint64_t *)malloc(config.TOTAL_THREADS * sizeof(uint64_t));
  uint64_t *stopClk = (uint64_t *)malloc(config.TOTAL_THREADS * sizeof(uint64_t));
  T *data1 = (T *)malloc(M_SIZE * sizeof(T));
  T *data2 = (T *)malloc(M_SIZE * sizeof(T));
  R *res = (R *)malloc(OUT_SIZE * sizeof(R));

  uint64_t *startClk_g;
  uint64_t *stopClk_g;
  T *data1_g;
  T *data2_g;
  R *res_g;

  for (uint32_t i = 0; i < M_SIZE; i++) {
    data1[i] = (T)1.0f;
    data2[i] = (T)1.0f;
  }

  gpuErrchk(cudaMalloc(&startClk_g, config.BLOCKS_NUM * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, config.BLOCKS_NUM * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&data1_g, M_SIZE * sizeof(T)));
  gpuErrchk(cudaMalloc(&data2_g, M_SIZE * sizeof(T)));
  gpuErrchk(cudaMalloc(&res_g, OUT_SIZE * sizeof(R)));

  gpuErrchk(
      cudaMemcpy(data1_g, data1, M_SIZE * sizeof(T), cudaMemcpyHostToDevice));
  gpuErrchk(
      cudaMemcpy(data2_g, data2, M_SIZE * sizeof(T), cudaMemcpyHostToDevice));

  // Calculate shared memory size
  int smem_size = sizeof(SharedStorage_Bandwidth<T, T, decltype(sA_layout), decltype(sB_layout)>);

  // Set shared memory configuration
  auto* kernel_ptr = &tensor_bandwidth<T, T, R, TiledMma, decltype(sA_layout), decltype(sB_layout), Row, Col>;
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
  float tcgen05_mma_cycles_per_op, umma_cycles_per_op;
  uint64_t total_time = stopClk[0] - startClk[0];
  tcgen05_mma_cycles_per_op = ((float)(total_time)) / ((float)(REPEAT_ITERS));
  // On Blackwell SM100, tcgen05.mma directly maps to UMMA instructions
  umma_cycles_per_op = tcgen05_mma_cycles_per_op;

  std::cout << "tcgen05.mma cycles per op (CuTE) = " << tcgen05_mma_cycles_per_op << " (clk)\n";
  std::cout << "UMMA cycles per op = " << umma_cycles_per_op << " (clk)\n";
  std::cout << "Total Clk number = " << total_time << "\n";
  std::cout << "Calculated throughput = " << float(size(mma_tiler)) / tcgen05_mma_cycles_per_op << " MACs/clk\n";

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

  return tcgen05_mma_cycles_per_op;
}

#endif
