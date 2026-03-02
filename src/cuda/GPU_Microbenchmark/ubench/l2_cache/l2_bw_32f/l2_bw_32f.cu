// This code is a modification of L2 cache benchmark from
//"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking":
// https://arxiv.org/pdf/1804.06826.pdf

// This benchmark measures the maximum read bandwidth of L2 cache for 32f
// Compile this file using the following command to disable L1 cache:
//    nvcc -Xptxas -dlcm=cg -Xptxas -dscm=wt l2_bw.cu

#include <algorithm>
#include <assert.h>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../../hw_def/hw_def.h"

// Change this to control the number of hardcoded loads (or pass -DNUM_LOADS=N to nvcc)
#ifndef NUM_LOADS
#define NUM_LOADS 8
#endif

// Compile-time recursive helpers to generate N load instructions and sum N elements
template <int N>
struct LoadHelper {
  __device__ __forceinline__ static void load(float *tmp, float *posArray, uint32_t uid) {
    float *ptr = posArray + (N-1) * warpSize + uid;
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(tmp[N-1]) : "l"(ptr) : "memory");
    LoadHelper<N-1>::load(tmp, posArray, uid);
  }
  __device__ __forceinline__ static float sum(float *tmp) {
    return tmp[N-1] + LoadHelper<N-1>::sum(tmp);
  }
};

template <>
struct LoadHelper<0> {
  __device__ __forceinline__ static void load(float *, float *, uint32_t) {}
  __device__ __forceinline__ static float sum(float *) { return 0.0f; }
};

/*
L2 cache is warmed up by loading posArray and adding sink
Start timing after warming up
Load posArray and add sink to generate read traffic
Repeat the previous step while offsetting posArray by one each iteration
Stop timing and store data
*/

__global__ void l2_bw(uint64_t *startClk, uint64_t *stopClk, float *dsink,
                      float *posArray, unsigned ARRAY_SIZE, uint32_t repeat_times)
{
  // block and thread index
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t uid = bid * blockDim.x + tid;
  uint32_t warp_id = uid / warpSize;
  uint32_t lane_id = uid % warpSize;

  // a register to avoid compiler optimization
  float sink = 0;

  // warm up l2 cache
  for (uint32_t i = uid; i < ARRAY_SIZE; i += blockDim.x * gridDim.x)
  {
    float *ptr = posArray + i;
    // every warp loads all data in l2 cache
    // use cg modifier to cache the load in L2 and bypass L1
    asm volatile("{\t\n"
                 ".reg .f32 data;\n\t"
                 "ld.global.cg.f32 data, [%1];\n\t"
                 "add.f32 %0, data, %0;\n\t"
                 "}"
                 : "+f"(sink)
                 : "l"(ptr)
                 : "memory");
  }

  asm volatile("membar.gl;");

  float tmp[NUM_LOADS];

  // start timing
  uint64_t start = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start)::"memory");

  for (uint32_t i = 0; i < repeat_times; i++) {
    
  LoadHelper<NUM_LOADS>::load(tmp, posArray + warp_id * NUM_LOADS * warpSize, lane_id);

  asm volatile("membar.gl;");
  // stop timing
  sink += LoadHelper<NUM_LOADS>::sum(tmp);
}
uint64_t stop = 0;
asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stop)::"memory");


  // store the result
  startClk[bid * blockDim.x + tid] = start;
  stopClk[bid * blockDim.x + tid] = stop;
  dsink[bid * blockDim.x + tid] = sink;
}

int main(int argc, char *argv[])
{

  initializeDeviceProp(0, argc, argv);

  uint32_t repeat_times = 32;

  unsigned ARRAY_SIZE = config.TOTAL_THREADS * NUM_LOADS;
  assert(ARRAY_SIZE * sizeof(float) <
         config.L2_SIZE); // Array size must not exceed L2 size

  config.BLOCKS_NUM = config.SM_NUMBER * 2; // 2 blocks per SM

  uint64_t *startClk = (uint64_t *)malloc(config.TOTAL_THREADS * sizeof(uint64_t));
  uint64_t *stopClk = (uint64_t *)malloc(config.TOTAL_THREADS * sizeof(uint64_t));

  float *posArray = (float *)malloc(ARRAY_SIZE * sizeof(float));
  float *dsink = (float *)malloc(config.TOTAL_THREADS * sizeof(float));

  float *posArray_g;
  float *dsink_g;
  uint64_t *startClk_g;
  uint64_t *stopClk_g;

  for (int i = 0; i < ARRAY_SIZE; i++)
    posArray[i] = (float)i;

  gpuErrchk(cudaMalloc(&posArray_g, ARRAY_SIZE * sizeof(float)));
  gpuErrchk(cudaMalloc(&dsink_g, config.TOTAL_THREADS * sizeof(float)));
  gpuErrchk(cudaMalloc(&startClk_g, config.TOTAL_THREADS * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, config.TOTAL_THREADS * sizeof(uint64_t)));

  gpuErrchk(cudaMemcpy(posArray_g, posArray, ARRAY_SIZE * sizeof(float),
                       cudaMemcpyHostToDevice));

  l2_bw<<<config.BLOCKS_NUM, config.THREADS_PER_BLOCK>>>(startClk_g, stopClk_g, dsink_g,
                                                         posArray_g, ARRAY_SIZE, repeat_times);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(startClk, startClk_g, config.TOTAL_THREADS * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk, stopClk_g, config.TOTAL_THREADS * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(dsink, dsink_g, config.TOTAL_THREADS * sizeof(float),
                       cudaMemcpyDeviceToHost));

  float bw, BW;
  unsigned long long data =
      (unsigned long long)config.TOTAL_THREADS * NUM_LOADS * sizeof(float) * repeat_times;
  uint64_t min_start = startClk[0], max_stop = stopClk[0];
  for (int i = config.THREADS_PER_BLOCK; i < config.TOTAL_THREADS; i += config.THREADS_PER_BLOCK) {
    if (startClk[i] < min_start) min_start = startClk[i];
    if (stopClk[i] > max_stop) max_stop = stopClk[i];
  }
  uint64_t total_time = max_stop - min_start;
  std::cout << "Total Clk number = " << total_time << "\n";

  // uint64_t total_time =
  // *std::max_element(&stopClk[0],&stopClk[TOTAL_THREADS])-*std::min_element(&startClk[0],&startClk[TOTAL_THREADS]);
  bw = (float)(data) / ((float)(total_time));
  BW = bw * config.CLK_FREQUENCY * 1000000 / 1024 / 1024 / 1024;
  std::cout << "L2 bandwidth = " << bw << "(byte/clk), " << BW << "(GB/s)\n";

  float max_bw = config.L2_BANKS * 64.f;
  BW = max_bw * config.CLK_FREQUENCY * 1000000 / 1024 / 1024 / 1024;
  std::cout << "Max Theortical L2 bandwidth = " << max_bw << "(byte/clk), "
            << BW << "(GB/s)\n";
  std::cout << "L2 BW achievable = " << (bw / max_bw) * 100 << "%\n";
  return 0;
}
