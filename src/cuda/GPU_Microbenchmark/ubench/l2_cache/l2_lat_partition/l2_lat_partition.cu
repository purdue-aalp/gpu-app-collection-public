// L2 cache latency partition benchmark
//
// Measures per-load latency from L2 cache misses across L2 partitions.
// 1. Allocates two L2-sized buffers
// 2. All threads pollute L2 with the first buffer to evict the second
// 3. After bar.sync, thread 0 loads from the second buffer at 128B stride,
//    256 times, measuring each load's latency individually
// 4. membar.gl serializes each load so latencies are accurate

#include <algorithm>
#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../../hw_def/hw_def.h"

#define NUM_LOADS 256
#define STRIDE_BYTES 128

__global__ void l2_lat_partition(uint64_t *latencies, uint32_t *pollute_buf,
                                 uint32_t *measure_buf, size_t pollute_elements,
                                 uint32_t *dsink) {
  uint32_t uid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t total_threads = gridDim.x * blockDim.x;

  // Phase 1: All threads pollute L2 by streaming through pollute_buf
  uint32_t sink = 0;
  for (size_t i = uid; i < pollute_elements; i += total_threads) {
    uint32_t *ptr = pollute_buf + i;
    asm volatile("{\n\t"
                 ".reg .u32 data;\n\t"
                 "ld.global.cg.u32 data, [%1];\n\t"
                 "add.u32 %0, data, %0;\n\t"
                 "}"
                 : "+r"(sink)
                 : "l"(ptr)
                 : "memory");
  }

  // Ensure all threads finished polluting before measurement
  asm volatile("bar.sync 0;");

  // Phase 2: First warp measures latency from measure_buf
  if (uid < warpSize) {
    for (uint32_t i = 0; i < NUM_LOADS; i++) {
      uint32_t *ptr = measure_buf + i * (STRIDE_BYTES / sizeof(uint32_t));
      uint32_t data;

      uint64_t start, stop;
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");

      asm volatile("ld.global.cg.u32 %0, [%1];"
                   : "=r"(data)
                   : "l"(ptr)
                   : "memory");
      asm volatile("membar.gl;" ::: "memory");

      asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");

      sink += data;
      latencies[i] = stop - start;
    }
  }

  dsink[uid] = sink;
}

int main(int argc, char *argv[]) {
  initializeDeviceProp(0, argc, argv);

  size_t l2_size = config.L2_SIZE;
  size_t buf_elements = l2_size / sizeof(uint32_t);

  printf("\nL2 Cache Size: %zu bytes\n", l2_size);
  printf("Stride: %d bytes\n", STRIDE_BYTES);
  printf("Number of loads: %d\n", NUM_LOADS);
  printf("Total measured region: %d bytes\n", NUM_LOADS * STRIDE_BYTES);

  // Allocate two L2-sized buffers on device
  uint32_t *pollute_buf_g, *measure_buf_g;
  gpuErrchk(cudaMalloc(&pollute_buf_g, l2_size));
  gpuErrchk(cudaMalloc(&measure_buf_g, l2_size));

  // Initialize both buffers to ensure pages are allocated
  gpuErrchk(cudaMemset(pollute_buf_g, 1, l2_size));
  gpuErrchk(cudaMemset(measure_buf_g, 2, l2_size));

  // Allocate latency output array and sink buffer
  uint64_t *latencies_g;
  uint32_t *dsink_g;
  gpuErrchk(cudaMalloc(&latencies_g, NUM_LOADS * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&dsink_g, config.TOTAL_THREADS * sizeof(uint32_t)));

  uint64_t *latencies = (uint64_t *)malloc(NUM_LOADS * sizeof(uint64_t));

  l2_lat_partition<<<config.BLOCKS_NUM, config.THREADS_PER_BLOCK>>>(
      latencies_g, pollute_buf_g, measure_buf_g, buf_elements, dsink_g);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  // Copy results back
  gpuErrchk(cudaMemcpy(latencies, latencies_g, NUM_LOADS * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));

  // Print per-load results
  uint64_t total = 0;
  printf("\n=== L2 Latency Partition Results ===\n");
  printf("Load#  Offset(B)  Latency(cycles)\n");
  for (int i = 0; i < NUM_LOADS; i++) {
    printf("%3d    %8d   %lu\n", i, i * STRIDE_BYTES, latencies[i]);
    total += latencies[i];
  }

  printf("\nAverage latency: %.1f cycles\n", (double)total / NUM_LOADS);
  printf("Min latency: %lu cycles\n",
         *std::min_element(latencies, latencies + NUM_LOADS));
  printf("Max latency: %lu cycles\n",
         *std::max_element(latencies, latencies + NUM_LOADS));

  free(latencies);
  gpuErrchk(cudaFree(pollute_buf_g));
  gpuErrchk(cudaFree(measure_buf_g));
  gpuErrchk(cudaFree(latencies_g));
  gpuErrchk(cudaFree(dsink_g));

  return 0;
}
