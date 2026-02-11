// L2 cache latency partition benchmark
//
// Measures per-load L2 miss latency from each SM.
// For each target SM:
//   1. All blocks (1 per SM) pollute L2 with pollute_buf, evicting measure_buf
//   2. bar.sync, then only blockIdx.x == target_sm's first warp measures
//   3. 256 loads at 128B stride, membar.gl serializes each load
// Output: CSV with rows = load index, columns = SM

#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../../hw_def/hw_def.h"

#define NUM_LOADS 256
#define STRIDE_BYTES 128

__global__ void l2_lat_partition(uint64_t *latencies, uint32_t *pollute_buf,
                                 uint32_t *measure_buf, size_t pollute_elements,
                                 uint32_t target_sm, uint32_t *dsink) {
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

  // Ensure all threads in this block finished polluting
  asm volatile("bar.sync 0;");

  // Phase 2: First warp of target SM block measures latency
  if (blockIdx.x == target_sm && threadIdx.x < warpSize) {
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

  // Parse --fast flag: only measure SM 0
  uint32_t fast_mode = 0;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--fast") == 0) {
      fast_mode = 1;
      break;
    }
  }

  size_t l2_size = config.L2_SIZE;
  size_t buf_elements = l2_size / sizeof(uint32_t);
  uint32_t num_sms = fast_mode ? 1 : config.SM_NUMBER;
  uint32_t blocks = config.SM_NUMBER; // always 1 block per SM for pollution

  printf("\nL2 Cache Size: %zu bytes\n", l2_size);
  printf("Stride: %d bytes\n", STRIDE_BYTES);
  printf("Number of loads: %d\n", NUM_LOADS);
  printf("Total measured region: %d bytes\n", NUM_LOADS * STRIDE_BYTES);
  printf("Number of SMs: %u\n", num_sms);

  // Allocate device buffers
  uint32_t *pollute_buf_g, *measure_buf_g;
  gpuErrchk(cudaMalloc(&pollute_buf_g, l2_size));
  gpuErrchk(cudaMalloc(&measure_buf_g, l2_size));
  gpuErrchk(cudaMemset(pollute_buf_g, 1, l2_size));
  gpuErrchk(cudaMemset(measure_buf_g, 2, l2_size));

  uint64_t *latencies_g;
  uint32_t *dsink_g;
  gpuErrchk(cudaMalloc(&latencies_g, NUM_LOADS * sizeof(uint64_t)));
  gpuErrchk(
      cudaMalloc(&dsink_g, blocks * config.THREADS_PER_BLOCK * sizeof(uint32_t)));

  // Host storage: num_sms columns x NUM_LOADS rows
  uint64_t *all_latencies =
      (uint64_t *)malloc(num_sms * NUM_LOADS * sizeof(uint64_t));
  uint64_t *latencies = (uint64_t *)malloc(NUM_LOADS * sizeof(uint64_t));

  // Launch once per SM
  for (uint32_t sm = 0; sm < num_sms; sm++) {
    l2_lat_partition<<<blocks, config.THREADS_PER_BLOCK>>>(
        latencies_g, pollute_buf_g, measure_buf_g, buf_elements, sm, dsink_g);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(latencies, latencies_g,
                          NUM_LOADS * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < NUM_LOADS; i++)
      all_latencies[sm * NUM_LOADS + i] = latencies[i];
  }

  // CSV header
  printf("\nLoad#,Offset(B)");
  for (uint32_t s = 0; s < num_sms; s++)
    printf(",SM%u", s);
  printf("\n");

  // CSV rows
  for (int i = 0; i < NUM_LOADS; i++) {
    printf("%d,%d", i, i * STRIDE_BYTES);
    for (uint32_t s = 0; s < num_sms; s++)
      printf(",%lu", all_latencies[s * NUM_LOADS + i]);
    printf("\n");
  }

  // Per-SM summary
  printf("\nPer-SM average latency:\n");
  for (uint32_t s = 0; s < num_sms; s++) {
    uint64_t total = 0;
    for (int i = 0; i < NUM_LOADS; i++)
      total += all_latencies[s * NUM_LOADS + i];
    printf("SM%u: %.1f cycles\n", s, (double)total / NUM_LOADS);
  }

  free(all_latencies);
  free(latencies);
  gpuErrchk(cudaFree(pollute_buf_g));
  gpuErrchk(cudaFree(measure_buf_g));
  gpuErrchk(cudaFree(latencies_g));
  gpuErrchk(cudaFree(dsink_g));

  return 0;
}
