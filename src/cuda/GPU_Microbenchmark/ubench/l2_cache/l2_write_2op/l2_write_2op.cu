// L2 cache two-write benchmark
//
// Launches a single warp (32 threads). Each thread performs two 4-byte stores
// separated by 8192 bytes. Measures the latency of each write.
// Output: per-thread latency for write 0 and write 1.

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../../hw_def/hw_def.h"

#define WRITE_SEPARATION 8192

__global__ void l2_write_2op(uint32_t *buf, uint64_t *lat_w0, uint64_t *lat_w1,
                             uint32_t *dsink) {
  uint32_t tid = threadIdx.x;

  // Each thread writes 4 bytes at stride = tid * 4
  // Write 0 at base, write 1 at base + 8192B
  uint32_t *ptr0 = buf + tid;
  uint32_t *ptr1 = (uint32_t *)((char *)buf + WRITE_SEPARATION) + tid;

  uint64_t start, stop;
  uint32_t val = tid + 1;

  // Write 0
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  asm volatile("st.global.cg.u32 [%0], %1;" ::"l"(ptr0), "r"(val) : "memory");
  asm volatile("membar.gl;" ::: "memory");
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
  lat_w0[tid] = stop - start;

  // Write 1
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  asm volatile("st.global.cg.u32 [%0], %1;" ::"l"(ptr1), "r"(val) : "memory");
  asm volatile("membar.gl;" ::: "memory");
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
  lat_w1[tid] = stop - start;

  dsink[tid] = val;
}

int main(int argc, char *argv[]) {
  initializeDeviceProp(0, argc, argv);

  printf("\nL2 Two-Write Benchmark\n");
  printf("Warp size: 32 threads, 1 block\n");
  printf("Write separation: %d bytes\n", WRITE_SEPARATION);

  // Allocate device buffers
  size_t buf_size = WRITE_SEPARATION + 32 * sizeof(uint32_t);
  uint32_t *buf_g;
  gpuErrchk(cudaMalloc(&buf_g, buf_size));
  gpuErrchk(cudaMemset(buf_g, 0, buf_size));

  uint64_t *lat_w0_g, *lat_w1_g;
  uint32_t *dsink_g;
  gpuErrchk(cudaMalloc(&lat_w0_g, 32 * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&lat_w1_g, 32 * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&dsink_g, 32 * sizeof(uint32_t)));

  // Launch exactly 1 warp
  l2_write_2op<<<1, 32>>>(buf_g, lat_w0_g, lat_w1_g, dsink_g);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  // Copy results back
  uint64_t lat_w0[32], lat_w1[32];
  gpuErrchk(
      cudaMemcpy(lat_w0, lat_w0_g, 32 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  gpuErrchk(
      cudaMemcpy(lat_w1, lat_w1_g, 32 * sizeof(uint64_t), cudaMemcpyDeviceToHost));

  // Print results
  printf("\nThread,Write0_lat,Write1_lat\n");
  uint64_t sum_w0 = 0, sum_w1 = 0;
  for (int t = 0; t < 32; t++) {
    printf("%2d,%lu,%lu\n", t, lat_w0[t], lat_w1[t]);
    sum_w0 += lat_w0[t];
    sum_w1 += lat_w1[t];
  }
  printf("\nAvg write0: %.1f cycles\n", (double)sum_w0 / 32);
  printf("Avg write1: %.1f cycles\n", (double)sum_w1 / 32);

  gpuErrchk(cudaFree(buf_g));
  gpuErrchk(cudaFree(lat_w0_g));
  gpuErrchk(cudaFree(lat_w1_g));
  gpuErrchk(cudaFree(dsink_g));

  return 0;
}
