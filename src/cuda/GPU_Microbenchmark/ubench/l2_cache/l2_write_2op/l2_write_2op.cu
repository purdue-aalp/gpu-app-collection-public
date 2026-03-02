// L2 cache two-write benchmark
//
// Launches a single warp (32 threads). Each thread performs two 4-byte stores
// separated by 8192 bytes, then loads back from each address.
// All accesses use .cg to bypass L1.
// Output: per-thread latency for write 0, write 1, load 0, load 1.

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../../hw_def/hw_def.h"

#define WRITE_SEPARATION 8192

__global__ void l2_write_2op(uint32_t *buf, uint64_t *lat_w0, uint64_t *lat_w1,
                             uint64_t *lat_r0, uint64_t *lat_r1,
                             uint32_t *dsink) {
  uint32_t tid = threadIdx.x;

  // Each thread writes 4 bytes at stride = tid * 4
  // Write 0 at base, write 1 at base + 8192B
  uint32_t *ptr0 = buf + tid;
  uint32_t *ptr1 = (uint32_t *)((char *)buf + WRITE_SEPARATION) + tid;

  uint64_t start, stop;
  uint32_t val = tid + 1;
  uint32_t sink = 0;
  
  // one sector only 4*8B = 32
  if (tid < 8) {

  // Write 0
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  asm volatile("st.global.cg.u32 [%0], %1;" ::"l"(ptr0), "r"(val) : "memory");
  asm volatile("membar.gl;" ::: "memory");
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
//   lat_w0[tid] = stop - start;

  // Write 1
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  asm volatile("st.global.cg.u32 [%0], %1;" ::"l"(ptr1), "r"(val) : "memory");
  asm volatile("membar.gl;" ::: "memory");
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
//   lat_w1[tid] = stop - start;

  // Load 0 (read back from ptr0, .cg bypasses L1)
  uint32_t data;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(data) : "l"(ptr0) : "memory");
  asm volatile("membar.gl;" ::: "memory");
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
//   lat_r0[tid] = stop - start;
  sink += data;

  // Load 1 (read back from ptr1, .cg bypasses L1)
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(data) : "l"(ptr1) : "memory");
  asm volatile("membar.gl;" ::: "memory");
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
//   lat_r1[tid] = stop - start;
  sink += data;

  dsink[tid] = sink;
  }
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

  uint64_t *lat_w0_g, *lat_w1_g, *lat_r0_g, *lat_r1_g;
  uint32_t *dsink_g;
  gpuErrchk(cudaMalloc(&lat_w0_g, 32 * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&lat_w1_g, 32 * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&lat_r0_g, 32 * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&lat_r1_g, 32 * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&dsink_g, 32 * sizeof(uint32_t)));

  // Launch exactly 1 warp
  l2_write_2op<<<1, 32>>>(buf_g, lat_w0_g, lat_w1_g, lat_r0_g, lat_r1_g, dsink_g);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  // Copy results back
  uint64_t lat_w0[32], lat_w1[32], lat_r0[32], lat_r1[32];
  gpuErrchk(
      cudaMemcpy(lat_w0, lat_w0_g, 32 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  gpuErrchk(
      cudaMemcpy(lat_w1, lat_w1_g, 32 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  gpuErrchk(
      cudaMemcpy(lat_r0, lat_r0_g, 32 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  gpuErrchk(
      cudaMemcpy(lat_r1, lat_r1_g, 32 * sizeof(uint64_t), cudaMemcpyDeviceToHost));

  // Print results
  printf("write0: %.1f cycles\n", (double)lat_w0[0]);
  printf("write1: %.1f cycles\n", (double)lat_w1[0]);
  printf("load0:  %.1f cycles\n", (double)lat_r0[0]);
  printf("load1:  %.1f cycles\n", (double)lat_r1[0]);

  gpuErrchk(cudaFree(buf_g));
  gpuErrchk(cudaFree(lat_w0_g));
  gpuErrchk(cudaFree(lat_w1_g));
  gpuErrchk(cudaFree(lat_r0_g));
  gpuErrchk(cudaFree(lat_r1_g));
  gpuErrchk(cudaFree(dsink_g));

  return 0;
}
