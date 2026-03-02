// L2 cache sector benchmark: SM A writes, SM B reads (ordered by atomic), two rounds
//
// Round 1: SM A writes to buf+tid,        SM B reads from buf+tid
// Round 2: SM A writes to buf+tid+8192,   SM B reads from buf+tid+8192
//
// 4-phase atomic barrier enforces strict ordering across rounds:
//   SM A writes R1 → (barrier=1) → SM B reads R1 → (barrier=2)
//   → SM A writes R2 → (barrier=3) → SM B reads R2
//
// Both SMs bypass L1 (cg cache operator), targeting L2.
// Output: per-thread write/read latency for each round in cycles.

#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../../hw_def/hw_def.h"

#define ACTIVE_THREADS 8  // first 8 threads of each warp → 8 × 4B = 32B = 1 sector

// ---------------------------------------------------------------------------
// Kernel
// lat_a[0..ACTIVE_THREADS-1]              : SM A write latencies, round 1
// lat_a[ACTIVE_THREADS..2*ACTIVE_THREADS-1]: SM A write latencies, round 2
// lat_b[0..ACTIVE_THREADS-1]              : SM B read  latencies, round 1
// lat_b[ACTIVE_THREADS..2*ACTIVE_THREADS-1]: SM B read  latencies, round 2
// ---------------------------------------------------------------------------
__global__ void l2_sector_2sm(uint32_t *buf,
                               uint64_t *lat_a,
                               uint64_t *lat_b,
                               uint32_t sm_a, uint32_t sm_b,
                               int *barrier_cnt,  // atomic ordering flag (init = 0)
                               uint32_t *dsink) {
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));

  if (smid != sm_a && smid != sm_b) return;

  uint32_t tid = threadIdx.x;
  uint32_t sink = 0;

  if (smid == sm_a) {

    // ---- Round 1: write buf+tid ----
    // if (tid < ACTIVE_THREADS) {
    //   uint32_t *ptr = buf + tid;
    //   uint64_t start, stop;
    //   asm volatile("mov.u64 %0, %%clock64;" : "=l"(start) :: "memory");
    //   asm volatile("st.global.cg.u32 [%0], %1;"
    //                :: "l"(ptr), "r"(tid + 1)
    //                : "memory");
    //   asm volatile("membar.gl;" ::: "memory");
    //   asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop) :: "memory");
    //   // lat_a[tid] = stop - start;
    // }
    if (tid < ACTIVE_THREADS) {
      uint32_t *ptr = buf + tid + 8192;
      uint32_t data;
      uint64_t start, stop;
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(start) :: "memory");
      asm volatile("ld.global.cg.u32 %0, [%1];"
                   : "=r"(data)
                   : "l"(ptr)
                   : "memory");
      asm volatile("membar.gl;" ::: "memory");
      sink += data;
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop) :: "memory");
      // lat_b[ACTIVE_THREADS + tid] = stop - start;
    }
    // Signal round 1 writes done; wait for SM B to finish reading round 1
    __syncthreads();
    if (tid == 0) {
      asm volatile("membar.gl;" ::: "memory");
      atomicAdd(barrier_cnt, 1);                          // barrier → 1
      while (atomicAdd(barrier_cnt, 0) < 2) { /* spin */ } // wait for SM B round 1 done
    }
    __syncthreads();

    // ---- Round 2: write buf+tid+8192 ----
    // if (tid < ACTIVE_THREADS) {
    //   uint32_t *ptr = buf + tid + 8192;
    //   uint64_t start, stop;
    //   asm volatile("mov.u64 %0, %%clock64;" : "=l"(start) :: "memory");
    //   asm volatile("st.global.cg.u32 [%0], %1;"
    //                :: "l"(ptr), "r"(tid + 1)
    //                : "memory");
    //   asm volatile("membar.gl;" ::: "memory");
    //   asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop) :: "memory");
    //   // lat_a[ACTIVE_THREADS + tid] = stop - start;
    // }
    // Signal round 2 writes done
    __syncthreads();
    if (tid == 0) {
      asm volatile("membar.gl;" ::: "memory");
      atomicAdd(barrier_cnt, 1);                          // barrier → 3
    }

  } else {  // smid == sm_b

    // ---- Wait for SM A round 1 writes ----
    if (tid == 0) {
      while (atomicAdd(barrier_cnt, 0) < 1) { /* spin */ }
    }
    __syncthreads();
    asm volatile("membar.gl;" ::: "memory");

    // ---- Round 1: read buf+tid ----
    if (tid < ACTIVE_THREADS) {
      uint32_t *ptr = buf + tid;
      uint32_t data;
      uint64_t start, stop;
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(start) :: "memory");
      asm volatile("ld.global.cg.u32 %0, [%1];"
                   : "=r"(data)
                   : "l"(ptr)
                   : "memory");
      asm volatile("membar.gl;" ::: "memory");
      sink += data;
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop) :: "memory");
      // lat_b[tid] = stop - start;
    }
    // Signal round 1 reads done; wait for SM A to finish writing round 2
    __syncthreads();
    if (tid == 0) {
      atomicAdd(barrier_cnt, 1);                          // barrier → 2
      while (atomicAdd(barrier_cnt, 0) < 3) { /* spin */ } // wait for SM A round 2 done
    }
    __syncthreads();
    asm volatile("membar.gl;" ::: "memory");

    // ---- Round 2: read buf+tid+8192 ----
    if (tid < ACTIVE_THREADS) {
      uint32_t *ptr = buf + tid + 8192;
      uint32_t data;
      uint64_t start, stop;
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(start) :: "memory");
      asm volatile("ld.global.cg.u32 %0, [%1];"
                   : "=r"(data)
                   : "l"(ptr)
                   : "memory");
      asm volatile("membar.gl;" ::: "memory");
      sink += data;
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop) :: "memory");
      // lat_b[ACTIVE_THREADS + tid] = stop - start;
    }
  }

  if (tid == 0)
    dsink[smid] = sink;
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
  initializeDeviceProp(0, argc, argv);

  // Parse --sm_a and --sm_b
  uint32_t sm_a = 0, sm_b = 1;
  for (int i = 1; i < argc - 1; i++) {
    if (strcmp(argv[i], "--sm_a") == 0) sm_a = (uint32_t)atoi(argv[i + 1]);
    if (strcmp(argv[i], "--sm_b") == 0) sm_b = (uint32_t)atoi(argv[i + 1]);
  }

  uint32_t num_sms = config.SM_NUMBER;
  if (sm_a >= num_sms || sm_b >= num_sms || sm_a == sm_b) {
    fprintf(stderr,
            "Error: sm_a=%u and sm_b=%u must be distinct and < %u (SM count)\n",
            sm_a, sm_b, num_sms);
    return 1;
  }

  printf("\nL2 Sector 2-SM Write/Read Benchmark (2 rounds)\n");
  printf("Active threads per block  : %d (threads 0..%d of warp)\n",
         ACTIVE_THREADS, ACTIVE_THREADS - 1);
  printf("Bytes per SM              : %d (%d threads × 4B = 1 sector)\n",
         ACTIVE_THREADS * 4, ACTIVE_THREADS);
  printf("SM A (writer)             : %u\n", sm_a);
  printf("SM B (reader)             : %u\n", sm_b);
  printf("Round 1 offset            : buf+tid\n");
  printf("Round 2 offset            : buf+tid+8192\n");
  printf("Total SMs launched        : %u (others return immediately)\n", num_sms);

  // Buffer: round 1 uses [0..7], round 2 uses [8192..8199]
  uint32_t *buf_g;
  gpuErrchk(cudaMalloc(&buf_g, (8192 + ACTIVE_THREADS) * sizeof(uint32_t)));
  gpuErrchk(cudaMemset(buf_g, 0, (8192 + ACTIVE_THREADS) * sizeof(uint32_t)));

  // Latency outputs: 2 rounds × ACTIVE_THREADS each
  uint64_t *lat_a_g, *lat_b_g;
  gpuErrchk(cudaMalloc(&lat_a_g, 2 * ACTIVE_THREADS * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&lat_b_g, 2 * ACTIVE_THREADS * sizeof(uint64_t)));
  gpuErrchk(cudaMemset(lat_a_g, 0, 2 * ACTIVE_THREADS * sizeof(uint64_t)));
  gpuErrchk(cudaMemset(lat_b_g, 0, 2 * ACTIVE_THREADS * sizeof(uint64_t)));

  // Barrier counter and sink
  int *barrier_g;
  gpuErrchk(cudaMalloc(&barrier_g, sizeof(int)));
  gpuErrchk(cudaMemset(barrier_g, 0, sizeof(int)));

  uint32_t *dsink_g;
  gpuErrchk(cudaMalloc(&dsink_g, num_sms * sizeof(uint32_t)));

  // Launch one warp per SM; only sm_a and sm_b do the measurement
  l2_sector_2sm<<<num_sms, 32>>>(buf_g, lat_a_g, lat_b_g,
                                  sm_a, sm_b, barrier_g, dsink_g);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  // Copy results
  uint64_t lat_a[2 * ACTIVE_THREADS], lat_b[2 * ACTIVE_THREADS];
  gpuErrchk(cudaMemcpy(lat_a, lat_a_g, 2 * ACTIVE_THREADS * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(lat_b, lat_b_g, 2 * ACTIVE_THREADS * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));

  // Print results — Round 1
  printf("\n[Round 1] ptr = buf + tid\n");
  printf("Thread,SM%u_write_lat(cycles),SM%u_read_lat(cycles)\n", sm_a, sm_b);
  uint64_t sum_a = 0, sum_b = 0;
  for (int t = 0; t < ACTIVE_THREADS; t++) {
    printf("%d,%lu,%lu\n", t, lat_a[t], lat_b[t]);
    sum_a += lat_a[t];
    sum_b += lat_b[t];
  }
  printf("Avg SM%u write : %.1f cycles\n", sm_a, (double)sum_a / ACTIVE_THREADS);
  printf("Avg SM%u read  : %.1f cycles\n", sm_b, (double)sum_b / ACTIVE_THREADS);

  // Print results — Round 2
  printf("\n[Round 2] ptr = buf + tid + 8192\n");
  printf("Thread,SM%u_write_lat(cycles),SM%u_read_lat(cycles)\n", sm_a, sm_b);
  sum_a = 0; sum_b = 0;
  for (int t = 0; t < ACTIVE_THREADS; t++) {
    printf("%d,%lu,%lu\n", t, lat_a[ACTIVE_THREADS + t], lat_b[ACTIVE_THREADS + t]);
    sum_a += lat_a[ACTIVE_THREADS + t];
    sum_b += lat_b[ACTIVE_THREADS + t];
  }
  printf("Avg SM%u write : %.1f cycles\n", sm_a, (double)sum_a / ACTIVE_THREADS);
  printf("Avg SM%u read  : %.1f cycles\n", sm_b, (double)sum_b / ACTIVE_THREADS);

  gpuErrchk(cudaFree(buf_g));
  gpuErrchk(cudaFree(lat_a_g));
  gpuErrchk(cudaFree(lat_b_g));
  gpuErrchk(cudaFree(barrier_g));
  gpuErrchk(cudaFree(dsink_g));

  return 0;
}
