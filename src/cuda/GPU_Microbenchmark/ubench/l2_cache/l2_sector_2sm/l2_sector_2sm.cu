// L2 cache sector load benchmark: two SMs, one 32-byte sector
//
// Each participating SM launches 8 active threads (first 8 of a 32-thread warp).
// Each thread loads 4B at buf[tid], so 8 threads cover one 32B L2 sector.
// Threads 8-31 are idle.
//
// Two target SMs are selected via --sm_a <id> and --sm_b <id> (default 0 and 1).
// All other SMs return immediately.
//
// A device-side barrier synchronizes the two active SMs so they hit L2 together,
// letting you observe contention / sharing effects.
//
// Uses ld.global.cg to bypass L1.
// Output: per-thread load latency (cycles) from SM A and SM B.

#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../../hw_def/hw_def.h"

#define ACTIVE_THREADS 8  // first 8 threads of each warp → 8 × 4B = 32B = 1 sector

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__global__ void l2_sector_2sm(uint32_t *buf,
                               uint64_t *lat_a,   // [ACTIVE_THREADS], SM A latencies
                               uint64_t *lat_b,   // [ACTIVE_THREADS], SM B latencies
                               uint32_t sm_a, uint32_t sm_b,
                               int *barrier_cnt,  // global atomic barrier (init = 0)
                               uint32_t *dsink) {
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));

  if (smid != sm_a && smid != sm_b) return;

  uint32_t tid = threadIdx.x;
  uint32_t sink = 0;

  // ----- device-side barrier: both active SMs must arrive before loading -----
  // Only thread 0 of each block participates in the counter.
  if (tid == 0) {
    atomicAdd(barrier_cnt, 1);
    // Spin until both SMs have incremented.
    while (atomicAdd(barrier_cnt, 0) < 2) { /* spin */ }
  }
  // Broadcast: all threads in the warp wait for thread 0.
  __syncthreads();

  // ----- load: only first ACTIVE_THREADS threads -----
  if (tid < ACTIVE_THREADS) {
    uint32_t *ptr = buf + tid;  // thread t → word t → byte offset t*4 within sector
    uint32_t data;
    uint64_t start, stop;

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start) :: "memory");
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(data) : "l"(ptr) : "memory");
    asm volatile("membar.gl;" ::: "memory");
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop) :: "memory");

    sink = data;
    uint64_t lat = stop - start;

    if (smid == sm_a)
      lat_a[tid] = lat;
    else
      lat_b[tid] = lat;
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

  printf("\nL2 Sector 2-SM Load Benchmark\n");
  printf("Active threads per block  : %d (threads 0..%d of warp)\n",
         ACTIVE_THREADS, ACTIVE_THREADS - 1);
  printf("Bytes loaded per SM       : %d (%d threads × 4B = 1 sector)\n",
         ACTIVE_THREADS * 4, ACTIVE_THREADS);
  printf("SM A                      : %u\n", sm_a);
  printf("SM B                      : %u\n", sm_b);
  printf("Total SMs launched        : %u (others return immediately)\n", num_sms);

  // 32B-aligned buffer covering one sector (8 uint32s)
  uint32_t *buf_g;
  gpuErrchk(cudaMalloc(&buf_g, ACTIVE_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMemset(buf_g, 1, ACTIVE_THREADS * sizeof(uint32_t)));

  // Latency outputs
  uint64_t *lat_a_g, *lat_b_g;
  gpuErrchk(cudaMalloc(&lat_a_g, ACTIVE_THREADS * sizeof(uint64_t)));
  gpuErrchk(cudaMalloc(&lat_b_g, ACTIVE_THREADS * sizeof(uint64_t)));
  gpuErrchk(cudaMemset(lat_a_g, 0, ACTIVE_THREADS * sizeof(uint64_t)));
  gpuErrchk(cudaMemset(lat_b_g, 0, ACTIVE_THREADS * sizeof(uint64_t)));

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
  uint64_t lat_a[ACTIVE_THREADS], lat_b[ACTIVE_THREADS];
  gpuErrchk(cudaMemcpy(lat_a, lat_a_g, ACTIVE_THREADS * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(lat_b, lat_b_g, ACTIVE_THREADS * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));

  // Print results
  printf("\nThread,SM%u_lat(cycles),SM%u_lat(cycles)\n", sm_a, sm_b);
  uint64_t sum_a = 0, sum_b = 0;
  for (int t = 0; t < ACTIVE_THREADS; t++) {
    printf("%d,%lu,%lu\n", t, lat_a[t], lat_b[t]);
    sum_a += lat_a[t];
    sum_b += lat_b[t];
  }
  printf("\nAvg SM%u : %.1f cycles\n", sm_a, (double)sum_a / ACTIVE_THREADS);
  printf("Avg SM%u : %.1f cycles\n", sm_b, (double)sum_b / ACTIVE_THREADS);

  gpuErrchk(cudaFree(buf_g));
  gpuErrchk(cudaFree(lat_a_g));
  gpuErrchk(cudaFree(lat_b_g));
  gpuErrchk(cudaFree(barrier_g));
  gpuErrchk(cudaFree(dsink_g));

  return 0;
}
