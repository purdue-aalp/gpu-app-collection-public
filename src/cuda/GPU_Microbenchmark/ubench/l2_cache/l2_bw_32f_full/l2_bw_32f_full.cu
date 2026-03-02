// L2 cache read bandwidth benchmark — full-L2 coverage, warp-per-cacheline.
//
// Design:
//   Launch: 2 CTAs per SM, 1024 threads/CTA  →  64 warps/SM.
//   Data:   floor(L2_SIZE / 128) cachelines, each 128 bytes (32 floats).
//   Each warp owns a strided subset of cachelines:
//     warp w, pass cl → posArray[(w + cl * total_warps) * 32 + lane_id]
//   All threads across all warps together cover the entire data array
//   exactly once per repeat, so every access hits L2 and nothing spills
//   to DRAM.
//
// Compile:
//   nvcc -Xptxas -dlcm=cg -Xptxas -dscm=wt l2_bw_32f_full.cu

#include <assert.h>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "../../../hw_def/hw_def.h"

__global__ void l2_bw_full(uint64_t *startClk, uint64_t *stopClk, float *dsink,
                            float *posArray, uint32_t cachelines_per_warp,
                            uint32_t total_warps, uint32_t repeat_times)
{
    uint32_t tid     = threadIdx.x;
    uint32_t bid     = blockIdx.x;
    uint32_t uid     = bid * blockDim.x + tid;
    uint32_t warp_id = uid / warpSize;
    uint32_t lane_id = uid % warpSize;

    float sink = 0.0f;

    // Warm up: bring every cacheline assigned to this warp into L2
    for (uint32_t cl = 0; cl < cachelines_per_warp; cl++) {
        float *ptr = posArray + (warp_id + cl * total_warps) * warpSize + lane_id;
        float tmp;
        asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(tmp) : "l"(ptr) : "memory");
        sink += tmp;
    }

    asm volatile("membar.gl;" ::: "memory");

    uint64_t start = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start) :: "memory");

    // Timed: repeat_times passes over all cachelines (full L2 per pass)
    for (uint32_t r = 0; r < repeat_times; r++) {
#pragma unroll 4
        for (uint32_t cl = 0; cl < cachelines_per_warp; cl++) {
            float *ptr = posArray + (warp_id + cl * total_warps) * warpSize + lane_id;
            float tmp;
            asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(tmp) : "l"(ptr) : "memory");
            sink += tmp;
        }
        asm volatile("membar.gl;" ::: "memory");
    }

    uint64_t stop = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stop) :: "memory");

    startClk[uid] = start;
    stopClk[uid]  = stop;
    dsink[uid]    = sink;
}

int main(int argc, char *argv[])
{
    initializeDeviceProp(0, argc, argv);

    uint32_t repeat_times = 32;

    // 2 CTAs per SM, 1024 threads per CTA
    config.THREADS_PER_BLOCK = 1024;
    config.BLOCKS_NUM        = config.SM_NUMBER * 2;
    config.TOTAL_THREADS     = config.THREADS_PER_BLOCK * config.BLOCKS_NUM;

    uint32_t warps_per_cta   = config.THREADS_PER_BLOCK / config.WARP_SIZE; // 32
    uint32_t total_warps     = config.BLOCKS_NUM * warps_per_cta;            // SM_NUM * 64

    // One cacheline = warpSize floats = 128 bytes
    uint32_t cacheline_floats    = config.WARP_SIZE;
    uint32_t total_cachelines    = (uint32_t)(config.L2_SIZE
                                              / (cacheline_floats * sizeof(float)));
    uint32_t cachelines_per_warp = total_cachelines / total_warps;

    // Actual array (may be slightly under L2_SIZE due to integer division)
    uint32_t ARRAY_SIZE = cachelines_per_warp * total_warps * cacheline_floats;

    std::cout << "total_warps          = " << total_warps                 << "\n";
    std::cout << "total_cachelines     = " << total_cachelines            << "\n";
    std::cout << "cachelines_per_warp  = " << cachelines_per_warp         << "\n";
    std::cout << "ARRAY_SIZE (floats)  = " << ARRAY_SIZE                  << "\n";
    std::cout << "ARRAY_SIZE (bytes)   = " << ARRAY_SIZE * sizeof(float)  << "\n";
    std::cout << "L2_SIZE    (bytes)   = " << config.L2_SIZE              << "\n";

    assert(cachelines_per_warp > 0);
    assert((size_t)ARRAY_SIZE * sizeof(float) <= config.L2_SIZE);

    uint64_t *startClk = (uint64_t *)malloc(config.TOTAL_THREADS * sizeof(uint64_t));
    uint64_t *stopClk  = (uint64_t *)malloc(config.TOTAL_THREADS * sizeof(uint64_t));
    float    *posArray = (float    *)malloc(ARRAY_SIZE            * sizeof(float));
    float    *dsink    = (float    *)malloc(config.TOTAL_THREADS  * sizeof(float));

    for (uint32_t i = 0; i < ARRAY_SIZE; i++)
        posArray[i] = (float)i;

    float    *posArray_g;
    float    *dsink_g;
    uint64_t *startClk_g;
    uint64_t *stopClk_g;

    gpuErrchk(cudaMalloc(&posArray_g, (size_t)ARRAY_SIZE           * sizeof(float)));
    gpuErrchk(cudaMalloc(&dsink_g,    (size_t)config.TOTAL_THREADS * sizeof(float)));
    gpuErrchk(cudaMalloc(&startClk_g, (size_t)config.TOTAL_THREADS * sizeof(uint64_t)));
    gpuErrchk(cudaMalloc(&stopClk_g,  (size_t)config.TOTAL_THREADS * sizeof(uint64_t)));

    gpuErrchk(cudaMemcpy(posArray_g, posArray,
                         (size_t)ARRAY_SIZE * sizeof(float),
                         cudaMemcpyHostToDevice));

    l2_bw_full<<<config.BLOCKS_NUM, config.THREADS_PER_BLOCK>>>(
        startClk_g, stopClk_g, dsink_g,
        posArray_g, cachelines_per_warp, total_warps, repeat_times);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(startClk, startClk_g,
                         (size_t)config.TOTAL_THREADS * sizeof(uint64_t),
                         cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(stopClk,  stopClk_g,
                         (size_t)config.TOTAL_THREADS * sizeof(uint64_t),
                         cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(dsink,    dsink_g,
                         (size_t)config.TOTAL_THREADS * sizeof(float),
                         cudaMemcpyDeviceToHost));

    // Total bytes read in timed section: full array × repeat_times
    unsigned long long data =
        (unsigned long long)ARRAY_SIZE * sizeof(float) * repeat_times;

    // Earliest start / latest stop across CTAs (sample first thread of each CTA)
    uint64_t min_start = startClk[0], max_stop = stopClk[0];
    for (uint32_t i = config.THREADS_PER_BLOCK; i < config.TOTAL_THREADS;
         i += config.THREADS_PER_BLOCK) {
        if (startClk[i] < min_start) min_start = startClk[i];
        if (stopClk[i]  > max_stop)  max_stop  = stopClk[i];
    }
    uint64_t total_time = max_stop - min_start;
    std::cout << "Total Clk number = " << total_time << "\n";

    float bw = (float)data / (float)total_time;
    float BW = bw * config.CLK_FREQUENCY * 1e6f / (1024.f * 1024.f * 1024.f);
    std::cout << "L2 bandwidth = " << bw << " (byte/clk), " << BW << " (GB/s)\n";

    float max_bw = (float)config.L2_BANKS * 64.f;
    float MaxBW  = max_bw * config.CLK_FREQUENCY * 1e6f / (1024.f * 1024.f * 1024.f);
    std::cout << "Max Theoretical L2 bandwidth = " << max_bw
              << " (byte/clk), " << MaxBW << " (GB/s)\n";
    std::cout << "L2 BW achievable = " << (bw / max_bw) * 100.f << "%\n";

    free(startClk); free(stopClk); free(posArray); free(dsink);
    cudaFree(posArray_g); cudaFree(dsink_g); cudaFree(startClk_g); cudaFree(stopClk_g);
    return 0;
}
