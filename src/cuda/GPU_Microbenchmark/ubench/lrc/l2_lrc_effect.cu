// L2 Request Coalescer (LRC) Microbenchmark for H100
// This benchmark investigates LRC coalescing behavior:
// 1. How many requests are coalesced?
// 2. What is the period/interval that LRC coalesces requests?
// 3. What is the condition for coalescing? (same address? same cacheline?)

// Compile with: nvcc -Xptxas -dlcm=cg -Xptxas -dscm=wt l2_lrc_effect.cu -o l2_lrc_effect.o
// Run with NSight Compute to measure LRC metrics:
// ncu --metrics lrc__lts2lrc_sectors_op_read.sum,lrc__xbar2gpc_sectors_op_read.sum,lrc__average_xbar2gpc_sectors_op_read.ratio ./l2_lrc_effect.o [test_mode] [num_warps] [offset_bytes] [delay_cycles]

// Test modes:
// 0: Same address test (all warps access same address)
// 1: Same cacheline test (all warps access same cacheline, different offsets)
// 2: Different cachelines test (all warps access different cachelines)
// 3: Temporal coalescing test (staggered accesses with delays)

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
#include <iomanip>

#ifdef TUNER
#include "../../../../hw_def/hw_def.h"
#else
#include "../../../../hw_def/common/gpuConfig.h"
#define CLK_FREQUENCY 1980 // H100 frequency 
#endif

#define L2_LINE_SIZE 128 // bytes
#define WARP_SIZE 32

// Test mode enum
enum TestMode {
    SAME_ADDRESS = 0,      // All warps access the exact same address
    SAME_CACHELINE = 1,    // All warps access same cacheline, different offsets
    DIFF_CACHELINE = 2,    // All warps access different cachelines
    TEMPORAL_COALESCE = 3  // Staggered accesses to test temporal coalescing
};

__global__ void lrc_coalesce_test(
    unsigned long long *startClk,
    unsigned long long *stopClk,
    float *dsink,
    float *posArray,
    unsigned ARRAY_SIZE,
    unsigned num_warps,
    unsigned test_mode,
    unsigned offset_bytes,
    unsigned delay_cycles
) {
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t uid = bid * blockDim.x + tid;
    uint32_t warp_id = uid / WARP_SIZE;
    uint32_t lane_id = uid % WARP_SIZE;
    
    float sink = 0;
    
    // Synchronize all threads
    asm volatile("bar.sync 0;");
    
    unsigned long long start = clock64();
    
    if (warp_id < num_warps) {
        float *ptr = nullptr;
        unsigned long long access_time = 0;
        
        switch (test_mode) {
            case SAME_ADDRESS: {
                // All warps access the exact same address
                ptr = posArray;
                break;
            }
            
            case SAME_CACHELINE: {
                // All warps access same cacheline with different byte offsets
                // offset_bytes should be < L2_LINE_SIZE
                uint32_t byte_offset = (warp_id * offset_bytes) % L2_LINE_SIZE;
                ptr = (float*)((char*)posArray + byte_offset);
                break;
            }
            
            case DIFF_CACHELINE: {
                // Each warp accesses a different cacheline
                uint32_t cacheline_offset = warp_id * (L2_LINE_SIZE / sizeof(float));
                ptr = posArray + cacheline_offset;
                break;
            }
            
            case TEMPORAL_COALESCE: {
                // Staggered accesses: each warp delays by delay_cycles before accessing
                // All access the same address to test temporal coalescing
                unsigned long long delay_start = clock64();
                unsigned long long target_time = delay_start + (warp_id * delay_cycles);
                
                // Busy wait until target time
                while (clock64() < target_time) {
                    // Spin
                }
                
                ptr = posArray;
                break;
            }
        }
        
        // Perform the memory access
        unsigned long long t0 = clock64();
        asm volatile("{\t\n"
            ".reg .f32 data;\n\t"
            "ld.global.cg.f32 data, [%1];\n\t"
            "add.f32 %0, data, %0;\n\t"
            "}" : "+f"(sink) : "l"(ptr) : "memory"
        );
        unsigned long long t1 = clock64();
        access_time = t1 - t0;
    }
    
    asm volatile("bar.sync 0;");
    
    unsigned long long stop = clock64();
    
    // Store timing results
    if (lane_id == 0 && warp_id < num_warps) {
        startClk[warp_id] = start;
        stopClk[warp_id] = stop;
    }
    dsink[uid] = sink;
}

// Variant for testing maximum coalescing capacity
__global__ void lrc_coalesce_capacity_test(
    unsigned long long *startClk,
    unsigned long long *stopClk,
    float *dsink,
    float *posArray,
    unsigned ARRAY_SIZE,
    unsigned num_warps,
    unsigned num_accesses_per_warp
) {
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t uid = bid * blockDim.x + tid;
    uint32_t warp_id = uid / WARP_SIZE;
    uint32_t lane_id = uid % WARP_SIZE;
    
    float sink = 0;
    
    asm volatile("bar.sync 0;");
    unsigned long long start = clock64();
    
    if (warp_id < num_warps) {
        // Each warp makes multiple accesses to the same address
        // This tests how many requests can be coalesced
        float *ptr = posArray; // Same address for all
        
        for (unsigned i = 0; i < num_accesses_per_warp; i++) {
            asm volatile("{\t\n"
                ".reg .f32 data;\n\t"
                "ld.global.cg.f32 data, [%1];\n\t"
                "add.f32 %0, data, %0;\n\t"
                "}" : "+f"(sink) : "l"(ptr) : "memory"
            );
        }
    }
    
    asm volatile("bar.sync 0;");
    unsigned long long stop = clock64();
    
    if (lane_id == 0 && warp_id < num_warps) {
        startClk[warp_id] = start;
        stopClk[warp_id] = stop;
    }
    dsink[uid] = sink;
}

int main(int argc, char *argv[]) {
    intilizeDeviceProp(0, argc, argv);
    
    // Default parameters
    unsigned test_mode = 0;  // SAME_ADDRESS
    unsigned num_warps = 32;
    unsigned offset_bytes = 4;  // 4 bytes (1 float) offset for same cacheline test
    unsigned delay_cycles = 10; // Delay in cycles for temporal test
    unsigned capacity_test = 0; // 0 = normal test, 1 = capacity test
    unsigned num_accesses_per_warp = 1;
    
    // Parse command line arguments
    if (argc > 1) test_mode = atoi(argv[1]);
    if (argc > 2) num_warps = atoi(argv[2]);
    if (argc > 3) offset_bytes = atoi(argv[3]);
    if (argc > 4) delay_cycles = atoi(argv[4]);
    if (argc > 5) capacity_test = atoi(argv[5]);
    if (argc > 6) num_accesses_per_warp = atoi(argv[6]);
    
    printf("=== L2 Request Coalescer (LRC) Microbenchmark ===\n");
    printf("Test Mode: %u ", test_mode);
    switch (test_mode) {
        case SAME_ADDRESS: printf("(Same Address)\n"); break;
        case SAME_CACHELINE: printf("(Same Cacheline)\n"); break;
        case DIFF_CACHELINE: printf("(Different Cachelines)\n"); break;
        case TEMPORAL_COALESCE: printf("(Temporal Coalescing)\n"); break;
        default: printf("(Unknown)\n"); break;
    }
    printf("Number of warps: %u\n", num_warps);
    printf("Offset bytes: %u\n", offset_bytes);
    printf("Delay cycles: %u\n", delay_cycles);
    printf("Capacity test: %u\n", capacity_test);
    if (capacity_test) printf("Accesses per warp: %u\n", num_accesses_per_warp);
    
    // Calculate array size needed
    // uint64_t threads_per_block = config.THREADS_PER_BLOCK;
    uint64_t total_threads = num_warps * WARP_SIZE;
    uint64_t threads_per_block = 1024; // Assuming max 1024 threads per block
    uint64_t blocks_launched = 1;
    if (total_threads < threads_per_block) {
        threads_per_block = total_threads;
    }
    else{
        blocks_launched = (total_threads + threads_per_block - 1) / threads_per_block;
        threads_per_block = 1024;
    }    
    
    // Ensure we have enough warps
    uint32_t actual_num_warps = (total_threads + WARP_SIZE - 1) / WARP_SIZE;
    if (num_warps > actual_num_warps) {
        printf("Warning: Requested %u warps but only %u available. Using %u.\n", 
               num_warps, actual_num_warps, actual_num_warps);
        num_warps = actual_num_warps;
    }
    
    // Array size: need enough space for different cacheline accesses
    unsigned ARRAY_SIZE = (num_warps + 1) * (L2_LINE_SIZE / sizeof(float)) + 1024;
    
    // Allocate host memory
    unsigned long long *startClk = (unsigned long long*)malloc(num_warps * sizeof(unsigned long long));
    unsigned long long *stopClk = (unsigned long long*)malloc(num_warps * sizeof(unsigned long long));
    float *posArray = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float *dsink = (float*)malloc(total_threads * sizeof(float));
    
    // Initialize array
    for (unsigned i = 0; i < ARRAY_SIZE; i++) {
        posArray[i] = (float)i;
    }
    
    // Allocate device memory
    float *posArray_g;
    float *dsink_g;
    unsigned long long *startClk_g;
    unsigned long long *stopClk_g;
    
    gpuErrchk(cudaMalloc(&posArray_g, ARRAY_SIZE * sizeof(float)));
    gpuErrchk(cudaMalloc(&dsink_g, total_threads * sizeof(float)));
    gpuErrchk(cudaMalloc(&startClk_g, num_warps * sizeof(unsigned long long)));
    gpuErrchk(cudaMalloc(&stopClk_g, num_warps * sizeof(unsigned long long)));
    
    gpuErrchk(cudaMemcpy(posArray_g, posArray, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    if (capacity_test) {
        lrc_coalesce_capacity_test<<<blocks_launched, threads_per_block>>>(
            startClk_g, stopClk_g, dsink_g, posArray_g, ARRAY_SIZE,
            num_warps, num_accesses_per_warp
        );
    } else {
        lrc_coalesce_test<<<blocks_launched, threads_per_block>>>(
            startClk_g, stopClk_g, dsink_g, posArray_g, ARRAY_SIZE,
            num_warps, test_mode, offset_bytes, delay_cycles
        );
    }
    
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Copy results back
    gpuErrchk(cudaMemcpy(startClk, startClk_g, num_warps * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(stopClk, stopClk_g, num_warps * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(dsink, dsink_g, total_threads * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Analyze results
    unsigned long long min_start = startClk[0];
    unsigned long long max_start = startClk[0];
    unsigned long long min_stop = stopClk[0];
    unsigned long long max_stop = stopClk[0];
    
    for (unsigned i = 0; i < num_warps; i++) {
        if (startClk[i] < min_start) min_start = startClk[i];
        if (startClk[i] > max_start) max_start = startClk[i];
        if (stopClk[i] < min_stop) min_stop = stopClk[i];
        if (stopClk[i] > max_stop) max_stop = stopClk[i];
    }
    
    unsigned long long total_time = max_stop - min_start;
    unsigned long long start_spread = max_start - min_start;
    unsigned long long stop_spread = max_stop - min_stop;
    
    printf("\n=== Results ===\n");
    printf("Total execution time: %llu cycles\n", total_time);
    printf("Start time spread: %llu cycles\n", start_spread);
    printf("Stop time spread: %llu cycles\n", stop_spread);
    printf("Number of warps: %u\n", num_warps);
    
    // Calculate per-warp timing
    printf("\nPer-warp timing (cycles):\n");
    printf("Warp ID | Start | Stop | Duration\n");
    printf("--------|-------|------|----------\n");
    for (unsigned i = 0; i < num_warps; i++) {
        unsigned long long duration = stopClk[i] - startClk[i];
        printf("%7u | %5llu | %4llu | %8llu\n", i, startClk[i] - min_start, stopClk[i] - min_start, duration);
    }
    
    // Output CSV for analysis
    printf("\n=== CSV Output (for analysis) ===\n");
    printf("test_mode,num_warps,offset_bytes,delay_cycles,total_time,start_spread,stop_spread\n");
    printf("%u,%u,%u,%u,%llu,%llu,%llu\n", 
           test_mode, num_warps, offset_bytes, delay_cycles, total_time, start_spread, stop_spread);
    
    printf("\n=== Instructions for NSight Compute Analysis ===\n");
    printf("To measure LRC coalescing metrics, run:\n");
    printf("ncu --metrics lrc__lts2lrc_sectors_op_read.sum,lrc__xbar2gpc_sectors_op_read.sum,");
    printf("lrc__average_xbar2gpc_sectors_op_read.ratio ./l2_lrc_effect.o %u %u %u %u\n",
           test_mode, num_warps, offset_bytes, delay_cycles);
    printf("\nKey metrics to observe:\n");
    printf("- lrc__lts2lrc_sectors_op_read.sum: Sectors sent from LTS to LRC\n");
    printf("- lrc__xbar2gpc_sectors_op_read.sum: Sectors sent from LRC to GPC (after coalescing)\n");
    printf("- Ratio: If xbar2gpc < lts2lrc, coalescing occurred\n");
    printf("- Coalescing ratio = xbar2gpc / lts2lrc (lower = more coalescing)\n");
    
    // Cleanup
    free(startClk);
    free(stopClk);
    free(posArray);
    free(dsink);
    cudaFree(posArray_g);
    cudaFree(dsink_g);
    cudaFree(startClk_g);
    cudaFree(stopClk_g);
    
    return 0;
}

