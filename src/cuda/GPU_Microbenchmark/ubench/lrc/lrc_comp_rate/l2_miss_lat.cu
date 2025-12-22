//This code is a modification of L2 cache benchmark from 
//"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking": https://arxiv.org/pdf/1804.06826.pdf

//This benchmark measures the maximum read bandwidth of L2 cache for 32f
//Compile this file using the following command to disable L1 cache:
//    nvcc -Xptxas -dlcm=cg -Xptxas -dscm=wt l2_miss_lat.cu

//This code have been tested on Volta V100 architecture
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#ifdef TUNER
#include "../../../hw_def/hw_def.h"
#define REPEAT_TIMES 2048

#else
#include "../../../hw_def/common/gpuConfig.h"
#define REPEAT_TIMES 2048
#define CLK_FREQUENCY 1980 // Assume A100 freq
#endif

#define L2_LINE_SIZE 128 // in bytes

__global__ void l2_miss_lat (
	unsigned long long *startClk, unsigned long long*stopClk, float*dsink, 
	float*posArray, unsigned ARRAY_SIZE, unsigned num_misses, unsigned stride_floats, 
	unsigned num_warps, unsigned long long* per_access_times) {
	// block and thread index
	uint32_t tid = threadIdx.x;
	uint32_t bid = blockIdx.x;
	uint32_t uid = bid * blockDim.x + tid;
	uint32_t TOTAL_THREADS = gridDim.x * blockDim.x;
	uint32_t warp_id = uid / warpSize; 
	uint32_t lane_id = uid % warpSize;

	// a register to avoid compiler optimization
	float sink = 0;
	float* ptr = posArray;
	for(unsigned i = 1; i < num_misses; i++){
		asm volatile("bar.sync 0;");

		unsigned long long stop = 0;
		unsigned long long start = 0;
		// Sweep the number of misses
		
		// start timing
		// asm volatile("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
		start = clock64();

		// Each warp accesses a unique cache line to generate L2 misses
		if(warp_id < i){
            // compute a unique index per (i, warp) in units of floats
            uint64_t idx = (uint64_t)i * (uint64_t)num_warps + (uint64_t)warp_id;
            idx = idx * (uint64_t)stride_floats;
            // wrap into array bounds
            idx = idx % (uint64_t)ARRAY_SIZE;
            float* ptr = posArray + idx;

            // uint32_t start_access, stop_access;
            // asm volatile("mov.u32 %0, %%clock;" : "=r"(start_access) :: "memory");
			 unsigned long long t0 = clock64();
            asm volatile("{\t\n"
                ".reg .f32 data;\n\t"
                "ld.global.cg.f32 data, [%1];\n\t"
                "add.f32 %0, data, %0;\n\t"
                "}" : "+f"(sink) : "l"(ptr) : "memory"
            );
			 unsigned long long t1 = clock64();
			 if (lane_id == 0) per_access_times[warp_id + i * num_warps] = (t1 - t0);
            // asm volatile("mov.u32 %0, %%clock;" : "=r"(stop_access) :: "memory");
            // if(lane_id==0) printf("Thread %u, Miss %u, Address: %p, Latency: %u\n", uid, i, ptr, stop_access - start_access);
        }
		asm volatile("bar.sync 0;");

		// stop timing
		
		// asm volatile("mov.u32 %0, %%clock;" : "=r"(stop) :: "memory");
		stop = clock64();
		ptr = posArray + i;
		// store the result
		startClk[uid + i*TOTAL_THREADS] = start;
		stopClk[uid + i*TOTAL_THREADS] = stop;
		dsink[uid] = sink;
	}
	
	
}
int main(int argc, char *argv[]){
	intilizeDeviceProp(0, argc, argv);

	uint32_t num_misses = 128; 
	if(argc > 1){
		num_misses = atoi(argv[1]);
	}
	printf("Number of L2 misses to be generated = %d\n", num_misses);
	uint64_t blocks_launched = 1;
	uint64_t threads_per_block = config.THREADS_PER_BLOCK;
	uint64_t total_threads = blocks_launched * threads_per_block;
	uint32_t num_warps = (total_threads + config.WARP_SIZE - 1) / config.WARP_SIZE;
	unsigned stride_floats = L2_LINE_SIZE / sizeof(float); // one cache line per access
	uint32_t min_array_size = num_misses * num_warps * stride_floats; // Each warp accesses a unique cache line to generate L2 misses
	printf("Minimum array size to generate %u L2 misses = %lu bytes %u elements\n", num_misses, min_array_size*sizeof(float), min_array_size);

	if(min_array_size < config.L2_SIZE/sizeof(float)){
		printf("Warning:: Array size fits in L2 cache\n");
	}

  unsigned ARRAY_SIZE = min_array_size + 1024; // Add some extra space to avoid boundary checking in the kernel
	unsigned long long *startClk = (unsigned long long*) malloc(total_threads*num_misses*sizeof(unsigned long long));
    unsigned long long *stopClk = (unsigned long long*) malloc(total_threads*num_misses*sizeof(unsigned long long));

	float *posArray = (float*) malloc(ARRAY_SIZE*sizeof(float));
	float *dsink = (float*) malloc(total_threads*sizeof(float));

	

    for (int i=0; i<ARRAY_SIZE; i++)
            posArray[i] = (float)i;
	// for(int misses = 1; misses <= num_misses; misses*=2){
	// 	float *posArray_g;
	// 	float *dsink_g;
	// 	uint32_t *startClk_g;
	// 	uint32_t *stopClk_g;
	// 	gpuErrchk( cudaMalloc(&posArray_g, ARRAY_SIZE*sizeof(float)) );
	// 	gpuErrchk( cudaMalloc(&dsink_g, config.TOTAL_THREADS*sizeof(float)) );
	// 	gpuErrchk( cudaMalloc(&startClk_g, config.TOTAL_THREADS*sizeof(uint32_t)) );
	// 	gpuErrchk( cudaMalloc(&stopClk_g, config.TOTAL_THREADS*sizeof(uint32_t)) );

	// 	gpuErrchk( cudaMemcpy(posArray_g, posArray, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice) );
	// 	l2_miss_lat<<<config.BLOCKS_NUM, config.THREADS_PER_BLOCK>>>(startClk_g, stopClk_g, dsink_g, posArray_g, ARRAY_SIZE, num_misses);
	// 	gpuErrchk( cudaPeekAtLastError() );
		
	// 	gpuErrchk( cudaMemcpy(startClk, startClk_g, config.TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	// 	gpuErrchk( cudaMemcpy(stopClk, stopClk_g, config.TOTAL_THREADS*sizeof(uint32_t), cudaMemcpyDeviceToHost) );
	// 	gpuErrchk( cudaMemcpy(dsink, dsink_g, config.TOTAL_THREADS*sizeof(float), cudaMemcpyDeviceToHost) );

	// 	uint64_t total_time = stopClk[0] - startClk[0];
	// 	std::cout << "============================\n";
	// 	std::cout << "Number of misses = " << misses << " ";
	// 	std::cout << "Total Clk number = " << total_time << " ";
	// 	std::cout << "Number of misses = " << misses << " ";
	// 	float avg_time = (float)total_time/misses;
	// 	std::cout << "Average L2 Miss Latency = " << avg_time << " clk ";
	// 	float l2_miss_lat = avg_time * CLK_FREQUENCY * 1e-6;
	// 	std::cout << "L2 Miss Latency = " << l2_miss_lat << " sec\n";
	// }

	float *posArray_g;
	float *dsink_g;
	unsigned long long *startClk_g;
	unsigned long long *stopClk_g;

	
	
	gpuErrchk( cudaMalloc(&posArray_g, ARRAY_SIZE*sizeof(float)) );
	gpuErrchk( cudaMalloc(&dsink_g, total_threads*sizeof(float)) );
	gpuErrchk( cudaMalloc(&startClk_g, blocks_launched*threads_per_block*num_misses*sizeof(unsigned long long)) );
	gpuErrchk( cudaMalloc(&stopClk_g, blocks_launched*threads_per_block*num_misses*sizeof(unsigned long long)) );
	gpuErrchk( cudaMemcpy(posArray_g, posArray, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice) );
	unsigned long long *per_warp_lat_h = (unsigned long long*) malloc(sizeof(unsigned long long) * num_warps * num_misses);
	unsigned long long *per_warp_lat_g;
	gpuErrchk( cudaMalloc(&per_warp_lat_g, sizeof(unsigned long long) * num_warps * num_misses) );
	gpuErrchk( cudaMemset(per_warp_lat_g, 0xff, sizeof(unsigned long long) * num_warps * num_misses) ); // init
	l2_miss_lat<<<1, threads_per_block>>>(
		startClk_g, stopClk_g, dsink_g, posArray_g, ARRAY_SIZE, num_misses, stride_floats, num_warps, per_warp_lat_g);
	
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaMemcpy(startClk, startClk_g, blocks_launched*threads_per_block*num_misses*sizeof(unsigned long long), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(stopClk, stopClk_g, blocks_launched*threads_per_block*num_misses*sizeof(unsigned long long), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(dsink, dsink_g, total_threads*sizeof(float), cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaMemcpy(per_warp_lat_h, per_warp_lat_g, sizeof(unsigned long long) * num_warps * num_misses, cudaMemcpyDeviceToHost) );
	uint32_t total_misses = 0;
	std::cout << "Average per-warp latencies (clk), #warps counted, i\n";
	for (unsigned i = 1; i < num_misses; i++){
		// uint64_t total_time = stopClk[0 + i*config.THREADS_PER_BLOCK*config.BLOCKS_NUM] - startClk[0 + i*config.THREADS_PER_BLOCK*config.BLOCKS_NUM];
		// std::cout << "Number of misses = " << i << " ";
		// std::cout << "Total Clk number = " << total_time << " ";
		// std::cout << "Number of misses = " << i << " ";
		// float avg_time = (float)total_time/i;
		// std::cout << "Average L2 Miss Latency = " << avg_time << " clk ";
		// float l2_miss_lat = avg_time * CLK_FREQUENCY * 1e-6;
		// std::cout << "L2 Miss Latency = " << l2_miss_lat << " sec ";
		total_misses += i;
		// std::cout << "Per-warp access latencies (clk): ";
		unsigned long long sum_lat = 0;
		int num_warps_counted = 0;
		for (unsigned w = 0; w < i; w++){
			unsigned long long lat = per_warp_lat_h[w + i * num_warps];
			if (lat == 0xffffffffffffff) continue; // not set
			// std::cout << lat << " ";
			sum_lat += lat;
			num_warps_counted++;
		}
		float avg_warp_lat = (float)sum_lat / (float)num_warps_counted;
		// std::cout << "\n";
		std::cout << avg_warp_lat << "," << num_warps_counted << "," << i << "\n";

	}
	printf("Total misses generated = %u\n", total_misses);

    return 0;
}
