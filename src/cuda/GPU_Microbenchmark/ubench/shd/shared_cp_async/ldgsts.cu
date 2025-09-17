#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#include <cuda_pipeline.h>

template <typename T>
__global__ void pipeline_kernel_async(T *global, uint64_t *clock,
                                     // size_t copy_count, size_t loop)
                                    size_t global_mem_size, size_t shmem_size, size_t loop_over_global, int cta_count)
{
    extern __shared__ char s[];
    T *shared = reinterpret_cast<T *>(s);
    size_t shmem_elem_count = shmem_size / sizeof(T);

    size_t cta_cover_global_mem_size = global_mem_size / cta_count;
    size_t cta_cover_global_mem_elem_count = cta_cover_global_mem_size / sizeof(T);
    size_t block_offset = cta_cover_global_mem_elem_count * blockIdx.x;

    uint64_t clock_start = clock64();
    for (int j = 0; j < loop_over_global; j++)
    {
#pragma unroll(32)
        for (size_t i = 0; i < cta_cover_global_mem_elem_count; i += blockDim.x)
        {
            __pipeline_memcpy_async(&shared[(i + threadIdx.x) % 8192],
                                    &global[block_offset + i + threadIdx.x],
                                    sizeof(T));
        }
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);

    uint64_t clock_end = clock64();

    __syncthreads();
    if (threadIdx.x == 0)
        atomicAdd(reinterpret_cast<unsigned long long *>(clock),
                  clock_end - clock_start);
}
int main(int argc, char **argv)
{
    using T = double;
    size_t loop = 1024;
    size_t num_blocks = 4;
    size_t threads_per_block = 256;

    if (argc < 4 || argc > 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <loop> <num_blocks> <threads_per_block> \n";
        return 1;
    }

    loop = std::atoi(argv[1]);
    num_blocks = std::atoi(argv[2]);
    threads_per_block = std::atoi(argv[3]);

    // Get device max shared memory
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    // size_t max_shared_mem = prop.sharedMemPerBlockOptin;
    // if (max_shared_mem == 0)
    //     max_shared_mem = prop.sharedMemPerBlock;
    size_t max_shared_mem = 100 << 10;

    size_t shared_mem_size = max_shared_mem;

    size_t bytes = 40 << 20; // 40 MB
    size_t total_elems = bytes / sizeof(T);

    // Host data
    T *h_data = new T[total_elems];
    for (size_t i = 0; i < total_elems; i++)
        h_data[i] = static_cast<T>(i);

    // Device memory
    T *d_data;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    uint64_t *d_clock;
    uint64_t zero = 0;
    cudaMalloc(&d_clock, sizeof(uint64_t));
    cudaMemcpy(d_clock, &zero, sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Opt-in to use max shared memory if needed
    cudaFuncSetAttribute(pipeline_kernel_async<T>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem);

    std::cout << "Threads per block        = " << threads_per_block << "\n";
    std::cout << "Max shared mem per block = " << max_shared_mem / 1024 << " KB\n";
    std::cout << "Total elems              = " << total_elems << "\n";
    std::cout << "Loop count               = " << loop << "\n";

    pipeline_kernel_async<T><<<num_blocks, threads_per_block, shared_mem_size>>>(
       // d_data, d_clock, copy_count, loop);
       d_data, d_clock, bytes, shared_mem_size, loop, num_blocks);

    // Copy and print clock result
    uint64_t h_clock;
    cudaMemcpy(&h_clock, d_clock, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    printf("Total clock cycles (summed across blocks): %llu\n", h_clock);

    // Clean up
    cudaFree(d_data);
    cudaFree(d_clock);
    delete[] h_data;

    return 0;
}