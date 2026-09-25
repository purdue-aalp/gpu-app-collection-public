#ifndef SHFL_SCAN_DEF_H
#define SHFL_SCAN_DEF_H

#include <cuda_runtime.h>
#include <stdio.h>

#include "../../../hw_def/hw_def.h"

// Fixed sizes keep every warp full and make the scan stages easy to follow.
const int SCAN_THREADS = 128;
const int SCAN_ITEMS = 4;
const int SCAN_WIDTH = SCAN_THREADS * SCAN_ITEMS;
const int SCAN_ROWS = 2;

__global__ void shuffle_test(int *output) {
  int tid = threadIdx.x;
  int value = tid + 10;

  output[tid] = __shfl_up_sync(0xffffffff, value, 1);
  output[64 + tid] = __shfl_down_sync(0xffffffff, value, 1);
  output[128 + tid] = __shfl_xor_sync(0xffffffff, value, 1);
  output[192 + tid] = __shfl_sync(0xffffffff, value, 7);
}

bool run_shuffle_test() {
  int output[4 * 64];
  int *output_g;
  gpuErrchk(cudaMalloc(&output_g, sizeof(output)));

  shuffle_test<<<1, 64>>>(output_g);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaMemcpy(output, output_g, sizeof(output), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(output_g));

  const char *names[] = {"up", "down", "xor", "idx"};
  bool passed = true;
  for (int mode = 0; mode < 4; ++mode) {
    bool matched = true;
    for (int tid = 0; tid < 64; ++tid) {
      int lane = tid % 32;
      int source[] = {lane == 0 ? tid : tid - 1,
                      lane == 31 ? tid : tid + 1,
                      tid ^ 1, (tid / 32) * 32 + 7};
      int expected = source[mode] + 10;
      if (output[mode * 64 + tid] != expected) {
        printf("shfl_%s: thread %d, expected %d, got %d\n", names[mode],
               tid, expected, output[mode * 64 + tid]);
        matched = false;
        break;
      }
    }
    printf("shfl_%s: %s\n", names[mode], matched ? "PASSED" : "FAILED");
    passed = matched && passed;
  }
  return passed;
}

__device__ int warp_scan(int value) {
  int lane = threadIdx.x % 32;
  for (int offset = 1; offset < 32; offset *= 2) {
    int previous = __shfl_up_sync(0xffffffff, value, offset);
    if (lane >= offset) value += previous;
  }
  return value;
}

// One block per row: local scan -> warp scan -> scan of warp totals.
__global__ void scan_rows(const int *input, int *output) {
  __shared__ int warp_totals[SCAN_THREADS / 32];
  int tid = threadIdx.x;
  int lane = tid % 32;
  int warp = tid / 32;
  int base = blockIdx.x * SCAN_WIDTH + tid * SCAN_ITEMS;

  int values[SCAN_ITEMS];
  int sum = 0;
  for (int i = 0; i < SCAN_ITEMS; ++i) {
    sum += input[base + i];
    values[i] = sum;
  }

  int prefix = warp_scan(sum);
  if (lane == 31) warp_totals[warp] = prefix;
  __syncthreads();

  // All 32 lanes participate; unused lanes contribute zero.
  if (warp == 0) {
    int total = lane < SCAN_THREADS / 32 ? warp_totals[lane] : 0;
    total = warp_scan(total);
    if (lane < SCAN_THREADS / 32) warp_totals[lane] = total;
  }
  __syncthreads();

  int carry = prefix - sum;
  if (warp > 0) carry += warp_totals[warp - 1];
  for (int i = 0; i < SCAN_ITEMS; ++i) output[base + i] = values[i] + carry;
}

bool run_scan_test() {
  int input[SCAN_ROWS * SCAN_WIDTH];
  int output[SCAN_ROWS * SCAN_WIDTH];
  int *input_g, *output_g;
  gpuErrchk(cudaMalloc(&input_g, sizeof(input)));
  gpuErrchk(cudaMalloc(&output_g, sizeof(output)));

  bool passed = true;
  for (int pattern = 0; pattern < 2; ++pattern) {
    for (int i = 0; i < SCAN_ROWS * SCAN_WIDTH; ++i)
      input[i] = pattern == 0 ? 1 : (i * 7 % 17) - 8;

    gpuErrchk(cudaMemcpy(input_g, input, sizeof(input), cudaMemcpyHostToDevice));
    scan_rows<<<SCAN_ROWS, SCAN_THREADS>>>(input_g, output_g);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpy(output, output_g, sizeof(output), cudaMemcpyDeviceToHost));

    bool matched = true;
    for (int row = 0; row < SCAN_ROWS; ++row) {
      int expected = 0;
      for (int col = 0; col < SCAN_WIDTH; ++col) {
        int index = row * SCAN_WIDTH + col;
        expected += input[index];
        if (output[index] != expected) {
          if (matched)
            printf("Row scan: row %d, column %d, expected %d, got %d\n",
                   row, col, expected, output[index]);
          matched = false;
        }
      }
    }
    printf("Row scan (%s): %s\n", pattern == 0 ? "ones" : "mixed",
           matched ? "PASSED" : "FAILED");
    passed = matched && passed;
  }

  gpuErrchk(cudaFree(input_g));
  gpuErrchk(cudaFree(output_g));
  return passed;
}

#endif
