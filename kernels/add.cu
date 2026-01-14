#include <cuda_runtime.h>
#include <stdio.h>

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}


extern "C" {

void launch_add(const float* a, const float* b, float* out, int n) {
  int threads_per_block = 256;

  // round up
  int blocks_per_grid = (n + threads_per_block -1) / threads_per_block;

  // launch kernel
  add_kernel<<<blocks_per_grid, threads_per_block>>>(a, b, out, n);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }

  // cudaDeviceSynchronize();
}

}