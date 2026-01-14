#include "common.h"

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}


extern "C" {

void launch_add(const float* a, const float* b, float* out, int n) {
  // round up
  int blocks = CEIL_DIV(n, BLOCK_SIZE);
  // launch kernel
  add_kernel<<<blocks, BLOCK_SIZE>>>(a, b, out, n);
}

}