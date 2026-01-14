#include "common.h"

__global__ void silu_kernel(const float* d_in, float* d_out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    float x = d_in[idx];
    // SiLU: x / (1 + e^-x)
    float sigmoid = 1.0f / (1.0f + expf(-x));
    d_out[idx] = x * sigmoid;
  }
}

extern "C" {

void launch_silu(const float* d_in, float* d_out, int n) {
  int blocks = CEIL_DIV(n, BLOCK_SIZE);
  silu_kernel<<<blocks, BLOCK_SIZE>>>(d_in, d_out, n);
}

}