#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {
  void hello_gpu() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess) {
      printf("CUDA Error: %s\n", cudaGetErrorString(err));
      return;
    }

    printf("Hello from C++! Found %d CUDA device.\n", device_count);
  }
}