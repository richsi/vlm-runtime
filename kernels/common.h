#pragma once
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define CEIL_DIV(M, N) (((M) + (N) -1) / (N))