#pragma once
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CALLABLE __host__ __device__
#define CHECK_CUDA(code)                                                       \
  { maniskill::check_cuda(code, __FILE__, __LINE__); }

//#define launch_kernel(kernel, dim, stream, args)                               \
//  {                                                                            \
//    const int num_threads = 256;                                               \
//    const int num_blocks = (dim + num_threads - 1) / num_threads;              \
//    kernel<<<num_blocks, 256, 0, stream>>> args;                               \
//    cudaError_t code = cudaGetLastError();                                     \
//    if (code != cudaSuccess)                                                   \
//      printf("CUDA Error: %s %s %d\n", cudaGetErrorString((cudaError_t)code),  \
//             __FILE__, __LINE__);                                              \
//  }

#define launch_kernel(kernel, dim, stream, args)                               \
  {                                                                            \
    const int num_threads = 256;                                               \
    const int num_blocks = (dim + num_threads - 1) / num_threads;              \
    kernel<<<num_blocks, 256, 0, stream>>> args;                               \
  }

namespace maniskill {

void check_cuda(int code, const char *file, int line) {
  if (code != cudaSuccess) {
    printf("CUDA Error: %s %s %d\n", cudaGetErrorString((cudaError_t)code),
           file, line);
  }
}

inline CUDA_CALLABLE int get_tid() {
  return blockDim.x * blockIdx.x + threadIdx.x;
}

} // namespace maniskill
