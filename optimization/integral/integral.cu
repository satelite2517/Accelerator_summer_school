#include <cstdio>

#include "integral.h"

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

// Device(GPU) pointers
static double *pi_gpu;

static __device__ double f(double x) { return 4.0 / (1 + x * x); }

#define THREADS_PER_BLOCK 1024
#define ELEMENTS_PER_BLOCK (THREADS_PER_BLOCK * 2) //since one thread calculates two elements
#define output_elements ((num_intervals + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK) //number of blocks


__global__ void integral_sum_kernel(double *pi_gpu, size_t num_intervals) {
  // (TODO) Implement integral calculation on GPU
  extern __shared__ double L[];

  unsigned int tid = threadIdx.x;
  unsigned int offset = blockIdx.x * blockDim.x * 2;
  unsigned int stride = blockDim.x;

  double dx = 1.0 / (double) num_intervals;
  L[tid] = 0;

  unsigned int x1 = tid + offset;
  unsigned int x2 = tid + stride + offset;
  if (x1 < num_intervals) L[tid] += f(x1 * dx) * dx;
  if (x2 < num_intervals) L[tid] += f(x2 * dx) * dx;
  __syncthreads();

  for (stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) L[tid] += L[tid + stride];
    __syncthreads();
  }

  if (tid == 0) pi_gpu[blockIdx.x] = L[0];

}

double integral(size_t num_intervals) {
  double pi_value = 0.0;

  dim3 gridDim(output_elements);
  dim3 blockDim(THREADS_PER_BLOCK);

  integral_sum_kernel<<<gridDim, blockDim, THREADS_PER_BLOCK * sizeof(double), 0>>>(pi_gpu, num_intervals);

  double *output_cpu = (double *)malloc(sizeof(double) * output_elements);
  CHECK_CUDA(cudaMemcpy(output_cpu, pi_gpu, sizeof(double) * output_elements, cudaMemcpyDeviceToHost));

  double sum = 0.0;
  for (size_t i = 0; i < output_elements; i++) {
    sum += output_cpu[i];
  }
  pi_value = sum;
  
  return pi_value;
}

void integral_init(size_t num_intervals) {
  // (TODO) Allocate device memory
  CHECK_CUDA(cudaMalloc(&pi_gpu, sizeof(double) * output_elements));

  // DO NOT REMOVE; NEED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void integral_cleanup() {
  // (TODO) Free device memory
  CHECK_CUDA(cudaFree(pi_gpu));


  // DO NOT REMOVE; NEED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
