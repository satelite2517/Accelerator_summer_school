#include <cstdio>

#include "matmul.h"

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
static float *A_gpu, *B_gpu, *C_gpu;

void naive_cpu_matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        _C[i * N + j] += _A[i * K + k] * _B[k * N + j];
      }
    }
  }
}

__global__ void matmal(const float *A, const float *B, float *C, const int M, const int N, const int K) {
  // if you change thx and thy then this will be faster
  int thx = blockIdx.x * blockDim.x + threadIdx.x;
  int thy = blockIdx.y * blockDim.y + threadIdx.y;
  if (thx>=M || thy>=N) return;

  C[thx * N + thy]= 0.0f;
  for (int i = 0; i <K; i++) {
    C[thx * N + thy] += A[thx * K + i] * B[i * N + thy];
  }

}

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  // Remove this line after you complete the matmul on GPU
  //naive_cpu_matmul(_A, _B, _C, M, N, K);

  // (TODO) Upload A and B matrix to GPU
  CHECK_CUDA(cudaMemcpy(A_gpu, _A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B_gpu, _B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  // (TODO) Launch kernel on a GPU
  dim3 block(32, 32);
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
  matmal<<<grid, block>>>(A_gpu, B_gpu, C_gpu, M, N, K);

  // (TODO) Download C matrix from GPU
  CHECK_CUDA(cudaMemcpy(_C, C_gpu,M*N*sizeof(float), cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_init(int M, int N, int K) {
  // (TODO) Allocate device memory
  CHECK_CUDA(cudaMalloc(&A_gpu, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&B_gpu, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&C_gpu, M * N * sizeof(float)));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_cleanup(float *_A, float *_B, float *_C, int M, int N, int K) {
  // (TODO) Do any post-matmul cleanup work here.
  CHECK_CUDA(cudaFree(A_gpu));
  CHECK_CUDA(cudaFree(B_gpu));
  CHECK_CUDA(cudaFree(C_gpu));

  free(_A)
  free(_B)
  free(_C)

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
