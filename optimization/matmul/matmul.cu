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
  const int tdix = blockIdx.x * blockDim.x + threadIdx.x;

  if (tdix >= M * N ) return;

  const int i = tdix / N;
  const int j = tdix % N;

  if (i >= M || j >= N) return;

  float sum = 0;

  float a0, a1, a2, a3, a4, a5, a6, a7;
  float b0, b1, b2, b3, b4, b5, b6, b7;

  for (int k = 0; k + 7 < K; k += 8) {
  a0 = A[i * K + (k + 0)];
  a1 = A[i * K + (k + 1)];
  a2 = A[i * K + (k + 2)];
  a3 = A[i * K + (k + 3)];
  a4 = A[i * K + (k + 4)];
  a5 = A[i * K + (k + 5)];
  a6 = A[i * K + (k + 6)];
  a7 = A[i * K + (k + 7)];
  b0 = B[(k + 0) * N + j];
  b1 = B[(k + 1) * N + j];
  b2 = B[(k + 2) * N + j];
  b3 = B[(k + 3) * N + j];
  b4 = B[(k + 4) * N + j];
  b5 = B[(k + 5) * N + j];
  b6 = B[(k + 6) * N + j];
  b7 = B[(k + 7) * N + j];
  sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 +
                  a6 * b6 + a7 * b7;
  }
  for (int k = K - K % 8; k < K; k++) {
    sum += A[i * K + k] * B[k * N + j];
  }
  C[i * N + j] = sum;

}

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  // Remove this line after you complete the matmul on GPU
  //naive_cpu_matmul(_A, _B, _C, M, N, K);

  // (TODO) Upload A and B matrix to GPU
  CHECK_CUDA(cudaMemcpy(A_gpu, _A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B_gpu, _B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  // (TODO) Launch kernel on a GPU
  dim3 block(16*16);
  dim3 grid((M*N + block.x - 1) / block.x);
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

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
