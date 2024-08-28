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

#define BLOCK_SIZE 32
#define TILE_HEIGHT 32
#define TILE_WIDTH 64
#define TILE_ 4

__global__ void matmul_kernel(const float *A, const float *B, float *C, const int M, const int N, const int K) {
  // if you change thx and thy then this will be faster
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int gj = blockIdx.x;
  int gi = blockIdx.y;
  
  int lj = threadIdx.x;
  int li = threadIdx.y;
  __shared__ float Alocal[TILE_HEIGHT][TILE_];
  __shared__ float Blocal[TILE_][TILE_WIDTH];
  float c = 0.f;
  int A_row_index = (gi * BLOCK_SIZE + li);
  int B_col_index = (gj * BLOCK_SIZE + lj);
  for (int bk = 0; bk < K; bk += BLOCK_SIZE) {
    int A_col_index = bk + lj;
    Alocal[li][lj] = (A_row_index < M && A_col_index < K)
                         ? A[A_row_index * K + A_col_index]
                         : 0.f;
    int B_row_index = bk + li;
    Blocal[li][lj] = (B_row_index < K && B_col_index < N)
                         ? B[B_row_index * N + B_col_index]
                         : 0.f;
    __syncthreads();
    for (int lk = 0; lk < BLOCK_SIZE; ++lk) {
      c += Alocal[li][lk] * Blocal[lk][lj];
    }
    __syncthreads();
  }
  if (i < M && j < N) C[i * N + j] = c;

}

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  // Remove this line after you complete the matmul on GPU
  //naive_cpu_matmul(_A, _B, _C, M, N, K);

  // (TODO) Upload A and B matrix to GPU
  CHECK_CUDA(cudaMemcpy(A_gpu, _A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B_gpu, _B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  // (TODO) Launch kernel on a GPU
  dim3 blockDim(TILE_WIDTH, TILE_HEIGHT);
  dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_HEIGHT - 1) / TILE_HEIGHT);
  matmul_kernel<<<gridDim, blockDim>>>(A_gpu, B_gpu, C_gpu, M, N, K);

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
