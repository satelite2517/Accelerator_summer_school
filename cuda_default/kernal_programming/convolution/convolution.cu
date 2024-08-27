#include <cstdio>

#include "convolution.h"

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

void naive_cpu_convolution(float *_I, float *_F, float *_O, int N, int C, int H,
                           int W, int K, int R, int S, int pad_h, int pad_w,
                           int stride_h, int stride_w, int dilation_h,
                           int dilation_w) {
  float *I = _I, *F = _F, *O = _O;
  // Naive CPU convolution
  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  for (int on = 0; on < ON; ++on) {
    for (int oc = 0; oc < OC; ++oc) {
      for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
          float sum = 0;
          for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
              for (int s = 0; s < S; ++s) {
                const int n = on;
                const int h = oh * stride_h - pad_h + r * dilation_h;
                const int w = ow * stride_w - pad_w + s * dilation_w;
                const int k = oc;
                if (h < 0 || h >= H || w < 0 || w >= W) continue;
                sum += I[((n * C + c) * H + h) * W + w] *
                       F[((k * C + c) * R + r) * S + s];
              }
            }
          }
          O[((on * OC + oc) * OH + oh) * OW + ow] = sum;
        }
      }
    }
  }
}

static float *Input_gpu, *Filter_gpu, *Output_gpu;

__global__ void convolution_cuda(const float *Input_gpu, const float *Filter_gpu, float *Output_gpu,  
                                   int N, int C,
                                   int H, int W, int K, int R, int S, int pad_h,
                                   int pad_w, int stride_h, int stride_w,
                                   int dilation_h, int dilation_w){

  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  
  // parse (n, c, h, w) from thread index
  const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  const int on = tidx / (OC * OH * OW);
  const int oc = (tidx / (OH * OW)) % OC;
  const int oh = (tidx / OW) % OH;
  const int ow = tidx % OW;

  if (on >= ON || oc >= OC || oh >= OH || ow >= OW) return;
  
  float sum = 0;
  for (int c = 0; c < C; ++c) {
    for (int r = 0; r < R; ++r) {
      for (int s = 0; s < S; ++s) {
        const int n = on;
        const int h = oh * stride_h - pad_h + r * dilation_h;
        const int w = ow * stride_w - pad_w + s * dilation_w;
        const int k = oc;
        if (h < 0 || h >= H || w < 0 || w >= W) continue;
        sum +=
            Input_gpu[((n * C + c) * H + h) * W + w] * Filter_gpu[((k * C + c) * R + r) * S + s];
      }
    }
  }
  Output_gpu[((on * OC + oc) * OH + oh) * OW + ow] = sum;



}

void convolution(float *_I, float *_F, float *_O, int N, int C, int H, int W,
                 int K, int R, int S, int pad_h, int pad_w, int stride_h,
                 int stride_w, int dilation_h, int dilation_w) {
  // Remove this line after you complete the convolution on GPU
  // naive_cpu_convolution(_I, _F, _O, N, C, H, W, K, R, S, pad_h, pad_w, stride_h,
  //                       stride_w, dilation_h, dilation_w);

  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  CHECK_CUDA(cudaMemcpy(Input_gpu, _I, N * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(Filter_gpu, _F, K * C * R * S * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(512);
  dim3 grid((N*K*OH*OW + 512 - 1)/512);
  convolution_cuda<<<grid, block>>>(Input_gpu,Filter_gpu,Output_gpu,N, C, H, W, K, R, S, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);

  CHECK_CUDA(cudaMemcpy(_O, Output_gpu, N * K * OH * OW * sizeof(float),cudaMemcpyDeviceToHost));
  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void convolution_initialize(int N, int C, int H, int W, int K, int R, int S,
                            int pad_h, int pad_w, int stride_h, int stride_w,
                            int dilation_h, int dilation_w) {
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  CHECK_CUDA(cudaMalloc(&Input_gpu, N * C * H * W *sizeof(float)));
  CHECK_CUDA(cudaMalloc(&Filter_gpu, K * C * R * S * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&Output_gpu, N * K * OH * OW * sizeof (float)));


  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void convolution_cleanup(float *_I, float *_F, float *_O, int N, int C, int H,
                         int W, int K, int R, int S, int pad_h, int pad_w,
                         int stride_h, int stride_w, int dilation_h,
                         int dilation_w) {
  
  CHECK_CUDA(cudaFree(Input_gpu));
  CHECK_CUDA(cudaFree(Output_gpu));
  CHECK_CUDA(cudaFree(Filter_gpu));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}