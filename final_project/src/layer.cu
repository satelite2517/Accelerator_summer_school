#include "layer.h"

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)


//This part is for kernel functions

__global__ void Linear_kernel(const float *in_, const float *w_, const float *b_, float *out_, const size_t M, const size_t N, const size_t K){
  const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  const int i = tidx / N;
  const int j = tidx % N;

  if (i >= M || j >= N) return;
  out_[i*M+j]=b_[j];
  for (size_t k = 0; k<K; k++){
    out_[i*M+j] += in_[i * K +k] * w_[j * K + k];
  }

}

__global__ void Reshape_kernel(const float *in_, float *out_, const size_t N, const size_t D, const size_t C, const size_t H, const size_t W){
  const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  const int n = tidx / (C*H*W);
  const int c = (tidx / (H*W)) % C;
  const int h = (tidx / W) % H;
  const int w = tidx % W;

  if (n >= N || c >= C || h >= H || w >= W) return;
  out_[n * C * H * W + c * H * W + h * W + w] = in_[n * D + c * H * W + h * W + w];
}

__global__ void ConvTranspose2d_kernel(const float *in_, const float *w_, const float *b_, float *out_, 
                                      const size_t C, const size_t H, const size_t W, const size_t K, 
                                      const size_t R, const size_t S, const size_t OH, const size_t OW, 
                                      const size_t stride, const size_t pad, const size_t dilation){
  const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx >= K * OH * OW) return;

  const int oc = tidx / (OH*OW);
  const int oh = (tidx / OH) % OW;
  const int ow = tidx  % OW;

  if (oc >= K || oh >= OH || ow >= OW) return;

  float o = b_[oc];
  for (size_t c = 0; c < C; ++c) {
    for (size_t r = 0; r < R; ++r) {
      for (size_t s = 0; s < S; ++s) {
        if ((oh - (r * dilation - pad)) % stride != 0) continue;
        if ((ow - (s * dilation - pad)) % stride != 0) continue;
        size_t h = (oh - (r * dilation - pad)) / stride;
        size_t w = (ow - (s * dilation - pad)) / stride;
        if (h >= H || w >= W) continue;
        o += in_[c * H * W + h * W + w] * w_[c * K * R * S + oc * R * S + r * S + s];
      }
    }
  }
  out_[oc * OH * OW + oh * OW + ow] = o;
  
}

__global__ void BatchNorm2d_kernel(const float *in_, const float *w_, const float *b_, float *out_, 
                                    const size_t C, const size_t H, const size_t W){

  const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx >= C * H * W) return;

  const int c = tidx / (H * W);
  const int h = (tidx / W) % H;
  const int w = tidx % W;

  if (c >= C || h >= H || w >= W) return;

  float mean = 0.0f;
  float var = 0.0f;
  for (size_t i = 0; i < H; i++) {
    for (size_t j = 0; j < W; j++) {
      float val = in_[c * H * W + i * W + j];
      mean += val;
    }
  }

  mean /= (H * W);

  for (size_t i = 0; i < H; i++) {
    for (size_t j = 0; j < W; j++) {
      float val = in_[c * H * W + i * W + j];
      var += (val - mean) * (val - mean);
    }
  }

  var /= (H * W);

  for (size_t i = 0; i < H; i++) {
    for (size_t j = 0; j < W; j++) {
      out_[c * H * W + i * W + j] = w_[c] * (in_[c * H * W + i * W + j] - mean) / sqrt(var + 1e-5) + b_[c];
    }
  }

}

__global__ void LeakyReLU_kernel(float *inout, size_t N, float alpha) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) { 
    if (inout[idx] < 0) { inout[idx] *= alpha; }
  }
}  

__global__ void Conv2d_Tanh_fusion_kernel(const float *input_, const float *w_, const float *b_, float *output_, const size_t N, const size_t K,
                              const size_t C, const size_t R, const size_t S, const size_t H, const size_t W, const size_t OH, const size_t OW,
                              const size_t stride, const size_t pad, const size_t dilation){
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * K * OH * OW) return;

  size_t n = idx / (K * OH * OW);
  size_t oc = (idx / (OH * OW)) % K;
  size_t oh = (idx / OW) % OH;
  size_t ow = idx % OW;


  float o = b_[oc];
  for (size_t c = 0; c < C; c++) {
    for (size_t r = 0; r < R; r++) {
      for (size_t s = 0; s < S; s++) {
        size_t h = oh * stride - pad + r * dilation;
        size_t w = ow * stride - pad + s * dilation;
        if (h >= H || w >= W) continue;
        o += input_[n * C * H * W + c * H * W + h * W + w] *
          w_[oc * C * R * S + c * R * S + r * S + s];
      }
    }
  }
  output_[n * K * OH * OW + oc * OH * OW + oh * OW + ow] = tanh(o);  

}



//Will upload with this part
void data_upload(Tensor *z) {
    // Calculate the total number of elements in the tensor
    size_t total_elements = 1;
    for (size_t i = 0; i < z->ndim; i++) {
        if (z->shape[i] > 0) {
            total_elements *= z->shape[i];
        }
    }

    // Allocate memory directly to z->gpu_buf
    CHECK_CUDA(cudaMalloc(&(z->gpu_buf), total_elements * sizeof(float)));
    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(z->gpu_buf, z->buf, total_elements * sizeof(float), cudaMemcpyHostToDevice));
}

void data_cleanup(Tensor *z) {
    // Free the memory allocated for the tensor
    CHECK_CUDA(cudaFree(z->gpu_buf));
    z->gpu_buf = nullptr;
}



//This part is for the cuda fuctions to run kernel
void Linear_cuda(Tensor *in, Tensor *w, Tensor *b, Tensor *out){
  size_t M = out->shape[0];
  size_t N = out->shape[1];
  size_t K = w->shape[1];

  Linear_kernel<<<((M*N+1024-1)/1024), 1024>>>(in->gpu_buf, w->gpu_buf, b->gpu_buf,out->gpu_buf, M, N, K);
}

void Reshape_cuda(Tensor *in, Tensor *out){
  size_t N = in->shape[0];
  size_t D = in->shape[1];
  size_t C = out->shape[1];
  size_t H = out->shape[2];
  size_t W = out->shape[3];

  Reshape_kernel<<<((N*C*H*W+1024-1)/1024), 1024>>>(in->gpu_buf, out->gpu_buf, N, D, C, H, W);
}

void ConvTran_Batch_ReLU_fusion_cuda(Tensor *in, Tensor *Conv_weight, Tensor *Conv_bias, Tensor *Conv_ans, Tensor *Batch_weight, Tensor *Batch_bias, Tensor *out){
  size_t Conv_C = in->shape[1];
  size_t H = in->shape[2];
  size_t W = in->shape[3];
  size_t Conv_K = Conv_weight->shape[1];
  size_t Conv_R = Conv_weight->shape[2];
  size_t Conv_S = Conv_weight->shape[3];
  size_t Batch_C = Batch_weight->shape[0];
  size_t OH = out->shape[2];
  size_t OW = out->shape[3];
 
  const size_t stride = 2;
  const size_t pad = 1;
  const size_t dilation = 1;

  ConvTranspose2d_kernel<<<(Conv_K*OH*OW+1024-1)/1024, 1024>>>(in->gpu_buf, Conv_weight->gpu_buf, Conv_bias->gpu_buf, Conv_ans->gpu_buf, Conv_C, H, W, Conv_K, Conv_R, Conv_S, OH, OW, stride, pad, dilation);
  BatchNorm2d_kernel<<<(Batch_C*OH*OW+1024-1)/1024, 1024>>>(Conv_ans->gpu_buf, Batch_weight->gpu_buf, Batch_bias->gpu_buf, out->gpu_buf, Batch_C, OH, OW);
  LeakyReLU_kernel<<<(Batch_C*OH*OW+1024-1)/1024, 1024>>>(out->gpu_buf, Batch_C*OH*OW, 0.01);

}

void Conv2d_Tanh_fusion_cuda(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out){
  size_t N = in->shape[0];
  size_t C = in->shape[1];
  size_t H = in->shape[2];
  size_t W = in->shape[3];
  size_t K = weight->shape[0];
  size_t R = weight->shape[2];
  size_t S = weight->shape[3];
  size_t OH = out->shape[2];
  size_t OW = out->shape[3];

  const size_t stride = 1;
  const size_t pad = 1;
  const size_t dilation = 1;

  Conv2d_Tanh_fusion_kernel<<<(N*K*OH*OW+1024-1)/1024, 1024>>>(in->gpu_buf, weight->gpu_buf, bias->gpu_buf, out->gpu_buf, N, K, C, R, S, H, W, OH, OW, stride, pad, dilation);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(out->buf, out->gpu_buf, N*K*OH*OW*sizeof(float), cudaMemcpyDeviceToHost));

}
