#pragma once

#include "tensor.h"

void data_upload(Tensor *in);
void data_cleanup(Tensor *in);

void Linear_cuda(Tensor *in, Tensor *w, Tensor *b, Tensor *out, cudaStream_t stream);
void Reshape_cuda(Tensor *in, Tensor *out, cudaStream_t stream);
void Conv2d_Tanh_fusion_cuda(Tensor *in, Tensor *w, Tensor *b, Tensor *out, cudaStream_t stream);
void ConvTran_Batch_ReLU_fusion_cuda(Tensor *in, Tensor *Conv_weight, Tensor *Conv_bias, Tensor *Conv_ans,  Tensor *Batch_weight, Tensor *Batch_bias, Tensor *out, cudaStream_t stream);
