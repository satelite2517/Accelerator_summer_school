#pragma once

#include "tensor.h"

void data_stream(Tensor *in);

void Linear_kernel_cuda(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void Reshape_kernel_cuda(Tensor *in, Tensor *out);
void Conv2d_Tanh_fusion_kernel_cuda(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void ConvTran_Batch_ReLU_fusion_kernel_cuda(Tensor *in, Tensor *Conv_weight, Tensor *Conv_bias,  Tensor *Batch_weight, Tensor *Batch_bias, Tensor *out);


/* Example GPU kernel */
void Linear_cuda(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
float* Linear_cuda_(const float *in, Tensor *w, Tensor *b,  Tensor *out);
void Reshape_cuda(Tensor *in, Tensor *out);
void Conv2d_Tanh_fusion_cuda(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void ConvTran_Batch_ReLU_fusion_cuda(Tensor *in, Tensor *Conv_weight, Tensor *Conv_bias,  Tensor *Batch_weight, Tensor *Batch_bias, Tensor *out);

void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out);

