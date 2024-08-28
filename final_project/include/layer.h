#pragma once

#include "tensor.h"

/* Elementwise operations */
void LeakyReLU(Tensor *inout);
void Tanh(Tensor *inout);

/* Matmul operations */
void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out);

/* Data movement operations */
void Reshape(Tensor *in, Tensor *out);

/* Convolutional operations */
void ConvTranspose2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out);
void Conv2d(Tensor *in, Tensor *w, Tensor *b, Tensor *out);

/* Other operations */
void BatchNorm2d(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out);

/* Example GPU kernel */
void LeakyReLU_cuda(Tensor *inout);
void Linear_cuda(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void ConvTranspose2d_cuda(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out);
void Conv2d_cuda(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void BatchNorm2d_cuda(Tensor *in, Tensor *weight, Tensor *bias, Tensor *out);
void Tanh_cuda(Tensor *inout);
void Reshape_cuda(Tensor *in, Tensor *out);


void Conv2d_Tanh_fusion_cuda(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void ConvTran_Batch_ReLU_fusion_cuda(Tensor *in, Tensor *Conv_weight, Tensor *Conv_bias,  Tensor *Batch_weight, Tensor *Batch_bias, Tensor *out);