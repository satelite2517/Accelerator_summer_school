#include <cstdio>

#include "layer.h"
#include "model.h"


#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

/* [Model Parameters]
 * _w: Weight parameter
 * _b: Bias parameter
 */
Parameter *mlp1_w, *mlp1_b;
Parameter *mlp2_w, *mlp2_b;
Parameter *convtrans1_w, *convtrans1_b;
Parameter *batchnorm1_w, *batchnorm1_b;
Parameter *convtrans2_w, *convtrans2_b;
Parameter *batchnorm2_w, *batchnorm2_b;
Parameter *convtrans3_w, *convtrans3_b;
Parameter *batchnorm3_w, *batchnorm3_b;
Parameter *convtrans4_w, *convtrans4_b;
Parameter *batchnorm4_w, *batchnorm4_b;
Parameter *convtrans5_w, *convtrans5_b;
Parameter *batchnorm5_w, *batchnorm5_b;
Parameter *convtrans6_w, *convtrans6_b;
Parameter *batchnorm6_w, *batchnorm6_b;
Parameter *conv_w, *conv_b;

void alloc_and_set_parameters(float *param, size_t param_size) {
  size_t pos = 0;

  	mlp1_w = new Parameter({16384, 128}, param + pos);
	pos += 16384 * 128;
	mlp1_b = new Parameter({16384}, param + pos);
	pos += 16384;
	
	mlp2_w = new Parameter({4096, 16384}, param + pos);
	pos += 4096 * 16384;
	mlp2_b = new Parameter({4096}, param + pos);
	pos += 4096;

	convtrans1_w = new Parameter({1024, 512, 3, 3}, param + pos);
	pos += 1024 * 512 * 3 * 3;
	convtrans1_b = new Parameter({512}, param + pos);
	pos += 512;
	batchnorm1_w = new Parameter({512}, param + pos);
	pos += 512;
	batchnorm1_b = new Parameter({512}, param + pos);
	pos += 512;
	
	convtrans2_w = new Parameter({512, 256, 3, 3}, param + pos);
	pos += 512 * 256 * 3 * 3;
	convtrans2_b = new Parameter({256}, param + pos);
	pos += 256;
	batchnorm2_w = new Parameter({256}, param + pos);
	pos += 256;
	batchnorm2_b = new Parameter({256}, param + pos);
	pos += 256;

	convtrans3_w = new Parameter({256, 128, 3, 3}, param + pos);
	pos += 256 * 128 * 3 * 3;
	convtrans3_b = new Parameter({128}, param + pos);
	pos += 128;
	batchnorm3_w = new Parameter({128}, param + pos);
	pos += 128;
	batchnorm3_b = new Parameter({128}, param + pos);
	pos += 128;

	convtrans4_w = new Parameter({128, 64, 3, 3}, param + pos);
	pos += 128 * 64 * 3 * 3;
	convtrans4_b = new Parameter({64}, param + pos);
	pos += 64;
	batchnorm4_w = new Parameter({64}, param + pos);
	pos += 64;
	batchnorm4_b = new Parameter({64}, param + pos);
	pos += 64;

	convtrans5_w = new Parameter({64, 32, 3, 3}, param + pos);
	pos += 64 * 32 * 3 * 3;
	convtrans5_b = new Parameter({32}, param + pos);
	pos += 32;
	batchnorm5_w = new Parameter({32}, param + pos);
	pos += 32;
	batchnorm5_b = new Parameter({32}, param + pos);
	pos += 32;

	convtrans6_w = new Parameter({32, 32, 3, 3}, param + pos);
	pos += 32 * 32 * 3 * 3;
	convtrans6_b = new Parameter({32}, param + pos);
	pos += 32;
	batchnorm6_w = new Parameter({32}, param + pos);
	pos += 32;
	batchnorm6_b = new Parameter({32}, param + pos);
	pos += 32;

	conv_w = new Parameter({3, 32, 3, 3}, param + pos);
	pos += 3 * 32 * 3 * 3;
	conv_b = new Parameter({3}, param + pos);
	pos += 3;
	
	if (pos != param_size) {
		fprintf(stderr, "Parameter size mismatched: %zu vs %zu\n", pos, param_size);
		exit(1);
	}
}

void free_parameters() {
	delete mlp1_w;
	delete mlp1_b;
	delete mlp2_w;
	delete mlp2_b;
	delete convtrans1_w;
	delete convtrans1_b;
	delete batchnorm1_w;
	delete batchnorm1_b;
	delete convtrans2_w;
	delete convtrans2_b;
	delete batchnorm2_w;
	delete batchnorm2_b;
	delete convtrans3_w;
	delete convtrans3_b;
	delete batchnorm3_w;
	delete batchnorm3_b;
	delete convtrans4_w;
	delete convtrans4_b;
	delete batchnorm4_w;
	delete batchnorm4_b;
	delete convtrans5_w;
	delete convtrans5_b;
	delete batchnorm5_w;
	delete batchnorm5_b;
	delete convtrans6_w;
	delete convtrans6_b;
	delete batchnorm6_w;
	delete batchnorm6_b;
	delete conv_w;
	delete conv_b;
}

/* [Model Activations] 
 * _a: Activation buffer
 */
Activation *linear1_a, *linear2_a;
Activation *reshape_a;
Activation *convtrans1_a, *batchnorm1_a;
Activation *convtrans2_a, *batchnorm2_a;
Activation *convtrans3_a, *batchnorm3_a;
Activation *convtrans4_a, *batchnorm4_a;
Activation *convtrans5_a, *batchnorm5_a;
Activation *convtrans6_a, *batchnorm6_a;
Activation *conv_a;

#define IMAGE_CHUNK 128
void alloc_activations() {
  	linear1_a = new Activation({IMAGE_CHUNK, 16384});
	linear2_a = new Activation({IMAGE_CHUNK, 4096});
	reshape_a = new Activation({IMAGE_CHUNK, 1024, 2, 2});
	convtrans1_a = new Activation({IMAGE_CHUNK, 512, 4, 4});
	batchnorm1_a = new Activation({IMAGE_CHUNK, 512, 4, 4});
	convtrans2_a = new Activation({IMAGE_CHUNK, 256, 8, 8});
	batchnorm2_a = new Activation({IMAGE_CHUNK, 256, 8, 8});
	convtrans3_a = new Activation({IMAGE_CHUNK, 128, 16, 16});
	batchnorm3_a = new Activation({IMAGE_CHUNK, 128, 16, 16});
	convtrans4_a = new Activation({IMAGE_CHUNK, 64, 32, 32});
	batchnorm4_a = new Activation({IMAGE_CHUNK, 64, 32, 32});
	convtrans5_a = new Activation({IMAGE_CHUNK, 32, 64, 64});
	batchnorm5_a = new Activation({IMAGE_CHUNK, 32, 64, 64});
	convtrans6_a = new Activation({IMAGE_CHUNK, 32, 128, 128});
	batchnorm6_a = new Activation({IMAGE_CHUNK, 32, 128, 128});
	conv_a = new Activation({IMAGE_CHUNK, 3, 128, 128});
}

void free_activations() {
  	delete linear1_a;
	delete linear2_a;
	delete reshape_a;
	delete convtrans1_a;
	delete batchnorm1_a;
	delete convtrans2_a;
	delete batchnorm2_a;
	delete convtrans3_a;
	delete batchnorm3_a;
	delete convtrans4_a;
	delete batchnorm4_a;
	delete convtrans5_a;
	delete batchnorm5_a;
	delete convtrans6_a;
	delete batchnorm6_a;
	delete conv_a;
}

/* [Model Computation: Image Generation] */
void generate_images(float *input, float *output, size_t n_img) {

    size_t image_chunk = IMAGE_CHUNK;  // IMAGE_CHUNK variable
    cudaStream_t kernel_stream, data_stream;
    cudaStreamCreate(&kernel_stream);
    cudaStreamCreate(&data_stream);

    cudaEvent_t kernel_end, memcpy_end;
    cudaEventCreate(&kernel_end);
    cudaEventCreate(&memcpy_end);

    // Allocate pinned memory for output
    float *pinned_output;
    cudaMallocHost((void**)&pinned_output, n_img * 3 * 128 * 128 * sizeof(float));

    /* Generate images for each chunk of latent vectors in the input */
    for (size_t n = 0; n < n_img; n += image_chunk) {

        /* Calculate the number of images in the current chunk */
        size_t current_chunk_size = (n + image_chunk <= n_img) ? image_chunk : (n_img - n);

        /* Initialize input latent vectors z [current_chunk_size, LATENT_DIM] */
        Tensor *z = new Tensor({current_chunk_size, LATENT_DIM});
        memcpy(z->buf, input + n * LATENT_DIM, current_chunk_size * LATENT_DIM * sizeof(float));

        data_upload(z);

        Linear_cuda(z, mlp1_w, mlp1_b, linear1_a, kernel_stream);
        Linear_cuda(linear1_a, mlp2_w, mlp2_b, linear2_a, kernel_stream);
        Reshape_cuda(linear2_a, reshape_a, kernel_stream);
        ConvTran_Batch_ReLU_fusion_cuda(reshape_a, convtrans1_w, convtrans1_b, convtrans1_a, batchnorm1_w, batchnorm1_b, batchnorm1_a, kernel_stream);
        ConvTran_Batch_ReLU_fusion_cuda(batchnorm1_a, convtrans2_w, convtrans2_b, convtrans2_a, batchnorm2_w, batchnorm2_b, batchnorm2_a, kernel_stream);
        ConvTran_Batch_ReLU_fusion_cuda(batchnorm2_a, convtrans3_w, convtrans3_b, convtrans3_a, batchnorm3_w, batchnorm3_b, batchnorm3_a, kernel_stream);
        ConvTran_Batch_ReLU_fusion_cuda(batchnorm3_a, convtrans4_w, convtrans4_b, convtrans4_a, batchnorm4_w, batchnorm4_b, batchnorm4_a, kernel_stream);
        ConvTran_Batch_ReLU_fusion_cuda(batchnorm4_a, convtrans5_w, convtrans5_b, convtrans5_a, batchnorm5_w, batchnorm5_b, batchnorm5_a, kernel_stream);
        ConvTran_Batch_ReLU_fusion_cuda(batchnorm5_a, convtrans6_w, convtrans6_b, convtrans6_a, batchnorm6_w, batchnorm6_b, batchnorm6_a, kernel_stream);
        
        if (n != 0) cudaStreamWaitEvent(kernel_stream, memcpy_end, 0);

        Conv2d_Tanh_fusion_cuda(batchnorm6_a, conv_w, conv_b, conv_a, kernel_stream);
        cudaEventRecord(kernel_end, kernel_stream);

        cudaStreamWaitEvent(data_stream, kernel_end, 0);
        CHECK_CUDA(cudaMemcpyAsync(pinned_output + n * 3 * 128 * 128, conv_a->gpu_buf, current_chunk_size * 3 * 128 * 128 * sizeof(float), cudaMemcpyDeviceToHost, data_stream));
        cudaEventRecord(memcpy_end, data_stream);

        /* Free the input latent vector z */
        delete z;
    }

    // Wait for all operations to complete
    cudaStreamSynchronize(data_stream);
    cudaStreamSynchronize(kernel_stream);

    // Copy from pinned memory to the final output
    memcpy(output, pinned_output, n_img * 3 * 128 * 128 * sizeof(float));

    // Clean up pinned memory, streams, and events
    cudaFreeHost(pinned_output);
    cudaEventDestroy(kernel_end);
    cudaEventDestroy(memcpy_end);
    cudaStreamDestroy(data_stream);
    cudaStreamDestroy(kernel_stream);
}