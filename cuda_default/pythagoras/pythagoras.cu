#include <cstdio>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

__global__ void pythagoras(int *pa, int *pb, int *pc, int *presult) {
  int a = *pa;
  int b = *pb;
  int c = *pc;

  if ((a * a + b * b) == c * c)
    *presult = 1;
  else
    *presult = 0;
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <num 1> <num 2> <num 3>\n", argv[0]);
    return 0;
  }

  int a = atoi(argv[1]);
  int b = atoi(argv[2]);
  int c = atoi(argv[3]);
  int result = 0;

  // TODO: 1. allocate device memory
  int *dev_a, *dev_b, *dev_c, *dev_result;
  CHECK_CUDA(cudaMalloc(&dev_a, sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dev_b, sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dev_c, sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dev_result, sizeof(int)));


  // TODO: 2. copy data to device
  CHECK_CUDA(cudaMemcpy(dev_a, &a, sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dev_b, &b, sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dev_c, &c, sizeof(int), cudaMemcpyHostToDevice));

  // TODO: 3. launch kernel
  pythagoras<<<1,1>>>(dev_a, dev_b, dev_c, dev_result);


  // TODO: 4. copy result back to host
  CHECK_CUDA(cudaMemcpy(&result, dev_result, sizeof(int), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dev_a));
  CHECK_CUDA(cudaFree(dev_b));
  CHECK_CUDA(cudaFree(dev_c));
  CHECK_CUDA(cudaFree(dev_result));

  if (result) printf("YES\n");
  else printf("NO\n");

  return 0;
}
