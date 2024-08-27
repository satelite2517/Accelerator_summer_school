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

int main() {
  int count;
  CHECK_CUDA(cudaGetDeviceCount(&count));

  printf("Number of devices: %d\n", count);
  cudaDeviceProp props[4];
  for (int i = 0; i < count; ++i) {
    printf("\tdevice %d:\n", i);

    // Fetch and print device properties
    CHECK_CUDA(cudaGetDeviceProperties(&props[i], i));

    printf("  \tName: %s\n", props[i].name);
    printf("  \tSM count: %d\n", props[i].multiProcessorCount);
    printf("  \tMaxThreadsPerBlock: %d\n", props[i].maxThreadsPerBlock);
    printf("  \tTotal Global Memory: %lu bytes\n", props[i].totalGlobalMem);
    printf("  \tShared Memory Per Block: %lu bytes\n\n", props[i].sharedMemPerBlock);
  }

  return 0;
}
