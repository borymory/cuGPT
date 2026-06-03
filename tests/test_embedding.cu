#include <cstdio>
#include "hpc_utils.cuh"
#include "embedding.cuh"

// -- VERIFY FUNCTIONS --
// For elements wise verification, use cuGPT::validate(...) func given in ./common/hpc_utils.cu

bool test_embedding_v1() {
  cudaStream_t stream;
  
  int *inputs;
  float *out;
  float *wte;
  float *wpe;
  float *out_cpu;

  int B = 32;
  int T = 32;
  int C = 512;
  int vocab_size = 50257;
  int max_length = 1024;

  
  // USE UNIFIED MEMORY - INITIALIZATIONS
  CUDA_CHECK(cudaMallocManaged(&inputs, B * T * sizeof(int)));
  CUDA_CHECK(cudaMallocManaged(&out, B * T * C * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&wte, vocab_size * C * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&wpe, max_length * C * sizeof(float)));
  out_cpu = (float*)std::malloc(B * T * C * sizeof(float));
  CUDA_CHECK(cudaStreamCreate(&stream));

  // -- VERIFY KERNEL RUN --
  for (unsigned int i = 0; i < B * T; ++i) {
    inputs[i] = std::rand() % vocab_size;
  }
  cuGPT::initMatrix(wte, vocab_size, C);
  cuGPT::initMatrix(wpe, max_length, C);

  std::printf("Running CPU Embedding... | ");
  cpu_embedding(inputs, out, wte, wpe, B, T, C);
  std::printf("✅ CPU Embedding Finished\n");

  std::printf("Running GPU Embedding... | ");
  launch_embedding_v1(inputs, out, wte, wpe, B, T, C, max_length, stream);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::printf("✅ GPU Embedding Finished\n");

  bool isExact = cuGPT::validate(out, out_cpu, B * T * C);

  // FREE MEMORY ALLOCATION
  CUDA_CHECK(cudaFree(inputs));
  CUDA_CHECK(cudaFree(out));
  CUDA_CHECK(cudaFree(wte));
  CUDA_CHECK(cudaFree(wpe));
  std::free(out_cpu);

  // DESTROY STREAM
  CUDA_CHECK(cudaStreamDestroy(stream));

  return isExact; // VERIFY KERNEL
}

int main(void) {
  std::printf("Running Test...\n");
  if (test_embedding_v1()) std::printf("Succes!\n");
}