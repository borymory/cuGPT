#include <cstdio>
#include "hpc_utils.cuh"
#include "lm_head.cuh"

// -- VERIFY FUNCTIONS --
// For elements wise verification, use cuGPT::validate(...) func given in ./common/hpc_utils.cu

bool test_lm_head_v1() {
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  cudaStream_t stream;
  
  float *X_final
  float *wte;
  float *logits;
  float *logits_cpu;

  int B = 4;
  int T = 256;
  int C = 512;
  int vocab_size = 50257;

  
  // USE UNIFIED MEMORY - INITIALIZATIONS
  CUDA_CHECK(cudaMallocManaged(&X_final, B * T * C * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&wte, vocab_size * C * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&logits, B * T * vocab_size * sizeof(float)));
  logits_cpu = (float*)std::malloc(B * T * vocab_size * sizeof(float));
  CUDA_CHECK(cudaStreamCreate(&stream));

  // -- VERIFY KERNEL RUN --
  cuGPT::initMatrix(X_final, B*T, C);
  cuGPT::initMatrix(wte, vocab_size, C);

  std::printf("Running CPU LM Head... | ");
  cpu_lm_head_fwd(X_final, wte, logits_cpu, B * T, C, vocab_size);
  std::printf("✅ CPU LM Head Finished\n");

  std::printf("Running GPU LM Head... | ");
  lm_head_fwd(cublas_handle, X_final, wte, logits, B * T, C, vocab_size, stream);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::printf("✅ GPU Embedding LM Head\n");

  bool isExact = cuGPT::validate(logits, logits_cpu, B * T * vocab_size);

  // FREE MEMORY ALLOCATION
  CUDA_CHECK(cudaFree(inputs));
  CUDA_CHECK(cudaFree(out));
  CUDA_CHECK(cudaFree(wte));
  CUDA_CHECK(cudaFree(wpe));
  std::free(out_cpu);

  // DESTROY STREAM
  cublasDestroy(cublas_handle);
  CUDA_CHECK(cudaStreamDestroy(stream));

  return isExact; // VERIFY KERNEL
}

int main(void) {
  std::printf("Running Test...\n");
  if (test_lm_head_v1()) std::printf("Succes!\n");
}