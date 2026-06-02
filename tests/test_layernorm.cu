#include <cstdio>
#include "hpc_utils.cuh"
#include "layernorm.cuh"

// -- VERIFY FUNCTIONS --
// For elements wise verification, use cuGPT::validate(...) func given in ./common/hpc_utils.cu

bool test_layernorm_forward_v1() {
  cudaStream_t stream;
  
  float *X;
  float *X_norm;
  float *alpha;
  float *beta;
  float *X_norm_cpu;

  int B = 32;
  int T = 32;
  int C = 512;

  
  // USE UNIFIED MEMORY - INITIALIZATIONS
  CUDA_CHECK(cudaMallocManaged(&X, B * T * C * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&X_norm, B * T * C * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&alpha, C * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&beta, C * sizeof(float)));
  X_norm_cpu = (float*)std::malloc(B * T * C * sizeof(float));
  
  CUDA_CHECK(cudaStreamCreate(&stream));

  // -- VERIFY KERNEL RUN --
  cuGPT::initMatrix(X, B * T * C);
  cuGPT::initMatrix(alpha, 1, C);
  cuGPT::initMatrix(beta, 1, C);

  std::printf("Running CPU Layernorm... | ");
  cpu_layernorm_fwd(X, X_norm_cpu, alpha, beta, BT, C);
  std::printf("✅ CPU Layernorm Finished\n");

  std::printf("Running GPU Layernorm Kernel... | ");
  layernorm_forward_v1(X, X_norm, alpha, beta, BT, C, stream);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::printf("✅ GPU Layernorm Finished\n");
  
  bool isExact = cuGPT::validate(X_norm, X_norm_cpu, B * T * C);

  // FREE MEMORY ALLOCATION
  CUDA_CHECK(cudaFree(X));
  CUDA_CHECK(cudaFree(X_norm));
  CUDA_CHECK(cudaFree(alpha));
  CUDA_CHECK(cudaFree(beta));
  std::free(X_norm_cpu);

  // DESTROY STREAM
  CUDA_CHECK(cudaStreamDestroy(stream));

  return isExact; // VERIFY KERNEL
}

int main(void) {
  std::printf("Running Test...\n");
  if (test_layernorm_forward_v1()) std::printf("Succes!\n");
}