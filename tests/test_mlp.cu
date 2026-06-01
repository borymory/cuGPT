#include <cstdio>
#include "hpc_utils.cuh"
#include "mlp.cuh"

// -- VERIFY FUNCTIONS --
// For elements wise verification, use cuGPT::validate(...) func given in ./common/hpc_utils.cu

bool test_fused_bias_ReLU_v1() {
  cudaStream_t stream;
  
  float *h_out;
  float *b1;
  float *h_out_cpu;

  int B = 32;
  int T = 32;
  int C = 512;

  
  // USE UNIFIED MEMORY - INITIALIZATIONS
  CUDA_CHECK(cudaMallocManaged(&h_out, B * T * 4 * C * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&b1, 4 * C * sizeof(float)));
  h_out_cpu = (float*)std::malloc(B * T * 4 * C * sizeof(float));
  CUDA_CHECK(cudaStreamCreate(&stream));

  // -- BENCHMARK STATISTICS --
  

  // -- BENCHMARK KERNEL RUN --
  

  // -- VERIFY KERNEL RUN --
  cuGPT::initMatrix(h_out, B*T, 4*C);
  cuGPT::initMatrix(b1, 1, 4*C);
  cpu_bias_ReLU(h_out, h_out_cpu, b1, B*T, C);            // Obtain CPU result
  launch_fused_bias_ReLU_v1 (h_out, b1, B*T, C, stream);  // Obtain GPU result
  CUDA_CHECK(cudaDeviceSynchronize());
  bool isExact = cuGPT::validate(h_out, h_out_cpu, B*T*4*C);

  // FREE MEMORY ALLOCATION
  CUDA_CHECK(cudaFree(h_out));
  CUDA_CHECK(cudaFree(b1));
  std::free(h_out_cpu);

  // DESTROY STREAM
  CUDA_CHECK(cudaStreamDestroy(stream));

  return isExact; // VERIFY KERNEL
}

bool test_fused_bias_residual_v1() {
  cudaStream_t stream;
  
  float *h_out;
  float *X;
  float *b2;
  float *h_out_cpu;

  int B = 32;
  int T = 32;
  int C = 512;

  
  // USE UNIFIED MEMORY - INITIALIZATIONS
  CUDA_CHECK(cudaMallocManaged(&h_out, B * T * C * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&X, B * T * C * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&b2, C * sizeof(float)));
  h_out_cpu = (float*)std::malloc(B * T * C * sizeof(float));
  CUDA_CHECK(cudaStreamCreate(&stream));

  // -- BENCHMARK STATISTICS --
  

  // -- BENCHMARK KERNEL RUN --
  

  // -- VERIFY KERNEL RUN --
  cuGPT::initMatrix(h_out, B*T, C);
  cuGPT::initMatrix(X, B*T, C);
  cuGPT::initMatrix(b2, 1, C);
  cpu_bias_residual(X, h_out, h_out_cpu, b2, B*T, C);           // Obtain CPU result
  launch_fused_bias_residual_v1 (X, h_out, b2, B*T, C, stream); // Obtain GPU result
  CUDA_CHECK(cudaDeviceSynchronize());
  bool isExact = cuGPT::validate(h_out, h_out_cpu, B*T*C);

  // FREE MEMORY ALLOCATION
  CUDA_CHECK(cudaFree(h_out));
  CUDA_CHECK(cudaFree(X));
  CUDA_CHECK(cudaFree(b2));
  std::free(h_out_cpu);

  // DESTROY STREAM
  CUDA_CHECK(cudaStreamDestroy(stream));

  return isExact; // VERIFY KERNEL
}

int main(void) {
  std::printf("Running MLP Test...\n");
  
  string kernel_name = "fused_bias_residual_v1";
  if (test_fused_bias_residual_v1()) std::printf("Kernel Name: %s | Status: Succes!\n", kernel_name);
}