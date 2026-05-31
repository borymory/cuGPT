#include <cstdio>
#include "hpc_utils.cuh"
#include "mlp.cuh"
#include "tests.cuh"

// -- VERIFY FUNCTIONS --
// For elements wise verification, use validate func given in ./common/hpc_utils.cu

bool test_fused_bias_ReLU_v1() {
  cudaStream_t stream;
  
  float *h_out;
  float *b1;
  float *h_out_cpu;

  int B = 32;
  int T = 32;
  int C = 512;

  
  // USE UNIFIED MEMORY - INITIALIATONS
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

  if (isExact) return true; // VERIFY KERNEL
  return false;
}