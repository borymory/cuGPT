#include <cstdio>
#include "hpc_utils.cuh"
#include "attention.cuh"

// -- VERIFY FUNCTIONS --
// For elements wise verification, use cuGPT::validate(...) func given in ./common/hpc_utils.cu

bool test_flashattn_fwd() {
  cudaStream_t stream;
  
  float* Q;
  float* K;
  float* V;
  float* O;

  float* O_cpu;
  float* S_cpu;

  int B = 2;
  int H = 12;
  int seq_len = 64; // (N)
  int d = 64;

  size_t tensor_size = B * H * seq_len * d;
  size_t scratch_size = seq_len * seq_len;

  size_t tensor_size_bytes = tensor_size * sizeof(float);
  size_t scratch_size_bytes = scratch_size * sizeof(float);

  // Memory Allocation
  CUDA_CHECK(cudaMallocManaged((void**)&Q, tensor_size_bytes));
  CUDA_CHECK(cudaMallocManaged((void**)&K, tensor_size_bytes));
  CUDA_CHECK(cudaMallocManaged((void**)&V, tensor_size_bytes));
  CUDA_CHECK(cudaMallocManaged((void**)&O, tensor_size_bytes));
  CUDA_CHECK(cudaStreamCreate(&stream));  
  O_cpu = (float*)malloc(tensor_size_bytes);
  S_cpu = (float*)malloc(scratch_size_bytes);
  

  // CPU, GPU Runs
  cuGPT::initMatrix(Q, B * H * seq_len, d);
  cuGPT::initMatrix(K, B * H * seq_len, d);
  cuGPT::initMatrix(V, B * H * seq_len, d);

  fprintf(stderr, "Running CPU_ATTENTION: ");
  cpu_attention(Q, K, V, S_cpu, O_cpu, B, H, seq_len, d);
  fprintf(stderr, "DONE!\n");
  fprintf(stderr, "Running GPU_FLASHATTN_FWD: ");
  launch_flash_attn_forward_kernel(Q, K, V, O, B, H, seq_len, d, stream);
  fprintf(stderr, "DONE!\n");
  CUDA_CHECK(cudaDeviceSynchronize());
  bool isExact = cuGPT::validate(O, O_cpu, tensor_size);

  // FREE MEMORY ALLOCATION
  CUDA_CHECK(cudaFree(Q));
  CUDA_CHECK(cudaFree(K));
  CUDA_CHECK(cudaFree(V));
  CUDA_CHECK(cudaFree(O));
  free(O_cpu);
  free(S_cpu);

  // DESTROY STREAM
  CUDA_CHECK(cudaStreamDestroy(stream));

  return isExact; // VERIFY KERNEL
}

int main(void) {
  std::printf("Running Test...\n");
  if (test_flashattn_fwd()) std::printf("Succes!\n");
}