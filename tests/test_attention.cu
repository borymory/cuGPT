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

  // Context
  int B = 3;
  int current_seq_len = 3; // (N)
  
  // Config
  int max_B = 5;
  int max_T = 5;
  int C = 768
  int H = 12;
  int d = 64;
  
  size_t KV_size = max_B * max_T * C * sizeof(float);
  size_t Q_size = max_B * max_T * C * sizeof(float);
  size_t scratch_size = current_seq_len * current_seq_len * sizeof(float);

  // Memory Allocation
  CUDA_CHECK(cudaMallocManaged((void**)&Q, Q_size));
  CUDA_CHECK(cudaMallocManaged((void**)&K, KV_size));
  CUDA_CHECK(cudaMallocManaged((void**)&V, KV_size));
  CUDA_CHECK(cudaMallocManaged((void**)&O, Q_size));
  CUDA_CHECK(cudaStreamCreate(&stream));  
  O_cpu = (float*)malloc(Q_size);
  S_cpu = (float*)malloc(scratch_size);
  

  // CPU, GPU Runs
  for (int b = 0; b < B; ++b) {
    int KV_offset = b * (max_T * C);
    float* K_batch = K + KV_offset;
    float* V_batch = V + KV_offset;
    cuGPT::initMatrix(K_batch, current_seq_len, C);
    cuGPT::initMatrix(V_batch, current_seq_len, C);
  }
  cuGPT::initMatrix(Q, B * current_seq_len, C);

  fprintf(stderr, "Running CPU_ATTENTION: ");
  cpu_attention(Q, K, V, S_cpu, O_cpu, B, H, current_seq_len, d, max_T);
  fprintf(stderr, "DONE!\n");
  fprintf(stderr, "Running GPU_FLASHATTN_FWD: ");
  launch_flash_mha_fwd_v1(Q, K, V, O, B, H, current_seq_len, d, max_T, stream);
  fprintf(stderr, "DONE!\n");
  CUDA_CHECK(cudaDeviceSynchronize());
  if(cuGPT::validate(O, O_cpu, Q_size / sizeof(float))){
    printf("Succes! Outputs match!");
  }

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
  test_flashattn_fwd();
}