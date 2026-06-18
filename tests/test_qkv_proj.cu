#include <cstdio>
#include "hpc_utils.cuh"
#include "qkv_proj.cuh"

// -- VERIFY FUNCTIONS --
// For elements wise verification, use cuGPT::validate(...) func given in ./common/hpc_utils.cu

void test_qkv_proj_append_to_kv() {
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  cudaStream_t stream;
  
  // INPUTS
  float* X_norm;  // [B * seq_len * C]

  // WEIGHTS AND BIASES
  float* w_q; float* b_q; // [C * C], [C]
  float* w_k; float* b_k; // [C * C], [C]
  float* w_v; float* b_v; // [C * C], [C]

  // OUTPUTS
  float* Q;   // [B * seq_len * C]
  float* K;   // [B * seq_len * C]
  float* V;   // [B * seq_len * C]
  float* Q_cpu;   // [B * seq_len * C]
  float* K_cpu;   // [B * seq_len * C]
  float* V_cpu;   // [B * seq_len * C]
 
  int B = 2;
  int seq_len = 32;
  int C = 64;

  size_t IO_size = B * seq_len * C * sizeof(float);
  size_t weight_size = C * C * sizeof(float);
  size_t bias_size = C * sizeof(float);

  // Memory Allocation
  CUDA_CHECK(cudaMallocManaged((void**)&X_norm, IO_size));
  CUDA_CHECK(cudaMallocManaged((void**)&Q, IO_size));
  CUDA_CHECK(cudaMallocManaged((void**)&K, IO_size));
  CUDA_CHECK(cudaMallocManaged((void**)&V, IO_size));
  CUDA_CHECK(cudaStreamCreate(&stream));  
  Q_cpu = (float*)malloc(IO_size);
  K_cpu = (float*)malloc(IO_size);
  V_cpu = (float*)malloc(IO_size);

  CUDA_CHECK(cudaMallocManaged((void**)&w_q, weight_size));
  CUDA_CHECK(cudaMallocManaged((void**)&w_k, weight_size));
  CUDA_CHECK(cudaMallocManaged((void**)&w_v, weight_size));

  CUDA_CHECK(cudaMallocManaged((void**)&b_q, bias_size));
  CUDA_CHECK(cudaMallocManaged((void**)&b_k, bias_size));
  CUDA_CHECK(cudaMallocManaged((void**)&b_v, bias_size));
  

  // INIT INPUT, WEIGHT AND BIAS
  cuGPT::initMatrix(X_norm, B * seq_len, C);
  cuGPT::initMatrix(w_q, C, C);
  cuGPT::initMatrix(w_k, C, C);
  cuGPT::initMatrix(w_v, C, C);

  cuGPT::initMatrix(b_q, 1, C);
  cuGPT::initMatrix(b_k, 1, C);
  cuGPT::initMatrix(b_v, 1, C);

  // CPU GPU RUNS
  fprintf(stderr, "Running CPU_QKV_PROJ: ");
  cpu_proj_append_to_KV_cache(
    X_norm, 
    w_q, w_k, w_v,
    b_q, b_k, b_v,
    K_cpu, V_cpu, Q_cpu,
    B, seq_len, C);
  fprintf(stderr, "DONE!\n");

  fprintf(stderr, "Running GPU_QKV_PROJ: ");
  qkv_proj_append_to_KV_cache(
    cublas_handle,
    X_norm,
    w_q, w_k, w_v,
    b_q, b_k, b_v,
    K, V, Q,
    B, seq_len, C,
    stream
  );
  fprintf(stderr, "DONE!\n");
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Compare outputs
  int num_elements = IO_size / sizeof(float);
  if (cuGPT::validate(Q, Q_cpu, num_elements)) {
    printf("Succes: query_scratch matches!\n");
  } else {
    printf("Error: mismatch in query_scratch\n");
  }

  if (cuGPT::validate(K, K_cpu, num_elements)) {
    printf("Succes: key_cache matches!\n");
  } else {
    printf("Error: mismatch in key_cache\n");
  }

  if (cuGPT::validate(V, V_cpu, num_elements)) {
    printf("Succes: value_cache matches!\n");
  } else {
    printf("Error: mismatch in value_cache\n");
  }

  // FREE MEMORY ALLOCATION
  CUDA_CHECK(cudaFree(X_norm));
  CUDA_CHECK(cudaFree(w_q));
  CUDA_CHECK(cudaFree(w_k));
  CUDA_CHECK(cudaFree(w_v));
  CUDA_CHECK(cudaFree(b_q));
  CUDA_CHECK(cudaFree(b_k));
  CUDA_CHECK(cudaFree(b_v));
  CUDA_CHECK(cudaFree(Q));
  CUDA_CHECK(cudaFree(K));
  CUDA_CHECK(cudaFree(V));
  free(Q_cpu);
  free(K_cpu);
  free(V_cpu);

  // DESTROY STREAM
  CUDA_CHECK(cudaStreamDestroy(stream));
  cublasDestroy(cublas_handle);
}

int main(void) {
  printf("TEST_QKV_PROJ...\n");
  test_qkv_proj_append_to_kv();
}