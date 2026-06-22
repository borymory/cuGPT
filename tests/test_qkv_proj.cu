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
  float* X_norm;  // [B * current_seq_len * C]

  // WEIGHTS AND BIASES
  float* w_q; float* b_q; // [C * C], [C]
  float* w_k; float* b_k; // [C * C], [C]
  float* w_v; float* b_v; // [C * C], [C]

  // OUTPUTS
  float* Q;   // [B * current_seq_len * C]
  float* K;   // [max_B * max_T * C]
  float* V;   // [max_B * max_T * C]
  float* Q_cpu;   // [B * current_seq_len * C]
  float* K_cpu;   // [max_B * max_T * C]
  float* V_cpu;   // [max_B * max_T * C]
 
  int B = 3;
  int current_seq_len = 32;
  
  int max_B = 5;
  int max_T = 64;
  int C = 64;

  size_t X_size = B * current_seq_len * C * sizeof(float);
  size_t KV_size = max_B * max_T * C * sizeof(float);
  size_t Q_size = max_B * max_T * C * sizeof(float);
  size_t weight_size = C * C * sizeof(float);
  size_t bias_size = C * sizeof(float);

  // Memory Allocation
  CUDA_CHECK(cudaMallocManaged((void**)&X_norm, X_size));
  CUDA_CHECK(cudaMallocManaged((void**)&Q, Q_size));
  CUDA_CHECK(cudaMallocManaged((void**)&K, KV_size));
  CUDA_CHECK(cudaMallocManaged((void**)&V, KV_size));
  CUDA_CHECK(cudaStreamCreate(&stream));  
  Q_cpu = (float*)malloc(Q_size);
  K_cpu = (float*)malloc(KV_size);
  V_cpu = (float*)malloc(KV_size);

  CUDA_CHECK(cudaMallocManaged((void**)&w_q, weight_size));
  CUDA_CHECK(cudaMallocManaged((void**)&w_k, weight_size));
  CUDA_CHECK(cudaMallocManaged((void**)&w_v, weight_size));

  CUDA_CHECK(cudaMallocManaged((void**)&b_q, bias_size));
  CUDA_CHECK(cudaMallocManaged((void**)&b_k, bias_size));
  CUDA_CHECK(cudaMallocManaged((void**)&b_v, bias_size));
  

  // INIT INPUT, WEIGHT AND BIAS
  cuGPT::initMatrix(X_norm, B * current_seq_len, C);
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
    B, current_seq_len, C,
    max_T
  );
  fprintf(stderr, "DONE!\n");

  fprintf(stderr, "Running GPU_QKV_PROJ: ");
  qkv_proj_append_to_KV_cache(
    cublas_handle,
    X_norm,
    w_q, w_k, w_v,
    b_q, b_k, b_v,
    K, V, Q,
    B, current_seq_len, C,
    max_T,
    stream
  );
  fprintf(stderr, "DONE!\n");
  CUDA_CHECK(cudaDeviceSynchronize());

  int Q_num_elements = B * current_seq_len * C;
  if(cuGPT::validate(Q, Q_cpu, Q_num_elements)) {
    printf("Query_cache matches!\n");
  } else {
    printf("Error: mismatch in query_cache\n");
  }
  
  // Compare outputs
  int KV_num_elements = current_seq_len * C;
  for (int b = 0; b < B; ++b) {
    int batch_offset = b * (max_T * C);

    float* K_batch = K + batch_offset;
    printf("Current batch_offset=%d\n", batch_offset);
    float* K_batch_cpu = K_cpu + batch_offset;
    if (cuGPT::validate(K_batch, K_batch_cpu, KV_num_elements)) {
      printf("Succes: Key_cache matches!\n");
    } else {
      printf("Error: Mismatch first seen in key_cache, at batch b=%d\n", b);
      break;
    }
  }

  for (int b = 0; b < B; ++b) {
    int batch_offset = b * (max_T * C);

    float* V_batch = V + batch_offset;
    float* V_batch_cpu = V_cpu + batch_offset;
    printf("Current batch_offset=%d\n", batch_offset);
    if (cuGPT::validate(V_batch, V_batch_cpu, KV_num_elements)) {
      printf("Succes: Value_cache matches!\n");
    } else {
      printf("Error: Mismatch first seen in value_cache, at batch b=%d\n", b);
      break;
    }
  }
  size_t KV_utilized = (size_t)KV_num_elements * B * 2 * sizeof(float);
  float perc_usage = ((float)KV_utilized / (2 * KV_size)) * 100;
  printf("KV_Size (Bytes): %zu\n", 2 * KV_size);
  printf("KV_Size Utilized (Bytes): %zu\n", KV_utilized);
  printf("Total KV Cache usage: %f\n", perc_usage);

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