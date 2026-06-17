#include "hpc_utils.cuh"
#include "qkv_proj.cuh"

// CPU Func



//
// GPU Kernel
//
__global__ void proj_bias_add (
    float* __restrict__ d_out,
    const float* __restrict__ d_bias,
    const int B,
    const int current_seq_len,
    const int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_elements = B * current_seq_len * C;

    if (idx < num_elements) {
        int col = idx % C;  // Broadcast: corresponsing bias entry
        d_out[idx] += d_bias[col];
    }
}

//
// Kernel Wrappers
//
void launch_proj_bias_add (
    float* __restrict__ d_out,
    const float* __restrict__ d_bias,
    const int B,
    const int current_seq_len,
    const int C,
    cudaStream_t stream)
{
    int total_elements = B * current_seq_len * C;
    int block_size = 256;
    int grid_size = CEIL_DIV(total_elements, block_size);

    proj_bias_add<<<grid_size, block_size, 0, stream>>>(d_out, d_bias, B, current_seq_len, C);
    CHECK_LAST_CUDA_ERROR();
}

// L dimension is handled by the pointer arithmetic.
void qkv_proj_append_to_KV_cache(
    cublasHandle_t cublas_handle,
    const float* __restrict__ X_norm, // [B * seq_len, C]
    const float* __restrict__ w_q, // [C, C]
    const float* __restrict__ w_k, // [C, C]
    const float* __restrict__ w_v, // [C, C]
    const float* __restrict__ b_q, // [C]
    const float* __restrict__ b_k, // [C]
    const float* __restrict__ b_v, // [C]
    float* __restrict__ key_cache,  // Written: [B, seq_len, C]
    float* __restrict__ value_cache, // Written: [B, seq_len, C]
    float* __restrict__ q_scratch, // Written: [B, seq_len, C]
    const int B,
    const int current_seq_len,
    const int C,
    cudaStream_t stream
)
{
    cublasSetStream(cublas_handle, stream);

    // QKV Proj
   cuGPT::gemm(cublas_handle, X_norm, w_q, q_scratch, B * current_seq_len, C, C);
   cuGPT::gemm(cublas_handle, X_norm, w_k, key_cache, B * current_seq_len, C, C);
   cuGPT::gemm(cublas_handle, X_norm, w_v, value_cache, B * current_seq_len, C, C);

   // QKV Bias
   launch_proj_bias_add(q_scratch, b_q, B, current_seq_len, C, stream);
   launch_proj_bias_add(key_cache, b_k, B, current_seq_len, C, stream);
   launch_proj_bias_add(value_cache, b_v, B, current_seq_len, C, stream);
}