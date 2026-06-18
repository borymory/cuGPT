#include "hpc_utils.cuh"
#include "qkv_proj.cuh"

// CPU Func
void cpu_proj_append_to_KV_cache(
    float* X_norm, // [B * seq_len, C]
    float* w_q, // [C, C]
    float* w_k, // [C, C]
    float* w_v, // [C, C]
    float* b_q, // [C]
    float* b_k, // [C]
    float* b_v, // [C]
    float* key_cache,  // Written: [B, seq_len, C]
    float* value_cache, // Written: [B, seq_len, C]
    float* q_scratch, // Written: [B, seq_len, C]
    int B,
    int current_seq_len,
    int C
) 
{
    // When appending caches for different batches, 
    // each batch's KV cache must be stored at an offset of
    // b_idx * current_seq_len * C
    // from the base KV_cache.

    // We do not have to treat batches independently
    // and can process them as a single huge chunk, 
    // thus can share the whole KV cache buffer together

    // For the sake of explicitness, I am writing with a batch offset to see
    // whether my kernel behaves exact to this behaviour.
    // One problem is, all batches must have exact seq_len.
    for (int b = 0; b < B; ++b) {
        int batch_offset = b * (current_seq_len * C);
        float* X_norm_local = X_norm + batch_offset;
        float* key_cache_local = key_cache + batch_offset;
        float* value_cache_local = value_cache + batch_offset;
        float* q_scratch_local = q_scratch + batch_offset;

        for (int r = 0; r < seq_len; ++r) {
            for (int c = 0; c < C; ++c) {

                float q_sum = 0.0f;
                float k_sum = 0.0f;
                float v_sum = 0.0f;
                for (int k = 0; k < C; ++k) {
                    float X_val = X_norm_local[r * C + k];
                    float wq_val = w_q[k * C + c];
                    float wk_val = w_k[k * C + c];
                    float wv_val = w_v[k * C + c];

                    q_sum += X_val * wq_val;
                    k_sum += X_val * wk_val;
                    v_sum += X_val * wv_val;
                }
                q_scratch_local[r * C + c] = q_sum + b_q[c];
                key_cache_local[r * C + c] = k_sum + b_k[c];
                value_cache_local[r * C + c] = v_sum + b_v[c];
            }
        }
    }
}
// Validate QKV cache results


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