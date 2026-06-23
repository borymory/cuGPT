#include "hpc_utils.cuh"
#include "layernorm.cuh"

//
// CPU Reference
//

void cpu_layernorm_fwd(
    float* X,       // [B * current_seq_len, C]
    float* X_norm,  // [B * current_seq_len, C]
    float* alpha,   // [C]
    float* beta,    // [C]
    int B,
    int current_seq_len,
    int C
) {
    int BT = B * current_seq_len;
    const float eps = 1e-5f;    // to prevent divide by zero error

    for (unsigned int i = 0; i < BT; ++i) {

        // find mean
        float sum = 0.0f;
        for (unsigned int k = 0; k < C; ++k){
            sum += X[i * C + k];
        }
        float mean = sum / C;

        // find variance
        sum = 0.0f;
        for (unsigned int k = 0; k < C; ++k){
            float diff = X[i * C + k] - mean;
            sum += diff * diff;
        }
        float std_dev = std::sqrt((sum / C) + eps);

        // update value
        for (unsigned int k = 0; k < C; ++k){
            X_norm[i * C + k] = ((X[i * C + k] - mean) / std_dev) * alpha[k] + beta[k];
        }
    }
}

//
// GPU Kernels
//

template<const int block_BT>
__global__ void layernorm_fwd_v1(
    const float* __restrict__ X,        // [B * current_seq_len, C]
    float* X_norm,                      // [B * current_seq_len, C]
    const float* __restrict__ alpha,    // [C]
    const float* __restrict__ beta,     // [C]
    const int B,
    const int current_seq_len, 
    const int C
) {
    // Launch CEIL_DIV(BT, block_BT) many blocks
    // Launch block_BT * 32 many threads
    const float eps = 1e-5f;    // to prevent divide by zero error

    int tx = threadIdx.x % 32;
    int ty = threadIdx.x / 32;

    // Offset blocks to rows
    int rowIdx = blockIdx.x * block_BT;
    X += rowIdx * C;
    X_norm += rowIdx * C;

    int global_row = ty + rowIdx;
    if (global_row < B * current_seq_len) {

        // Mean Calculation
        float mean = 0.0f;
        for (unsigned int idx = tx; idx < C; idx += 32) {
            mean += X[ty * C + idx];
        }
    
        for (unsigned int mirrorIdx = 1; mirrorIdx <= 16; mirrorIdx <<= 1) {
            mean += __shfl_xor_sync(FULL_MASK, mean, mirrorIdx);
        }
        mean /= C;
        
        // Standard Deviation Calculation
        float std_dev = 0.0f;
        for (unsigned int idx = tx; idx < C; idx += 32) {
            float diff = X[ty * C + idx] - mean;
            std_dev += diff * diff;
        }
    
        for (unsigned int mirrorIdx = 1; mirrorIdx <= 16; mirrorIdx <<= 1) {
            std_dev += __shfl_xor_sync(FULL_MASK, std_dev, mirrorIdx);
        }
        std_dev = sqrtf((std_dev / C) + eps);
    
        // Update and load to X_norm
        for (unsigned int idx = tx; idx < C; idx += 32) {
            X_norm[ty * C + idx] = ((X[ty * C + idx] - mean) / std_dev) * alpha[idx] + beta[idx];
        }
    }
}

//
// Layer norm implementation
//

void layernorm_forward_v1(
    const float* __restrict__ X,        // [B * current_seq_len, C]
    float* X_norm,                      // [B * current_seq_len, C]
    const float* __restrict__ alpha,    // [C]
    const float* __restrict__ beta,     // [C]
    const int B,
    const int current_seq_len, 
    const int C
    cudaStream_t stream
) {
    const int block_BT = 32;
    int block_count = CEIL_DIV(BT, block_BT);
    int thread_count = block_BT * 32;

    layernorm_fwd_v1<block_BT><<<block_count, thread_count, 0, stream>>> (X, X_norm, alpha, beta, B, current_seq_len, C);
    CHECK_LAST_CUDA_ERROR();
}