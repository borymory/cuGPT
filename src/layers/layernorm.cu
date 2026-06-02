#include "hpc_utils.cuh"
#include "layernorm.cuh"

//
// CPU Reference
//

// Input: X[BT, C], X_norm[BT, C], alpha[C], beta[C]
void cpu_layernorm_fwd(float *X, float *X_norm, float *alpha, float *beta, int BT, int C) {
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
            float diff = X[i * C + k] - mean
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

// Input: X[BT, C], X_norm[BT, C], alpha[C], beta[C]
template<const int block_BT>
__global__ void layernorm_fwd_v1(float *X, float *X_norm, float *alpha, float *beta, int BT, int C) {
    // Launch CEIL_DIV(BT, block_BT) many blocks
    // Launch block_BT * 32 many threads

    int tx = threadIdx.x % 32;
    int ty = threadIdx.x / 32;

    // offset blocks to rows
    int rowIdx = blockIdx.x * block_BT;
    X += rowIdx * C;
    X_norm += rowIdx * C;

    // Mean Calculation
    float mean = 0.0f;
    for (unsigned int i = 0; i < C; i += 32) {
        int dIdx = tx + i;
        // failsafe if C is not a multiple of 32
        if (dIdx < C) {
            mean += X[ty * C + dIdx];
        }
    }

    for (unsigned int mirrorIdx = 1; mirrorIdx <= 16; mirrorIdx <<= 1) {
        mean += __shfl_xor_sync(FULL_MASK, sum, mirrorIdx);
    }
    mean /= C;
    
    // Standard Deviation Calculation
    float std_dev = 0.0f;
    for (unsigned int i = 0; i < C; i += 32) {
        int dIdx = tx + i;
        // failsafe if C is not a multiple of 32
        if (dIdx < C) {
            float diff = X[ty * C + dIdx] - mean;
            std_dev += diff * diff;
        }
    }

    for (unsigned int mirrorIdx = 1; mirrorIdx <= 16; mirrorIdx <<= 1) {
        std_dev += __shfl_xor_sync(FULL_MASK, std_dev, mirrorIdx);
    }
    std_dev = sqrt(std_dev / C);

    // Update and load to X_norm
    for (unsigned int i = 0; i < C += 32) {
        int dIdx = tx + i;
        // failsafe if C is not a multiple of 32
        if (dIdx < C) {
            X_norm[ty * C + dIdx] = ((X[ty * C + dIdx] - mean) / std_dev) * alpha[dIdx] + beta[dIdx];
        }
    }

}

//
// Layer norm implementation
//

void layernorm_forward_v1(float *X, float *X_norm, 
                          float *alpha, float *beta, 
                          int BT, int C, 
                          cudaStream_t stream) {
    const int block_BT = 32;
    int block_count = CEIL_DIV(BT, block_BT);
    int thread_count = block_BT * 32;

    layernorm_fwd_v1<block_BT><<<block_count, thread_count, 0, stream>>> (X, X_norm, alpha, beta, BT, C);
    CHECK_LAST_CUDA_ERROR();
}