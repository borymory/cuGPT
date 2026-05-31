#include "hpc_utils.cuh"
#include "mlp.cuh"

//
// CPU Reference: MLP forward pass with ReLU
//
// Input shapes:
//   X: [B * T, C]
//   W1: [C, 4 * C],  b1: [4 * C]
//   W2: [4 * C, C],  b2: [C]
void cpu_mlp_forward(float *X, float *out,
                    float *W1, float *b1,
                    float *W2, float *b2,
                    int BT, int C) {
    
    // Allocate memory for hidden state tensor of shape [BT, 4 * C]
    float *h_out = (float*)malloc(BT * 4 * C * sizeof(float));

    // h_out = ReLU(X * W1 + b1)
    // h_out: [BT, 4*C]
    for (unsigned int i = 0; i < BT; ++i) {
        for (unsigned int j = 0; j < 4*C; ++j) {

            float xw1_sum = 0.0f;
            for (unsigned int k = 0; k < C; ++k) {
                xw1_sum += X[i * C + k] * W1[k * (4*C) + j];
            }
            h_out[i * (4*C) + j] = fmaxf(0.0f, xw1_sum + b1[j]);
        }
    }

    // out = h_out * W2 + b2
    // out: [BT, C]
    for (unsigned int i = 0; i < BT; ++i) {
        for (unsigned int j = 0; j < C; ++j) {

            float hw2sum = 0.0f;
            for (unsigned int k = 0; k < 4*C; ++k) {
                hw2sum += h_out[i * (4*C) + k] * W2[k * C + j];
            }
            out[i * C + j] = hw2sum + b2[j];
        }
    }

    free(h_out);
}

// CPU first bias + ReLU
// h_out: [B * T, 4 * C]
// b1: [4 * C]
void cpu_bias_ReLU (float *h_out, float *out, float *b1,
                    int BT, int C) {
    
    for (unsigned int idx = 0; idx < BT * 4 * C; ++idx) {
        int bias_idx = idx % (4*C);
        out[idx] = fmaxf(0.0f, h_out[idx] + b1[bias_idx]);
    }
}

//
// GPU Kernels
//

// First layer bias + ReLU:
// h_out: [B * T, 4 * C]
// b1: [4 * C]
template<const int block_BT>
__global__ void fused_bias_ReLU_v1 (float *h_out, float *b1, 
                                        int BT, int C) {
    // Each block is responsible for block_BT many rows
    // We launch CEIL_DIV(BT, block_BT) many blocks

    // Offset blocks to rows
    int rowIdx = blockIdx.x * block_BT;
    h_out += rowIdx * (4 * C);

    int tx = threadIdx.x;
    for (unsigned int idx = 0; idx < 4 * C * block_BT; idx += blockDim.x) {
        if (tx + idx < 4 * C * block_BT) {
            int bias_idx = idx % (4*C);
            h_out[tx + idx] = fmaxf(0.0f, h_out[tx + idx] + b1[bias_idx]);
        }
    }
}

// Second layer bias + residual
// X: [B * T, C]
// h_out: [B * T, C]
// b2: [C]
__global__ void fused_bias_residual_v1 (float *X, float *h_out, float *b2, 
                                            int BT, int C) {
    // code here
}

//
// Kernel Launchers
//
void launch_fused_bias_ReLU_v1 (float *h_out, float *b1, 
                                    int BT, int C, cudaStream_t stream) {

    const int block_BT = 32;
    dim3 gridDim(CEIL_DIV(BT, block_BT));
    dim3 blockDim(block_BT * 32);

    fused_bias_ReLU_v1<block_BT><<<gridDim, blockDim, 0, stream>>>(h_out, b1, BT, C);

    // Check for launch errors (like passing a CPU pointer!)
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
}