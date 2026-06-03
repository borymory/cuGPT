#include "hpc_utils.cuh"
#include "embedding.cuh"

//
// CPU Reference
//

// inputs[B, T],  out[B, T, C], wte[vocab_size, C], wpe[max_length, C]
void cpu_embedding(int *inputs, float *out, float *wte, float *wpe, int B, int T, int C) {
    for (unsigned int i = 0; i < B; ++i) {
        int batch_offset = i * (T * C);
        for (unsigned int j = 0; j < T; ++j){
            int token_offset = j * C;
            int token_id = inputs[i * T + j];    // ranges 0 - vocab_size-1

            for (unsigned int k = 0; k < C; ++k) {
                out[batch_offset + token_offset + k] = wte[token_id * C + k] + wpe[token_offset + k];
            }
        }
    }
}

//
// GPU Kernels
//

// inputs[BT],  out[BT * C], wte[vocab_size, C], wpe[max_length, C]
__global__ void embedding_v1(const int *inputs, float *out, const float *wte, const float *wpe, const int BT, const int C, const int max_length) {
    // View inputs and out as 1D
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = BT * C;

    // grid-stride loop
    for (int idx = tid; idx < total_elements; idx += blockDim.x * gridDim.x) {
        
        int c_idx = idx % C;        // which channel in the token
        int token_idx = idx / C;    // which token in the sequence
        int token_id = inputs[token_idx];
        int pos_idx = token_idx % max_length;

        out[idx] = wte[token_id * C + c_idx] + wpe[pos_idx * C + c_idx];
    }
}

//
// Kernel Launchers
//



//
// Embedding Implementations
//
void launch_embedding_v1(int *inputs, float *out, float *wte, float *wpe, int B, int T, int C, int max_length, cudaStream_t stream) {
    int BT = B * T;
    int total_elements = BT * C;

    int thread_count = 256;
    int block_count = CEIL_DIV(total_elements, thread_count);

    embedding_v1<<<block_count, thread_count, 0, stream>>>(inputs, out, wte, wpe, BT, C, max_length);
    CHECK_LAST_CUDA_ERROR();
}