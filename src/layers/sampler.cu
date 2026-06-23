#include "hpc_utils.cuh"
#include "sampler.cuh"

//
// Helpers
//

// LCG Rand num generator
__device__ unsigned int lcg_random(unsigned int* seed) {
    *seed = 1664525U * (*seed) + 1013904223U;
    return *seed;
}

__device__ float lcg_random_float(unsigned int* seed) {
    return (float)lcg_random(seed) / (float)4294967295U;
}

//
// CPU Funcs
//

// logits[B, vocab_size], top_k_probs[B, MAX_K], top_k_indices[B, MAX_K]
// returns list of indices and respective probs for sampling the next token.
template<const int MAX_K>
void online_softmax_topk_temp(
    const float *logits, 
    float *top_k_probs, 
    int *top_k_indices, 
    const int B, 
    const int vocab_size, 
    const float temp)
{
    if (MAX_K > vocab_size) {
        printf("Invalid arguments: MAX_K > vocab_size!\n");
        return;
    }
    const float T_inv = (temp > 0.0f) ? (1.0f / temp) : 1.0f;
    for (unsigned int batch_offset = 0; batch_offset < B; ++batch_offset) {
        float m = -INFINITY;
        float norm = 1.0f;

        float u[MAX_K];
        int p[MAX_K];
        for (unsigned int k = 0; k < MAX_K; ++k) {
            u[k] = -INFINITY;
            p[k] = -1;
        }

        const float *batch_logits = logits + (batch_offset * vocab_size);

        // Finding Top-K elements
        for (int elem_idx = 0; elem_idx < vocab_size; ++elem_idx) {
            float elem = batch_logits[elem_idx];

            // Pre-insertion, if elem is big enough
            if (elem > u[MAX_K - 1]) {

                u[MAX_K-1] = elem;
                p[MAX_K-1] = elem_idx;

                for (int k = MAX_K-2; k >= 0; --k) {
                    if (u[k+1] > u[k]) { // commence swap
                        float temp_u = u[k];
                        int temp_p = p[k];

                        u[k] = u[k+1];
                        p[k] = p[k+1];

                        u[k+1] = temp_u;
                        p[k+1] = temp_p;
                    } else {
                        break;
                    }
                }
            }
        }

        m = u[0];   // By sorting, we are sure that the 0th entry is the maximum value,
                    // with contribution to norm of 1.0f
        for (int k = 1; k < MAX_K; ++k) {
            float elem = u[k];
            norm += expf((elem - m) * T_inv);
        }

        float *dest_probs = top_k_probs + (batch_offset * MAX_K);
        int *dest_indices = top_k_indices + (batch_offset * MAX_K);

        for (int k = 0; k < MAX_K; ++k) {
            dest_probs[k] = expf((u[k] - m) * T_inv) / norm;
            dest_indices[k] = p[k];
        }
    }
}

// u [B, MAX_K], p [B, MAX_K], next_tokens[B] (gives selected token ID for each batch)
void sample_top_k_from_probs(
    const float *u, 
    const int *p, 
    int *next_tokens, 
    const int B, 
    const int MAX_K)
{
    for (unsigned int b = 0; b < B; ++b) {

        int token_id = p[b * MAX_K + (MAX_K-1)]; // If probs' sum != 1 and coin_flip is really close to 1. Default back to lowest prob.
        float coin_flip = (float)std::rand() / RAND_MAX;
        float cdf_range = 0.0f;

        for (unsigned int k = 0; k < MAX_K; ++k) {
            float prob_offset = u[b * MAX_K + k];
            cdf_range += prob_offset;
            if (coin_flip <= cdf_range) {
                // Range found, break loop, write batch's selected token_id
                token_id = p[b * MAX_K + k];
                break;
            }

        }
        next_tokens[b] = token_id;
    }
}

//
// GPU Kernels
//

// Inputs d_logits[B, current_seq_len, vocab_size]
// Outputs d_sampled_token[B]
// BLOCK_SIZE = 512
template<int BLOCK_SIZE>
__global__ void fused_sample_kernel(
    const float* __restrict__ d_logits,     // [B * current_seq_len, V]
    int* __restrict__ d_sampled_token,      // [B]
    unsigned int* d_seeds,                  // [B]
    const int current_seq_len,
    const int vocab_size,
    const float temperature
) {
    int b = blockIdx.x;
    int tid = threadIdx.x;

    int block_offset = b * (current_seq_len * vocab_size) + (current_seq_len - 1) * vocab_size;
    const float* logits_local = d_logits + block_offset;
    const float inv_temp = 1.0f / temperature;

    __shared__ float s_max_val;
    __shared__ float s_total_sum;
    __shared__ float s_threshold;
    __shared__ float s_sums[BLOCK_SIZE]; // Local sums of the 512 threads

    // Finding global max
    float thread_max = -INFINITY;
    for (int idx = tid; idx < vocab_size; idx += BLOCK_SIZE) {
        thread_max = fmaxf(thread_max, logits_local[idx]);
    }
    s_sums[tid] = thread_max;
    __syncthreads();

    // Block-level reduction for global max
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sums[tid] = fmaxf(s_sums[tid], s_sums[tid + offset]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        s_max_val = s_sums[0];
    }
    __syncthreads();

    // Calculating local sum
    int stride = (vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int start_idx = tid * stride;
    int end_idx = min(start_idx + stride, vocab_size);
    float thread_sum = 0.0f;
    for (int i = start_idx; i < end_idx; ++i) {
        thread_sum += expf((logits_local[i] - s_max_val) * inv_temp);
    }
    s_sums[tid] = thread_sum;
    __syncthreads();

    float val = s_sums[tid];
    for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
        float temp = 0.0f;
        if (tid >= offset) {
            temp = s_sums[tid - offset];
        }
        __syncthreads();
        s_sums[tid] += temp;
        __syncthreads();
    }
    float thread_start_cumulative = (tid == 0) ? 0.0f : s_sums[tid - 1];
    __syncthreads();
    float thread_end_cumulative = thread_start_cumulative + val;

    if (tid == BLOCK_SIZE - 1) {
        s_total_sum = thread_end_cumulative;
    }
    __syncthreads();

    if (tid == 0) {
        unsigned int seed = d_seeds[b];
        float r = lcg_random_float(&seed);
        d_seeds[b] = seed;
        s_threshold = r * s_total_sum;
    }
    __syncthreads();

    if (s_threshold >= thread_start_cumulative && s_threshold < thread_end_cumulative) {
        float running_accum = thread_start_cumulative;

        for (int i = start_idx; i < end_idx; ++i) {
            running_accum += expf((logits_local[i] - s_max_val) * inv_temp);
            if (running_accum >= s_threshold) {
                d_sampled_token[b] = i;     // The winning token ID
                break;
            }
        }
    }
}

//
// Sampler Implementations
//

void launch_sampler(
    const float* d_logits,
    int* d_sampled_token,
    unsigned int* d_seeds,
    const int B,
    const int current_seq_len,
    const int vocab_size,
    const float temperature,
    cudaStream_t stream
) {
    dim3 gridDim(B);
    dim3 blockDim(512);

    fused_sample_kernel<512><<<gridDim, blockDim, 0, stream>>>(
        d_logits, 
        d_sampled_token, 
        d_seeds, 
        current_seq_len, 
        vocab_size, 
        temperature
    );
    CHECK_LAST_CUDA_ERROR();
}


// PREFILL and DECODE
// Input: logits[B, vocab_size]
// Intermediate: u[B, MAX_K], p[B, MAX_K]
// Output: next_tokens[B]
void sample_top_k_from_logits(
    const float *logits, 
    float *u, int *p, 
    int *next_tokens, 
    const int B, 
    const int vocab_size, 
    const float temp) 
{
    // Time based seed on std::rand for new sampling at every run
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    const int MAX_K = 5;
    online_softmax_topk_temp<MAX_K>(logits, u, p, B, vocab_size, temp);
    sample_top_k_from_probs(u, p, next_tokens, B, MAX_K);
}