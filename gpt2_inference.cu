#include "hpc_utils.cuh"
#include <stdio.h>
#include <stdlib.h>

// Reference
// H: number of heads
// C = d_model, the whole embedding dimension
// d_head: =C/H
// B: batch
// N or T: sequence_length
// max_T, max_seq_len, the maximum sequence length that can be processed
// L: number of layers (number of forward passes)


//
// Helper Funcs
//

__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

// LCG Rand num generator
__device__ unsigned int lcg_random(unsigned int* seed) {
    *seed = 1664525U * (*seed) + 1013904223U;
    return *seed;
}

__device__ float lcg_random_float(unsigned int* seed) {
    return (float)lcg_random(seed) / (float)4294967295U;
}

// Assumes Row-Major layout: Pads SMEM with zeros if N is not divisible by Br or Bc
__device__ __forceinline__ void fload_to_smem(
    float* __restrict__ shared_dst, 
    const float* __restrict__ global_src,
    int transpose, 
    const int row_dim,  // Tile rows (Br or Bc)
    const int col_dim,  // Tile col (d)
    const int padding,
    const int global_row_offset,    // Row offset from the beggining of [N, d]
    const int max_rows,
    const int ldgmem
) 
{
    int num_elements = row_dim * col_dim;

    if (transpose) {
        int ldsmem = row_dim + padding; // How many elements to get to the other row

        for (int idx = threadIdx.x; idx < num_elements; idx += blockDim.x) {
            int s_col = idx / col_dim;
            int s_row = idx % col_dim;
            int s_idx = s_row * ldsmem + s_col;
            int g_idx = s_col * ldgmem + s_row;

            if ((global_row_offset + s_col) < max_rows) {
                shared_dst[s_idx] = global_src[g_idx];
            } else {
                shared_dst[s_idx] = 0.0f;
            }
            // smem shape: [col_dim, row_dim + padding]
        }
    } else {
        int ldsmem = col_dim + padding;

        for (int idx = threadIdx.x; idx < num_elements; idx += blockDim.x) {
            int s_col = idx % col_dim;
            int s_row = idx / col_dim;
            int s_idx = s_row * ldsmem + s_col;
            int g_idx = s_row * ldgmem + s_col;

            if ((global_row_offset + s_row) < max_rows) {
                shared_dst[s_idx] = global_src[g_idx];
            } else {
                shared_dst[s_idx] = 0.0f;
            }
            // smem shape: [row_dim, col_dim + padding]
        }
    }
}

template<
const int Br, 
const int Bc, 
const int d, 
const int ROWS_PER_WARP, 
const int COLS_PER_WARP, 
const int ELEMENTS_PER_ROW, 
const int warp_count
>
__device__ __forceinline__ void tile_attention(
const float* __restrict__ s_Q, 
const float* __restrict__ s_K, 
const float* __restrict__ s_V,
float* __restrict__ s_S, 
float* __restrict__ m_i,
float* __restrict__ l_i,
float* __restrict__ m_prev,
float* __restrict__ l_prev,
float* __restrict__ thread_res_O,
const float scale, 
const int i,
const int j,
const int N,
int warpLane,
int warpId
) {   
    // Warp-Level tiling of Q, S, and O.
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        int warp_row = warpId + (r * warp_count);
        int global_row = i + warp_row;
        if (global_row < N) {

            // QK Matmul: Q [Br, d], K^T [d, Bc]
            #pragma unroll
            for (int c = 0; c < COLS_PER_WARP; ++c) {
                int idx = warpLane + (c * 32);
                int global_col = j + idx;
                // Casual Masking s_S
                if (global_col < N && global_row >= global_col) {
                    float qk_sum = 0.0f;

                    #pragma unroll
                    for (int k = 0; k < d; ++k) {
                        float q_val = s_Q[warp_row * (d+1) + k];
                        float k_val = s_K[k * (Bc+1) + idx];
                        qk_sum += q_val * k_val;
                    }
                    s_S[warp_row * (Bc+1) + idx] = qk_sum * scale;
                }
            }
            __syncwarp();

            // Reduce s_S into registers, doing statistics
            #pragma unroll
            for (int c = 0; c < COLS_PER_WARP; ++c) {
                int idx = warpLane + (c * 32);
                int global_col = j + idx;
                if (global_col < N && global_row >= global_col) {
                    float val = s_S[warp_row * (Bc+1) + idx];
                    
                    float m_old = m_i[r];     // store old local max
                    m_i[r] = fmaxf(m_i[r], val);    // obtain new local max
                    l_i[r] *= expf(m_old - m_i[r]); // scale old norm
                    l_i[r] += expf(val - m_i[r]);   // add current contribution   
                }
            }

            // Warp shuffle online softmax
            #pragma unroll
            for (unsigned int mirrorIdx = 1; mirrorIdx <= 16; mirrorIdx <<= 1) {
                float m_j = __shfl_xor_sync(FULL_MASK, m_i[r], mirrorIdx);     // obtain m_j from another thread
                float l_j = __shfl_xor_sync(FULL_MASK, l_i[r], mirrorIdx);     // obtain l_j from another thread

                float max = fmaxf(m_i[r], m_j);    // max = max(m_i, m_j)

                // To prevent out-of-bound thread to propogate NaN
                if (max != -INFINITY) {
                    l_i[r] *= expf(m_i[r] - max);    // rescale old sum
                    l_i[r] += l_j * expf(m_j - max);       // add contribution from the new sum
                    m_i[r] = max;                 // store new max
                } else {
                    l_i[r] = 0.0f;
                    m_i[r] = -INFINITY;
                }
            }

            // Calculate unnorm P
            #pragma unroll
            for (int c = 0; c < COLS_PER_WARP; ++c) {
                int idx = warpLane + (c * 32);
                int global_col = j + idx;
                if (global_col < N) {
                    if (global_row >= global_col) {
                        float val = s_S[warp_row * (Bc+1) + idx];
                        s_S[warp_row * (Bc+1) + idx] = expf(val - m_i[r]);
                    } else {
                        s_S[warp_row * (Bc+1) + idx] = 0.0f;
                    }
                }
            }
            __syncwarp();

            // Calculate global stats - store the new stats
            float m_new = fmaxf(m_prev[r], m_i[r]);
            float prev_scale = expf(m_prev[r] - m_new);
            float current_scale = expf(m_i[r] - m_new);
            m_prev[r] = m_new;
            l_prev[r] = prev_scale * l_prev[r] + current_scale * l_i[r];

            // PV Matmul: P [Br, Bc], V [Bc, d]
            #pragma unroll
            for (int c = 0; c < ELEMENTS_PER_ROW; ++c) {
                int idx = warpLane + (c * 32);
                float pv_sum = 0.0f;

                #pragma unroll
                for (int k = 0; k < Bc; ++k) {
                    // Inner dimension boundary check:
                    int global_col = j + k;
                    if (global_col < N) {
                        float p_val = s_S[warp_row * (Bc+1) + k];
                        float v_val = s_V[k * (d+1) + idx];
                        pv_sum += p_val * v_val;
                    }
                }
                thread_res_O[r * ELEMENTS_PER_ROW + c] *= prev_scale;   // scale prev O chunk
                pv_sum *= current_scale;    // scale pv_sum w.r.t. global stats
                thread_res_O[r * ELEMENTS_PER_ROW + c] += pv_sum;   // add current PV chunk
            }
        }
    }
}

//
// KERNELS
//
__global__ void embedding_v1(const int *inputs, float *out, const float *wte, const float *wpe, const int B, const int T, const int C, const int max_length) {
    // View inputs and out as 1D
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * T * C;

    // grid-stride loop
    for (int idx = tid; idx < total_elements; idx += blockDim.x * gridDim.x) {
        
        int c_idx = idx % C;        // which channel in the token
        int token_idx = idx / C;    // which token in the sequence
        int token_id = inputs[token_idx];
        int pos_idx = token_idx % T;

        out[idx] = wte[token_id * C + c_idx] + wpe[pos_idx * C + c_idx];
    }
}

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

// Op: d_out [B * current_seq_len, C] + d_bias[C] (broadcasted to every row)
// Out: d_out [B * current_seq_len, C]
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

template<
const int Br, 
const int Bc, 
const int d, 
const int ROWS_PER_WARP, 
const int COLS_PER_WARP, 
const int ELEMENTS_PER_ROW, 
const int warp_count
>
__global__ void flash_mha_fwd_v1(
    const float* __restrict__ Q, // Shape: [B, N, H, d]
    const float* __restrict__ K, // Shape: [B, N, H, d]
    const float* __restrict__ V, // Shape: [B, N, H, d]
    float* __restrict__ O,       // Shape: [B, N, H, d]
    const int H,
    const int N,                // Same as sequence length
    const int max_T
) {
    int head_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int C = H * d;

    int block_offset = batch_idx * (N * C) + head_idx * d;
    int block_offset_KV = batch_idx * (max_T * C) + head_idx * d;

    // Block Pointer Offsets
    const float *Q_local = Q + block_offset;
    const float *K_local = K + block_offset_KV;
    const float *V_local = V + block_offset_KV;
    float *O_local = O + block_offset;

    // Shared Memory
    extern __shared__ float s_mem[];
    float* s_Q = s_mem;         // Size Br * (d + 1)
    float* s_K = s_Q + Br * (d + 1);  // Size d * (Bc + 1)
    float* s_V = s_K + d * (Bc + 1);  // Size Bc * (d + 1)
    float* s_S = s_V + Bc * (d + 1);  // Size Br * (Bc + 1)

    // 1/sqrt(d) from the attention paper
    const float scale = 1.0f / sqrtf(d);

    // Thread Specific data
    float m_prev[ROWS_PER_WARP];
    float l_prev[ROWS_PER_WARP];
    float m_i[ROWS_PER_WARP];
    float l_i[ROWS_PER_WARP];
    float thread_res_O[ROWS_PER_WARP * ELEMENTS_PER_ROW];
    int warpLane = threadIdx.x % 32;    // 0 to 31
    int warpId = threadIdx.x / 32;  // 0 to 7
    
    // Outer Loop over Q and O
    for (unsigned int i = 0; i < N; i += Br) {

        // Initialize global stats and registers for O_i block
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            m_prev[r] = -INFINITY;
            l_prev[r] = 0.0f;

            #pragma unroll
            for (int c = 0; c < ELEMENTS_PER_ROW; ++c){
                thread_res_O[r * ELEMENTS_PER_ROW + c] = 0.0f;
            }
        }

        // Populate s_Q
        fload_to_smem(s_Q, Q_local + i * C, 0, Br, d, 1, i, N, C);
        __syncthreads();

        // Inner Loop over K and V: 
        // Populates m_prev, l_prev with final softmax and
        // thread_res_O register cache with the final values
        for (unsigned int j = 0; j < N; j += Bc) {

            // Populate rest smem: transpose K
            fload_to_smem(s_K, K_local + j * C, 1, Bc, d, 1, j, N, C);
            fload_to_smem(s_V, V_local + j * C, 0, Bc, d, 1, j, N, C);
            __syncthreads();

            // Initialize block-local stats per inner-loop
            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                m_i[r] = -INFINITY;
                l_i[r] = 0.0f;
            }

            // QK matmul, softmax, PV matmul
            // Populates: m_i, l_i, m_prev, l_prev, thread_res_O
            tile_attention<Br, Bc, d, ROWS_PER_WARP, COLS_PER_WARP, ELEMENTS_PER_ROW, warp_count>(
                s_Q, s_K, s_V, s_S,
                m_i, l_i,
                m_prev, l_prev,
                thread_res_O,
                scale,
                i, j, N,
                warpLane, warpId
            );
            __syncthreads();
        }

        // Load final output from registers to GMEM
        float* O_block = O_local + i * d;
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            int warp_row = warpId + (r * warp_count);
            int global_row = i + warp_row;
            if (global_row < N) {
                float inv_l = 1.0f / l_prev[r];
                
                #pragma unroll
                for (int c = 0; c < ELEMENTS_PER_ROW; ++c) {
                    int idx = warpLane + (c * 32);
                    if (idx < d) {
                        O_block[warp_row * C + idx] = inv_l * thread_res_O[r * ELEMENTS_PER_ROW + c];
                    }
                }
            }
        }
    }
}

// Residual
__global__ void residual_add(
    float* __restrict__ d_out,
    const float* __restrict__ d_res,
    const int B,
    const int current_seq_len,
    const int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_elements = B * current_seq_len * C;

    if (idx < num_elements) {
        d_out[idx] += d_res[idx];   // Elementwise addition
    }
}

// ReLU(h_out [BT, 4C] + b1 [4C]) -> h_out [BT, 4C]
// MLP bias+gelu and bias+residual kernels
__global__ void fused_bias_gelu_v1 (
    float *h_out, // [BT, 4C]
    float *b1,    // [4C]
    int BT, 
    int C
) {
    // Launch CEIL_DIV(BT * 4*C, 256) many blocks
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_elements = BT * 4 * C;

    if (idx < num_elements) {
        int bias_idx = idx % (4*C);
        h_out[idx] = gelu(h_out[idx] + b1[bias_idx]);
    }
}

__global__ void fused_bias_residual_v1 (float *X, float *out, float *b2, 
                                            int BT, int C) {
    // Launch CEIL_DIV(BT*C, 256) many blocks
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_elements = BT * C;
    if (idx < num_elements) {
        int bias_idx = idx % C;
        out[idx] += X[idx] + b2[bias_idx];
    }
}

// Inputs d_logits[B, current_seq_len, vocab_size]
// Appends to d_prompt
// BLOCK_SIZE = 512
template<int BLOCK_SIZE>
__global__ void fused_sample_kernel(
    float* __restrict__ d_logits,     // [B * current_seq_len, V]
    int* __restrict__ d_prompt,             // [B, current_seq_len]
    unsigned int* d_seeds,                  // [B]
    const int current_seq_len,
    const int vocab_size,
    const int max_seq_len,
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
        float val = logits_local[i];
        if (val != -INFINITY) {
            thread_sum += expf((val - s_max_val) * inv_temp);
        }
    }
    s_sums[tid] = thread_sum;
    __syncthreads();

    // Calculate CDF
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
                d_prompt[b * max_seq_len + current_seq_len] = i;     // The winning token ID
                break;
            }
        }
    }
}

template<int TOP_K>
__global__ void top_k_filter(
    float* __restrict__ d_logits,     // [B * current_seq_len, V]
    const int current_seq_len,
    const int vocab_size,
    const int max_seq_len
) {
    // We launch B many blocks
    // blockDim.x = 32
    int b = blockIdx.x;
    float* logits_local = d_logits + b * (current_seq_len * vocab_size) + (current_seq_len - 1) * vocab_size;

    float top_k_i[TOP_K];
    float top_k_j[TOP_K];
    float merged[TOP_K];
    for (int i = 0; i < TOP_K; ++i) {
        top_k_i[i] = -INFINITY;
    }

    // Thread level top-k sorting
    for (int idx = threadIdx.x; idx < vocab_size; idx += 32) {
        float val = logits_local[idx];
        if (val > top_k_i[TOP_K-1]) {
            top_k_i[TOP_K-1] = val;
            
            for (int i = TOP_K-2; i >= 0; --i) {
                // Commence local top-k swap
                if (val > top_k_i[i]) {
                    float temp_top_k = top_k_i[i];
                    top_k_i[i] = val;
                    top_k_i[i+1] = temp_top_k;
                } else {
                    break;
                }
            }
        }
    }

    // Each thread chunk does TOP_K reduction
    #pragma unroll
    for (unsigned int mirrorIdx = 1; mirrorIdx <= 16; mirrorIdx <<= 1) {

        // Copy other threads top_k into our temp top_k register
        #pragma unroll
        for (int k = 0; k < TOP_K; ++k) {
            top_k_j[k] = __shfl_xor_sync(FULL_MASK, top_k_i[k], mirrorIdx);
        }

        // Merge both our TOP_Ks
        int i = 0;
        int j = 0;
        #pragma unroll
        for (int k = 0; k < TOP_K; ++k) {
            if(top_k_i[i] >= top_k_j[j]) {
                merged[k] = top_k_i[i];
                i++;
            } else {
                merged[k] = top_k_j[j];
                j++;
            }
        }

        // Copy result into top_k_i
        #pragma unroll
        for (int k = 0; k < TOP_K; ++k) {
            float val = merged[k];
            top_k_i[k] = val;
        }
    }

    // Filter top-k elements
    float cutoff = top_k_i[TOP_K-1];
    for (int idx = threadIdx.x; idx < vocab_size; idx += 32) {
        float val = logits_local[idx];
        if (val < cutoff) {
            logits_local[idx] = -INFINITY;
        }
    }
}

//
// Kernel Launchers
//
void launch_embedding_v1(int *inputs, float *out, float *wte, float *wpe, int B, int T, int C, int max_length, cudaStream_t stream) {
    int total_elements = B * T * C;

    int thread_count = 256;
    int block_count = CEIL_DIV(total_elements, thread_count);

    embedding_v1<<<block_count, thread_count, 0, stream>>>(inputs, out, wte, wpe, B, T, C, max_length);
    CHECK_LAST_CUDA_ERROR();
}

void layernorm_forward_v1(
    const float* __restrict__ X,        // [B * current_seq_len, C]
    float* X_norm,                      // [B * current_seq_len, C]
    const float* __restrict__ alpha,    // [C]
    const float* __restrict__ beta,     // [C]
    const int B,
    const int current_seq_len, 
    const int C,
    cudaStream_t stream
) {
    int BT = B * current_seq_len;
    const int block_BT = 32;
    int block_count = CEIL_DIV(BT, block_BT);
    int thread_count = block_BT * 32;

    layernorm_fwd_v1<block_BT><<<block_count, thread_count, 0, stream>>> (X, X_norm, alpha, beta, B, current_seq_len, C);
    CHECK_LAST_CUDA_ERROR();
}

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

void qkv_proj_append_to_KV_cache(
    cublasHandle_t cublas_handle,
    float* __restrict__ X_norm, // [B * current_seq_len, C]
    float* __restrict__ w_q, // [C, C]
    float* __restrict__ w_k, // [C, C]
    float* __restrict__ w_v, // [C, C]
    const float* __restrict__ b_q, // [C]
    const float* __restrict__ b_k, // [C]
    const float* __restrict__ b_v, // [C]
    float* __restrict__ key_cache,  // Written: [B, current_seq_len, C]
    float* __restrict__ value_cache, // Written: [B, current_seq_len, C]
    float* __restrict__ q_scratch, // Written: [B * current_seq_len, C]
    const int B,
    const int current_seq_len,
    const int C,
    const int max_seq_len,
    cudaStream_t stream
)
{
    cublasSetStream(cublas_handle, stream);

    // Q Proj
    cuGPT::gemm(cublas_handle, X_norm, w_q, q_scratch, B * current_seq_len, C, C);
    // Q Bias
    launch_proj_bias_add(q_scratch, b_q, B, current_seq_len, C, stream);

    // KV Cache appending is done by batch offsets
    for (int b = 0; b < B; ++b) {
        float* key_cache_batch = key_cache + (b * max_seq_len * C);
        float* value_cache_batch = value_cache + (b * max_seq_len * C);
        float* X_norm_batch = X_norm + (b * current_seq_len * C);
        
        // KV Proj
       cuGPT::gemm(cublas_handle, X_norm_batch, w_k, key_cache_batch, current_seq_len, C, C);
       cuGPT::gemm(cublas_handle, X_norm_batch, w_v, value_cache_batch, current_seq_len, C, C);

       // KV Bias
       launch_proj_bias_add(key_cache_batch, b_k, 1, current_seq_len, C, stream);
       launch_proj_bias_add(value_cache_batch, b_v, 1, current_seq_len, C, stream);
    }

}

void launch_flash_mha_fwd_v1(
    const float* __restrict__ Q, // Shape: [B, N, H, d]
    const float* __restrict__ K, // Shape: [B, N, H, d]
    const float* __restrict__ V, // Shape: [B, N, H, d]
    float* __restrict__ O,       // Shape: [B, N, H, d]
    const int B,
    const int H,
    const int N,                 // Same as sequence length
    const int d,
    const int max_T,
    cudaStream_t stream
)
{
    dim3 gridDim(H, B);
    dim3 blockDim(256);
    const int warp_count = 256 / 32;
    const int Br = 32;
    const int Bc = 32;
    const int ROWS_PER_WARP = Br / warp_count;  // 32 / 8 = 4 rows
    const int COLS_PER_WARP = Bc / 32;  // 32 / 32 = 1 element
    const int ELEMENTS_PER_ROW = 64 / 32; // d / 32 = 2 registers

    size_t shared_mem_bytes = (size_t)Br * (d + 1); // Padding for s_Q
    shared_mem_bytes += (size_t)d * (Bc+1);         // Padding for s_K: TRANPOSED
    shared_mem_bytes += (size_t)Bc * (d + 1);       // Padding for s_V
    shared_mem_bytes += (size_t)Br * (Bc + 1);      // Padding for s_S
    shared_mem_bytes *= sizeof(float);

    switch (d) {
        case 64: // GPT2 Case
            flash_mha_fwd_v1<Br, Bc, 64, ROWS_PER_WARP, COLS_PER_WARP, ELEMENTS_PER_ROW, warp_count><<<gridDim, blockDim, shared_mem_bytes, stream>>>(Q, K, V, O, H, N, max_T);
            CHECK_LAST_CUDA_ERROR();
            break;
        default:
            fprintf(stderr, "FlashAttn ERROR: No given d case is found: d=%d\n", d);
            exit(EXIT_FAILURE);
    }
}

void o_proj(
    cublasHandle_t cublas_handle,
    float* __restrict__ attn_out,   // [B * seq_len, C]
    float* __restrict__ w_o,        // [C, C]
    const float* __restrict__ b_o,  // [C]
    float* __restrict__ o_proj_out, // Written: [B, seq_len, C]
    const int B,
    const int current_seq_len,
    const int C,
    cudaStream_t stream 
) {
    cublasSetStream(cublas_handle, stream);

    // O Proj
    cuGPT::gemm(cublas_handle, attn_out, w_o, o_proj_out, B * current_seq_len, C, C);

    // O Bias
    launch_proj_bias_add(o_proj_out, b_o, B, current_seq_len, C, stream);
}

void launch_residual_add (
    float* __restrict__ d_out,
    const float* __restrict__ d_res,
    const int B,
    const int current_seq_len,
    const int C,
    cudaStream_t stream
) {
    int total_elements = B * current_seq_len * C;
    int block_size = 256;
    int grid_size = CEIL_DIV(total_elements, block_size);

    residual_add<<<grid_size, block_size, 0, stream>>>(
        d_out, d_res, B, current_seq_len, C);
    CHECK_LAST_CUDA_ERROR();
}

// RESIDUAL FUSED MLP
void mlp_forward_v1(
    cublasHandle_t cublas_handle, 
    float *X,       // Our residual
    float* X_norm,  // Layernorm input to MLP
    float *h_out,   // Hidden output buffer [BT, 4*C]
    float *mlp_out,     // Final output [BT, C]
    float *w1, float *b1, // Weight and biases of layer 1
    float *w2, float *b2, // Weight and biases of layer 2
    int B, int T, int C, cudaStream_t stream
) {
    cublasSetStream(cublas_handle, stream);
    int thread_count = 256;
    int BT = B * T;
    // X_norm [BT, C] * W1 [C, 4C] -> h_out [BT, 4C]
    cuGPT::gemm(cublas_handle, X_norm, w1, h_out, BT, 4 * C, C);

    // ReLU(h_out [BT, 4C] + b1 [4C]) -> h_out [BT, 4C]
    int block_count_1 = CEIL_DIV(BT * 4*C, 256);

    fused_bias_gelu_v1<<<block_count_1, thread_count, 0, stream>>>(
        h_out, 
        b1, 
        BT, 
        C
    );

    // h_out [BT, 4C] * W2 [4C, C] -> mlp_out [BT, C]
    cuGPT::gemm(cublas_handle, h_out, w2, mlp_out, BT, C, 4*C);

    // mlp_out [BT, C] + b2 [C] -> mlp_out [BT, C]
    int block_count_2 = CEIL_DIV(BT * C, 256);

    // mlp_out + residual
    fused_bias_residual_v1<<<block_count_2, thread_count, 0, stream>>>(
        X, 
        mlp_out, 
        b2, 
        BT, 
        C
    );
}

// X_final[BT, C], wte[vocab_size, C], logits[BT, vocab_size]
void lm_head_fwd(
    cublasHandle_t cublas_handle, 
    float *X_final, 
    float *wte, 
    float *logits, 
    int B,
    int current_seq_len, 
    int C, 
    int vocab_size, 
    cudaStream_t stream
) {    
    cublasSetStream(cublas_handle, stream);

    // X_final * wte^T
    // X_final [BT, C]
    // wte  [vocab_size, C]
    cuGPT::gemm_transposed(cublas_handle, X_final, wte, logits, B * current_seq_len, vocab_size, C);
}

void launch_sampler_top_k(
    float* d_logits,        // [B * current_seq_len, C]
    int* d_prompt,          // [B, current_seq_len]
    unsigned int* d_seeds,
    const int B,
    const int current_seq_len,
    const int vocab_size,
    const int max_seq_len,
    const float temperature,
    cudaStream_t stream
) {
    const int TOP_K = 50;
    int grid_dim = B;
    int block_dim = 32;
    top_k_filter<TOP_K><<<grid_dim, block_dim, 0, stream>>>(
        d_logits,
        current_seq_len,
        vocab_size,
        max_seq_len
    );
    CHECK_LAST_CUDA_ERROR();

    block_dim = 512;
    fused_sample_kernel<512><<<grid_dim, block_dim, 0, stream>>>(
        d_logits, 
        d_prompt, 
        d_seeds, 
        current_seq_len, 
        vocab_size, 
        max_seq_len,
        temperature
    );
    CHECK_LAST_CUDA_ERROR();
}

//
// MODEL DEFINITION
//

// for comparing weight config with model config
typedef struct {
    int32_t magic;
    int32_t max_seq_len;
    int32_t vocab_size;
    int32_t layers;
    int32_t heads;
    int32_t channels;
} file_header;

typedef enum {
    EMBED_OUT_IDX = 0,
    X_NORM_IDX,

    Q_SCRATCH_IDX,
    ATTN_OUT_IDX,
    O_PROJ_OUT_IDX,

    MLP_H_IDX,
    LOGITS_IDX,

    NUM_ACTIVATIONS
} ActivationIndex;

typedef enum {
    WTE_IDX = 0,
    WPE_IDX,
    LN_1_ALPHA_IDX,
    LN_1_BETA_IDX,

    PROJ_W_Q_IDX,
    PROJ_W_K_IDX,
    PROJ_W_V_IDX,
    PROJ_B_Q_IDX,
    PROJ_B_K_IDX,
    PROJ_B_V_IDX,
    PROJ_W_O_IDX,
    PROJ_B_O_IDX,

    LN_2_ALPHA_IDX,
    LN_2_BETA_IDX,

    FFN_W1_IDX,
    FFN_B1_IDX,
    FFN_W2_IDX,
    FFN_B2_IDX,

    LN_FINAL_ALPHA_IDX,
    LN_FINAL_BETA_IDX,

    NUM_TENSORS
} TensorIndex;

typedef struct {
    int max_seq_len;// max sequence length (e.g., 1024)
    int max_batch;  // max batch count
    int vocab_size; // vocab_size (e.g., 50257)
    int layers;     // num of layers (e.g., 12)
    int heads;      // num of heads (e.g., 12)
    int channels;   // channel count (e.g., 768)
} model_config;

// holds model's weight ptrs
typedef struct {
    float* wte; // [vocab_size, C]
    float* wpe; // [max_T, C]

    // ---- For layers 0 to 11 -----
    float* ln_1_alpha;  // [L, C]
    float* ln_1_beta;   // [L, C]

    float* proj_w_q;    // [L, C, C]
    float* proj_w_k;    // [L, C, C]
    float* proj_w_v;    // [L, C, C]
    float* proj_b_q;    // [L, C]
    float* proj_b_k;    // [L, C]
    float* proj_b_v;    // [L, C]

    float* proj_w_o;    // [L, C, C]
    float* proj_b_o;    // [L, C]

    float* ln_2_alpha;  // [L, C]
    float* ln_2_beta;   // [L, C]

    float* ffn_w1;  // [L, C, 4C]
    float* ffn_b1;  // [L, 4C]
    float* ffn_w2;  // [L, 4C, C]
    float* ffn_b2;  // [L, C]
    // ------------------------

    float* ln_final_alpha;  // [C]
    float* ln_final_beta;   // [C]
} model_parameters;

typedef struct {
    float* embedding_out; // [max_B, max_T, C]
    float* X_norm;  // [max_B, max_T, C]

    float* scratch_query;   // [max_B, max_T, C]
    float* attn_out;    // [max_B, max_T, C]
    float* o_proj_out;  // [max_B, max_T, C]

    float* mlp_hidden;  // [max_B, max_T, 4 * C]
    float* logits;  // [max_B, max_T, V]
} model_activations;

typedef struct {
    float* key_cache;   // [L, max_B, max_T, C]
    float* value_cache; // [L, max_B, max_T, C]
} KV_cache;

typedef struct{
    int current_seq_len = 0;
    int current_batch = 0;
} model_context;

typedef struct {
    model_config config;
    model_context context;      // model context tracking
    model_activations d_activations;    // sliced device activations
    model_parameters d_weights; // sliced device weights
    KV_cache d_kv_cache;        // sliced device KV cache

    float* h_weights_base;      // base ptr to host side weights
    float* d_weights_base;      // base ptr to device side weights
    float* d_activations_base;  // base ptr to device side activations
    float* d_kv_cache_base;     // base ptr to device side KV cache

    // Model prompts
    int* h_prompt;      // [max_B, max_T]
    int* d_prompt;      // [max_B, max_T]
    unsigned int* d_seeds;  // [max_B]

    // Creates host and device prompt buffer.
    // Currently copies a single batch, with no batch offsets from CPU to GPU
    void prompt(char* argv[], const int prompt_len) {
        context.current_seq_len = prompt_len;
        context.current_batch += 1;
        size_t max_context_size = (size_t)config.max_batch * config.max_seq_len * sizeof(int);
        h_prompt = (int*)malloc(max_context_size);
        if (!h_prompt) {
            fprintf(stderr, "Failed to allocate CPU memory for prompt tokens.\n");
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < prompt_len; ++i) {
            h_prompt[i] = atoi(argv[i+2]);
        }

        CUDA_CHECK(cudaMalloc((void**)&d_prompt, max_context_size));
        CUDA_CHECK(cudaMemcpy(d_prompt, h_prompt, prompt_len * sizeof(int), cudaMemcpyHostToDevice));

        fprintf(stderr, "Model Prompted. Prompt ptr located at: %p\n", (void*)d_prompt);
    }
} model;

void calculate_param_buffer_size (size_t* param_size, model_config config) {
    int max_T = config.max_seq_len;
    int V = config.vocab_size;
    int L = config.layers;
    int C = config.channels;
    
    // EMBEDDING
    param_size[WTE_IDX] = (size_t)V * C; // token embedding
    param_size[WPE_IDX] = (size_t)max_T * C; // positional embedding

    // LAYERNORM w1, b1
    param_size[LN_1_ALPHA_IDX] = (size_t)L * C; 
    param_size[LN_1_BETA_IDX] = (size_t)L * C;

    // ATTENTION proj_w_kqv, proj_b_kqv, proj_w_o, proj_b_o
    for (int tensor_idx = 4; tensor_idx < 7; ++tensor_idx) {
        param_size[tensor_idx] = (size_t)L * C * C;
    }

    for (int tensor_idx = 7; tensor_idx < 10; ++tensor_idx) {
        param_size[tensor_idx] = (size_t)L * C;
    }

    // ATTENTION proj_w_o, proj_b_o
    param_size[PROJ_W_O_IDX] = (size_t)L * C * C;
    param_size[PROJ_B_O_IDX] = (size_t)L * C;

    // LAYERNORM w2, b2
    param_size[LN_2_ALPHA_IDX] = (size_t)L * C; 
    param_size[LN_2_BETA_IDX] = (size_t)L * C;

    // FFN w1, b1, w2, b2
    param_size[FFN_W1_IDX] = (size_t)L * C * 4 * C;
    param_size[FFN_B1_IDX] = (size_t)L * 4 * C;
    param_size[FFN_W2_IDX] = (size_t)L * 4 * C * C;
    param_size[FFN_B2_IDX] = (size_t)L * C;

    // LAYERNORM fw, fb
    param_size[LN_FINAL_ALPHA_IDX] = (size_t)C;
    param_size[LN_FINAL_BETA_IDX] = (size_t)C;
}

// keeping batch==1 for now. declaring max_T instead of current 
// since model can progress to max_seq_len if <EOS> is never met...
// Excluded max_B, since currently working at a single batch. Normally, these activations must be declared of size max_B*max_T*C.
void calculate_activ_buffer_size (size_t* activ_size, model_config config) {
    int max_T = config.max_seq_len;
    int max_B = config.max_batch;
    int V = config.vocab_size;
    int L = config.layers; // overwrite at each layer (seems possible)
    int C = config.channels;

    activ_size[EMBED_OUT_IDX] = (size_t)max_B * max_T * C;
    activ_size[X_NORM_IDX] = (size_t)max_B * max_T * C;

    activ_size[Q_SCRATCH_IDX] = (size_t)max_B * max_T * C;
    activ_size[ATTN_OUT_IDX] = (size_t)max_B * max_T * C;
    activ_size[O_PROJ_OUT_IDX] = (size_t)max_B * max_T * C;

    activ_size[MLP_H_IDX] = (size_t)max_B * max_T * 4 * C;
    activ_size[LOGITS_IDX] = (size_t)max_B * max_T * V;
    // populate activ_size array with further hidden activations
}

// Implementation inspired from Andrej Karpathy's llm.c repo :)
// Returns a base ptr to the GPU buffer and slices the GPU buffers to tensor parameters using model_parameters or model_activations
float* malloc_and_point_to_weights(size_t* param_size, model_parameters* params) {

    size_t total_param_size = 0;
    for (size_t i = 0; i < NUM_TENSORS; ++i) {
        total_param_size += param_size[i];
    }

    float* params_memory;
    CUDA_CHECK(cudaMalloc((void**)&params_memory, total_param_size * sizeof(float)));

    // array of adresses to model weight pointers
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln_1_alpha, &params->ln_1_beta, &params->proj_w_q, &params->proj_w_k, &params->proj_w_v, &params->proj_b_q,
        &params->proj_b_k, &params->proj_b_v, &params->proj_w_o, &params->proj_b_o, &params->ln_2_alpha, &params->ln_2_beta, &params->ffn_w1, &params->ffn_b1,
        &params->ffn_w2, &params->ffn_b2, &params->ln_final_alpha, &params->ln_final_beta
    };
    int num_ptrs = sizeof(ptrs) / sizeof(ptrs[0]);
    if (num_ptrs != NUM_TENSORS) {
        fprintf(stderr, "Error: Mismatch in parameter tensor and pointer count! Tensor: %d, Ptr: %d\n", NUM_TENSORS, num_ptrs);
        cudaFree(&params_memory);
        exit(EXIT_FAILURE);
    }

    float* base_offset = params_memory;
    for (size_t i = 0; i < NUM_TENSORS; ++i) {
        *(ptrs[i]) = base_offset;
        base_offset += param_size[i];
    }

    return params_memory;
}

float* malloc_and_point_to_activations(size_t* activ_size, model_activations* activations) {

    size_t total_activ_size = 0;
    for (size_t i = 0; i < NUM_ACTIVATIONS; ++i) {
        total_activ_size += activ_size[i];
    }

    float* activs_memory;
    CUDA_CHECK(cudaMalloc((void**)&activs_memory, total_activ_size * sizeof(float)));

    // array of adresses to model activ pointers
    float** ptrs[] = {
        &activations->embedding_out, &activations->X_norm, &activations->scratch_query, &activations->attn_out, &activations->o_proj_out, 
        &activations->mlp_hidden, &activations->logits
        // add more pointer adresses as more activations need to be sliced
    };
    int num_ptrs = sizeof(ptrs) / sizeof(ptrs[0]);
    if (num_ptrs != NUM_ACTIVATIONS) {
        fprintf(stderr, "Error: Mismatch in activation and pointer count! Activ: %d, Ptr: %d\n", NUM_ACTIVATIONS, num_ptrs);
        cudaFree(&activs_memory);
        exit(EXIT_FAILURE);
    }

    float* base_offset = activs_memory;
    for (size_t i = 0; i < NUM_ACTIVATIONS; ++i) {
        *(ptrs[i]) = base_offset;
        base_offset += activ_size[i];
    }

    return activs_memory;
}

float* malloc_and_point_to_KV_cache(model_config config, KV_cache* model_kv_cache) {
    int max_T = config.max_seq_len;
    int max_B = config.max_batch;
    int L = config.layers;
    int C = config.channels;

    size_t total_KV_cache_size = (size_t)2 * L * max_B * max_T * C;

    float* kv_cache_memory;
    CUDA_CHECK(cudaMalloc((void**)&kv_cache_memory, total_KV_cache_size * sizeof(float)));

    model_kv_cache->key_cache = kv_cache_memory;
    model_kv_cache->value_cache = kv_cache_memory + (size_t)L * max_B * max_T * C;

    return kv_cache_memory;
}

unsigned int* init_model_seeds(model_config config) {
    int max_B = config.max_batch;

    unsigned int* h_seeds;
    h_seeds = (unsigned int*)malloc(max_B * sizeof(unsigned int));
    for (int b = 0; b < max_B; ++b) {
        h_seeds[b] = (unsigned int)time(NULL) + b;
    }

    unsigned int* d_seeds_memory;
    CUDA_CHECK(cudaMalloc((void**)&d_seeds_memory, max_B * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_seeds_memory, h_seeds, max_B * sizeof(unsigned int), cudaMemcpyHostToDevice));
    free(h_seeds);

    return d_seeds_memory;
}

// Initializes the model on GPU with weights loaded
model init_model(model_config config, const char* checkpoint_path) {
    model m;
    m.config = config;
    m.d_seeds = init_model_seeds(m.config);

    // calculate the total memory needed for specific model
    size_t activ_size[NUM_ACTIVATIONS];
    size_t param_size[NUM_TENSORS];
    calculate_param_buffer_size(param_size, m.config);
    calculate_activ_buffer_size(activ_size, m.config);

    // create GPU side buffer with correct weight/activation ptrs and base ptr
    m.d_weights_base = malloc_and_point_to_weights(param_size, &m.d_weights);
    m.d_activations_base = malloc_and_point_to_activations(activ_size, &m.d_activations);
    m.d_kv_cache_base = malloc_and_point_to_KV_cache(m.config, &m.d_kv_cache);

    // load weights onto CPU, then copy to GPU
    size_t total_elements = 0;
    for (int i = 0; i < NUM_TENSORS; ++i) {
        total_elements += param_size[i];
    }
    size_t total_bytes = total_elements * sizeof(float);

    FILE* file = fopen(checkpoint_path, "rb");
    if (!file) {
        fprintf(stderr, "Error: Failed to open checkpoint file at %s\n", checkpoint_path);
        cudaFree(m.d_weights_base);
        exit(EXIT_FAILURE);
    }

    file_header header;
    if (fread(&header, sizeof(file_header), 1, file) != 1) {
        fprintf(stderr, "Error: Failed to read checkpoint header.\n");
        fclose(file);
        cudaFree(m.d_weights_base);
        exit(EXIT_FAILURE);
    }

    if (header.magic != 20241027) {
        fprintf(stderr, "Error: Mismatch in magic nubmer!\n");
        fclose(file);
        cudaFree(m.d_weights_base);
        exit(EXIT_FAILURE);
    }

    if (header.layers != (config.layers) ||
        header.channels != config.channels ||
        header.vocab_size != config.vocab_size) {
        fprintf(stderr, "Error: Mismatch in model config!\n");
        fclose(file);
        cudaFree(m.d_weights_base);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    size_t expected_size = sizeof(file_header) + total_bytes;
    if (file_size != expected_size) {
        fprintf(stderr, "Error: Mismatch on file and expected size!\n");
        fprintf(stderr, "Expected: %zu bytes, Actual: %zu bytes", expected_size, file_size);
        fclose(file);
        cudaFree(m.d_weights_base);
        exit(EXIT_FAILURE);
    }

    fseek(file, sizeof(file_header), SEEK_SET);
    m.h_weights_base = (float*)malloc(total_bytes);
    if (!m.h_weights_base) {
        fprintf(stderr, "Error: Failed to allocate CPU memory for weights.\n");
        fclose(file);
        cudaFree(m.d_weights_base);
        exit(EXIT_FAILURE);
    }

    size_t read_count = fread(m.h_weights_base, sizeof(float), total_elements, file);
    if (read_count != total_elements) {
        fprintf(stderr, "Error: Failed to read %zu elements. Read: %zu elements", total_elements, read_count);
        fclose(file);
        cudaFree(m.d_weights_base);
        exit(EXIT_FAILURE);
    }
    fclose(file);

    CUDA_CHECK(cudaMemcpy(m.d_weights_base, m.h_weights_base, total_bytes, cudaMemcpyHostToDevice));

    free(m.h_weights_base);
    m.h_weights_base = NULL;

    fprintf(stderr, "Model Initialized. GPU weight base: %p\n", (void*)m.d_weights_base);
    fprintf(stderr, "GPU activations base: %p\n", (void*)m.d_activations_base);
    fprintf(stderr, "GPU KV Cache base: %p\n", (void*)m.d_kv_cache_base);
    fprintf(stderr, "GPU seed base: %p\n", (void*)m.d_seeds);

    return m;
}

void free_model(model* m) {
    cudaFree(m->d_weights_base);    // free device side weights
    cudaFree(m->d_activations_base);    // free device side activations
    cudaFree(m->d_kv_cache_base);   // free device side KV cache
    cudaFree(m->d_prompt);
    free(m->h_prompt);
    cudaFree(m->d_seeds);
}

//
// Forward Pass (B == 1) for now
//
void prefill_forward(model* m, cudaStream_t stream, cublasHandle_t cublas_handle) {
    float temperature = 0.7f;
    int current_seq_len = m->context.current_seq_len;
    int B = m->context.current_batch;
    int L = m->config.layers;
    int H = m->config.heads;
    int C = m->config.channels;
    int d_head = C / H; // 768/12 = 64
    int vocab_size = m->config.vocab_size;
    int max_seq_len = m->config.max_seq_len;
    int max_B = m->config.max_batch;
    unsigned int* seeds = m->d_seeds;



    // ACTIVATIONS: 
    // can be declared out of the loop,
    // and can be overwritten at each step
    float* embedding_out = m->d_activations.embedding_out;
    float* X_norm = m->d_activations.X_norm;
    float* q_cache_scratch = m->d_activations.scratch_query;
    float* attn_out = m->d_activations.attn_out;
    float* o_proj_out = m->d_activations.o_proj_out;
    float* mlp_hidden = m->d_activations.mlp_hidden;
    float* logits = m->d_activations.logits;

    // Preprocessing
    // Output: embedding_out [B * current_seq_len, C]
    int* user_prompt = m->d_prompt;
    float* wte = m->d_weights.wte;
    float* wpe = m->d_weights.wpe;
    launch_embedding_v1(
        user_prompt,
        embedding_out, wte, wpe,
        1, current_seq_len, C, 
        max_seq_len,
        stream
    );

    for (int l = 0; l < L; ++l) {

        // embedding_out is the current residual
        // Output: X_norm [B * current_seq_len, C]
        float* alpha_layer = m->d_weights.ln_1_alpha + (l * C);
        float* beta_layer = m->d_weights.ln_1_beta + (l * C);
        layernorm_forward_v1(
            embedding_out,  // Our pre-attn residual
            X_norm, 
            alpha_layer, beta_layer,
            1, current_seq_len, C,
            stream
        );

        // Output: key_cache_layer, value_cache_layer, q_cache_scratch [B * current_seq_len, C]
        float* w_q_layer = m->d_weights.proj_w_q + (l * C * C);
        float* w_k_layer = m->d_weights.proj_w_k + (l * C * C);
        float* w_v_layer = m->d_weights.proj_w_v + (l * C * C);
        float* b_q_layer = m->d_weights.proj_b_q + (l * C);
        float* b_k_layer = m->d_weights.proj_b_k + (l * C);
        float* b_v_layer = m->d_weights.proj_b_v + (l * C);
        // Activation: q_cache_scratch
        float* key_cache_layer = m->d_kv_cache.key_cache + (l * max_B * max_seq_len * C);
        float* value_cache_layer = m->d_kv_cache.value_cache + (l * max_B * max_seq_len * C);
        qkv_proj_append_to_KV_cache(
            cublas_handle, 
            X_norm, 
            w_q_layer, w_k_layer, w_v_layer, 
            b_q_layer, b_k_layer, b_v_layer, 
            key_cache_layer, value_cache_layer, q_cache_scratch,
            1, current_seq_len, C, // Format: (current_batch, current_seq_len, channel)
            max_seq_len,
            stream
        );

        // masking is missing
        // Output: attn_out [B * current_seq_len, C]
        launch_flash_mha_fwd_v1(
            q_cache_scratch, 
            key_cache_layer, 
            value_cache_layer,
            attn_out,
            1, H, current_seq_len, d_head,   // Format: (current_batch, head, current_seq_len, d_head)
            max_seq_len,
            stream
        );

        // Output: o_proj_out [B * current_seq_len, C]
        float* w_o_layer = m->d_weights.proj_w_o + (l * C * C);
        float* b_o_layer = m->d_weights.proj_b_o + (l * C);
        // Activation: attn_out, o_proj_out
        o_proj(cublas_handle,
            attn_out,
            w_o_layer, b_o_layer,
            o_proj_out,
            1, current_seq_len, C,
            stream
        );

        // Output: o_proj_out [B * current_seq_len, C]
        // Activation: o_proj_out(OVERWRITTEN), embedding_out
        launch_residual_add(
            o_proj_out, 
            embedding_out,
            1, current_seq_len, C,
            stream
        );

        // Output: X_norm [B * current_seq_len, C]
        alpha_layer = m->d_weights.ln_2_alpha + (l * C);
        beta_layer = m->d_weights.ln_2_beta + (l * C);
        // Activation: X_norm(OVERWRITTEN), o_proj_out
        layernorm_forward_v1(
            o_proj_out, // Our pre-MLP residual
            X_norm,
            alpha_layer, beta_layer,
            1, current_seq_len, C,
            stream
        );

        // Output: embedding_out [B * current_seq_len, C]
        float* w1_layer = m->d_weights.ffn_w1 + (l * C * 4 * C);
        float* b1_layer = m->d_weights.ffn_b1 + (l * 4 * C);
        float* w2_layer = m->d_weights.ffn_w2 + (l * 4 * C * C);
        float* b2_layer = m->d_weights.ffn_b2 + (l * C);
        // Activation: o_proj_out, X_norm, mlp_hidden, embedding_out(OVERWRITTEN)
        mlp_forward_v1(
            cublas_handle,
            o_proj_out,     // residual
            X_norm,
            mlp_hidden,
            embedding_out,  // overwritten for the next layer
            w1_layer, b1_layer,
            w2_layer, b2_layer,
            1, current_seq_len, C,
            stream
        );
    }

    // Output: embedding_out [B * current_seq_len, C]
    float* final_alpha_layer = m->d_weights.ln_final_alpha;
    float* final_beta_layer = m->d_weights.ln_final_beta;
    // Activation: embedding_out, X_norm(OVERWRITTEN)
    layernorm_forward_v1(
        embedding_out,
        X_norm,
        final_alpha_layer, final_beta_layer,
        1, current_seq_len, C,
        stream
    );

    // Output: logits [B * current_seq_len, vocab_size]
    // Activation: embedding_out, logits
    lm_head_fwd(
        cublas_handle,
        X_norm,
        wte,
        logits,
        1, current_seq_len, C,
        vocab_size,
        stream
    );

    // Output: user_prompt [B, current_seq_len]
    // Activation: logits
    launch_sampler_top_k(
        logits,
        user_prompt,
        seeds,
        1, current_seq_len, vocab_size,
        max_seq_len,
        temperature,
        stream
    );

    m->context.current_seq_len += 1;
}

//
// Main Inference Loop
//
void generate(model* m) {
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int EOS_TOKEN_ID = 50256; // FOR GPT2
    int max_T = m->config.max_seq_len;
    int max_B = m->config.max_batch;
    int current_seq_len = m->context.current_seq_len;
    int current_batch = m->context.current_batch;
    int* h_prompt = m->h_prompt;
    int* d_prompt = m->d_prompt;

    while (m->context.current_seq_len < m->config.max_seq_len) {
        prefill_forward(m, stream, cublas_handle);
        current_seq_len = m->context.current_seq_len;
        CUDA_CHECK(cudaMemcpyAsync(
            h_prompt, 
            d_prompt, 
            max_B * max_T * sizeof(int), 
            cudaMemcpyDeviceToHost,
            stream
        ));
        cudaStreamSynchronize(stream);

        if (h_prompt[current_seq_len-1] == EOS_TOKEN_ID) {
            break;
        } else {
            printf("%d ", h_prompt[current_seq_len - 1]);
            fflush(stdout);
        }
    }
    // Decode Phase
    // include check whether <EOS> token is reached

    // Copy d_prompt to h_prompt

    cublasDestroy(cublas_handle);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char* argv[]) {
    // argc: argument count
    // argv[0] = program name
    // argv[1] = checkpoint path
    // argv[2...] = token IDs

    if (argc < 3) {
        fprintf(stderr, "Error: Missing Arguments.\n");
        fprintf(stderr, "Usage: %s <checkpoint_path> <token 1> <token 2> ...\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Initialize model
    const char* checkpoint_path = argv[1]; // "checkpoints/gpt2_124m.bin"
    model_config GPT2Config = {1024, 1, 50257, 12, 12, 768}; // max_batch = 1
    model cuGPT = init_model(GPT2Config, checkpoint_path);

    // Prompt model
    int prompt_len = argc - 2;
    cuGPT.prompt(argv, prompt_len);
    
    // Print Host Prompt
    for (int i = 0; i < prompt_len; ++i) {
        printf("%d ", cuGPT.h_prompt[i]);
        fflush(stdout);
    }
    
    generate(&cuGPT);

    free_model(&cuGPT);
    return 0;
}