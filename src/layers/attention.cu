#include "hpc_utils.cuh"
#include "attention.cuh"

// CPU Func
void cpu_attention(
    const float* Q, // Shape: [B, H, N, d]
    const float* K, // Shape: [B, H, N, d]
    const float* V, // Shape: [B, H, N, d]
    float *S,       // Shape: [N, N] (scrap)
    float *O,       // Shape: [B, H, N, d]
    int B,
    int H,
    int N,
    int d
) 
{
    float scale = 1.0f / sqrtf(d);
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            int offset = b * (H * N * d) + h * (N * d);
            const float* Q_local = Q + offset;
            const float* K_local = K + offset;
            const float* V_local = V + offset;
            float* O_local = O + offset;

            // QK matmul: Q[N, d], K[N, d]
            for (int q_row = 0; q_row < N; ++q_row) {
                for (int k_row = 0; k_row < N; ++k_row) {
                    // Casual Mask
                    if (q_row >= k_row) {
                        float qk_sum = 0.0f;
                        for (int k = 0; k < d; ++k) {
                            qk_sum += Q_local[q_row * d + k] * K_local[k_row * d + k];
                        }
                        S[q_row * N + k_row] = qk_sum * scale;
                    } else {
                        S[q_row * N + k_row] = -INFINITY;
                    }
                }
            }

            // Online Softmax
            for (int s_row = 0; s_row < N; ++s_row) {
                float m = -INFINITY;
                float l = 0.0f;

                for (int k = 0; k < N; ++k) {
                    float val = S[s_row * N + k];
                    
                    float m_prev = m;
                    m = fmaxf(m, val);
                    
                    l *= expf(m_prev - m);
                    l += expf(val - m);
                }

                for (int k = 0; k < N; ++k) {
                    if (s_row >= k) {
                        S[s_row * N + k] = expf(S[s_row * N + k] - m) / l;
                    } else {
                        S[s_row * N + k] = 0.0f;
                    }
                }
            }

            // PV Matmul: P[N, N], V[N, d]
            for (int p_row = 0; p_row < N; ++p_row) {
                for (int v_col = 0; v_col < d; ++v_col) {
                    float pv_sum = 0.0f;
                    for (int k = 0; k < N; ++k) {
                        pv_sum += S[p_row * N + k] * V_local[k * d + v_col];
                    }
                    O_local[p_row * d + v_col] = pv_sum;
                }
            }
        }
    }
}


//
// Helper Funcs
//

// Assumes Row-Major layout: Pads SMEM with zeros if N is not divisible by Br or Bc
__device__ __forceinline__ void fload_to_smem(float* __restrict__ shared_dst, 
    const float* __restrict__ global_src,
    int transpose, 
    const int row_dim,
    const int col_dim,
    const int padding,
    const int global_row_offset,    // row offset from the beggining of [N, d]
    const int max_rows
) 
{
    int num_elements = row_dim * col_dim;

    if (transpose) {
        int ldsmem = row_dim + padding; // How many elements to get to the other row

        for (int idx = threadIdx.x; idx < num_elements; idx += blockDim.x) {
            int s_col = idx / col_dim;
            int s_row = idx % col_dim;
            int s_idx = s_row * ldsmem + s_col;
            if ((global_row_offset + s_col) < max_rows) {
                shared_dst[s_idx] = global_src[idx];
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
            if ((global_row_offset + s_row) < max_rows) {
                shared_dst[s_idx] = global_src[idx];
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
                if (global_col < N) {
                    // Casual Masking s_S
                    if (global_row >= global_col) {
                        float qk_sum = 0.0f;

                        #pragma unroll
                        for (int k = 0; k < d; ++k) {
                            float q_val = s_Q[warp_row * (d+1) + k];
                            float k_val = s_K[k * (Bc+1) + idx];
                            qk_sum += q_val * k_val;
                        }
                        s_S[warp_row * (Bc+1) + idx] = qk_sum * scale;
                    } else {
                        s_S[warp_row * (Bc+1) + idx] = -INFINITY;
                    }
                }
            }
            __syncwarp();

            // Reduce s_S into registers, doing statistics
            #pragma unroll
            for (int c = 0; c < COLS_PER_WARP; ++c) {
                int idx = warpLane + (c * 32);
                int global_col = j + idx;
                if (global_col < N) {
                    float val = s_S[warp_row * (Bc+1) + idx];
                    
                    float m_old = m_i[r];     // store old local max
                    m_i[r] = fmaxf(m_i[r], val);    // obtain new local max

                    if (m_i[r] != -INFINITY) {
                        l_i[r] *= expf(m_old - m_i[r]); // scale old norm
                        l_i[r] += expf(val - m_i[r]);   // add current contribution   
                    } else {
                        l_i[r] = 0.0f;
                    }
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
                float val = s_S[warp_row * (Bc+1) + idx];
                int global_col = j + idx;
                if (global_col < N) {
                    s_S[warp_row * (Bc+1) + idx] = expf(val - m_i[r]);
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
// GPU Kernel
//

// For GPT2: d = 64, H = 12, C = 768
// gridDim.x = H
// gridDim.y = B
// BlockDim.x = 256
// Br, Bc = 64
// Register of thread elements per row of O = d / 32.
// Number of rows of a warp = Br / warp_count.
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
    const float* __restrict__ Q, // Shape: [B, H, N, d]
    const float* __restrict__ K, // Shape: [B, H, N, d]
    const float* __restrict__ V, // Shape: [B, H, N, d]
    float* __restrict__ O,       // Shape: [B, H, N, d]
    const int H,
    const int N                 // Same as sequence length
) {
    int head_idx = blockIdx.x;
    int batch_idx = blockIdx.y;

    int stride_h = N * d;
    int stride_b = H * N * d;

    int block_offset = batch_idx * stride_b + head_idx * stride_h;

    // Block Pointer Offsets
    const float *Q_local = Q + block_offset;
    const float *K_local = K + block_offset;
    const float *V_local = V + block_offset;
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
        fload_to_smem(s_Q, Q_local + i * d, 0, Br, d, 1, i, N);
        __syncthreads();

        // Inner Loop over K and V: 
        // Populates m_prev, l_prev with final softmax and
        // thread_res_O register cache with the final values
        for (unsigned int j = 0; j < N; j += Bc) {

            // Populate rest smem: transpose K
            fload_to_smem(s_K, K_local + j * d, 1, Bc, d, 1, j, N);
            fload_to_smem(s_V, V_local + j * d, 0, Bc, d, 1, j, N);
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
                        O_block[warp_row * d + idx] = inv_l * thread_res_O[r * ELEMENTS_PER_ROW + c];
                    }
                }
            }
        } 
    }
}

//
// Kernel Wrappers
//
void launch_flash_mha_fwd_v1(
    const float* __restrict__ Q, // Shape: [B, H, N, d]
    const float* __restrict__ K, // Shape: [B, H, N, d]
    const float* __restrict__ V, // Shape: [B, H, N, d]
    float* __restrict__ O,       // Shape: [B, H, N, d]
    const int B,
    const int H,
    const int N,                 // Same as sequence length
    const int d,
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
            flash_mha_fwd_v1<Br, Bc, 64, ROWS_PER_WARP, COLS_PER_WARP, ELEMENTS_PER_ROW, warp_count><<<gridDim, blockDim, shared_mem_bytes, stream>>>(Q, K, V, O, H, N);
            CHECK_LAST_CUDA_ERROR();
            break;
        default:
            fprintf(stderr, "FlashAttn ERROR: No given d case is found: d=%d\n", d);
            exit(EXIT_FAILURE);
    }
}