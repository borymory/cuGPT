#include "hpc_utils.cuh"
#include "attention.cuh"

//
// Helper Funcs
//
__device__ __forceinline__ void online_softmax(float* __restrict__ s_S, 
half * __restrict__ s_P,
float* __restrict__ m_local,
float* __restrict__ l_local,
const int Br, 
const int Bc,
const int lds)
{
    // we have 128 threads (4 warps)
    // A warp works on 16 many rows.
    int warp_id = threadIdx.x / 32;
    float *s_warp = s_S + (warp_id * 16) * lds;
    half *p_warp = s_P + (warp_id * 16) * lds;

    for (int row_offset = 0; row_offset < 16; ++row_offset) {
        s_warp += row_offset * lds;
        float m_i = m[row_offset];
        float d_i = l[row_offset];
        // Reduce s_warp into reg
        for (int i = threadIdx.x; i < Bc; i += 32) {
            float m_old = m_i;
            float val = s_warp[i]
            if (val > m_i) {
                m_i = val;
                l_i = l_i * expf(m_old - m_i) + 1.0f;
            } else {
                l_i += expf(val - m_i);
            }
        }

        for (unsigned int mirrorIdx = 1; mirrorIdx <= 16; mirrorIdx <<= 1) {
            float m_j = __shfl_xor_sync(FULL_MASK, m_i, mirrorIdx);     // obtain m_j from another thread
            float l_j = __shfl_xor_sync(FULL_MASK, l_i, mirrorIdx);     // obtain l_j from another thread

            float m_old = m_i;
            if (m_j > m_i) {
                m_i = m_j;
                l_i = l_i * expf(m_old - m_i) + l_j;
            } else {
                l_i = l_i + l_j * expf(m_j - m_i);
            }
        }

        // Load to s_P as FP16
        for (int i = threadIdx.x; i < Bc; i += 32) {
            p_warp[i] = __float2half(expf(s_warp[i] - m_i));
        }

        // Update row result
        m_local[row_offset] = m_i;
        l_local[row_offset] = l_i;
    }

}


// GMEM->SMEM load helper with FP32->FP16 casting. Block stride loop, functional with only 1D blocks
__device__ __forceinline__ void float2half_load_to_smem(half* __restrict__ shared_dst, 
    const float* __restrict__ global_src, 
    const int num_elements) 
{
    const int tid = threadIdx.x;

    for (int i = tid; i < num_elements; i += blockDim.x) {
        shared_dst[i] = __float2half(global_src[i]);
    }
}

__device__ __forceinline__ void tensor_tile_attention(const half* __restrict__ s_Q, 
const half* __restrict__ s_K, 
const half* __restrict__ s_V,
float* __restrict__ s_S, 
half * __restrict__ s_P,
float* __restrict__ m_local,
float* __restrict__ l_local,
float* __restrict__ m_old,
float* __restrict__ l_old,
wmma::fragment<wmma::accumulator, 16, 16, 16, float> (&o_frag)[4],
const float scale, 
const int d,
const int Br,
const int Bc)
{   
    int warp_id = threadIdx.x / 32; // Ranges [0, 3]

    const int lda = d;  // s_Q is [Br, d]
    const int ldb = d;  // s_K is [Bc, d]
    const int lds = Bc;  // s_S is [Br, Bc]

    half* q_warp = const_cast<half>(s_Q) + (warp_id * 16) * lda;
    half* k_warp = const_cast<half>(s_K);
    float* s_warp = s_S + (warp_id * 16) * lds;

    // Declare fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;

    // Compute QK^T
    #pragma unroll
    for (int col_step = 0; col_step < Bc; col_step += 16) {

        wmma::fill_fragment(s_frag, 0.0f);

        // Loop over dimension d in tiles
        #pragma unroll
        for (int k = 0; k < d; k += 16) {
            // Load 16x16 tiles into respective fragments
            half* q_tile = q_warp + k;
            wmma::load_matrix_sync(q_frag, q_tile, lda);

            half* k_tile = k_warp + col_step * ldb + k;
            wmma::load_matrix_sync(k_frag, k_tile, ldb);

            // Accumulate s_frag += q_frag * k_frag
            wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
        }

        // Load 16x16 result back to SMEM
        float* s_tile = s_warp + col_step;
        wmma::store_matrix_sync(s_tile, s_frag, lds, wmma:layout_row_major);
    }

    // This fills m_local and l_local, and updates s_P in FP16
    online_softmax(s_S, s_P, m_local, l_local, Br, Bc, lds);
    __syncthreads();

    // Calculate new global stats
    float m_new[16];
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        m_new[r] = fmaxf(m_old[r], m_local[r]);
    }

    // Update previous O accumulators in registers
    int lane_id = threadIdx.x % 32;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int tile_row = (lane_id % 4) * 2 + (i / 4) + (lane_id >= 16 ? 8 : 0);

        // Previous block scale factor
        float scale_factor = expf(m_old[tile_row] - m_new[tile_row]);

        #pragma unroll
        for (int o_col = 0; o_col < 4; ++o_col) {
            o_frag[o_col].x[i] *= scale_factor;
        }
    }

    // PV Matmul
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;

    half* p_warp = s_P + (warp_id * 16) * Bc;
    half* v_warp = const_cast<half>(s_V);

    #pragma unroll
    for (int k_step = 0; k_step < Bc; k_step += 16) {
        half* p_tile = p_warp + k_step;
        wmma::load_matrix_sync(p_frag, p_tile, Bc);

        #pragma unroll
        for (int o_col = 0; o_col < 4; ++o_col) {
            half* v_tile = v_warp + k_step * d + (o_col * 16);
            wmma::load_matrix_sync(v_frag, v_tile, d);

            // Accumulate onto scaled o_frag
            wmma::mma_sync(o_frag[o_col], p_frag, v_frag, o_frag[o_col]);
        }
    }

    // Store new stats as old
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        float scale_old = expf(m_old[r] - m_local[r]);
        float scale_local = expf(m_local[r] - m_new[r]);

        l_old[r] = l_old[r] * scale_old + l_local[r] * scale_local;
        m_old[r] = m_new[r];
    }

}

//
// GPU Kernel
//

// Assert Br/(blockDim.x/32) = 16 for Maximum FP16 Tensor utilization
// Assert H*d = C
// d = 64, H = 12 for GPT-2, C = 768
// gridDim.x = C/d
// gridDim.y = B
// BlockDim.x = 128
// Br, Bc = 64
template<const int Br, const int Bc>
__global__ void flash_attn_forward_kernel(
    const float* __restrict__ Q, // Shape: [B, H, N, d]
    const float* __restrict__ K, // Shape: [B, H, N, d]
    const float* __restrict__ V, // Shape: [B, H, N, d]
    float* __restrict__ O,       // Shape: [B, H, N, d]
    const int H,
    const int N,                 // Same as sequence length
    const int d
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
    extern __shared__ half s_mem[];
    half* s_Q = s_mem;         // Size Br * d
    half* s_K = s_Q + Br * d;  // Size Bc * d
    half* s_V = s_K + Bc * d;  // Size Bc * d
    half* s_P = s_V + Bc * d;  // Size Br * Bc
    float* s_S = reinetpret_cast<float>(s_P + Br * Bc); // Size Br * Bc

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    const float scale = 1.0f / sqrtf(d);

    // Running Statistics
    float m_local[16];
    float l_local[16];
    float m_old[16];
    float l_old[16];
    
    // Outer Loop over Q and O
    for (unsigned int i = 0; i < N; i += Br) {

        // Initialize running statistics for Q_i block
        for (int r = 0; r < 16; r++) {
            m_old[r] = -INFINITY;
            l_old[r] = 0.0f;
        } 

        // Initialize o_frag accumulator fragments in registers
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[4];
        #pragma unroll
        for(int col = 0; col < 4; ++col) {
            wmma::fill_fragments(o_frag[col], 0.0f);
        }


        // Populate s_Q
        float2half_load_to_smem(s_Q, Q_local + i * d, Br * d);
        __syncthreads();

        // Inner Loop over K and V
        for (unsigned int j = 0; j < N; j += Bc) {

            // Populate rest smem
            float2half_load_to_smem(s_K, K_local + j * d, Bc * d);
            float2half_load_to_smem(s_V, V_local + j * d, Bc * d);
            __syncthreads();

            // Attention Tile: Scales O_frag and adds current PV contribution
            tensor_tile_attention(
                s_Q, s_K, s_V, s_S, s_P,
                m_local, l_local, m_old, l_old,
                o_frag, scale, d, Br, Bc
            );
            __syncthreads();
        }

        // Normalize O registers
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            int tile_row = (lane_id % 4) * 2 + (k / 4) + (lane_id >= 16 ? 8 : 0);

            float inv_l = 1.0f / l_old[tile_row];

            #pragma unroll
            for (int o_col = 0; o_col < 4; ++o_col) {
                o_frag[o_col].x[k] *= inv_l;
            }
        }

        // Store output registers to s_S as buffer
        #pragma unroll
        for (int o_col = 0; o_col < 4; ++o_col) {
            float* s_out_tile = s_S + (warp_id * 16) * d + (o_col * 16);
            wmma::store_matrix_sync(s_out_tile, o_frag[o_col], d, wmma::layout_row_major);
        }
        __syncthreads();

        // Copy finalized O output from buffer to Global Memory o
        int total_elements = Br * d;
        #pragma unroll
        for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
            (O_local + i * d)[idx] = s_S[idx];
        }
        __syncthreads();
    }
}

//
// Kernel Wrappers
//
void flash_attn_forward(const float* __restrict__ Q, // Shape: [B, H, N, d]
    const float* __restrict__ K, // Shape: [B, H, N, d]
    const float* __restrict__ V, // Shape: [B, H, N, d]
    float* __restrict__ O,       // Shape: [B, H, N, d]
    const int B,
    const int T,                 // Same as N
    const int C,
    const int H,
    cudaStream_t stream)
{
    const int d = CEIL_DIV(C, H);
    int grid_y = B;
    int grid_x = H;

    dim3 blockDim(128);;
    dim3 gridDim(grid_x, grid_y);

    flash_attn_forward_kernel<gridDim, blockDim, 0, stream>(K, Q, V, O, H, T, d);
    CHECK_LAST_CUDA_ERROR();
}