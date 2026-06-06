#include "hpc_utils.cuh"
#include "lm_head.cuh"

//
// CPU Reference
//

// X_final[BT, C], wte[vocab_size, C], logits[BT, vocab_size]
void cpu_lm_head_fwd(float *X_final, float *wte, 
                    float *logits, int BT, 
                    int C, int vocab_size) {
    for (unsigned int i = 0; i < BT; ++i) {
        for (unsigned int j = 0; j < vocab_size; ++j) {
            
            float sum = 0.0f;
            for (unsigned int k = 0; k < C; ++k) {
                sum += X_final[i * C + k] * wte[j * C + k];
            }
            logits[i * vocab_size + j] = sum;
        }
    }
}

//
// LM Head implementation
//

// X_final[BT, C], wte[vocab_size, C], logits[BT, vocab_size]
void lm_head_fwd(cublasHandle_t cublas_handle, 
                float *X_final, 
                float *wte, 
                float *logits, 
                int BT, 
                int C, 
                int vocab_size, 
                cudaStream_t stream) {
    
    cublasSetStream(cublas_handle, stream);

    // X_final * wte^T
    // X_final [BT, C]
    // wte  [vocab_size, C]
    cuGPT::gemm_transposed(cublas_handle, X_final, wte, logits, BT, vocab_size, C);

}
