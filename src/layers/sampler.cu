#include "hpc_utils.cuh"
#include "sampler.cuh"

// Simple sampler implementation from NVIDIA paper

// Generation Case: T==1 (if T!=1, use GPU Kernels!)
// logits[B, vocab_size], top_k_probs[B, MAX_K], top_k_indices[B, MAX_K]
// returns list of indices and respective probs for sampling the next token.
void online_softmax_topk(const float *logits, float *top_k_probs, int *top_k_indices, const int B, const int vocab_size, const int MAX_K) {

    for (unsigned int batch_offset = 0; batch_offset < B; ++batch_offset) {
        float m = -INFINITY;
        float norm = 0;

        float u[MAX_K];
        int p[MAX_K];
        for (unsigned int k = 0; k < MAX_K; ++k) {
            u[k] = -INFINITY;
            p[k] = -1;
        }

        const float *batch_logits = logits + (batch_offset * vocab_size);

        for (int elem_idx = 0; elem_idx < vocab_size; ++elem_idx) {
            float elem = batch_logits[elem_idx];

            float m_old = m;
            if (elem > m) {
                m = elem;
                norm = norm * expf(m_old - m) + 1.0f; // if its the new max, then the running contribution is 1.of
            } else {
                norm += expf(elem - m);
            }

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

        float *dest_probs = top_k_probs + (batch_offset * MAX_K);
        int *dest_indices = top_k_indices + (batch_offset * MAX_K);

        for (int k = 0; k < MAX_K; ++k) {
            dest_probs[k] = expf(u[k] - m) / norm;
            dest_indices[k] = p[k];
        }
    }
}


// u [B, MAX_K], p [B, MAX_K], next_tokens[B] (gives selected token ID for each batch)
void sample_top_k_probs(const float *u, const int *p, int *next_tokens, const int B, const int MAX_K) {
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

// Input: logits[B, vocab_size]
// Output: next_tokens[B]
void sample_top_k_from_logits(const float *logits, float *u, int *p, int *next_tokens, const int B, const int vocab_size, const int MAX_K) {
    online_softmax_topk(logits, u, p, B, vocab_size, MAK_K);
    sample_top_k_probs(u, p, next_tokens, B, MAX_K);
}