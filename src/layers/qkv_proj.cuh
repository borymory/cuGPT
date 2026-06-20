#pragma once
#include <algorithm>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string> // for std::string

#define FULL_MASK 0xffffffffu // unsigned, safer in bit shifting

// CPU Reference Funcs
void cpu_proj_append_to_KV_cache(
    float* X_norm, // [B * seq_len, C]
    float* w_q, // [C, C]
    float* w_k, // [C, C]
    float* w_v, // [C, C]
    float* b_q, // [C]
    float* b_k, // [C]
    float* b_v, // [C]
    float* key_cache,  // Written: [B, seq_len, C]
    float* value_cache, // Written: [B, seq_len, C]
    float* q_scratch, // Written: [B, seq_len, C]
    int B,
    int current_seq_len,
    int C
);


// qkv_proj kernel
void qkv_proj_append_to_KV_cache(
    cublasHandle_t cublas_handle,
    float* __restrict__ X_norm, // [B * seq_len, C]
    float* __restrict__ w_q, // [C, C]
    float* __restrict__ w_k, // [C, C]
    float* __restrict__ w_v, // [C, C]
    const float* __restrict__ b_q, // [C]
    const float* __restrict__ b_k, // [C]
    const float* __restrict__ b_v, // [C]
    float* __restrict__ key_cache,  // Written: [B, seq_len, C]
    float* __restrict__ value_cache, // Written: [B, seq_len, C]
    float* __restrict__ q_scratch, // Written: [B, seq_len, C]
    const int B,
    const int current_seq_len,
    const int C,
    cudaStream_t stream
);
