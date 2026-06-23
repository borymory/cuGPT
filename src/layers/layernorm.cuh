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
void cpu_layernorm_fwd(
    float* X,       // [B * current_seq_len, C]
    float* X_norm,  // [B * current_seq_len, C]
    float* alpha,   // [C]
    float* beta,    // [C]
    int B,
    int current_seq_len,
    int C
);

// Layernorm implementations
void layernorm_forward_v1(
    const float* __restrict__ X,        // [B * current_seq_len, C]
    float* X_norm,                      // [B * current_seq_len, C]
    const float* __restrict__ alpha,    // [C]
    const float* __restrict__ beta,     // [C]
    const int B,
    const int current_seq_len, 
    const int C,
    cudaStream_t stream
);
