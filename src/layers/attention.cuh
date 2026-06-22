#pragma once
#include <algorithm>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string> // for std::string

#define FULL_MASK 0xffffffffu // unsigned, safer in bit shifting

// CPU Functions
void cpu_attention(
    const float* Q, // Shape: [B, N, H, d]
    const float* K, // Shape: [B, N, H, d]
    const float* V, // Shape: [B, N, H, d]
    float *S,       // Shape: [N, N] (scrap)
    float *O,       // Shape: [B, N, H, d]
    int B,
    int H,
    int N,
    int d,
    int max_T
);

// Kernel Launcher
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
);