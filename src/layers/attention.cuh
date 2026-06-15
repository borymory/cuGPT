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
    const float* Q, // Shape: [B, H, N, d]
    const float* K, // Shape: [B, H, N, d]
    const float* V, // Shape: [B, H, N, d]
    float *S,       // Shape: [N, N] (scrap)
    float *O,       // Shape: [B, H, N, d]
    int B,
    int H,
    int N,
    int d
);

// Kernel Launcher
void launch_flash_attn_forward_kernel(
    const float* __restrict__ Q, // Shape: [B, H, N, d]
    const float* __restrict__ K, // Shape: [B, H, N, d]
    const float* __restrict__ V, // Shape: [B, H, N, d]
    float* __restrict__ O,       // Shape: [B, H, N, d]
    const int H,
    const int N,                 // Same as sequence length
    const int d,
    cudaStream_t stream
);