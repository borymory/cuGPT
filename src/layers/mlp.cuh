#pragma once
#include <algorithm>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string> // for std::string

#define FULL_MASK 0xffffffffu // unsigned, safer in bit shifting

void cpu_mlp_forward(float *X, float *out, float *W1, float *b1, float *W2, float *b2, int BT, int C);

void cpu_bias_ReLU (float *h_out, float *out, float *b1, int BT, int C);

void launch_fused_bias_ReLU_v1 (float *h_out, float *b1, int BT, int C, cudaStream_t stream);
