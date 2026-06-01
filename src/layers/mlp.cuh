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
void cpu_mlp_forward(float *X, float *out, float *W1, float *b1, float *W2, float *b2, int BT, int C);

void cpu_bias_ReLU (float *h_out, float *out, float *b1, int BT, int C);

void cpu_bias_residual (float *X, float *h_out, float *out, float *b2, int BT, int C);

// Kernel Wrappers
void launch_fused_bias_ReLU_v1 (float *h_out, float *b1, int BT, int C, cudaStream_t stream);

void launch_fused_bias_residual_v1 (float *X, float *h_out, float *b2, int BT, int C, cudaStream_t stream);

// MLP Layer implementations

void mlp_forward_v1(cublasHandle_t cublas_handle, 
                    float *X,       // Input [BT, C]
                    float *h_out,   // Hidden output buffer [BT, 4*C]
                    float *out,     // Final output [BT, C]
                    float *W1, float *b1, // Weight and biases of layer 1
                    float *W2, float *b2, // Weight and biases of layer 2
                    int BT, int C, cudaStream_t stream);
