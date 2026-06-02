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
void cpu_layernorm_fwd(float *X, float *X_norm, 
                       float *alpha, float *beta, 
                       int BT, int C);

// Kernel Wrappers (no need)


// Layernorm implementations
void layernorm_forward_v1(float *X, float *X_norm, 
                          float *alpha, float *beta, 
                          int BT, int C, 
                          cudaStream_t stream);
