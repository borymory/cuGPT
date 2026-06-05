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
void cpu_lm_head_fwd(float *X_final, float *wte, 
    float *logits, int BT, 
    int C, int vocab_size);


// LM head implementations
void lm_head_fwd(cublasHandle_t cublas_handle, 
    float *X_final, 
    float *wte, 
    float *logits, 
    int BT, 
    int C, 
    int vocab_size, 
    cudaStream_t stream);