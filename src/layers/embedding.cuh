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
void cpu_embedding(int *inputs, float *out, 
                    float *wte, float *wpe, 
                    int B, int T, int C);

// Kernel Wrappers (no need)


// Layernorm implementations
void launch_embedding_v1(int *inputs, float *out, 
                        float *wte, float *wpe, 
                        int B, int T, int C, 
                        int max_length, cudaStream_t stream);
