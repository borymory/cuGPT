#pragma once
#include <algorithm>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string> // for std::string

#define FULL_MASK 0xffffffffu // unsigned, safer in bit shifting

// Helpers
void sample_top_k_from_probs(
    const float *u, 
    const int *p, 
    int *next_tokens, 
    const int B, 
    const int MAX_K);


// Sampler Implementation
void sample_top_k_from_logits(
    const float *logits, 
    float *u, int *p, 
    int *next_tokens, 
    const int B, 
    const int vocab_size, 
    const float temp);