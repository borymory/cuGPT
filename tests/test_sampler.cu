#include <cstdio>
#include "hpc_utils.cuh"
#include "lm_head.cuh"

// -- VERIFY FUNCTIONS --
// For elements wise verification, use cuGPT::validate(...) func given in ./common/hpc_utils.cu

void print_logits(float* logits, const int B, const int vocab_size) {
  for (int b = 0; b < B; ++b) {
    std::printf("Batch %b:\n", b);
    const float* batch_logits = logits + b * vocab_size;

    std::printf("Logits: [");
    for (int i = 0; i < vocab_size - 1; ++i) {
      float val = batch_logits[i];
      std::printf("%f, ", val);
    }
    std::printf("%f]\n", batch_logits[vocab_size - 1]);
  }
}

void test_sampler_v1() {  
  float* logits;        // [B, vocab_size]
  float* u              // [B, MAX_K]
  int* p                // [B, MAX_K]
  int* next_tokens      // [B]

  int B = 2;
  int T = 1;  // For inference, generation phase
  int vocab_size = 20; // Accurately: 50257
  int MAX_K = 5;
  float temp = 0.1f;  // Almost greedy-decoding

  
  // MEMORY - INITIALIZATIONS
  logits = (float*)std::malloc(B * vocab_size * sizeof(float));
  u = (float*)std::malloc(B * MAX_K * sizeof(float));
  p = (int*)std::malloc(B * MAX_K * sizeof(int));
  next_tokens = (int*)std::malloc(B * sizeof(int));

  // -- Init Logits --
  for (int i = 0; i < B * vocab_size; ++i) {
    int num = i % vocab_size;
    logits[i] = (float)num + 0.1f;
  }
  print_logits(logits, B, vocab_size);

  std::printf("Running CPU Softmax Sampling... | ");
  std::fflush(stdout);
  sample_top_k_from_logits(logits, 
    u, p, 
    next_tokens, 
    B, vocab_size, temp);
  std::printf("✅\n");

  // FREE MEMORY ALLOCATION
  std::free(logits);
  std::free(u);
  std::free(p);
  std::free(next_tokens);
}

int main(void) {
  std::printf("Running Test...\n");
  if (test_lm_head_v1()) std::printf("It ran!\n");
}