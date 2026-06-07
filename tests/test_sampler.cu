#include <cstdio>
#include "hpc_utils.cuh"
#include "sampler.cuh"

// -- VERIFY FUNCTIONS --
// For elements wise verification, use cuGPT::validate(...) func given in ./common/hpc_utils.cu

void print_data(const float *data, const int num_elements, const int width) {
  for (int i = 0; i < num_elements; ++i) {
    int col = i % width;
    int row = i / width;

    if (col == 0) std::printf("Row %d: [", row);
    if (col == width-1) {
      std::printf("%f", data[i]);
      std::printf("]\n");
    } else {
      std::printf("%f, ", data[i]);
    }
  }
}

bool test_generation_sampler_v1() {  
  float* logits;        // [B, vocab_size]
  float* u;             // [B, MAX_K]
  int* p;               // [B, MAX_K]
  int* next_tokens;     // [B]

  int B = 2;
  // T = 1 for inference (unused)
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
  print_data(logits, B * vocab_size, vocab_size);

  std::printf("Running CPU Softmax Sampling... | ");
  std::fflush(stdout);
  sample_top_k_from_logits(logits, 
    u, p, 
    next_tokens, 
    B, vocab_size, temp);
  std::printf("✅\n");
  print_data(next_tokens, B, B);

  // FREE MEMORY ALLOCATION
  std::free(logits);
  std::free(u);
  std::free(p);
  std::free(next_tokens);

  return true;
}

int main(void) {
  std::printf("Running Test...\n");
  if (test_generation_sampler_v1()) std::printf("It ran!\n");
}