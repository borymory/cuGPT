#include "tests.cuh"

// -- VERIFY FUNCTIONS --
// For elements wise verification, use validate func given in ./common/hpc_utils.cu

int main(void) {
  std::printf("Running MLP Test...\n");
  if (test_fused_bias_ReLU_v1()) std::printf("Succes!");
}