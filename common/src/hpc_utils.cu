#include "hpc_utils.cuh"

namespace cuGPT {

  double get_time_ms() {
      struct timeval tv;
      gettimeofday(&tv, nullptr);
      // Convert seconds and microseconds to a single millisecond value
      return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
  }

  void initArray(float *A, const int d) {
    for (unsigned int i = 0; i < d; ++i) A[i] = (float)std::rand() / RAND_MAX;
  }

  void initMatrix(float *B, const int N, const int d) {
    for (unsigned int i = 0; i < N; i++) {
      for (unsigned int j = 0; j < d; ++j) B[i * d + j] = (float)std::rand() / RAND_MAX;
    }
  }

  void copyArray (float *src, float *dst, const int d) {
    for (unsigned int i = 0; i < d; ++i) dst[i] = src[i];
  }

  void copyMatrix(float *src, float *dst, const int N, const int d) {
    for (unsigned int i = 0; i < N; i++) {
      for (unsigned int j = 0; j < d; ++j) dst[i * d + j] = src[i * d + j];
    }
  }

  void copy(float *src, float *dst, const int size) {
    for (unsigned int i = 0; i < size; ++i) dst[i] = src[i];
  }

  // generic verifier
  bool validate(float *gpu_res, float *cpu_res, int size) {
    for (unsigned int i = 0; i < size; i++) {

        float diff = std::abs(gpu_res[i] - cpu_res[i]);
        float relative_err = diff / std::abs(cpu_res[i]);

        if (std::isnan(gpu_res[i]) || std::isnan(cpu_res[i]) ||
            std::isinf(gpu_res[i]) || std::isinf(cpu_res[i])) {
            std::printf("Validation Failure: Inf or NaN detected at index %d\n", i);
            std::printf("GPU: %f\n", gpu_res[i]);
            std::printf("CPU: %f\n", cpu_res[i]);
            return false;
        }
        
        if (relative_err > 1e-5) {
          std::printf("First error at index %d\n", i);
          std::printf("GPU: %f\n", gpu_res[i]);
          std::printf("CPU: %f\n", cpu_res[i]);
          return false;
        }
    }
    return true;
  }

  void checkLast(const char* const file, const int line)
    {
        cudaError_t const err{cudaGetLastError()};
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                    << std::endl;
            std::cerr << cudaGetErrorString(err) << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

}


