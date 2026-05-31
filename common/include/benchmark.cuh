#pragma once
#include "hpc_utils.cuh"

namespace cuGPT::Benchmark {

    template <typename T>
    void benchmark_kernel (
        std::function<T(cudaStream_t)> bound_function, 
        cudaStream_t stream, 
        size_t bytes_moved,
        float cpu_ref_time, 
        size_t num_repeats, 
        size_t num_warmups, 
        bool flush_l2_cache)
    {
        // CREATE CUDA EVENT
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // L2 CACHE INFO
        int device = 0;
        int l2_size = 0;
        float *d_flush_ptr;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, device));
        size_t flush_bytes = l2_size * 2;

        // ALLOCATE L2 BUFFER
        CUDA_CHECK(cudaMalloc(&d_flush_ptr, flush_bytes));

        // WARMUP RUNS
        for (size_t i = 0; i < num_warmups; ++i) {
            bound_function(stream);
        }
        cudaStreamSynchronize(stream);

        // EXECUTION LOOP
        float total_ms = 0.0f;
        float partial_ms;
        for (size_t i = 0; i < num_repeats; ++i) {
            if (flush_l2_cache) {
                CUDA_CHECK(cudaMemsetAsync(d_flush_ptr, 0, flush_bytes, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CHECK_LAST_CUDA_ERROR();
            }

            CUDA_CHECK(cudaEventRecord(start, stream));
            bound_function(stream);
            CUDA_CHECK(cudaEventRecord(stop, stream));

            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&partial_ms, start, stop));
            total_ms += partial_ms;
        }

        // CALC AVG_TIME
        float avg_ms = total_ms / num_repeats;

        // BANDWIDTH CALCULATION
        // Formula: Bytes moved = (Read N * d + write N) * 4 Bytes, time = avg_ms
        double gb = static_cast<double>(bytes_moved) / 1e9;
        double bandwidth = gb / (avg_ms / 1000.0);

        // PRINT RESULT
        std::printf("-- Benchmark Result --\n");
        std::printf("Average Time:  %.4f ms\n", avg_ms);
        std::printf("Throughput:    %.2f GB/s\n", bandwidth);
        std::printf("Speedup from CPU:  %.2fx\n", cpu_ref_time / avg_ms);
        std::cout << "Flush: " << std::boolalpha << flush_l2_cache << std::endl;

        // CLEANUP
        CUDA_CHECK(cudaFree(d_flush_ptr));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
}