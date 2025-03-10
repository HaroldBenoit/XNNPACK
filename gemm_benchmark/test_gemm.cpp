#include <chrono>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <tuple>
#include "subgraph_gemm.h"

// Naive reference GEMM implementation
template<typename T>
void reference_gemm(
    size_t batch_size,
    size_t in_features,
    size_t out_features,
    const T* input,
    const T* weight,
    const T* bias,
    T* output) {
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t o = 0; o < out_features; o++) {
            T sum = bias ? bias[o] : T(0);
            for (size_t i = 0; i < in_features; i++) {
                sum += input[b * in_features + i] * weight[o * in_features + i];
            }
            output[b * out_features + o] = sum;
        }
    }
}

// Helper to measure execution time
template<typename F>
double measure_time_ms(F&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Helper to generate random data
template<typename T>
void fill_random(std::vector<T>& data, T min = -1.0, T max = 1.0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    for (auto& x : data) {
        x = static_cast<T>(dis(gen));
    }
}

// Helper to compare results
template<typename T>
bool compare_results(const std::vector<T>& a, const std::vector<T>& b, float tolerance = 1e-6) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (std::abs(a[i] - b[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": " 
                      << a[i] << " vs " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

void run_benchmark(size_t batch_size, size_t in_features, size_t out_features, int num_runs = 10) {
    std::cout << "\nBenchmarking GEMM: " 
              << batch_size << "x" << in_features << "x" << out_features << std::endl;
    
    // Allocate memory
    std::vector<float> input(batch_size * in_features);
    std::vector<float> weight(out_features * in_features);
    std::vector<float> bias(out_features);
    std::vector<float> output_ref(batch_size * out_features);
    std::vector<float> output_xnn(batch_size * out_features);
    
    // Initialize data
    fill_random(input);
    fill_random(weight);
    fill_random(bias);
    
    // Warm up runs
    reference_gemm(batch_size, in_features, out_features, 
                  input.data(), weight.data(), bias.data(), output_ref.data());
    
    linear_xnn_subgraph_raw_impl(
        batch_size, in_features, out_features,
        input.data(), weight.data(), bias.data(),
        nullptr, 0, 0, output_xnn.data(),
        xnn_datatype_fp32, xnn_datatype_fp32, xnn_datatype_fp32,
        true, false, false);
    
    // Verify results match
    bool results_match = compare_results(output_ref, output_xnn);
    if (!results_match) {
        std::cout << "ERROR: Results don't match!" << std::endl;
        return;
    }
    
    // Benchmark reference implementation
    double ref_time = 0;
    for (int i = 0; i < num_runs; i++) {
        ref_time += measure_time_ms([&]() {
            reference_gemm(batch_size, in_features, out_features,
                         input.data(), weight.data(), bias.data(), output_ref.data());
        });
    }
    ref_time /= num_runs;
    
    // Benchmark XNNPACK implementation
    double xnn_time = 0;
    for (int i = 0; i < num_runs; i++) {
        xnn_time += measure_time_ms([&]() {
            linear_xnn_subgraph_raw_impl(
                batch_size, in_features, out_features,
                input.data(), weight.data(), bias.data(),
                nullptr, 0, 0, output_xnn.data(),
                xnn_datatype_fp32, xnn_datatype_fp32, xnn_datatype_fp32,
                true, false, false);
        });
    }
    xnn_time /= num_runs;
    
    // Print results
    std::cout << std::fixed << std::setprecision(2)
              << "Reference implementation: " << ref_time << " ms\n"
              << "XNNPACK implementation:   " << xnn_time << " ms\n"
              << "Speedup:                  " << ref_time / xnn_time << "x\n";
}

int main() {
    // Test various sizes
    std::vector<std::tuple<size_t, size_t, size_t>> test_sizes = {
        {1, 256, 256},      // Small batch
        {32, 512, 512},     // Medium
        {128, 1024, 1024},  // Large
    };
    
    for (const auto& [b, m, n] : test_sizes) {
        run_benchmark(b, m, n);
    }
    
    return 0;
} 