# XNNPACK GEMM Benchmark

This project benchmarks XNNPACK's GEMM implementation against a naive reference implementation.

## Prerequisites

- CMake 3.15 or higher
- C++17 compiler
- XNNPACK library and headers
- pthreadpool library

## Building

```bash
# From XNNPACK root directory
mkdir -p build
cd build
cmake ..
make -j gemm_test

```

Or use the one-liner:
```bash
mkdir -p build && cd build && cmake .. && make -j gemm_test
```

## Running

After building, run the benchmark:

```bash
./gemm_benchmark/gemm_test
```

The benchmark will test various matrix sizes and compare:
- Naive single-threaded reference implementation
- XNNPACK optimized GEMM implementation

For each size, it will:
1. Verify that both implementations produce matching results (within tolerance of 1e-6)
2. Measure execution time over multiple runs (default: 10 runs)
3. Report average execution time and speedup

## Test Sizes

The benchmark includes the following matrix sizes:
- Small: 1x256x256 (batch x input_features x output_features)
- Medium: 32x512x512
- Large: 128x1024x1024

Each test reports:
- Reference implementation time
- XNNPACK implementation time
- Speedup factor
