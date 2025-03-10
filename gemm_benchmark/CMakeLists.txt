cmake_minimum_required(VERSION 3.15)
project(xnnpack_gemm_benchmark CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add XNNPACK directory to include path
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

# Add source files
add_executable(gemm_test 
    subgraph_gemm.cpp
    test_gemm.cpp
)

# Link libraries
target_link_libraries(gemm_test
    PRIVATE
    XNNPACK
)

# Enable optimizations
target_compile_options(gemm_test PRIVATE -O3 -march=native) 