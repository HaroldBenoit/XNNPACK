#pragma once

#include <cstddef>
#include <xnnpack.h>

void linear_xnn_subgraph_raw_impl(
    size_t batch_size,
    size_t in_features,
    size_t out_features,
    const void* input_data,
    const void* weight_data,
    const void* bias_data,
    const void* weight_scale_data,
    int64_t weight_zero_point,
    int64_t block_size,
    void* output_data,
    xnn_datatype input_datatype,
    xnn_datatype weight_datatype,
    xnn_datatype output_datatype,
    bool use_bias,
    bool channelwise_quantized_weight,
    bool blockwise_quantized_weight); 