#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <array>
#include <xnnpack.h>
#include <pthreadpool.h>  // For pthreadpool_t
#include <cfloat>  // For FLT_MAX
#include <limits>
#include <sstream>  // For stringstream     
#include <mutex>
#include <iostream>

// Helper struct to manage XNNPACK operator lifecycle
struct XNNOperator {
    xnn_operator_t op = nullptr;
    
    ~XNNOperator() {
        if (op) xnn_delete_operator(op);
    }
};

// Helper struct to manage XNNPACK subgraph lifecycle
struct XNNSubgraph {
    xnn_subgraph_t subgraph = nullptr;
    
    ~XNNSubgraph() {
        if (subgraph) xnn_delete_subgraph(subgraph);
    }
};

// Helper struct to manage XNNPACK runtime lifecycle
struct XNNRuntime {
    xnn_runtime_t runtime = nullptr;
    
    ~XNNRuntime() {
        if (runtime) xnn_delete_runtime(runtime);
    }
};

// Helper struct to manage threadpool lifecycle
struct ThreadpoolGuard {
    pthreadpool_t threadpool = nullptr;
    
    ThreadpoolGuard() {
        threadpool = pthreadpool_create(0);
    }
    
    ~ThreadpoolGuard() {
        if (threadpool) {
            pthreadpool_destroy(threadpool);
        }
    }
    
    pthreadpool_t get() const {
        return threadpool;
    }
};

// Helper function to convert xnn_status to string for error reporting
std::string xnn_status_to_string(xnn_status status) {
    switch (status) {
        case xnn_status_success: return "success";
        case xnn_status_uninitialized: return "uninitialized";
        case xnn_status_invalid_parameter: return "invalid_parameter";
        case xnn_status_invalid_state: return "invalid_state";
        case xnn_status_unsupported_parameter: return "unsupported_parameter";
        case xnn_status_unsupported_hardware: return "unsupported_hardware";
        case xnn_status_out_of_memory: return "out_of_memory";
        default: return "unknown error (" + std::to_string(static_cast<int>(status)) + ")";
    }
}

// Helper function to get threadpool
pthreadpool_t get_threadpool() {
    // Use a thread-local threadpool to avoid creating/destroying for each call
    thread_local ThreadpoolGuard guard;
    return guard.get();
}

// Helper function to ensure XNNPACK is initialized
bool ensure_xnnpack_initialized() {
    static bool initialized = false;
    static std::mutex init_mutex;
    
    if (!initialized) {
        std::lock_guard<std::mutex> lock(init_mutex);
        if (!initialized) {
            xnn_status status = xnn_initialize(/*allocator=*/nullptr);
            initialized = (status == xnn_status_success);
            return initialized;
        }
    }
    return initialized;
}

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
    bool blockwise_quantized_weight) {
    
    // Ensure XNNPACK is initialized
    if (!ensure_xnnpack_initialized()) {
        throw std::runtime_error("Failed to initialize XNNPACK");
    }
    
    
    // Create subgraph
    XNNSubgraph xnn_subgraph;
    xnn_status status = xnn_create_subgraph(
        /*external_value_ids=*/2,
        /*flags=*/0,
        &xnn_subgraph.subgraph);
    if (status != xnn_status_success) {
        std::stringstream ss;
        ss << "Failed to create XNNPACK subgraph: " << xnn_status_to_string(status);
        throw std::runtime_error(ss.str());
    }
    
    // Define input tensor
    uint32_t input_id;
    {
        std::vector<size_t> input_dims = {batch_size, in_features};
        status = xnn_define_tensor_value(
            xnn_subgraph.subgraph,
            input_datatype,
            /*num_dims=*/input_dims.size(),
            /*dims=*/input_dims.data(),
            /*data=*/nullptr,  // Data will be provided during setup
            /*external_id=*/0,  // Input external ID
            /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,  // Mark as external input
            &input_id);
        if (status != xnn_status_success) {
            std::stringstream ss;
            ss << "Failed to define input tensor: " << xnn_status_to_string(status);
            throw std::runtime_error(ss.str());
        }
    }
    
    // Define weight tensor
    uint32_t weight_id;
    {
        std::vector<size_t> weight_dims = {out_features, in_features};

        if (!channelwise_quantized_weight && !blockwise_quantized_weight) {
        status = xnn_define_tensor_value(
            xnn_subgraph.subgraph,
            weight_datatype,
            /*num_dims=*/weight_dims.size(),
            /*dims=*/weight_dims.data(),
            /*data=*/weight_data,  // Provide weight data directly
            /*external_id=*/XNN_INVALID_VALUE_ID,  // Weight is not external
            /*flags=*/0,  // No special flags needed
            &weight_id);
        } else if (channelwise_quantized_weight) {
            status = xnn_define_channelwise_quantized_tensor_value_v2(
                xnn_subgraph.subgraph,
                weight_datatype,
                /*zero_point=*/weight_zero_point,
                /*scale=*/static_cast<const float*>(weight_scale_data),
                /*num_dims=*/weight_dims.size(),
                /*channel_dim=*/0, // Channel dimension is the first dimension (out_features)
                /*dims=*/weight_dims.data(),
                /*data=*/weight_data,
                /*external_id=*/XNN_INVALID_VALUE_ID,
                /*flags=*/0,
                &weight_id);
        } else if (blockwise_quantized_weight) {
            status = xnn_define_blockwise_quantized_tensor_value(
                xnn_subgraph.subgraph,
                weight_datatype,
                /*zero_point=*/weight_zero_point,
                /*scale=*/static_cast<const uint16_t*>(weight_scale_data),
                /*num_dims=*/weight_dims.size(),
                /*channel_dim=*/0, // TODO: Check if this is correct
                /*block_size=*/block_size, // TODO: Check if this is correct
                /*dims=*/weight_dims.data(),
                /*data=*/weight_data,
                /*external_id=*/XNN_INVALID_VALUE_ID,
                /*flags=*/0,
                &weight_id);
        }


        if (status != xnn_status_success) {
            std::stringstream ss;
            ss << "Failed to define weight tensor: " << xnn_status_to_string(status)
               << " (channelwise_quantized=" << channelwise_quantized_weight 
               << ", blockwise_quantized=" << blockwise_quantized_weight << ")";
            ss << "Failed to define weight tensor: " << xnn_status_to_string(status);
                throw std::runtime_error(ss.str());
            }

    }
    
    // Define bias tensor
    uint32_t bias_id = XNN_INVALID_VALUE_ID;
    if (use_bias) {
        std::vector<size_t> bias_dims = {out_features};
        status = xnn_define_tensor_value(
            xnn_subgraph.subgraph,
            input_datatype == xnn_datatype_fp16 ? xnn_datatype_fp16 : xnn_datatype_fp32,
            /*num_dims=*/bias_dims.size(),
            /*dims=*/bias_dims.data(),
            /*data=*/bias_data,
            /*external_id=*/XNN_INVALID_VALUE_ID,
            /*flags=*/0,
            &bias_id);
        if (status != xnn_status_success) {
            std::stringstream ss;
            ss << "Failed to define bias tensor: " << xnn_status_to_string(status);
            throw std::runtime_error(ss.str());
        }
    }
    
    // Define output tensor
    uint32_t output_id;
    {
        std::vector<size_t> output_dims = {batch_size, out_features};
        
        status = xnn_define_tensor_value(
            xnn_subgraph.subgraph,
            output_datatype,
            /*num_dims=*/output_dims.size(),
            /*dims=*/output_dims.data(),
            /*data=*/nullptr,  // Data will be provided during setup
            /*external_id=*/1,  // Output external ID
            /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT,  // Mark as external output
            &output_id);
        if (status != xnn_status_success) {
            std::stringstream ss;
            ss << "Failed to define output tensor: " << xnn_status_to_string(status);
            throw std::runtime_error(ss.str());
        }
    }
    
    // Define fully connected operation
    status = xnn_define_fully_connected(
        xnn_subgraph.subgraph,
        /*output_min=*/-std::numeric_limits<float>::infinity(),  // No clamping
        /*output_max=*/std::numeric_limits<float>::infinity(),   // No clamping
        /*input_id=*/input_id,
        /*filter_id=*/weight_id,
        /*bias_id=*/bias_id,
        /*output_id=*/output_id,
        /*flags=*/0);  // No special flags needed
    if (status != xnn_status_success) {
        std::stringstream ss;
        ss << "Failed to define fully connected operation: " << xnn_status_to_string(status);
        throw std::runtime_error(ss.str());
    }
    
    // Create threadpool
    pthreadpool_t threadpool = get_threadpool();

    if (threadpool == nullptr) {
        throw std::runtime_error("Threadpool is null pointer");
    }

    // Create runtime
    XNNRuntime xnn_runtime;
    status = xnn_create_runtime_v3(
        xnn_subgraph.subgraph,
        /*mempool=*/nullptr,
        /*threadpool=*/threadpool,
        /*flags=*/0,
        &xnn_runtime.runtime);
    if (status != xnn_status_success) {
        std::stringstream ss;
        ss << "Failed to create XNNPACK runtime: " << xnn_status_to_string(status);
        throw std::runtime_error(ss.str());
    }
    
    // Setup external tensors
    std::vector<xnn_external_value> external_values(2);
    
    // Input tensor (external ID 0)
    external_values[0] = {
        /*id=*/0,
        /*data=*/const_cast<void*>(input_data)
    };

    // Output tensor (external ID 1)
    external_values[1] = {
        /*id=*/1,
        /*data=*/output_data
    };
    
    // Setup runtime with external values
    status = xnn_setup_runtime(
        xnn_runtime.runtime,
        /*num_external_values=*/external_values.size(),
        /*external_values=*/external_values.data());
    if (status != xnn_status_success) {
        std::stringstream ss;
        ss << "Failed to setup XNNPACK runtime: " << xnn_status_to_string(status);
        throw std::runtime_error(ss.str());
    }
    
    // Execute the computation
    status = xnn_invoke_runtime(xnn_runtime.runtime);
    if (status != xnn_status_success) {
        std::stringstream ss;
        ss << "Failed to invoke XNNPACK runtime: " << xnn_status_to_string(status);
        throw std::runtime_error(ss.str());
    }
}