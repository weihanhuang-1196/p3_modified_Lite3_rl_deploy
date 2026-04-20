/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "inference_runtime.hpp"
#include <stdexcept>
#include <iostream>
#include <numeric>

#ifdef USE_TORCH
#include <ATen/Parallel.h>
#endif

namespace InferenceRuntime
{

// ============================================================================
// TorchModel Implementation
// ============================================================================

TorchModel::TorchModel()
{
#ifdef USE_TORCH
    // Set threads before model load
    torch::set_num_threads(1);
#endif
}

TorchModel::~TorchModel()
{
}

bool TorchModel::load(const std::string& model_path)
{
    try
    {
#ifdef USE_TORCH
        // Load TorchScript model
        model_ = torch::jit::load(model_path, torch::kCPU);
        model_path_ = model_path;
        loaded_ = true;
        model_.eval(); // Set to evaluation mode
        std::cout << LOGGER::INFO << "Successfully loaded Torch model: " << model_path << std::endl;
        return true;
#else
        std::cout << LOGGER::WARNING << "Torch support not compiled. Please define USE_TORCH." << std::endl;
        loaded_ = false;
        return false;
#endif
    }
    catch (const std::exception& e)
    {
        std::cout << LOGGER::ERROR << "Failed to load Torch model: " << e.what() << std::endl;
        loaded_ = false;
        return false;
    }
}

std::vector<float> TorchModel::forward(const std::vector<std::vector<float>>& inputs)
{
    if (!loaded_)
    {
        throw std::runtime_error("Model not loaded");
    }

#ifdef USE_TORCH
    try
    {

        // Disable gradient computation before each forward pass
        // torch::autograd::GradMode::set_enabled(false);
        torch::InferenceMode guard;

        // Ensure single-threaded execution (critical for performance!)
        torch::set_num_threads(1);

        std::vector<torch::jit::IValue> input_tensors;
        for (const auto& input : inputs)
        {
            auto tensor = torch::tensor(input, torch::kFloat32)
                            .reshape({1, static_cast<int64_t>(input.size())}); // batch=1
            input_tensors.push_back(tensor);
        }
        auto output = model_.forward(input_tensors).toTensor();
        return torch_to_vector(output);

    
        // if(inputs.size() > 1)
        // {
        //     // Convert input vector to Torch tensor (use first input only)
        //     const auto& input = inputs[0];
        //     auto input_tensor = torch::tensor(input, torch::kFloat32).reshape({1,static_cast<int64_t>(input.size())});
        //     const auto& input2 = inputs[1];
        //     auto input2_tensor = torch::tensor(input2, torch::kFloat32).reshape({1,static_cast<int64_t>(input2.size())});

        //     // std::vector<torch::jit::IValue> inputs_obs;
        //     // inputs_obs.push_back(input_tensor);
        //     // inputs_obs.push_back(input2_tensor);

        //             // Execute forward inference
        //     auto output = model_.forward({input_tensor, input2_tensor}).toTensor();
        //             // Convert output tensor to vector
        //     return torch_to_vector(output);
        // }else
        // {
        //     // Convert input vector to Torch tensor (use first input only)
        //     const auto& input = inputs[0];
        //     auto input_tensor = torch::tensor(input, torch::kFloat32).reshape({1, static_cast<int64_t>(input.size())});
        //     auto output = model_.forward({input_tensor}).toTensor();
        //             // Convert output tensor to vector
        //     return torch_to_vector(output);
        // }

    }
    catch (const std::exception& e)
    {
        std::cout << LOGGER::ERROR << "Torch inference error: " << e.what() << std::endl;
        throw;
    }
#else
    throw std::runtime_error("Torch support not compiled");
#endif
}



std::vector<std::vector<float>> TorchModel::forward_world(const std::vector<std::vector<float>>& inputs)
{
    if (!loaded_)
    {
        throw std::runtime_error("Model not loaded");
    }
#ifdef USE_TORCH
        // Disable gradient computation before each forward pass
        // torch::autograd::GradMode::set_enabled(false);
        torch::InferenceMode guard;

        // Ensure single-threaded execution (critical for performance!)
        torch::set_num_threads(1);

        const auto& input_wm_prop = inputs[0];
        auto input_wm_prop_tensor = vector_to_torch(input_wm_prop, {1, static_cast<int64_t>(input_wm_prop.size())});
        const auto& wm_input_image = inputs[1];
        auto wm_input_image_tensor = vector_to_torch(wm_input_image, {1, 64, 64, 1});
        const auto& wm_logit = inputs[2];
        auto wm_logit_tensor =vector_to_torch(wm_logit, {1, 32, 32});
        const auto& wm_stoch = inputs[3];
        auto wm_stoch_tensor = vector_to_torch(wm_stoch, {1, 32, 32});
        const auto& wm_deter = inputs[4];
        auto wm_deter_tensor = vector_to_torch(wm_deter, {1, static_cast<int64_t>(wm_deter.size())});
        const auto& wm_action = inputs[5];
        auto wm_action_tensor = vector_to_torch(wm_action, {1, static_cast<int64_t>(wm_action.size())});
        const auto& wm_is_first = inputs[6];
        auto wm_is_first_tensor = vector_to_torch(wm_is_first, {static_cast<int64_t>(wm_is_first.size())});

        auto output = model_.forward({input_wm_prop_tensor, wm_input_image_tensor, wm_logit_tensor, wm_stoch_tensor, wm_deter_tensor, wm_action_tensor, wm_is_first_tensor}).toTuple()->elements();

        std::vector<std::vector<float>> result;
        result.reserve(output.size());
        for (auto& o : output)
        {
            result.push_back(torch_to_vector(o.toTensor()));
        }
        
        return result;


#else
    throw std::runtime_error("Torch support not compiled");
#endif

}





#ifdef USE_TORCH
torch::Tensor TorchModel::vector_to_torch(const std::vector<float>& data, const std::vector<int64_t>& shape)
{
    // Use torch::tensor() + reshape() to match test program behavior
    // auto tensor = torch::tensor(data, torch::kFloat32).reshape(shape);
    auto tensor = torch::from_blob(const_cast<float*>(data.data()), shape, torch::kFloat32).clone();
    return tensor;
}

std::vector<float> TorchModel::torch_to_vector(const torch::Tensor& tensor)
{
    // Ensure tensor is contiguous and on CPU
    auto cpu_tensor = tensor.is_contiguous() ? tensor : tensor.contiguous();
    if (cpu_tensor.device().type() != torch::kCPU)
    {
        cpu_tensor = cpu_tensor.to(torch::kCPU);
    }

    // Get data pointer and size
    float* data_ptr = cpu_tensor.data_ptr<float>();
    int64_t num_elements = cpu_tensor.numel();

    // Copy data to vector
    return std::vector<float>(data_ptr, data_ptr + num_elements);
}
#endif

// ============================================================================
// ONNXModel Implementation
// ============================================================================

ONNXModel::ONNXModel()
#ifdef USE_ONNX
    : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
#endif
{
#ifdef USE_ONNX
    // Initialize ONNX Runtime environment
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ONNXModel");
#endif
}

ONNXModel::~ONNXModel()
{
#ifdef USE_ONNX
    session_.reset();
    env_.reset();
#endif
}

bool ONNXModel::load(const std::string& model_path)
{
    try
    {
#ifdef USE_ONNX
        // Configure session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(2);
        // session_options.SetInterOpNumThreads(2);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Create inference session
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

        // Setup input/output information
        setup_input_output_info();

        model_path_ = model_path;
        loaded_ = true;
        std::cout << LOGGER::INFO << "Successfully loaded ONNX model: " << model_path << std::endl;
        return true;
#else
        std::cout << LOGGER::WARNING << "ONNX support not compiled. Please define USE_ONNX." << std::endl;
        loaded_ = false;
        return false;
#endif
    }
    catch (const std::exception& e)
    {
        std::cout << LOGGER::ERROR << "Failed to load ONNX model: " << e.what() << std::endl;
        loaded_ = false;
        return false;
    }
}

std::vector<float> ONNXModel::forward(const std::vector<std::vector<float>>& inputs)
{
    if (!loaded_)
    {
        throw std::runtime_error("Model not loaded");
    }

#ifdef USE_ONNX
    try
    {
        // Create memory info
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        if(input_node_names_.size() != inputs.size())
        {
            std::cout << LOGGER::ERROR << "Input size mismatch: expected " << input_node_names_.size() << ", got " << inputs.size() << std::endl;
            throw;
        }

        std::vector<Ort::Value> ort_inputs;
        ort_inputs.reserve(input_node_names_.size());
        
        for (size_t i = 0; i < input_node_names_.size(); i++)
        {
            const auto& input = inputs[i];
            auto input_shape = session_->GetInputTypeInfo(i)
                                   .GetTensorTypeAndShapeInfo()
                                   .GetShape();
            
            // 修正动态维度
            for (auto& dim : input_shape)
                if (dim < 0) dim = 1;
                
            // 检查总元素是否一致，不一致则强制用 input.size() 作为最后一维
            int64_t total = 1;
            for (auto d : input_shape) total *= d;
            if (total != static_cast<int64_t>(input.size())) {
                input_shape = {1, static_cast<int64_t>(input.size())};
            }

            auto input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float*>(input.data()),
                input.size(),
                input_shape.data(),
                input_shape.size());

            ort_inputs.emplace_back(std::move(input_tensor));
            
        }

                // --- 输入/输出名称 ---
        std::vector<const char*> input_names;
        for (auto& n : input_node_names_)
            input_names.push_back(n.c_str());

        std::vector<const char*> output_names;
        for (auto& n : output_node_names_)
            output_names.push_back(n.c_str());


         auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            ort_inputs.data(),
            ort_inputs.size(),
            output_names.data(),
            output_names.size());


        // Extract output data
        return extract_output_data(outputs);
    }
    catch (const std::exception& e)
    {
        std::cout << LOGGER::ERROR << "ONNX inference error: " << e.what() << std::endl;
        throw;
    }
#else
    throw std::runtime_error("ONNX support not compiled");
#endif
}


std::vector<std::vector<float>> ONNXModel::forward_world(const std::vector<std::vector<float>>& inputs)
{
    if (!loaded_)
    {
        throw std::runtime_error("Model not loaded");
    }

#ifdef USE_ONNX
    try
    {

        // Create memory info
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        if(input_node_names_.size() != inputs.size())
        {
            std::cout << LOGGER::ERROR << "Input size mismatch: expected " << input_node_names_.size() << ", got " << inputs.size() << std::endl;
            throw;
        }

        std::vector<Ort::Value> ort_inputs;
        ort_inputs.reserve(input_node_names_.size());
        
        for (size_t i = 0; i < input_node_names_.size(); i++)
        {
            const auto& input = inputs[i];
            auto input_shape = session_->GetInputTypeInfo(i)
                                   .GetTensorTypeAndShapeInfo()
                                   .GetShape();
            
            // 修正动态维度
            for (auto& dim : input_shape)
                if (dim < 0) dim = 1;
                
            // 检查总元素是否一致，不一致则强制用 input.size() 作为最后一维
            int64_t total = 1;
            for (auto d : input_shape) total *= d;
            if (total != static_cast<int64_t>(input.size())) {
                input_shape = {1, static_cast<int64_t>(input.size())};
            }

            auto input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float*>(input.data()),
                input.size(),
                input_shape.data(),
                input_shape.size());

            ort_inputs.emplace_back(std::move(input_tensor));
            
        }

                // --- 输入/输出名称 ---
        std::vector<const char*> input_names;
        for (auto& n : input_node_names_)
            input_names.push_back(n.c_str());

        std::vector<const char*> output_names;
        for (auto& n : output_node_names_)
            output_names.push_back(n.c_str());

        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            ort_inputs.data(),
            ort_inputs.size(),
            output_names.data(),
            output_names.size());



        auto result = extract_output_all_data(outputs);
        

        // Extract output data
        return result;

    }
    catch (const std::exception& e)
    {
        std::cout << LOGGER::ERROR << "ONNX inference error: " << e.what() << std::endl;
        throw;
    }

#else
    throw std::runtime_error("ONNX support not compiled");
#endif
}




#ifdef USE_ONNX
void ONNXModel::setup_input_output_info()
{
    // Get input node information
    size_t num_input_nodes = session_->GetInputCount();
    input_node_names_.reserve(num_input_nodes);
    input_shapes_.reserve(num_input_nodes);

    for (size_t i = 0; i < num_input_nodes; ++i)
    {
        // Get input name
        auto input_name = session_->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        input_node_names_.push_back(std::string(input_name.get()));

        // Get input shape
        Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();

        std::vector<int64_t> shape;
        for (auto dim : input_dims)
        {
            // Handle dynamic dimensions
            if (dim == -1)
            {
                shape.push_back(1);
            }
            else
            {
                shape.push_back(dim);
            }
        }
        input_shapes_.push_back(shape);
    }

    // Get output node information
    size_t num_output_nodes = session_->GetOutputCount();
    output_node_names_.reserve(num_output_nodes);
    output_shapes_.reserve(num_output_nodes);

    for (size_t i = 0; i < num_output_nodes; ++i)
    {
        // Get output name
        auto output_name = session_->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        output_node_names_.push_back(std::string(output_name.get()));

        // Get output shape
        Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();

        std::vector<int64_t> shape;
        for (auto dim : output_dims)
        {
            // Handle dynamic dimensions
            if (dim == -1)
            {
                shape.push_back(1);
            }
            else
            {
                shape.push_back(dim);
            }
        }
        output_shapes_.push_back(shape);
    }
}


std::vector<float> ONNXModel::forward_motor_policy(const std::vector<float>& inputs)
{
    int NUM_MOTORS = 12;
    std::vector<int64_t> input_shape = {NUM_MOTORS, 6};

    const char* input_name  = this->input_node_names_[0].c_str();
    const char* output_name = this->output_node_names_[0].c_str();
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(inputs.data()),
            inputs.size(),
            input_shape.data(),
            input_shape.size()
        );

    // 2. 推理
    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        &input_name,
        &input_tensor,
        1,
        &output_name,
        1
    );


    // 3. 读取输出
    float* tau_est = outputs[0].GetTensorMutableData<float>();

    // for (int i = 0; i < NUM_MOTORS; ++i) {
    //     std::cout << "motor " << i
    //               << " tau_est = "
    //               << tau_est[i] << std::endl;
    // }

    return std::vector<float>(tau_est, tau_est + NUM_MOTORS);


}


std::vector<std::vector<float>> ONNXModel::extract_output_all_data(const std::vector<Ort::Value>& outputs)
{
    if (outputs.empty())
    {
        throw std::runtime_error("No outputs from ONNX model");
    }
    std::vector<std::vector<float>> results;

    for (const auto& output : outputs)
    {
        if (!output.IsTensor())
            continue;

        auto info = output.GetTensorTypeAndShapeInfo();

        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            continue;

        const float* data = output.GetTensorData<float>();
        size_t count = info.GetElementCount();

        results.emplace_back(data, data + count);
    }

    return results;
}


std::vector<float> ONNXModel::extract_output_data(const std::vector<Ort::Value>& outputs)
{
    if (outputs.empty())
    {
        throw std::runtime_error("No outputs from ONNX model");
    }

    // Get first output tensor
    auto& output = outputs[0];
    float* output_data = const_cast<float*>(output.GetTensorData<float>());

    // Calculate total number of output elements
    auto output_shape = output.GetTensorTypeAndShapeInfo().GetShape();

    int64_t num_elements = 1;
    for (auto dim : output_shape)
    {
        if (dim > 0)
        {
            num_elements *= dim;
        }
    }

    // Copy output data to vector
    std::vector<float> result(output_data, output_data + num_elements);

    return result;
}
#endif

// ============================================================================
// ModelFactory Implementation
// ============================================================================

std::unique_ptr<Model> ModelFactory::create_model(ModelType type)
{
    switch (type)
    {
        case ModelType::TORCH:
            return std::make_unique<TorchModel>();
        case ModelType::ONNX:
            return std::make_unique<ONNXModel>();
        default:
            return nullptr;
    }
}

ModelFactory::ModelType ModelFactory::detect_model_type(const std::string& model_path)
{
    // Extract file extension from path
    std::filesystem::path path(model_path);
    std::string extension = path.extension().string();

    // Convert to lowercase for case-insensitive comparison
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    // Determine model type based on extension
    if (extension == ".pt" || extension == ".pth")
    {
        return ModelType::TORCH;
    }
    else if (extension == ".onnx")
    {
        return ModelType::ONNX;
    }
    else
    {
        throw std::runtime_error("Unknown model file extension: " + extension + ". Supported: .pt, .pth, .onnx");
    }
}

std::unique_ptr<Model> ModelFactory::load_model(const std::string& model_path, ModelType type)
{
    // If type is AUTO, automatically detect model type
    if (type == ModelType::AUTO)
    {
        type = detect_model_type(model_path);
    }

    // Create and load model
    auto model = create_model(type);
    if (model && model->load(model_path))
    {
        return model;
    }
    return nullptr;
}

} // namespace InferenceRuntime
