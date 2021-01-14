#pragma once

#include <core/session/onnxruntime_cxx_api.h>

#include <Acts/Definitions/Algebra.hpp>

///
/// 
template<int numInputs, int numOutputs>
class OnnxModel
{    
    template<int D>
    using VectorF = Eigen::Matrix<float, D, 1>;

    std::unique_ptr<Ort::Session> m_session;
    
    std::array<const char*, numInputs> m_inputNodeNames;
    std::array<std::vector<int64_t>, numInputs> m_inputNodeDims;
    
    std::array<const char*, numOutputs> m_outputNodeNames;
    std::array<std::vector<int64_t>, numOutputs> m_outputNodeDims;
    
public:
    /// @param env the ONNX runtime environment
    /// @param modelPath the path to the ML model in *.onnx format
    OnnxModel(Ort::Env env, std::string modelPath)
    {
        ////////////////////////
        // Basic session setup
        ////////////////////////
        Ort::SessionOptions sessionOptions;
        
        sessionOptions.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_BASIC);
        
        m_session = std::make_unique<Ort::Session>(env, modelPath.c_str(), sessionOptions);

        Ort::AllocatorWithDefaultOptions allocator;

        if( m_session->GetInputCount() != numInputs || m_session->GetOutputCount() != numOutputs )
            throw std::invalid_argument("Input model must have exactely 2 inputs and 1 output");

        //////////////////
        // Handle inputs
        //////////////////
        for (size_t i = 0; i < numInputs; ++i) 
        {
            m_inputNodeNames[i] = m_session->GetInputName(i, allocator);

            Ort::TypeInfo inputTypeInfo = m_session->GetInputTypeInfo(i);
            auto tensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
            m_inputNodeDims[i] = tensorInfo.GetShape();
            
            // fix for symbolic dim = -1 from python
            for (size_t j = 0; j < m_inputNodeDims.size(); j++)
                if (m_inputNodeDims[i][j] < 0)
                    m_inputNodeDims[i][j] = 1;
        }
        

        //////////////////
        // Handle outputs
        //////////////////
        for (auto i=0ul; i < numOutputs; ++i)
        {        
            m_outputNodeNames[i] = m_session->GetOutputName(0, allocator);

            Ort::TypeInfo outputTypeInfo = m_session->GetOutputTypeInfo(0);
            auto tensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
            m_outputNodeDims[i] = tensorInfo.GetShape();
            
            // fix for symbolic dim = -1 from python
            for (size_t j = 0; j < m_outputNodeDims.size(); j++)
                if (m_outputNodeDims[i][j] < 0)
                    m_outputNodeDims[i][j] = 1;
        }
    }
    
    OnnxModel(const OnnxModel &) = delete;
    OnnxModel &operator=(const OnnxModel &) = delete;

    /// @brief Run the ONNX inference function
    template<typename InTuple, typename OutTuple>
    void predict(OutTuple &outputVectors, const InTuple &inputVectors) const
    {
        static_assert( std::tuple_size_v<OutTuple> + std::tuple_size_v<InTuple> == numInputs + numOutputs );
        
        // Helper functions
        auto vectorToTensor = [](auto &tensor, const auto &vector, const auto &shape, const auto &memInfo)
        {
            tensor = Ort::Value::CreateTensor<float>(memInfo, vector.data(), vector.size(), shape.data(), shape.size());
        };
        
        auto tensorToVector = [](auto &vector, const auto &tensor, const std::size_t size)
        {
            const float *data = tensor.template GetTensorData<float>();
            
            for(auto i = 0ul; i<size; ++i)
                vector[i] = data[i];
        };
        
        // Init memory Info
        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        // Allocate Tensors
        std::array<Ort::Value, std::tuple_size_v<InTuple>> inputTensors;
        std::array<Ort::Value, std::tuple_size_v<OutTuple>> outputTensors;
        
        // Set input tensors
        std::apply([&, i=0](const auto &... vector) mutable
        {
            ( (vectorToTensor(inputTensors[i], vector, m_inputNodeDims[i++], memInfo)), ... );
        },
        inputVectors);
        
        // Run model
        m_session->Run(Ort::RunOptions{nullptr}, m_inputNodeNames.data(),
                       inputTensors.data(), m_inputNodeNames.size(),
                       m_outputNodeNames.data(),
                       outputTensors.data(), outputTensors.size());
        
        // double-check that outputTensors contains Tensors
        if( !std::all_of(outputTensors.begin(), outputTensors.end(), [](auto &a){ return a.isTensor(); }) )
            throw std::runtime_error("runONNXInference: calculation of output failed. ");

            
        std::apply([&, i=0](const auto &... vector) mutable
        {
            ( (tensorToVector(vector, outputTensors[i], m_outputNodeDims[i++][1])), ... );
        },
        outputVectors);
    }
};
