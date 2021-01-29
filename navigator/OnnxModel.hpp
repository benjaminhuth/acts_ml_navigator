#pragma once

#include <core/session/onnxruntime_cxx_api.h>

#include <Acts/Definitions/Algebra.hpp>

///
/// 
template<int NumInputs, int NumOutputs>
class OnnxModel
{    
    template<int D>
    using VectorF = Eigen::Matrix<float, D, 1>;

    std::unique_ptr<Ort::Session> m_session;
    
    std::array<const char*, NumInputs> m_inputNodeNames;
    std::array<std::vector<int64_t>, NumInputs> m_inputNodeDims;
    
    std::array<const char*, NumOutputs> m_outputNodeNames;
    std::array<std::vector<int64_t>, NumOutputs> m_outputNodeDims;
    
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

        if( m_session->GetInputCount() != NumInputs || m_session->GetOutputCount() != NumOutputs )
            throw std::invalid_argument("Input or Output dimension mismatch");

        //////////////////
        // Handle inputs
        //////////////////
        for (size_t i = 0; i < NumInputs; ++i) 
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
        for (auto i=0ul; i < NumOutputs; ++i)
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
    void predict(OutTuple &outputVectors, InTuple &inputVectors) const
    {
        static_assert( std::tuple_size_v<OutTuple> == NumOutputs );
        static_assert( std::tuple_size_v<InTuple> == NumInputs );
        
        std::apply([]<typename... Ts>(Ts...){ static_assert( std::conjunction_v<std::is_same<float, typename Ts::Scalar>...> ); }, outputVectors);
        std::apply([]<typename... Ts>(Ts...){ static_assert( std::conjunction_v<std::is_same<float, typename Ts::Scalar>...> ); }, inputVectors);
        
        // Init memory Info
        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        // Helper function
        auto make_tensor = [&](auto &vector, auto &shape)
        {                
            return Ort::Value::CreateTensor<float>(memInfo, vector.data(), static_cast<std::size_t>(vector.size()),
                                                   shape.data(), shape.size());
        };
        
        auto fill_input_tensors = [&]<std::size_t... I>(std::index_sequence<I...>)
        {
            return std::array<Ort::Value, NumInputs>
            {
                make_tensor(std::get<I>(inputVectors), std::get<I>(m_inputNodeDims))...
            };
        };
        

        auto fill_output_tensors = [&]<std::size_t... I>(std::index_sequence<I...>)
        {
            return std::array<Ort::Value, NumOutputs>
            {
                make_tensor(std::get<I>(outputVectors), std::get<I>(m_outputNodeDims))...
            };
        };
        
        // Create Tensors        
        auto inputTensors = fill_input_tensors(std::make_index_sequence<NumInputs>());
        auto outputTensors = fill_output_tensors(std::make_index_sequence<NumOutputs>());
        
        // Run model
        m_session->Run(Ort::RunOptions{nullptr}, m_inputNodeNames.data(),
                       inputTensors.data(), m_inputNodeNames.size(),
                       m_outputNodeNames.data(),
                       outputTensors.data(), outputTensors.size());
        
        // double-check that outputTensors contains Tensors
        if( !std::all_of(outputTensors.begin(), outputTensors.end(), [](auto &a){ return a.IsTensor(); }) )
            throw std::runtime_error("runONNXInference: calculation of output failed. ");
    }
};
