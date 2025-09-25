#include "include/TrieMethodSelector.h"
#include <iostream>
#include <stdexcept> // 用于抛出运行时错误

// 构造函数
TrieMethodSelector::TrieMethodSelector(const std::string& model_path)
    : _env(ORT_LOGGING_LEVEL_WARNING, "TrieMethodSelector"),
      _session(nullptr) {
    
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::cout << "  - [Selector] Initializing ONNX Runtime session..." << std::endl;
    try {
        _session = Ort::Session(_env, model_path.c_str(), session_options);
    } catch (const Ort::Exception& e) {
        throw std::runtime_error("Failed to load ONNX model: " + std::string(e.what()));
    }
    std::cout << "  - [Selector] Session initialized successfully." << std::endl;

    // --- 正确地获取并存储模型的输入/输出信息 ---

    // 1. 获取输入节点的名称
    //    使用一个临时的 unique_ptr 来接收 ONNX Runtime 分配的内存
    Ort::AllocatedStringPtr input_name_ptr = _session.GetInputNameAllocated(0, _allocator);
    //    将 C 风格字符串复制到自己的 std::string 成员变量中进行管理
    _input_name_str = input_name_ptr.get();
    //    将 std::string 的 C 风格字符串指针 (.c_str()) 存入向量，供 Run() 函数使用
    _input_node_names.push_back(_input_name_str.c_str());

    // 2. 获取输出节点的名称 (同理)
    Ort::AllocatedStringPtr output_name_ptr = _session.GetOutputNameAllocated(0, _allocator);
    _output_name_str = output_name_ptr.get();
    _output_node_names.push_back(_output_name_str.c_str());

    // 3. 获取输入的维度信息
    Ort::TypeInfo type_info = _session.GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    _input_node_dims = tensor_info.GetShape();
    
    if (_input_node_dims[0] < 0) {
       _input_node_dims[0] = 1; 
    }
}

// 预测函数
bool TrieMethodSelector::predict(const std::vector<float>& features) {
    // 验证传入的特征数量是否与模型期望的一致
    if (features.size() != _input_node_dims[1]) {
        throw std::runtime_error("Feature size mismatch. Model expects " + 
                                 std::to_string(_input_node_dims[1]) + 
                                 " features, but got " + std::to_string(features.size()));
    }

    // 1. 创建内存信息，指定数据在CPU上
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // 2. 将输入的 std::vector<float> 包装成 ONNX Runtime 需要的张量 (Tensor)
    //    注意：这里使用 const_cast 是安全的，因为 ONNX Runtime 不会修改输入数据
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, 
                                                              const_cast<float*>(features.data()), 
                                                              features.size(), 
                                                              _input_node_dims.data(), 
                                                              _input_node_dims.size());
    
    // 3. 运行推理
    //    将输入张量传递给会话，获取输出张量
    auto output_tensors = _session.Run(Ort::RunOptions{nullptr}, 
                                       _input_node_names.data(), 
                                       &input_tensor, 
                                       1, // 输入张量的数量
                                       _output_node_names.data(), 
                                       1); // 输出张量的数量

    // 4. 从输出张量中提取结果
    //    Scikit-learn 分类器的输出通常是 int64_t 类型的类别标签
    int64_t* pred_ptr = output_tensors[0].GetTensorMutableData<int64_t>();
    int prediction = static_cast<int>(pred_ptr[0]); // 获取第一个（也是唯一一个）预测结果

    // 根据 Python 脚本的逻辑，1 代表 'choose_method_T' (应选方法T - 递归法)
    return (prediction == 1);
}