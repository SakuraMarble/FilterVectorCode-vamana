// 文件名: TrieMethodSelector.h

#pragma once // 防止头文件被重复包含

#include <string>
#include <vector>
#include <memory> // for std::unique_ptr

// 包含 ONNX Runtime C++ API 的核心头文件
#include "../third_party/onnxruntime-linux-x64-1.16.3/include/onnxruntime_cxx_api.h"

class TrieMethodSelector {
public:
    // 构造函数：加载 ONNX 模型并初始化推理会话
    // 参数: model_path - 我们之前导出的 "trie_method_selector.onnx" 文件的路径
    TrieMethodSelector(const std::string& model_path);

    // 预测函数：根据输入的特征，返回 true (代表应使用新方法) 或 false (使用旧方法)
    // 参数: features - 一个包含所有输入特征的浮点数向量
    bool predict(const std::vector<float>& features);

private:
    // ONNX Runtime 的核心对象
    Ort::Env _env;
    Ort::Session _session;
    Ort::AllocatorWithDefaultOptions _allocator;
    
    // 存储模型元信息，避免每次预测都查询，提升性能
    std::vector<const char*> _input_node_names;
    std::vector<const char*> _output_node_names;
    std::vector<int64_t> _input_node_dims;

    std::string _input_name_str;
    std::string _output_name_str;
};