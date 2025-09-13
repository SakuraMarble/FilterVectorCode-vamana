#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <queue>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <set>
#include <memory>
#include <algorithm>
#include <map>
#include <cstdint>
#include <stdexcept>
#include <atomic>
#include <boost/program_options.hpp>
#include <omp.h>


// ANNS Namespace contains the core Trie data structure for indexing label sets.
namespace ANNS {
    using IdxType = uint32_t;
    using LabelType = uint32_t;

    struct TrieNode {
        std::vector<std::pair<LabelType, std::unique_ptr<TrieNode>>> children;
        IdxType group_id = 0;
        IdxType group_size = 0;
        std::vector<LabelType> label_set;
    };

    class TrieIndex {
    public:
        TrieIndex() : root(std::make_unique<TrieNode>()), max_group_id(0), max_label_id(0) {}

        IdxType insert(const std::vector<LabelType>& labels, IdxType& new_group_id_counter) {
            auto node = root.get();
            std::vector<LabelType> sorted_labels = labels;
            std::sort(sorted_labels.begin(), sorted_labels.end());

            for (auto label : sorted_labels) {
                auto it = std::lower_bound(node->children.begin(), node->children.end(), label,
                    [](const auto& child, LabelType val) {
                        return child.first < val;
                    });
                
                if (it == node->children.end() || it->first != label) {
                    auto new_node = std::make_unique<TrieNode>();
                    it = node->children.insert(it, {label, std::move(new_node)});
                }
                node = it->second.get();
            }

            if (node->group_id == 0) {
                node->group_id = new_group_id_counter++;
                node->label_set = sorted_labels;
                max_group_id = std::max(max_group_id, node->group_id);
            }
            node->group_size++;
            if (!sorted_labels.empty()) {
                max_label_id = std::max(max_label_id, sorted_labels.back());
            }
            return node->group_id;
        }
        
        IdxType calculate_coverage(const std::vector<LabelType>& query_labels) const {
            if (query_labels.empty()) return 0;

            std::vector<LabelType> sorted_labels = query_labels;
            std::sort(sorted_labels.begin(), sorted_labels.end());

            IdxType total_coverage = 0;
            std::set<IdxType> visited_groups;
            
            find_and_sum_supersets(root.get(), sorted_labels, 0, total_coverage, visited_groups);
            
            return total_coverage;
        }

        LabelType get_max_label_id() const { return max_label_id; }
        IdxType get_max_group_id() const { return max_group_id; }

    private:
        void find_and_sum_supersets(
            const TrieNode* node,
            const std::vector<LabelType>& sorted_query_labels,
            size_t query_idx,
            IdxType& total_coverage,
            std::set<IdxType>& visited_groups) const
        {
            if (query_idx == sorted_query_labels.size()) {
                collect_all_children_coverage(node, total_coverage, visited_groups);
                return;
            }

            LabelType current_query_label = sorted_query_labels[query_idx];
            
            auto it = std::lower_bound(node->children.begin(), node->children.end(), current_query_label,
                [](const auto& child, LabelType val) {
                    return child.first < val;
                });

            while (it != node->children.end()) {
                const auto& child_node = it->second;
                LabelType child_label = it->first;

                if (child_label == current_query_label) {
                    find_and_sum_supersets(child_node.get(), sorted_query_labels, query_idx + 1, total_coverage, visited_groups);
                }
                else {
                    find_and_sum_supersets(child_node.get(), sorted_query_labels, query_idx, total_coverage, visited_groups);
                }
                it++;
            }
        }
        
        void collect_all_children_coverage(const TrieNode* node, IdxType& total_coverage, std::set<IdxType>& visited_groups) const {
            std::queue<const TrieNode*> q;
            q.push(node);

            while (!q.empty()) {
                const TrieNode* cur = q.front();
                q.pop();
                
                if (cur->group_id != 0 && visited_groups.find(cur->group_id) == visited_groups.end()) {
                    total_coverage += cur->group_size;
                    visited_groups.insert(cur->group_id);
                }

                for (const auto& child_pair : cur->children) {
                    q.push(child_pair.second.get());
                }
            }
        }

    private:
        std::unique_ptr<TrieNode> root;
        LabelType max_label_id;
        IdxType max_group_id;
    };
}


namespace po = boost::program_options;
using namespace ANNS;


// 用于加载 fvecs 格式文件的帮助函数
void load_fvecs(const std::string& filename, std::vector<std::vector<float>>& vectors, int& dimension) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        throw std::runtime_error("无法打开文件: " + filename);
    }
    input.read(reinterpret_cast<char*>(&dimension), sizeof(int));
    if (input.gcount() != sizeof(int)) {
        dimension = 0;
        return; 
    }
    input.seekg(0);
    int d;
    while (input.read(reinterpret_cast<char*>(&d), sizeof(int))) {
        if (d != dimension) {
            throw std::runtime_error("fvecs 文件中的向量维度不一致。");
        }
        std::vector<float> vec(d);
        input.read(reinterpret_cast<char*>(vec.data()), d * sizeof(float));
        if (input.gcount() != d * sizeof(float)) {
             throw std::runtime_error("从 fvecs 文件读取向量数据时出错。");
        }
        vectors.push_back(vec);
    }
}

// 用于写入 fvecs 格式文件的帮助函数
void write_fvecs(const std::string& filename, const std::vector<std::vector<float>>& vectors) {
    if (vectors.empty()) return;
    std::ofstream output(filename, std::ios::binary);
    if (!output) {
        throw std::runtime_error("无法打开文件进行写入: " + filename);
    }
    int dimension = vectors[0].size();
    for (const auto& vec : vectors) {
        if (vec.size() != static_cast<size_t>(dimension)) {
             throw std::runtime_error("写入 fvecs 文件时向量维度不一致。");
        }
        output.write(reinterpret_cast<const char*>(&dimension), sizeof(int));
        output.write(reinterpret_cast<const char*>(vec.data()), dimension * sizeof(float));
    }
}


// 'generate' 模式: 并行生成候选查询池
void run_generate_mode(const po::variables_map& vm) {
   // --- 获取命令行选项 ---
   std::string input_file = vm["input_file"].as<std::string>();
   std::string output_file = vm["output_file"].as<std::string>();
   IdxType num_points = vm["num_points"].as<IdxType>();
   IdxType K = vm["K"].as<IdxType>();
   std::string distribution_type = vm["distribution_type"].as<std::string>();
   bool truncate_to_fixed_length = vm["truncate_to_fixed_length"].as<bool>();
   IdxType num_labels_per_query = vm["num_labels_per_query"].as<IdxType>();
   IdxType expected_num_label = vm["expected_num_label"].as<IdxType>();

   // --- 获取向量文件相关的选项 ---
   bool generate_vectors = vm.count("base_vectors_file") && vm.count("output_vectors_file");
   std::vector<std::vector<float>> base_vectors;
   
   if (generate_vectors) {
       std::string base_vectors_file = vm["base_vectors_file"].as<std::string>();
       std::cout << "Loading base vectors from " << base_vectors_file << "..." << std::endl;
       try {
           int dim;
           load_fvecs(base_vectors_file, base_vectors, dim);
           std::cout << "Loaded " << base_vectors.size() << " base vectors with dimension " << dim << "." << std::endl;
       } catch (const std::exception& e) {
           std::cerr << "Error loading base vectors: " << e.what() << std::endl;
           return;
       }
   }

   // --- 从基础文件构建Trie索引 ---
   TrieIndex trie_index;
   std::ifstream infile(input_file);
   IdxType new_label_set_id = 1;
   std::string line, label_str;

   std::vector<std::vector<LabelType>> base_label_sets;
   while (std::getline(infile, line)) {
       std::vector<LabelType> label_set;
       std::stringstream ss(line);
       while (std::getline(ss, label_str, ',')) if (!label_str.empty()) label_set.emplace_back(std::stoul(label_str));
       if(!label_set.empty()) {
           trie_index.insert(label_set, new_label_set_id);
           base_label_sets.push_back(label_set);
       }
   }
   
   // --- 检查基础标签集和基础向量数量是否匹配 ---
   if (generate_vectors && !base_vectors.empty() && base_vectors.size() != base_label_sets.size()) {
       std::cerr << "Warning: The number of base label sets (" << base_label_sets.size() 
                 << ") does not match the number of base vectors (" << base_vectors.size() 
                 << "). Vector sampling might be inaccurate." << std::endl;
   }

   std::vector<std::vector<LabelType>> generated_label_sets;
   std::vector<std::vector<float>> generated_vectors;

   LabelType num_labels = trie_index.get_max_label_id();

   std::cout << "======================================" << std::endl;
   std::cout << "Generating " << num_points << " candidate queries in parallel..." << std::endl;
   if (generate_vectors) {
       std::cout << "Corresponding query vectors will also be generated." << std::endl;
   }

   // [并行] 使用OpenMP并行生成查询
   #pragma omp parallel for schedule(dynamic)
   for (IdxType i = 0; i < num_points; ++i) {
       std::mt19937 gen(std::random_device{}() + omp_get_thread_num());
       std::vector<LabelType> label_set;
       std::vector<float> corresponding_vector;
       bool found_valid = false;

       for (uint8_t j = 0; j < 100; ++j) {
           label_set.clear();
           
           // --- 查询生成的条件逻辑 ---
           if (truncate_to_fixed_length) {
               // --- A) 生成固定长度的查询 ---
               if (distribution_type == "zipf") {
                   std::vector<LabelType> temp_set;
                   for (LabelType label = 1; label <= num_labels; ++label) {
                       double probability = 0.7 / static_cast<double>(label);
                       std::bernoulli_distribution dist(probability);
                       if (dist(gen)) {
                           temp_set.push_back(label);
                       }
                   }
                   if (temp_set.size() >= num_labels_per_query) {
                       // std::shuffle(temp_set.begin(), temp_set.end(), gen); // --- 移除此处的随机打乱步骤 ---
                       temp_set.resize(num_labels_per_query);
                       label_set = temp_set;
                   }
               } else { // multi_normial
                   if (base_label_sets.empty()) break; // 如果基础数据集为空，无法生成

                   // 1. 从缓存的原始标签集中随机选择一个“源集合”
                   std::uniform_int_distribution<> dis(0, base_label_sets.size() - 1);
                   size_t source_idx = dis(gen);
                   const auto& source_set = base_label_sets[source_idx];

                   // 2. 如果源集合本身不够长，则此次尝试失败，继续下一次重试
                   if (source_set.size() < num_labels_per_query) {
                       continue;
                   }

                   // 3. 如果源集合够长，从中随机选出所需数量的标签
                   std::vector<LabelType> temp_set = source_set;
                   std::shuffle(temp_set.begin(), temp_set.end(), gen);
                   label_set.assign(temp_set.begin(), temp_set.begin() + num_labels_per_query);
                   
                   // --- 从与源标签集对应的索引中获取向量 ---
                   if(generate_vectors && !base_vectors.empty()) {
                       corresponding_vector = base_vectors[source_idx];
                   }
               }
           } else {
               // --- B) 生成可变长度的查询---
               if (distribution_type == "zipf") {
                   for (LabelType label = 1; label <= num_labels; ++label) {
                       double probability = 0.7 / static_cast<double>(label);
                       std::bernoulli_distribution dist(probability);
                       if (dist(gen)) {
                           label_set.push_back(label);
                       }
                   }
                   std::shuffle(label_set.begin(), label_set.end(), gen);
                   if(label_set.size() > expected_num_label){
                       label_set.resize(expected_num_label);
                   }
               } else { // multi_normial
                   for (LabelType label = 1; label <= num_labels; ++label) {
                       if (static_cast<float>(gen()) / gen.max() < static_cast<float>(expected_num_label) / num_labels) {
                           label_set.push_back(label);
                       }
                   }
               }
           }

           // --- 如果其他生成方式没有指定源，则随机挑选一个向量 ---
           if (generate_vectors && corresponding_vector.empty() && !base_vectors.empty()) {
               std::uniform_int_distribution<> dis(0, base_vectors.size() - 1);
               corresponding_vector = base_vectors[dis(gen)];
           }

           if (label_set.empty()) continue; // 如果生成失败，重试
           
           // 检查生成的查询是否覆盖至少K个结果
           if (trie_index.calculate_coverage(label_set) >= K) {
               found_valid = true;
               break; // 成功，退出重试循环
           }
       }
       
       // 备用机制: 如果100次尝试都失败了，从已成功的查询中随机挑选一个
       if (!found_valid) {
           #pragma omp critical
           {
               if (!generated_label_sets.empty()) {
                   std::uniform_int_distribution<> dis(0, generated_label_sets.size() - 1);
                   size_t fallback_idx = dis(gen);
                   label_set = generated_label_sets[fallback_idx];
                   // --- 备用机制也需要同步复制向量 ---
                   if (generate_vectors && generated_vectors.size() > fallback_idx) {
                       corresponding_vector = generated_vectors[fallback_idx];
                   }
               }
           }
       }

       if (!label_set.empty()) {
           #pragma omp critical
           {
               generated_label_sets.push_back(label_set);
               // --- 将对应的向量也存入结果 ---
               if (generate_vectors && !corresponding_vector.empty()) {
                   generated_vectors.push_back(corresponding_vector);
               }
               
               if (generated_label_sets.size() % 100 == 0) {
                  std::cout << "\rGenerated " << generated_label_sets.size() << " / " << num_points << " candidates..." << std::flush;
               }
           }
       }
   }

   // --- 将标签集结果写入输出文件 ---
   std::ofstream outfile(output_file);
   for (const auto& ls : generated_label_sets) {
       std::vector<LabelType> sorted_ls = ls;
       std::sort(sorted_ls.begin(), sorted_ls.end());
       for (size_t k = 0; k < sorted_ls.size(); ++k) {
           outfile << sorted_ls[k] << (k == sorted_ls.size() - 1 ? "" : ",");
       }
       outfile << std::endl;
   }
   std::cout << "\nGeneration complete. " << generated_label_sets.size() << " candidate queries written to " << output_file << std::endl;
   
   // --- 将向量结果写入输出文件 ---
   if (generate_vectors) {
       std::string output_vectors_file = vm["output_vectors_file"].as<std::string>();
       try {
           write_fvecs(output_vectors_file, generated_vectors);
           std::cout << generated_vectors.size() << " corresponding query vectors written to " << output_vectors_file << std::endl;
       } catch (const std::exception& e) {
           std::cerr << "Error writing output vectors: " << e.what() << std::endl;
       }
   }
}

// 'analyze' 模式: 分析候选查询池
void run_analyze_mode(const po::variables_map& vm) {
    std::string input_file = vm["input_file"].as<std::string>();
    std::string candidate_file = vm["candidate_file"].as<std::string>();
    std::string profiled_output = vm["profiled_output"].as<std::string>();

    TrieIndex trie_index;
    std::ifstream infile(input_file);
    IdxType new_label_set_id = 1;
    std::string line, label_str;
    while (std::getline(infile, line)) {
        std::vector<LabelType> label_set;
        std::stringstream ss(line);
        while (std::getline(ss, label_str, ',')) if (!label_str.empty()) label_set.emplace_back(std::stoul(label_str));
        if(!label_set.empty()) trie_index.insert(label_set, new_label_set_id);
    }

    std::cout << "Reading candidate file..." << std::endl;
    std::ifstream cand_file(candidate_file);
    
    std::vector<std::vector<LabelType>> queries_to_process;
    while(std::getline(cand_file, line)) {
        std::vector<LabelType> label_set;
        std::stringstream ss(line);
        while (std::getline(ss, label_str, ',')) if (!label_str.empty()) label_set.emplace_back(std::stoul(label_str));
        if (!label_set.empty()) {
            queries_to_process.push_back(label_set);
        }
    }
    cand_file.close();

    size_t num_queries = queries_to_process.size();
    std::cout << "Read " << num_queries << " queries. Analyzing in parallel..." << std::endl;
    
    std::vector<std::pair<IdxType, std::vector<LabelType>>> results(num_queries);
    std::atomic<size_t> progress_counter{0};

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_queries; ++i) {
        IdxType coverage = trie_index.calculate_coverage(queries_to_process[i]);
        results[i] = {coverage, queries_to_process[i]};
        
        size_t processed_count = ++progress_counter;
        if (processed_count % 100 == 0) {
            std::cout << "\rAnalyzed " << processed_count << " / " << num_queries << " queries..." << std::flush;
        }
    }

    std::cout << "\nAnalysis complete. Writing results..." << std::endl;
    
    std::ofstream prof_file(profiled_output);
    prof_file << "coverage_count,labels\n";
    for (const auto& res : results) {
        prof_file << res.first << ",";
        std::vector<LabelType> sorted_labels = res.second;
        std::sort(sorted_labels.begin(), sorted_labels.end());
        for (size_t k = 0; k < sorted_labels.size(); ++k) {
            prof_file << sorted_labels[k] << (k == sorted_labels.size() - 1 ? "" : " ");
        }
        prof_file << "\n";
    }
    std::cout << "Results written to " << profiled_output << std::endl;
}


int main(int argc, char** argv) {
    po::options_description generic_opts("Generic options");
    generic_opts.add_options()
        ("help,h", "Print help message")
        ("mode", po::value<std::string>()->default_value("generate"), "Run mode: generate, analyze");
    
    po::options_description generate_opts("Mode 'generate': Generate a candidate query pool");
    generate_opts.add_options()
        ("input_file", po::value<std::string>(), "Path to the base label file (required for generate/analyze)")
        ("output_file", po::value<std::string>(), "Output path for the generated candidate queries")
        ("base_vectors_file", po::value<std::string>(), "[Optional] Path to the base vectors file (.fvecs format)")
        ("output_vectors_file", po::value<std::string>(), "[Optional] Output path for the generated query vectors (.fvecs format)")
        ("num_points", po::value<IdxType>(), "Number of candidate queries to generate")
        ("K", po::value<IdxType>()->default_value(20), "Minimum number of potential results for a query to be valid")
        ("distribution_type", po::value<std::string>()->default_value("multi_normial"), "Distribution type: zipf, multi_normial")
        ("truncate_to_fixed_length", po::value<bool>()->default_value(true), "Use fixed-length queries. If false, query length is variable.")
        ("num_labels_per_query", po::value<IdxType>()->default_value(2), "The fixed number of labels per query (if truncate_to_fixed_length is true).")
        ("expected_num_label", po::value<IdxType>()->default_value(3), "Expected number of labels per query (if truncate_to_fixed_length is false).");

    po::options_description analyze_opts("Mode 'analyze': Analyze a candidate pool and profile coverage");
    analyze_opts.add_options()
        ("candidate_file", po::value<std::string>(), "Path to the candidate query file to be analyzed")
        ("profiled_output", po::value<std::string>(), "Output path for the analysis result (.csv)");

    po::options_description cmdline_opts;
    cmdline_opts.add(generic_opts).add(generate_opts).add(analyze_opts);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, cmdline_opts), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << "Multi-purpose Query Generation and Analysis Tool." << std::endl;
        std::cout << "Workflow: Use 'generate' to create a large pool, then 'analyze' to check its coverage distribution." << std::endl << std::endl;
        std::cout << generic_opts << std::endl;
        std::cout << generate_opts << std::endl;
        std::cout << analyze_opts << std::endl;
        return 0;
    }

    std::string mode = vm["mode"].as<std::string>();
    if (mode == "generate" || mode == "analyze") {
        if (!vm.count("input_file")) {
            std::cerr << "Error: Mode '" << mode << "' requires the --input_file parameter." << std::endl;
            return 1;
        }
    }
    
    if (mode == "generate") {
        run_generate_mode(vm);
    } else if (mode == "analyze") {
        run_analyze_mode(vm);
    } else {
        std::cerr << "Error: Unknown mode '" << mode << "'. Valid modes are: generate, analyze" << std::endl;
        return 1;
    }

    return 0;
}