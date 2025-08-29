#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <utility>
#include <random>
#include <iomanip>
#include <chrono>
#include <functional> // for hash
#include <omp.h>      // for OpenMP

using namespace std;

// 优化后的标签集节点结构体
struct LabelSetNode
{
   vector<int> label_set;
   vector<unique_ptr<LabelSetNode>> children;
   vector<uint32_t> vectors;

   LabelSetNode(const vector<int> &labels) : label_set(labels)
   {
      sort(label_set.begin(), label_set.end()); // 保持有序便于比较和哈希
   }
};

class LNGVectorGenerator
{
private:
   int max_depth;
   vector<tuple<int, int, double>> segment_rules;
   int base_attr_start;
   unique_ptr<LabelSetNode> root;
   unordered_map<int, vector<LabelSetNode *>> nodes_by_depth;
   unordered_map<uint32_t, vector<int>> vector_to_labels;

   // 优化后的哈希函数
   struct VectorHash
   {
      size_t operator()(const vector<int> &v) const
      {
         size_t seed = v.size();
         for (const auto &i : v)
         {
            seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
         }
         return seed;
      }
   };

   unordered_map<vector<int>, vector<uint32_t>, VectorHash> label_set_to_vectors;
   vector<int> layer_group_counts;
   uint32_t total_vectors; // 总向量数使用32位
   vector<LabelSetNode *> all_nodes;

   // 验证和规范化分段规则
   vector<tuple<int, int, double>> validate_rules(const vector<tuple<int, int, double>> &rules, int max_depth)
   {
      if (rules.empty())
      {
         return {make_tuple(1, max_depth, 1.5)}; // 默认规则
      }

      // 按起始深度排序
      vector<tuple<int, int, double>> sorted_rules = rules;
      sort(sorted_rules.begin(), sorted_rules.end(),
           [](const auto &a, const auto &b)
           { return get<0>(a) < get<0>(b); });

      // 检查覆盖范围
      unordered_set<int> covered;
      for (const auto &rule : sorted_rules)
      {
         int start = get<0>(rule);
         int end = get<1>(rule);
         for (int d = start; d <= end; ++d)
         {
            covered.insert(d);
         }
      }

      // 查找未覆盖的深度
      vector<int> missing;
      for (int d = 1; d <= max_depth; ++d)
      {
         if (covered.find(d) == covered.end())
         {
            missing.push_back(d);
         }
      }

      // 为未覆盖的深度添加默认规则
      if (!missing.empty())
      {
         vector<pair<int, int>> ranges;
         if (!missing.empty())
         {
            int current_start = missing[0];
            int current_end = current_start;
            for (size_t i = 1; i < missing.size(); ++i)
            {
               if (missing[i] == current_end + 1)
               {
                  current_end = missing[i];
               }
               else
               {
                  ranges.emplace_back(current_start, current_end);
                  current_start = missing[i];
                  current_end = current_start;
               }
            }
            ranges.emplace_back(current_start, current_end);
         }

         for (const auto &range : ranges)
         {
            sorted_rules.emplace_back(range.first, range.second, 1.0);
         }
      }

      // 重新排序
      sort(sorted_rules.begin(), sorted_rules.end(),
           [](const auto &a, const auto &b)
           { return get<0>(a) < get<0>(b); });

      return sorted_rules;
   }

   // 计算每层的组数
   vector<int> calculate_layer_groups()
   {
      vector<int> layer_counts = {1}; // 第1层默认1个组

      for (int depth = 2; depth <= max_depth; ++depth)
      {
         int last_count = layer_counts.back();
         double factor = 1.0; // 默认增长因子

         // 查找适用于当前深度的规则
         for (const auto &rule : segment_rules)
         {
            int start = get<0>(rule);
            int end = get<1>(rule);
            if (start <= depth && depth <= end)
            {
               factor = get<2>(rule);
               break;
            }
         }

         int next_count = static_cast<int>(last_count * factor);
         layer_counts.push_back(max(static_cast<int>(1), next_count));
      }

      return layer_counts;
   }

   // 构建树结构
   void build_tree_structure()
   {
      cout << "开始构建树结构..." << endl;
      auto start_time = chrono::steady_clock::now();

      root = make_unique<LabelSetNode>(vector<int>{base_attr_start});
      nodes_by_depth[1].push_back(root.get());

      int attr_counter = base_attr_start + 1;
      uint32_t total_groups = accumulate(layer_group_counts.begin(), layer_group_counts.end(), 0);
      uint32_t processed_groups = 1; // 根节点已处理

      // 显示进度条
      auto print_progress = [&](int depth)
      {
         float progress = static_cast<float>(processed_groups) / total_groups * 100.0f;
         cout << "\r构建进度: 深度 " << depth << "/" << max_depth
              << " | 已完成 " << processed_groups << "/" << total_groups << " 组 ("
              << fixed << setprecision(1) << progress << "%)" << flush;
      };

      for (int depth = 2; depth <= max_depth; ++depth)
      {
         int target_count = layer_group_counts[depth - 1];
         auto &parent_nodes = nodes_by_depth[depth - 1];

         int children_per_parent = target_count / parent_nodes.size();
         int remainder = target_count % parent_nodes.size();

         vector<int> allocations(parent_nodes.size(), children_per_parent);
         for (int i = 0; i < remainder; ++i)
         {
            allocations[i] += 1;
         }

         for (size_t parent_idx = 0; parent_idx < parent_nodes.size(); ++parent_idx)
         {
            auto parent = parent_nodes[parent_idx];
            for (int _ = 0; _ < allocations[parent_idx]; ++_)
            {
               vector<int> new_label = parent->label_set;
               new_label.push_back(attr_counter);
               auto child_node = make_unique<LabelSetNode>(new_label);
               nodes_by_depth[depth].push_back(child_node.get());
               parent->children.push_back(move(child_node));
               attr_counter += 1;
               processed_groups++;
            }
         }

         // 每层完成后更新进度
         print_progress(depth);
      }

      auto end_time = chrono::steady_clock::now();
      auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
      cout << "\n树结构构建完成! 耗时: " << duration.count() << " 毫秒" << endl;
   }

   // 获取所有节点
   void collect_all_nodes()
   {
      if (!root)
         return;

      queue<LabelSetNode *> q;
      q.push(root.get());

      cout << "收集所有节点..." << endl;
      uint32_t node_count = 0;

      while (!q.empty())
      {
         auto node = q.front();
         q.pop();
         if (node)
         {
            all_nodes.push_back(node);
            node_count++;
            for (const auto &child : node->children)
            {
               q.push(child.get());
            }
         }
      }

      cout << "共收集 " << node_count << " 个节点" << endl;
   }

   // 随机分配向量
   void assign_vectors_random()
   {
      cout << "开始随机分配向量..." << endl;

      // 分配基础向量（每个节点先分配一个）
      for (size_t i = 0; i < all_nodes.size(); ++i)
      {
         all_nodes[i]->vectors.push_back(static_cast<uint32_t>(i));
      }

      uint32_t remaining = total_vectors - static_cast<uint32_t>(all_nodes.size());
      cout << "需要分配 " << remaining << " 个额外向量" << endl;

      const uint32_t report_interval = max(static_cast<uint32_t>(1), remaining / 10);

// 使用OpenMP并行化
#pragma omp parallel
      {
         random_device rd;
         mt19937 gen(rd());
         uniform_int_distribution<uint32_t> dis(0, static_cast<uint32_t>(all_nodes.size()) - 1);

#pragma omp for schedule(dynamic)
         for (uint32_t vec_id = static_cast<uint32_t>(all_nodes.size()); vec_id < total_vectors; ++vec_id)
         {
            uint32_t chosen_group = dis(gen);
#pragma omp critical
            {
               all_nodes[chosen_group]->vectors.push_back(vec_id);
            }

            if ((vec_id - static_cast<uint32_t>(all_nodes.size())) % report_interval == 0)
            {
               float progress = static_cast<float>(vec_id - all_nodes.size()) / remaining * 100.0f;
#pragma omp critical
               {
                  cout << "\r分配进度: " << fixed << setprecision(1) << progress << "%" << flush;
               }
            }
         }
      }

      cout << "\n随机向量分配完成!" << endl;
   }

   // 按权重分配向量
   void assign_vectors_weighted(double exponent = 0.5)
   {
      cout << "开始按权重分配向量 (指数=" << exponent << ")..." << endl;

      // 分配基础向量
      for (size_t i = 0; i < all_nodes.size(); ++i)
      {
         all_nodes[i]->vectors.push_back(static_cast<uint32_t>(i));
      }

      // 计算层权重
      vector<double> layer_weights;
      double total_weight = 0.0;
      for (int depth = 1; depth <= max_depth; ++depth)
      {
         double weight = 1.0 / pow(depth, exponent);
         layer_weights.push_back(weight);
         total_weight += weight;
      }

      // 归一化权重
      for (auto &weight : layer_weights)
      {
         weight /= total_weight;
      }

      // 计算每层分配的向量数
      uint32_t remaining_vectors = total_vectors - static_cast<uint32_t>(all_nodes.size());
      vector<uint32_t> layer_vector_counts(max_depth);
      uint32_t total_allocated = 0;

      for (int i = 0; i < max_depth; ++i)
      {
         layer_vector_counts[i] = static_cast<uint32_t>(layer_weights[i] * remaining_vectors);
         total_allocated += layer_vector_counts[i];
      }

      // 调整舍入误差
      int32_t diff = remaining_vectors - total_allocated;
      if (diff != 0)
      {
         layer_vector_counts.back() += diff;
      }

      // 分配向量到各层
      uint32_t next_vec_id = static_cast<uint32_t>(all_nodes.size());
      cout << "需要分配 " << remaining_vectors << " 个额外向量到各层" << endl;

      for (int depth = 1; depth <= max_depth; ++depth)
      {
         auto &groups_in_layer = nodes_by_depth[depth];
         uint32_t vectors_for_layer = layer_vector_counts[depth - 1];

         if (vectors_for_layer == 0 || groups_in_layer.empty())
         {
            continue;
         }

         uint32_t vectors_per_group = vectors_for_layer / groups_in_layer.size();
         uint32_t remainder = vectors_for_layer % groups_in_layer.size();

         cout << "正在分配层 " << depth << ": " << vectors_for_layer << " 个向量到 "
              << groups_in_layer.size() << " 个组..." << endl;

         for (size_t i = 0; i < groups_in_layer.size(); ++i)
         {
            uint32_t additional = vectors_per_group + (i < remainder ? 1 : 0);
            if (additional > 0)
            {
               for (uint32_t j = 0; j < additional; ++j)
               {
                  groups_in_layer[i]->vectors.push_back(next_vec_id++);
               }
            }
         }
      }

      cout << "权重向量分配完成!" << endl;
   }

   // 构建映射关系
   void build_mappings()
   {
      cout << "开始构建映射关系..." << endl;
      uint32_t total_vectors = 0;

      for (const auto &node : all_nodes)
      {
         total_vectors += node->vectors.size();
         for (uint32_t vec_id : node->vectors)
         {
            vector_to_labels[vec_id] = node->label_set;
            label_set_to_vectors[node->label_set].push_back(vec_id);
         }
      }

      cout << "映射关系构建完成! 共处理 " << total_vectors << " 个向量" << endl;
   }

public:
   LNGVectorGenerator(int max_depth, const vector<tuple<int, int, double>> &segment_rules, int base_attr_start = 1)
       : max_depth(max_depth), segment_rules(validate_rules(segment_rules, max_depth)), base_attr_start(base_attr_start)
   {
      cout << "初始化LNG生成器..." << endl;
      cout << "最大深度: " << max_depth << endl;
      cout << "基础属性起始值: " << base_attr_start << endl;
      this->segment_rules = validate_rules(segment_rules, max_depth);
      layer_group_counts = calculate_layer_groups();
      cout << "每层组数计算完成" << endl;
   }

   void build(const string &dataset, uint32_t total_vectors, const string &distribution_mode = "weighted", double layer_weight_exponent = 0.5)
   {
      auto total_start = chrono::steady_clock::now();
      cout << "\n=== 开始构建LNG ===" << endl;
      cout << "数据集: " << dataset << endl;
      cout << "总向量数: " << total_vectors << endl;
      cout << "分配模式: " << distribution_mode << endl;

      if (accumulate(layer_group_counts.begin(), layer_group_counts.end(), 0) > total_vectors)
      {
         throw runtime_error("错误: 总向量数不足");
      }

      this->total_vectors = total_vectors;

      if (!root)
      {
         build_tree_structure();
      }
      collect_all_nodes();

      auto assign_start = chrono::steady_clock::now();
      if (distribution_mode == "random")
      {
         assign_vectors_random();
      }
      else if (distribution_mode == "weighted")
      {
         assign_vectors_weighted(layer_weight_exponent);
      }
      else
      {
         throw runtime_error("错误: 无效的分配模式");
      }
      auto assign_end = chrono::steady_clock::now();
      cout << "向量分配耗时: "
           << chrono::duration_cast<chrono::milliseconds>(assign_end - assign_start).count()
           << " 毫秒" << endl;

      build_mappings();
      print_and_save_stats(dataset + "_base_labels_info.log");

      auto total_end = chrono::steady_clock::now();
      cout << "\n=== LNG构建完成 ===" << endl;
      cout << "总耗时: "
           << chrono::duration_cast<chrono::milliseconds>(total_end - total_start).count()
           << " 毫秒" << endl;
   }

   // 打印并保存统计信息
   void print_and_save_stats(const string &output_file)
   {
      cout << "\n正在生成统计信息..." << endl;
      ofstream f(output_file);
      if (!f.is_open())
      {
         cerr << "无法打开文件: " << output_file << endl;
         return;
      }

      vector<string> stats;
      stats.push_back("=== LNG树统计信息 ===");
      stats.push_back("总层数: " + to_string(max_depth));
      stats.push_back("总group数: " + to_string(accumulate(layer_group_counts.begin(), layer_group_counts.end(), 0)));
      stats.push_back("总向量数: " + to_string(total_vectors));
      stats.push_back("\n层级分布:");
      stats.push_back("层号\tgroup数\t向量数\t平均向量/group");

      for (int depth = 1; depth <= max_depth; ++depth)
      {
         int groups = nodes_by_depth[depth].size();
         uint32_t vectors = 0;
         for (const auto &node : nodes_by_depth[depth])
         {
            vectors += node->vectors.size();
         }
         double avg = groups > 0 ? static_cast<double>(vectors) / groups : 0.0;
         stats.push_back(to_string(depth) + "\t" + to_string(groups) + "\t" +
                         to_string(vectors) + "\t" + to_string(avg));
      }

      // 输出到控制台和文件
      for (const auto &line : stats)
      {
         cout << line << endl;
         f << line << endl;
      }

      f.close();
      cout << "统计信息已保存到 " << output_file << endl;
   }

   // 保存向量到文件
   void save_to_file(const string &filename = "lng_tree_vectors.txt")
   {
      cout << "\n正在保存向量到文件..." << endl;
      ofstream f(filename);
      if (!f.is_open())
      {
         cerr << "无法打开文件: " << filename << endl;
         return;
      }

      // 收集并排序所有向量ID
      vector<uint32_t> sorted_vector_ids;
      for (const auto &pair : vector_to_labels)
      {
         sorted_vector_ids.push_back(pair.first);
      }
      sort(sorted_vector_ids.begin(), sorted_vector_ids.end());

      // 写入文件
      const uint32_t report_interval = max(static_cast<uint32_t>(1), static_cast<uint32_t>(sorted_vector_ids.size()) / 10);
      for (size_t i = 0; i < sorted_vector_ids.size(); ++i)
      {
         uint32_t vec_id = sorted_vector_ids[i];
         const auto &labels = vector_to_labels[vec_id];
         vector<int> sorted_labels(labels.begin(), labels.end());
         sort(sorted_labels.begin(), sorted_labels.end());

         for (size_t j = 0; j < sorted_labels.size(); ++j)
         {
            if (j != 0)
               f << ",";
            f << sorted_labels[j];
         }
         f << "\n";

         if (i % report_interval == 0)
         {
            float progress = static_cast<float>(i) / sorted_vector_ids.size() * 100.0f;
            cout << "\r保存进度: " << fixed << setprecision(1) << progress << "%" << flush;
         }
      }

      f.close();
      cout << "\n向量-标签映射已保存到 " << filename << endl;
   }
};

int main()
{
   string dataset = "captcha";
   vector<tuple<int, int, double>> segment_rules;
   int max_depth;
   uint32_t total_vectors;

   // 配置参数
   if (dataset == "words")
   {
      segment_rules = {
          make_tuple(1, 3, 1.0),
          make_tuple(4, 5, 2.0),
          make_tuple(6, 9, 1.25),
          make_tuple(10, 1000, 1.0)};
      max_depth = 1000;
      total_vectors = 8000;
   }
   else if (dataset == "MTG")
   {
      segment_rules = {
          make_tuple(1, 3, 2.0),
          make_tuple(4, 7, 1.25),
          make_tuple(8, 5000, 1.03)};
      max_depth = 5000;
      total_vectors = 40274;
   }
   else if (dataset == "arxiv")
   {
      segment_rules = {
          make_tuple(1, 3, 2.0),
          make_tuple(3, 8, 1.3),
          make_tuple(9, 20000, 1.03)};
      max_depth = 20000;
      total_vectors = 157606;
   }
   else if (dataset == "captcha")
   {
      segment_rules = {
          make_tuple(1, 3, 2.0),
          make_tuple(3, 7, 1.3),
          make_tuple(8, 10000, 1.03)};
      max_depth = 10000;
      total_vectors = 1365874;
   }

   try
   {
      cout << "=== LNG向量生成器 ===" << endl;
      cout << "配置: " << dataset << " 数据集" << endl;
      cout << "最大深度: " << max_depth << endl;
      cout << "总向量数: " << total_vectors << endl;

      LNGVectorGenerator generator(max_depth, segment_rules, 1);
      cout << "\n=== 随机分配模式 ===" << endl;
      generator.build(dataset, total_vectors, "random");
      generator.save_to_file(dataset + "_base_labels.txt");
   }
   catch (const exception &e)
   {
      cerr << "错误: " << e.what() << endl;
      return 1;
   }

   return 0;
}