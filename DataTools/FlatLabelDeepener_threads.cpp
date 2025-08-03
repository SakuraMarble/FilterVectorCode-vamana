#include <iostream>
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <stdexcept>
#include <mutex>

// --- MODIFICATION: Headers for memory usage monitoring ---
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <unistd.h>
#include <ios>
#include <fstream>
#include <string>
#elif defined(__APPLE__)
#include <mach/mach.h>
#endif

// OpenMP header for multi-threading
#include <omp.h>

// --- 1. 参数设置 ---
const int MIN_SUPPORT_COUNT = 10;
const int MIN_PATH_DEPTH = 3;
const double COVERAGE_TARGET = 0.99;

// --- 数据结构定义 ---
using LabelSet = std::unordered_set<int>;
using DataSet = std::vector<LabelSet>;
using InvertedIndex = std::unordered_map<int, std::vector<int>>;

struct TreeNode
{
   int label;
   std::vector<TreeNode> children;
};

struct ChildInfo
{
   int label;
   int support;
   std::vector<int> indices;
};

// --- 工具函数 ---

// --- MODIFICATION: New function to get memory usage ---
double get_memory_usage_mb()
{
#if defined(_WIN32)
   PROCESS_MEMORY_COUNTERS_EX pmc;
   if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pmc, sizeof(pmc)))
   {
      return static_cast<double>(pmc.PrivateUsage) / (1024.0 * 1024.0);
   }
   return 0.0;
#elif defined(__linux__)
   std::ifstream statm_file("/proc/self/statm");
   if (!statm_file.is_open())
      return 0.0;
   long rss_pages = 0;
   statm_file >> rss_pages >> rss_pages; // First value is virtual mem, second is resident set size (RSS)
   statm_file.close();
   long page_size_kb = sysconf(_SC_PAGESIZE) / 1024; // Page size in KB
   return static_cast<double>(rss_pages) * static_cast<double>(page_size_kb) / 1024.0;
#elif defined(__APPLE__)
   mach_task_basic_info_data_t info;
   mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
   if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &infoCount) == KERN_SUCCESS)
   {
      return static_cast<double>(info.resident_size) / (1024.0 * 1024.0);
   }
   return 0.0;
#else
   // Platform not supported
   return 0.0;
#endif
}

// --- MODIFICATION: log_info now includes memory usage ---
void log_info(const std::string &message)
{
   static std::mutex log_mutex;
   std::lock_guard<std::mutex> lock(log_mutex);

   auto now = std::chrono::system_clock::now();
   auto in_time_t = std::chrono::system_clock::to_time_t(now);

   std::stringstream ss;

#ifdef _WIN32
   std::tm buf;
   localtime_s(&buf, &in_time_t);
   ss << std::put_time(&buf, "%Y-%m-%d %X");
#else
   std::tm buf;
   ss << std::put_time(localtime_r(&in_time_t, &buf), "%Y-%m-%d %X");
#endif

   ss << " - INFO - [Mem: " << std::fixed << std::setprecision(2) << get_memory_usage_mb() << " MB] - " << message;

   std::cout << ss.str() << std::endl;
}

std::vector<int> sorted_vector_intersection(const std::vector<int> &vec1, const std::vector<int> &vec2)
{
   std::vector<int> intersection;
   if (vec1.empty() || vec2.empty())
      return intersection;

   std::set_intersection(vec1.begin(), vec1.end(),
                         vec2.begin(), vec2.end(),
                         std::back_inserter(intersection));
   return intersection;
}

// --- 2. 数据加载与预处理 ---
DataSet load_labels(const std::string &filepath)
{
   log_info("开始从文件加载数据: " + filepath);
   DataSet all_labels;
   std::ifstream file(filepath);
   std::string line;
   int count = 0;
   while (std::getline(file, line))
   {
      if (line.empty())
         continue;
      LabelSet labels;
      std::stringstream ss(line);
      std::string item;
      while (std::getline(ss, item, ','))
      {
         labels.insert(std::stoi(item));
      }
      all_labels.push_back(labels);
      if (++count % 100000 == 0)
      {
         log_info("已加载 " + std::to_string(count) + " 行...");
      }
   }
   log_info("数据加载完成，共 " + std::to_string(all_labels.size()) + " 行。");
   return all_labels;
}

InvertedIndex build_inverted_index(const DataSet &all_data)
{
   log_info("正在构建倒排索引，请耐心等待...");
   InvertedIndex index;
   for (size_t i = 0; i < all_data.size(); ++i)
   {
      for (int label : all_data[i])
      {
         index[label].push_back(i);
      }
   }

   log_info("正在排序倒排索引...");
   for (auto &pair : index)
   {
      std::sort(pair.second.begin(), pair.second.end());
   }

   log_info("倒排索引构建完成。索引了 " + std::to_string(index.size()) + " 个独立标签。");
   return index;
}

// --- 3. 阶段一：构建“主干树”森林 ---

std::vector<TreeNode> build_tree_recursive(const std::vector<int> &parent_path, const std::vector<int> &parent_row_indices, const InvertedIndex &inverted_index, int num_threads);

std::vector<TreeNode> build_tree_recursive(const std::vector<int> &parent_path, const std::vector<int> &parent_row_indices, const InvertedIndex &inverted_index, int num_threads)
{
   if (parent_row_indices.empty())
   {
      return {};
   }

   std::unordered_set<int> parent_path_set(parent_path.begin(), parent_path.end());

   std::vector<ChildInfo> children_with_support;
   for (const auto &pair : inverted_index)
   {
      int child_label = pair.first;
      if (parent_path_set.count(child_label))
         continue;
      auto intersected_vec = sorted_vector_intersection(parent_row_indices, pair.second);
      if (!intersected_vec.empty())
      {
         children_with_support.push_back({child_label, (int)intersected_vec.size(), intersected_vec});
      }
   }

   std::vector<ChildInfo> strong_children;
   for (const auto &info : children_with_support)
   {
      if (info.support >= MIN_SUPPORT_COUNT)
      {
         strong_children.push_back(info);
      }
   }

   if (!strong_children.empty())
   {
      std::vector<TreeNode> nodes;
      for (const auto &child_info : strong_children)
      {
         std::vector<int> new_path = parent_path;
         new_path.push_back(child_info.label);
         nodes.push_back({child_info.label, build_tree_recursive(new_path, child_info.indices, inverted_index, num_threads)});
      }
      return nodes;
   }
   else
   {
      std::sort(children_with_support.begin(), children_with_support.end(), [](const auto &a, const auto &b)
                { return a.support > b.support; });
      std::vector<TreeNode> leaf_nodes;
      for (const auto &info : children_with_support)
      {
         leaf_nodes.push_back({info.label, {}});
      }
      return leaf_nodes;
   }
}

int get_max_depth(const TreeNode &node)
{
   if (node.children.empty())
      return 1;
   int max_d = 0;
   for (const auto &child : node.children)
   {
      max_d = std::max(max_d, get_max_depth(child));
   }
   return 1 + max_d;
}

// --- MODIFICATION: Function signature changed to accept num_threads ---
std::pair<TreeNode, std::vector<int>> find_single_theme_tree(const std::vector<int> &data_indices, const InvertedIndex &inverted_index, int num_threads)
{
   log_info("在当前数据子集上寻找最佳根节点...");

   std::unordered_map<int, int> counter;
   for (const auto &pair : inverted_index)
   {
      counter[pair.first] = sorted_vector_intersection(data_indices, pair.second).size();
   }

   std::vector<std::pair<int, int>> sorted_counts;
   for (const auto &pair : counter)
      sorted_counts.push_back(pair);
   std::sort(sorted_counts.begin(), sorted_counts.end(), [](const auto &a, const auto &b)
             { return a.second > b.second; });

   int root_label = -1;
   for (const auto &pair : sorted_counts)
   {
      if (pair.second >= MIN_SUPPORT_COUNT)
      {
         root_label = pair.first;
         log_info("已选定本轮候选根: " + std::to_string(root_label) + " (支持度: " + std::to_string(pair.second) + ")");
         break;
      }
   }

   if (root_label == -1)
   {
      log_info("数据池中已没有任何标签满足最小支持度。");
      return {{0, {}}, {}};
   }

   auto root_row_indices = sorted_vector_intersection(data_indices, inverted_index.at(root_label));
   std::vector<int> parent_path = {root_label};

   std::vector<ChildInfo> strong_children;
   for (const auto &pair : inverted_index)
   {
      if (pair.first == root_label)
         continue;
      auto intersected_set = sorted_vector_intersection(root_row_indices, pair.second);
      if (intersected_set.size() >= MIN_SUPPORT_COUNT)
      {
         strong_children.push_back({pair.first, (int)intersected_set.size(), intersected_set});
      }
   }

   std::sort(strong_children.begin(), strong_children.end(), [](const auto &a, const auto &b)
             { return a.support > b.support; });

   std::vector<TreeNode> children_nodes(strong_children.size());
   if (!strong_children.empty())
   {
      log_info("路径 [" + std::to_string(root_label) + "]: 使用 OpenMP (最多 " + std::to_string(num_threads) + " 线程) 并行构建 " + std::to_string(strong_children.size()) + " 个子树...");

// --- BUG FIX: Added num_threads(num_threads) clause to control thread count ---
#pragma omp parallel for num_threads(num_threads)
      for (size_t i = 0; i < strong_children.size(); ++i)
      {
         const auto &child_info = strong_children[i];
         std::vector<int> new_path = parent_path;
         new_path.push_back(child_info.label);

         children_nodes[i] = {child_info.label, build_tree_recursive(new_path, child_info.indices, inverted_index, num_threads)};

         log_info("    -> 子树 " + std::to_string(child_info.label) + " 构建完成。(" + std::to_string(i + 1) + "/" + std::to_string(strong_children.size()) + ")");
      }
      log_info("... " + std::to_string(strong_children.size()) + " 个子树并行构建全部完成");
   }

   TreeNode tree = {root_label, children_nodes};
   int depth = get_max_depth(tree);
   log_info("为根 " + std::to_string(root_label) + " 构建的树，最大深度为: " + std::to_string(depth));

   if (depth >= MIN_PATH_DEPTH)
   {
      return {tree, root_row_indices};
   }

   log_info("根 " + std::to_string(root_label) + " 的树不满足最小深度要求。");
   return {{0, {}}, {}};
}

// --- MODIFICATION: Function signature changed to accept and pass num_threads ---
std::pair<std::vector<TreeNode>, std::vector<int>> build_theme_forest(const DataSet &original_data, const InvertedIndex &inverted_index, int num_threads)
{
   log_info("\n==================================================\n--- 阶段一：开始构建主干树森林 ---");
   std::vector<TreeNode> forest;
   std::vector<int> root_labels;

   std::vector<bool> remaining_flags(original_data.size(), true);
   long remaining_count = original_data.size();
   long total_data_count = original_data.size();

   int iteration_count = 1;
   while (true)
   {
      log_info("\n--- 森林构建迭代轮次: " + std::to_string(iteration_count) + " ---");
      if (remaining_count == 0)
      {
         log_info("所有数据已处理完毕。");
         break;
      }

      double coverage_ratio = 1.0 - (double)remaining_count / total_data_count;
      if (coverage_ratio >= COVERAGE_TARGET)
      {
         log_info("数据覆盖率 " + std::to_string(coverage_ratio * 100) + "% 已达到目标，停止挖掘。");
         break;
      }
      log_info("当前剩余数据: " + std::to_string(remaining_count) + " 行。已处理覆盖率: " + std::to_string(coverage_ratio * 100) + "%");

      std::vector<int> current_remaining_indices;
      for (int i = 0; i < total_data_count; ++i)
      {
         if (remaining_flags[i])
            current_remaining_indices.push_back(i);
      }

      // --- MODIFICATION: Pass num_threads to the tree finding function ---
      auto result = find_single_theme_tree(current_remaining_indices, inverted_index, num_threads);

      if (result.first.label == 0)
      {
         log_info("在剩余数据中已无法找到新树，挖掘结束。");
         break;
      }

      forest.push_back(result.first);
      root_labels.push_back(result.first.label);

      long newly_covered_count = 0;
      for (int idx : result.second)
      {
         if (remaining_flags[idx])
         {
            remaining_flags[idx] = false;
            newly_covered_count++;
         }
      }
      remaining_count -= newly_covered_count;
      iteration_count++;
   }
   log_info("\n--- 阶段一结束：共发现 " + std::to_string(forest.size()) + " 棵主干树 ---");
   return {forest, root_labels};
}

// 主程序
int main()
{
   // 在Windows上请使用类似 "C:/Users/YourUser/Data/celeba_attributes.txt" 的路径
   std::string input_file = "/home/fengxiaoyao/Data/data/celeba/data/celeba_attributes.txt";
   std::string roots_file = "tree_roots.txt";

   auto original_data = load_labels(input_file);
   if (original_data.empty())
      return 1;

   auto inverted_index = build_inverted_index(original_data);

   // --- MODIFICATION: Ensure num_threads is at least 1 ---
   int num_threads = omp_get_max_threads() / 3;
   if (num_threads < 1)
   {
      num_threads = 1;
   }
   log_info("OpenMP 将最多使用: " + std::to_string(num_threads) + " 个核心。");

   auto [forest, root_labels] = build_theme_forest(original_data, inverted_index, num_threads);

   if (forest.empty())
   {
      log_info("未能发现任何主干树。程序退出。");
      return 0;
   }

   log_info("正在生成树根信息文件 (" + roots_file + ")...");
   std::ofstream roots_out(roots_file);
   for (int label : root_labels)
   {
      roots_out << label << "\n";
   }
   roots_out.close();

   log_info("\n==================================================\n阶段一（建树）已成功完成！");
   log_info("The next step would be Phase 2 (label reconstruction), which is not included in this specific script.");

   return 0;
}