#include <set>
#include <queue>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <thread>
#include <chrono>
#include <omp.h>
#include <unordered_set>
#include "trie.h"

namespace ANNS
{
   TrieIndex::TrieIndex()
   {
      _root = std::make_shared<TrieNode>(0, nullptr);
   }

   // insert a new label set into the trie tree, increase the group size
   IdxType TrieIndex::insert(const std::vector<LabelType> &label_set, IdxType &new_label_set_id)
   {
      std::shared_ptr<TrieNode> cur = _root;
      for (const LabelType label : label_set)
      {
         // create a new node
         if (cur->children.find(label) == cur->children.end())
         {
            cur->children[label] = std::make_shared<TrieNode>(label, cur);

            // update max label id and label_to_nodes
            if (label > _max_label_id)
            {
               _max_label_id = label;
               _label_to_nodes.resize(_max_label_id + 1);
            }
            // _label_to_nodes.resize(_max_label_id + 1);
            _label_to_nodes[label].push_back(cur->children[label]);
         }
         cur = cur->children[label];
      }

      // set the group_id and group_size
      if (cur->group_id == 0)
      {
         cur->group_id = new_label_set_id++;
         cur->label_set_size = label_set.size();
         cur->group_size = 1;
      }
      else
      {
         cur->group_size++;
      }
      return cur->group_id;
   }

   // find the exact match of the label set
   std::shared_ptr<TrieNode> TrieIndex::find_exact_match(const std::vector<LabelType> &label_set) const
   {
      std::shared_ptr<TrieNode> cur = _root;
      for (const LabelType label : label_set)
      {
         if (cur->children.find(label) == cur->children.end())
            return nullptr;
         cur = cur->children[label];
      }

      // check whether it is a terminal node
      if (cur->group_id == 0)
         return nullptr;
      return cur;
   }

   // get the top entrances of all super sets in the trie tree, assume the label_set has been sorted in ascending order
   void TrieIndex::get_super_set_entrances(const std::vector<LabelType> &label_set,
                                           std::vector<std::shared_ptr<TrieNode>> &super_set_entrances,
                                           bool avoid_self, bool need_containment) const
   {
      super_set_entrances.clear();

      // find the existing node for the input label set
      std::shared_ptr<TrieNode> avoided_node = nullptr;
      if (avoid_self)
         avoided_node = find_exact_match(label_set);
      std::queue<std::shared_ptr<TrieNode>> q;

      // if the label set is empty, find all children of the root
      if (label_set.empty())
      {
         for (const auto &child : _root->children)
            q.push(child.second);
      }
      else
      {

         // if need containing the input label set, obtain candidate nodes for the last label
         if (need_containment)
         {
            for (auto node : _label_to_nodes[label_set[label_set.size() - 1]])
               if (examine_containment(label_set, node))
                  q.push(node);
         }
         else // if no need for containing the whole label set
         {
            for (auto label : label_set)
               for (auto node : _label_to_nodes[label])
                  if (examine_smallest(label_set, node))
                     q.push(node);
         }
      }

      // search in the trie tree to find the candidate super sets
      std::set<IdxType> group_ids;
      while (!q.empty())
      {
         auto cur = q.front();
         q.pop();

         // add to candidates if it is a terminal node
         if (cur->group_id > 0 && cur != avoided_node && group_ids.find(cur->group_id) == group_ids.end())
         {
            group_ids.insert(cur->group_id);
            super_set_entrances.push_back(cur);
         }
         else
         {
            for (const auto &child : cur->children)
               q.push(child.second);
         }
      }
   }

   // fxy_add:方法一debug
   void TrieIndex::get_super_set_entrances_debug(const std::vector<LabelType> &label_set,
                                                 std::vector<std::shared_ptr<TrieNode>> &super_set_entrances,
                                                 bool avoid_self, bool need_containment,
                                                 std::atomic<int> &print_counter, TrieMethod1Metrics &metrics) const // 1. 添加 print_counter 参数
   {
      super_set_entrances.clear();
      metrics = {};

      // --- Metrics & Timers Initialization ---
      auto function_start_time = std::chrono::high_resolution_clock::now();
      double time_candidate_gen = 0.0;
      double time_bfs = 0.0;
      long long initial_candidates_from_map = 0;
      long long examine_containment_calls = 0;
      long long successful_containment_checks = 0;
      long long bfs_nodes_processed = 0;
      long long upward_traversal_nodes = 0; // 用于累加向上回溯的节点数

      // find the existing node for the input label set
      std::shared_ptr<TrieNode> avoided_node = nullptr;
      if (avoid_self)
         avoided_node = find_exact_match(label_set);

      std::queue<std::shared_ptr<TrieNode>> q;

      // --- Phase 1: Candidate Generation & Verification ---
      auto start_candidate_gen = std::chrono::high_resolution_clock::now();

      if (label_set.empty())
      {
         for (const auto &child : _root->children)
            q.push(child.second);
         successful_containment_checks = _root->children.size();
      }
      else
      {
         if (need_containment)
         {
            const auto &candidates_from_map = _label_to_nodes.at(label_set.back());
            initial_candidates_from_map = candidates_from_map.size(); // [METRIC]

            for (auto node : candidates_from_map)
            {
               examine_containment_calls++; // [METRIC]
               if (examine_containment_debug(label_set, node, upward_traversal_nodes))
               {
                  successful_containment_checks++; // [METRIC]
                  q.push(node);
               }
            }
         }
         else // if no need for containing the whole label set
         {
            for (auto label : label_set)
            {
               const auto &candidates_from_map = _label_to_nodes.at(label);
               initial_candidates_from_map += candidates_from_map.size(); // [METRIC]

               for (auto node : candidates_from_map)
               {
                  // Assuming examine_smallest is a similar check. We count it as a "call".
                  examine_containment_calls++; // [METRIC]
                  if (examine_smallest(label_set, node))
                  {
                     successful_containment_checks++; // [METRIC]
                     q.push(node);
                  }
               }
            }
         }
      }
      auto end_candidate_gen = std::chrono::high_resolution_clock::now();
      time_candidate_gen = std::chrono::duration<double, std::milli>(end_candidate_gen - start_candidate_gen).count();

      // 填充 metrics 结构体 ===
      metrics.initial_candidates = initial_candidates_from_map;
      metrics.successful_checks = successful_containment_checks;
      metrics.upward_traversals = upward_traversal_nodes;

      // --- Phase 2: Downward BFS Search ---
      auto start_bfs = std::chrono::high_resolution_clock::now();

      std::set<IdxType> group_ids;
      while (!q.empty())
      {
         auto cur = q.front();
         q.pop();
         bfs_nodes_processed++; // [METRIC]

         if (cur->group_id > 0 && cur != avoided_node && group_ids.find(cur->group_id) == group_ids.end())
         {
            group_ids.insert(cur->group_id);
            super_set_entrances.push_back(cur);
         }
         else
         {
            for (const auto &child : cur->children)
               q.push(child.second);
         }
      }
      auto end_bfs = std::chrono::high_resolution_clock::now();
      time_bfs = std::chrono::duration<double, std::milli>(end_bfs - start_bfs).count();

      metrics.bfs_nodes_processed = bfs_nodes_processed;

      // --- 4. 使用原子计数器控制打印次数 ---
      if (print_counter.fetch_add(1, std::memory_order_relaxed) < 10) // 2. 使用原子计数器
      {
#pragma omp critical
         {
            long long total_traversed = upward_traversal_nodes + bfs_nodes_processed; // 计算总数
            std::cout << "\n--- Method 1 (Shortcut) Performance Analysis ---\n"
                      << "Query: " << "size=" << label_set.size() << ", last_label=" << (label_set.empty() ? -1 : label_set.back()) << "\n"
                      << "--- Phase 1: Candidate Generation & Verification ---\n"
                      << "   - Initial candidates from map: " << initial_candidates_from_map << "\n"
                      << "   - `examine_containment` calls: " << examine_containment_calls << "\n"
                      << "   - Nodes traversed UPWARDS: " << upward_traversal_nodes << " <--- (New Metric)\n" // 新指标
                      << "   - Successful checks (queue size): " << successful_containment_checks << "\n"
                      << "   - Time for this phase: " << time_candidate_gen << " ms\n"
                      << "--- Phase 2: Downward BFS ---\n"
                      << "   - Nodes traversed DOWNWARDS (BFS): " << bfs_nodes_processed << "\n" // 明确这是向下
                      << "   - Time for this phase: " << time_bfs << " ms\n"
                      << "--- Summary ---\n"
                      << "   - Total super sets found: " << super_set_entrances.size() << "\n"
                      << "   - Total Nodes Traversed (Up + Down): " << total_traversed << "\n" // 总数
                      << "   - Total function time: " << std::chrono::duration<double, std::milli>(end_bfs - function_start_time).count() << " ms\n"
                      << "--------------------------------------------------\n"
                      << std::endl;
         }
      }
   }

   // fxy_add: 方法二新的主入口函数调试输出版本
   void TrieIndex::get_super_set_entrances_new_debug(const std::vector<LabelType> &label_set,
                                                     std::vector<std::shared_ptr<TrieNode>> &super_set_entrances,
                                                     bool avoid_self, bool need_containment,
                                                     std::atomic<int> &print_counter, TrieSearchMetricsRecursive &metrics) const
   {
      super_set_entrances.clear();
      metrics = {}; // 清空传入的 metrics

      if (!need_containment)
         return;

      // --- 初始化 ---
      auto function_start_time = std::chrono::high_resolution_clock::now();

      std::shared_ptr<TrieNode> avoided_node = nullptr;
      if (avoid_self)
      {
         avoided_node = find_exact_match(label_set);
      }
      std::set<IdxType> visited_groups;

      // --- 启动核心递归搜索 ---
      find_supersets_recursive_debug(
          _root, label_set, 0, super_set_entrances,
          visited_groups, avoided_node, metrics, 0 // 传入指标对象和初始深度0
      );

      auto function_end_time = std::chrono::high_resolution_clock::now();
      double total_time = std::chrono::duration<double, std::milli>(function_end_time - function_start_time).count();

      // --- 统一打印报告 ---
      if (print_counter.fetch_add(1, std::memory_order_relaxed) < 10)
      {
#pragma omp critical
         {
            long long total_traversed = metrics.recursive_calls + metrics.nodes_processed_in_bfs; // 计算总数
            std::cout << "\n--- Method 2 (Recursive) Performance Analysis ---\n"
                      << "Query: size=" << label_set.size() << ", first_label=" << label_set[0] << "\n"
                      << "--- Phase 1: Recursive Search (DFS) ---\n"
                      << "   - Nodes traversed in DFS: " << metrics.recursive_calls << "\n"
                      << "   - Pruning events (IMPORTANT): " << metrics.pruning_events << "\n"
                      << "   - Max recursion depth: " << metrics.max_recursion_depth << "\n"
                      << "--- Phase 2: Result Collection (BFS) ---\n"
                      << "   - Collection function calls: " << metrics.collection_calls << "\n"
                      << "   - Nodes traversed in all BFS: " << metrics.nodes_processed_in_bfs << "\n"
                      << "--- Summary ---\n"
                      << "   - Total super sets found: " << super_set_entrances.size() << "\n"
                      << "   - Total Nodes Traversed (DFS + BFS): " << total_traversed << "\n" // 总数
                      << "   - Total function time: " << total_time << " ms\n"
                      << "-----------------------------------------------------------\n"
                      << std::endl;
         }
      }
   }

   // fxy_add:方法二新的主入口函数调试输出版本,不从root进入
   void TrieIndex::get_super_set_entrances_new_more_sp_debug(const std::vector<LabelType> &label_set,
                                                             std::vector<std::shared_ptr<TrieNode>> &super_set_entrances,
                                                             bool avoid_self, bool need_containment,
                                                             std::atomic<int> &print_counter, TrieSearchMetricsRecursive &metrics) const
   {
      // --- 0. 预处理与边界检查 (逻辑不变) ---
      super_set_entrances.clear();
      metrics = {}; // 重置性能指标

      if (!need_containment || label_set.empty())
      {
         return;
      }
      if (label_set[0] >= _label_to_nodes.size())
      {
         return;
      }

      // --- 1. 初始化 (逻辑不变) ---
      auto function_start_time = std::chrono::high_resolution_clock::now();

      std::shared_ptr<TrieNode> avoided_node = nullptr;
      if (avoid_self)
      {
         avoided_node = find_exact_match(label_set);
      }

      std::set<IdxType> visited_groups;

      const auto &entry_points = _label_to_nodes.at(label_set[0]);
      if (entry_points.empty())
      {
         return;
      }

      // --- 2. 串行搜索 ---
      for (const auto &start_node : entry_points)
      {
         if (label_set.size() == 1)
         {
            // [MODIFIED] 直接将结果写入主输出变量 super_set_entrances 和 metrics
            collect_all_terminals_debug(start_node, super_set_entrances, visited_groups, avoided_node, metrics);
         }
         else
         {
            // [MODIFIED] 直接将结果写入主输出变量 super_set_entrances 和 metrics
            find_supersets_recursive_debug(
                start_node,
                label_set,
                1,
                super_set_entrances, // 直接使用主结果容器
                visited_groups,
                avoided_node,
                metrics, // 直接使用主指标结构体
                1);
         }
      }
      // --- 3. 最终报告 ---
      auto function_end_time = std::chrono::high_resolution_clock::now();
      double total_time = std::chrono::duration<double, std::milli>(function_end_time - function_start_time).count();

      if (print_counter.fetch_add(1, std::memory_order_relaxed) < 10)
      {
// 这里的临界区可以保留，以防止此函数本身在外部被多线程调用时打印混乱
#pragma omp critical(PrintSuperSetReport)
         {
            long long total_traversed = metrics.recursive_calls + metrics.nodes_processed_in_bfs; // 计算总数
            std::cout << "\n--- Method 2 (Recursive) Performance Analysis ---\n"
                      << "Query: size=" << label_set.size() << ", first_label=" << label_set[0] << "\n"
                      << "--- Phase 1: Recursive Search (DFS) ---\n"
                      << "   - Nodes traversed in DFS: " << metrics.recursive_calls << "\n"
                      << "   - Pruning events (IMPORTANT): " << metrics.pruning_events << "\n"
                      << "   - Max recursion depth: " << metrics.max_recursion_depth << "\n"
                      << "--- Phase 2: Result Collection (BFS) ---\n"
                      << "   - Collection function calls: " << metrics.collection_calls << "\n"
                      << "   - Nodes traversed in all BFS: " << metrics.nodes_processed_in_bfs << "\n"
                      << "--- Summary ---\n"
                      << "   - Total super sets found: " << super_set_entrances.size() << "\n"
                      << "   - Total Nodes Traversed (DFS + BFS): " << total_traversed << "\n" // 总数
                      << "   - Total function time: " << total_time << " ms\n"
                      << "-----------------------------------------------------------\n"
                      << std::endl;
         }
      }
   }

   // fxy_mod: 增加了 metrics 参数来累积BFS相关的统计数据
   void TrieIndex::collect_all_terminals_debug(
       std::shared_ptr<TrieNode> start_node,
       std::vector<std::shared_ptr<TrieNode>> &results,
       std::set<IdxType> &visited_groups,
       const std::shared_ptr<TrieNode> &avoided_node,
       TrieSearchMetricsRecursive &metrics) const
   {
      // [METRIC] 记录收集函数被调用
      metrics.collection_calls++;

      std::queue<std::shared_ptr<TrieNode>> q;
      if (start_node)
         q.push(start_node);

      while (!q.empty())
      {
         auto cur = q.front();
         q.pop();

         // [METRIC] 记录BFS中处理的节点数
         metrics.nodes_processed_in_bfs++;

         if (cur->group_id > 0 && cur != avoided_node && visited_groups.find(cur->group_id) == visited_groups.end())
         {
            visited_groups.insert(cur->group_id);
            results.push_back(cur);
         }

         for (const auto &child_pair : cur->children)
         {
            q.push(child_pair.second);
         }
      }
   }

   // fxy_mod: 增加了 metrics 和 current_depth 参数来收集指标
   void TrieIndex::find_supersets_recursive_debug(
       std::shared_ptr<TrieNode> current_node,
       const std::vector<LabelType> &sorted_query,
       size_t query_idx,
       std::vector<std::shared_ptr<TrieNode>> &results,
       std::set<IdxType> &visited_groups,
       const std::shared_ptr<TrieNode> &avoided_node,
       TrieSearchMetricsRecursive &metrics, // [MOD]
       int current_depth) const             // [MOD]
   {
      // [METRIC] 更新递归调用次数和最大深度
      metrics.recursive_calls++;
      metrics.max_recursion_depth = std::max(metrics.max_recursion_depth, current_depth);

      // === 递归终止条件 ===
      if (query_idx == sorted_query.size())
      {
         // [METRIC] 计时并调用收集函数
         auto bfs_start_time = std::chrono::high_resolution_clock::now();
         collect_all_terminals_debug(current_node, results, visited_groups, avoided_node, metrics);
         auto bfs_end_time = std::chrono::high_resolution_clock::now();
         metrics.time_in_collection_bfs += std::chrono::duration<double, std::milli>(bfs_end_time - bfs_start_time).count();
         return;
      }

      LabelType target_label = sorted_query[query_idx];

      // === 递归递进逻辑 ===
      for (const auto &child_pair : current_node->children)
      {
         LabelType child_label = child_pair.first;
         auto child_node = child_pair.second;

         if (child_label < target_label)
         {
            find_supersets_recursive_debug(child_node, sorted_query, query_idx, results, visited_groups, avoided_node, metrics, current_depth + 1);
         }
         else if (child_label == target_label)
         {
            find_supersets_recursive_debug(child_node, sorted_query, query_idx + 1, results, visited_groups, avoided_node, metrics, current_depth + 1);
         }
         else // child_label > target_label
         {
            // [METRIC] 关键！记录剪枝事件
            metrics.pruning_events++;
         }
      }
   }

   // bottom to top, examine whether the current node is the smallest in the label set
   bool TrieIndex::examine_smallest(const std::vector<LabelType> &label_set,
                                    const std::shared_ptr<TrieNode> &node) const
   {
      auto cur = node->parent;
      while (cur != nullptr && cur->label >= label_set[0])
      {
         if (std::binary_search(label_set.begin(), label_set.end(), cur->label))
            return false;
         cur = cur->parent;
      }
      return true;
   }

   // bottom to top, examine whether is a super set of the label set
   bool TrieIndex::examine_containment(const std::vector<LabelType> &label_set,
                                       const std::shared_ptr<TrieNode> &node) const
   {
      auto cur = node->parent;
      for (int64_t i = label_set.size() - 2; i >= 0; --i)
      {
         while (cur->label > label_set[i] && cur->parent != nullptr)
            cur = cur->parent;
         if (cur->parent == nullptr || cur->label != label_set[i])
            return false;
      }
      return true;
   }

   // fxy_add: debug版本的 examine_containment，增加了节点遍历计数
   bool TrieIndex::examine_containment_debug(const std::vector<LabelType> &label_set,
                                             const std::shared_ptr<TrieNode> &node,
                                             long long &nodes_traversed) const
   {
      auto cur = node->parent;
      nodes_traversed++; // <--- 计数起始节点（的父节点）

      for (int64_t i = label_set.size() - 2; i >= 0; --i)
      {
         while (cur->label > label_set[i] && cur->parent != nullptr)
         {
            cur = cur->parent;
            nodes_traversed++; // <--- 在循环中计数每次向上移动
         }
         if (cur->parent == nullptr || cur->label != label_set[i])
            return false;
      }
      return true;
   }

   // save the trie tree to a file
   void TrieIndex::save(std::string filename) const
   {
      std::ofstream out(filename);

      // save the max label id and number of nodes
      out << _max_label_id << std::endl;
      IdxType num_nodes = 1;
      for (const auto &nodes : _label_to_nodes)
         num_nodes += nodes.size();
      out << num_nodes << std::endl;

      // save the root node
      std::unordered_map<std::shared_ptr<TrieNode>, IdxType> node_to_id;
      out << 0 << " " << _root->label << " " << _root->group_id << " "
          << _root->label_set_size << " " << _root->group_size << std::endl;
      node_to_id[_root] = 0;

      // save the other nodes
      IdxType id = 1;
      for (const auto &nodes : _label_to_nodes)
         for (const auto &node : nodes)
         {
            out << id << " " << node->label << " " << node->group_id << " "
                << node->label_set_size << " " << node->group_size << std::endl;
            node_to_id[node] = id;
            ++id;
         }

      // save the parent of each node
      for (const auto &each : node_to_id)
      {
         if (each.first == _root)
            out << each.second << " 0" << std::endl;
         else
            out << each.second << " " << node_to_id[each.first->parent] << std::endl;
      }

      // save the children of each node
      for (const auto &each : node_to_id)
      {
         out << each.second << " " << each.first->children.size() << " ";
         for (const auto &child : each.first->children)
            out << child.first << " " << node_to_id[child.second] << " ";
         out << std::endl;
      }
   }

   // load the trie tree from a file
   void TrieIndex::load(std::string filename)
   {
      std::ifstream in(filename);
      LabelType label, num_children, label_set_size;
      IdxType id, group_id, group_size, parent_id, child_id;

      // load the max label id and number of nodes
      in >> _max_label_id;
      IdxType num_nodes;
      in >> num_nodes;

      // load the nodes
      std::vector<std::shared_ptr<TrieNode>> nodes(num_nodes);
      for (IdxType i = 0; i < num_nodes; ++i)
      {
         in >> id >> label >> group_id >> label_set_size >> group_size;
         nodes[id] = std::make_shared<TrieNode>(label, group_id, label_set_size, group_size);
      }
      _root = nodes[0];

      // load the parent of each node
      for (IdxType i = 0; i < num_nodes; ++i)
      {
         in >> id >> parent_id;
         if (id > 0)
            nodes[id]->parent = nodes[parent_id];
      }
      _root->parent = nullptr;

      // load the children of each node
      for (IdxType i = 0; i < num_nodes; ++i)
      {
         in >> id >> num_children;
         for (IdxType j = 0; j < num_children; ++j)
         {
            in >> label >> child_id;
            nodes[id]->children[label] = nodes[child_id];
         }
      }

      // build label_to_nodes
      _label_to_nodes.resize(_max_label_id + 1);
      for (const auto each : nodes)
         _label_to_nodes[each->label].push_back(each);
   }

   float TrieIndex::get_index_size()
   {
      float index_size = 0;
      for (const auto &nodes : _label_to_nodes)
      {
         index_size += nodes.size() * (sizeof(TrieNode) + sizeof(std::shared_ptr<TrieNode>));
         for (const auto &node : nodes)
            index_size += node->children.size() * (sizeof(LabelType) + sizeof(std::shared_ptr<TrieNode>));
      }
      return index_size;
   }

   // fxy_add
   TrieStaticMetrics TrieIndex::calculate_static_metrics() const
   {
      TrieStaticMetrics metrics;
      if (_root == nullptr)
      {
         return metrics; // Return empty metrics if Trie is not built
      }

      // --- Use a queue for level-order traversal to visit all nodes ---
      std::queue<std::shared_ptr<TrieNode>> q;
      q.push(_root);
      std::unordered_set<std::shared_ptr<TrieNode>> visited;
      visited.insert(_root);

      // --- Variables for calculation ---
      double total_path_length_sum = 0;
      size_t path_count = 0; // Equivalent to num_groups
      double total_branching_factor_sum = 0;
      size_t non_leaf_node_count = 0;

      while (!q.empty())
      {
         std::shared_ptr<TrieNode> current = q.front();
         q.pop();

         metrics.total_nodes++;

         // --- Metric: Average Path Length ---
         if (current->group_id > 0)
         {
            total_path_length_sum += current->label_set_size;
            path_count++;
         }

         // --- Metric: Average Branching Factor ---
         if (!current->children.empty())
         {
            total_branching_factor_sum += current->children.size();
            non_leaf_node_count++;
         }

         // --- Enqueue children for traversal ---
         for (const auto &child_pair : current->children)
         {
            if (visited.find(child_pair.second) == visited.end())
            {
               q.push(child_pair.second);
               visited.insert(child_pair.second);
            }
         }
      }

      // --- Final Calculations ---
      metrics.label_cardinality = _label_to_nodes.size();
      if (path_count > 0)
      {
         metrics.avg_path_length = total_path_length_sum / path_count;
      }
      if (non_leaf_node_count > 0)
      {
         metrics.avg_branching_factor = total_branching_factor_sum / non_leaf_node_count;
      }

      // --- Metric: Label Frequency Distribution ---
      for (ANNS::LabelType label = 0; label < _label_to_nodes.size(); ++label)
      {
         metrics.label_frequency[label] = _label_to_nodes[label].size();
      }

      return metrics;
   }

}
