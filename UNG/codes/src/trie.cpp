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
                                                 std::atomic<int> &print_counter) const // 1. 添加 print_counter 参数
   {
      super_set_entrances.clear();

      // --- Metrics & Timers Initialization ---
      auto function_start_time = std::chrono::high_resolution_clock::now();
      double time_candidate_gen = 0.0;
      double time_bfs = 0.0;
      long long initial_candidates_from_map = 0;
      long long examine_containment_calls = 0;
      long long successful_containment_checks = 0;
      long long bfs_nodes_processed = 0;

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
               if (examine_containment(label_set, node))
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

      // --- 4. 使用原子计数器控制打印次数 ---
      if (print_counter.fetch_add(1, std::memory_order_relaxed) < 10) // 2. 使用原子计数器
      {
#pragma omp critical
         {
            std::cout << "\n--- Method 1 (Shortcut) Performance Analysis ---\n"
                      << "Query: " << "size=" << label_set.size() << ", last_label=" << (label_set.empty() ? -1 : label_set.back()) << "\n"
                      << "--- Phase 1: Candidate Generation & Verification ---\n"
                      << "  - Initial candidates from map: " << initial_candidates_from_map << "\n"
                      << "  - `examine_containment` calls: " << examine_containment_calls << "\n"
                      << "  - Successful checks (queue size): " << successful_containment_checks << "\n"
                      << "  - Time for this phase: " << time_candidate_gen << " ms\n"
                      << "--- Phase 2: Downward BFS ---\n"
                      << "  - Nodes processed in BFS: " << bfs_nodes_processed << "\n"
                      << "  - Time for this phase: " << time_bfs << " ms\n"
                      << "--- Summary ---\n"
                      << "  - Total super sets found: " << super_set_entrances.size() << "\n"
                      << "  - Total function time: " << std::chrono::duration<double, std::milli>(end_bfs - function_start_time).count() << " ms\n"
                      << "--------------------------------------------------\n"
                      << std::endl;
         }
      }
   }

   // fxy_add:方法二新的主入口函数，负责初始化并启动整个超集搜索过程
   void TrieIndex::get_super_set_entrances_new(const std::vector<LabelType> &label_set,
                                               std::vector<std::shared_ptr<TrieNode>> &super_set_entrances,
                                               bool avoid_self, bool need_containment) const
   {
      // 清空上一次的搜索结果
      super_set_entrances.clear();
      if (!need_containment)
      {
         return;
      }

      // 如果需要排除查询标签集自身，则先找到该标签集对应的Trie节点
      std::shared_ptr<TrieNode> avoided_node = nullptr;
      if (avoid_self)
      {
         avoided_node = find_exact_match(label_set);
      }

      std::set<IdxType> visited_groups; // 使用 std::set 来防止重复添加同一个 group_id 的节点
      // 从Trie树的根节点开始，启动核心的递归搜索过程(假设 label_set 已被排序)
      find_supersets_recursive(_root, label_set, 0, super_set_entrances, visited_groups, avoided_node);
   }

   // fxy_add: 方法二新的主入口函数调试输出版本
   void TrieIndex::get_super_set_entrances_new_debug(const std::vector<LabelType> &label_set,
                                                     std::vector<std::shared_ptr<TrieNode>> &super_set_entrances,
                                                     bool avoid_self, bool need_containment,
                                                     std::atomic<int> &print_counter) const
   {
      super_set_entrances.clear();
      if (!need_containment)
         return;

      // --- 初始化 ---
      auto function_start_time = std::chrono::high_resolution_clock::now();
      TrieSearchMetricsRecursive metrics; // 创建指标对象

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
            std::cout << "\n--- Method 2 (Recursive) Performance Analysis ---\n"
                      << "Query: size=" << label_set.size() << "\n"
                      << "--- Phase 1: Recursive Search (DFS) ---\n"
                      << "  - Recursive calls: " << metrics.recursive_calls << "\n"
                      << "  - Pruning events (IMPORTANT): " << metrics.pruning_events << "\n"
                      << "  - Max recursion depth: " << metrics.max_recursion_depth << "\n"
                      << "--- Phase 2: Result Collection (BFS) ---\n"
                      << "  - Collection function calls: " << metrics.collection_calls << "\n"
                      << "  - Nodes processed in all BFS: " << metrics.nodes_processed_in_bfs << "\n"
                      << "  - Time spent in all BFS: " << metrics.time_in_collection_bfs << " ms\n"
                      << "--- Summary ---\n"
                      << "  - Total super sets found: " << super_set_entrances.size() << "\n"
                      << "  - Total function time: " << total_time << " ms\n"
                      << "---------------------------------------------------\n"
                      << std::endl;
         }
      }
   }

   // fxy_add:参与get_super_set_entrances_new的辅助函数：从一个起始节点开始，收集其子树中所有的终端节点
   void TrieIndex::collect_all_terminals(
       std::shared_ptr<TrieNode> start_node,
       std::vector<std::shared_ptr<TrieNode>> &results,
       std::set<IdxType> &visited_groups,
       const std::shared_ptr<TrieNode> &avoided_node) const
   {
      // 使用队列进行广度优先搜索 (BFS)
      std::queue<std::shared_ptr<TrieNode>> q;
      if (start_node)
         q.push(start_node);
      while (!q.empty())
      {
         auto cur = q.front();
         q.pop();
         // 检查当前节点是否是一个有效的、未被访问过的、非排除的终端节点
         if (cur->group_id > 0 && cur != avoided_node && visited_groups.find(cur->group_id) == visited_groups.end())
         {
            // 如果是，则加入结果集，并标记为已访问
            visited_groups.insert(cur->group_id);
            results.push_back(cur);
         }
         // 将所有子节点加入队列，继续搜索子树的下一层
         for (const auto &child_pair : cur->children)
         {
            q.push(child_pair.second);
         }
      }
   }

   // fxy_add:参与get_super_set_entrances_new的递归函数,在Trie树中智能地查找所有超集
   void TrieIndex::find_supersets_recursive(
       std::shared_ptr<TrieNode> current_node,
       const std::vector<LabelType> &sorted_query, // 必须是已排序的查询标签
       size_t query_idx,                           // 当前正在匹配的查询标签的索引
       std::vector<std::shared_ptr<TrieNode>> &results,
       std::set<IdxType> &visited_groups,
       const std::shared_ptr<TrieNode> &avoided_node) const
   {
      // === 递归终止条件 ===
      // 如果查询中的所有标签都已成功匹配 (query_idx 到达末尾)
      if (query_idx == sorted_query.size())
      { // 那么当前节点之下的所有终端节点都是合法的超集,调用辅助函数将它们全部收集起来
         collect_all_terminals(current_node, results, visited_groups, avoided_node);
         return;
      }

      // 获取当前需要匹配的目标标签
      LabelType target_label = sorted_query[query_idx];

      // === 递归递进逻辑 ===
      // 遍历当前节点的所有子节点（Trie树保证子节点按label值有序）
      for (const auto &child_pair : current_node->children)
      {
         LabelType child_label = child_pair.first;
         auto child_node = child_pair.second;

         if (child_label < target_label)
         {
            // 情况1: 子标签小于目标。路径仍有可能构成超集。
            // 例如，查询{10, 20}，当前在路径{5}，仍需在{5}的子树里找{10, 20}。
            // 递归进入子节点，但 query_idx 不变，因为我们还在找同一个 target_label。
            find_supersets_recursive(child_node, sorted_query, query_idx, results, visited_groups, avoided_node);
         }
         else if (child_label == target_label)
         {
            // 情况2: 子标签等于目标。成功匹配一个！
            // 递归进入子节点，并将 query_idx 加 1，去匹配查询中的下一个标签。
            find_supersets_recursive(child_node, sorted_query, query_idx + 1, results, visited_groups, avoided_node);
         }
         // 当 child_label > target_label 时，【什么都不做】，直接跳过这个分支。这是最关键的剪枝
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
}
