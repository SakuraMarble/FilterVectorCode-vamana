#ifndef TRIE_TREE_H
#define TRIE_TREE_H

#include <vector>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include "config.h"

namespace ANNS
{

   // fxy_add:用于方法一 (Shortcut) 的指标传递
   struct TrieMethod1Metrics
   {
      size_t initial_candidates = 0;
      size_t successful_checks = 0;

      long long upward_traversals;   // 向上回溯的节点数
      long long bfs_nodes_processed; // 向下BFS的节点数

      long long redundant_upward_steps = 0; // 向上回溯过程中重复的节点
   };

   // fxy_add:用于从方法二(递归法)中收集详细性能指标的结构体
   struct TrieSearchMetricsRecursive
   {
      // --- 递归搜索阶段 (DFS) ---
      long long recursive_calls = 0; // 递归函数被调用的总次数
      long long pruning_events = 0;  // 关键指标：剪枝发生的次数
      int max_recursion_depth = 0;   // 到达过的最大递归深度

      // --- 结果收集阶段 (BFS) ---
      long long collection_calls = 0;       // collect_all_terminals被调用的次数
      long long nodes_processed_in_bfs = 0; // 在所有收集中，BFS处理的总节点数
      double time_in_collection_bfs = 0.0;  // 在所有收集中，BFS花费的总时间
   };

   // fxy_add:Add a struct to hold the calculated metrics
   struct TrieStaticMetrics
   {
      size_t label_cardinality = 0;
      size_t total_nodes = 0;
      float avg_path_length = 0.0;
      float avg_branching_factor = 0.0;
      std::map<ANNS::LabelType, size_t> label_frequency;
   };

   // trie tree node
   struct TrieNode
   {
      LabelType label;
      IdxType group_id;         // group_id>0, and 0 if not a terminal node
      LabelType label_set_size; // number of elements in the label set if it is a terminal node
      IdxType group_size;       // number of elements in the group if it is a terminal node

      std::shared_ptr<TrieNode> parent;
      // std::map<LabelType, std::shared_ptr<TrieNode>> children;
      std::unordered_map<LabelType, std::shared_ptr<TrieNode>> children;

      TrieNode(LabelType x, std::shared_ptr<TrieNode> y)
          : label(x), parent(y), group_id(0), label_set_size(0), group_size(0) {}
      TrieNode(LabelType a, IdxType b, LabelType c, IdxType d)
          : label(a), group_id(b), label_set_size(c), group_size(d) {}
      ~TrieNode() = default;
   };

   // trie tree construction and search for super sets
   class TrieIndex
   {

   public:
      TrieIndex();

      // construction
      IdxType insert(const std::vector<LabelType> &label_set, IdxType &new_label_set_id);

      // query
      LabelType get_max_label_id() const { return _max_label_id; }
      std::shared_ptr<TrieNode> find_exact_match(const std::vector<LabelType> &label_set) const;
      void get_super_set_entrances(const std::vector<LabelType> &label_set,
                                   std::vector<std::shared_ptr<TrieNode>> &super_set_entrances,
                                   bool avoid_self = false, bool need_containment = true) const;
      void get_super_set_entrances_debug(const std::vector<LabelType> &label_set,
                                         std::vector<std::shared_ptr<TrieNode>> &super_set_entrances,
                                         bool avoid_self, bool need_containment,
                                         std::atomic<int> &print_counter, TrieMethod1Metrics &metrics) const;
      void get_super_set_entrances_new_debug(const std::vector<LabelType> &label_set,
                                             std::vector<std::shared_ptr<TrieNode>> &super_set_entrances,
                                             bool avoid_self, bool need_containment,
                                             std::atomic<int> &print_counter, TrieSearchMetricsRecursive &metrics) const;
      void get_super_set_entrances_new_more_sp_debug(const std::vector<LabelType> &label_set,
                                                     std::vector<std::shared_ptr<TrieNode>> &super_set_entrances,
                                                     bool avoid_self, bool need_containment,
                                                     std::atomic<int> &print_counter, TrieSearchMetricsRecursive &metrics) const;

      // fxy_add
      size_t get_candidate_count_for_label(LabelType label) const
      {
         // 步骤1: 检查标签ID是否在 _label_to_nodes 向量的有效范围内
         if (label >= _label_to_nodes.size())
         {
            return 0; // 标签越界，不可能有对应的候选集
         }
         // 步骤2: 直接通过索引访问并返回内部向量的大小
         return _label_to_nodes[label].size();
      }
      // fxy_add:计算并返回Trie树的静态结构指标
      TrieStaticMetrics calculate_static_metrics() const;

      // I/O
      void save(std::string filename) const;
      void load(std::string filename);
      float get_index_size();

   private:
      LabelType _max_label_id = 0;
      std::shared_ptr<TrieNode> _root;
      std::vector<std::vector<std::shared_ptr<TrieNode>>> _label_to_nodes;

      // help function for get_super_set_entrances
      bool examine_smallest(const std::vector<LabelType> &label_set, const std::shared_ptr<TrieNode> &node) const;
      bool examine_containment(const std::vector<LabelType> &label_set, const std::shared_ptr<TrieNode> &node) const;
      // bool examine_containment_debug(const std::vector<LabelType> &label_set, const std::shared_ptr<TrieNode> &node, long long &nodes_traversed) const;
      bool examine_containment_debug(const std::vector<LabelType> &label_set,
                                     const std::shared_ptr<TrieNode> &node,
                                     long long &nodes_traversed,
                                     std::unordered_set<std::shared_ptr<TrieNode>> &visited_upward,
                                     long long &redundant_steps) const;
      void find_supersets_recursive_debug(
          std::shared_ptr<TrieNode> current_node,
          const std::vector<LabelType> &sorted_query,
          size_t query_idx,
          std::vector<std::shared_ptr<TrieNode>> &results,
          std::set<IdxType> &visited_groups,
          const std::shared_ptr<TrieNode> &avoided_node,
          TrieSearchMetricsRecursive &metrics,
          int current_depth) const;

      void collect_all_terminals_debug(
          std::shared_ptr<TrieNode> start_node,
          std::vector<std::shared_ptr<TrieNode>> &results,
          std::set<IdxType> &visited_groups,
          const std::shared_ptr<TrieNode> &avoided_node,
          TrieSearchMetricsRecursive &metrics) const;
   };
}

#endif // TRIE_TREE_H