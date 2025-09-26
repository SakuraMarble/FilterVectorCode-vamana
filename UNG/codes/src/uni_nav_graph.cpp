#include <omp.h>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <boost/filesystem.hpp>
#include <vector>
#include <queue>
#include <stack>
#include <bitset>

#include <random>
#include <fstream>
#include <algorithm>
#include <unordered_set>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <filesystem>
#include <bitset>
#include <boost/dynamic_bitset.hpp>
#include <iomanip>
#include <atomic> // 引入原子操作头文件

#include "utils.h"
#include "vamana/vamana.h"
#include "include/uni_nav_graph.h"
#include <roaring/roaring.h>
#include <roaring/roaring.hh>

namespace fs = boost::filesystem;
using BitsetType = boost::dynamic_bitset<>;

// 文件格式常量
const std::string FVEC_EXT = ".fvecs";
const std::string TXT_EXT = ".txt";
const size_t FVEC_HEADER_SIZE = sizeof(uint32_t);
struct QueryTask
{
   ANNS::IdxType vec_id;         // 向量ID
   std::vector<uint32_t> labels; // 查询标签集
};

struct TreeInfo
{
   ANNS::IdxType root_group_id;
   ANNS::LabelType root_label_id; // 存储概念上的根标签ID
   size_t coverage_count;
   // 用于按覆盖率降序排序
   bool operator<(const TreeInfo &other) const
   {
      return coverage_count > other.coverage_count;
   }
};

namespace ANNS
{

   void UniNavGraph::build(std::shared_ptr<IStorage> base_storage, std::shared_ptr<DistanceHandler> distance_handler,
                           std::string scenario, std::string index_name, uint32_t num_threads, IdxType num_cross_edges,
                           IdxType max_degree, IdxType Lbuild, float alpha, std::string dataset,
                           ANNS::AcornInUng new_cross_edge)
   {
      auto all_start_time = std::chrono::high_resolution_clock::now();
      _base_storage = base_storage;
      _num_points = base_storage->get_num_points();
      _distance_handler = distance_handler;
      std::cout << "- Scenario: " << scenario << std::endl;

      // index parameters
      _index_name = index_name;
      _num_cross_edges = num_cross_edges;
      _max_degree = max_degree;
      _Lbuild = Lbuild;
      _alpha = alpha;
      _num_threads = num_threads;
      _scenario = scenario;

      std::cout << "Dividing groups and building the trie tree index ..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      build_trie_and_divide_groups();
      _graph = std::make_shared<ANNS::Graph>(base_storage->get_num_points());
      _global_graph = std::make_shared<ANNS::Graph>(base_storage->get_num_points());
      std::cout << "begin prepare_group_storages_graphs" << std::endl;
      prepare_group_storages_graphs();
      _label_processing_time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
      std::cout << "- Finished in " << _label_processing_time << " ms" << std::endl;

      // build graph index for each group
      build_graph_for_all_groups();
      // build_global_vamana_graph();
      build_vector_and_attr_graph(); // fxy_add

      // for label equality scenario, there is no need for label navigating graph and cross-group edges
      if (_scenario == "equality")
      {
         add_offset_for_uni_nav_graph();
      }
      else
      {

         // build the label navigating graph
         build_label_nav_graph();
         get_descendants_info(); // fxy_add

         // calculate the coverage ratio
         cal_f_coverage_ratio(); // fxy_add

         // initialize_lng_descendants_coverage_bitsets();
         initialize_roaring_bitsets();

         // precompute the node depths in the label navigating graph,used in hard sandwitch
         // _precompute_lng_node_depths();

         if (!new_cross_edge.ung_and_acorn)
            build_cross_group_edges();
         else
         {
            _num_cross_edges = _num_cross_edges / 2;
            finalize_intra_group_graphs(); // fxy_add:从 build_cross_group_edges中拎出转换id新旧转换的部分

            add_new_distance_oriented_edges(
                dataset,
                num_threads,
                new_cross_edge);
         }
      }

      // index time
      _index_time = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - all_start_time)
                        .count();
   }

   void UniNavGraph::build_trie_and_divide_groups()
   {
      // create groups for base label sets
      IdxType new_group_id = 1;
      for (auto vec_id = 0; vec_id < _num_points; ++vec_id)
      {
         const auto &label_set = _base_storage->get_label_set(vec_id);
         auto group_id = _trie_index.insert(label_set, new_group_id);

         // deal with new label setinver
         if (group_id + 1 > _group_id_to_vec_ids.size())
         {
            _group_id_to_vec_ids.resize(group_id + 1);
            _group_id_to_label_set.resize(group_id + 1);
            _group_id_to_label_set[group_id] = label_set;
         }
         _group_id_to_vec_ids[group_id].emplace_back(vec_id);
      }

      // logs
      _num_groups = new_group_id - 1;
      std::cout << "- Number of groups: " << _num_groups << std::endl;
   }

   void UniNavGraph::get_min_super_sets(const std::vector<LabelType> &query_label_set, std::vector<IdxType> &min_super_set_ids,
                                        bool avoid_self, bool need_containment)
   {
      min_super_set_ids.clear();

      // obtain the candidates
      std::vector<std::shared_ptr<TrieNode>> candidates;
      _trie_index.get_super_set_entrances(query_label_set, candidates, avoid_self, need_containment); // 搜索候选超集

      // special cases
      if (candidates.empty())
         return;
      if (candidates.size() == 1)
      {
         min_super_set_ids.emplace_back(candidates[0]->group_id);
         return;
      }

      // obtain the minimum size
      // 所有candidates 按照其标签集的尺寸 label_set_size 从小到大排序
      std::sort(candidates.begin(), candidates.end(),
                [](const std::shared_ptr<TrieNode> &a, const std::shared_ptr<TrieNode> &b)
                {
                   return a->label_set_size < b->label_set_size;
                });
      auto min_size = _group_id_to_label_set[candidates[0]->group_id].size();

      // get the minimum super sets
      for (auto candidate : candidates)
      {
         const auto &cur_group_id = candidate->group_id;
         const auto &cur_label_set = _group_id_to_label_set[cur_group_id];
         bool is_min = true;

         // check whether contains existing minimum super sets (label ids are in ascending order)
         if (cur_label_set.size() > min_size)
         {
            for (auto min_group_id : min_super_set_ids)
            {
               const auto &min_label_set = _group_id_to_label_set[min_group_id];
               if (std::includes(cur_label_set.begin(), cur_label_set.end(), min_label_set.begin(), min_label_set.end()))
               {
                  is_min = false;
                  break;
               }
            }
         }

         // add to the minimum super sets
         if (is_min)
            min_super_set_ids.emplace_back(cur_group_id);
      }
   }

   void UniNavGraph::get_min_super_sets_debug(const std::vector<LabelType> &query_label_set,
                                              std::vector<IdxType> &min_super_set_ids,
                                              bool avoid_self, bool need_containment,
                                              std::atomic<int> &print_counter, bool is_new_trie_method, bool is_rec_more_start, QueryStats &stats)
   {
      // --- 计时器变量定义 ---
      double time_trie_lookup = 0.0;
      double time_sorting = 0.0;
      double time_filtering_loop = 0.0;

      auto function_start_time = std::chrono::high_resolution_clock::now();

      min_super_set_ids.clear();

      // --- 1. 测量Trie查找候选者的时间 ---
      auto start_trie = std::chrono::high_resolution_clock::now();
      std::vector<std::shared_ptr<TrieNode>> candidates;

      if (is_new_trie_method)
      {
         // --- 调用方法二 (Recursive) ---
         // 1. 创建一个用于接收底层指标的结构体实例
         TrieSearchMetricsRecursive trie_metrics_m2 = {};

         // 2. 调用修改后的 TrieIndex 方法，将结构体传入
         if (is_rec_more_start)
            _trie_index.get_super_set_entrances_new_more_sp_debug(query_label_set, candidates, avoid_self, need_containment, print_counter, trie_metrics_m2);
         else
            _trie_index.get_super_set_entrances_new_debug(query_label_set, candidates, avoid_self, need_containment, print_counter, trie_metrics_m2);

         // 3. 将从底层获取的指标填充到高层的 QueryStats 中
         stats.recursive_calls = trie_metrics_m2.recursive_calls;
         stats.pruning_events = trie_metrics_m2.pruning_events;
         stats.trie_nodes_traversed = trie_metrics_m2.recursive_calls + trie_metrics_m2.nodes_processed_in_bfs;

         // 4. 计算并填充衍生指标：剪枝效率
         if (stats.recursive_calls > 0)
         {
            stats.pruning_efficiency = static_cast<float>(stats.pruning_events) / stats.recursive_calls;
         }
         else
         {
            stats.pruning_efficiency = 0.0f;
         }
      }
      else
      {
         // --- 调用方法一 (Shortcut) ---
         // 1. 创建一个用于接收底层指标的结构体实例
         TrieMethod1Metrics trie_metrics_m1 = {};

         // 2. 调用修改后的 TrieIndex 方法，将结构体传入
         _trie_index.get_super_set_entrances_debug(query_label_set, candidates, avoid_self, need_containment, print_counter, trie_metrics_m1);

         // 3. 将从底层获取的指标填充到高层的 QueryStats 中
         stats.successful_checks = trie_metrics_m1.successful_checks;
         stats.trie_nodes_traversed = trie_metrics_m1.upward_traversals + trie_metrics_m1.bfs_nodes_processed;
         stats.redundant_upward_steps = trie_metrics_m1.redundant_upward_steps;

         // 4. 计算并填充衍生指标：捷径命中率
         if (stats.candidate_set_size > 0)
         {
            stats.shortcut_hit_ratio = static_cast<float>(stats.successful_checks) / stats.candidate_set_size;
         }
         else
         {
            stats.shortcut_hit_ratio = 0.0f;
         }
      }

      auto end_trie = std::chrono::high_resolution_clock::now();
      time_trie_lookup = std::chrono::duration<double, std::milli>(end_trie - start_trie).count();

      // 特殊情况处理
      if (candidates.empty())
      {
         std::cout << "No candidates found for the given query label set." << std::endl;
         return;
      }
      if (candidates.size() == 1)
      {
         min_super_set_ids.emplace_back(candidates[0]->group_id);
         // std::cout<<"group_id_to_label_set: "<<std::endl;
         // for(auto label_id: _group_id_to_label_set[candidates[0]->group_id]){
         //    std::cout<<label_id<<" ";
         // }
         // std::cout<<std::endl;
         return;
      }

      // --- 2. 测量排序时间 ---
      auto start_sort = std::chrono::high_resolution_clock::now();
      std::sort(candidates.begin(), candidates.end(),
                [](const std::shared_ptr<TrieNode> &a, const std::shared_ptr<TrieNode> &b)
                {
                   return a->label_set_size < b->label_set_size;
                });
      auto end_sort = std::chrono::high_resolution_clock::now();
      time_sorting = std::chrono::duration<double, std::milli>(end_sort - start_sort).count();

      auto min_size = _group_id_to_label_set[candidates[0]->group_id].size();

      // --- 3. 测量过滤循环的时间 ---
      auto start_filter = std::chrono::high_resolution_clock::now();
      for (auto candidate : candidates)
      {
         const auto &cur_group_id = candidate->group_id;
         const auto &cur_label_set = _group_id_to_label_set[cur_group_id];
         bool is_min = true;

         if (cur_label_set.size() > min_size)
         {
            for (auto min_group_id : min_super_set_ids)
            {
               const auto &min_label_set = _group_id_to_label_set[min_group_id];
               if (std::includes(cur_label_set.begin(), cur_label_set.end(), min_label_set.begin(), min_label_set.end()))
               {
                  is_min = false;
                  break;
               }
            }
         }

         if (is_min)
         {
            min_super_set_ids.emplace_back(cur_group_id);
         }
      }
      auto end_filter = std::chrono::high_resolution_clock::now();
      time_filtering_loop = std::chrono::duration<double, std::milli>(end_filter - start_filter).count();

      auto function_total_time = std::chrono::duration<double, std::milli>(end_filter - function_start_time).count();
   }

   // 为了方便排序，定义一个结构体来存储候选组及其评分
   struct ScoredCandidate
   {
      double score;
      ANNS::IdxType group_id;

      // 重载小于运算符，方便使用 std::sort 进行降序排序
      bool operator<(const ScoredCandidate &other) const
      {
         return score > other.score; // score 越大越靠前
      }
   };

   std::vector<ANNS::IdxType> UniNavGraph::select_entry_groups(
       const std::vector<ANNS::IdxType> &minimum_entry_sets,
       ANNS::SelectionMode mode,
       size_t top_k,
       double beta,
       ANNS::IdxType true_query_group_id) const
   {
      bool verbose = false;
      assert(_label_nav_graph != nullptr && "Error: _label_nav_graph should not be null.");

      if (top_k == 0 && true_query_group_id == 0)
      {
         return minimum_entry_sets;
      }
      if (minimum_entry_sets.empty() && true_query_group_id == 0)
      {
         return {};
      }

      // --- 步骤 1: 运行BFS，并收集所有可达的候选节点 ---
      std::queue<ANNS::IdxType> q;
      std::vector<int> lng_distance(_num_groups + 1, -1);

      // === 核心优化 1: 创建一个容器，仅用于存储BFS访问过的、距离大于0的节点 ===
      std::vector<ANNS::IdxType> reachable_nodes;
      reachable_nodes.reserve(_num_groups / 10); // 预分配一些内存，避免多次重分配

      for (const auto &group_id : minimum_entry_sets)
      {
         if (group_id > 0 && group_id <= _num_groups && lng_distance[group_id] == -1)
         {
            q.push(group_id);
            lng_distance[group_id] = 0;
         }
      }

      while (!q.empty())
      {
         ANNS::IdxType current_group = q.front();
         q.pop();

         if (current_group >= _label_nav_graph->out_neighbors.size())
            continue;

         for (const auto &child_group : _label_nav_graph->out_neighbors[current_group])
         {
            if (child_group > 0 && child_group <= _num_groups && lng_distance[child_group] == -1)
            {
               lng_distance[child_group] = lng_distance[current_group] + 1;
               q.push(child_group);
               // === 核心优化 2: 将新发现的候选节点存入列表 ===
               reachable_nodes.push_back(child_group);
            }
         }
      }

      // --- 步骤 2: 仅对可达节点进行评分 ---
      std::vector<ANNS::ScoredCandidate> candidates;
      candidates.reserve(reachable_nodes.size());

      // === 核心优化 3: 直接遍历数量少得多的 reachable_nodes 列表 ===
      for (const auto group_id : reachable_nodes)
      {
         const double group_size = static_cast<double>(_group_id_to_vec_ids[group_id].size());
         double score = 0.0;
         if (mode == ANNS::SelectionMode::SizeOnly)
         {
            score = group_size;
         }
         else
         { // SizeAndDistance with new formula
            const int dist = lng_distance[group_id];
            // 新评分公式: 深度优先，大小作为次要排序依据
            score = static_cast<double>(dist) * static_cast<double>(_num_points) + group_size;
         }
         candidates.push_back({score, group_id});
      }

      // --- 步骤 3: 部分排序并选择 ---
      size_t num_to_sort = std::min(candidates.size(), top_k + 5); // +5 作为安全缓冲
      if (num_to_sort > 0)
      {
         std::partial_sort(candidates.begin(),
                           candidates.begin() + num_to_sort,
                           candidates.end());
      }

      std::vector<ANNS::IdxType> final_entry_groups = minimum_entry_sets;
      std::unordered_set<ANNS::IdxType> existing_groups(minimum_entry_sets.begin(), minimum_entry_sets.end());

      bool oracle_injected = false;
      bool oracle_existed = false;

      if (true_query_group_id > 0 && true_query_group_id <= _num_groups)
      {
         if (existing_groups.find(true_query_group_id) == existing_groups.end())
         {
            final_entry_groups.push_back(true_query_group_id);
            existing_groups.insert(true_query_group_id);
            oracle_injected = true;
         }
         else
         {
            oracle_existed = true;
         }
      }

      for (size_t i = 0; i < num_to_sort; ++i)
      {
         const auto &candidate = candidates[i];
         if (final_entry_groups.size() >= minimum_entry_sets.size() + top_k)
         {
            break;
         }
         if (existing_groups.find(candidate.group_id) == existing_groups.end())
         {
            final_entry_groups.push_back(candidate.group_id);
            existing_groups.insert(candidate.group_id);
         }
      }

      // --- 精简后的日志输出 ---
      if (verbose)
      {
         if (oracle_injected)
         {
            std::cout << "[Info] Oracle group " << true_query_group_id << " was successfully injected." << std::endl;
         }
         else if (oracle_existed && true_query_group_id > 0)
         {
            std::cout << "[Info] Oracle group " << true_query_group_id << " was already in the initial set." << std::endl;
         }

         size_t added_count = final_entry_groups.size() > minimum_entry_sets.size() ? final_entry_groups.size() - minimum_entry_sets.size() : 0;
         if (oracle_injected)
            added_count--;

         std::cout << "[Info] Final selected entry groups count: " << final_entry_groups.size()
                   << " (Initial: " << minimum_entry_sets.size()
                   << ", Heuristically Added: " << added_count << ")" << std::endl;
      }

      return final_entry_groups;
   }

   void UniNavGraph::prepare_group_storages_graphs()
   {
      _new_vec_id_to_group_id.resize(_num_points);

      // reorder the vectors
      _group_id_to_range.resize(_num_groups + 1);
      _new_to_old_vec_ids.resize(_num_points);
      IdxType new_vec_id = 0;
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         _group_id_to_range[group_id].first = new_vec_id;
         for (auto old_vec_id : _group_id_to_vec_ids[group_id])
         {
            _new_to_old_vec_ids[new_vec_id] = old_vec_id;
            _new_vec_id_to_group_id[new_vec_id] = group_id;
            ++new_vec_id;
         }
         _group_id_to_range[group_id].second = new_vec_id;
      }

      // reorder the underlying storage
      _base_storage->reorder_data(_new_to_old_vec_ids);

      // init storage and graph for each group
      _group_storages.resize(_num_groups + 1);
      _group_graphs.resize(_num_groups + 1);
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         auto start = _group_id_to_range[group_id].first;
         auto end = _group_id_to_range[group_id].second;
         _group_storages[group_id] = create_storage(_base_storage, start, end);
         _group_graphs[group_id] = std::make_shared<Graph>(_graph, start, end);
      }
   }

   void UniNavGraph::build_graph_for_all_groups()
   {
      std::cout << "Building graph for each group ..." << std::endl;
      omp_set_num_threads(_num_threads);
      auto start_time = std::chrono::high_resolution_clock::now();

      // build vamana index
      if (_index_name == "Vamana")
      {
         _vamana_instances.resize(_num_groups + 1);
         _group_entry_points.resize(_num_groups + 1);

#pragma omp parallel for schedule(dynamic, 1)
         for (auto group_id = 1; group_id <= _num_groups; ++group_id)
         {
            // if (group_id % 100 == 0)
            //    std::cout << "\r" << (100.0 * group_id) / _num_groups << "%" << std::flush;

            // if there are less than _max_degree points in the group, just build a complete graph
            const auto &range = _group_id_to_range[group_id];
            if (range.second - range.first <= _max_degree)
            {
               build_complete_graph(_group_graphs[group_id], range.second - range.first);
               _vamana_instances[group_id] = std::make_shared<Vamana>(_group_storages[group_id], _distance_handler,
                                                                      _group_graphs[group_id], 0);

               // build the vamana graph
            }
            else
            {
               _vamana_instances[group_id] = std::make_shared<Vamana>(false);
               _vamana_instances[group_id]->build(_group_storages[group_id], _distance_handler,
                                                  _group_graphs[group_id], _max_degree, _Lbuild, _alpha, 1);
            }

            // set entry point
            _group_entry_points[group_id] = _vamana_instances[group_id]->get_entry_point() + range.first;
         }

         // if none of the above
      }
      else
      {
         std::cerr << "Error: invalid index name " << _index_name << std::endl;
         exit(-1);
      }
      _build_graph_time = std::chrono::duration<double, std::milli>(
                              std::chrono::high_resolution_clock::now() - start_time)
                              .count();
      std::cout << "\r- Finished in " << _build_graph_time << " ms" << std::endl;
   }

   // fxy_add：构建全局Vamana图
   void UniNavGraph::build_global_vamana_graph()
   {
      std::cout << "Building global Vamana graph..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      _global_vamana = std::make_shared<Vamana>(false);

      _global_vamana->build(_base_storage, _distance_handler, _global_graph, _max_degree, _Lbuild, _alpha, 1);

      auto build_time = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - start_time)
                            .count();
      _global_vamana_entry_point = _global_vamana->get_entry_point();

      std::cout << "- Global Vamana graph built in " << build_time << " ms" << std::endl;
   }
   //=====================================begin 数据预处理：构建向量-属性二分图=========================================
   // fxy_add: 构建向量-属性二分图
   void UniNavGraph::build_vector_and_attr_graph()
   {
      std::cout << "Building vector-attribute bipartite graph..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      // 初始化属性到ID的映射和反向映射
      _attr_to_id.clear();
      _id_to_attr.clear();
      _vector_attr_graph.clear();

      // 第一遍：收集所有唯一属性并分配ID
      AtrType attr_id = 0;
      for (IdxType vec_id = 0; vec_id < _num_points; ++vec_id)
      {
         const auto &label_set = _base_storage->get_label_set(vec_id);
         for (const auto &label : label_set)
         {
            if (_attr_to_id.find(label) == _attr_to_id.end())
            {
               _attr_to_id[label] = attr_id;
               _id_to_attr[attr_id] = label;
               attr_id++;
            }
         }
      }

      // 初始化图结构（向量节点 + 属性节点）
      _vector_attr_graph.resize(_num_points + static_cast<size_t>(attr_id));

      // 第二遍：构建图结构
      for (IdxType vec_id = 0; vec_id < _num_points; ++vec_id)
      {
         const auto &label_set = _base_storage->get_label_set(vec_id);
         for (const auto &label : label_set)
         {
            AtrType a_id = _attr_to_id[label];

            // 添加双向边
            // 向量节点ID范围: [0, _num_points-1]
            // 属性节点ID范围: [_num_points, _num_points+attr_id-1]
            _vector_attr_graph[vec_id].push_back(_num_points + static_cast<IdxType>(a_id));
            _vector_attr_graph[_num_points + static_cast<IdxType>(a_id)].push_back(vec_id);
         }
      }

      // 统计信息
      _num_attributes = attr_id;
      _build_vector_attr_graph_time = std::chrono::duration<double, std::milli>(
                                          std::chrono::high_resolution_clock::now() - start_time)
                                          .count();
      std::cout << "- Finish in " << _build_vector_attr_graph_time << " ms" << std::endl;
      std::cout << "- Total edges: " << count_graph_edges() << std::endl;

      // 可选：保存图结构供调试
      // save_bipartite_graph_info();
   }

   // fxy_add: 计算向量-属性二分图的边数
   size_t UniNavGraph::count_graph_edges() const
   {
      size_t total_edges = 0;
      for (const auto &neighbors : _vector_attr_graph)
      {
         total_edges += neighbors.size();
      }
      return total_edges / 2; // 因为是双向边，实际边数是总数的一半
   }

   // fxy_add: 保存二分图信息到txt文件调试用
   void UniNavGraph::save_bipartite_graph_info() const
   {
      std::ofstream outfile("bipartite_graph_info.txt");
      if (!outfile.is_open())
      {
         std::cerr << "Warning: Could not open file to save bipartite graph info" << std::endl;
         return;
      }

      outfile << "Bipartite Graph Information\n";
      outfile << "==========================\n";
      outfile << "Total vectors: " << _num_points << "\n";
      outfile << "Total attributes: " << _num_attributes << "\n";
      outfile << "Total edges: " << count_graph_edges() << "\n\n";

      // 输出属性映射
      outfile << "Attribute to ID Mapping:\n";
      for (const auto &pair : _attr_to_id)
      {
         outfile << pair.first << " -> " << static_cast<IdxType>(pair.second) << "\n";
      }
      outfile << "\n";

      // 输出部分图结构示例
      outfile << "Sample Graph Connections (first 10 vectors and attributes):\n";
      outfile << "Vector connections:\n";
      for (IdxType i = 0; i < std::min(_num_points, static_cast<IdxType>(10)); ++i)
      {
         outfile << "Vector " << i << " connects to attributes: ";
         for (auto a_node : _vector_attr_graph[i])
         {
            AtrType attr_id = static_cast<AtrType>(a_node - _num_points);
            outfile << _id_to_attr.at(attr_id) << " ";
         }
         outfile << "\n";
      }

      outfile << "\nAttribute connections:\n";
      for (AtrType i = 0; i < std::min(_num_attributes, static_cast<AtrType>(5)); ++i)
      {
         outfile << "Attribute " << _id_to_attr.at(i) << " connects to vectors: ";
         for (auto v_node : _vector_attr_graph[_num_points + static_cast<IdxType>(i)])
         {
            outfile << v_node << " ";
         }
         outfile << "\n";
      }

      outfile.close();
      std::cout << "- Bipartite graph info saved to bipartite_graph_info.txt" << std::endl;
   }

   uint32_t UniNavGraph::compute_checksum() const
   {
      // 简单的校验和计算示例
      uint32_t sum = 0;
      for (const auto &neighbors : _vector_attr_graph)
      {
         for (IdxType node : neighbors)
         {
            sum ^= (node << (sum % 32));
         }
      }
      return sum;
   }

   // fxy_add: 保存二分图到文件
   void UniNavGraph::save_bipartite_graph(const std::string &filename)
   {
      std::ofstream out(filename, std::ios::binary);
      if (!out)
      {
         throw std::runtime_error("Cannot open file for writing: " + filename);
      }

      // 1. 写入文件头标识和版本
      const char header[8] = {'B', 'I', 'P', 'G', 'R', 'P', 'H', '1'};
      out.write(header, 8);

      // 2. 写入基本元数据
      out.write(reinterpret_cast<const char *>(&_num_points), sizeof(IdxType));
      out.write(reinterpret_cast<const char *>(&_num_attributes), sizeof(AtrType));

      // 3. 写入属性映射表
      // 3.1 先写入条目数量
      uint64_t map_size = _attr_to_id.size();
      out.write(reinterpret_cast<const char *>(&map_size), sizeof(uint64_t));

      // 3.2 写入每个映射条目（LabelType是uint16_t，直接存储）
      for (const auto &[label, id] : _attr_to_id)
      {
         out.write(reinterpret_cast<const char *>(&label), sizeof(LabelType));
         out.write(reinterpret_cast<const char *>(&id), sizeof(AtrType));
      }

      // 4. 写入邻接表数据
      // 4.1 先写入节点总数
      uint64_t total_nodes = _vector_attr_graph.size();
      out.write(reinterpret_cast<const char *>(&total_nodes), sizeof(uint64_t));

      // 4.2 写入每个节点的邻居列表
      for (const auto &neighbors : _vector_attr_graph)
      {
         // 先写入邻居数量
         uint32_t neighbor_count = neighbors.size();
         out.write(reinterpret_cast<const char *>(&neighbor_count), sizeof(uint32_t));

         // 写入邻居ID列表
         if (!neighbors.empty())
         {
            out.write(reinterpret_cast<const char *>(neighbors.data()),
                      neighbors.size() * sizeof(IdxType));
         }
      }

      // 5. 写入文件尾校验和
      uint32_t checksum = compute_checksum();
      out.write(reinterpret_cast<const char *>(&checksum), sizeof(uint32_t));

      std::cout << "Successfully saved bipartite graph to " << filename
                << " (" << out.tellp() << " bytes)" << std::endl;
   }

   // fxy_add: 读取二分图
   void UniNavGraph::load_bipartite_graph(const std::string &filename)
   {
      std::cout << "Loading bipartite graph from " << filename << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      std::ifstream in(filename, std::ios::binary);
      if (!in)
      {
         throw std::runtime_error("Cannot open file for reading: " + filename);
      }

      // 1. 验证文件头
      char header[8];
      in.read(header, 8);
      if (std::string(header, 8) != "BIPGRPH1")
      {
         throw std::runtime_error("Invalid file format");
      }

      // 2. 读取基本元数据
      in.read(reinterpret_cast<char *>(&_num_points), sizeof(IdxType));
      in.read(reinterpret_cast<char *>(&_num_attributes), sizeof(AtrType));

      // 3. 读取属性映射表
      _attr_to_id.clear();
      _id_to_attr.clear();

      // 3.1 读取条目数量
      uint64_t map_size;
      in.read(reinterpret_cast<char *>(&map_size), sizeof(uint64_t));

      // 3.2 读取每个映射条目
      for (uint64_t i = 0; i < map_size; ++i)
      {
         LabelType label;
         AtrType id;

         in.read(reinterpret_cast<char *>(&label), sizeof(LabelType));
         in.read(reinterpret_cast<char *>(&id), sizeof(AtrType));

         _attr_to_id[label] = id;
         _id_to_attr[id] = label;
      }

      // 4. 读取邻接表数据
      _vector_attr_graph.clear();

      // 4.1 读取节点总数
      uint64_t total_nodes;
      in.read(reinterpret_cast<char *>(&total_nodes), sizeof(uint64_t));
      _vector_attr_graph.resize(total_nodes);

      // 4.2 读取每个节点的邻居列表
      for (uint64_t i = 0; i < total_nodes; ++i)
      {
         uint32_t neighbor_count;
         in.read(reinterpret_cast<char *>(&neighbor_count), sizeof(uint32_t));

         _vector_attr_graph[i].resize(neighbor_count);
         if (neighbor_count > 0)
         {
            in.read(reinterpret_cast<char *>(_vector_attr_graph[i].data()),
                    neighbor_count * sizeof(IdxType));
         }
      }

      // 5. 验证校验和
      uint32_t stored_checksum;
      in.read(reinterpret_cast<char *>(&stored_checksum), sizeof(uint32_t));

      uint32_t computed_checksum = compute_checksum();
      if (stored_checksum != computed_checksum)
      {
         throw std::runtime_error("Checksum verification failed");
      }

      std::cout << "- Loaded bipartite graph with " << _num_points << " vectors and "
                << _num_attributes << " attributes in "
                << std::chrono::duration<double, std::milli>(
                       std::chrono::high_resolution_clock::now() - start_time)
                       .count()
                << " ms" << std::endl;
   }

   // fxy_add: 比较两个向量-属性二分图
   bool UniNavGraph::compare_graphs(const ANNS::UniNavGraph &g1, const ANNS::UniNavGraph &g2)
   {
      // 1. 验证基本属性
      if (g1._num_points != g2._num_points)
      {
         std::cerr << "Mismatch in _num_points: "
                   << g1._num_points << " vs " << g2._num_points << std::endl;
         return false;
      }

      if (g1._num_attributes != g2._num_attributes)
      {
         std::cerr << "Mismatch in _num_attributes: "
                   << g1._num_attributes << " vs " << g2._num_attributes << std::endl;
         return false;
      }

      // 2. 验证属性映射
      if (g1._attr_to_id.size() != g2._attr_to_id.size())
      {
         std::cerr << "Mismatch in _attr_to_id size" << std::endl;
         return false;
      }

      for (const auto &[label, id] : g1._attr_to_id)
      {
         auto it = g2._attr_to_id.find(label);
         if (it == g2._attr_to_id.end())
         {
            std::cerr << "Label " << label << " missing in g2" << std::endl;
            return false;
         }
         if (it->second != id)
         {
            std::cerr << "Mismatch ID for label " << label
                      << ": " << id << " vs " << it->second << std::endl;
            return false;
         }
      }

      // 3. 验证反向属性映射
      for (const auto &[id, label] : g1._id_to_attr)
      {
         auto it = g2._id_to_attr.find(id);
         if (it == g2._id_to_attr.end())
         {
            std::cerr << "ID " << id << " missing in g2" << std::endl;
            return false;
         }
         if (it->second != label)
         {
            std::cerr << "Mismatch label for ID " << id
                      << ": " << label << " vs " << it->second << std::endl;
            return false;
         }
      }

      // 4. 验证邻接表
      if (g1._vector_attr_graph.size() != g2._vector_attr_graph.size())
      {
         std::cerr << "Mismatch in graph size" << std::endl;
         return false;
      }

      for (size_t i = 0; i < g1._vector_attr_graph.size(); ++i)
      {
         const auto &neighbors1 = g1._vector_attr_graph[i];
         const auto &neighbors2 = g2._vector_attr_graph[i];

         if (neighbors1.size() != neighbors2.size())
         {
            std::cerr << "Mismatch in neighbors count for node " << i << std::endl;
            return false;
         }

         for (size_t j = 0; j < neighbors1.size(); ++j)
         {
            if (neighbors1[j] != neighbors2[j])
            {
               std::cerr << "Mismatch in neighbor " << j << " for node " << i
                         << ": " << neighbors1[j] << " vs " << neighbors2[j] << std::endl;
               return false;
            }
         }
      }

      std::cout << "Graphs are identical!" << std::endl;

      return true;
   }
   //=====================================end 数据预处理：构建向量-属性二分图=========================================

   //=====================================begein 查询过程：计算bitmap=========================================

   // fxy_add: 构建bitmap
   std::pair<std::bitset<10000001>, double> UniNavGraph::compute_attribute_bitmap(const std::vector<LabelType> &query_attributes) const
   {
      // 1. 初始化全true的bitmap（表示开始时所有点都满足条件）
      std::bitset<10000001> bitmap;
      bitmap.set(); // 设置所有位为1
      double per_query_bitmap_time = 0.0;

      // 2. 处理每个查询属性
      for (LabelType attr_label : query_attributes)
      {
         //  2.1 查找属性ID
         auto it = _attr_to_id.find(attr_label);
         if (it == _attr_to_id.end())
         {
            // 属性不存在，没有任何点能满足所有条件
            return {std::bitset<10000001>(), 0.0};
         }

         // 2.2 获取属性节点ID
         AtrType attr_id = it->second;
         IdxType attr_node_id = _num_points + static_cast<IdxType>(attr_id);

         // 2.3 创建临时bitmap记录当前属性的满足情况
         std::bitset<10000001> temp_bitmap;
         auto start_time = std::chrono::high_resolution_clock::now();
         for (IdxType vec_id : _vector_attr_graph[attr_node_id])
         {
            temp_bitmap.set(vec_id); // 使用set()方法设置对应的位为1
         }

         // 2.4 与主bitmap进行AND操作
         bitmap &= temp_bitmap; // 使用&=操作符进行位与操作

         per_query_bitmap_time += std::chrono::duration<double, std::milli>(
                                      std::chrono::high_resolution_clock::now() - start_time)
                                      .count();
      }

      return {bitmap, per_query_bitmap_time};
   }

   //====================================end 查询过程：计算bitmap=========================================
   void UniNavGraph::build_complete_graph(std::shared_ptr<Graph> graph, IdxType num_points)
   {
      for (auto i = 0; i < num_points; ++i)
         for (auto j = 0; j < num_points; ++j)
            if (i != j)
               graph->neighbors[i].emplace_back(j);
   }

   //=====================================begin LNG中每个f覆盖率计算=========================================
   // fxy_add: 递归打印孩子节点及其覆盖率
   void print_children_recursive(const std::shared_ptr<ANNS::LabelNavGraph> graph, IdxType group_id, std::ofstream &outfile, int indent_level)
   {
      // 根据缩进级别生成前缀空格
      std::string indent(indent_level * 4, ' '); // 每一层缩进4个空格

      // 打印当前节点
      outfile << "\n"
              << indent << "[" << group_id << "] (" << graph->coverage_ratio[group_id] << ")";

      // 如果有子节点，递归打印子节点
      if (!graph->out_neighbors[group_id].empty())
      {
         for (size_t i = 0; i < graph->out_neighbors[group_id].size(); ++i)
         {
            auto child_group_id = graph->out_neighbors[group_id][i];
            print_children_recursive(graph, child_group_id, outfile, indent_level + 1); // 增加缩进级别
         }
      }
   }

   // fxy_add: 输出覆盖率及孩子节点,调用print_children_recursive
   void output_coverage_ratio(const std::shared_ptr<ANNS::LabelNavGraph> _label_nav_graph, IdxType _num_groups, std::ofstream &outfile)
   {
      outfile << "Coverage Ratio for each group\n";
      outfile << "=====================================\n";
      outfile << "Format: [GroupID] -> CoverageRatio \n";
      outfile << "-> [child_GroupID (CoverageRatio)]\n\n";

      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         // 打印当前组的覆盖率
         outfile << "[" << group_id << "] -> " << _label_nav_graph->coverage_ratio[group_id];

         // 打印孩子节点
         if (!_label_nav_graph->out_neighbors[group_id].empty())
         {
            outfile << " ->";
            for (size_t i = 0; i < _label_nav_graph->out_neighbors[group_id].size(); ++i)
            {
               auto child_group_id = _label_nav_graph->out_neighbors[group_id][i];
               print_children_recursive(_label_nav_graph, child_group_id, outfile, 1); // 第一层子节点缩进为1
            }
         }
         outfile << "\n";
      }
   }

   // fxy_add：计算每个label set的向量覆盖比率
   void UniNavGraph::cal_f_coverage_ratio()
   {
      std::cout << "Calculating coverage ratio..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      // Step 0: 初始化covered_sets
      _label_nav_graph->coverage_ratio.clear();
      _label_nav_graph->covered_sets.clear();
      _label_nav_graph->covered_sets.resize(_num_groups + 1);

      // Step 1: 初始化每个 group 的覆盖集合
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         const auto &vec_ids = _group_id_to_vec_ids[group_id];
         if (vec_ids.empty())
            continue;

         _label_nav_graph->covered_sets[group_id].insert(vec_ids.begin(), vec_ids.end());
      }

      // Step 2: 找出所有叶子节点（出度为 0）
      std::queue<IdxType> q;
      std::vector<int> out_degree(_num_groups + 1, 0); // 复制一份出度用于拓扑传播

      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         out_degree[group_id] = _label_nav_graph->out_neighbors[group_id].size();
         if (out_degree[group_id] == 0)
            q.push(group_id);
      }
      std::cout << "- Number of leaf nodes: " << q.size() << std::endl;

      // Step 3: 自底向上合并集合
      while (!q.empty())
      {
         IdxType current = q.front();
         q.pop();

         for (auto parent : _label_nav_graph->in_neighbors[current])
         {
            // 将当前节点的集合合并到父节点中
            _label_nav_graph->covered_sets[parent].insert(
                _label_nav_graph->covered_sets[current].begin(),
                _label_nav_graph->covered_sets[current].end());

            // 减少父节点剩余未处理的子节点数
            out_degree[parent]--;
            if (out_degree[parent] == 0)
            {
               q.push(parent);
            }
         }
      }

      // Step 4: 计算最终覆盖率
      _label_nav_graph->coverage_ratio.resize(_num_groups + 1, 0.0);
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         size_t covered_count = _label_nav_graph->covered_sets[group_id].size();
         _label_nav_graph->coverage_ratio[group_id] = static_cast<double>(covered_count) / _num_points;
      }

      // Step 5: 输出时间
      _cal_coverage_ratio_time = std::chrono::duration<double, std::milli>(
                                     std::chrono::high_resolution_clock::now() - start_time)
                                     .count();
      std::cout << "- Finish in " << _cal_coverage_ratio_time << " ms" << std::endl;
   }
   // =====================================end LNG中每个f覆盖率计算=========================================

   // =====================================begin 计算LNG中后代的个数=========================================
   // fxy_add：使用广度优先搜索（BFS）来确保所有后代都被找到
   void UniNavGraph::get_descendants_info()
   {
      std::cout << "Calculating descendants info (using corrected BFS method)..." << std::endl;
      using PairType = std::pair<IdxType, int>;

      std::vector<PairType> descendants_num(_num_groups + 1);
      std::vector<std::unordered_set<IdxType>> descendants_set(_num_groups + 1);

      auto start_time = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic)
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         std::vector<bool> visited(_num_groups + 1, false);
         std::queue<IdxType> q;
         std::unordered_set<IdxType> temp_set;

         visited[group_id] = true;
         q.push(group_id);

         while (!q.empty())
         {
            IdxType u = q.front();
            q.pop();

            for (const auto &v : _label_nav_graph->out_neighbors[u])
            {
               if (!visited[v])
               {
                  visited[v] = true;
                  temp_set.insert(v);
                  q.push(v);
               }
            }
         }
         descendants_num[group_id] = PairType(group_id, temp_set.size());
         descendants_set[group_id] = std::move(temp_set);
      }

      // 单线程写入类成员变量
      _label_nav_graph->_lng_descendants_num = std::move(descendants_num);
      _label_nav_graph->_lng_descendants = std::move(descendants_set);

      // 计算平均后代个数
      double total_descendants = 0;
      for (const auto &pair : _label_nav_graph->_lng_descendants_num)
      {
         total_descendants += pair.second;
      }
      _label_nav_graph->avg_descendants = _num_groups > 0 ? total_descendants / _num_groups : 0.0;
      _cal_descendants_time = std::chrono::duration<double, std::milli>(
                                  std::chrono::high_resolution_clock::now() - start_time)
                                  .count();
      std::cout << "- Finish in " << _cal_descendants_time << " ms" << std::endl;
      std::cout << "- Number of groups: " << _num_groups << std::endl;
      std::cout << "- Average number of descendants per group: " << _label_nav_graph->avg_descendants << std::endl;
   }

   // =====================================end 计算LNG中后代的个数=========================================

   // =====================================begin 添加新的跨组边=========================================
   void UniNavGraph::finalize_intra_group_graphs()
   {
      std::cout << "Finalizing intra-group graphs by converting neighbor IDs to global..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      add_offset_for_uni_nav_graph();
      auto duration = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
      std::cout << "- Finished in " << duration << " ms" << std::endl;
   }

   void UniNavGraph::add_new_distance_oriented_edges(
       const std::string &dataset,
       uint32_t num_threads,
       ANNS::AcornInUng new_cross_edge) // MODIFIED: 参数类型改为 const std::string&
   {
      // 防御性断言
      assert(_base_storage != nullptr && "Error: _base_storage is not initialized!");
      assert(_label_nav_graph != nullptr && "Error: _label_nav_graph is not initialized! Did you call build_label_nav_graph()?");
      assert(!_vamana_instances.empty() && "Error: _vamana_instances is empty! Did you call build_graph_for_all_groups()?");
      assert(_vamana_instances.size() == _num_groups + 1 && "Error: _vamana_instances has incorrect size!");

      // 根据传入的字符串策略决定执行哪些步骤
      bool add_acorn_edges = (new_cross_edge.new_edge_policy == "acorn_and_guarantee" || new_cross_edge.new_edge_policy == "just_acorn");
      bool add_guarantee_edges = (new_cross_edge.new_edge_policy == "acorn_and_guarantee" || new_cross_edge.new_edge_policy == "just_guarantee");
      auto all_start_time = std::chrono::high_resolution_clock::now();

      // 提前清空新边集合，无论执行哪个分支都需要
      _my_new_edges_set.clear();

      // ==========================================================================================
      // PART 1 - ACORN 边构建逻辑 (Steps 1-3)
      // ==========================================================================================
      if (add_acorn_edges)
      {
         std::cout << "\n========================================================" << std::endl;
         std::cout << "--- Augmenting Graph with New Distance-Oriented Edges (ACORN) ---" << std::endl;
         std::cout << "========================================================" << std::endl;
         auto acorn_part_start_time = std::chrono::high_resolution_clock::now();

         const size_t MAX_ID_SAMPLES = 15;
         const size_t SAMPLING_THRESHOLD = 100000;

         // FXY_ADD: Step 1 (识别、采样) 计时开始
         std::cout << "\n--- [Cross-Edge Sub-step 1: Identifying & Sampling Query Vectors] ---" << std::endl;
         auto step1_start_time = std::chrono::high_resolution_clock::now();

         // --- Step 0: 构建反向ID映射 ---
         std::cout << "[Step 0] Building reverse ID map for safe lookup..." << std::endl;
         std::unordered_map<IdxType, IdxType> old_to_new_map;
         old_to_new_map.reserve(_num_points);
         for (IdxType new_id = 0; new_id < _num_points; ++new_id)
         {
            old_to_new_map[_new_to_old_vec_ids[new_id]] = new_id;
         }
         std::cout << "  - Reverse ID map built successfully." << std::endl;

         // --- Step 1.1: 识别“概念根属性” ---
         std::cout << "[Step 1.1] Discovering conceptual root attributes via global coverage..." << std::endl;
         std::unordered_map<LabelType, size_t> attribute_counts;
         for (IdxType i = 0; i < _num_points; ++i)
         {
            for (const auto &label : _base_storage->get_label_set(i))
            {
               attribute_counts[label]++;
            }
         }
         std::unordered_set<LabelType> conceptual_root_labels;
         for (const auto &pair : attribute_counts)
         {
            if (static_cast<float>(pair.second) / _num_points >= new_cross_edge.root_coverage_threshold)
            {
               conceptual_root_labels.insert(pair.first);
            }
         }
         if (conceptual_root_labels.empty())
         {
            std::cerr << "  - WARNING: No conceptual root attributes found. Aborting edge addition." << std::endl;
            return;
         }
         std::cout << "  - Discovered " << conceptual_root_labels.size() << " conceptual root attributes." << std::endl;

         std::cout << "  	- Conceptual Root Labels are: { ";
         for (const auto &label : conceptual_root_labels)
         {
            std::cout << label << " ";
         }
         std::cout << "}" << std::endl;

         // --- Step 1.2: 根据层级比例识别候选组 ---
         std::cout << "[Step 1.2] Precisely identifying candidate groups using layer depth ratio..." << std::endl;
         size_t max_observed_depth = 0;
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            if (_group_id_to_label_set[group_id].size() > max_observed_depth)
            {
               max_observed_depth = _group_id_to_label_set[group_id].size();
            }
         }
         int target_max_depth = std::max(1, static_cast<int>(max_observed_depth * new_cross_edge.layer_depth_retio));
         std::cout << "  - Max observed layer depth in dataset: " << max_observed_depth << std::endl;
         std::cout << "  - Using layer_depth_ratio=" << new_cross_edge.layer_depth_retio << ", target max depth is " << target_max_depth << std::endl;
         std::unordered_set<IdxType> candidate_groups;
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            const auto &label_set = _group_id_to_label_set[group_id];
            if (label_set.empty() || label_set.size() > target_max_depth)
               continue;
            for (const auto &label : label_set)
            {
               if (conceptual_root_labels.count(label))
               {
                  candidate_groups.insert(group_id);
                  break;
               }
            }
         }
         std::unordered_set<IdxType> topmost_group_ids;
         for (IdxType group_id : candidate_groups)
         {
            bool is_topmost = true;
            for (IdxType parent_id : _label_nav_graph->in_neighbors[group_id])
            {
               if (candidate_groups.count(parent_id))
               {
                  is_topmost = false;
                  break;
               }
            }
            if (is_topmost)
            {
               topmost_group_ids.insert(group_id);
            }
         }
         std::cout << "  - Identified " << candidate_groups.size() << " total candidate groups within target depth." << std::endl;
         std::cout << "  - Identified " << topmost_group_ids.size() << " true topmost groups among them." << std::endl;

         // --- 步骤 1.3: 将向量分入两个桶 ---
         std::cout << "[Step 1.3] Separating vectors into 'topmost' and 'subsequent' buckets..." << std::endl;
         std::vector<IdxType> topmost_vec_ids, subsequent_vec_ids;
         std::vector<float *> topmost_vec_data, subsequent_vec_data;
         std::vector<std::vector<LabelType>> topmost_vec_labels, subsequent_vec_labels;
         for (IdxType group_id : candidate_groups)
         {
            const auto &range = _group_id_to_range[group_id];
            const auto &label_set = _group_id_to_label_set[group_id];
            if (topmost_group_ids.count(group_id))
            {
               for (IdxType u = range.first; u < range.second; ++u)
               {
                  topmost_vec_ids.push_back(u);
                  topmost_vec_data.push_back(const_cast<float *>(reinterpret_cast<const float *>(_base_storage->get_vector(u))));
                  topmost_vec_labels.push_back(label_set);
               }
            }
            else
            {
               for (IdxType u = range.first; u < range.second; ++u)
               {
                  subsequent_vec_ids.push_back(u);
                  subsequent_vec_data.push_back(const_cast<float *>(reinterpret_cast<const float *>(_base_storage->get_vector(u))));
                  subsequent_vec_labels.push_back(label_set);
               }
            }
         }
         std::cout << "  - Topmost-layer bucket size: " << topmost_vec_ids.size() << " vectors." << std::endl;
         std::cout << "  - Subsequent-layer bucket size: " << subsequent_vec_ids.size() << " vectors." << std::endl;

         // --- 步骤 1.4: 应用分层抽样逻辑 ---
         std::cout << "[Step 1.4] Applying stratified sampling logic..." << std::endl;
         std::vector<IdxType> final_query_vec_ids;
         std::vector<float *> final_query_vec_data;
         std::vector<std::vector<LabelType>> final_query_vec_labels;
         size_t total_candidates_vecs = topmost_vec_ids.size() + subsequent_vec_ids.size();
         const auto original_topmost_vec_ids = topmost_vec_ids;

         if (total_candidates_vecs < SAMPLING_THRESHOLD || new_cross_edge.query_vector_ratio >= 1.0f)
         {
            std::cout << "  - Decision: Using ALL candidates without sampling." << std::endl;
            final_query_vec_ids = std::move(topmost_vec_ids);
            final_query_vec_ids.insert(final_query_vec_ids.end(), subsequent_vec_ids.begin(), subsequent_vec_ids.end());
            final_query_vec_data = std::move(topmost_vec_data);
            final_query_vec_data.insert(final_query_vec_data.end(), subsequent_vec_data.begin(), subsequent_vec_data.end());
            final_query_vec_labels = std::move(topmost_vec_labels);
            final_query_vec_labels.insert(final_query_vec_labels.end(), subsequent_vec_labels.begin(), subsequent_vec_labels.end());
         }
         else
         {
            std::cout << "  - Decision: Performing stratified sampling." << std::endl;
            size_t target_total_count = static_cast<size_t>(total_candidates_vecs * new_cross_edge.query_vector_ratio);
            target_total_count = std::max(target_total_count, topmost_vec_ids.size());

            final_query_vec_ids = std::move(topmost_vec_ids);
            final_query_vec_data = std::move(topmost_vec_data);
            final_query_vec_labels = std::move(topmost_vec_labels);

            size_t num_to_sample = target_total_count > final_query_vec_ids.size() ? target_total_count - final_query_vec_ids.size() : 0;
            if (num_to_sample > 0 && !subsequent_vec_ids.empty())
            {
               num_to_sample = std::min(num_to_sample, subsequent_vec_ids.size());
               std::vector<size_t> indices(subsequent_vec_ids.size());
               std::iota(indices.begin(), indices.end(), 0);
               std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

               for (size_t i = 0; i < num_to_sample; ++i)
               {
                  size_t original_idx = indices[i];
                  final_query_vec_ids.push_back(subsequent_vec_ids[original_idx]);
                  final_query_vec_data.push_back(subsequent_vec_data[original_idx]);
                  final_query_vec_labels.push_back(subsequent_vec_labels[original_idx]);
               }
            }
         }
         std::cout << "  - Final number of query vectors for edge augmentation: " << final_query_vec_ids.size() << std::endl;

         if (final_query_vec_ids.empty())
         {
            std::cout << "  - No vectors to process. Skipping edge addition." << std::endl;
            return;
         }

         // FXY_ADD: Step 1 计时结束
         _cross_edge_step1_time_ms = std::chrono::duration<double, std::milli>(
                                         std::chrono::high_resolution_clock::now() - step1_start_time)
                                         .count();
         std::cout << "--- [Finished Sub-step 1 in " << _cross_edge_step1_time_ms << " ms] ---" << std::endl;

         // --- 步骤 1.5: 执行 ACORN ---
         // FXY_ADD: Step 2 (执行ACORN) 计时开始
         std::cout << "\n--- [Cross-Edge Sub-step 2: Executing ACORN Script] ---" << std::endl;
         auto step2_start_time = std::chrono::high_resolution_clock::now();
         std::cout << "[Step 1.5] Preparing input files and executing ACORN script..." << std::endl;
         std::string query_vec_path = new_cross_edge.acorn_in_ung_output_path + "/acorn_query_vectors.fvecs";
         std::string query_attr_path = new_cross_edge.acorn_in_ung_output_path + "/acorn_query_attrs.txt";
         write_fvecs(query_vec_path, final_query_vec_data, _base_storage->get_dim());
         write_labels_txt(query_attr_path, final_query_vec_labels);
         std::string output_neighbors_file = new_cross_edge.acorn_in_ung_output_path + "/R_neighbors_with_vectors.txt";
         std::stringstream cmd_ss;
         cmd_ss << "/data/fxy/FilterVector/FilterVectorCode/ACORN/run_acorn_for_ung.sh "
                << dataset << " " << new_cross_edge.R_in_add_new_edge << " " << _base_storage->get_num_points() << " "
                << new_cross_edge.M << " " << new_cross_edge.M_beta << " " << new_cross_edge.gamma << " " << new_cross_edge.efs << " "
                << num_threads << " "
                << query_vec_path << " " << query_attr_path << " " << new_cross_edge.acorn_in_ung_output_path << " 1";
         std::string cmd = cmd_ss.str();
         std::cout << "  - Executing command: " << cmd << std::endl;
         int ret = system(cmd.c_str());
         if (ret != 0)
         {
            std::cerr << "  - FATAL ERROR: ACORN script execution failed! Check logs in " << new_cross_edge.acorn_in_ung_output_path << std::endl;
            return;
         }
         std::cout << "  - ACORN script finished successfully." << std::endl;

         // FXY_ADD: Step 2 计时结束
         _cross_edge_step2_acorn_time_ms = std::chrono::duration<double, std::milli>(
                                               std::chrono::high_resolution_clock::now() - step2_start_time)
                                               .count();
         std::cout << "--- [Finished Sub-step 2 in " << _cross_edge_step2_acorn_time_ms << " ms] ---" << std::endl;

         // --- 步骤 1.6 & 2: 解析、过滤并添加距离驱动边 ---
         // FXY_ADD: Step 3 (添加距离驱动边) 计时开始
         std::cout << "\n--- [Cross-Edge Sub-step 3: Adding Distance-Oriented Edges] ---" << std::endl;
         auto step3_start_time = std::chrono::high_resolution_clock::now();

         std::cout << "[Step 1.6] Parsing ACORN output and generating candidates..." << std::endl;
         std::vector<NewEdgeCandidate> candidates;
         std::ifstream neighbor_file(output_neighbors_file);
         if (!neighbor_file.is_open())
         {
            std::cerr << "  - FATAL ERROR: Cannot open ACORN output file: " << output_neighbors_file << std::endl;
            return;
         }
         std::string line;
         long long current_query_local_idx = -1;
         IdxType u_id_new = 0;
         size_t intra_group_edges_skipped = 0;
         while (std::getline(neighbor_file, line))
         {
            if (line.rfind("Query: ", 0) == 0)
            {
               current_query_local_idx = std::stoll(line.substr(7));
               if (current_query_local_idx >= 0 && current_query_local_idx < final_query_vec_ids.size())
               {
                  u_id_new = final_query_vec_ids[current_query_local_idx];
               }
            }
            else if (line.rfind("NeighborID: ", 0) == 0)
            {
               std::stringstream line_ss(line);
               std::string token;
               IdxType v_id_old;
               line_ss >> token >> v_id_old;
               if (v_id_old < 0)
                  continue;
               auto it = old_to_new_map.find(v_id_old);
               if (it == old_to_new_map.end())
                  continue;
               IdxType v_id_new = it->second;
               IdxType group_u = _new_vec_id_to_group_id[u_id_new];
               IdxType group_v = _new_vec_id_to_group_id[v_id_new];
               if (group_u != group_v)
               {
                  float dist = _distance_handler->compute(
                      _base_storage->get_vector(u_id_new),
                      _base_storage->get_vector(v_id_new),
                      _base_storage->get_dim());
                  candidates.push_back({u_id_new, v_id_new, dist});
               }
               else
               {
                  intra_group_edges_skipped++;
               }
            }
         }
         neighbor_file.close();
         std::cout << "  - Generated " << candidates.size() << " initial cross-group candidates from ACORN results." << std::endl;
         std::cout << "  - Skipped " << intra_group_edges_skipped << " potential edges because they were intra-group." << std::endl;

         // ============================================================================================
         // FXY_FIX: 这是策略一的最终修复版本
         // ============================================================================================
         std::cout << "[Step 2] Adding new edges with smart selection logic and in-degree control (M=" << new_cross_edge.M_in_add_new_edge << ")..." << std::endl;

         // 1. 按“查询点”对所有候选邻居进行分组
         std::unordered_map<IdxType, std::vector<NewEdgeCandidate>> query_to_candidates;
         for (const auto &cand : candidates)
         {
            query_to_candidates[cand.from].push_back(cand);
         }

         _num_distance_oriented_edges = 0;

         // FXY_FIX: 1. 重新引入全局的入度计数器
         std::vector<std::atomic<IdxType>> new_edge_in_degree(_num_points);
         for (size_t i = 0; i < _num_points; ++i)
         {
            new_edge_in_degree[i] = 0;
         }

         // 2. 遍历每个查询点，独立地为其选择最佳出边
         for (auto const &[query_id, cand_list] : query_to_candidates)
         {
            std::vector<NewEdgeCandidate> hierarchical_candidates;
            std::vector<NewEdgeCandidate> non_hierarchical_candidates;

            IdxType from_group = _new_vec_id_to_group_id[query_id];

            // 3. 将候选邻居分为“层级邻居”和“普通邻居”两个桶
            for (const auto &cand : cand_list)
            {
               IdxType to_group = _new_vec_id_to_group_id[cand.to];
               const auto &children = _label_nav_graph->out_neighbors[from_group];
               bool is_hierarchical = false;
               for (const auto &true_child : children)
               {
                  if (to_group == true_child)
                  {
                     is_hierarchical = true;
                     break;
                  }
               }
               if (is_hierarchical)
               {
                  hierarchical_candidates.push_back(cand);
               }
               else
               {
                  non_hierarchical_candidates.push_back(cand);
               }
            }

            // 4. 对两个桶内的候选者分别按距离排序
            std::sort(hierarchical_candidates.begin(), hierarchical_candidates.end(),
                      [](const auto &a, const auto &b)
                      { return a.distance < b.distance; });
            std::sort(non_hierarchical_candidates.begin(), non_hierarchical_candidates.end(),
                      [](const auto &a, const auto &b)
                      { return a.distance < b.distance; });

            // 5. 智能添加边：同时考虑出度、入度限制
            size_t edges_added_for_this_query = 0;
            const size_t max_out_degree = new_cross_edge.M_in_add_new_edge;
            const size_t max_in_degree = new_cross_edge.M_in_add_new_edge; // 通常入度和出度限制相同

            // 5.1. 优先添加层级邻居
            for (const auto &cand : hierarchical_candidates)
            {
               if (edges_added_for_this_query >= max_out_degree)
                  break;

               // FXY_FIX: 2. 在添加边时检查目标节点的入度
               if (new_edge_in_degree[cand.to].load() < max_in_degree)
               {
                  uint64_t edge_key = (uint64_t)cand.from << 32 | cand.to;
                  if (_my_new_edges_set.find(edge_key) == _my_new_edges_set.end())
                  {
                     _graph->neighbors[cand.from].push_back(cand.to);
                     _num_distance_oriented_edges++;
                     edges_added_for_this_query++;
                     // FXY_FIX: 3. 成功添加后，才增加目标节点的入度计数
                     new_edge_in_degree[cand.to]++;
                     _my_new_edges_set.insert(edge_key);
                  }
               }
            }

            // 5.2. 如果出度名额还有剩余，用普通邻居来填充
            for (const auto &cand : non_hierarchical_candidates)
            {
               if (edges_added_for_this_query >= max_out_degree)
                  break;

               // FXY_FIX: 2. 同样检查目标节点的入度
               if (new_edge_in_degree[cand.to].load() < max_in_degree)
               {
                  uint64_t edge_key = (uint64_t)cand.from << 32 | cand.to;
                  if (_my_new_edges_set.find(edge_key) == _my_new_edges_set.end())
                  {
                     _graph->neighbors[cand.from].push_back(cand.to);
                     _num_distance_oriented_edges++;
                     edges_added_for_this_query++;
                     // FXY_FIX: 3. 成功添加后，才增加目标节点的入度计数
                     new_edge_in_degree[cand.to]++;
                     _my_new_edges_set.insert(edge_key);
                  }
               }
            }
         }

         std::cout << "  - Added a total of " << _num_distance_oriented_edges << " new distance-oriented edges to the graph with smart selection and in-degree control." << std::endl;

         // 诊断分析部分
         std::set<std::pair<IdxType, IdxType>> acorn_connected_group_pairs;
         for (const auto &edge_key : _my_new_edges_set)
         {
            IdxType from_id = edge_key >> 32;
            IdxType to_id = edge_key & 0xFFFFFFFF;
            acorn_connected_group_pairs.insert({_new_vec_id_to_group_id[from_id], _new_vec_id_to_group_id[to_id]});
         }

         std::cout << "\n--- [Cross-Edge Sub-step 3.5: ACORN Edge Connectivity Analysis] ---" << std::endl;
         // ... (诊断分析的 cout 代码完全不变) ...
         size_t total_required_hierarchical_links = 0;
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            total_required_hierarchical_links += _label_nav_graph->out_neighbors[group_id].size();
         }
         std::cout << "  - Total parent->child links in hierarchy: " << total_required_hierarchical_links << std::endl;
         std::cout << "  - ACORN edges created " << acorn_connected_group_pairs.size() << " unique group-to-group connections." << std::endl;

         size_t satisfied_hierarchical_links = 0;
         for (const auto &group_pair : acorn_connected_group_pairs)
         {
            IdxType parent_group = group_pair.first;
            IdxType child_group = group_pair.second;
            const auto &children = _label_nav_graph->out_neighbors[parent_group];
            bool is_hierarchical = false;
            for (const auto &true_child : children)
            {
               if (child_group == true_child)
               {
                  is_hierarchical = true;
                  break;
               }
            }
            if (is_hierarchical)
            {
               satisfied_hierarchical_links++;
            }
         }
         std::cout << "  - Among them, " << satisfied_hierarchical_links << " connections are valid parent->child links." << std::endl;
         double satisfaction_rate = total_required_hierarchical_links > 0 ? (double)satisfied_hierarchical_links / total_required_hierarchical_links * 100.0 : 0.0;
         std::cout << "  - Satisfaction rate of hierarchical links by ACORN: " << satisfaction_rate << "%" << std::endl;
         std::cout << "--- [Finished ACORN Edge Analysis] ---\n"
                   << std::endl;

         // FXY_ADD: Step 3 计时结束
         _cross_edge_step3_add_dist_edges_time_ms = std::chrono::duration<double, std::milli>(
                                                        std::chrono::high_resolution_clock::now() - step3_start_time)
                                                        .count();
         std::cout << "--- [Finished Sub-step 3 in " << _cross_edge_step3_add_dist_edges_time_ms << " ms] ---" << std::endl;

         // 计算并保存总的跨组边构建时间
         _build_cross_edges_time = std::chrono::duration<double, std::milli>(
                                       std::chrono::high_resolution_clock::now() - acorn_part_start_time)
                                       .count();
      }

      // ==========================================================================================
      // PART 2 - 层级保障边修复逻辑 (Step 4)
      // ==========================================================================================
      if (add_guarantee_edges)
      {
         std::cout << "\n--- [Cross-Edge Sub-step 4: Adding Targeted Edges for Unconnected Children] ---" << std::endl;
         auto step4_start_time = std::chrono::high_resolution_clock::now();

         // 1. 初始化，注意SearchCacheList仅用于获取临时的搜索队列
         size_t max_group_size = 0;
         if (_num_groups > 0)
         {
            for (auto group_id = 1; group_id <= _num_groups; ++group_id)
               max_group_size = std::max(max_group_size, static_cast<size_t>(_group_id_to_range[group_id].second - _group_id_to_range[group_id].first));
         }
         SearchCacheList search_cache_list(1, max_group_size, _Lbuild);

         // 2. 准备容器存储补充边
         std::vector<std::vector<std::pair<IdxType, IdxType>>> additional_edges_hierarchical(_num_groups + 1);

         // 3. 以单线程模式遍历所有组
         std::cout << "  - Checking for and repairing broken parent->child links (in single-thread mode)..." << std::endl;
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            const auto &cur_range = _group_id_to_range[group_id];
            if (cur_range.first >= cur_range.second || _label_nav_graph->out_neighbors[group_id].empty())
            {
               continue;
            }

            std::unordered_set<IdxType> connected_groups;
            for (IdxType i = cur_range.first; i < cur_range.second; ++i)
            {
               for (const auto &neighbor : _graph->neighbors[i])
                  connected_groups.insert(_new_vec_id_to_group_id[neighbor]);
            }

            for (IdxType out_group_id : _label_nav_graph->out_neighbors[group_id])
            {
               if (connected_groups.find(out_group_id) == connected_groups.end())
               {
                  const auto &out_range = _group_id_to_range[out_group_id];
                  size_t child_group_size = out_range.second - out_range.first;
                  if (child_group_size == 0)
                     continue;
                  if (child_group_size <= _max_degree)
                  { // 小规模组逻辑（蛮力）
                     IdxType query_vec_id = _group_entry_points[group_id];
                     const char *query = _base_storage->get_vector(query_vec_id);
                     float min_dist = std::numeric_limits<float>::max();
                     IdxType best_neighbor_id = -1;
                     for (IdxType i = out_range.first; i < out_range.second; ++i)
                     {
                        float dist = _distance_handler->compute(query, _base_storage->get_vector(i), _base_storage->get_dim());
                        if (dist < min_dist)
                        {
                           min_dist = dist;
                           best_neighbor_id = i;
                        }
                     }
                     if (best_neighbor_id != -1)
                     {
                        additional_edges_hierarchical[group_id].emplace_back(query_vec_id, best_neighbor_id);
                     }
                  }
                  else
                  { // 大规模组逻辑：使用手动图搜索，并配合正确的“已访问”集合
                     IdxType cnt = 0;
                     for (auto vec_id = cur_range.first; vec_id < cur_range.second && cnt < _num_cross_edges; ++vec_id)
                     {
                        const char *query_vec = _base_storage->get_vector(vec_id);
                        auto search_cache = search_cache_list.get_free_cache();
                        search_cache->search_queue.clear();
                        std::unordered_set<IdxType> visited_nodes_global;
                        IdxType child_group_entry_point = _group_entry_points[out_group_id];
                        float dist = _distance_handler->compute(query_vec, _base_storage->get_vector(child_group_entry_point), _base_storage->get_dim());

                        visited_nodes_global.insert(child_group_entry_point);
                        search_cache->search_queue.insert(child_group_entry_point, dist);
                        while (search_cache->search_queue.has_unexpanded_node())
                        {
                           const Candidate &cur = search_cache->search_queue.get_closest_unexpanded();
                           for (auto neighbor_id : _graph->neighbors[cur.id])
                           {
                              if (neighbor_id >= out_range.first && neighbor_id < out_range.second)
                              {
                                 if (visited_nodes_global.find(neighbor_id) == visited_nodes_global.end())
                                 {
                                    visited_nodes_global.insert(neighbor_id);
                                    float d = _distance_handler->compute(query_vec, _base_storage->get_vector(neighbor_id), _base_storage->get_dim());
                                    search_cache->search_queue.insert(neighbor_id, d);
                                 }
                              }
                           }
                        }
                        for (size_t k = 0; k < search_cache->search_queue.size() && k < _num_cross_edges / 2; ++k)
                        {
                           additional_edges_hierarchical[group_id].emplace_back(vec_id, search_cache->search_queue[k].id);
                           cnt++;
                        }
                        search_cache_list.release_cache(search_cache);
                     }
                  }

                  // // 不再区分大小组，总是执行以下逻辑
                  // IdxType query_vec_id = _group_entry_points[group_id];
                  // const char* query = _base_storage->get_vector(query_vec_id);
                  // float min_dist = std::numeric_limits<float>::max();
                  // IdxType best_neighbor_id = -1;

                  // for (IdxType i = out_range.first; i < out_range.second; ++i) {
                  //    float dist = _distance_handler->compute(query, _base_storage->get_vector(i), _base_storage->get_dim());
                  //    if (dist < min_dist) { min_dist = dist; best_neighbor_id = i; }
                  // }

                  // if (best_neighbor_id != -1) {
                  //    additional_edges_hierarchical[group_id].emplace_back(query_vec_id, best_neighbor_id);
                  // }
               }
            }
         }
         std::cout << std::endl;

         // 4. 单线程合并所有补充边...
         std::cout << "  - Merging newly created fill-in edges..." << std::endl;
         size_t num_added_hierarchical_edges = 0;
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            for (const auto &[from_id, to_id] : additional_edges_hierarchical[group_id])
            {
               uint64_t edge_key = (uint64_t)from_id << 32 | to_id;
               if (_my_new_edges_set.find(edge_key) == _my_new_edges_set.end())
               {
                  _graph->neighbors[from_id].push_back(to_id);
                  _my_new_edges_set.insert(edge_key);
                  num_added_hierarchical_edges++;
               }
            }
         }
         std::cout << "  - Merged " << num_added_hierarchical_edges << " new hierarchical edges to repair the graph." << std::endl;

         _cross_edge_step4_add_hierarchy_edges_time_ms = std::chrono::duration<double, std::milli>(
                                                             std::chrono::high_resolution_clock::now() - step4_start_time)
                                                             .count();
         std::cout << "--- [Finished Sub-step 4 in " << _cross_edge_step4_add_hierarchy_edges_time_ms << " ms] ---" << std::endl;
      }

      auto total_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - all_start_time).count();
      std::cout << "\n--- Finished augmenting graph in " << total_time_ms << " ms. ---" << std::endl;
      std::cout << "========================================================" << std::endl;
   }

   // =====================================begin 添加新的跨组边=========================================

   void UniNavGraph::build_label_nav_graph()
   {
      std::cout << "Building label navigation graph... " << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      _label_nav_graph = std::make_shared<LabelNavGraph>(_num_groups + 1);
      omp_set_num_threads(_num_threads);

// obtain out-neighbors
#pragma omp parallel for schedule(dynamic, 256)
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         // if (group_id % 100 == 0)
         //    std::cout << "\r" << (100.0 * group_id) / _num_groups << "%" << std::flush;
         std::vector<IdxType> min_super_set_ids;
         get_min_super_sets(_group_id_to_label_set[group_id], min_super_set_ids, true);
         _label_nav_graph->out_neighbors[group_id] = min_super_set_ids;
      }

      // obtain in-neighbors
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
         for (auto each : _label_nav_graph->out_neighbors[group_id])
            _label_nav_graph->in_neighbors[each].emplace_back(group_id);

      _build_LNG_time = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - start_time)
                            .count();
      std::cout << "\r- Finished in " << _build_LNG_time << " ms" << std::endl;
   }

   // fxy_add : 打印信息的build_label_nav_graph
   //    void UniNavGraph::build_label_nav_graph()
   //    {
   //       std::cout << "Building label navigation graph... " << std::endl;
   //       auto start_time = std::chrono::high_resolution_clock::now();
   //       _label_nav_graph = std::make_shared<LabelNavGraph>(_num_groups + 1);
   //       omp_set_num_threads(_num_threads);
   //       std::ofstream outfile("lng_structure.txt");
   //       if (!outfile.is_open())
   //       {
   //          std::cerr << "Error: Could not open lng_structure.txt for writing!" << std::endl;
   //          return;
   //       }
   //       outfile << "Label Navigation Graph (LNG) Structure\n";
   //       outfile << "=====================================\n";
   //       outfile << "Format: [GroupID] {LabelSet} -> [OutNeighbor1]{LabelSet}, [OutNeighbor2]{LabelSet}, ...\n\n";
   // // obtain out-neighbors
   // #pragma omp parallel for schedule(dynamic, 256)
   //       for (auto group_id = 1; group_id <= _num_groups; ++group_id)
   //       {
   //          if (group_id % 100 == 0)
   //          {
   // #pragma omp critical
   //             std::cout << "\r" << (100.0 * group_id) / _num_groups << "%" << std::flush;
   //          }
   //          std::vector<IdxType> min_super_set_ids;
   //          get_min_super_sets(_group_id_to_label_set[group_id], min_super_set_ids, true);
   //          _label_nav_graph->out_neighbors[group_id] = min_super_set_ids;
   //          _label_nav_graph->out_degree[group_id] = min_super_set_ids.size();
   // #pragma omp critical
   //          {
   //             outfile << "[" << group_id << "] {";
   //             // 打印标签集
   //             for (const auto &label : _group_id_to_label_set[group_id])
   //                outfile << label << ",";
   //             outfile << "} -> ";
   //             // 打印出边（包含目标节点的标签集）
   //             for (size_t i = 0; i < min_super_set_ids.size(); ++i)
   //             {
   //                auto target_id = min_super_set_ids[i];
   //                outfile << "[" << target_id << "] {";
   //                // 打印目标节点的标签集
   //                for (const auto &label : _group_id_to_label_set[target_id])
   //                {
   //                   outfile << label << ",";
   //                }
   //                outfile << "}";
   //                if (i != min_super_set_ids.size() - 1)
   //                   outfile << ", ";
   //             }
   //             outfile << "\n";
   //          }
   //       }
   //       // obtain in-neighbors (不需要打印入边，但保留原有逻辑)
   //       for (auto group_id = 1; group_id <= _num_groups; ++group_id)
   //       {
   //          for (auto each : _label_nav_graph->out_neighbors[group_id])
   //          {
   //             _label_nav_graph->in_neighbors[each].emplace_back(group_id);
   //             _label_nav_graph->in_degree[group_id] += 1;
   //          }
   //       }
   //       // outfile.close();
   //       _build_LNG_time = std::chrono::duration<double, std::milli>(
   //                             std::chrono::high_resolution_clock::now() - start_time)
   //                             .count();
   //       std::cout << "\r- Finished in " << _build_LNG_time << " ms" << std::endl;
   //       std::cout << "- LNG structure saved to lng_structure.txt" << std::endl;
   //    }

   // fxy_add:初始化求flag的几个数据结构
   void UniNavGraph::initialize_lng_descendants_coverage_bitsets()
   {
      std::cout << "Initializing LNG descendants and coverage bitsets..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      _lng_descendants_bits.resize(_num_groups + 1);
      _covered_sets_bits.resize(_num_groups + 1);

      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         // 初始化大小
         _lng_descendants_bits[group_id].resize(_num_groups);
         _covered_sets_bits[group_id].resize(_num_points);

         // 填充后代的group的集合
         const auto &descendants = _label_nav_graph->_lng_descendants[group_id];
         for (auto id : descendants)
         {
            _lng_descendants_bits[group_id].set(id);
         }

         // 填充覆盖的向量的集合
         const auto &coverage = _label_nav_graph->covered_sets[group_id];
         for (auto id : coverage)
         {
            _covered_sets_bits[group_id].set(id);
         }
      }
   }

   // fxy_add:初始化求flag的几个数据结构

   void UniNavGraph::initialize_roaring_bitsets()
   {
      std::cout << "enter initialize_roaring_bitsets" << std::endl;

      // 打印 _num_groups 的值，确保它是一个预期的、合理的值
      std::cout << "_num_groups = " << _num_groups << std::endl;

      // 检查指针是否为空
      if (!_label_nav_graph)
      {
         std::cerr << "Error: _label_nav_graph is a null pointer!" << std::endl;
         return; // 或者抛出异常
      }

      _lng_descendants_rb.resize(_num_groups + 1);
      _covered_sets_rb.resize(_num_groups + 1);

      // 关键诊断：打印所有容器的大小
      std::cout << "_lng_descendants_rb.size() = " << _lng_descendants_rb.size() << std::endl;
      std::cout << "_covered_sets_rb.size() = " << _covered_sets_rb.size() << std::endl;
      std::cout << "_label_nav_graph->_lng_descendants.size() = " << _label_nav_graph->_lng_descendants.size() << std::endl;
      std::cout << "_label_nav_graph->covered_sets.size() = " << _label_nav_graph->covered_sets.size() << std::endl;

      // std::cout << "begin for " << std::endl;

      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         // std::cout << "group_id: " << group_id << std::endl;
         const auto &descendants = _label_nav_graph->_lng_descendants[group_id];
         const auto &coverage = _label_nav_graph->covered_sets[group_id];

         // 初始化后代 bitset
         for (auto id : descendants)
         {
            _lng_descendants_rb[group_id].add(id);
         }

         // 初始化覆盖 bitset
         for (auto id : coverage)
         {
            _covered_sets_rb[group_id].add(id);
         }
      }

      std::cout << "Roaring bitsets initialized." << std::endl;
   }

   // 将分组内的局部索引转换为全局索引
   void UniNavGraph::add_offset_for_uni_nav_graph()
   {
      omp_set_num_threads(_num_threads);
#pragma omp parallel for schedule(dynamic, 4096)
      for (auto i = 0; i < _num_points; ++i)
         for (auto &neighbor : _graph->neighbors[i])
            neighbor += _group_id_to_range[_new_vec_id_to_group_id[i]].first;
   }

   void UniNavGraph::build_cross_group_edges()
   {
      std::cout << "Building cross-group edges ..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      // allocate memory for storaging cross-group neighbors
      std::vector<SearchQueue> cross_group_neighbors;
      cross_group_neighbors.resize(_num_points);
      for (auto point_id = 0; point_id < _num_points; ++point_id)
         cross_group_neighbors[point_id].reserve(_num_cross_edges);

      // allocate memory for search caches
      size_t max_group_size = 0;
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
         max_group_size = std::max(max_group_size, _group_id_to_vec_ids[group_id].size());
      SearchCacheList search_cache_list(_num_threads, max_group_size, _Lbuild);
      omp_set_num_threads(_num_threads);

      // for each group
      for (auto group_id = 1; group_id <= _num_groups; ++group_id)
      {
         if (_label_nav_graph->in_neighbors[group_id].size() > 0)
         {
            // if (group_id % 100 == 0)
            //    std::cout << "\r" << (100.0 * group_id) / _num_groups << "%" << std::flush;
            IdxType offset = _group_id_to_range[group_id].first;

            // query vamana index
            if (_index_name == "Vamana")
            {
               auto index = _vamana_instances[group_id];
               if (_num_cross_edges > _Lbuild)
               {
                  std::cerr << "Error: num_cross_edges should be less than or equal to Lbuild" << std::endl;
                  exit(-1);
               }

               // for each in-neighbor group
               for (auto in_group_id : _label_nav_graph->in_neighbors[group_id])
               {
                  const auto &range = _group_id_to_range[in_group_id];

// take each vector in the group as the query
#pragma omp parallel for schedule(dynamic, 1)
                  for (auto vec_id = range.first; vec_id < range.second; ++vec_id)
                  {
                     const char *query = _base_storage->get_vector(vec_id);
                     auto search_cache = search_cache_list.get_free_cache();
                     index->iterate_to_fixed_point(query, search_cache);

                     // update the cross-group edges for vec_id
                     for (auto k = 0; k < search_cache->search_queue.size(); ++k)
                        cross_group_neighbors[vec_id].insert(search_cache->search_queue[k].id + offset,
                                                             search_cache->search_queue[k].distance);
                     search_cache_list.release_cache(search_cache);
                  }
               }

               // if none of the above
            }
            else
            {
               std::cerr << "Error: invalid index name " << _index_name << std::endl;
               exit(-1);
            }
         }
      }

      // add additional edges
      std::vector<std::vector<std::pair<IdxType, IdxType>>> additional_edges(_num_groups + 1);
#pragma omp parallel for schedule(dynamic, 256)
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         const auto &cur_range = _group_id_to_range[group_id];
         std::unordered_set<IdxType> connected_groups;

         // obtain connected groups
         for (IdxType i = cur_range.first; i < cur_range.second; ++i)
            for (IdxType j = 0; j < cross_group_neighbors[i].size(); ++j)
               connected_groups.insert(_new_vec_id_to_group_id[cross_group_neighbors[i][j].id]);

         // add additional cross-group edges for unconnected groups
         for (IdxType out_group_id : _label_nav_graph->out_neighbors[group_id])
            if (connected_groups.find(out_group_id) == connected_groups.end())
            {
               IdxType cnt = 0;
               for (auto vec_id = cur_range.first; vec_id < cur_range.second && cnt < _num_cross_edges; ++vec_id)
               {
                  auto search_cache = search_cache_list.get_free_cache();
                  _vamana_instances[out_group_id]->iterate_to_fixed_point(_base_storage->get_vector(vec_id), search_cache);

                  for (auto k = 0; k < search_cache->search_queue.size() && k < _num_cross_edges / 2; ++k)
                  {
                     additional_edges[group_id].emplace_back(vec_id,
                                                             search_cache->search_queue[k].id + _group_id_to_range[out_group_id].first);
                     cnt += 1;
                  }
                  search_cache_list.release_cache(search_cache);
               }
            }
      }

      // add offset for uni-nav graph
      add_offset_for_uni_nav_graph();

// merge cross-group edges
#pragma omp parallel for schedule(dynamic, 4096)
      for (auto point_id = 0; point_id < _num_points; ++point_id)
         for (auto k = 0; k < cross_group_neighbors[point_id].size(); ++k)
            _graph->neighbors[point_id].emplace_back(cross_group_neighbors[point_id][k].id);

// merge additional cross-group edges
#pragma omp parallel for schedule(dynamic, 256)
      for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
      {
         for (const auto &[from_id, to_id] : additional_edges[group_id])
            _graph->neighbors[from_id].emplace_back(to_id);
      }

      _build_cross_edges_time = std::chrono::duration<double, std::milli>(
                                    std::chrono::high_resolution_clock::now() - start_time)
                                    .count();
      std::cout << "\r- Finish in " << _build_cross_edges_time << " ms" << std::endl;
   }

   /*void UniNavGraph::search(std::shared_ptr<IStorage> query_storage, std::shared_ptr<DistanceHandler> distance_handler,
                            uint32_t num_threads, IdxType Lsearch, IdxType num_entry_points, std::string scenario,
                            IdxType K, std::pair<IdxType, float> *results, std::vector<float> &num_cmps,
                            std::vector<std::bitset<10000001>> &bitmap)
   {
      auto num_queries = query_storage->get_num_points();
      _query_storage = query_storage;
      _distance_handler = distance_handler;
      _scenario = scenario;

      // preparation
      if (K > Lsearch)
      {
         std::cerr << "Error: K should be less than or equal to Lsearch" << std::endl;
         exit(-1);
      }
      SearchCacheList search_cache_list(num_threads, _num_points, Lsearch);

      // run queries
      omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
      for (auto id = 0; id < num_queries; ++id)
      {
         auto search_cache = search_cache_list.get_free_cache();
         const char *query = _query_storage->get_vector(id);
         SearchQueue cur_result;

         // for overlap or nofilter scenario
         if (scenario == "overlap" || scenario == "nofilter")
         {
            num_cmps[id] = 0;
            search_cache->visited_set.clear();
            cur_result.reserve(K);

            // obtain entry group
            std::vector<IdxType> entry_group_ids;
            if (scenario == "overlap")
               get_min_super_sets(_query_storage->get_label_set(id), entry_group_ids, false, false);
            else
               get_min_super_sets({}, entry_group_ids, true, true);

            // for each entry group
            for (const auto &group_id : entry_group_ids)
            {
               std::vector<IdxType> entry_points;
               get_entry_points_given_group_id(num_entry_points, search_cache->visited_set, group_id, entry_points);

               // graph search and dump to current result
               num_cmps[id] += iterate_to_fixed_point(query, search_cache, id, entry_points, true, false);
               for (auto k = 0; k < search_cache->search_queue.size() && k < K; ++k)
                  cur_result.insert(search_cache->search_queue[k].id, search_cache->search_queue[k].distance);
            }

            // for the other scenarios: containment, equality
         }
         else
         {

            // obtain entry points
            auto entry_points = get_entry_points(_query_storage->get_label_set(id), num_entry_points, search_cache->visited_set);
            if (entry_points.empty())
            {
               num_cmps[id] = 0;
               for (auto k = 0; k < K; ++k)
                  results[id * K + k].first = -1;
               continue;
            }

            // graph search
            num_cmps[id] = iterate_to_fixed_point(query, search_cache, id, entry_points);
            cur_result = search_cache->search_queue;
         }

         // write results
         for (auto k = 0; k < K; ++k)
         {
            if (k < cur_result.size())
            {
               results[id * K + k].first = _new_to_old_vec_ids[cur_result[k].id];
               results[id * K + k].second = cur_result[k].distance;
            }
            else
               results[id * K + k].first = -1;
         }

         // clean
         search_cache_list.release_cache(search_cache);
      }
   }*/

   // fxy_add
   void UniNavGraph::search_hybrid(std::shared_ptr<IStorage> query_storage,
                                   std::shared_ptr<DistanceHandler> distance_handler,
                                   uint32_t num_threads, IdxType Lsearch,
                                   IdxType num_entry_points, std::string scenario,
                                   IdxType K, std::pair<IdxType, float> *results,
                                   std::vector<float> &num_cmps,
                                   std::vector<QueryStats> &query_stats,
                                   std::vector<std::bitset<10000001>> &bitmaps,
                                   bool is_ori_ung,
                                   bool is_new_trie_method, bool is_rec_more_start,
                                   bool is_ung_more_entry, 
                                   int lsearch_start, int lsearch_step,
                                   int efs_start, int efs_step,
                                   const std::vector<IdxType> &true_query_group_ids)
   {
      auto num_queries = query_storage->get_num_points();
      _query_storage = query_storage;
      _distance_handler = distance_handler;
      _scenario = scenario;

      // 初始化统计信息
      query_stats.resize(num_queries);

      // 参数检查
      if (K > Lsearch)
      {
         std::cerr << "Error: K should be less than or equal to Lsearch" << std::endl;
         exit(-1);
      }

      // 搜索参数
      const float COVERAGE_THRESHOLD = 0.8f;
      const int MIN_LNG_DESCENDANTS_THRESHOLD = _num_points / 2.5;
      SearchCacheList search_cache_list(num_threads, _num_points, Lsearch);

      // 并行查询处理
      omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
      for (auto id = 0; id < num_queries; ++id)
      {
         auto &stats = query_stats[id];
         auto total_search_start_time = std::chrono::high_resolution_clock::now();

         auto search_cache = search_cache_list.get_free_cache();
         const char *query = _query_storage->get_vector(id);
         SearchQueue cur_result;
         cur_result.reserve(K);

         // 获取查询标签集
         const auto &query_labels = _query_storage->get_label_set(id);

         stats.query_length = query_labels.size(); // 记录查询长度
         if (!query_labels.empty())                // 记录候选集大小
            stats.candidate_set_size = _trie_index.get_candidate_count_for_label(query_labels.back());
         else
            stats.candidate_set_size = 0;

         // 计算入口组信息
         auto get_entry_group_start_time = std::chrono::high_resolution_clock::now();
         std::vector<IdxType> entry_group_ids;

         // ======================= idea1 selector begin =======================
         if (is_new_trie_method)
         {
            // --- 【路径 A: 启用智能决策模型 (idea1 selector)】 ---
            // 1. 默认决策
            bool is_new_trie_method_decision; 

            // 2. 实时预测
            if (_trie_method_selector != nullptr) {
               auto idea1_flag_satrt_time = std::chrono::high_resolution_clock::now();

               // 2.1 准备特征
               const auto& trie_metrics = _trie_static_metrics;
               const float epsilon = 1e-9f;
               const float avg_query_size = static_cast<float>(stats.query_length);
               const float avg_cand_size = static_cast<float>(stats.candidate_set_size);
               
               const float query_depth_ratio = avg_query_size / (trie_metrics.avg_path_length + epsilon);
               const float cand_set_coverage_ratio = avg_cand_size / (static_cast<float>(trie_metrics.total_nodes) + epsilon);
               const float query_path_density = avg_query_size * trie_metrics.avg_branching_factor;
               const float avg_nodes_per_label = static_cast<float>(trie_metrics.total_nodes) / (static_cast<float>(trie_metrics.label_cardinality) + epsilon);
               const float cand_set_selectivity = avg_nodes_per_label / (avg_cand_size + epsilon);
               const float query_cand_ratio = avg_query_size / (avg_cand_size + epsilon);
               const float avg_cand_size_sq = avg_cand_size * avg_cand_size;
               const float query_depth_ratio_sq = query_depth_ratio * query_depth_ratio;
               const float cand_x_query_interaction = avg_cand_size * avg_query_size;
               const float log_avg_cand_size = std::log1p(avg_cand_size);
               const float branching_x_candsize = trie_metrics.avg_branching_factor * avg_cand_size;

               // 2.2 构建特征向量
               std::vector<float> features = {
                  query_depth_ratio, cand_set_coverage_ratio, query_path_density,
                  cand_set_selectivity, query_cand_ratio, avg_cand_size_sq,
                  query_depth_ratio_sq, cand_x_query_interaction, log_avg_cand_size,
                  branching_x_candsize
               };

               // 2.3 调用模型预测
               try {
                  is_new_trie_method_decision = _trie_method_selector->predict(features);
               } catch (const std::exception& e) {
                  std::cerr << "Query " << id << " prediction error: " << e.what() << ". Falling back to default method." << std::endl;
               }
               stats.idea1_flag_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - idea1_flag_satrt_time).count();
            }
            
            
            // 3. 应用决策结果
            if (!is_ung_more_entry) {
               static std::atomic<int> trie_debug_print_counter{0};
               auto get_min_super_sets_satrt_time = std::chrono::high_resolution_clock::now();
               get_min_super_sets_debug(query_labels, entry_group_ids, false, true,
                                       trie_debug_print_counter, 
                                       is_new_trie_method_decision, // <-- 使用模型的决策结果
                                       is_rec_more_start, stats);
               stats.get_min_super_sets_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - get_min_super_sets_satrt_time).count();
               stats.num_entry_points = entry_group_ids.size();
            } else {
               IdxType true_group_id = 0;
               if (id < true_query_group_ids.size()) { true_group_id = true_query_group_ids[id]; }
               std::vector<IdxType> base_entry_groups;
               static std::atomic<int> trie_debug_print_counter{0};
               auto get_min_super_sets_satrt_time = std::chrono::high_resolution_clock::now();
               get_min_super_sets_debug(query_labels, base_entry_groups, false, true,
                                       trie_debug_print_counter, 
                                       is_new_trie_method_decision, // <-- 使用模型的决策结果
                                       is_rec_more_start, stats);
               stats.get_min_super_sets_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - get_min_super_sets_satrt_time).count();
               const size_t extra_k = base_entry_groups.size() / 5;
               const SelectionMode current_mode = SelectionMode::SizeAndDistance;
               const double beta_value = 1.0;
               entry_group_ids = select_entry_groups(base_entry_groups, current_mode, extra_k, beta_value, true_group_id);
               stats.num_entry_points = entry_group_ids.size();
            }
         }
         else 
         {
            // --- 【路径 B: 使用原始 UNG 算法逻辑】 ---
            // 传给 get_min_super_sets_debug 的 is_new_trie_method 参数固定为 false。
            if (!is_ung_more_entry) {
               static std::atomic<int> trie_debug_print_counter{0};
               auto get_min_super_sets_satrt_time = std::chrono::high_resolution_clock::now();
               get_min_super_sets_debug(query_labels, entry_group_ids, false, true,
                                       trie_debug_print_counter, 
                                       false, // <-- 固定为 false，执行原始的捷径法
                                       false, stats);
               stats.get_min_super_sets_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - get_min_super_sets_satrt_time).count();
               stats.num_entry_points = entry_group_ids.size();
            } else {
               IdxType true_group_id = 0;
               if (id < true_query_group_ids.size()) { true_group_id = true_query_group_ids[id]; }
               std::vector<IdxType> base_entry_groups;
               static std::atomic<int> trie_debug_print_counter{0};
               auto get_min_super_sets_satrt_time = std::chrono::high_resolution_clock::now();
               get_min_super_sets_debug(query_labels, base_entry_groups, false, true,
                                       trie_debug_print_counter, 
                                       false, // <-- 固定为 false，执行原始的捷径法
                                       false, stats);
               stats.get_min_super_sets_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - get_min_super_sets_satrt_time).count();
               const size_t extra_k = base_entry_groups.size() / 5;
               const SelectionMode current_mode = SelectionMode::SizeAndDistance;
               const double beta_value = 1.0;
               entry_group_ids = select_entry_groups(base_entry_groups, current_mode, extra_k, beta_value, true_group_id);
               stats.num_entry_points = entry_group_ids.size();
            }
         }
         
         stats.get_group_entry_time_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - get_entry_group_start_time).count();
         // ======================= idea1 selector end =======================

         // idea:计算flag
         auto flag_start_time = std::chrono::high_resolution_clock::now();
         stats.num_lng_descendants = [&]()
         {
            roaring::Roaring combined_descendants;
            auto desc_start = std::chrono::high_resolution_clock::now();
            for (auto group_id : entry_group_ids)
            {
               if (group_id > 0 && group_id <= _num_groups)
               {
                  combined_descendants |= _lng_descendants_rb[group_id];
               }
            }
            auto desc_end = std::chrono::high_resolution_clock::now();
            stats.descendants_merge_time_ms = std::chrono::duration<double, std::milli>(desc_end - desc_start).count();
            return combined_descendants.cardinality();
         }();

         float total_unique_coverage = [&]()
         {
            roaring::Roaring combined_coverage;
            auto cov_start = std::chrono::high_resolution_clock::now();
            for (auto group_id : entry_group_ids)
            {
               if (group_id > 0 && group_id <= _num_groups)
               {
                  combined_coverage |= _covered_sets_rb[group_id];
               }
            }
            auto cov_end = std::chrono::high_resolution_clock::now();
            stats.coverage_merge_time_ms = std::chrono::duration<double, std::milli>(cov_end - cov_start).count();
            return static_cast<float>(combined_coverage.cardinality()) / _num_points;
         }();

         stats.entry_group_total_coverage = total_unique_coverage;
         bool use_global_search = (total_unique_coverage > COVERAGE_THRESHOLD) ||
                                  (stats.num_lng_descendants > MIN_LNG_DESCENDANTS_THRESHOLD);
         stats.flag_time_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::high_resolution_clock::now() - flag_start_time)
                                  .count();

         if (is_ori_ung)
            use_global_search = false;

         use_global_search = true;
         stats.is_global_search = use_global_search;

         // 4. 执行搜索
         /*if (use_global_search)
         {
            // 4.1 全局图搜索模式
            search_cache->visited_set.clear();

            // 获取全局入口点
            std::vector<IdxType> global_entry_points;
            if (_global_vamana_entry_point != -1)
            {
               global_entry_points.push_back(_global_vamana_entry_point);
            }
            else
            {
               // 随机选择入口点
               for (int i = 0; i < num_entry_points; ++i)
               {
                  global_entry_points.push_back(rand() % _num_points);
               }
            }

            // 记录初始距离计算次数
            num_cmps[id] = iterate_to_fixed_point_global(query, search_cache, id, global_entry_points);
            stats.num_distance_calcs = num_cmps[id];

            // 过滤结果
            int valid_count = 0;
            for (size_t k = 0; k < search_cache->search_queue.size() && valid_count < K; k++)
            {
               auto candidate = search_cache->search_queue[k];
               const auto &candidate_labels = _base_storage->get_label_set(candidate.id);

               // 检查候选是否满足查询条件
               bool is_valid = true;
               if (scenario == "equality")
               {
                  is_valid = (candidate_labels == query_labels);
               }

               // 使用bitmaps进行过滤
               if (bitmaps.size() > id && bitmaps[id].size() > candidate.id)
               {
                  if (scenario == "containment")
                  {
                     is_valid = bitmaps[id][candidate.id];
                  }
                  else
                  {
                     is_valid = is_valid && bitmaps[id][candidate.id];
                  }
               }

               if (is_valid)
               {
                  cur_result.insert(candidate.id, candidate.distance);
                  valid_count++;
               }
            }
         }*/
         if (use_global_search)
         {
               // === ACORN集成 === 
               stats.is_global_search = true; 

               if (!_acorn_index) {
                  std::cerr << "Warning: ACORN index not loaded for query " << id 
                           << ". Skipping." << std::endl;
                  for (auto k = 0; k < K; ++k) results[id * K + k].first = -1;
                  continue;
               }

               // auto acorn_call_start_time = std::chrono::high_resolution_clock::now();

               // 1. 根据 Lsearch 动态计算 efs
               int current_efs = efs_start + ((Lsearch - lsearch_start) / lsearch_step) * efs_step;
               stats.acorn_efs_used = current_efs;

               // 2. 利用预计算的 bitmaps 和 ID映射表 来正确生成 filter_map
               std::vector<char> filter_map(_num_points);
               const auto& current_bitmap = bitmaps[id];
               for (IdxType new_id = 0; new_id < _num_points; ++new_id) {
                  IdxType original_id = _new_to_old_vec_ids[new_id];
                  filter_map[original_id] = current_bitmap[new_id] ? 1 : 0;// 在 filter_map 的 original_id 位置，填入 new_id 对应的过滤结果
               }

               // 3. 准备其他ACORN搜索参数
               const float* query_vector_float = reinterpret_cast<const float*>(query);
               _acorn_index->acorn.efSearch = current_efs;

               // 4. 准备结果容器并执行ACORN搜索
               std::vector<faiss::idx_t> result_original_ids(K); // 变量名清晰化，表示这是原始ID
               std::vector<float> result_dists(K);
               
               _acorn_index->search(1, query_vector_float, K, result_dists.data(), result_original_ids.data(), filter_map.data(), nullptr, nullptr, nullptr, true);

               // 5. 将ACORN返回的 original_id 转换为 UNG 的 new_id
               cur_result.clear();
               for (size_t i = 0; i < K; ++i) {
                  faiss::idx_t original_id = result_original_ids[i];
                  if (original_id != -1) {
                     // 使用反向映射表找到对应的 new_id
                     IdxType new_id = _old_to_new_vec_ids[original_id];
                     // 将转换后的 new_id 插入结果队列
                     cur_result.insert(new_id, result_dists[i]);
                  }
               }

               // 统计距离计算次数
               num_cmps[id] = 0; 

               // stats.acorn_search_time_ms = std::chrono::duration<double, std::milli>(
               //                      std::chrono::high_resolution_clock::now() - acorn_call_start_time)
               //                      .count();
         }
         else
         {
            // 4.2 传统搜索模式
            search_cache->visited_set.clear();

            if (scenario == "overlap" || scenario == "nofilter")
            {
               // 获取入口点
               std::vector<IdxType> entry_points;
               for (const auto &group_id : entry_group_ids)
               {
                  std::vector<IdxType> group_entry_points;
                  get_entry_points_given_group_id(num_entry_points,
                                                  search_cache->visited_set,
                                                  group_id,
                                                  group_entry_points);
                  entry_points.insert(entry_points.end(),
                                      group_entry_points.begin(),
                                      group_entry_points.end());
               }

               // 执行搜索
               num_cmps[id] = iterate_to_fixed_point(query, search_cache, id,
                                                     entry_points, stats.num_nodes_visited, true, false);
               stats.num_distance_calcs = num_cmps[id];

               // 收集结果
               for (auto k = 0; k < search_cache->search_queue.size() && k < K; ++k)
               {
                  cur_result.insert(search_cache->search_queue[k].id,
                                    search_cache->search_queue[k].distance);
               }
            }
            else
            {
               // containment/equality场景
               // auto entry_points = get_entry_points(query_labels, num_entry_points,
               //                                      search_cache->visited_set);

               std::vector<IdxType> entry_points;
               for (const auto &group_id : entry_group_ids)
               {
                  get_entry_points_given_group_id(num_entry_points,
                                                  search_cache->visited_set,
                                                  group_id,
                                                  entry_points);
               }

               if (entry_points.empty())
               {
                  stats.num_distance_calcs = 0;
                  continue;
               }

               num_cmps[id] = iterate_to_fixed_point(query, search_cache, id, entry_points, stats.num_nodes_visited);
               stats.num_distance_calcs = num_cmps[id];
               cur_result = search_cache->search_queue;
            }
         }

         // 5. 记录结果
         for (auto k = 0; k < K; ++k)
         {
            if (k < cur_result.size())
            {
               results[id * K + k].first = _new_to_old_vec_ids[cur_result[k].id];
               results[id * K + k].second = cur_result[k].distance;
            }
            else
            {
               results[id * K + k].first = -1;
            }
         }

         // 6. 记录统计信息
         stats.time_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::high_resolution_clock::now() - total_search_start_time)
                             .count();

         search_cache_list.release_cache(search_cache);
      }
   }

   std::vector<IdxType> UniNavGraph::get_entry_points(const std::vector<LabelType> &query_label_set,
                                                      IdxType num_entry_points, VisitedSet &visited_set)
   {
      std::vector<IdxType> entry_points;
      entry_points.reserve(num_entry_points);
      visited_set.clear();

      // obtain entry points for label-equality scenario
      if (_scenario == "equality")
      {
         auto node = _trie_index.find_exact_match(query_label_set);
         if (node == nullptr)
            return entry_points;
         get_entry_points_given_group_id(num_entry_points, visited_set, node->group_id, entry_points);

         // obtain entry points for label-containment scenario
      }
      else if (_scenario == "containment")
      {
         std::vector<IdxType> min_super_set_ids;
         get_min_super_sets(query_label_set, min_super_set_ids);
         for (auto group_id : min_super_set_ids)
            get_entry_points_given_group_id(num_entry_points, visited_set, group_id, entry_points);
      }
      else
      {
         std::cerr << "Error: invalid scenario " << _scenario << std::endl;
         exit(-1);
      }

      return entry_points;
   }

   void UniNavGraph::get_entry_points_given_group_id(IdxType num_entry_points, VisitedSet &visited_set,
                                                     IdxType group_id, std::vector<IdxType> &entry_points)
   {
      const auto &group_range = _group_id_to_range[group_id];

      // not enough entry points, use all of them
      if (group_range.second - group_range.first <= num_entry_points)
      {
         for (auto i = 0; i < group_range.second - group_range.first; ++i)
            entry_points.emplace_back(i + group_range.first);
         return;
      }

      // add the entry point of the group
      const auto &group_entry_point = _group_entry_points[group_id];
      visited_set.set(group_entry_point);
      entry_points.emplace_back(group_entry_point);

      // randomly sample the other entry points
      for (auto i = 1; i < num_entry_points; ++i)
      {
         auto entry_point = rand() % (group_range.second - group_range.first) + group_range.first;
         if (visited_set.check(entry_point) == false)
         {
            visited_set.set(entry_point);
            entry_points.emplace_back(i + group_range.first);
         }
      }
   }

   IdxType UniNavGraph::iterate_to_fixed_point(const char *query, std::shared_ptr<SearchCache> search_cache,
                                               IdxType target_id, const std::vector<IdxType> &entry_points,
                                               size_t &num_nodes_visited,
                                               bool clear_search_queue, bool clear_visited_set)
   {
      auto dim = _base_storage->get_dim();
      auto &search_queue = search_cache->search_queue;
      auto &visited_set = search_cache->visited_set;
      std::vector<IdxType> neighbors;
      if (clear_search_queue)
         search_queue.clear();
      if (clear_visited_set)
         visited_set.clear();

      // entry point
      for (const auto &entry_point : entry_points)
         search_queue.insert(entry_point, _distance_handler->compute(query, _base_storage->get_vector(entry_point), dim));
      IdxType num_cmps = entry_points.size();

      // greedily expand closest nodes
      while (search_queue.has_unexpanded_node())
      {
         const Candidate &cur = search_queue.get_closest_unexpanded();

         // iterate neighbors
         {
            std::lock_guard<std::mutex> lock(_graph->neighbor_locks[cur.id]);
            neighbors = _graph->neighbors[cur.id];
         }
         for (auto i = 0; i < neighbors.size(); ++i)
         {

            // prefetch
            if (i + 1 < neighbors.size() && visited_set.check(neighbors[i + 1]) == false)
               _base_storage->prefetch_vec_by_id(neighbors[i + 1]);

            // skip if visited
            auto &neighbor = neighbors[i];
            if (visited_set.check(neighbor))
               continue;
            visited_set.set(neighbor);

            num_nodes_visited++; // search过程中遍历的点的个数

            // push to search queue
            search_queue.insert(neighbor, _distance_handler->compute(query, _base_storage->get_vector(neighbor), dim));
            num_cmps++;
         }
      }
      return num_cmps;
   }

   // fxy_add
   IdxType UniNavGraph::iterate_to_fixed_point_global(const char *query, std::shared_ptr<SearchCache> search_cache,
                                                      IdxType target_id, const std::vector<IdxType> &entry_points,
                                                      bool clear_search_queue, bool clear_visited_set)
   {
      auto dim = _base_storage->get_dim();
      auto &search_queue = search_cache->search_queue;
      auto &visited_set = search_cache->visited_set;
      std::vector<IdxType> neighbors;
      if (clear_search_queue)
         search_queue.clear();
      if (clear_visited_set)
         visited_set.clear();

      // entry point
      for (const auto &entry_point : entry_points)
         search_queue.insert(entry_point, _distance_handler->compute(query, _base_storage->get_vector(entry_point), dim));
      IdxType num_cmps = entry_points.size();

      // greedily expand closest nodes
      while (search_queue.has_unexpanded_node())
      {
         const Candidate &cur = search_queue.get_closest_unexpanded();

         // iterate neighbors
         {
            std::lock_guard<std::mutex> lock(_global_graph->neighbor_locks[cur.id]);
            neighbors = _global_graph->neighbors[cur.id];
         }
         for (auto i = 0; i < neighbors.size(); ++i)
         {

            // prefetch
            if (i + 1 < neighbors.size() && visited_set.check(neighbors[i + 1]) == false)
               _base_storage->prefetch_vec_by_id(neighbors[i + 1]);

            // skip if visited
            auto &neighbor = neighbors[i];
            if (visited_set.check(neighbor))
               continue;
            visited_set.set(neighbor);

            // push to search queue
            search_queue.insert(neighbor, _distance_handler->compute(query, _base_storage->get_vector(neighbor), dim));
            num_cmps++;
         }
      }
      return num_cmps;
   }

   void UniNavGraph::save(std::string index_path_prefix, std::string results_path_prefix)
   {
      fs::create_directories(index_path_prefix);
      auto start_time = std::chrono::high_resolution_clock::now();

      // save meta data
      std::map<std::string, std::string> meta_data;
      statistics();
      meta_data["num_points"] = std::to_string(_num_points);
      meta_data["num_groups"] = std::to_string(_num_groups);
      meta_data["index_name"] = _index_name;
      meta_data["max_degree"] = std::to_string(_max_degree);
      meta_data["Lbuild"] = std::to_string(_Lbuild);
      meta_data["alpha"] = std::to_string(_alpha);
      meta_data["build_num_threads"] = std::to_string(_num_threads);
      meta_data["scenario"] = _scenario;
      meta_data["num_cross_edges"] = std::to_string(_num_cross_edges);
      meta_data["graph_num_edges"] = std::to_string(_graph_num_edges);
      meta_data["LNG_num_edges"] = std::to_string(_LNG_num_edges);
      meta_data["index_size(MB)"] = std::to_string(_index_size);
      meta_data["_index_size_add_rb(MB)"] = std::to_string(_index_size_add_rb);
      meta_data["index_time(ms)"] = std::to_string(_index_time);
      meta_data["label_processing_time(ms)"] = std::to_string(_label_processing_time);
      meta_data["build_graph_time(ms)"] = std::to_string(_build_graph_time);
      meta_data["build_vector_attr_graph_time(ms)"] = std::to_string(_build_vector_attr_graph_time);
      meta_data["cal_descendants_time(ms)"] = std::to_string(_cal_descendants_time);
      meta_data["cal_coverage_ratio_time(ms)"] = std::to_string(_cal_coverage_ratio_time);
      meta_data["build_LNG_time(ms)"] = std::to_string(_build_LNG_time);
      meta_data["build_cross_edges_time(ms)"] = std::to_string(_build_cross_edges_time);
      // FXY_ADD: 保存详细的跨组边构建时间到 meta 文件
      meta_data["cross_edge_step1_time(ms)"] = std::to_string(_cross_edge_step1_time_ms);
      meta_data["cross_edge_step2_acorn_time(ms)"] = std::to_string(_cross_edge_step2_acorn_time_ms);
      meta_data["cross_edge_step3_add_dist_edges_time(ms)"] = std::to_string(_cross_edge_step3_add_dist_edges_time_ms);
      meta_data["cross_edge_step4_add_hierarchy_edges_time(ms)"] = std::to_string(_cross_edge_step4_add_hierarchy_edges_time_ms);

      // FXY_ADD: 计算并保存 Trie 静态指标
      std::cout << "Calculating and saving Trie static metrics..." << std::endl;
      TrieStaticMetrics trie_metrics = _trie_index.calculate_static_metrics();
      meta_data["trie_label_cardinality"] = std::to_string(trie_metrics.label_cardinality);
      meta_data["trie_total_nodes"] = std::to_string(trie_metrics.total_nodes);
      meta_data["trie_avg_path_length"] = std::to_string(trie_metrics.avg_path_length);
      meta_data["trie_avg_branching_factor"] = std::to_string(trie_metrics.avg_branching_factor);
      // Save the detailed label frequency distribution to its own file for easier analysis
      std::string trie_freq_filename = results_path_prefix + "trie_label_frequency.csv";
      std::ofstream freq_file(trie_freq_filename);
      freq_file << "LabelID,Frequency\n";
      for (const auto &pair : trie_metrics.label_frequency)
         freq_file << pair.first << "," << pair.second << "\n";
      freq_file.close();
      std::cout << "- Trie label frequency distribution saved to " << trie_freq_filename << std::endl;

      std::string meta_filename = index_path_prefix + "meta";
      write_kv_file(meta_filename, meta_data);

      // save build_time to csv
      std::string build_time_filename = results_path_prefix + "build_time.csv";
      std::ofstream build_time_file(build_time_filename);
      build_time_file << "Index Name,Build Time (ms)\n";
      build_time_file << "index_time" << "," << _index_time << "\n";
      build_time_file << "label_processing_time" << "," << _label_processing_time << "\n";
      build_time_file << "build_graph_time" << "," << _build_graph_time << "\n";
      build_time_file << "build_vector_attr_graph_time" << "," << _build_vector_attr_graph_time << "\n";
      build_time_file << "cal_descendants_time" << "," << _cal_descendants_time << "\n";
      build_time_file << "cal_coverage_ratio_time" << "," << _cal_coverage_ratio_time << "\n";
      build_time_file << "build_LNG_time" << "," << _build_LNG_time << "\n";
      build_time_file << "build_cross_edges_time" << "," << _build_cross_edges_time << "\n";
      // FXY_ADD: 保存详细的跨组边构建时间到 CSV 文件
      build_time_file << "cross_edge_step1_time" << "," << _cross_edge_step1_time_ms << "\n";
      build_time_file << "cross_edge_step2_acorn_time" << "," << _cross_edge_step2_acorn_time_ms << "\n";
      build_time_file << "cross_edge_step3_add_dist_edges_time" << "," << _cross_edge_step3_add_dist_edges_time_ms << "\n";
      build_time_file << "cross_edge_step4_add_hierarchy_edges_time" << "," << _cross_edge_step4_add_hierarchy_edges_time_ms << "\n";
      build_time_file.close();

      // save vectors and label sets
      std::string bin_file = index_path_prefix + "vecs.bin";
      std::string label_file = index_path_prefix + "labels.txt";
      _base_storage->write_to_file(bin_file, label_file);

      // save group id to label set
      std::string group_id_to_label_set_filename = index_path_prefix + "group_id_to_label_set";
      write_2d_vectors(group_id_to_label_set_filename, _group_id_to_label_set);

      // save group id to range
      std::string group_id_to_range_filename = index_path_prefix + "group_id_to_range";
      write_2d_vectors(group_id_to_range_filename, _group_id_to_range);

      // save group id to entry point
      std::string group_entry_points_filename = index_path_prefix + "group_entry_points";
      write_1d_vector(group_entry_points_filename, _group_entry_points);

      // save group id to vec ids
      std::string group_id_to_vec_ids_filename = index_path_prefix + "group_id_to_vec_ids.dat";
      write_2d_vectors(group_id_to_vec_ids_filename, _group_id_to_vec_ids);
      std::cout << "group_id_to_vec_ids saved." << std::endl;

      // save new to old vec ids
      std::string new_to_old_vec_ids_filename = index_path_prefix + "new_to_old_vec_ids";
      write_1d_vector(new_to_old_vec_ids_filename, _new_to_old_vec_ids);

      // save trie index
      std::string trie_filename = index_path_prefix + "trie";
      _trie_index.save(trie_filename);

      // save graph data
      std::string graph_filename = index_path_prefix + "graph";
      _graph->save(graph_filename);

      std::string global_graph_filename = index_path_prefix + "global_graph";
      _global_graph->save(global_graph_filename);

      std::string global_vamana_entry_point_filename = index_path_prefix + "global_vamana_entry_point";
      write_one_T(global_vamana_entry_point_filename, _global_vamana_entry_point);

      // save LNG coverage ratio
      std::string coverage_ratio_filename = index_path_prefix + "lng_coverage_ratio";
      write_1d_vector(coverage_ratio_filename, _label_nav_graph->coverage_ratio);

      // save covered_sets in LNG
      std::string covered_sets_filename = index_path_prefix + "covered_sets";
      write_2d_vectors(covered_sets_filename, _label_nav_graph->covered_sets);
      std::cout << "LNG covered_sets saved." << std::endl;

      // save LNG descendant num
      std::string lng_descendants_num_filename = index_path_prefix + "lng_descendants_num";
      write_1d_pair_vector(lng_descendants_num_filename, _label_nav_graph->_lng_descendants_num);

      // save LNG descendants
      std::string lng_descendants_filename = index_path_prefix + "lng_descendants";
      write_2d_vectors(lng_descendants_filename, _label_nav_graph->_lng_descendants);

      // 保存 LNG 的核心图结构 (邻接表)
      std::string lng_out_neighbors_filename = index_path_prefix + "lng_out_neighbors.dat";
      write_2d_vectors(lng_out_neighbors_filename, _label_nav_graph->out_neighbors);
      std::cout << "LNG out_neighbors saved." << std::endl; // 增加一个打印，确认保存成功

      // save vector attr graph data
      std::string vector_attr_graph_filename = index_path_prefix + "vector_attr_graph";
      save_bipartite_graph(vector_attr_graph_filename);

      // // save _lng_descendants_bits and _covered_sets_bits
      // std::string lng_descendants_bits_filename = index_path_prefix + "lng_descendants_bits";
      // write_bitset_vector(lng_descendants_bits_filename, _lng_descendants_bits);
      // std::string covered_sets_bits_filename = index_path_prefix + "covered_sets_bits";
      // write_bitset_vector(covered_sets_bits_filename, _covered_sets_bits);

      // save lng_descendants_rb and _covered_sets_rb
      std::string lng_descendants_rb_filename = index_path_prefix + "lng_descendants_rb.bin";
      save_roaring_vector(lng_descendants_rb_filename, _lng_descendants_rb);
      std::string covered_sets_rb_filename = index_path_prefix + "covered_sets_rb.bin";
      save_roaring_vector(covered_sets_rb_filename, _covered_sets_rb);

      // print
      std::cout << "- Index saved in " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count() << " ms" << std::endl;
   }

   void UniNavGraph::load(std::string index_path_prefix,std::string selector_modle_prefix,const std::string &data_type, const std::string &acorn_index_path)
   {
      std::cout << "Loading index from " << index_path_prefix << " ..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      // load meta data
      std::string meta_filename = index_path_prefix + "meta";
      auto meta_data = parse_kv_file(meta_filename);
      _num_points = std::stoi(meta_data["num_points"]);
      _num_groups = std::stoi(meta_data["num_groups"]);

      // load vectors and label sets
      std::string bin_file = index_path_prefix + "vecs.bin";
      std::string label_file = index_path_prefix + "labels.txt";
      _base_storage = create_storage(data_type, false);
      _base_storage->load_from_file(bin_file, label_file);

      // load group id to label set
      std::string group_id_to_label_set_filename = index_path_prefix + "group_id_to_label_set";
      load_2d_vectors(group_id_to_label_set_filename, _group_id_to_label_set);

      // load group id to range
      std::string group_id_to_range_filename = index_path_prefix + "group_id_to_range";
      load_2d_vectors(group_id_to_range_filename, _group_id_to_range);

      // load group id to entry point
      std::string group_entry_points_filename = index_path_prefix + "group_entry_points";
      load_1d_vector(group_entry_points_filename, _group_entry_points);

      // load group id to vec ids
      std::string group_id_to_vec_ids_filename = index_path_prefix + "group_id_to_vec_ids.dat";
      load_2d_vectors(group_id_to_vec_ids_filename, _group_id_to_vec_ids);
      std::cout << "group_id_to_vec_ids loaded." << std::endl;

      // load new to old vec ids
      std::string new_to_old_vec_ids_filename = index_path_prefix + "new_to_old_vec_ids";
      load_1d_vector(new_to_old_vec_ids_filename, _new_to_old_vec_ids);

      // fxy_add: load old to new vec ids
      std::cout << "- Building reverse ID map (original_id -> new_id)..." << std::endl;
      _old_to_new_vec_ids.resize(_num_points);
      for (IdxType new_id = 0; new_id < _num_points; ++new_id) {
         IdxType old_id = _new_to_old_vec_ids[new_id];
         _old_to_new_vec_ids[old_id] = new_id;
      }
      std::cout << "- Reverse ID map built successfully." << std::endl;

      // load trie index
      std::string trie_filename = index_path_prefix + "trie";
      _trie_index.load(trie_filename);
      // --- 预计算并缓存 Trie 静态指标 ---
      std::cout << "Pre-calculating and caching Trie static metrics..." << std::endl;
      _trie_static_metrics = _trie_index.calculate_static_metrics();
      std::cout << "- Caching complete." << std::endl;

      // load graph data
      std::string graph_filename = index_path_prefix + "graph";
      _graph = std::make_shared<Graph>(_base_storage->get_num_points());
      _graph->load(graph_filename);

      // fxy_add:load global graph data
      std::string global_graph_filename = index_path_prefix + "global_graph";
      _global_graph = std::make_shared<Graph>(_base_storage->get_num_points());
      _global_graph->load(global_graph_filename);

      // fxy_add: load global vamana entry point
      std::string global_vamana_entry_point_filename = index_path_prefix + "global_vamana_entry_point";
      load_one_T(global_vamana_entry_point_filename, _global_vamana_entry_point);

      // fxy_add: load LNG coverage ratio
      std::string coverage_ratio_filename = index_path_prefix + "lng_coverage_ratio";
      std::cout << "_label_nav_graph->coverage_ratio size: " << _label_nav_graph->coverage_ratio.size() << std::endl;
      load_1d_vector(coverage_ratio_filename, _label_nav_graph->coverage_ratio);
      std::cout << "LNG coverage ratio loaded." << std::endl;

      // fxy_add: load LNG descendants num
      std::string lng_descendants_num_filename = index_path_prefix + "lng_descendants_num";
      load_1d_pair_vector(lng_descendants_num_filename, _label_nav_graph->_lng_descendants_num);
      std::cout << "LNG descendants num loaded." << std::endl;

      // fxy_add: load LNG descendants
      std::string lng_descendants_filename = index_path_prefix + "lng_descendants";
      load_2d_vectors(lng_descendants_filename, _label_nav_graph->_lng_descendants);

      // fxy_add: load covered_sets in LNG
      std::string covered_sets_filename = index_path_prefix + "covered_sets";
      load_2d_vectors(covered_sets_filename, _label_nav_graph->covered_sets);
      std::cout << "LNG covered_sets loaded." << std::endl;

      // fxy_add: load LNG out_neighbors
      std::string lng_out_neighbors_filename = index_path_prefix + "lng_out_neighbors.dat";
      load_2d_vectors(lng_out_neighbors_filename, _label_nav_graph->out_neighbors);
      std::cout << "LNG out_neighbors loaded." << std::endl;

      // // fxy_add: load  _lng_descendants_bits and _covered_sets_bits
      // std::string lng_descendants_bits_filename = index_path_prefix + "lng_descendants_bits";
      // load_bitset_vector(lng_descendants_bits_filename, _lng_descendants_bits);
      // std::cout << "_lng_descendants_bits loaded." << std::endl;
      // std::string covered_sets_bits_filename = index_path_prefix + "covered_sets_bits";
      // load_bitset_vector(covered_sets_bits_filename, _covered_sets_bits);
      // std::cout << "_covered_sets_bits loaded." << std::endl;

      // fxy_add: load lng_descendants_rb and _covered_sets_rb
      std::string lng_descendants_rb_filename = index_path_prefix + "lng_descendants_rb.bin";
      load_roaring_vector(lng_descendants_rb_filename, _lng_descendants_rb);
      std::cout << "_lng_descendants_rb loaded." << std::endl;
      std::string covered_sets_rb_filename = index_path_prefix + "covered_sets_rb.bin";
      load_roaring_vector(covered_sets_rb_filename, _covered_sets_rb);
      std::cout << "_covered_sets_rb loaded." << std::endl;
      std::cout << " _label_nav_graph->out_neighbors.size() = " << _label_nav_graph->out_neighbors.size() << std::endl;
      std::cout << " _num_groups = " << _num_groups << std::endl;

      //load idea1 selector
      std::string model_path = selector_modle_prefix + "trie_method_selector.onnx";
      std::cout << "Loading Trie method selector model from " << model_path << " ..." << std::endl;
      try {
         if (fs::exists(model_path)) {
               _trie_method_selector = std::make_unique<TrieMethodSelector>(model_path);
               std::cout << "- Model loaded successfully." << std::endl;
         } else {
               std::cerr << "- WARNING: Model file not found. Will use default method passed by command line." << std::endl;
               _trie_method_selector = nullptr;
         }
      } catch (const std::exception& e) {
         std::cerr << "- ERROR: Failed to load model: " << e.what() << ". Will use default method." << std::endl;
         _trie_method_selector = nullptr;
      }

      // === 新增逻辑：加载ACORN索引 === 
      if (!acorn_index_path.empty() && fs::exists(acorn_index_path)) {
            std::cout << "Loading ACORN index from " << acorn_index_path << " ..." << std::endl;
            try {
               // 1. 使用Faiss的IO函数读取原始索引
               faiss::Index* raw_index = faiss::read_index(acorn_index_path.c_str());
               
               // 2. 安全地将通用指针转换为ACORN专用指针
               _acorn_index = std::shared_ptr<faiss::IndexACORNFlat>(dynamic_cast<faiss::IndexACORNFlat*>(raw_index));
               
               if (_acorn_index) {
                  // 3. 重新关联元数据(标签)，这是ACORN正确工作的关键步骤
                  std::cout << "- Re-associating metadata with ACORN index..." << std::endl;
                  std::vector<std::vector<int>> metadata(_num_points);

                  // 从UNG的基础存储中提取标签，构建ACORN需要的元数据格式
                  for(size_t i = 0; i < _num_points; ++i) {
                        const auto& label_set = _base_storage->get_label_set(i);
                        // ACORN需要int类型的标签，并且需要排序
                        metadata[i].assign(label_set.begin(), label_set.end());
                        std::sort(metadata[i].begin(), metadata[i].end());
                  }
                  // 将构建好的元数据设置到ACORN索引实例中
                  _acorn_index->set_metadata(metadata);
                  
                  std::cout << "- ACORN index loaded and metadata associated successfully." << std::endl;
               } else {
                  std::cerr << "ERROR: Failed to cast loaded index to faiss::IndexACORNFlat. Is the index file correct?" << std::endl;
                  delete raw_index; // 防止因转换失败导致的内存泄漏
               }
            } catch (const std::exception& e) {
               std::cerr << "ERROR: Exception caught while loading ACORN index: " << e.what() << std::endl;
               _acorn_index = nullptr; // 确保加载失败时指针为空
            }
      } else if (!acorn_index_path.empty()) {
            std::cerr << "Warning: ACORN index path provided, but file not found at: " << acorn_index_path << std::endl;
      }
      // +++ === ACORN索引加载逻辑结束 === +++

      // print
      std::cout << "- Index loaded in " << std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count() << " ms" << std::endl;
   }

   void UniNavGraph::statistics()
   {

      // number of edges in the unified navigating graph
      _graph_num_edges = 0;
      for (IdxType i = 0; i < _num_points; ++i)
         _graph_num_edges += _graph->neighbors[i].size();

      // number of edges in the label navigating graph
      _LNG_num_edges = 0;
      if (_label_nav_graph != nullptr)
         for (IdxType i = 1; i <= _num_groups; ++i)
            _LNG_num_edges += _label_nav_graph->out_neighbors[i].size();

      // index size
      _index_size = 0;
      for (IdxType i = 1; i <= _num_groups; ++i)
         _index_size += _group_id_to_label_set[i].size() * sizeof(LabelType);
      _index_size += _group_id_to_range.size() * sizeof(IdxType) * 2;
      _index_size += _group_entry_points.size() * sizeof(IdxType);
      _index_size += _new_to_old_vec_ids.size() * sizeof(IdxType);
      _index_size += _trie_index.get_index_size();
      _index_size += _graph->get_index_size();

      _index_size_add_rb = _index_size; // 记录不含 Roaring Bitmap 部分的索引大小
      // fxy_add:计算为混合搜索策略预处理的 Roaring Bitmaps 的大小
      if (!_lng_descendants_rb.empty()) {
      for (const auto& rb : _lng_descendants_rb) {
            _index_size_add_rb += rb.getSizeInBytes();
      }
      }
      if (!_covered_sets_rb.empty()) {
         for (const auto& rb : _covered_sets_rb) {
               _index_size_add_rb += rb.getSizeInBytes();
         }
      }


      // return as MB
      _index_size /= 1024 * 1024;
      _index_size_add_rb /= 1024 * 1024;
   }
}