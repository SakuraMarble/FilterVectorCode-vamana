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
         _precompute_lng_node_depths();

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
         std::cout << "Only one candidate found. Returning it as the minimal super set." << std::endl;
         min_super_set_ids.emplace_back(candidates[0]->group_id);
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

      /*// --- 4. 使用原子计数器控制打印次数 ---
      // fetch_add(1) 会原子地将计数器加1，并返回增加前的值。所以这个条件对于前10次调用 (0到9) 会成立。
      if (print_counter.fetch_add(1) < 10)
      {
// 使用 #pragma omp critical 来防止多线程输出混乱
#pragma omp critical
         {
            std::cout << "--- get_min_super_sets Performance Analysis---\n"
                      << "  - Trie查找总耗时: " << time_trie_lookup << " ms\n"
                      << "  - 排序耗时:       " << time_sorting << " ms\n"
                      << "  - 过滤循环耗时:   " << time_filtering_loop << " ms\n"
                      << "Total Function Time: " << function_total_time << " ms\n"
                      << "-----------------------------------------------------------\n"
                      << std::endl;
         }
      }*/
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
   // fxy_add：使用广度优先搜索（BFS）来确保所有后代都被找到
   void UniNavGraph::get_descendants_info()
   {
      std::cout << "Calculating descendants info (using corrected BFS method)..." << std::endl;
      using PairType = std::pair<IdxType, int>;

      // ------------------- 修改点 1 -------------------
      // 将大小从 _num_groups 改为 _num_groups + 1
      // 以便能安全地使用 1-based 索引 (group_id)
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

         // ------------------- 修改点 2 -------------------
         // 使用 group_id 直接作为索引，而不是 group_id - 1
         // 这与容器的新大小相匹配
         descendants_num[group_id] = PairType(group_id, temp_set.size());
         descendants_set[group_id] = std::move(temp_set);
      }

      // 单线程写入类成员变量 (这部分无需修改)
      _label_nav_graph->_lng_descendants_num = std::move(descendants_num);
      _label_nav_graph->_lng_descendants = std::move(descendants_set);

      // 计算平均后代个数 (这部分无需修改)
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
                                   bool is_ung_more_entry, const std::vector<IdxType> &true_query_group_ids)
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

         // 新增指标记录
         stats.query_length = query_labels.size(); // 记录查询长度
         if (!query_labels.empty())                // 记录候选集大小
            stats.candidate_set_size = _trie_index.get_candidate_count_for_label(query_labels.back());
         else
            stats.candidate_set_size = 0;

         // 计算入口组信息
         auto get_entry_group_start_time = std::chrono::high_resolution_clock::now();
         std::vector<IdxType> entry_group_ids;

         // 是否使用更多的入口组
         if (!is_ung_more_entry)
         {
            static std::atomic<int> trie_debug_print_counter{0};
            auto get_min_super_sets_satrt_time = std::chrono::high_resolution_clock::now();
            get_min_super_sets_debug(query_labels, entry_group_ids, true, true,
                                     trie_debug_print_counter, is_new_trie_method, is_rec_more_start, stats);
            stats.get_min_super_sets_time_ms = std::chrono::duration<double, std::milli>(
                                                   std::chrono::high_resolution_clock::now() - get_min_super_sets_satrt_time)
                                                   .count();
            stats.num_entry_points = entry_group_ids.size();
         }
         else
         {

            IdxType true_group_id = 0;
            if (id < true_query_group_ids.size())
            {
               true_group_id = true_query_group_ids[id]; // 获取当前查询的真实组ID
            }
            std::vector<IdxType> base_entry_groups;
            static std::atomic<int> trie_debug_print_counter{0};
            auto get_min_super_sets_satrt_time = std::chrono::high_resolution_clock::now();
            get_min_super_sets_debug(query_labels, base_entry_groups, true, true,
                                     trie_debug_print_counter, is_new_trie_method, is_rec_more_start, stats);
            stats.get_min_super_sets_time_ms = std::chrono::duration<double, std::milli>(
                                                   std::chrono::high_resolution_clock::now() - get_min_super_sets_satrt_time)
                                                   .count();
            const size_t extra_k = base_entry_groups.size() / 5;
            const SelectionMode current_mode = SelectionMode::SizeAndDistance;
            const double beta_value = 1.0;
            entry_group_ids = select_entry_groups(
                base_entry_groups,
                current_mode,
                extra_k,
                beta_value,
                true_group_id);
            stats.num_entry_points = entry_group_ids.size();
         }
         stats.get_group_entry_time_ms = std::chrono::duration<double, std::milli>(
                                             std::chrono::high_resolution_clock::now() - get_entry_group_start_time)
                                             .count();

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
         stats.is_global_search = use_global_search;

         // 4. 执行搜索
         if (use_global_search)
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

   void UniNavGraph::load(std::string index_path_prefix, const std::string &data_type)
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

      // load trie index
      std::string trie_filename = index_path_prefix + "trie";
      _trie_index.load(trie_filename);

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

      // return as MB
      _index_size /= 1024 * 1024;
   }

   // ===================================begin：生成query task========================================
   // fxy_add:计算一个标签子集在整个数据集中被包含的次数
   int UniNavGraph::count_subset_occurrences(const std::vector<unsigned int> &sorted_subset)
   {
      // 步骤1: 检查缓存中是否已有记录
      auto it = _subset_count_cache.find(sorted_subset);
      if (it != _subset_count_cache.end())
      {
         return it->second; // 缓存命中，直接返回结果
      }

      // 步骤2: 缓存未命中，遍历整个数据集进行计算
      int count = 0;
      for (ANNS::IdxType i = 0; i < _num_points; ++i)
      {
         const auto &base_labels = _base_storage->get_label_set(i);

         // 优化: 如果基向量的标签数量少于子集，它不可能包含该子集
         if (base_labels.size() < sorted_subset.size())
         {
            continue;
         }

         // 检查 base_labels 是否是 sorted_subset 的超集
         bool is_superset = true;
         for (int label_in_subset : sorted_subset)
         {
            // 在基向量的标签集中查找是否存在子集中的每一个标签
            if (std::find(base_labels.begin(), base_labels.end(), label_in_subset) == base_labels.end())
            {
               is_superset = false; // 只要有一个标签不匹配，就不是超集
               break;
            }
         }

         if (is_superset)
         {
            count++;
         }
      }

      // 步骤3: 将计算结果存入缓存，供后续使用
      _subset_count_cache[sorted_subset] = count;
      return count;
   }

   // fxy_add: 根据LNG中f点，生成查询向量和标签
   void UniNavGraph::query_generate(std::string &output_prefix, int n, float keep_prob, int K, bool stratified_sampling, bool verify)
   {
      // 在每次调用开始时清空缓存，确保不同生成任务之间的结果是独立的
      _subset_count_cache.clear();

      std::ofstream fvec_file(output_prefix + FVEC_EXT, std::ios::binary);
      std::ofstream txt_file(output_prefix + "_labels" + TXT_EXT);

      uint32_t dim = _base_storage->get_dim();
      std::cout << "Vector dimension: " << dim << std::endl;
      std::cout << "Number of points: " << _num_points << std::endl;
      std::cout << "Minimum occurrence threshold (K): " << K << std::endl;

      // 随机数生成器
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

      std::vector<QueryTask> all_queries;
      size_t total_queries = 0;

      // min(10000个查询,_num_groups)
      total_queries = std::min(10000, int(_num_groups));
      for (ANNS::IdxType group_id = 1; group_id <= total_queries; ++group_id)
      {
         auto [start, end] = _group_id_to_range[group_id];
         size_t group_size = end - start;
         if (group_size == 0)
            continue;

         // 计算该组实际采样数量
         int sample_num = n;
         if (stratified_sampling)
         {
            sample_num = std::max(1, static_cast<int>(n * group_size / _num_points)); // 分层采样：按组大小比例调整采样数
         }
         sample_num = std::min(sample_num, static_cast<int>(group_size));

         // 非重复采样 (逻辑不变)
         std::vector<ANNS::IdxType> vec_ids(group_size);
         std::iota(vec_ids.begin(), vec_ids.end(), start);
         std::shuffle(vec_ids.begin(), vec_ids.end(), gen);

         for (int i = 0; i < sample_num; ++i) // 每个组采样的个数
         {
            ANNS::IdxType vec_id = vec_ids[i];
            if (_base_storage->get_label_set(vec_id).empty())
            {
               continue; // 跳过无 base 属性的向量
            }
            QueryTask task;
            task.vec_id = vec_id;

            // 生成查询标签集
            for (auto label : _group_id_to_label_set[group_id])
            {
               if (prob_dist(gen) <= keep_prob)
               {
                  task.labels.push_back(label);
               }
            }

            // 确保至少保留一个标签
            if (task.labels.empty())
            {
               task.labels.push_back(*_group_id_to_label_set[group_id].begin());
            }

            // ==================== 新增核心逻辑: 检查出现次数是否满足 K ====================
            // 为了能够使用map作为缓存的key，必须对标签进行排序，以获得一个唯一的表示
            auto sorted_labels = task.labels;
            std::sort(sorted_labels.begin(), sorted_labels.end());

            // 调用辅助函数计算或从缓存中获取该标签组合的出现次数
            int occurrences = count_subset_occurrences(sorted_labels);

            // 如果出现次数小于K，则这个查询不合格，直接丢弃并开始下一次循环
            if (occurrences < K)
            {
               continue;
            }
            // ========================================================================

            // 验证标签是组的子集
            if (verify)
            {
               auto &group_labels = _group_id_to_label_set[group_id];
               for (auto label : task.labels)
               {
                  assert(std::find(group_labels.begin(), group_labels.end(), label) != group_labels.end());
               }
            }

            all_queries.push_back(task);
         }
      }

      // 写入fvecs文件前验证标签
      std::ofstream verify_file(output_prefix + "_verify.txt");
      for (const auto &task : all_queries)
      {
         const auto &original_labels = _base_storage->get_label_set(task.vec_id);
         verify_file << task.vec_id << " base_labels:";
         for (auto l : original_labels)
            verify_file << " " << l;
         verify_file << " query_labels:";
         for (auto l : task.labels)
            verify_file << " " << l;
         verify_file << "\n";

         if (verify)
         {
            for (auto label : task.labels)
            {
               if (std::find(original_labels.begin(), original_labels.end(), label) == original_labels.end())
               {
                  std::cerr << "Error: Label " << label << " not found in original label set for vector " << task.vec_id << std::endl;
                  throw std::runtime_error("Label verification failed");
               }
            }
         }
      }
      verify_file.close();

      // 写入fvecs文件
      for (const auto &task : all_queries)
      {
         const char *vec_data = _base_storage->get_vector(task.vec_id);
         fvec_file.write((char *)&dim, sizeof(uint32_t));
         fvec_file.write(vec_data, dim * sizeof(float));
      }

      // 写入txt文件
      for (const auto &task : all_queries)
      {
         for (auto label : task.labels)
         {
            txt_file << label << ",";
         }
         txt_file << "\n";
      }

      std::cout << "Generated " << all_queries.size() << " queries (that meet K=" << K << " criteria)\n";
      std::cout << "FVECS file: " << output_prefix + FVEC_EXT << "\n";
      std::cout << "TXT file: " << output_prefix + "_labels" + TXT_EXT << "\n";
   }

   // fxy_add:根据LNG中f点，生成多个查询任务
   void UniNavGraph::generate_multiple_queries(
       std::string dataset,
       UniNavGraph &index, int K,
       const std::string &base_output_path,
       int num_sets,
       int n_per_set,
       float keep_prob,
       bool stratified_sampling,
       bool verify)
   {
      std::cout << "enter generate_multiple_queries" << std::endl;
      // namespace fs = std::filesystem; // 假设 std::filesystem 已经可用

      fs::create_directories(base_output_path);

      for (int i = 1; i <= num_sets; ++i)
      {
         std::cout << "Generating query set " << i << "..." << std::endl;
         std::string folder_name = base_output_path;
         fs::create_directories(folder_name);
         std::string output_prefix = folder_name + "/" + dataset + "_query";
         index.query_generate(output_prefix, n_per_set, keep_prob, K, stratified_sampling, verify);
         std::cout << "Generated query set " << i << " at " << folder_name << std::endl;
      }
   }

   // fxy_add:极端方法1数据，生成覆盖率高的查询任务
   void UniNavGraph::generate_queries_method1_high_coverage(std::string &output_prefix, std::string dataset, int query_n, std::string &base_label_file, float coverage_threshold)
   {
      // Step 1: 读取base_label_file并统计每列标签频率
      std::ifstream label_file(base_label_file);
      if (!label_file.is_open())
      {
         std::cerr << "Failed to open label file: " << base_label_file << std::endl;
         return;
      }

      std::vector<std::unordered_map<int, int>> column_label_counts;
      std::vector<std::vector<int>> all_label_rows;
      std::string line;
      int total_lines = 0;
      size_t num_columns = 0;

      while (std::getline(label_file, line))
      {
         std::vector<int> labels;
         std::stringstream ss(line);
         std::string label_str;

         while (std::getline(ss, label_str, ','))
         {
            labels.push_back(std::stoi(label_str));
         }

         if (!labels.empty())
         {
            if (num_columns == 0)
            {
               num_columns = labels.size();
               column_label_counts.resize(num_columns);
            }
            if (labels.size() == num_columns)
            {
               for (size_t col = 0; col < num_columns; ++col)
               {
                  column_label_counts[col][labels[col]]++;
               }
               all_label_rows.push_back(labels);
               total_lines++;
            }
         }
      }
      label_file.close();

      if (total_lines == 0 || num_columns == 0)
      {
         std::cerr << "No valid labels found in base label file" << std::endl;
         return;
      }

      // Step 2: 计算每列标签覆盖率，筛选高覆盖率标签
      struct LabelInfo
      {
         int label;
         float coverage;
         size_t column; // 记录标签所属列
      };
      std::vector<std::vector<LabelInfo>> high_coverage_labels_per_column(num_columns);
      std::vector<LabelInfo> all_high_coverage_labels;

      for (size_t col = 0; col < num_columns; ++col)
      {
         for (const auto &entry : column_label_counts[col])
         {
            float coverage = static_cast<float>(entry.second) / total_lines;
            if (coverage >= coverage_threshold)
            {
               high_coverage_labels_per_column[col].push_back({entry.first, coverage, col});
               all_high_coverage_labels.push_back({entry.first, coverage, col});
            }
         }
         // 按覆盖率降序排序
         std::sort(high_coverage_labels_per_column[col].begin(), high_coverage_labels_per_column[col].end(),
                   [](const auto &a, const auto &b)
                   { return a.coverage > b.coverage; });
      }

      // Step 3: 构建最高覆盖率标签组合并计算幂集覆盖率
      std::vector<int> max_coverage_labels;
      std::vector<size_t> label_to_column;

      // 动态选择每列覆盖率最高的标签
      for (size_t col = 0; col < num_columns; ++col)
      {
         int max_label = 0;
         float max_coverage = 0.0f;
         for (const auto &entry : column_label_counts[col])
         {
            float coverage = static_cast<float>(entry.second) / total_lines;
            if (coverage > max_coverage)
            {
               max_coverage = coverage;
               max_label = entry.first;
            }
         }
         if (max_coverage > 0.0f)
         {
            max_coverage_labels.push_back(max_label);
            label_to_column.push_back(col);
         }
      }

      std::vector<std::pair<std::vector<int>, float>> high_coverage_combinations;

      // 生成幂集（非空子集）
      if (!max_coverage_labels.empty())
      {
         int total_subsets = (1 << max_coverage_labels.size()) - 1;
         for (int mask = 1; mask <= total_subsets; ++mask)
         {
            std::vector<int> subset;
            std::vector<size_t> subset_columns; // 记录子集对应的列
            for (size_t i = 0; i < max_coverage_labels.size(); ++i)
            {
               if (mask & (1 << i))
               {
                  subset.push_back(max_coverage_labels[i]);
                  subset_columns.push_back(label_to_column[i]);
               }
            }

            // 计算子集覆盖率
            int combo_count = 0;
            for (const auto &row : all_label_rows)
            {
               bool match = true;
               for (size_t i = 0; i < subset.size(); ++i)
               {
                  if (row[subset_columns[i]] != subset[i])
                  {
                     match = false;
                     break;
                  }
               }
               if (match)
                  combo_count++;
            }
            float coverage = static_cast<float>(combo_count) / total_lines;
            if (coverage >= coverage_threshold)
            {
               high_coverage_combinations.emplace_back(subset, coverage);
            }
         }
      }

      // 按覆盖率降序排序组合
      std::sort(high_coverage_combinations.begin(), high_coverage_combinations.end(),
                [](const auto &a, const auto &b)
                { return a.second > b.second; });

      // Step 4: 准备候选标签（组合 + 单个标签）
      struct CandidateLabel
      {
         std::vector<int> labels; // 组合或单个标签
         float coverage;
         bool is_combination; // 是否为组合
      };
      std::vector<CandidateLabel> candidate_labels;

      // 添加高覆盖率组合
      for (const auto &combo : high_coverage_combinations)
      {
         candidate_labels.push_back({combo.first, combo.second, true});
      }

      // 添加单个高覆盖率标签
      for (const auto &label_info : all_high_coverage_labels)
      {
         candidate_labels.push_back({{label_info.label}, label_info.coverage, false});
      }

      if (candidate_labels.empty())
      {
         std::cerr << "No labels or combinations with coverage >= " << coverage_threshold << std::endl;
         return;
      }

      // Step 5: 准备输出文件
      std::ofstream txt_file(output_prefix + "/" + dataset + "_query_labels.txt");
      std::ofstream fvec_file(output_prefix + "/" + dataset + "_query.fvecs", std::ios::binary);
      std::ofstream stats_file(output_prefix + "/" + dataset + "_query_stats.txt");

      if (!txt_file.is_open() || !fvec_file.is_open() || !stats_file.is_open())
      {
         std::cerr << "Failed to open output files" << std::endl;
         return;
      }

      // Step 6: 写入统计信息
      stats_file << "Coverage Threshold: " << coverage_threshold << std::endl;
      stats_file << "Number of Columns: " << num_columns << std::endl;
      stats_file << "Total Rows: " << total_lines << std::endl;
      stats_file << "\nPer-Column Max-Coverage Labels:\n";
      for (size_t col = 0; col < num_columns; ++col)
      {
         stats_file << "Column " << col + 1 << ":\n";
         // 找到该列覆盖率最高的标签
         int max_label = 0;
         float max_coverage = 0.0f;
         for (const auto &entry : column_label_counts[col])
         {
            float coverage = static_cast<float>(entry.second) / total_lines;
            if (coverage > max_coverage)
            {
               max_coverage = coverage;
               max_label = entry.first;
            }
         }
         if (max_coverage > 0.0f)
         {
            stats_file << "  Label: " << max_label << ", Coverage: " << max_coverage << "\n";
         }
         else
         {
            stats_file << "  No labels found\n";
         }
      }
      stats_file << "\nHigh-Coverage Combinations (from max coverage labels):\n";
      for (const auto &combo : high_coverage_combinations)
      {
         stats_file << "  Labels: ";
         for (size_t i = 0; i < combo.first.size(); ++i)
         {
            stats_file << combo.first[i];
            if (i != combo.first.size() - 1)
               stats_file << ",";
         }
         stats_file << ", Coverage: " << combo.second << "\n";
      }

      // Step 7: 生成查询任务
      uint32_t dim = _base_storage->get_dim();
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<size_t> label_dis(0, candidate_labels.size() - 1);
      std::uniform_int_distribution<ANNS::IdxType> vec_dis(0, _base_storage->get_num_points() - 1);
      std::unordered_set<ANNS::IdxType> used_vec_ids;

      for (int i = 0; i < query_n; ++i)
      {
         // 选择候选标签（循环使用）
         const auto &candidate = candidate_labels[i % candidate_labels.size()];

         // 写入标签文件
         for (size_t j = 0; j < candidate.labels.size(); ++j)
         {
            txt_file << candidate.labels[j];
            if (j != candidate.labels.size() - 1)
               txt_file << ",";
         }
         txt_file << std::endl;

         // 随机选择不重复的向量
         ANNS::IdxType vec_id;
         do
         {
            vec_id = vec_dis(gen);
         } while (used_vec_ids.count(vec_id) > 0 && used_vec_ids.size() < _base_storage->get_num_points());

         if (used_vec_ids.size() >= _base_storage->get_num_points())
         {
            std::cerr << "Warning: Not enough unique vectors" << std::endl;
            break;
         }
         used_vec_ids.insert(vec_id);

         // 写入向量文件
         const char *vec_data = _base_storage->get_vector(vec_id);
         fvec_file.write((char *)&dim, sizeof(uint32_t));
         fvec_file.write(vec_data, dim * sizeof(float));
      }

      txt_file.close();
      fvec_file.close();
      stats_file.close();

      // Step 8: 输出统计信息到控制台
      std::cout << "Generated " << std::min(query_n, (int)used_vec_ids.size()) << " high-coverage queries" << std::endl;
      std::cout << "Labels written to: " << output_prefix + "/" + dataset + "_query_labels.txt" << std::endl;
      std::cout << "Vectors written to: " << output_prefix + "/" + dataset + "_query.fvecs" << std::endl;
      std::cout << "Statistics written to: " << output_prefix + "/" + dataset + "_query_stats.txt" << std::endl;
   }

   // fxy_add:极端方法1数据，生成覆盖率低的查询任务:选出覆盖率在 (0, coverage_threshold] 区间且出现次数 ≥ K 的组合
   void UniNavGraph::generate_queries_method1_low_coverage(
       std::string &output_prefix,
       std::string dataset,
       int query_n,
       std::string &base_label_file,
       int num_of_per_query_labels,
       float coverage_threshold,
       int K)
   {
      // Step 1: 从标签文件加载并分析标签分布
      std::ifstream label_file(base_label_file);
      if (!label_file.is_open())
      {
         std::cerr << "Failed to open label file: " << base_label_file << std::endl;
         return;
      }

      // 统计标签出现频率
      std::unordered_map<int, int> label_counts;
      std::vector<std::vector<int>> all_label_sets;
      std::string line;

      while (std::getline(label_file, line))
      {
         std::vector<int> labels;
         std::stringstream ss(line);
         std::string label_str;

         while (std::getline(ss, label_str, ','))
         {
            int label = std::stoi(label_str);
            labels.push_back(label);
            label_counts[label]++;
         }

         std::sort(labels.begin(), labels.end());
         all_label_sets.push_back(labels);
      }
      label_file.close();

      // 如果没有标签数据，使用默认标签
      if (all_label_sets.empty())
      {
         std::cerr << "No label data found, using default label 1" << std::endl;
         all_label_sets.push_back({1});
         label_counts[1] = 1;
      }

      // Step 2: 识别低覆盖率标签组合（排除覆盖率为0的组合）
      struct LabelSetInfo
      {
         std::vector<int> labels;
         int count;
         float coverage;
      };

      std::vector<LabelSetInfo> valid_low_coverage_sets;
      int total_vectors = all_label_sets.size();

      // 统计标签组合的覆盖率（最多num_of_per_query_labels个标签的组合）
      std::map<std::vector<int>, int> combination_counts;
      for (const auto &labels : all_label_sets)
      {
         // 生成所有可能的1-num_of_per_query_labels标签组合
         for (int k = 1; k <= num_of_per_query_labels && k <= labels.size(); ++k)
         {
            std::vector<bool> mask(labels.size(), false);
            std::fill(mask.begin(), mask.begin() + k, true);

            do
            {
               std::vector<int> combination;
               for (size_t i = 0; i < mask.size(); ++i)
               {
                  if (mask[i])
                     combination.push_back(labels[i]);
               }
               combination_counts[combination]++;
            } while (std::prev_permutation(mask.begin(), mask.end()));
         }
      }

      // 筛选出覆盖率大于0且低于阈值的组合
      for (const auto &entry : combination_counts)
      {
         // 跳过覆盖率为0的组合
         if (entry.second == 0)
            continue;

         float coverage = static_cast<float>(entry.second) / total_vectors;
         if (coverage <= coverage_threshold && entry.second >= K)
         {
            valid_low_coverage_sets.push_back({entry.first, entry.second, coverage});
         }
      }

      // 按覆盖率升序排序
      std::sort(valid_low_coverage_sets.begin(), valid_low_coverage_sets.end(),
                [](const auto &a, const auto &b)
                {
                   return a.coverage < b.coverage;
                });

      // 检查是否有足够的有效组合
      if (valid_low_coverage_sets.empty())
      {
         std::cerr << "Error: No valid low-coverage label combinations found (all either have coverage=0 or > "
                   << coverage_threshold << ")" << std::endl;
         return;
      }

      // Step 3: 生成查询文件和标签
      std::ofstream txt_file(output_prefix + "/" + dataset + "_query_labels.txt");
      std::ofstream txt__coverage_file(output_prefix + "/" + dataset + "_query_and_coverage_labels.txt");
      std::ofstream fvec_file(output_prefix + "/" + dataset + "_query.fvecs", std::ios::binary);

      if (!txt_file.is_open() || !fvec_file.is_open())
      {
         std::cerr << "Failed to open output files." << std::endl;
         return;
      }

      uint32_t dim = _base_storage->get_dim();
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<ANNS::IdxType> vec_dis(0, _base_storage->get_num_points() - 1);
      std::unordered_set<ANNS::IdxType> used_vec_ids;

      // 选择前query_n个最稀有的有效标签组合
      int queries_generated = 0;
      for (size_t i = 0; i < valid_low_coverage_sets.size() && queries_generated < query_n; ++i)
      {
         const auto &label_set = valid_low_coverage_sets[i].labels;

         // 写入标签文件
         for (size_t j = 0; j < label_set.size(); ++j)
         {
            txt_file << label_set[j];
            if (j != label_set.size() - 1)
               txt_file << ",";

            txt__coverage_file << label_set[j];
            if (j != label_set.size() - 1)
               txt__coverage_file << ",";
         }
         txt__coverage_file << " coverage:" << valid_low_coverage_sets[i].coverage;
         txt_file << std::endl;
         txt__coverage_file << std::endl;

         // 随机选择一个向量（确保不重复）
         ANNS::IdxType vec_id;
         do
         {
            vec_id = vec_dis(gen);
         } while (used_vec_ids.count(vec_id) > 0 &&
                  used_vec_ids.size() < _base_storage->get_num_points());

         used_vec_ids.insert(vec_id);
         const char *vec_data = _base_storage->get_vector(vec_id);

         // 写入向量文件
         fvec_file.write((char *)&dim, sizeof(uint32_t));
         fvec_file.write(vec_data, dim * sizeof(float));

         queries_generated++;
      }

      txt_file.close();
      txt__coverage_file.close();
      fvec_file.close();

      std::cout << "Generated " << queries_generated << " low-coverage queries with 0 < coverage <= "
                << coverage_threshold << std::endl;
      std::cout << "Labels written to: " << output_prefix + "/" + dataset + "_lowcov_query_labels.txt" << std::endl;
      std::cout << "Vectors written to: " << output_prefix + "/" + dataset + "_lowcov_query.fvecs" << std::endl;

      // 输出统计信息
      if (!valid_low_coverage_sets.empty())
      {
         std::cout << "\nStatistics of generated low-coverage queries:" << std::endl;
         std::cout << "Min coverage: " << valid_low_coverage_sets.front().coverage << std::endl;
         std::cout << "Max coverage: " << valid_low_coverage_sets.back().coverage << std::endl;
         std::cout << "Median coverage: "
                   << valid_low_coverage_sets[valid_low_coverage_sets.size() / 2].coverage << std::endl;
      }
   }

   // fxy_add: 极端方法2的高覆盖率查询任务生成(使用Deepener处理过的)
   void UniNavGraph::generate_queries_method2_high_coverage(int N, int K, int top_M_trees, std::string dataset, const std::string &output_prefix, const std::string &base_label_tree_roots)
   {
      std::cout << "\n==================================================" << std::endl;
      std::cout << "--- 开始生成极端查询任务 ---" << std::endl;

      if (_num_groups == 0)
      {
         std::cerr << "错误: 索引未构建或不包含任何分组。" << std::endl;
         return;
      }

      // --- 阶段1: 读取并验证用户指定的概念根 ---
      std::cout << "\n步骤1: 读取并解析用户指定的概念根文件..." << std::endl;
      std::cout << "  - 概念根文件路径: " << base_label_tree_roots << std::endl;
      std::vector<LabelType> conceptual_root_labels;
      std::ifstream roots_file(base_label_tree_roots);
      if (!roots_file.is_open())
      {
         std::cerr << "FATAL ERROR: 无法打开概念根文件！程序无法继续。" << std::endl;
         return;
      }
      LabelType root_label_from_file;
      while (roots_file >> root_label_from_file)
      {
         conceptual_root_labels.push_back(root_label_from_file);
      }
      roots_file.close();

      if (conceptual_root_labels.empty())
      {
         std::cerr << "FATAL ERROR: 从 " << base_label_tree_roots << " 读取到的概念根数量为0，无法生成任何查询。" << std::endl;
         return;
      }
      std::cout << "  - 成功读取 " << conceptual_root_labels.size() << " 个概念树根标签，将基于这些标签生成查询。" << std::endl;

      // --- 阶段2: 分析并打印物理图结构 (仅用于诊断) ---
      std::cout << "\n步骤2: [仅供诊断] 分析图的物理结构..." << std::endl;
      {
         // 这部分代码块的计算结果仅用于打印诊断信息，不影响后续查询生成
         std::map<IdxType, size_t> physical_root_coverage;
         std::unordered_map<IdxType, IdxType> group_to_root_group;
         std::vector<int> parent(_num_groups + 1, 0);
         for (ANNS::IdxType i = 1; i <= _num_groups; ++i)
         {
            for (auto child : _label_nav_graph->out_neighbors[i])
               parent[child] = i;
         }
         for (ANNS::IdxType i = 1; i <= _num_groups; ++i)
         {
            ANNS::IdxType current = i;
            while (parent[current] != 0)
               current = parent[current];
            group_to_root_group[i] = current;
         }
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            physical_root_coverage[group_to_root_group[group_id]] += _group_id_to_vec_ids[group_id].size();
         }

         std::cout << "  [DIAGNOSTICS] 物理根详情如下:" << std::endl;
         std::cout << "  --------------------------------------------------" << std::endl;
         for (auto const &[root_id, coverage] : physical_root_coverage)
         {
            std::cout << "  - 物理根 Group ID: " << root_id << ", 覆盖点数: " << coverage << std::endl;
            if (root_id < _group_id_to_label_set.size())
            {
               const auto &labels = _group_id_to_label_set.at(root_id);
               if (labels.empty())
               {
                  std::cout << "    > 注意: 此物理根的标签集为空 (通常代表无标签向量的总根)。" << std::endl;
               }
               else
               {
                  std::cout << "    > 物理根标签: ";
                  for (const auto &label : labels)
                     std::cout << label << " ";
                  std::cout << std::endl;
               }
            }
         }
         std::cout << "  --------------------------------------------------" << std::endl;
         std::cout << "  - 诊断信息结束，后续流程将严格按照您提供的概念根执行。" << std::endl;
      }

      // --- 阶段3: 基于概念根构建查询树信息 ---
      std::cout << "\n步骤3: 基于您提供的概念根构建查询树..." << std::endl;
      std::vector<TreeInfo> sorted_trees;
      std::map<IdxType, std::vector<IdxType>> per_tree_groups; // 存储每棵概念树包含的所有group ID
      std::map<IdxType, int> tree_max_depth;                   // 存储每棵概念树的最大深度

      for (LabelType root_label : conceptual_root_labels)
      {
         auto node = _trie_index.find_exact_match({root_label});
         if (!node)
         {
            std::cerr << "  - 警告: 无法在Trie索引中找到概念根标签 " << root_label << "，将跳过此根。" << std::endl;
            continue;
         }

         IdxType conceptual_root_group_id = node->group_id;
         std::cout << "  - 正在处理概念根: " << root_label << " (Group ID: " << conceptual_root_group_id << ")" << std::endl;

         // 使用BFS/DFS遍历计算概念树的覆盖范围和所有成员
         size_t current_tree_coverage = 0;
         int current_max_depth = 0;
         std::vector<IdxType> groups_in_this_tree;
         std::vector<bool> visited(_num_groups + 1, false);
         std::queue<IdxType> q;

         q.push(conceptual_root_group_id);
         visited[conceptual_root_group_id] = true;

         while (!q.empty())
         {
            IdxType current_group = q.front();
            q.pop();

            groups_in_this_tree.push_back(current_group);
            current_tree_coverage += _group_id_to_vec_ids[current_group].size();
            current_max_depth = std::max(current_max_depth, (int)_group_id_to_label_set[current_group].size());

            for (auto child_id : _label_nav_graph->out_neighbors[current_group])
            {
               if (!visited[child_id])
               {
                  visited[child_id] = true;
                  q.push(child_id);
               }
            }
         }

         sorted_trees.push_back({conceptual_root_group_id, root_label, current_tree_coverage});
         per_tree_groups[conceptual_root_group_id] = groups_in_this_tree;
         tree_max_depth[conceptual_root_group_id] = current_max_depth;
         std::cout << "    > 此概念树共包含 " << groups_in_this_tree.size() << " 个Group, 总覆盖点数: " << current_tree_coverage << std::endl;
      }

      if (sorted_trees.empty())
      {
         std::cerr << "FATAL ERROR: 您提供的所有概念根标签均无效或无法在索引中找到，无法生成查询。" << std::endl;
         return;
      }

      // --- 阶段4: 查询生成 ---
      std::cout << "\n步骤4: 开始为K近邻场景生成查询..." << std::endl;
      std::sort(sorted_trees.begin(), sorted_trees.end()); // 按覆盖率降序排序

      if (top_M_trees > 0 && top_M_trees < sorted_trees.size())
      {
         sorted_trees.resize(top_M_trees);
      }
      std::cout << "  - 将从覆盖率最高的 " << sorted_trees.size() << " 棵概念树中生成查询。" << std::endl;

      // 按比例分配N个查询
      std::vector<int> query_allocations;
      size_t total_top_coverage = 0;
      for (const auto &tree_info : sorted_trees)
      {
         total_top_coverage += tree_info.coverage_count;
      }
      if (total_top_coverage > 0)
      {
         int allocated_queries = 0;
         for (size_t i = 0; i < sorted_trees.size(); ++i)
         {
            int num = (i == sorted_trees.size() - 1) ? (N - allocated_queries) : static_cast<int>(N * (static_cast<double>(sorted_trees[i].coverage_count) / total_top_coverage));
            query_allocations.push_back(num);
            allocated_queries += num;
         }
      }

      // 准备文件和工具
      std::ofstream query_vec_file(output_prefix + "/" + dataset + "_query.fvecs", std::ios::binary);
      std::ofstream query_label_file(output_prefix + "/" + dataset + "_query_labels.txt");
      std::ofstream ground_truth_file(output_prefix + "/" + dataset + "_groundtruth.txt");
      uint32_t dim = _base_storage->get_dim();
      std::unordered_map<IdxType, IdxType> old_to_new_map;
      for (IdxType new_id = 0; new_id < _num_points; ++new_id)
      {
         old_to_new_map[_new_to_old_vec_ids[new_id]] = new_id;
      }
      std::mt19937 rng(std::random_device{}());
      int queries_generated = 0;

      // 开始生成
      for (size_t i = 0; i < sorted_trees.size(); ++i)
      {
         IdxType root_group_id = sorted_trees[i].root_group_id;
         LabelType root_label_id = sorted_trees[i].root_label_id;
         int queries_to_gen_for_this_tree = query_allocations[i];

         // 构建深度样本池
         std::vector<IdxType> local_deep_pool;
         int deep_threshold = static_cast<int>(tree_max_depth.at(root_group_id) * 0.67);
         for (IdxType group_id : per_tree_groups.at(root_group_id))
         {
            if (_group_id_to_label_set[group_id].size() > deep_threshold)
            {
               local_deep_pool.insert(local_deep_pool.end(), _group_id_to_vec_ids[group_id].begin(), _group_id_to_vec_ids[group_id].end());
            }
         }

         if (local_deep_pool.size() < K)
         {
            std::cout << "  - 警告: 树 (根标签 " << root_label_id << ") 的深度样本数 (" << local_deep_pool.size() << ") 小于 K (" << K << ")，跳过此树。" << std::endl;
            continue;
         }

         std::cout << "  - 正在为树 (根标签 " << root_label_id << ") 生成 " << queries_to_gen_for_this_tree << " 个查询..." << std::endl;

         for (int j = 0; j < queries_to_gen_for_this_tree && queries_generated < N; ++j)
         {
            // [这部分寻找GT和质心的代码与原版完全相同]
            std::uniform_int_distribution<size_t> dist(0, local_deep_pool.size() - 1);
            IdxType seed_original_id = local_deep_pool[dist(rng)];
            IdxType seed_new_id = old_to_new_map.at(seed_original_id);

            std::priority_queue<std::pair<float, IdxType>> top_k_queue;
            for (IdxType candidate_original_id : local_deep_pool)
            {
               if (candidate_original_id == seed_original_id)
                  continue;
               IdxType candidate_new_id = old_to_new_map.at(candidate_original_id);
               float distance = _distance_handler->compute(_base_storage->get_vector(seed_new_id), _base_storage->get_vector(candidate_new_id), dim);
               if (top_k_queue.size() < (size_t)K - 1)
               {
                  top_k_queue.push({distance, candidate_original_id});
               }
               else if (distance < top_k_queue.top().first)
               {
                  top_k_queue.pop();
                  top_k_queue.push({distance, candidate_original_id});
               }
            }

            std::vector<IdxType> ground_truth_ids;
            ground_truth_ids.push_back(seed_original_id);
            while (!top_k_queue.empty())
            {
               ground_truth_ids.push_back(top_k_queue.top().second);
               top_k_queue.pop();
            }

            std::vector<float> centroid_vec(dim, 0.0f);
            for (IdxType gt_id : ground_truth_ids)
            {
               const float *vec = reinterpret_cast<const float *>(_base_storage->get_vector(old_to_new_map.at(gt_id)));
               for (uint32_t d = 0; d < dim; ++d)
                  centroid_vec[d] += vec[d];
            }
            for (uint32_t d = 0; d < dim; ++d)
               centroid_vec[d] /= K;

            query_vec_file.write(reinterpret_cast<const char *>(&dim), sizeof(uint32_t));
            query_vec_file.write(reinterpret_cast<const char *>(centroid_vec.data()), dim * sizeof(float));
            query_label_file << root_label_id << "\n";
            for (size_t gt_idx = 0; gt_idx < ground_truth_ids.size(); ++gt_idx)
            {
               ground_truth_file << ground_truth_ids[gt_idx] << (gt_idx == ground_truth_ids.size() - 1 ? "" : "\t");
            }
            ground_truth_file << "\n";
            queries_generated++;
         }
      }

      query_vec_file.close();
      query_label_file.close();
      ground_truth_file.close();

      std::cout << "\n--- 查询生成结束：共生成 " << queries_generated << " 个查询任务 ---" << std::endl;
      std::cout << "查询向量已保存到: " << output_prefix + "/" + dataset + "_query.fvecs" << std::endl;
      std::cout << "查询标签已保存到: " << output_prefix + "/" + dataset + "_query_labels.txt" << std::endl;
      std::cout << "Ground Truth已保存到: " << output_prefix + "/" + dataset + "_groundtruth.txt" << std::endl;
   }

   // fxy_add: 极端方法2的查询任务生成低覆盖率的查询任务（基于 count ∈ [K, max_K]）
   void UniNavGraph::generate_queries_method2_low_coverage(
       std::string &output_prefix,
       std::string dataset,
       int query_n,
       std::string &base_label_file,
       int num_of_per_query_labels, // 每个查询的标签数量
       int K,                       // 标签组合出现次数的下界
       int max_K,                   // 标签组合出现次数的上界
       int min_K)                   // 有效标签组合的最小数量,即符合罕见度要求的“查询模板”个数
   {
      // ==================== 阶段1：统计标签频率 ====================
      std::unordered_map<int, int> label_counts;

      // 第一次扫描：仅统计标签频率
      {
         std::ifstream label_file(base_label_file);
         if (!label_file.is_open())
         {
            std::cerr << "Failed to open label file: " << base_label_file << std::endl;
            return;
         }

         std::string line;
         while (std::getline(label_file, line))
         {
            std::stringstream ss(line);
            std::string label_str;
            while (std::getline(ss, label_str, ','))
            {
               int label = std::stoi(label_str);
               label_counts[label]++;
            }
         }
         label_file.close();
      }

      if (label_counts.empty())
      {
         std::cerr << "No labels found, using default label 1" << std::endl;
         label_counts[1] = 1;
      }

      // ==================== 阶段2：准备低频标签数据 ====================
      // 计算频率阈值（只保留出现次数<=max_K的标签）
      const int freq_threshold = max_K * 2; // 经验值，可根据数据调整

      // 第二次扫描：仅缓存包含低频标签的行
      std::vector<std::vector<int>> low_freq_label_sets;
      {
         std::ifstream label_file(base_label_file);
         std::string line;

         while (std::getline(label_file, line))
         {
            std::vector<int> labels;
            std::stringstream ss(line);
            std::string label_str;
            bool has_low_freq_label = false;

            while (std::getline(ss, label_str, ','))
            {
               int label = std::stoi(label_str);
               if (label_counts[label] <= freq_threshold)
               {
                  has_low_freq_label = true;
               }
               labels.push_back(label);
            }

            if (has_low_freq_label)
            {
               std::sort(labels.begin(), labels.end());
               low_freq_label_sets.push_back(labels);
            }
         }
         label_file.close();
      }

      // ==================== 阶段3：生成有效标签组合 ====================
      // 按频率排序标签（低频优先）
      std::vector<std::pair<int, int>> label_freq_pairs(label_counts.begin(), label_counts.end());
      std::sort(label_freq_pairs.begin(), label_freq_pairs.end(),
                [](const auto &a, const auto &b)
                { return a.second < b.second; });

      std::vector<int> sorted_labels;
      for (const auto &p : label_freq_pairs)
      {
         sorted_labels.push_back(p.first);
      }

      struct LabelSetInfo
      {
         std::vector<int> labels;
         int count;
      };
      std::vector<LabelSetInfo> valid_combinations;
      std::unordered_set<std::string> seen_combinations;

      // 生成候选组合（仅使用低频标签）
      for (size_t start_idx = 0; start_idx < sorted_labels.size() &&
                                 valid_combinations.size() < min_K;
           ++start_idx)
      {
         if (label_counts[sorted_labels[start_idx]] > freq_threshold)
            continue;

         for (int k = 1; k <= num_of_per_query_labels &&
                         k <= sorted_labels.size() - start_idx;
              ++k)
         {
            std::vector<int> candidate;
            for (int i = 0; i < k; ++i)
            {
               candidate.push_back(sorted_labels[start_idx + i]);
               if (label_counts[sorted_labels[start_idx + i]] > freq_threshold)
               {
                  candidate.clear();
                  break;
               }
            }
            if (candidate.empty())
               continue;

            // 去重检查
            std::sort(candidate.begin(), candidate.end());
            std::string key;
            for (int l : candidate)
               key += std::to_string(l) + ",";
            if (seen_combinations.count(key))
               continue;
            seen_combinations.insert(key);

            // 统计组合出现次数（使用缓存的低频数据）
            int occurrence = 0;
            for (const auto &labels : low_freq_label_sets)
            {
               bool match = true;
               for (int l : candidate)
               {
                  if (!std::binary_search(labels.begin(), labels.end(), l))
                  {
                     match = false;
                     break;
                  }
               }
               if (match)
                  occurrence++;
            }

            if (occurrence >= K && occurrence <= max_K)
            {
               valid_combinations.push_back({candidate, occurrence});
               if (valid_combinations.size() >= min_K)
                  break;
            }
         }
      }

      // ==================== 阶段4：生成查询文件 ====================
      if (valid_combinations.empty())
      {
         std::cerr << "Error: No valid combinations in range [" << K << ", " << max_K << "]" << std::endl;
         return;
      }

      std::ofstream txt_file(output_prefix + "/" + dataset + "_query_labels.txt");
      std::ofstream coverage_file(output_prefix + "/" + dataset + "_query_and_coverage_labels.txt");
      std::ofstream fvec_file(output_prefix + "/" + dataset + "_query.fvecs", std::ios::binary);

      if (!txt_file.is_open() || !fvec_file.is_open())
      {
         std::cerr << "Failed to open output files" << std::endl;
         return;
      }

      uint32_t dim = _base_storage->get_dim();
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<int> combo_dis(0, valid_combinations.size() - 1);
      std::uniform_int_distribution<ANNS::IdxType> vec_dis(0, _base_storage->get_num_points() - 1);
      std::unordered_set<ANNS::IdxType> used_vec_ids;

      for (int i = 0; i < query_n; ++i)
      {
         const auto &combo = valid_combinations[combo_dis(gen)];

         // 获取唯一向量ID
         ANNS::IdxType vec_id;
         do
         {
            vec_id = vec_dis(gen);
         } while (used_vec_ids.count(vec_id) &&
                  used_vec_ids.size() < _base_storage->get_num_points());

         if (used_vec_ids.size() >= _base_storage->get_num_points())
         {
            std::cerr << "Warning: Not enough unique vectors" << std::endl;
            break;
         }
         used_vec_ids.insert(vec_id);

         // 写入标签
         for (size_t j = 0; j < combo.labels.size(); ++j)
         {
            txt_file << combo.labels[j];
            coverage_file << combo.labels[j];
            if (j != combo.labels.size() - 1)
            {
               txt_file << ",";
               coverage_file << ",";
            }
         }
         coverage_file << " count:" << combo.count;
         txt_file << "\n";
         coverage_file << "\n";

         // 写入向量
         const char *vec_data = _base_storage->get_vector(vec_id);
         fvec_file.write((char *)&dim, sizeof(uint32_t));
         fvec_file.write(vec_data, dim * sizeof(float));
      }

      // 关闭文件
      txt_file.close();
      coverage_file.close();
      fvec_file.close();

      std::cout << "Generated " << std::min(query_n, (int)valid_combinations.size())
                << " queries with count in [" << K << ", " << max_K << "]\n";
      std::cout << "Unique combinations found: " << valid_combinations.size() << "\n";
   }

   // fxy_add: 真实数据不处理，自动发现高覆盖率概念根并生成查询
   void UniNavGraph::generate_queries_true_data_high_coverage(
       int N,                             // 要生成的查询总数
       int K,                             // 用于计算质心的邻居数量
       int top_M_trees,                   // 选择覆盖率最高的M棵概念树
       std::string dataset,               // 数据集名称，用于生成文件名
       const std::string &output_prefix,  // 输出文件路径前缀
       float min_root_coverage_threshold) // 单个属性被视为“概念根”的最低覆盖率阈值
   {
      std::cout << "\n==========================================================" << std::endl;
      std::cout << "--- 开始执行真实数据的高覆盖率查询生成函数 (自动发现根节点) ---" << std::endl;
      std::cout << "==========================================================" << std::endl;

      // 基本检查
      if (_num_groups == 0 || _num_points == 0)
      {
         std::cerr << "[错误] 索引未构建或不包含任何数据点，无法生成查询。" << std::endl;
         return;
      }
      uint32_t dim = _base_storage->get_dim();

      // ==================== 步骤 1: 自动发现高覆盖率的单个属性作为概念根 ====================
      std::cout << "\n[步骤 1] 自动发现概念根..." << std::endl;
      std::cout << "  - 策略：统计数据集中每个单独属性的出现频率，筛选出覆盖最广的属性。" << std::endl;

      std::unordered_map<LabelType, size_t> attribute_counts;
      // 遍历所有数据点，统计每个标签的出现次数
      for (IdxType i = 0; i < _num_points; ++i)
      {
         const auto &label_set = _base_storage->get_label_set(i);
         for (const auto &label : label_set)
         {
            attribute_counts[label]++;
         }
      }
      std::cout << "  - [信息] 发现数据集中共有 " << attribute_counts.size() << " 个独立属性。" << std::endl;

      // 将统计结果转换为可排序的 vector
      std::vector<std::pair<LabelType, size_t>> sorted_attributes;
      for (const auto &pair : attribute_counts)
      {
         sorted_attributes.push_back(pair);
      }
      // 按出现次数（覆盖率）降序排序
      std::sort(sorted_attributes.begin(), sorted_attributes.end(), [](const auto &a, const auto &b)
                { return a.second > b.second; });

      // 筛选出符合最低覆盖率阈值的属性作为概念根
      std::vector<LabelType> conceptual_root_labels;
      std::cout << "  - [调试] 筛选覆盖率 > " << min_root_coverage_threshold * 100 << "% 的属性作为根..." << std::endl;
      for (const auto &pair : sorted_attributes)
      {
         float coverage = static_cast<float>(pair.second) / _num_points;
         if (coverage >= min_root_coverage_threshold)
         {
            conceptual_root_labels.push_back(pair.first);
            std::cout << "    - [候选根] 属性: " << pair.first
                      << ", 出现次数: " << pair.second
                      << ", 覆盖率: " << coverage * 100 << "%" << std::endl;
         }
      }

      if (conceptual_root_labels.empty())
      {
         std::cerr << "[致命错误] 未找到任何属性满足覆盖率阈值 > " << min_root_coverage_threshold * 100
                   << "%。无法生成查询。请尝试降低阈值。" << std::endl;
         return;
      }
      std::cout << "  - [成功] 最终确定 " << conceptual_root_labels.size() << " 个概念根用于构建查询树。" << std::endl;

      // ==================== 步骤 2: 基于发现的概念根构建查询树信息 ====================
      std::cout << "\n[步骤 2] 基于概念根构建查询树..." << std::endl;
      std::vector<TreeInfo> sorted_trees;
      std::map<LabelType, std::vector<IdxType>> per_tree_groups; // 存储每棵概念树包含的所有group ID
      std::map<LabelType, int> tree_max_depth;                   // 存储每棵概念树的最大深度

      for (LabelType root_label : conceptual_root_labels)
      {
         std::cout << "  - 正在处理概念根: " << root_label << std::endl;

         std::vector<bool> visited_groups(_num_groups + 1, false);
         std::queue<IdxType> q;
         std::vector<IdxType> groups_in_this_tree;
         size_t current_tree_coverage = 0;
         int current_max_depth = 0;

         // 找到所有包含该根标签的 group 作为遍历的起点
         for (IdxType group_id = 1; group_id <= _num_groups; ++group_id)
         {
            const auto &labels = _group_id_to_label_set[group_id];
            if (std::find(labels.begin(), labels.end(), root_label) != labels.end())
            {
               if (!visited_groups[group_id])
               {
                  q.push(group_id);
                  visited_groups[group_id] = true;
               }
            }
         }

         std::cout << "    - [调试] 找到 " << q.size() << " 个包含此根标签的初始Group。" << std::endl;

         // 从所有起点开始BFS，找到完整的概念树
         while (!q.empty())
         {
            IdxType current_group = q.front();
            q.pop();

            groups_in_this_tree.push_back(current_group);
            current_tree_coverage += _group_id_to_vec_ids[current_group].size();
            current_max_depth = std::max(current_max_depth, (int)_group_id_to_label_set[current_group].size());

            for (auto child_id : _label_nav_graph->out_neighbors[current_group])
            {
               if (!visited_groups[child_id])
               {
                  visited_groups[child_id] = true;
                  q.push(child_id);
               }
            }
         }

         if (current_tree_coverage > 0)
         {
            sorted_trees.push_back({0, root_label, current_tree_coverage}); // group_id 占位，以 root_label_id 为主键
            per_tree_groups[root_label] = groups_in_this_tree;
            tree_max_depth[root_label] = current_max_depth;
            std::cout << "    - [结果] 此概念树共包含 " << groups_in_this_tree.size()
                      << " 个Group, 总覆盖点数: " << current_tree_coverage
                      << ", 最大标签集深度: " << current_max_depth << std::endl;
         }
         else
         {
            std::cout << "    - [警告] 概念根 " << root_label << "未能构建有效的树 (覆盖点数为0)，已跳过。" << std::endl;
         }
      }

      if (sorted_trees.empty())
      {
         std::cerr << "[致命错误] 所有候选概念根都未能构建有效的查询树，无法生成查询。" << std::endl;
         return;
      }

      // ==================== 步骤 3: 筛选树并分配查询数量 ====================
      std::cout << "\n[步骤 3] 筛选顶层树并分配查询..." << std::endl;
      // 按覆盖率对树进行最终排序
      std::sort(sorted_trees.begin(), sorted_trees.end());

      if (top_M_trees > 0 && top_M_trees < sorted_trees.size())
      {
         sorted_trees.resize(top_M_trees);
      }
      std::cout << "  - [信息] 将从覆盖率最高的 " << sorted_trees.size() << " 棵概念树中生成 " << N << " 个查询。" << std::endl;

      std::vector<int> query_allocations;
      size_t total_top_coverage = 0;
      for (const auto &tree_info : sorted_trees)
      {
         total_top_coverage += tree_info.coverage_count;
      }

      if (total_top_coverage > 0)
      {
         int allocated_queries = 0;
         for (size_t i = 0; i < sorted_trees.size(); ++i)
         {
            // 按比例分配，最后一个树获得剩余所有名额，确保总数为N
            int num = (i == sorted_trees.size() - 1) ? (N - allocated_queries) : static_cast<int>(round(N * (static_cast<double>(sorted_trees[i].coverage_count) / total_top_coverage)));
            query_allocations.push_back(num);
            allocated_queries += num;
            std::cout << "    - 树 (根 " << sorted_trees[i].root_label_id << ", 覆盖率 " << (double)sorted_trees[i].coverage_count / total_top_coverage * 100 << "%) 分配到 " << num << " 个查询名额。" << std::endl;
         }
      }
      else
      {
         std::cerr << "[错误] 顶层树的总覆盖率为0，无法分配查询。" << std::endl;
         return;
      }

      // ==================== 步骤 4: 生成查询并写入文件 ====================
      std::cout << "\n[步骤 4] 生成查询向量和真值集..." << std::endl;
      // 准备文件和工具
      std::string query_vec_filename = output_prefix + "/" + dataset + "_query.fvecs";
      std::string query_label_filename = output_prefix + "/" + dataset + "_query_labels.txt";
      std::string ground_truth_filename = output_prefix + "/" + dataset + "_groundtruth.txt";

      std::ofstream query_vec_file(query_vec_filename, std::ios::binary);
      std::ofstream query_label_file(query_label_filename);
      std::ofstream ground_truth_file(ground_truth_filename);

      std::unordered_map<IdxType, IdxType> old_to_new_map;
      for (IdxType new_id = 0; new_id < _num_points; ++new_id)
      {
         old_to_new_map[_new_to_old_vec_ids[new_id]] = new_id;
      }
      std::mt19937 rng(std::random_device{}());
      int queries_generated_total = 0;

      // 开始生成
      for (size_t i = 0; i < sorted_trees.size(); ++i)
      {
         LabelType root_label_id = sorted_trees[i].root_label_id;
         int queries_to_gen_for_this_tree = query_allocations[i];
         std::cout << "  - 正在为树 (根 " << root_label_id << ") 生成 " << queries_to_gen_for_this_tree << " 个查询..." << std::endl;

         // ======================== 新的、更健壮的样本池构建逻辑 ========================
         std::vector<IdxType> local_deep_pool;
         int max_d = tree_max_depth.at(root_label_id);
         int deep_threshold = std::max(1, static_cast<int>(max_d * 0.67));

         std::cout << "    - [调试] 此树最大深度为 " << max_d << "，初始深度阈值设为 > " << deep_threshold << std::endl;

         // 为了高效地按层添加，先将 group 按深度分类
         std::map<int, std::vector<IdxType>> groups_by_depth;
         for (IdxType group_id : per_tree_groups.at(root_label_id))
         {
            groups_by_depth[_group_id_to_label_set[group_id].size()].push_back(group_id);
         }

         // 1. 优先填充最深层的节点
         for (int d = max_d; d > deep_threshold; --d)
         {
            if (groups_by_depth.count(d))
            {
               for (IdxType group_id : groups_by_depth.at(d))
               {
                  const auto &vecs_in_group = _group_id_to_vec_ids[group_id];
                  local_deep_pool.insert(local_deep_pool.end(), vecs_in_group.begin(), vecs_in_group.end());
               }
            }
         }
         std::cout << "    - [信息] 初始深度池 (深度 > " << deep_threshold << ") 大小为: " << local_deep_pool.size() << std::endl;

         // 2. 如果池大小不足 K，则逐层向下（深度值减小）放宽标准
         if (local_deep_pool.size() < K)
         {
            std::cout << "    - [警告] 初始池大小小于 K (" << K << ")。将逐层添加候选向量..." << std::endl;
            for (int d = deep_threshold; d >= 1; --d)
            {
               if (groups_by_depth.count(d))
               {
                  std::cout << "      - 尝试添加深度为 " << d << " 的节点..." << std::endl;
                  for (IdxType group_id : groups_by_depth.at(d))
                  {
                     const auto &vecs_in_group = _group_id_to_vec_ids[group_id];
                     local_deep_pool.insert(local_deep_pool.end(), vecs_in_group.begin(), vecs_in_group.end());
                  }
                  std::cout << "      - 新的池大小: " << local_deep_pool.size() << std::endl;
               }
               // 每添加一层后都检查是否已满足要求
               if (local_deep_pool.size() >= K)
               {
                  std::cout << "    - [成功] 样本池大小已满足要求，停止添加。" << std::endl;
                  break; // 样本数已够，跳出循环
               }
            }
         }

         // 3. 最终检查：如果遍历完所有层仍然不够，则只能放弃此树
         if (local_deep_pool.size() < K)
         {
            std::cout << "    - [错误] 即使包含了树中所有节点，样本池大小 (" << local_deep_pool.size()
                      << ") 仍然小于 K (" << K << ")，无法为此树生成查询，已跳过。" << std::endl;
            continue;
         }
         std::cout << "    - [信息] 最终用于生成查询的样本池大小为: " << local_deep_pool.size() << std::endl;

         for (int j = 0; j < queries_to_gen_for_this_tree && queries_generated_total < N; ++j)
         {
            // 从深度池中随机选一个种子点
            std::uniform_int_distribution<size_t> dist(0, local_deep_pool.size() - 1);
            IdxType seed_original_id = local_deep_pool[dist(rng)];
            IdxType seed_new_id = old_to_new_map.at(seed_original_id);

            // 在深度池中暴力搜索种子的K-1个最近邻
            std::priority_queue<std::pair<float, IdxType>> top_k_queue;
            for (IdxType candidate_original_id : local_deep_pool)
            {
               if (candidate_original_id == seed_original_id)
                  continue;

               IdxType candidate_new_id = old_to_new_map.at(candidate_original_id);
               float distance = _distance_handler->compute(_base_storage->get_vector(seed_new_id), _base_storage->get_vector(candidate_new_id), dim);

               if (top_k_queue.size() < (size_t)K - 1)
               {
                  top_k_queue.push({distance, candidate_original_id});
               }
               else if (distance < top_k_queue.top().first)
               {
                  top_k_queue.pop();
                  top_k_queue.push({distance, candidate_original_id});
               }
            }

            // 提取GT并计算质心
            std::vector<IdxType> ground_truth_ids;
            ground_truth_ids.push_back(seed_original_id);
            while (!top_k_queue.empty())
            {
               ground_truth_ids.push_back(top_k_queue.top().second);
               top_k_queue.pop();
            }

            std::vector<float> centroid_vec(dim, 0.0f);
            for (IdxType gt_id : ground_truth_ids)
            {
               const float *vec = reinterpret_cast<const float *>(_base_storage->get_vector(old_to_new_map.at(gt_id)));
               for (uint32_t d = 0; d < dim; ++d)
                  centroid_vec[d] += vec[d];
            }
            for (uint32_t d = 0; d < dim; ++d)
               centroid_vec[d] /= K;

            // 写入文件
            query_vec_file.write(reinterpret_cast<const char *>(&dim), sizeof(uint32_t));
            query_vec_file.write(reinterpret_cast<const char *>(centroid_vec.data()), dim * sizeof(float));

            query_label_file << root_label_id << "\n";

            for (size_t gt_idx = 0; gt_idx < ground_truth_ids.size(); ++gt_idx)
            {
               ground_truth_file << ground_truth_ids[gt_idx] << (gt_idx == ground_truth_ids.size() - 1 ? "" : "\t");
            }
            ground_truth_file << "\n";
            queries_generated_total++;
         }
      }

      query_vec_file.close();
      query_label_file.close();
      ground_truth_file.close();

      std::cout << "\n--- 查询生成结束 ---" << std::endl;
      std::cout << "  - 成功生成 " << queries_generated_total << " 个查询任务。" << std::endl;
      std::cout << "  - 查询向量已保存到: " << query_vec_filename << std::endl;
      std::cout << "  - 查询标签已保存到: " << query_label_filename << std::endl;
      std::cout << "  - Ground Truth已保存到: " << ground_truth_filename << std::endl;
      std::cout << "==========================================================" << std::endl;
   }

   // fxy_add:真实数据不处理，生成覆盖率低的查询任务:选出覆盖率在 (0, coverage_threshold] 区间且出现次数 ≥ K 的组合
   void UniNavGraph::generate_queries_true_data_low_coverage(
       std::string &output_prefix,
       std::string dataset,
       int query_n,
       std::string &base_label_file,
       int num_of_per_query_labels,
       float coverage_threshold,
       int K)
   {
      // Step 1: 从标签文件加载并分析标签分布
      std::ifstream label_file(base_label_file);
      if (!label_file.is_open())
      {
         std::cerr << "Failed to open label file: " << base_label_file << std::endl;
         return;
      }

      // 统计标签出现频率
      std::unordered_map<int, int> label_counts;
      std::vector<std::vector<int>> all_label_sets;
      std::string line;

      while (std::getline(label_file, line))
      {
         std::vector<int> labels;
         std::stringstream ss(line);
         std::string label_str;

         while (std::getline(ss, label_str, ','))
         {
            int label = std::stoi(label_str);
            labels.push_back(label);
            label_counts[label]++;
         }

         std::sort(labels.begin(), labels.end());
         all_label_sets.push_back(labels);
      }
      label_file.close();

      // 如果没有标签数据，使用默认标签
      if (all_label_sets.empty())
      {
         std::cerr << "No label data found, using default label 1" << std::endl;
         all_label_sets.push_back({1});
         label_counts[1] = 1;
      }

      // Step 2: 识别低覆盖率标签组合（排除覆盖率为0的组合）
      struct LabelSetInfo
      {
         std::vector<int> labels;
         int count;
         float coverage;
      };

      std::vector<LabelSetInfo> valid_low_coverage_sets;
      int total_vectors = all_label_sets.size();

      // 统计标签组合的覆盖率（最多num_of_per_query_labels个标签的组合）
      std::map<std::vector<int>, int> combination_counts;
      for (const auto &labels : all_label_sets)
      {
         // 生成所有可能的1-num_of_per_query_labels标签组合
         for (int k = 1; k <= num_of_per_query_labels && k <= labels.size(); ++k)
         {
            std::vector<bool> mask(labels.size(), false);
            std::fill(mask.begin(), mask.begin() + k, true);

            do
            {
               std::vector<int> combination;
               for (size_t i = 0; i < mask.size(); ++i)
               {
                  if (mask[i])
                     combination.push_back(labels[i]);
               }
               combination_counts[combination]++;
            } while (std::prev_permutation(mask.begin(), mask.end()));
         }
      }

      // 筛选出覆盖率大于0且低于阈值的组合
      for (const auto &entry : combination_counts)
      {
         // 跳过覆盖率为0的组合
         if (entry.second == 0)
            continue;

         float coverage = static_cast<float>(entry.second) / total_vectors;
         if (coverage <= coverage_threshold && entry.second >= K)
         {
            valid_low_coverage_sets.push_back({entry.first, entry.second, coverage});
         }
      }

      // 按覆盖率升序排序
      std::sort(valid_low_coverage_sets.begin(), valid_low_coverage_sets.end(),
                [](const auto &a, const auto &b)
                {
                   return a.coverage < b.coverage;
                });

      // 检查是否有足够的有效组合
      if (valid_low_coverage_sets.empty())
      {
         std::cerr << "Error: No valid low-coverage label combinations found (all either have coverage=0 or > "
                   << coverage_threshold << ")" << std::endl;
         return;
      }

      // Step 3: 生成查询文件和标签
      std::ofstream txt_file(output_prefix + "/" + dataset + "_query_labels.txt");
      std::ofstream txt__coverage_file(output_prefix + "/" + dataset + "_query_and_coverage_labels.txt");
      std::ofstream fvec_file(output_prefix + "/" + dataset + "_query.fvecs", std::ios::binary);

      if (!txt_file.is_open() || !fvec_file.is_open())
      {
         std::cerr << "Failed to open output files." << std::endl;
         return;
      }

      uint32_t dim = _base_storage->get_dim();
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<ANNS::IdxType> vec_dis(0, _base_storage->get_num_points() - 1);
      std::unordered_set<ANNS::IdxType> used_vec_ids;

      // 选择前query_n个最稀有的有效标签组合
      int queries_generated = 0;
      for (size_t i = 0; i < valid_low_coverage_sets.size() && queries_generated < query_n; ++i)
      {
         const auto &label_set = valid_low_coverage_sets[i].labels;

         // 写入标签文件
         for (size_t j = 0; j < label_set.size(); ++j)
         {
            txt_file << label_set[j];
            if (j != label_set.size() - 1)
               txt_file << ",";

            txt__coverage_file << label_set[j];
            if (j != label_set.size() - 1)
               txt__coverage_file << ",";
         }
         txt__coverage_file << " coverage:" << valid_low_coverage_sets[i].coverage;
         txt_file << std::endl;
         txt__coverage_file << std::endl;

         // 随机选择一个向量（确保不重复）
         ANNS::IdxType vec_id;
         do
         {
            vec_id = vec_dis(gen);
         } while (used_vec_ids.count(vec_id) > 0 &&
                  used_vec_ids.size() < _base_storage->get_num_points());

         used_vec_ids.insert(vec_id);
         const char *vec_data = _base_storage->get_vector(vec_id);

         // 写入向量文件
         fvec_file.write((char *)&dim, sizeof(uint32_t));
         fvec_file.write(vec_data, dim * sizeof(float));

         queries_generated++;
      }

      txt_file.close();
      txt__coverage_file.close();
      fvec_file.close();

      std::cout << "Generated " << queries_generated << " low-coverage queries with 0 < coverage <= "
                << coverage_threshold << std::endl;
      std::cout << "Labels written to: " << output_prefix + "/" + dataset + "_lowcov_query_labels.txt" << std::endl;
      std::cout << "Vectors written to: " << output_prefix + "/" + dataset + "_lowcov_query.fvecs" << std::endl;

      // 输出统计信息
      if (!valid_low_coverage_sets.empty())
      {
         std::cout << "\nStatistics of generated low-coverage queries:" << std::endl;
         std::cout << "Min coverage: " << valid_low_coverage_sets.front().coverage << std::endl;
         std::cout << "Max coverage: " << valid_low_coverage_sets.back().coverage << std::endl;
         std::cout << "Median coverage: "
                   << valid_low_coverage_sets[valid_low_coverage_sets.size() / 2].coverage << std::endl;
      }
   }

   // fxy_add: 人造数据：极端方法2的查询任务生成：根据LNG树的深度信息生成查询任务
   void UniNavGraph::generate_queries_method2_high_coverage_human(
       std::string &output_prefix,
       std::string dataset,
       int query_n,
       std::string &base_label_file,
       std::string &base_label_info_file)
   {
      // Step 1: 读取LNG树信息文件获取总层数
      std::ifstream info_file(base_label_info_file);
      if (!info_file.is_open())
      {
         std::cerr << "Failed to open LNG info file: " << base_label_info_file << std::endl;
         return;
      }

      int total_layers = 0;
      std::string line;
      while (std::getline(info_file, line))
      {
         if (line.find("总层数:") != std::string::npos)
         {
            size_t pos = line.find(":");
            if (pos != std::string::npos)
            {
               total_layers = std::stoi(line.substr(pos + 1));
               break;
            }
         }
      }
      info_file.close();

      if (total_layers == 0)
      {
         std::cerr << "Error: Could not determine total layers from LNG info file" << std::endl;
         return;
      }

      // 计算顶层深度
      // int max_depth_top = std::max(1, total_layers / 10);
      int max_depth_top = 3;
      std::cout << "Total layers: " << total_layers << ", Top layers to use: " << max_depth_top << std::endl;

      // Step 2: 读取base_labels文件获取顶层标签
      std::ifstream label_file(base_label_file);
      if (!label_file.is_open())
      {
         std::cerr << "Failed to open base label file: " << base_label_file << std::endl;
         return;
      }

      std::vector<std::vector<int>> top_layer_labels;
      int lines_read = 0;
      while (std::getline(label_file, line) && lines_read < max_depth_top)
      {
         std::vector<int> labels;
         std::stringstream ss(line);
         std::string label_str;

         while (std::getline(ss, label_str, ','))
         {
            labels.push_back(std::stoi(label_str));
         }

         if (!labels.empty())
         {
            top_layer_labels.push_back(labels);
            lines_read++;
         }
      }
      label_file.close();

      if (top_layer_labels.empty())
      {
         std::cerr << "Error: No valid labels found in base label file" << std::endl;
         return;
      }

      // Step 3: 准备输出文件
      std::ofstream txt_file(output_prefix + "/" + dataset + "_query_labels.txt");
      std::ofstream fvec_file(output_prefix + "/" + dataset + "_query.fvecs", std::ios::binary);

      if (!txt_file.is_open() || !fvec_file.is_open())
      {
         std::cerr << "Failed to open output files" << std::endl;
         return;
      }

      uint32_t dim = _base_storage->get_dim();
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<ANNS::IdxType> vec_dis(0, _base_storage->get_num_points() - 1);
      std::unordered_set<ANNS::IdxType> used_vec_ids;

      // Step 4: 生成查询任务
      int queries_generated = 0;
      while (queries_generated < query_n)
      {
         // 循环使用顶层标签
         const auto &labels = top_layer_labels[queries_generated % top_layer_labels.size()];

         // 写入标签文件
         for (size_t j = 0; j < labels.size(); ++j)
         {
            txt_file << labels[j];
            if (j != labels.size() - 1)
            {
               txt_file << ",";
            }
         }
         txt_file << std::endl;

         // 随机选择一个不重复的向量
         ANNS::IdxType vec_id;
         do
         {
            vec_id = vec_dis(gen);
         } while (used_vec_ids.count(vec_id) > 0 &&
                  used_vec_ids.size() < _base_storage->get_num_points());

         used_vec_ids.insert(vec_id);
         const char *vec_data = _base_storage->get_vector(vec_id);

         // 写入向量文件
         fvec_file.write((char *)&dim, sizeof(uint32_t));
         fvec_file.write(vec_data, dim * sizeof(float));

         queries_generated++;
      }

      txt_file.close();
      fvec_file.close();

      std::cout << "Generated " << queries_generated << " queries based on LNG depth" << std::endl;
      std::cout << "Labels written to: " << output_prefix + "/" + dataset + "_query_labels.txt" << std::endl;
      std::cout << "Vectors written to: " << output_prefix + "/" + dataset + "_query.fvecs" << std::endl;
   }

   // ===================================end：生成query task========================================

   // ===================================begin：限制覆盖率的查询任务生成========================================
   // fxy_add: LNG节点深度预计算
   void UniNavGraph::_precompute_lng_node_depths()
   {
      std::cout << " - 正在预计算LNG节点的图深度..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      if (_num_groups == 0)
         return;

      _lng_node_depths.assign(_num_groups + 1, 0);
      std::vector<int> in_degree(_num_groups + 1, 0);

      // 1. 计算所有节点的入度
      for (IdxType u = 1; u <= _num_groups; ++u)
      {
         for (IdxType v : _label_nav_graph->out_neighbors[u])
         {
            in_degree[v]++;
         }
      }

      // 2. 找到所有根节点 (入度为0) 并加入队列
      std::queue<IdxType> q;
      for (IdxType i = 1; i <= _num_groups; ++i)
      {
         if (in_degree[i] == 0)
         {
            q.push(i);
         }
      }

      // 3. 类似拓扑排序的BFS过程来更新深度
      while (!q.empty())
      {
         IdxType u = q.front();
         q.pop();

         for (IdxType v : _label_nav_graph->out_neighbors[u])
         {
            // 更新子节点的深度
            _lng_node_depths[v] = std::max(_lng_node_depths[v], _lng_node_depths[u] + 1);
            in_degree[v]--;
            // 如果子节点的所有父节点都已处理完毕，则将其入队
            if (in_degree[v] == 0)
            {
               q.push(v);
            }
         }
      }

      auto end_time = std::chrono::high_resolution_clock::now();
      std::cout << "   - 完成，耗时: " << std::chrono::duration<double, std::milli>(end_time - start_time).count() << " ms" << std::endl;
   }

   // fxy_add: "困难夹心"策略，生成对UNG和ACORN都具有挑战性的查询任务
   void UniNavGraph::generate_queries_hard_sandwich(
       int N,                            // 要生成的查询总数
       const std::string &output_prefix, // 输出文件路径前缀
       const std::string &dataset,       // 数据集名称
       float parent_min_coverage_ratio,  // 父节点的最小覆盖率阈值
       float child_max_coverage_ratio,   // 子节点的最大覆盖率阈值
       float query_min_selectivity,      // 查询的最小选择率
       float query_max_selectivity)      // 查询的最大选择率
   {
      std::cout << "\n========================================================" << std::endl;
      std::cout << "--- 开始生成“困难夹心”查询 (Hard Sandwich Queries) ---" << std::endl;
      std::cout << "--- 策略: 优先从匹配向量中“图深度最深”的向量挑选查询 ---" << std::endl;
      std::cout << "========================================================" << std::endl;

      // --- 阶段0: 基本检查与准备 ---
      if (_label_nav_graph == nullptr || _label_nav_graph->out_neighbors.empty() || _lng_node_depths.empty())
      {
         std::cerr << "[错误] LNG或其深度信息未构建，请确保build()或load()后调用了_precompute_lng_node_depths()。" << std::endl;
         return;
      }
      if (_num_points == 0)
      {
         std::cerr << "[错误] 数据点数量为0。" << std::endl;
         return;
      }

      auto all_start_time = std::chrono::high_resolution_clock::now();

      // --- 阶段1: 构建倒排索引以快速计算覆盖率 ---
      std::cout << "[步骤 1/4] 正在构建倒排索引以加速覆盖率计算..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      std::unordered_map<LabelType, std::vector<IdxType>> label_to_vectors_map;
      for (IdxType i = 0; i < _num_points; ++i)
      {
         const auto &labels = _base_storage->get_label_set(i);
         for (const auto &label : labels)
         {
            label_to_vectors_map[label].push_back(i);
         }
      }
      auto end_time = std::chrono::high_resolution_clock::now();
      std::cout << " - 完成，耗时: " << std::chrono::duration<double, std::milli>(end_time - start_time).count() << " ms" << std::endl;

      // --- 阶段2: 寻找候选“父子对”并构造“夹心”查询模板 ---
      std::cout << "[步骤 2/4] 正在扫描LNG，寻找候选“父子对”并构造查询模板..." << std::endl;
      start_time = std::chrono::high_resolution_clock::now();

      struct QueryTemplate
      {
         std::vector<LabelType> labels;
         std::vector<IdxType> matching_vectors;
      };
      std::vector<QueryTemplate> valid_templates;

      for (IdxType parent_id = 1; parent_id <= _num_groups; ++parent_id)
      {

         float parent_coverage = (float)_label_nav_graph->covered_sets[parent_id].size() / _num_points;
         if (parent_coverage < parent_min_coverage_ratio)
            continue;

         for (IdxType child_id : _label_nav_graph->out_neighbors[parent_id])
         {
            float child_coverage = (float)_label_nav_graph->covered_sets[child_id].size() / _num_points;
            if (child_coverage > child_max_coverage_ratio)
               continue;

            const auto &parent_labels = _group_id_to_label_set[parent_id];
            const auto &child_labels = _group_id_to_label_set[child_id];

            if (child_labels.size() <= parent_labels.size())
               continue;

            std::vector<LabelType> diff_labels;
            std::set_difference(child_labels.begin(), child_labels.end(), parent_labels.begin(), parent_labels.end(), std::back_inserter(diff_labels));

            if (diff_labels.empty())
               continue;

            for (const auto &new_label : diff_labels)
            {
               std::vector<LabelType> query_labels = parent_labels;
               query_labels.push_back(new_label);
               std::sort(query_labels.begin(), query_labels.end());

               if (query_labels.empty())
                  continue;

               LabelType rarest_label = query_labels[0];
               size_t min_size = label_to_vectors_map.count(rarest_label) ? label_to_vectors_map[rarest_label].size() : 0;
               for (size_t i = 1; i < query_labels.size(); ++i)
               {
                  size_t current_size = label_to_vectors_map.count(query_labels[i]) ? label_to_vectors_map[query_labels[i]].size() : 0;
                  if (current_size < min_size)
                  {
                     min_size = current_size;
                     rarest_label = query_labels[i];
                  }
               }

               if (!label_to_vectors_map.count(rarest_label))
                  continue;

               std::vector<IdxType> result_vectors = label_to_vectors_map[rarest_label];

               for (const auto &label : query_labels)
               {
                  if (label == rarest_label)
                     continue;
                  if (!label_to_vectors_map.count(label))
                  {
                     result_vectors.clear();
                     break;
                  }
                  std::vector<IdxType> temp_intersection;
                  const auto &other_list = label_to_vectors_map[label];
                  std::set_intersection(result_vectors.begin(), result_vectors.end(), other_list.begin(), other_list.end(), std::back_inserter(temp_intersection));
                  result_vectors = std::move(temp_intersection);
               }

               int coverage_count = result_vectors.size();
               float selectivity = (float)coverage_count / _num_points;

               if (selectivity >= query_min_selectivity && selectivity <= query_max_selectivity)
               {
                  valid_templates.push_back({query_labels, result_vectors});
               }
            }
         }
      }
      std::cout << "\r - 扫描进度: 100.0%" << std::endl;
      end_time = std::chrono::high_resolution_clock::now();
      std::cout << " - 完成，找到 " << valid_templates.size() << " 个有效查询模板。耗时: "
                << std::chrono::duration<double, std::milli>(end_time - start_time).count() << " ms" << std::endl;

      if (valid_templates.empty())
      {
         std::cerr << "[警告] 未找到任何满足条件的查询模板，无法生成查询。请尝试放宽参数。" << std::endl;
         return;
      }

      // --- 阶段3: 生成最终查询文件 ---
      std::cout << "[步骤 3/4] 正在从模板库中抽样并生成查询文件..." << std::endl;
      start_time = std::chrono::high_resolution_clock::now();

      std::string fvec_filename = output_prefix + "/" + dataset + "_query.fvecs";
      std::string label_filename = output_prefix + "/" + dataset + "_query_labels.txt";
      std::string stats_filename = output_prefix + "/" + dataset + "_query_stats.txt";

      std::ofstream fvec_file(fvec_filename, std::ios::binary);
      std::ofstream label_file(label_filename);
      std::ofstream stats_file(stats_filename);

      if (!fvec_file.is_open() || !label_file.is_open() || !stats_file.is_open())
      {
         std::cerr << "[错误] 无法打开输出文件。" << std::endl;
         return;
      }

      // ===> 新增：为来源组ID创建一个新文件 <===
      std::string source_group_filename = output_prefix + "/" + dataset + "_query_source_groups.txt";
      std::ofstream source_group_file(source_group_filename);
      if (!source_group_file.is_open())
      {
         std::cerr << "[错误] 无法打开来源组ID的输出文件。" << std::endl;
      }

      uint32_t dim = _base_storage->get_dim();
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<size_t> dist(0, valid_templates.size() - 1);

      struct GeneratedQueryInfo
      {
         int id;
         int coverage_count;
         float selectivity;
         int max_depth;
         std::vector<LabelType> labels;
      };
      std::vector<GeneratedQueryInfo> generated_queries_info;
      int queries_generated = 0;
      for (int i = 0; i < N; ++i)
      {
         const auto &T = valid_templates[dist(gen)];
         const auto &all_matching_vectors = T.matching_vectors;

         if (all_matching_vectors.empty())
            continue;

         int max_depth = -1;
         std::vector<IdxType> deepest_matching_vectors;

         for (IdxType vec_idx : all_matching_vectors)
         {
            IdxType group_id = _new_vec_id_to_group_id[vec_idx];
            int current_depth = _lng_node_depths[group_id];

            if (current_depth > max_depth)
            {
               max_depth = current_depth;
               deepest_matching_vectors.clear();
               deepest_matching_vectors.push_back(vec_idx);
            }
            else if (current_depth == max_depth)
            {
               deepest_matching_vectors.push_back(vec_idx);
            }
         }

         IdxType query_vec_id = -1;

         const std::vector<IdxType> *pool_to_use = &all_matching_vectors;
         if (!deepest_matching_vectors.empty())
         {
            pool_to_use = &deepest_matching_vectors;
         }

         if (pool_to_use->empty())
            continue;

         std::uniform_int_distribution<size_t> vec_dist(0, pool_to_use->size() - 1);
         query_vec_id = (*pool_to_use)[vec_dist(gen)];

         for (size_t j = 0; j < T.labels.size(); ++j)
         {
            label_file << T.labels[j] << (j == T.labels.size() - 1 ? "" : ",");
         }
         label_file << "\n";

         const char *vec_data = _base_storage->get_vector(query_vec_id);
         fvec_file.write(reinterpret_cast<const char *>(&dim), sizeof(uint32_t));
         fvec_file.write(vec_data, dim * sizeof(float));

         // ===> 新增：获取并写入来源组ID <===
         // query_vec_id 是 reordered ID，所以用 _new_vec_id_to_group_id
         IdxType source_group_id = _new_vec_id_to_group_id[query_vec_id];
         source_group_file << source_group_id << "\n";

         int coverage_count = all_matching_vectors.size();
         generated_queries_info.push_back({queries_generated + 1, coverage_count, (float)coverage_count / _num_points, max_depth, T.labels});
         queries_generated++;
      }
      std::cout << std::endl;

      fvec_file.close();
      label_file.close();
      source_group_file.close();

      end_time = std::chrono::high_resolution_clock::now();
      std::cout << " - 完成，耗时: " << std::chrono::duration<double, std::milli>(end_time - start_time).count() << " ms" << std::endl;

      // --- 阶段4: 写入统计信息 ---
      std::cout << "[步骤 4/4] 正在写入统计信息..." << std::endl;

      stats_file << "--- Hard Sandwich Query Generation Stats ---" << std::endl;

      auto now = std::chrono::system_clock::now();
      auto in_time_t = std::chrono::system_clock::to_time_t(now);
      stats_file << "生成时间: " << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X") << std::endl;

      stats_file << "\n--- Summary ---" << std::endl;
      stats_file << "成功生成的查询总数: " << queries_generated << " / " << N << std::endl;
      stats_file << "用于生成的有效模板数: " << valid_templates.size() << std::endl;

      if (!generated_queries_info.empty())
      {
         long long total_coverage = 0;
         int min_coverage = generated_queries_info[0].coverage_count;
         int max_coverage = generated_queries_info[0].coverage_count;
         for (const auto &info : generated_queries_info)
         {
            total_coverage += info.coverage_count;
            if (info.coverage_count < min_coverage)
               min_coverage = info.coverage_count;
            if (info.coverage_count > max_coverage)
               max_coverage = info.coverage_count;
         }
         float avg_coverage = (float)total_coverage / generated_queries_info.size();
         stats_file << "\n--- Coverage Stats of Generated Queries ---" << std::endl;
         stats_file << "最小覆盖向量数: " << min_coverage << " (" << std::fixed << std::setprecision(5) << (float)min_coverage / _num_points * 100 << "%)" << std::endl;
         stats_file << "最大覆盖向量数: " << max_coverage << " (" << (float)max_coverage / _num_points * 100 << "%)" << std::endl;
         stats_file << "平均覆盖向量数: " << avg_coverage << " (" << (float)avg_coverage / _num_points * 100 << "%)" << std::endl;
      }

      stats_file << "\n--- Parameters Used ---" << std::endl;
      stats_file << " - parent_min_coverage_ratio: " << parent_min_coverage_ratio << std::endl;
      stats_file << " - child_max_coverage_ratio: " << child_max_coverage_ratio << std::endl;
      stats_file << " - query_min_selectivity: " << query_min_selectivity << std::endl;
      stats_file << " - query_max_selectivity: " << query_max_selectivity << std::endl;

      stats_file << "\n--- Detailed Query List ---" << std::endl;
      stats_file << "ID, CoverageCount, Selectivity(%), MaxDepth, Labels" << std::endl;
      for (const auto &info : generated_queries_info)
      {
         stats_file << info.id << ", " << info.coverage_count << ", "
                    << std::fixed << std::setprecision(5) << info.selectivity * 100 << ", "
                    << info.max_depth << ", "
                    << "{";
         for (size_t j = 0; j < info.labels.size(); ++j)
         {
            stats_file << info.labels[j] << (j == info.labels.size() - 1 ? "" : ",");
         }
         stats_file << "}" << std::endl;
      }

      stats_file.close();

      auto all_end_time = std::chrono::high_resolution_clock::now();
      std::cout << "\n--- “困难夹心”查询生成完毕 ---" << std::endl;
      std::cout << "总耗时: " << std::chrono::duration<double, std::milli>(all_end_time - all_start_time).count() << " ms" << std::endl;
      std::cout << "查询向量已保存到: " << fvec_filename << std::endl;
      std::cout << "查询标签已保存到: " << label_filename << std::endl;
      std::cout << "生成统计已保存到: " << stats_filename << std::endl;
      std::cout << "查询来源组ID已保存到: " << source_group_filename << std::endl;
      std::cout << "========================================================" << std::endl;
   }

   // fxy_add:生成“困难Top-N稀有标签”查询任务
   void UniNavGraph::generate_queries_hard_top_n_rare(
       int N,                            // 要生成的查询总数
       const std::string &output_prefix, // 输出文件路径前缀
       const std::string &dataset,       // 数据集名称
       int num_rare_labels_to_use,       // 要使用的频率最低的标签数量 (n)
       float query_min_selectivity,      // 查询的最小选择率
       float query_max_selectivity,      // 查询的最大选择率
       int min_frequency_for_rare_labels)
   {
      std::cout << "\n========================================================" << std::endl;
      std::cout << "--- 开始生成“困难Top-N稀有标签”查询 (Hard Top-N Rare Label Queries) ---" << std::endl;
      std::cout << "--- 策略: 轮流使用频率最低的 " << num_rare_labels_to_use << " 个标签生成查询 ---" << std::endl;
      std::cout << "========================================================" << std::endl;

      int min_vectors_for_fallback = 10; // 用于控制备用模式下的最小向量数

      // --- 阶段 0: 基本检查与准备 ---
      if (_label_nav_graph == nullptr || _label_nav_graph->out_neighbors.empty() || _lng_node_depths.empty())
      {
         std::cerr << "[错误] LNG或其深度信息未构建，请确保build()或load()后调用了_precompute_lng_node_depths()。" << std::endl;
         return;
      }
      if (_num_points == 0)
      {
         std::cerr << "[错误] 数据点数量为0。" << std::endl;
         return;
      }
      auto all_start_time = std::chrono::high_resolution_clock::now();

      // --- 阶段 1: 构建倒排索引以快速计算频率和覆盖率 ---
      std::cout << "[步骤 1/5] 正在构建倒排索引..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();
      std::unordered_map<LabelType, std::vector<IdxType>> label_to_vectors_map;
      for (IdxType i = 0; i < _num_points; ++i)
      {
         const auto &labels = _base_storage->get_label_set(i);
         for (const auto &label : labels)
         {
            label_to_vectors_map[label].push_back(i);
         }
      }
      auto end_time = std::chrono::high_resolution_clock::now();
      std::cout << " - 完成，耗时: " << std::chrono::duration<double, std::milli>(end_time - start_time).count() << " ms" << std::endl;

      // --- 阶段 2: 识别Top-N稀有标签 (带频率下限过滤) ---
      std::cout << "[步骤 2/5] 正在识别频率最低的 " << num_rare_labels_to_use
                << " 个标签 (要求频率不低于 " << min_frequency_for_rare_labels << ")..." << std::endl;
      start_time = std::chrono::high_resolution_clock::now();

      std::vector<std::pair<LabelType, size_t>> label_frequencies;
      for (const auto &pair : label_to_vectors_map)
      {
         label_frequencies.push_back({pair.first, pair.second.size()});
      }

      // 按频率升序排序
      std::sort(label_frequencies.begin(), label_frequencies.end(),
                [](const auto &a, const auto &b)
                {
                   return a.second < b.second;
                });

      // 新增逻辑 - 过滤掉频率低于下限的标签
      std::vector<std::pair<LabelType, size_t>> candidate_labels;
      for (const auto &freq_pair : label_frequencies)
      {
         if (freq_pair.second >= (size_t)min_frequency_for_rare_labels)
         {
            candidate_labels.push_back(freq_pair);
         }
      }

      if (candidate_labels.empty())
      {
         std::cerr << "[警告] 没有标签的频率满足不低于 " << min_frequency_for_rare_labels << " 的条件，无法生成查询。" << std::endl;
         return;
      }

      std::cout << " - 共有 " << candidate_labels.size() << " 个标签满足频率下限条件。" << std::endl;

      std::vector<LabelType> top_n_rare_labels;
      std::vector<std::pair<LabelType, size_t>> top_n_rare_labels_with_freq;
      // 从过滤后的候选列表中选择
      int actual_num_to_use = std::min((int)candidate_labels.size(), num_rare_labels_to_use);

      for (int i = 0; i < actual_num_to_use; ++i)
      {
         top_n_rare_labels.push_back(candidate_labels[i].first);
         top_n_rare_labels_with_freq.push_back(candidate_labels[i]);
      }

      end_time = std::chrono::high_resolution_clock::now();
      std::cout << " - 完成，最终选定 " << top_n_rare_labels.size() << " 个稀有标签。耗时: " << std::chrono::duration<double, std::milli>(end_time - start_time).count() << " ms" << std::endl;

      if (top_n_rare_labels.empty())
      {
         std::cerr << "[警告] 未找到任何标签，无法生成查询。" << std::endl;
         return;
      }

      // --- 阶段 3: 为每个Top-N稀有标签构建查询模板库 ---
      std::cout << "[步骤 3/5] 正在为每个稀有标签构建查询模板库..." << std::endl;
      start_time = std::chrono::high_resolution_clock::now();

      struct QueryTemplate
      {
         std::vector<LabelType> labels;
         std::vector<IdxType> matching_vectors;
      };

      std::unordered_map<LabelType, std::vector<QueryTemplate>> rare_label_to_templates_map;

      // 遍历所有唯一的标签组合 (groups) 来寻找模板
      for (const auto &query_labels_vec : _group_id_to_label_set)
      {

         if (query_labels_vec.empty())
            continue;

         // 检查这个组合是否包含我们关注的Top-N稀有标签之一
         for (const auto &rare_label : top_n_rare_labels)
         {
            bool contains_current_rare_label = false;
            for (const auto &label : query_labels_vec)
            {
               if (label == rare_label)
               {
                  contains_current_rare_label = true;
                  break;
               }
            }

            if (contains_current_rare_label)
            {
               // 如果包含，则计算其精确覆盖率并判断是否为有效模板
               std::vector<LabelType> query_labels = query_labels_vec;
               std::sort(query_labels.begin(), query_labels.end());

               // --- 开始计算向量交集 ---
               LabelType rarest_label_in_query = query_labels[0];
               size_t min_size = label_to_vectors_map.count(rarest_label_in_query) ? label_to_vectors_map[rarest_label_in_query].size() : 0;
               for (size_t i = 1; i < query_labels.size(); ++i)
               {
                  size_t current_size = label_to_vectors_map.count(query_labels[i]) ? label_to_vectors_map[query_labels[i]].size() : 0;
                  if (current_size < min_size)
                  {
                     min_size = current_size;
                     rarest_label_in_query = query_labels[i];
                  }
               }

               if (!label_to_vectors_map.count(rarest_label_in_query))
                  continue;

               std::vector<IdxType> result_vectors = label_to_vectors_map.at(rarest_label_in_query);
               for (const auto &label : query_labels)
               {
                  if (label == rarest_label_in_query)
                     continue;
                  if (!label_to_vectors_map.count(label))
                  {
                     result_vectors.clear();
                     break;
                  }
                  std::vector<IdxType> temp_intersection;
                  const auto &other_list = label_to_vectors_map.at(label);
                  std::set_intersection(result_vectors.begin(), result_vectors.end(), other_list.begin(), other_list.end(), std::back_inserter(temp_intersection));
                  result_vectors = std::move(temp_intersection);
                  if (result_vectors.empty())
                     break;
               }
               // --- 交集计算结束 ---

               float selectivity = (float)result_vectors.size() / _num_points;
               if (selectivity >= query_min_selectivity && selectivity <= query_max_selectivity)
               {
                  rare_label_to_templates_map[rare_label].push_back({query_labels, result_vectors});
               }
            }
         }
      }

      size_t total_templates = 0;
      for (const auto &pair : rare_label_to_templates_map)
         total_templates += pair.second.size();

      end_time = std::chrono::high_resolution_clock::now();
      std::cout << " - 完成，共找到 " << total_templates << " 个有效查询模板，分布在 "
                << rare_label_to_templates_map.size() << " 个稀有标签上。耗时: "
                << std::chrono::duration<double, std::milli>(end_time - start_time).count() << " ms" << std::endl;

      if (total_templates == 0)
      {
         std::cout << "[警告] 未找到任何满足条件的查询模板，启动带方向的随机生成模式..." << std::endl;
         std::cout << " - [策略] 所有随机生成的模板必须至少包含 " << min_vectors_for_fallback << " 个向量。" << std::endl;

         // 计算标签选择率
         std::unordered_map<LabelType, float> label_selectivity;
         for (auto &p : label_to_vectors_map)
         {
            label_selectivity[p.first] = (float)p.second.size() / _num_points;
         }

         // 按选择率分池
         std::vector<LabelType> low_freq, mid_freq, high_freq;
         for (auto &kv : label_selectivity)
         {
            if (kv.second < 0.001f)
               low_freq.push_back(kv.first);
            else if (kv.second < 0.2f)
               mid_freq.push_back(kv.first);
            else
               high_freq.push_back(kv.first);
         }

         std::mt19937 rng(std::random_device{}());
         auto pick_random = [&](const std::vector<LabelType> &pool) -> LabelType
         {
            std::uniform_int_distribution<size_t> dist(0, pool.size() - 1);
            return pool[dist(rng)];
         };

         for (auto &rare_label : top_n_rare_labels)
         {
            float rare_sel = label_selectivity[rare_label];
            const std::vector<LabelType> *primary_pool;

            if (rare_sel > query_max_selectivity)
               primary_pool = &low_freq;
            else if (rare_sel < query_min_selectivity)
               primary_pool = &high_freq;
            else
               primary_pool = &mid_freq;

            bool generated = false;
            float best_diff = 1.0f;
            QueryTemplate best_template;

            // 先试 1 标签
            {
               std::vector<LabelType> combo = {rare_label};
               auto &vecs = label_to_vectors_map[rare_label];
               float sel = (float)vecs.size() / _num_points;
               // fxy_add: 增加向量数检查
               if (sel >= query_min_selectivity && sel <= query_max_selectivity && vecs.size() >= (size_t)min_vectors_for_fallback)
               {
                  rare_label_to_templates_map[rare_label].push_back({combo, vecs});
                  generated = true;
                  std::cout << "  [生成] 稀有标签 " << rare_label
                            << " 单标签选择率 " << sel * 100 << "% (" << vecs.size() << " vectors)" << std::endl;
                  continue;
               }
            }

            // 再试 2-5 标签组合
            for (int combo_size = 2; combo_size <= 5 && !generated; ++combo_size)
            {
               for (int attempt = 0; attempt < 500; ++attempt)
               {
                  std::vector<LabelType> combo = {rare_label};
                  std::vector<LabelType> pool = *primary_pool;
                  std::shuffle(pool.begin(), pool.end(), rng);
                  for (int i = 0; i < combo_size - 1 && i < (int)pool.size(); ++i)
                     if (pool[i] != rare_label)
                        combo.push_back(pool[i]);

                  std::sort(combo.begin(), combo.end());
                  combo.erase(std::unique(combo.begin(), combo.end()), combo.end());

                  // 求交集 (此部分代码未变)
                  // ...
                  LabelType rarest_label_in_combo = combo[0];
                  size_t min_size = label_to_vectors_map[rarest_label_in_combo].size();
                  for (size_t i = 1; i < combo.size(); ++i)
                  {
                     size_t cur_size = label_to_vectors_map[combo[i]].size();
                     if (cur_size < min_size)
                     {
                        min_size = cur_size;
                        rarest_label_in_combo = combo[i];
                     }
                  }
                  std::vector<IdxType> result = label_to_vectors_map[rarest_label_in_combo];
                  for (auto &label : combo)
                  {
                     if (label == rarest_label_in_combo)
                        continue;
                     std::vector<IdxType> temp;
                     std::set_intersection(result.begin(), result.end(),
                                           label_to_vectors_map[label].begin(), label_to_vectors_map[label].end(),
                                           std::back_inserter(temp));
                     result.swap(temp);
                     if (result.empty())
                        break;
                  }

                  // fxy_add: 增加对结果向量数的检查
                  if (result.size() < (size_t)min_vectors_for_fallback)
                  {
                     continue; // 如果结果向量太少，直接跳过这次尝试
                  }

                  float sel = (float)result.size() / _num_points;
                  float diff = std::min(std::abs(sel - query_min_selectivity), std::abs(sel - query_max_selectivity));

                  // fxy_add: 此处的向量数检查是冗余但安全的，因为上面已经 continue
                  if (sel >= query_min_selectivity && sel <= query_max_selectivity && result.size() >= (size_t)min_vectors_for_fallback)
                  {
                     rare_label_to_templates_map[rare_label].push_back({combo, result});
                     generated = true;
                     std::cout << "  [生成] 稀有标签 " << rare_label << " 组合: {";
                     for (size_t k = 0; k < combo.size(); ++k)
                        std::cout << combo[k] << (k + 1 < combo.size() ? "," : "");
                     std::cout << "} 选择率 " << sel * 100 << "% (" << result.size() << " vectors)" << std::endl;
                     break;
                  }

                  // fxy_add: 记录最接近的组合时，也必须满足向量数下限
                  // (由于上面的continue，这里的检查也是冗余但安全的)
                  if (diff < best_diff && result.size() >= (size_t)min_vectors_for_fallback)
                  {
                     best_diff = diff;
                     best_template = {combo, result};
                  }
               }
            }

            // 如果还是没生成，使用满足条件的最佳兜底
            if (!generated && !best_template.matching_vectors.empty())
            {
               // fxy_add: 这里的 best_template 已经保证了向量数 >= min_vectors_for_fallback
               rare_label_to_templates_map[rare_label].push_back(best_template);
               std::cout << "  [兜底] 稀有标签 " << rare_label << " 组合: {";
               for (size_t k = 0; k < best_template.labels.size(); ++k)
                  std::cout << best_template.labels[k] << (k + 1 < best_template.labels.size() ? "," : "");
               std::cout << "} 最接近选择率: "
                         << (float)best_template.matching_vectors.size() / _num_points * 100 << "% ("
                         << best_template.matching_vectors.size() << " vectors)" << std::endl;
            }
            else if (!generated)
            {
               std::cout << "  [失败] 稀有标签 " << rare_label << " 未能生成满足条件的模板 (向量数需 >= " << min_vectors_for_fallback << ")" << std::endl;
            }
         }

         total_templates = 0;
         for (auto &p : rare_label_to_templates_map)
            total_templates += p.second.size();
         std::cout << "[完成] 带方向的随机生成完成，总模板数: " << total_templates << std::endl;
      }

      // --- 阶段 4: 轮流抽样并生成最终查询文件 ---
      std::cout << "[步骤 4/5] 正在轮流从模板库中抽样并生成查询文件..." << std::endl;
      start_time = std::chrono::high_resolution_clock::now();

      // fxy_add: --- [修正死循环] ---
      // 1. 创建一个只包含可用稀有标签的列表
      std::vector<LabelType> usable_rare_labels;
      for (const auto &rare_label : top_n_rare_labels)
      {
         if (rare_label_to_templates_map.count(rare_label) && !rare_label_to_templates_map.at(rare_label).empty())
         {
            usable_rare_labels.push_back(rare_label);
         }
      }

      // 2. 如果没有任何标签是可用的，则无法生成查询
      if (usable_rare_labels.empty())
      {
         std::cerr << "[错误] 没有任何稀有标签成功生成了查询模板，无法继续生成查询。" << std::endl;
         return;
      }
      std::cout << " - [信息] " << top_n_rare_labels.size() << " 个稀有标签中，有 " << usable_rare_labels.size() << " 个可用于生成查询。" << std::endl;
      // fxy_add: --- [修正结束] ---

      std::string fvec_filename = output_prefix + "/" + dataset + "_query.fvecs";
      std::string label_filename = output_prefix + "/" + dataset + "_query_labels.txt";
      std::string stats_filename = output_prefix + "/" + dataset + "_query_stats.txt";
      std::string source_group_filename = output_prefix + "/" + dataset + "_query_source_groups.txt";

      std::ofstream fvec_file(fvec_filename, std::ios::binary);
      std::ofstream label_file(label_filename);
      std::ofstream stats_file(stats_filename);
      std::ofstream source_group_file(source_group_filename);

      if (!fvec_file.is_open() || !label_file.is_open() || !stats_file.is_open() || !source_group_file.is_open())
      {
         std::cerr << "[错误] 无法打开一个或多个输出文件。" << std::endl;
         return;
      }

      uint32_t dim = _base_storage->get_dim();
      std::random_device rd;
      std::mt19937 gen(rd());

      struct GeneratedQueryInfo
      {
         int id;
         int coverage_count;
         float selectivity;
         int max_depth;
         std::vector<LabelType> labels;
      };
      std::vector<GeneratedQueryInfo> generated_queries_info;
      int queries_generated = 0;

      for (int i = 0; i < N; ++i)
      {
         // fxy_add: [修正] 从“可用”标签列表中轮流选择
         const auto &current_rare_label = usable_rare_labels[queries_generated % usable_rare_labels.size()];

         // fxy_add: [修正] 以下检查不再需要，因为列表中的所有标签都保证是可用的
         // if (rare_label_to_templates_map.find(current_rare_label) == ... )
         // {
         //     i--;
         //     continue;
         // }

         const auto &template_pool = rare_label_to_templates_map.at(current_rare_label);
         std::uniform_int_distribution<size_t> template_dist(0, template_pool.size() - 1);
         const auto &T = template_pool[template_dist(gen)];

         const auto &all_matching_vectors = T.matching_vectors;
         if (all_matching_vectors.empty())
         {
            // 理论上不应发生，因为我们在兜底模式已过滤了空结果，但作为安全措施保留
            i--; // 如果真的发生了，还是需要作废本次循环
            continue;
         }

         int max_depth = -1;
         std::vector<IdxType> deepest_matching_vectors;
         for (IdxType vec_idx : all_matching_vectors)
         {
            IdxType group_id = _new_vec_id_to_group_id[vec_idx];
            int current_depth = _lng_node_depths[group_id];
            if (current_depth > max_depth)
            {
               max_depth = current_depth;
               deepest_matching_vectors.clear();
               deepest_matching_vectors.push_back(vec_idx);
            }
            else if (current_depth == max_depth)
            {
               deepest_matching_vectors.push_back(vec_idx);
            }
         }

         const std::vector<IdxType> *pool_to_use = &all_matching_vectors;
         if (!deepest_matching_vectors.empty())
         {
            pool_to_use = &deepest_matching_vectors;
         }

         if (pool_to_use->empty())
         {
            i--;
            continue;
         }

         std::uniform_int_distribution<size_t> vec_dist(0, pool_to_use->size() - 1);
         IdxType query_vec_id = (*pool_to_use)[vec_dist(gen)];

         for (size_t j = 0; j < T.labels.size(); ++j)
         {
            label_file << T.labels[j] << (j == T.labels.size() - 1 ? "" : ",");
         }
         label_file << "\n";

         const char *vec_data = _base_storage->get_vector(query_vec_id);
         fvec_file.write(reinterpret_cast<const char *>(&dim), sizeof(uint32_t));
         fvec_file.write(vec_data, dim * sizeof(float));

         IdxType source_group_id = _new_vec_id_to_group_id[query_vec_id];
         source_group_file << source_group_id << "\n";

         int coverage_count = all_matching_vectors.size();
         generated_queries_info.push_back({queries_generated + 1, coverage_count, (float)coverage_count / _num_points, max_depth, T.labels});
         queries_generated++;
      }

      fvec_file.close();
      label_file.close();
      source_group_file.close();
      end_time = std::chrono::high_resolution_clock::now();
      std::cout << " - 完成，成功生成 " << queries_generated << " 个查询。耗时: " << std::chrono::duration<double, std::milli>(end_time - start_time).count() << " ms" << std::endl;

      // --- 阶段 5: 写入统计信息 ---
      std::cout << "[步骤 5/5] 正在写入统计信息..." << std::endl;

      stats_file << "--- Hard Top-N Rare Label Query Generation Stats ---" << std::endl;
      auto now = std::chrono::system_clock::now();
      auto in_time_t = std::chrono::system_clock::to_time_t(now);
#pragma warning(suppress : 4996) // 禁用VS对localtime的警告
      stats_file << "生成时间: " << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X") << std::endl;

      stats_file << "\n--- Summary ---" << std::endl;
      stats_file << "成功生成的查询总数: " << queries_generated << " / " << N << std::endl;
      stats_file << "用于生成的有效模板总数: " << total_templates << std::endl;

      if (!generated_queries_info.empty())
      {
         long long total_coverage = 0;
         int min_coverage = generated_queries_info[0].coverage_count;
         int max_coverage = generated_queries_info[0].coverage_count;
         for (const auto &info : generated_queries_info)
         {
            total_coverage += info.coverage_count;
            min_coverage = std::min(min_coverage, info.coverage_count);
            max_coverage = std::max(max_coverage, info.coverage_count);
         }
         float avg_coverage = (float)total_coverage / generated_queries_info.size();
         stats_file << "\n--- Coverage Stats of Generated Queries ---" << std::endl;
         stats_file << "最小覆盖向量数: " << min_coverage << " (" << std::fixed << std::setprecision(5) << (float)min_coverage / _num_points * 100 << "%)" << std::endl;
         stats_file << "最大覆盖向量数: " << max_coverage << " (" << (float)max_coverage / _num_points * 100 << "%)" << std::endl;
         stats_file << "平均覆盖向量数: " << avg_coverage << " (" << (float)avg_coverage / _num_points * 100 << "%)" << std::endl;
      }

      stats_file << "\n--- Parameters Used ---" << std::endl;
      stats_file << " - num_rare_labels_to_use: " << num_rare_labels_to_use << " (实际使用: " << top_n_rare_labels.size() << ")" << std::endl;
      stats_file << " - query_min_selectivity: " << query_min_selectivity << std::endl;
      stats_file << " - query_max_selectivity: " << query_max_selectivity << std::endl;

      stats_file << "\n--- Top-N Rare Labels Used (Label, Frequency) ---" << std::endl;
      for (const auto &pair : top_n_rare_labels_with_freq)
      {
         stats_file << " - " << pair.first << ", " << pair.second << std::endl;
      }

      stats_file << "\n--- Detailed Query List ---" << std::endl;
      stats_file << "ID, CoverageCount, Selectivity(%), MaxDepth, Labels" << std::endl;
      for (const auto &info : generated_queries_info)
      {
         stats_file << info.id << ", " << info.coverage_count << ", "
                    << std::fixed << std::setprecision(5) << info.selectivity * 100 << ", "
                    << info.max_depth << ", "
                    << "{";
         for (size_t j = 0; j < info.labels.size(); ++j)
         {
            stats_file << info.labels[j] << (j == info.labels.size() - 1 ? "" : ",");
         }
         stats_file << "}" << std::endl;
      }

      stats_file.close();

      auto all_end_time = std::chrono::high_resolution_clock::now();
      std::cout << "\n--- “困难Top-N稀有标签”查询生成完毕 ---" << std::endl;
      std::cout << "总耗时: " << std::chrono::duration<double, std::milli>(all_end_time - all_start_time).count() << " ms" << std::endl;
      std::cout << "查询向量已保存到: " << fvec_filename << std::endl;
      std::cout << "查询标签已保存到: " << label_filename << std::endl;
      std::cout << "生成统计已保存到: " << stats_filename << std::endl;
      std::cout << "查询来源组ID已保存到: " << source_group_filename << std::endl;
      std::cout << "========================================================" << std::endl;
   }

   // ===================================end：限制覆盖率的查询任务生成=========================================
}