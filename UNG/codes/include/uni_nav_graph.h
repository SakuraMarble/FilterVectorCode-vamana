#ifndef UNG_H
#define UNG_H
#include "trie.h"
#include "graph.h"
#include "storage.h"
#include "distance.h"
#include "search_cache.h"
#include "label_nav_graph.h"
#include "vamana/vamana.h"
#include <unordered_map>
#include <bitset>
#include <boost/dynamic_bitset.hpp>
#include <roaring/roaring.h>
#include <roaring/roaring.hh>

using BitsetType = boost::dynamic_bitset<>;

namespace ANNS
{
   struct QueryStats
   {
      float recall;
      double time_ms;
      double flag_time_ms;
      double descendants_merge_time_ms;  // descendants合并耗时
      double coverage_merge_time_ms;     // coverage合并耗时
      double get_min_super_sets_time_ms; // 获取最小入口集合耗时
      double get_group_entry_time_ms;    // 计算入口点耗时
      double entry_group_total_coverage;
      size_t num_distance_calcs;
      size_t num_entry_points;
      size_t num_lng_descendants;
      bool is_global_search;

      size_t num_nodes_visited = 0; // 用于存储search过程中的节点总数

      size_t query_length;            // 查询长度
      long long trie_nodes_traversed; // 存储两种方法的总遍历节点数

      // 方法一相关
      size_t candidate_set_size;
      size_t successful_checks = 0;
      float shortcut_hit_ratio = 0.0f;
      long long redundant_upward_steps = 0; // 向上回溯过程中重复的节点
      // 方法二相关
      size_t recursive_calls = 0;
      size_t pruning_events = 0;
      float pruning_efficiency = 0.0f;
   };
   struct NewEdgeCandidate
   {
      IdxType from;
      IdxType to;
      float distance;

      bool operator<(const NewEdgeCandidate &other) const
      {
         return distance < other.distance;
      }
   };
   struct AcornInUng
   {
      bool ung_and_acorn;
      std::string new_edge_policy;   // 只跑acorn，只跑保障边，两者都跑
      int R_in_add_new_edge;         // 让ACORN为每个查询找20个邻居
      int W_in_add_new_edge;         // 从20个中只考虑最近的10个
      int M_in_add_new_edge;         // 每个点最多有M条新的出/入边
      float layer_depth_retio;       // 候选总数占向量总数的比例
      float query_vector_ratio;      // 查询向量占候选总数的比例
      float root_coverage_threshold; // 单个属性被视为“概念根”的最低覆盖率
      std::string acorn_in_ung_output_path;

      // acorn中的参数
      int M, M_beta, gamma, efs, compute_recall;
   };

   // 定义入口点选择策略的枚举
   enum class SelectionMode
   {
      SizeOnly,       // 策略1：只考虑组的 size
      SizeAndDistance // 策略2：综合考虑 size 和 LNG 跳数
   };

   class UniNavGraph
   {
   public:
      UniNavGraph(IdxType num_nodes) : _label_nav_graph(std::make_shared<LabelNavGraph>(num_nodes)) {} // 修改构造函数以初始化 _label_nav_graph
      UniNavGraph() = default;
      ~UniNavGraph() = default;

      void build(std::shared_ptr<IStorage> base_storage, std::shared_ptr<DistanceHandler> distance_handler,
                 std::string scenario, std::string index_name, uint32_t num_threads, IdxType num_cross_edges,
                 IdxType max_degree, IdxType Lbuild, float alpha, std::string dataset,
                 ANNS::AcornInUng new_cross_edge);

      void search(std::shared_ptr<IStorage> query_storage, std::shared_ptr<DistanceHandler> distance_handler,
                  uint32_t num_threads, IdxType Lsearch, IdxType num_entry_points, std::string scenario,
                  IdxType K, std::pair<IdxType, float> *results, std::vector<float> &num_cmps,
                  std::vector<std::bitset<10000001>> &bitmap);
      void search_hybrid(std::shared_ptr<IStorage> query_storage,
                         std::shared_ptr<DistanceHandler> distance_handler,
                         uint32_t num_threads, IdxType Lsearch,
                         IdxType num_entry_points, std::string scenario,
                         IdxType K, std::pair<IdxType, float> *results,
                         std::vector<float> &num_cmps,
                         std::vector<QueryStats> &query_stats,
                         std::vector<std::bitset<10000001>> &bitmaps,
                         bool is_ori_ung,
                         bool is_select_entry_groups, bool is_rec_more_start,
                         bool is_ung_more_entry, const std::vector<IdxType> &true_query_group_ids = {}); // 包含每个查询其真实来源组ID的向量

      // I/O
      void save(std::string index_path_prefix, std::string results_path_prefix);
      void load(std::string index_path_prefix, const std::string &data_type);

      // query generator

      std::map<std::vector<unsigned int>, int> _subset_count_cache; // 用于缓存标签组合出现次数的map，避免重复进行昂贵的计算。key是排序后的标签组合，value是其在整个数据集中的出现次数。
      int count_subset_occurrences(const std::vector<unsigned int> &sorted_subset);
      void query_generate(std::string &output_prefix, int n, float keep_prob, int K, bool stratified_sampling, bool verify);
      void generate_multiple_queries(std::string dataset,
                                     UniNavGraph &index,
                                     int K,
                                     const std::string &base_output_path,
                                     int num_sets,
                                     int n_per_set,
                                     float keep_prob,
                                     bool stratified_sampling,
                                     bool verify);
      void generate_queries_method1_high_coverage(std::string &output_prefix, std::string dataset, int query_n, std::string &base_label_file, float coverage_threshold);
      void generate_queries_method1_low_coverage(
          std::string &output_prefix,
          std::string dataset,
          int query_n,
          std::string &base_label_file,
          int num_of_per_query_labels,
          float coverage_threshold,
          int K);
      void generate_queries_method2_high_coverage(int N, int K, int top_M_trees, std::string dataset, const std::string &output_prefix, const std::string &base_label_tree_roots);
      void generate_queries_method2_high_coverage_human(
          std::string &output_prefix,
          std::string dataset,
          int query_n,
          std::string &base_label_file,
          std::string &base_label_info_file);
      void generate_queries_method2_low_coverage(
          std::string &output_prefix,
          std::string dataset,
          int query_n,
          std::string &base_label_file,
          int num_of_per_query_labels,
          int K,
          int max_K,
          int min_K);
      void generate_queries_true_data_high_coverage(
          int N,                            // 要生成的查询总数
          int K,                            // 用于计算质心的邻居数量
          int top_M_trees,                  // 选择覆盖率最高的M棵概念树
          std::string dataset,              // 数据集名称，用于生成文件名
          const std::string &output_prefix, // 输出文件路径前缀
          float min_root_coverage_threshold);
      void generate_queries_true_data_low_coverage(
          std::string &output_prefix,
          std::string dataset,
          int query_n,
          std::string &base_label_file,
          int num_of_per_query_labels,
          float coverage_threshold,
          int K);
      void generate_queries_hard_sandwich(
          int N,                            // 要生成的查询总数
          const std::string &output_prefix, // 输出文件路径前缀
          const std::string &dataset,       // 数据集名称
          float parent_min_coverage_ratio,  // 父节点的最小覆盖率阈值 (例如 0.02)
          float child_max_coverage_ratio,   // 子节点的最大覆盖率阈值 (例如 0.005)
          float query_min_selectivity,      // 查询的最小选择率 (例如 0.0005, 即0.05%)
          float query_max_selectivity);     // 查询的最大选择率 (例如 0.01, 即1%)
      std::vector<int> _lng_node_depths;    // 存储每个group_id的图深度,generate_queries_hard_sandwich需要用
      void _precompute_lng_node_depths();
      void generate_queries_hard_top_n_rare(
          int N,                            // 要生成的查询总数
          const std::string &output_prefix, // 输出文件路径前缀
          const std::string &dataset,       // 数据集名称
          int num_rare_labels_to_use,       // 要使用的频率最低的标签数量 (n)
          float query_min_selectivity,      // 查询的最小选择率
          float query_max_selectivity,      // 查询的最大选择率
          int min_frequency_for_rare_labels = 1);

      void load_bipartite_graph(const std::string &filename);
      bool compare_graphs(const ANNS::UniNavGraph &g1, const ANNS::UniNavGraph &g2);
      IdxType _num_points;
      std::vector<std::vector<IdxType>> _vector_attr_graph; // 邻接表表示的图
      std::unordered_map<LabelType, AtrType> _attr_to_id;   // 属性到ID的映射
      std::unordered_map<AtrType, LabelType> _id_to_attr;   // ID到属性的映射
      AtrType _num_attributes;                              // 唯一属性数量

      std::pair<std::bitset<10000001>, double> compute_attribute_bitmap(const std::vector<LabelType> &query_attributes) const; // 构建bitmap

      // 求search中flag需要的数据结构
      std::vector<BitsetType> _lng_descendants_bits; // 每个 group 的后代集合
      std::vector<BitsetType> _covered_sets_bits;    // 每个 group 的覆盖集合
      std::vector<roaring::Roaring> _lng_descendants_rb;
      std::vector<roaring::Roaring> _covered_sets_rb;

      std::vector<IdxType> select_entry_groups(
          const std::vector<IdxType> &minimum_entry_sets, // 基础入口，必须包含
          SelectionMode mode,                             // 选择的策略模式
          size_t top_k,                                   // 除了基础入口外，额外选择 K 个最优入口
          double beta = 1.0,                              // 策略2中，用于调节跳数权重的 beta 值
          IdxType true_query_group_id = 0) const;         // 声明为 const 函数，因为它不应修改图的状态

   private:
      // data
      std::shared_ptr<IStorage> _base_storage,
          _query_storage;
      std::shared_ptr<DistanceHandler> _distance_handler;
      std::shared_ptr<Graph> _graph;
      // IdxType _num_points;

      // trie index and vector groups
      IdxType _num_groups;
      TrieIndex _trie_index;
      std::vector<IdxType> _new_vec_id_to_group_id;
      std::vector<std::vector<IdxType>> _group_id_to_vec_ids;
      std::vector<std::vector<LabelType>> _group_id_to_label_set;
      void build_trie_and_divide_groups();

      // label navigating graph
      std::shared_ptr<LabelNavGraph> _label_nav_graph = nullptr;
      void get_min_super_sets(const std::vector<LabelType> &query_label_set, std::vector<IdxType> &min_super_set_ids,
                              bool avoid_self = false, bool need_containment = true);
      void get_min_super_sets_pruning(const std::vector<LabelType> &query_label_set, std::vector<IdxType> &min_super_set_ids,
                                      bool avoid_self, bool need_containment);
      void get_min_super_sets_roaring(const std::vector<LabelType> &query_label_set, std::vector<IdxType> &min_super_set_ids,
                                      bool avoid_self = false, bool need_containment = true);
      void get_min_super_sets_debug(const std::vector<LabelType> &query_label_set,
                                    std::vector<IdxType> &min_super_set_ids,
                                    bool avoid_self, bool need_containment,
                                    std::atomic<int> &print_counter, bool is_new_trie_method, bool is_rec_more_start, QueryStats &stats);
      void cal_f_coverage_ratio();
      void build_label_nav_graph();
      size_t count_all_descendants(IdxType group_id) const;
      void print_lng_descendants_num(const std::string &filename) const;
      void get_descendants_info();

      // prepare vector storage for each group
      std::vector<IdxType> _new_to_old_vec_ids;
      std::vector<std::pair<IdxType, IdxType>> _group_id_to_range;
      std::vector<std::shared_ptr<IStorage>> _group_storages;
      void prepare_group_storages_graphs();

      // graph indices for each graph
      std::string _index_name;
      std::vector<std::shared_ptr<Graph>> _group_graphs;
      std::vector<IdxType> _group_entry_points;
      void build_graph_for_all_groups();
      void build_complete_graph(std::shared_ptr<Graph> graph, IdxType num_points);
      std::vector<std::shared_ptr<Vamana>> _vamana_instances;

      std::shared_ptr<Graph> _global_graph;
      std::shared_ptr<Vamana> _global_vamana; // 全局 Vamana 实例
      IdxType _global_vamana_entry_point;     // 全局 Vamana 实例的入口点
      void build_global_vamana_graph();

      // build attr_id_graph（数据预处理）
      // std::vector<std::vector<IdxType>> _vector_attr_graph; // 邻接表表示的图
      // std::unordered_map<LabelType, AtrType> _attr_to_id;   // 属性到ID的映射
      // std::unordered_map<AtrType, LabelType> _id_to_attr;   // ID到属性的映射
      // AtrType _num_attributes;                              // 唯一属性数量
      void build_vector_and_attr_graph();
      size_t count_graph_edges() const;
      void save_bipartite_graph_info() const;
      void save_bipartite_graph(const std::string &filename);
      uint32_t compute_checksum() const;
      // void load_bipartite_graph(const std::string &filename);

      // 处理flag的相关函数
      void initialize_lng_descendants_coverage_bitsets();
      void initialize_roaring_bitsets();

      // 添加新的跨组边
      void add_new_distance_oriented_edges(
          const std::string &dataset,
          uint32_t num_threads,
          ANNS::AcornInUng new_cross_edge);
      int _num_distance_oriented_edges;
      const bool ENABLE_SEARCH_PATH_LOGGING = true;
      std::unordered_set<uint64_t> _my_new_edges_set;

      void finalize_intra_group_graphs(); // 用于将局部图ID转换为全局ID

      // index parameters for each graph
      IdxType _max_degree,
          _Lbuild;
      float _alpha;
      uint32_t _num_threads;
      std::string _scenario;

      // cross-group edges
      IdxType _num_cross_edges;
      std::vector<SearchQueue> _cross_group_neighbors;
      void build_cross_group_edges();

      // obtain the final unified navigating graph
      void add_offset_for_uni_nav_graph();

      // obtain entry_points
      std::vector<IdxType> get_entry_points(const std::vector<LabelType> &query_label_set,
                                            IdxType num_entry_points, VisitedSet &visited_set);
      void get_entry_points_given_group_id(IdxType num_entry_points, VisitedSet &visited_set,
                                           IdxType group_id, std::vector<IdxType> &entry_points);

      // search in graph
      IdxType iterate_to_fixed_point(const char *query, std::shared_ptr<SearchCache> search_cache,
                                     IdxType target_id, const std::vector<IdxType> &entry_points,
                                     size_t &num_nodes_visited,
                                     bool clear_search_queue = true, bool clear_visited_set = true);
      // search in global graph
      IdxType iterate_to_fixed_point_global(const char *query, std::shared_ptr<SearchCache> search_cache,
                                            IdxType target_id, const std::vector<IdxType> &entry_points,
                                            bool clear_search_queue = true, bool clear_visited_set = true);

      // statistics
      float _index_time = 0, _label_processing_time = 0, _build_graph_time = 0, _build_vector_attr_graph_time = 0, _cal_descendants_time = 0, _cal_coverage_ratio_time = 0;
      float _build_LNG_time = 0, _build_cross_edges_time = 0;
      float _index_size;
      IdxType _graph_num_edges, _LNG_num_edges;

      // FXY_ADD: 为 add_new_distance_oriented_edges 添加详细计时
      double _cross_edge_step1_time_ms;                     // 步骤1 (识别、采样) 的耗时
      double _cross_edge_step2_acorn_time_ms;               // 步骤2 (执行ACORN) 的耗时
      double _cross_edge_step3_add_dist_edges_time_ms;      // 步骤3 (添加距离驱动边) 的耗时
      double _cross_edge_step4_add_hierarchy_edges_time_ms; // 步骤4 (添加层级保障边) 的耗时
      void statistics();
   };
}

#endif // UNG_H