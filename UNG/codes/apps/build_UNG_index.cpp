#include <chrono>
#include <fstream>
#include <iostream>
#include <boost/program_options.hpp>
#include "uni_nav_graph.h"

namespace po = boost::program_options;

int main(int argc, char **argv)
{

   // common auguments
   std::string data_type, dist_fn, base_bin_file, base_label_file, base_label_info_file, base_label_tree_roots, index_path_prefix, result_path_prefix;
   uint32_t num_threads;
   ANNS::IdxType num_cross_edges;

   // parameters for graph indices
   std::string index_type, scenario;
   ANNS::IdxType max_degree, Lbuild; // Vamana
   float alpha;                      // Vamana

   // parameters for acorn in ung
   bool ung_and_acorn;
   std::string new_edge_policy;
   int R_in_add_new_edge, W_in_add_new_edge, M_in_add_new_edge;
   float layer_depth_retio, query_vector_ratio, root_coverage_threshold;
   std::string acorn_in_ung_output_path;
   int M, M_beta, gamma, efs, compute_recall;

   // if query file is not provided, generate query file
   bool generate_query;
   std::string generate_query_task;
   std::string query_file_path;
   std::string dataset;
   float method1_high_coverage_p = 0.0f;

   try
   {
      po::options_description desc{"Arguments"};

      // common arguments
      desc.add_options()("help,h", "Print information on arguments");
      desc.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                         "data type <int8/uint8/float>");
      desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                         "distance function <L2/IP/cosine>");
      desc.add_options()("base_bin_file", po::value<std::string>(&base_bin_file)->required(),
                         "File containing the base vectors in binary format");
      desc.add_options()("base_label_file", po::value<std::string>(&base_label_file)->required(),
                         "Base label file in txt format");
      desc.add_options()("base_label_info_file", po::value<std::string>(&base_label_info_file)->required(),
                         "Base label info file in log format");
      desc.add_options()("base_label_tree_roots", po::value<std::string>(&base_label_tree_roots)->required(),
                         "base_label_tree_roots");
      desc.add_options()("num_threads", po::value<uint32_t>(&num_threads)->default_value(1),
                         "Number of threads to use");
      desc.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                         "Path prefix for saving the index");
      desc.add_options()("result_path_prefix", po::value<std::string>(&result_path_prefix)->required(),
                         "Path prefix for saving the results");

      // parameters for graph indices
      desc.add_options()("scenario", po::value<std::string>(&scenario)->default_value("general"),
                         "Scenario for building UniNavGraph, <equality/general>");
      desc.add_options()("index_type", po::value<std::string>(&index_type)->default_value("Vamana"),
                         "Type of index to build, <Vamana>");
      desc.add_options()("num_cross_edges", po::value<ANNS::IdxType>(&num_cross_edges)->default_value(ANNS::default_paras::NUM_CROSS_EDGES),
                         "Number of cross edges for building Vamana");
      desc.add_options()("max_degree", po::value<ANNS::IdxType>(&max_degree)->default_value(ANNS::default_paras::MAX_DEGREE),
                         "Max degree for building Vamana");
      desc.add_options()("Lbuild", po::value<uint32_t>(&Lbuild)->default_value(ANNS::default_paras::L_BUILD),
                         "Size of candidate set for building Vamana");
      desc.add_options()("alpha", po::value<float>(&alpha)->default_value(ANNS::default_paras::ALPHA),
                         "Alpha for building Vamana");

      // parameters for acorn in ung
      desc.add_options()("ung_and_acorn", po::value<bool>(&ung_and_acorn)->required(),
                         "ung_and_acorn");
      desc.add_options()("new_edge_policy", po::value<std::string>(&new_edge_policy)->required(),
                         "new_edge_policy");
      desc.add_options()("R_in_add_new_edge", po::value<int>(&R_in_add_new_edge)->required(),
                         "R_in_add_new_edge");
      desc.add_options()("W_in_add_new_edge", po::value<int>(&W_in_add_new_edge)->required(),
                         "W_in_add_new_edge");
      desc.add_options()("M_in_add_new_edge", po::value<int>(&M_in_add_new_edge)->required(),
                         "M_in_add_new_edge");
      desc.add_options()("layer_depth_retio", po::value<float>(&layer_depth_retio)->required(),
                         "The proportion of the total number of candidates to the total number of vectors");
      desc.add_options()("query_vector_ratio", po::value<float>(&query_vector_ratio)->required(),
                         "The proportion of the query vector to the total number of candidates");
      desc.add_options()("root_coverage_threshold", po::value<float>(&root_coverage_threshold)->required(),
                         "A single attribute is regarded as the minimum coverage of the concept root");
      desc.add_options()("acorn_in_ung_output_path", po::value<std::string>(&acorn_in_ung_output_path)->required(),
                         "acorn_in_ung_output_path");
      desc.add_options()("M", po::value<int>(&M)->required(),
                         "M in acorn");
      desc.add_options()("M_beta", po::value<int>(&M_beta)->required(),
                         "M_beta in acorn");
      desc.add_options()("gamma", po::value<int>(&gamma)->required(),
                         "gamma in acorn");
      desc.add_options()("efs", po::value<int>(&efs)->required(),
                         "efs in acorn");
      desc.add_options()("compute_recall", po::value<int>(&compute_recall)->required(),
                         "compute_recall in acorn");

      // query file
      desc.add_options()("generate_query", po::value<bool>(&generate_query)->required(),
                         "Whether to generate query file");
      desc.add_options()("generate_query_task", po::value<std::string>(&generate_query_task)->required(),
                         "generate_query_task");
      desc.add_options()("query_file_path", po::value<std::string>(&query_file_path)->default_value("my_words_query"),
                         "Query label file");
      desc.add_options()("dataset", po::value<std::string>(&dataset)->required(),
                         "dataset");
      desc.add_options()("method1_high_coverage_p", po::value<float>(&method1_high_coverage_p)->required(),
                         "method1_high_coverage_p");

      po::variables_map vm;
      po::store(po::parse_command_line(argc, argv, desc), vm);
      if (vm.count("help"))
      {
         std::cout << desc;
         return 0;
      }
      po::notify(vm);
   }
   catch (const std::exception &ex)
   {
      std::cerr << ex.what() << std::endl;
      return -1;
   }

   // check scenario
   if (scenario != "general" && scenario != "equality")
   {
      std::cerr << "Invalid scenario: " << scenario << std::endl;
      return -1;
   }

   // load base data
   std::shared_ptr<ANNS::IStorage> base_storage = ANNS::create_storage(data_type);
   base_storage->load_from_file(base_bin_file, base_label_file);

   // preparation
   std::cout << "Building Unified Navigating Graph index based on " << index_type << " algorithm ..." << std::endl;
   std::shared_ptr<ANNS::DistanceHandler> distance_handler = ANNS::get_distance_handler(data_type, dist_fn);

   // AcornInUng new_cross_edge
   ANNS::AcornInUng new_cross_edge;
   new_cross_edge.ung_and_acorn = ung_and_acorn;
   new_cross_edge.new_edge_policy = new_edge_policy;
   new_cross_edge.R_in_add_new_edge = R_in_add_new_edge;
   new_cross_edge.W_in_add_new_edge = W_in_add_new_edge;
   new_cross_edge.M_in_add_new_edge = M_in_add_new_edge;
   new_cross_edge.layer_depth_retio = layer_depth_retio;
   new_cross_edge.query_vector_ratio = query_vector_ratio;
   new_cross_edge.root_coverage_threshold = root_coverage_threshold;
   new_cross_edge.M = M;
   new_cross_edge.M_beta = M_beta;
   new_cross_edge.gamma = gamma;
   new_cross_edge.efs = efs;
   new_cross_edge.compute_recall = compute_recall;
   new_cross_edge.acorn_in_ung_output_path = acorn_in_ung_output_path;

   // build index
   ANNS::UniNavGraph index;
   auto start_time = std::chrono::high_resolution_clock::now();
   index.build(base_storage, distance_handler, scenario, index_type, num_threads, num_cross_edges, max_degree, Lbuild, alpha, dataset, new_cross_edge);
   std::cout << "Index time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

   // save index
   index.save(index_path_prefix, result_path_prefix);

   // 测试读取向量-属性二分图的函数
   // ANNS::UniNavGraph index2;
   // index2.load_bipartite_graph(index_path_prefix + "vector_attr_graph");
   // index2.compare_graphs(index, index2);

   if (generate_query)
   {
      // 生成查询标签和查询向量：(文件名，每个group中有几个查询向量, 每个属性的概率, 是否分层抽样, 是否验证是子集)
      std::cout << "Generating query file ..." << std::endl;
      std::cout << "Query file path: " << query_file_path << std::endl;
      // index.generate_multiple_queries(dataset, index, query_file_path, 1, 1, 0.5f, false, true); //根据LNG的group生成查询任务
      if (generate_query_task == "method1_high_coverage") // 极端数据方法1-高覆盖率
      {
         index.generate_queries_method1_high_coverage(query_file_path, dataset, 1000, base_label_file, method1_high_coverage_p);
      }
      else if (generate_query_task == "method1_low_coverage") // 极端数据方法1-低覆盖率:选出覆盖率在 (0, coverage_threshold=0.1] 区间且出现次数 ≥ K=10 的组合
      {
         index.generate_queries_method1_low_coverage(query_file_path, dataset, 1000, base_label_file, 7, 0.1f, 10);
      }
      else if (generate_query_task == "method2_high_coverage") // 极端数据方法2-高覆盖率
      {
         //(int N, int K, int top_M_trees, std::string dataset, const std::string &output_prefix, const std::string &base_label_tree_roots);
         index.generate_queries_method2_high_coverage(1000, 10, 2, dataset, query_file_path, base_label_tree_roots);
      }
      else if (generate_query_task == "method2_low_coverage") // 极端数据方法2-低覆盖率
      {
         index.generate_queries_method2_low_coverage(query_file_path, dataset, 1000, base_label_file, 7, 10, 50, 5);
      }
      else if (generate_query_task == "true_data_high_coverage")
      {
         index.generate_queries_true_data_high_coverage(1000, 10, 2, dataset, query_file_path, 0.05);
      }
      else if (generate_query_task == "true_data_low_coverage")
      {
         index.generate_queries_true_data_low_coverage(query_file_path, dataset, 1000, base_label_file, 7, 0.1f, 10);
      }
      else if (generate_query_task == "hard_sandwich") // 极端数据方法3-硬三明治
      {
         index.generate_queries_hard_sandwich(
             3000,            // 生成查询的个数
             query_file_path, // 输出目录
             dataset,         // 数据集名字
             0.0f,            // 父节点覆盖率至少
             1.0f,            // 子节点覆盖率至多
             0.0001f,         // 查询选择率至少
             0.1f             // 查询选择率至多
         );
      }

      else
      {
         std::cout << "Error in generate_query_task" << std::endl;
      }
   }

   return 0;
}