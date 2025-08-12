#include <chrono>
#include <fstream>
#include <numeric>
#include <iostream>
#include <bitset>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include "uni_nav_graph.h"
#include "utils.h"
#include <roaring/roaring.h>
#include <roaring/roaring.hh>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

// 辅助函数：计算单个查询的recall
float calculate_single_query_recall(const std::pair<ANNS::IdxType, float> *gt,
                                    const std::pair<ANNS::IdxType, float> *results,
                                    ANNS::IdxType K)
{
   std::unordered_set<ANNS::IdxType> gt_set;
   for (int i = 0; i < K; ++i)
   {
      if (gt[i].first != -1)
      {
         gt_set.insert(gt[i].first);
      }
   }

   int correct = 0;
   for (int i = 0; i < K; ++i)
   {
      if (results[i].first != -1 && gt_set.count(results[i].first))
      {
         correct++;
      }
   }

   return static_cast<float>(correct) / gt_set.size();
}

int main(int argc, char **argv)
{
   std::string data_type, dist_fn, scenario;
   std::string base_bin_file, query_bin_file, base_label_file, query_label_file, gt_file, index_path_prefix, result_path_prefix, query_group_id_file;
   ANNS::IdxType K, num_entry_points;
   std::vector<ANNS::IdxType> Lsearch_list;
   uint32_t num_threads;
   bool is_new_method = false;      // true: use new method
   bool is_ori_ung = false;         // true: use original ung
   bool is_new_trie_method = false; // false:默认的mini entry groups
   int num_repeats = 1;             // 默认重复1次

   try
   {
      po::options_description desc{"Arguments"};
      desc.add_options()("help,h", "Print information on arguments");
      desc.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                         "data type <int8/uint8/float>");
      desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                         "distance function <L2/IP/cosine>");
      desc.add_options()("base_bin_file", po::value<std::string>(&base_bin_file)->required(),
                         "File containing the base vectors in binary format");
      desc.add_options()("query_bin_file", po::value<std::string>(&query_bin_file)->required(),
                         "File containing the query vectors in binary format");
      desc.add_options()("base_label_file", po::value<std::string>(&base_label_file)->default_value(""),
                         "Base label file in txt format");
      desc.add_options()("query_label_file", po::value<std::string>(&query_label_file)->default_value(""),
                         "Query label file in txt format");
      desc.add_options()("gt_file", po::value<std::string>(&gt_file)->required(),
                         "Filename for the computed ground truth in binary format");
      desc.add_options()("K", po::value<ANNS::IdxType>(&K)->required(),
                         "Number of ground truth nearest neighbors to compute");
      desc.add_options()("num_threads", po::value<uint32_t>(&num_threads)->default_value(ANNS::default_paras::NUM_THREADS),
                         "Number of threads to use");
      desc.add_options()("result_path_prefix", po::value<std::string>(&result_path_prefix)->required(),
                         "Path to save the querying result file");
      desc.add_options()("query_group_id_file", po::value<std::string>(&query_group_id_file)->required(),
                         "query_group_id_file");

      // graph search parameters
      desc.add_options()("scenario", po::value<std::string>(&scenario)->default_value("containment"),
                         "Scenario for building UniNavGraph, <equality/containment/overlap/nofilter>");
      desc.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                         "Prefix of the path to load the index");
      desc.add_options()("num_entry_points", po::value<ANNS::IdxType>(&num_entry_points)->default_value(ANNS::default_paras::NUM_ENTRY_POINTS),
                         "Number of entry points in each entry group");
      desc.add_options()("Lsearch", po::value<std::vector<ANNS::IdxType>>(&Lsearch_list)->multitoken()->required(),
                         "Number of candidates to search in the graph");
      desc.add_options()("is_new_method", po::value<bool>(&is_new_method)->required(),
                         "is_new_method");
      desc.add_options()("is_ori_ung", po::value<bool>(&is_ori_ung)->required(),
                         "is_ori_ung");
      desc.add_options()("is_new_trie_method", po::value<bool>(&is_new_trie_method)->required(),
                         "is_new_trie_method");
      desc.add_options()("num_repeats", po::value<int>(&num_repeats)->default_value(1),
                         "Number of repeats for each Lsearch value");

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
   if (scenario != "containment" && scenario != "equality" && scenario != "overlap")
   {
      std::cerr << "Invalid scenario: " << scenario << std::endl;
      return -1;
   }

   // load query data
   std::shared_ptr<ANNS::IStorage> query_storage = ANNS::create_storage(data_type);
   query_storage->load_from_file(query_bin_file, query_label_file);

   // load index
   ANNS::UniNavGraph index(query_storage->get_num_points());
   index.load(index_path_prefix, data_type);
   index.load_bipartite_graph(index_path_prefix + "vector_attr_graph");

   // 加载查询来源组ID文件
   std::vector<ANNS::IdxType> true_query_group_ids;
   std::ifstream source_group_file(query_group_id_file);
   if (source_group_file.is_open())
   {
      ANNS::IdxType group_id;
      while (source_group_file >> group_id)
      {
         true_query_group_ids.push_back(group_id);
      }
      source_group_file.close();
      std::cout << "成功加载 " << true_query_group_ids.size() << " 个查询的来源组ID。" << std::endl;
   }
   else // 即使没找到，程序也可以继续，只是没有优化效果
   {
      std::cerr << "警告：未找到查询来源组ID文件: " << query_group_id_file << std::endl;
   }

   // preparation
   auto num_queries = query_storage->get_num_points();
   std::shared_ptr<ANNS::DistanceHandler> distance_handler = ANNS::get_distance_handler(data_type, dist_fn);
   auto gt = new std::pair<ANNS::IdxType, float>[num_queries * K];
   ANNS::load_gt_file(gt_file, gt, num_queries, K);
   auto results = new std::pair<ANNS::IdxType, float>[num_queries * K];

   // compute attribute bitmap
   std::vector<std::pair<std::bitset<10000001>, double>> bitmap_and_time(num_queries);
   std::vector<std::bitset<10000001>> bitmap(num_queries);
#pragma omp parallel for
   for (int id = 0; id < num_queries; id++)
   {
      bitmap_and_time[id] = index.compute_attribute_bitmap(query_storage->get_label_set(id));
      bitmap[id] = bitmap_and_time[id].first;
   }

   // init query stats
   std::vector<std::vector<std::vector<ANNS::QueryStats>>> query_stats(num_repeats, std::vector<std::vector<ANNS::QueryStats>>(Lsearch_list.size(), std::vector<ANNS::QueryStats>(num_queries))); //(repeat,Lsearch,queryID)

   for (int repeat = 0; repeat < num_repeats; ++repeat)
   {
      std::cout << "\n=== Repeat " << (repeat + 1) << "/" << num_repeats << " ===" << std::endl;

      // search
      std::vector<float> all_cmps, all_qpss, all_recalls;
      std::vector<float> all_time_ms, all_flag_time, all_bitmap_time, all_entry_points, all_lng_descendants, all_entry_group_coverage;
      std::vector<float> all_is_global_search; // 如果需要统计全局搜索比例

      std::cout << "Start querying ..." << std::endl;
      for (int LsearchId = 0; LsearchId < Lsearch_list.size(); LsearchId++) // auto Lsearch : Lsearch_list
      {
         std::vector<float> num_cmps(num_queries);
         auto start_time = std::chrono::high_resolution_clock::now();
         if (!is_new_method)
            index.search(query_storage, distance_handler, num_threads, Lsearch_list[LsearchId], num_entry_points, scenario, K, results, num_cmps, bitmap);
         else
            index.search_hybrid(query_storage, distance_handler, num_threads, Lsearch_list[LsearchId],
                                num_entry_points, scenario, K, results, num_cmps, query_stats[repeat][LsearchId], bitmap, is_ori_ung, is_new_trie_method, true_query_group_ids);
         auto time_cost = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
         for (int i = 0; i < num_queries; ++i)
            query_stats[repeat][LsearchId][i].recall = calculate_single_query_recall(gt + i * K, results + i * K, K);
      }
   }

   // 输出详细文件
   std::ofstream detail_out(result_path_prefix + "query_details_repeat" + std::to_string(num_repeats) + ".csv");
   detail_out << "repeat,Lsearch,QueryID,Time(ms),get_minsuper_sets_time(ms),get_entry_group_start_time(ms),descendants_merge_time(ms),coverage_merge_time(ms),flag_time(ms),bitmap_time(ms),UNG_time(ms),DistanceCalcs,EntryPoints,LNGDescendants,entry_group_total_coverage,QPS,Recall,is_global_search\n";

   for (int repeat = 0; repeat < num_repeats; repeat++)
   {
      for (int LsearchId = 0; LsearchId < Lsearch_list.size(); LsearchId++)
      {
         for (int i = 0; i < num_queries; ++i)
         {
            detail_out << repeat << ","
                       << Lsearch_list[LsearchId] << ","
                       << i << ","
                       << query_stats[repeat][LsearchId][i].time_ms << ","
                       << query_stats[repeat][LsearchId][i].get_min_super_sets_time_ms << ","
                       << query_stats[repeat][LsearchId][i].get_group_entry_time_ms << ","
                       << query_stats[repeat][LsearchId][i].descendants_merge_time_ms << ","
                       << query_stats[repeat][LsearchId][i].coverage_merge_time_ms << ","
                       << query_stats[repeat][LsearchId][i].flag_time_ms << ","
                       << bitmap_and_time[i].second << ","
                       << query_stats[repeat][LsearchId][i].time_ms - query_stats[repeat][LsearchId][i].flag_time_ms << ","
                       << query_stats[repeat][LsearchId][i].num_distance_calcs << ","
                       << query_stats[repeat][LsearchId][i].num_entry_points << ","
                       << query_stats[repeat][LsearchId][i].num_lng_descendants << ","
                       << query_stats[repeat][LsearchId][i].entry_group_total_coverage << ","
                       << 1000.0 / (query_stats[repeat][LsearchId][i].time_ms) << ","
                       << query_stats[repeat][LsearchId][i].recall << ","
                       << query_stats[repeat][LsearchId][i].is_global_search << "\n";
         }
      }
   }
   detail_out.close();
   // std::cout << "- Lsearch=" << Lsearch_list[LsearchId] << ", time=" << time_cost << "ms" << std::endl;
   // all_qpss.push_back(num_queries * 1000.0 / time_cost);
   // all_cmps.push_back(std::accumulate(num_cmps.begin(), num_cmps.end(), 0.00f) / num_queries);
   // all_recalls.push_back(ANNS::calculate_recall(gt, results, num_queries, K));
   // std::ofstream out(result_path_prefix + "result_avg_repeat" + std::to_string(repeat) + ".csv");
   // out << "L,Cmps,QPS,Recall,Time(ms),Flag_time(ms),Bitmap_time(ms),EntryPoints,LNGDescendants,entry_group_total_coverage\n";
   // for (auto i = 0; i < Lsearch_list.size(); i++)
   // {
   //    out << Lsearch_list[i] << ","
   //        << all_cmps[i] << ","
   //        << all_qpss[i] << ","
   //        << all_recalls[i] / 100.00 << ","
   //        << all_time_ms[i] << ","
   //        << all_flag_time[i] << ","
   //        << all_bitmap_time[i] << ","
   //        << all_entry_points[i] << ","
   //        << all_lng_descendants[i] << ","
   //        << all_entry_group_coverage[i] << "\n";
   // }
   // out.close();

   std::cout << "- all done" << std::endl;
   return 0;
}