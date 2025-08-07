#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <stdexcept> // For std::runtime_error
#include <unordered_set>

// FAISS and system headers
#include <sys/time.h>
#include <sys/stat.h>
#include <unistd.h>
#include "../faiss/IndexACORN.h"
#include "../faiss/index_io.h"
#include "../faiss/utils/distances.h"
#include "utils.cpp"

// Function to parse command-line arguments
std::map<std::string, std::string> parse_arguments(int argc, char *argv[])
{
   std::map<std::string, std::string> args;
   for (int i = 1; i < argc; ++i)
   {
      std::string arg = argv[i];
      if (arg.rfind("--", 0) == 0)
      {
         std::string key = arg.substr(2);
         if (i + 1 < argc && (std::string(argv[i + 1]).rfind("--", 0) != 0))
         {
            args[key] = argv[++i];
         }
         else
         {
            args[key] = "1"; // Flag argument
         }
      }
   }
   return args;
}

// Function to print usage instructions
void print_usage(const char *prog_name)
{
   std::cerr << "用法: " << prog_name << " [参数]\n\n"
             << "必需参数:\n"
             << "  --dataset <string>         数据集名称 (例如: sift1m)\n"
             << "  --base_path <path>         基础向量数据目录\n"
             << "  --base_label_path <path>   基础向量属性标签目录\n"
             << "  --query_vec_path <path>    查询向量文件路径\n"
             << "  --query_attr_path <path>   查询属性文件目录\n"
             << "  --output_path <path>       结果输出文件路径\n"
             << "  --N <int>                  数据库向量数量\n"
             << "  --M <int>                  ACORN图的邻居数\n"
             << "  --M_beta <int>             ACORN压缩层的邻居数\n"
             << "  --gamma <int>              ACORN的属性分区数\n"
             << "  --efs <int>                ACORN的搜索参数efSearch\n"
             << "  --k <int>                  要查找的最近邻数量\n"
             << "  --threads <int>            使用的线程数\n\n"
             << "可选参数:\n"
             << "  --compute_recall           启用召回率计算 (耗时较长)\n"
             << std::endl;
}

// Function to compute ground truth with attribute filtering
std::vector<std::vector<faiss::idx_t>> compute_ground_truth(
    size_t nq, size_t N, size_t d, int k,
    const std::vector<float> &xq, const std::vector<float> &xb,
    const std::vector<char> &filter_ids_map)
{
   std::vector<std::vector<faiss::idx_t>> ground_truth(nq);
#pragma omp parallel for
   for (int i = 0; i < nq; ++i)
   {
      const float *query_vector = xq.data() + i * d;
      std::vector<std::pair<float, faiss::idx_t>> distances;

      for (int j = 0; j < N; ++j)
      {
         if (filter_ids_map[i * N + j])
         {
            const float *base_vector = xb.data() + j * d;
            float dist = faiss::fvec_L2sqr(query_vector, base_vector, d);
            distances.push_back({dist, (faiss::idx_t)j});
         }
      }
      std::sort(distances.begin(), distances.end());
      for (int m = 0; m < k && m < distances.size(); ++m)
      {
         ground_truth[i].push_back(distances[m].second);
      }
   }
   return ground_truth;
}

// Function to calculate per-query recall and return a vector of recalls
std::vector<float> calculate_per_query_recall(
    const std::vector<faiss::idx_t> &results_labels,
    const std::vector<std::vector<faiss::idx_t>> &ground_truth,
    size_t nq, int k)
{
   std::vector<float> per_query_recalls;
   per_query_recalls.reserve(nq);

   for (int i = 0; i < nq; ++i)
   {
      const auto &gt_set = ground_truth[i];
      size_t gt_size = gt_set.size();

      if (gt_size == 0)
      {
         per_query_recalls.push_back(1.0f);
         continue;
      }

      std::unordered_set<faiss::idx_t> gt_unordered_set(gt_set.begin(), gt_set.end());
      long long found_count = 0;

      for (int j = 0; j < k; ++j)
      {
         faiss::idx_t result_id = results_labels[i * k + j];
         if (gt_unordered_set.count(result_id))
         {
            found_count++;
         }
      }
      per_query_recalls.push_back((float)found_count / gt_size);
   }
   return per_query_recalls;
}

int main(int argc, char *argv[])
{
   // --- 0. Initial Setup ---
   double t_start = elapsed();
   std::cout << "=======================================\n"
             << "===   ACORN Search Tool for UNG   ===\n"
             << "=======================================\n";

   // --- 1. Argument Parsing ---
   auto args = parse_arguments(argc, argv);
   std::vector<std::string> required_args = {
       "dataset", "base_path", "base_label_path", "query_vec_path",
       "query_attr_path", "output_path", "N", "M", "M_beta", "gamma", "efs", "k", "threads"};

   for (const auto &key : required_args)
   {
      if (args.find(key) == args.end())
      {
         std::cerr << "错误: 缺少必需参数 --" << key << std::endl;
         print_usage(argv[0]);
         return 1;
      }
   }

   // --- 2. Load Parameters ---
   std::string dataset = args["dataset"];
   std::string base_path = args["base_path"];
   std::string base_label_path = args["base_label_path"];
   std::string query_vec_path = args["query_vec_path"];
   std::string query_attr_path = args["query_attr_path"];
   std::string output_path = args["output_path"];
   size_t N = std::stoul(args["N"]);
   int M = std::stoi(args["M"]);
   int M_beta = std::stoi(args["M_beta"]);
   int gamma = std::stoi(args["gamma"]);
   int efs = std::stoi(args["efs"]);
   int k = std::stoi(args["k"]);
   int nthreads = std::stoi(args["threads"]);
   bool compute_recall = args.count("compute_recall") > 0;

   omp_set_num_threads(nthreads);
   std::cout << "参数解析完成. 使用 " << nthreads << " 个线程." << std::endl;
   if (compute_recall)
   {
      std::cout << "召回率计算已启用." << std::endl;
   }

   // --- 3. Load Data & Attributes ---
   // Note: We use std::vector for safer memory management.
   size_t d = 0;
   size_t nq = 0;
   std::vector<float> xb;
   std::vector<float> xq;
   std::vector<std::vector<int>> metadata;
   std::vector<std::vector<int>> aq;

   try
   {
      // Load base vectors
      std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] 正在加载数据库向量..." << std::endl;
      size_t d_base, nb;
      std::string base_filename = base_path + "/" + dataset + "_base.fvecs";
      float *xb_raw = fvecs_read(base_filename.c_str(), &d_base, &nb);
      if (!xb_raw)
         throw std::runtime_error("无法读取基础向量文件: " + base_filename);
      d = d_base;
      xb.assign(xb_raw, xb_raw + nb * d);
      delete[] xb_raw; // Immediately free raw pointer after copy
      std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] 加载了 " << nb << " 个数据库向量 (维度: " << d << ")." << std::endl;

      // Load query vectors
      std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] 正在加载查询向量..." << std::endl;
      size_t d_query;
      float *xq_raw = fvecs_read(query_vec_path.c_str(), &d_query, &nq);
      if (!xq_raw)
         throw std::runtime_error("无法读取查询向量文件: " + query_vec_path);
      if (d != d_query)
         throw std::runtime_error("基础向量和查询向量的维度不匹配!");
      xq.assign(xq_raw, xq_raw + nq * d);
      delete[] xq_raw;
      std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] 加载了 " << nq << " 个查询向量 (维度: " << d << ")." << std::endl;

      // Load base attributes
      std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] 正在加载数据库属性..." << std::endl;
      metadata = load_ab_muti(dataset, gamma, "rand", N, base_label_path);
      metadata.resize(N);
      for (auto &vec : metadata)
      {
         std::sort(vec.begin(), vec.end());
      }
      std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] 加载了 " << metadata.size() << " 个数据库条目的属性." << std::endl;

      // Load query attributes
      std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] 正在加载查询属性..." << std::endl;
      aq = load_txt_to_vector_multi<int>(query_attr_path);
      for (auto &vec : aq)
      {
         std::sort(vec.begin(), vec.end());
      }
      std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] 加载了 " << aq.size() << " 个查询的属性." << std::endl;

      if (nq != aq.size())
      {
         throw std::runtime_error("查询向量数量和查询属性数量不匹配!");
      }
   }
   catch (const std::exception &e)
   {
      std::cerr << "数据加载时发生错误: " << e.what() << std::endl;
      return 1;
   }

   // --- 4. Build ACORN Index ---
   std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] 正在创建 ACORN 索引 (M=" << M << ", gamma=" << gamma << ")..." << std::endl;
   faiss::IndexACORNFlat hybrid_index(d, M, gamma, metadata, M_beta);
   double t_build_0 = elapsed();
   hybrid_index.add(N, xb.data());
   double t_build_1 = elapsed();
   std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] 成功将 " << N << " 个向量添加到索引. 构建耗时: " << t_build_1 - t_build_0 << " s." << std::endl;
   hybrid_index.printStats(false);

   // --- 5. Perform Filtered Search ---
   std::cout << "==================== 开始搜索 ====================\n";
   std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] 正在为 " << nq << " 个查询搜索 " << k << " 个近邻 (efs=" << efs << ")..." << std::endl;

   hybrid_index.acorn.efSearch = efs;

   std::vector<faiss::idx_t> result_labels(k * nq, -1);
   std::vector<float> result_dists(k * nq, -1.0f);

   std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] 正在创建属性过滤器..." << std::endl;
   double t_filter_0 = elapsed();
   std::vector<char> filter_ids_map(nq * N);
#pragma omp parallel for
   for (int i_q = 0; i_q < nq; i_q++)
   {
      for (int i_b = 0; i_b < N; i_b++)
      {
         const auto &query_attrs = aq[i_q];
         const auto &data_attrs = metadata[i_b];
         filter_ids_map[i_q * N + i_b] = std::includes(
             data_attrs.begin(), data_attrs.end(),
             query_attrs.begin(), query_attrs.end());
      }
   }
   double t_filter_1 = elapsed();
   std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] 属性过滤器创建完成. 耗时: " << t_filter_1 - t_filter_0 << " s" << std::endl;

   double t_search_0 = elapsed();
   hybrid_index.search(nq, xq.data(), k, result_dists.data(), result_labels.data(), filter_ids_map.data(), nullptr, nullptr, nullptr, true);
   double t_search_1 = elapsed();

   double search_time = t_search_1 - t_search_0;
   double qps = (search_time > 0) ? (nq / search_time) : 0;
   std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] 搜索完成. 总耗时: " << search_time << " s, QPS: " << std::fixed << std::setprecision(2) << qps << std::endl;

   // --- 6. Calculate Recall (Optional) ---
   if (compute_recall)
   {
      std::cout << "==================== 计算召回率 ====================\n";
      std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] 正在计算真实近邻 (Ground Truth)，这可能需要很长时间..." << std::endl;
      double t_gt_0 = elapsed();
      auto ground_truth = compute_ground_truth(nq, N, d, k, xq, xb, filter_ids_map);
      double t_gt_1 = elapsed();
      std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] 真实近邻计算完成. 耗时: " << t_gt_1 - t_gt_0 << " s\n";

      // 1. Calculate recall for each query individually
      auto per_query_recalls = calculate_per_query_recall(result_labels, ground_truth, nq, k);

      double total_recall_sum = 0.0;
      int queries_with_gt = 0;

      std::cout << "\n--- 每个查询的召回率 (Recall@" << k << ") ---\n";
      std::cout << std::fixed << std::setprecision(4);
      for (size_t i = 0; i < per_query_recalls.size(); ++i)
      {
         std::cout << "Query " << std::setw(4) << i << ": " << per_query_recalls[i] << std::endl;
         if (!ground_truth[i].empty())
         {
            total_recall_sum += per_query_recalls[i];
            queries_with_gt++;
         }
      }

      // 2. Calculate the average of the per-query recalls
      double average_recall = (queries_with_gt > 0) ? total_recall_sum / queries_with_gt : 0.0;

      std::cout << "\n>>> 平均召回率 (Macro Average Recall@" << k << "): " << average_recall << "\n\n";
   }

   // --- 7. Save Results ---
   std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] 正在将结果 (包括完整向量) 写入到 " << output_path << "...\n";
   std::ofstream out_file(output_path);
   if (!out_file.is_open())
   {
      std::cerr << "错误: 无法打开输出文件 " << output_path << std::endl;
      return 1;
   }
   out_file << std::fixed << std::setprecision(6);
   for (int i = 0; i < nq; ++i)
   {
      out_file << "Query: " << i << "\n";
      for (int j = 0; j < k; ++j)
      {
         faiss::idx_t neighbor_id = result_labels[i * k + j];
         if (neighbor_id < 0)
         {
            out_file << "NeighborID: -1 Vector: \n";
            continue;
         }
         out_file << "NeighborID: " << neighbor_id << " Vector:";
         const float *vector_data = xb.data() + neighbor_id * d;
         for (int dim = 0; dim < d; ++dim)
         {
            out_file << " " << vector_data[dim];
         }
         out_file << "\n";
      }
      out_file << "---\n";
   }
   out_file.close();
   std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] 结果写入完成.\n";

   // --- 8. Cleanup & Exit ---
   // No manual delete[] needed for xb and xq thanks to std::vector!
   std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] -----   任务完成   -----\n";
   return 0;
}