#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <sys/stat.h>
#include <sys/time.h>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <set>
#include <thread>

#include "../faiss/Index.h"
#include "../faiss/IndexACORN.h"
#include "../faiss/IndexFlat.h"
#include "../faiss/impl/ACORN.h"
#include "../faiss/index_io.h"
#include "../faiss/impl/platform_macros.h"
#include "utils.cpp" 

// 辅助函数：获取文件大小（字节）
long get_file_size(const std::string& filename) {
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

// 存储单次查询的结果
struct QueryResult {
    int query_id;
    double acorn_time_ms;
    double acorn_qps;
    float acorn_recall;
    size_t acorn_n3;
    double acorn_1_time_ms;
    double acorn_1_qps;
    float acorn_1_recall;
    size_t acorn_1_n3;
    double filter_time_ms;
};

// 存储多次 repeat 聚合后的平均结果
struct AggregatedResult {
    double total_acorn_time_s = 0.0;
    float total_acorn_recall = 0.0;
    size_t total_acorn_n3 = 0;
    
    double total_acorn_1_time_s = 0.0;
    float total_acorn_1_recall = 0.0;
    size_t total_acorn_1_n3 = 0;
    
    double total_filter_time_s = 0.0;
};


int main(int argc, char* argv[]) {
    std::cout << "====================\nSTART: ACORN Test Suite --" << std::endl;
    double t0 = elapsed();

    // --- 参数定义 ---
    size_t d = 0; 
    int M, M_beta, gamma, k = 10;
    std::string dataset;
    size_t N = 0;
    int nthreads, repeat_num = 0;
    std::string base_path, base_label_path, query_path, csv_dir, avg_csv_dir, dis_output_path;
    std::vector<int> efs_list;
    bool if_bfs_filter = true;
    std::string mode, index_path_acorn, index_path_acorn1;

    // --- 参数解析 ---
    {
        if (argc < 20) {
            fprintf(stderr, "Usage: %s <mode> <N> <gamma> <dataset> <M> <M_beta> <base_path> <base_label_path> <query_path> <csv_dir> <avg_csv_dir> <dis_output_path> <nthreads> <repeat_num> <if_bfs_filter> <efs_list> <index_path_acorn> <index_path_acorn1><k> \n", argv[0]);
            fprintf(stderr, "       <mode> must be 'build' or 'search'\n");
            exit(1);
        }
        mode = argv[1];
        if (mode != "build" && mode != "search") {
            fprintf(stderr, "Invalid mode: %s. Must be 'build' or 'search'.\n", mode.c_str());
            exit(1);
        }
        N = strtoul(argv[2], NULL, 10);
        gamma = atoi(argv[3]);
        dataset = argv[4];
        M = atoi(argv[5]);
        M_beta = atoi(argv[6]);
        base_path = argv[7];
        base_label_path = argv[8];
        query_path = argv[9];
        csv_dir = argv[10];     
        avg_csv_dir = argv[11]; 
        dis_output_path = argv[12];
        nthreads = atoi(argv[13]);
        repeat_num = atoi(argv[14]);
        if_bfs_filter = atoi(argv[15]);
        char* efs_list_str = argv[16];
        index_path_acorn = argv[17];
        index_path_acorn1 = argv[18];
        k = atoi(argv[19]);

        printf("Running in '%s' mode\n", mode.c_str());
        printf("DEBUG: N=%zu, gamma=%d, M=%d, M_beta=%d, threads=%d, repeat=%d\n", N, gamma, M, M_beta, nthreads, repeat_num);

        char* token = strtok(efs_list_str, ",");
        while (token != NULL) {
            efs_list.push_back(atoi(token));
            token = strtok(NULL, ",");
        }
        std::sort(efs_list.begin(), efs_list.end());
    }
    
    // --- 加载元数据 ---
    std::vector<std::vector<int>> metadata = load_ab_muti(dataset, gamma, "rand", N, base_label_path);
    metadata.resize(N);
    assert(N == metadata.size());
    for (auto& inner_vector : metadata) {
        std::sort(inner_vector.begin(), inner_vector.end());
    }
    printf("[%.3f s] Loaded metadata, %ld vectors found\n", elapsed() - t0, metadata.size());

    // ==================== BUILD 模式 ====================
    if (mode == "build") {
      omp_set_num_threads(32);
      printf("Using 32 threads to build index\n");
      std::cout<<"Building ACORN index with parameters: " 
               << "N=" << N << ", gamma=" << gamma 
               << ", M=" << M << ", M_beta=" << M_beta 
               << ", dataset=" << dataset << std::endl;

      size_t nb, d2;
      float* xb = nullptr;
      {
          printf("[%.3f s] Loading database for building...\n", elapsed() - t0);
          std::string filename = base_path + "/" + dataset + "_base.fvecs";
          xb = fvecs_read(filename.c_str(), &d2, &nb);
          d = d2; 
          printf("[%.3f s] Base data loaded, dim: %zu, nb: %zu\n", elapsed() - t0, d, nb);
          if (N > nb) {
              fprintf(stderr, "Error: N=%zu is larger than database size nb=%zu\n", N, nb);
              exit(1);
          }
      }

      faiss::IndexACORNFlat hybrid_index(d, M, gamma, metadata, M_beta);
      faiss::IndexACORNFlat hybrid_index_gamma1(d, M, 1, metadata, M * 2);

      std::cout<<"Index parameters: " 
               << "d=" << d << ", M=" << M 
               << ", gamma=" << gamma << ", M_beta=" << M_beta 
               << std::endl;

      // 构建 ACORN
      printf("[%.3f s] Adding %zu vectors to ACORN index...\n", elapsed() - t0, N);
      double t_start_acorn = elapsed();
      hybrid_index.add(N, xb);
      double acorn_build_time_s = elapsed() - t_start_acorn;
      printf("[%.3f s] ACORN build time: %.3f s\n", elapsed() - t0, acorn_build_time_s);
      
      std::cout<<"ACORN index built. Now building ACORN-1 index with gamma=1 and M_beta=" << M * 2 << std::endl;

      // 构建 ACORN-1
      printf("[%.3f s] Adding %zu vectors to ACORN-1 index...\n", elapsed() - t0, N);
      double t_start_acorn1 = elapsed();
      hybrid_index_gamma1.add(N, xb);
      double acorn_1_build_time_s = elapsed() - t_start_acorn1;
      printf("[%.3f s] ACORN-1 build time: %.3f s\n", elapsed() - t0, acorn_1_build_time_s);

      delete[] xb;

      // 保存索引
      printf("[%.3f s] Saving ACORN index to %s\n", elapsed() - t0, index_path_acorn.c_str());
      faiss::write_index(&hybrid_index, index_path_acorn.c_str());
      
      printf("[%.3f s] Saving ACORN-1 index to %s\n", elapsed() - t0, index_path_acorn1.c_str());
      faiss::write_index(&hybrid_index_gamma1, index_path_acorn1.c_str());

      // --- ACORN 索引大小计算 ---
      long acorn_total_size_bytes = get_file_size(index_path_acorn);
      // 计算理论向量大小 (N * d * 4 bytes)
      long acorn_vectors_size_bytes = (long)N * d * sizeof(float);
      // 计算纯索引结构大小
      long acorn_index_only_size_bytes = acorn_total_size_bytes - acorn_vectors_size_bytes;

      // 保存元数据文件
      std::ofstream meta_file(index_path_acorn + ".meta");
      meta_file << "build_time_s:" << acorn_build_time_s << std::endl;
      meta_file << "total_size_bytes:" << acorn_total_size_bytes << std::endl;
      meta_file << "index_only_size_bytes:" << acorn_index_only_size_bytes << std::endl; // 新增行
      meta_file.close();
      
      printf("[%.3f s] ACORN index saved. Total size: %.2f MB. Vectors part: %.2f MB. Index-only part: %.2f MB.\n", 
             elapsed() - t0, 
             (double)acorn_total_size_bytes / (1024.0 * 1024.0),
             (double)acorn_vectors_size_bytes / (1024.0 * 1024.0),
             (double)acorn_index_only_size_bytes / (1024.0 * 1024.0));

      // --- ACORN-1 索引大小计算 ---
      long acorn_1_total_size_bytes = get_file_size(index_path_acorn1);
      long acorn_1_vectors_size_bytes = (long)N * d * sizeof(float);
      long acorn_1_index_only_size_bytes = acorn_1_total_size_bytes - acorn_1_vectors_size_bytes;
      
      // 保存元数据文件
      std::ofstream meta_file1(index_path_acorn1 + ".meta");
      meta_file1 << "build_time_s:" << acorn_1_build_time_s << std::endl;
      meta_file1 << "total_size_bytes:" << acorn_1_total_size_bytes << std::endl;
      meta_file1 << "index_only_size_bytes:" << acorn_1_index_only_size_bytes << std::endl; // 新增行
      meta_file1.close();

      printf("[%.3f s] ACORN-1 index saved. Total size: %.2f MB. Vectors part: %.2f MB. Index-only part: %.2f MB.\n",
             elapsed() - t0,
             (double)acorn_1_total_size_bytes / (1024.0 * 1024.0),
             (double)acorn_1_vectors_size_bytes / (1024.0 * 1024.0),
             (double)acorn_1_index_only_size_bytes / (1024.0 * 1024.0));

      printf("[%.3f s] Build finished successfully.\n", elapsed() - t0);
      return 0;
   }
    // ==================== SEARCH 模式 ====================
   else if (mode == "search") {
        omp_set_num_threads(nthreads);
        printf("Using %d threads for search\n", nthreads);
        std::cout<<"Searching ACORN index with parameters: " 
                 << "N=" << N << ", gamma=" << gamma 
                 << ", M=" << M << ", M_beta=" << M_beta 
                 << ", dataset=" << dataset 
                 <<", k= "<< k << std::endl;

        // --- 加载构建阶段的元数据 (build time, index size) ---
        double acorn_build_time_s = 0.0, acorn_1_build_time_s = 0.0;
        long acorn_index_size_bytes = 0, acorn_1_index_size_bytes = 0;
        
        std::ifstream meta_file(index_path_acorn + ".meta");
        std::string line;
        if (meta_file.is_open()) {
            while(std::getline(meta_file, line)) {
                std::stringstream ss(line);
                std::string key;
                std::getline(ss, key, ':');
                if (key == "build_time_s") ss >> acorn_build_time_s;
                if (key == "index_size_bytes") ss >> acorn_index_size_bytes;
            }
            meta_file.close();
        } else {
            printf("Warning: Could not open meta file for ACORN index: %s.meta\n", index_path_acorn.c_str());
        }

        std::ifstream meta_file1(index_path_acorn1 + ".meta");
        if (meta_file1.is_open()) {
            while(std::getline(meta_file1, line)) {
                std::stringstream ss(line);
                std::string key;
                std::getline(ss, key, ':');
                if (key == "build_time_s") ss >> acorn_1_build_time_s;
                if (key == "index_size_bytes") ss >> acorn_1_index_size_bytes;
            }
            meta_file1.close();
        } else {
             printf("Warning: Could not open meta file for ACORN-1 index: %s.meta\n", index_path_acorn1.c_str());
        }

        // --- 加载索引 ---
        faiss::IndexACORNFlat* hybrid_index = nullptr;
        faiss::IndexACORNFlat* hybrid_index_gamma1 = nullptr;

        printf("[%.3f s] Loading ACORN index from: %s\n", elapsed() - t0, index_path_acorn.c_str());
        hybrid_index = dynamic_cast<faiss::IndexACORNFlat*>(faiss::read_index(index_path_acorn.c_str()));
        if (!hybrid_index) { fprintf(stderr, "Error: Failed to load index from %s.\n", index_path_acorn.c_str()); exit(1); }
        d = hybrid_index->d;

        printf("[%.3f s] Loading ACORN-1 index from: %s\n", elapsed() - t0, index_path_acorn1.c_str());
        hybrid_index_gamma1 = dynamic_cast<faiss::IndexACORNFlat*>(faiss::read_index(index_path_acorn1.c_str()));
        if (!hybrid_index_gamma1) { fprintf(stderr, "Error: Failed to load index from %s.\n", index_path_acorn1.c_str()); exit(1); }

        // 重新关联元数据
        printf("[%.3f s] Re-associating metadata with loaded indexes...\n", elapsed() - t0);
        hybrid_index->set_metadata(metadata);
        hybrid_index_gamma1->set_metadata(metadata);
        printf("[%.3f s] Indexes loaded and ready for search.\n", elapsed() - t0);

        // --- 加载查询数据 ---
        size_t nq;
        float* xq;
        std::vector<std::vector<int>> aq;
        {
            printf("[%.3f s] Loading query vectors and attributes\n", elapsed() - t0);
            size_t d2;
            std::string filename = query_path + "/" + dataset + "_query.fvecs";
            xq = fvecs_read(filename.c_str(), &d2, &nq);
            assert(d == d2 || !"Query dimension mismatch!");
            std::string last_value = query_path.substr(query_path.find_last_of('_') + 1);
            aq = load_aq_multi(dataset, gamma, 0, N, query_path);
            #pragma omp parallel for
            for (size_t i = 0; i < aq.size(); ++i) {
                std::sort(aq[i].begin(), aq[i].end());
            }
            std::cout<<"Query data loaded. nq=" << nq << std::endl;
            printf("[%.3f s] Loaded %zu queries\n", elapsed() - t0, nq);
        }

        // --- 准备结果存储 ---
        int efs_cnt = efs_list.size();
        std::vector<std::vector<std::vector<QueryResult>>> all_query_results(repeat_num, std::vector<std::vector<QueryResult>>(efs_cnt, std::vector<QueryResult>(nq)));
        for (int r = 0; r < repeat_num; ++r) for (int e = 0; e < efs_cnt; ++e) for (int i = 0; i < nq; ++i) all_query_results[r][e][i].query_id = i;
        
        std::vector<AggregatedResult> final_avg_results(efs_cnt);

        // --- 生成 Ground Truth 和 Filter Map ---
        std::string last_value = query_path.substr(query_path.find_last_of('_') + 1);
        std::string MY_DIS_SORT_DIR = dis_output_path + "/" + dataset + "_query_" + last_value + "/my_dis_sort";

         // 串行计算 Filter Map
         double t_filter_0 = elapsed();
         std::vector<char> filter_ids_map(nq * N);
         for (int xq_idx = 0; xq_idx < nq; xq_idx++) {
            for (int xb_idx = 0; xb_idx < N; xb_idx++) {
               const auto& query_attrs = aq[xq_idx];
               const auto& data_attrs = metadata[xb_idx];
               bool is_subset = std::includes(data_attrs.begin(), data_attrs.end(), query_attrs.begin(), query_attrs.end());
               filter_ids_map[xq_idx * N + xb_idx] = is_subset;
            }
         }
         double filter_time_s = elapsed() - t_filter_0;
         printf("[%.3f s] Filter map created in %.4f s\n", elapsed() - t0, filter_time_s);

        // 并行计算 Filter Map
        double t_filter_para_0 = elapsed();
        std::vector<char> filter_ids_map_para(nq * N);
        #pragma omp parallel for
        for (int xq_idx = 0; xq_idx < nq; xq_idx++) {
            for (int xb_idx = 0; xb_idx < N; xb_idx++) {
                const auto& query_attrs = aq[xq_idx];
                const auto& data_attrs = metadata[xb_idx];
                bool is_subset = std::includes(data_attrs.begin(), data_attrs.end(), query_attrs.begin(), query_attrs.end());
                filter_ids_map_para[xq_idx * N + xb_idx] = is_subset;
            }
        }
        double filter_para_time_s = elapsed() - t_filter_para_0;
        printf("[%.3f s] Filter map created in %.4f s\n", elapsed() - t0, filter_para_time_s);

         // Ground Truth生成
         {
            // 检查基准文件是否已存在，如果不存在则生成
            std::string ground_truth_check_file = MY_DIS_SORT_DIR + "/filter_sorted_dist_0.txt";
            std::ifstream f(ground_truth_check_file.c_str());

            if (!f.good()) {
                  printf("\n[%.3f s] Ground truth not found. Generating now... (This may take a while)\n", elapsed() - t0);

                  // 为保存文件创建目录，避免写入失败，需要创建两级目录
                  std::string parent_dir = dis_output_path + "/" + dataset + "_query_" + last_value;
                  mkdir(parent_dir.c_str(), 0777);
                  mkdir(MY_DIS_SORT_DIR.c_str(), 0777);
                  
                  // 1. 分配内存
                  float* all_distances = new float[nq * N];

                  // 2. 调用函数计算所有距离
                  printf("[%.3f s] Calculating all distances using hybrid_index->calculate_distances...\n", elapsed() - t0);
                  hybrid_index->calculate_distances(nq, xq, k, all_distances, nullptr);
                  printf("[%.3f s] All distances calculated.\n", elapsed() - t0);

                  // 3. 筛选并排序结果
                  printf("[%.3f s] Sorting and filtering distances...\n", elapsed() - t0);
                  auto sorted_results = get_sorted_filtered_distances(all_distances, filter_ids_map_para, nq, N);
                  
                  // 4. 保存排序后的基准文件 
                  printf("[%.3f s] Saving sorted ground truth to %s...\n", elapsed() - t0, MY_DIS_SORT_DIR.c_str());
                  save_sorted_filtered_distances_to_txt(sorted_results, MY_DIS_SORT_DIR, "filter_sorted_dist_");
                  
                  // 5. 清理内存
                  delete[] all_distances;
                  printf("[%.3f s] Ground truth saved.\n\n", elapsed() - t0);

            } else {
                  printf("\n[%.3f s] Ground truth found at %s. Skipping generation.\n\n", elapsed() - t0, MY_DIS_SORT_DIR.c_str());
                  f.close();
            }
         }
         std::cout<<"Search will use ground truth from: " << MY_DIS_SORT_DIR << std::endl;

        // --- 多次重复执行搜索 ---
        for (int repeat = 0; repeat < repeat_num; repeat++) {
            std::cout<<"=============== Repeat "<< repeat + 1<<"/"<< repeat_num <<" ==============="<< std::endl;
            for (int efs_id = 0; efs_id < efs_cnt; efs_id++) {
                int current_efs = efs_list[efs_id];
                
                // --- 搜索 ACORN ---
                std::vector<faiss::idx_t> nns2(k * nq);
                std::vector<float> dis2(k * nq);
                std::vector<double> query_times(nq);
                std::vector<double> query_qps(nq);
                std::vector<size_t> query_n3(nq);
                
                std::cout<<"--- ACORN | efs = "<< current_efs <<" ---"<< std::endl;
                hybrid_index->acorn.efSearch = current_efs;
                faiss::acorn_stats.reset();
                double t1_acorn = elapsed();
                hybrid_index->search(nq, xq, k, dis2.data(), nns2.data(), filter_ids_map_para.data(), &query_times, &query_qps, &query_n3, if_bfs_filter);
                double search_time_acorn_s = elapsed() - t1_acorn;
                const faiss::ACORNStats& acorn_search_stats = faiss::acorn_stats;

                // --- 搜索 ACORN-1 ---
                std::vector<faiss::idx_t> nns3(k * nq);
                std::vector<float> dis3(k * nq);
                std::vector<double> query_times3(nq);
                std::vector<double> query_qps3(nq);
                std::vector<size_t> query_n33(nq);

                printf("--- ACORN-1 | efs = %d ---\n", current_efs);
                hybrid_index_gamma1->acorn.efSearch = current_efs;
                faiss::acorn_stats.reset();
                double t1_acorn1 = elapsed();
                hybrid_index_gamma1->search(nq, xq, k, dis3.data(), nns3.data(), filter_ids_map_para.data(), &query_times3, &query_qps3, &query_n33, if_bfs_filter);
                double search_time_acorn1_s = elapsed() - t1_acorn1;
                const faiss::ACORNStats& acorn1_search_stats = faiss::acorn_stats;
                
                // --- 计算 Recall ---
                auto sorted_results = read_all_sorted_filtered_distances_from_txt(std::string(MY_DIS_SORT_DIR), nq, N);
                auto recalls_acorn = compute_recall(nns2, sorted_results, nq, k);
                float recall_mean_acorn = std::accumulate(recalls_acorn.begin(), recalls_acorn.end(), 0.0f) / nq;
                auto recalls_acorn1 = compute_recall(nns3, sorted_results, nq, k);
                float recall_mean_acorn1 = std::accumulate(recalls_acorn1.begin(), recalls_acorn1.end(), 0.0f) / nq;

                std::cout<< "  ACORN   | Time: "<< search_time_acorn_s <<" s, QPS: "<< nq / search_time_acorn_s <<", Recall: "<< recall_mean_acorn << std::endl;
                std::cout<< "  ACORN-1 | Time: "<< search_time_acorn1_s <<" s, QPS: "<< nq / search_time_acorn1_s <<", Recall: "<< recall_mean_acorn1 << std::endl;

                // --- 存储单次结果 ---
                for (int i = 0; i < nq; i++) {
                    all_query_results[repeat][efs_id][i].acorn_time_ms = query_times[i] * 1000.0;
                    all_query_results[repeat][efs_id][i].acorn_qps = query_qps[i];
                    all_query_results[repeat][efs_id][i].acorn_recall = recalls_acorn[i];
                    all_query_results[repeat][efs_id][i].acorn_n3 = query_n3[i];
                    all_query_results[repeat][efs_id][i].acorn_1_time_ms = query_times3[i] * 1000.0;
                    all_query_results[repeat][efs_id][i].acorn_1_qps = query_qps3[i];
                    all_query_results[repeat][efs_id][i].acorn_1_recall = recalls_acorn1[i];
                    all_query_results[repeat][efs_id][i].acorn_1_n3 = query_n33[i];
                    //all_query_results[repeat][efs_id][i].filter_time_ms = filter_time_s * 1000.0;
                }
                
                // --- 累加结果用于最终平均 ---
                final_avg_results[efs_id].total_acorn_time_s += search_time_acorn_s;
                final_avg_results[efs_id].total_acorn_recall += recall_mean_acorn;
                final_avg_results[efs_id].total_acorn_n3 += (acorn_search_stats.n1 > 0) ? ((size_t)acorn_search_stats.n3) : 0;
                
                final_avg_results[efs_id].total_acorn_1_time_s += search_time_acorn1_s;
                final_avg_results[efs_id].total_acorn_1_recall += recall_mean_acorn1;
                final_avg_results[efs_id].total_acorn_1_n3 += (acorn1_search_stats.n1 > 0) ? ((size_t)acorn1_search_stats.n3) : 0;
                
                //final_avg_results[efs_id].total_filter_time_s += filter_time_s;

               // // 检查召回率是否都已达到 1.0
               // if (recall_mean_acorn >= 0.99f && recall_mean_acorn1 >= 0.99f) {
               //    std::cout << "--- Recall@ " << k << " for both ACORN and ACORN-1 reached 0.99 at efs=" 
               //             << current_efs << ". Skipping larger efs values for this repeat. ---" << std::endl;
               //    break;
               // }
            }
         }

        // --- 生成动态文件名 ---
        std::string query_num_str = query_path.substr(query_path.find_last_of('_') + 1);
        std::stringstream efs_str_stream;
        efs_str_stream << efs_list.front() << "-" << efs_list.back();
        if (efs_list.size() > 1) {
             efs_str_stream << "_" << (efs_list[1] - efs_list[0]);
        }

        std::stringstream file_suffix_stream;
        file_suffix_stream << dataset << "_query" << query_num_str
                   << "_M" << M << "_gamma" << gamma
                   << "_threads" << nthreads << "_repeat" << repeat_num
                   << "_ifbfs" << if_bfs_filter << "_efs" << efs_str_stream.str()
                   << ".csv";

        std::string output_filename = file_suffix_stream.str();
        std::string full_csv_path = csv_dir + "/" + output_filename;
        std::string full_avg_csv_path = avg_csv_dir + "/" + "avg_" + output_filename;
        
        printf("\n[%.3f s] Writing detailed results to: %s\n", elapsed() - t0, full_csv_path.c_str());
        printf("[%.3f s] Writing average results to: %s\n", elapsed() - t0, full_avg_csv_path.c_str());

        // --- 写详细CSV文件 (per query) ---
        std::ofstream csv_file(full_csv_path);
        csv_file << "repeat,efs,QueryID,acorn_Time_ms,acorn_QPS,acorn_Recall,acorn_n3_visited,"
                 << "acorn_build_time_ms,acorn_index_size_MB,"
                 << "acorn_1_Time_ms,acorn_1_QPS,acorn_1_Recall,acorn_1_n3_visited,"
                 << "acorn_1_build_time_ms,acorn_1_index_size_MB\n";
        for (int repeat = 0; repeat < repeat_num; repeat++) {
            for (int efs_id = 0; efs_id < efs_list.size(); efs_id++) {
                for (const auto &result : all_query_results[repeat][efs_id]) {
                    csv_file << repeat << ","
                             << efs_list[efs_id] << ","
                             << result.query_id << ","
                             << result.acorn_time_ms << "," << result.acorn_qps << ","
                             << result.acorn_recall << "," << result.acorn_n3 << ","
                             << acorn_build_time_s * 1000.0 << ","
                             << (double)acorn_index_size_bytes / (1024.0 * 1024.0) << ","
                             << result.acorn_1_time_ms << "," << result.acorn_1_qps << ","
                             << result.acorn_1_recall << "," << result.acorn_1_n3 << ","
                             << acorn_1_build_time_s * 1000.0 << ","
                             << (double)acorn_1_index_size_bytes / (1024.0 * 1024.0) << "\n";
                }
            }
        }
        csv_file.close();

        // --- 写平均CSV文件 (per efs) ---
        std::ofstream avg_csv_file(full_avg_csv_path);
        avg_csv_file << "efs,acorn_Time_ms,acorn_QPS,acorn_Recall,acorn_n3_visited_avg,"
                     << "acorn_build_time_ms,acorn_index_size_MB,"
                     << "acorn_1_Time_ms,acorn_1_QPS,acorn_1_Recall,acorn_1_n3_visited_avg,"
                     << "acorn_1_build_time_ms,acorn_1_index_size_MB,FilterMapTime_ms,FilterMapTime_para_ms\n";
        for (int efs_id = 0; efs_id < efs_list.size(); efs_id++) {
            const auto& aggregated = final_avg_results[efs_id];
            avg_csv_file << efs_list[efs_id] << ","
                         << (aggregated.total_acorn_time_s * 1000.0) / repeat_num << ","
                         << (nq * repeat_num) / aggregated.total_acorn_time_s << ","
                         << aggregated.total_acorn_recall / repeat_num << ","
                         << (double)aggregated.total_acorn_n3 / (nq * repeat_num) << ","
                         << acorn_build_time_s * 1000.0 << ","
                         << (double)acorn_index_size_bytes / (1024.0 * 1024.0) << ","
                         << (aggregated.total_acorn_1_time_s * 1000.0) / repeat_num << ","
                         << (nq * repeat_num) / aggregated.total_acorn_1_time_s << ","
                         << aggregated.total_acorn_1_recall / repeat_num << ","
                         << (double)aggregated.total_acorn_1_n3 / (nq * repeat_num) << ","
                         << acorn_1_build_time_s * 1000.0 << ","
                         << (double)acorn_1_index_size_bytes / (1024.0 * 1024.0) << ","
                         << filter_time_s * 1000.0 << ","
                         << filter_para_time_s * 1000.0 << "\n";
        }
        avg_csv_file.close();

        delete[] xq;
        delete hybrid_index;
        delete hybrid_index_gamma1;

        printf("\n[%.3f s] -----SEARCH DONE-----\n", elapsed() - t0);
        return 0;
   }

   return 0; 
}
