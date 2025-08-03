#include <set>
#include "utils.h"

namespace ANNS
{

   void write_kv_file(const std::string &filename, const std::map<std::string, std::string> &kv_map)
   {
      std::ofstream out(filename);
      for (auto &kv : kv_map)
      {
         out << kv.first << "=" << kv.second << std::endl;
      }
      out.close();
   }

   std::map<std::string, std::string> parse_kv_file(const std::string &filename)
   {
      std::map<std::string, std::string> kv_map;
      std::ifstream in(filename);
      std::string line;
      while (std::getline(in, line))
      {
         size_t pos = line.find("=");
         if (pos == std::string::npos)
            continue;
         std::string key = line.substr(0, pos);
         std::string value = line.substr(pos + 1);
         kv_map[key] = value;
      }
      in.close();
      return kv_map;
   }

   void write_gt_file(const std::string &filename, const std::pair<IdxType, float> *gt, uint32_t num_queries, uint32_t K)
   {
      std::ofstream fout(filename, std::ios::binary);
      fout.write(reinterpret_cast<const char *>(gt), num_queries * K * sizeof(std::pair<IdxType, float>));
      std::cout << "Ground truth written to " << filename << std::endl;
   }

   void load_gt_file(const std::string &filename, std::pair<IdxType, float> *gt, uint32_t num_queries, uint32_t K)
   {
      std::ifstream fin(filename, std::ios::binary);
      fin.read(reinterpret_cast<char *>(gt), num_queries * K * sizeof(std::pair<IdxType, float>));
      std::cout << "Ground truth loaded from " << filename << std::endl;
   }

   /*float calculate_recall(const std::pair<IdxType, float> *gt, const std::pair<IdxType, float> *results, uint32_t num_queries, uint32_t K)
   {
      float total_correct = 0;
      for (uint32_t i = 0; i < num_queries; i++)
      {

         // prepare ground truth set, offset records the last valid gt index
         std::set<IdxType> gt_set;
         int32_t offset = -1;
         for (uint32_t j = 0; j < K; j++)
            if (gt[i * K + j].first != -1)
            {
               offset = j;
               gt_set.insert(gt[i * K + j].first);
            }

         // count the correct
         for (uint32_t j = 0; j < K; j++)
         {
            if (results[i * K + j].first == -1)
               break;
            if (offset >= 0 && results[i * K + j].second == gt[i * K + offset].second)
            { // for ties
               total_correct++;
               offset--;
            }
            else
            {
               if (gt_set.find(results[i * K + j].first) != gt_set.end())
                  total_correct++;
            }
         }
      }
      return 100.0 * total_correct / (num_queries * K);
   }*/
   // fxy_add
   float calculate_recall(const std::pair<IdxType, float> *gt, const std::pair<IdxType, float> *results, uint32_t num_queries, uint32_t K)
   {
      float total_correct = 0;
      float total_relevant = 0; // 新增：统计所有查询的真实相关结果总数

      for (uint32_t i = 0; i < num_queries; i++)
      {
         // 构建 ground truth 集合，并计算当前查询的真实相关数
         std::set<IdxType> gt_set;
         int32_t offset = -1;
         uint32_t num_relevant = 0; // 当前查询的真实相关数

         for (uint32_t j = 0; j < K; j++)
         {
            if (gt[i * K + j].first != -1)
            {
               offset = j;
               gt_set.insert(gt[i * K + j].first);
               num_relevant++; // 统计有效 GT
            }
         }

         total_relevant += num_relevant; // 累加到总真实相关数

         // 统计正确匹配数
         for (uint32_t j = 0; j < K; j++)
         {
            if (results[i * K + j].first == -1)
               break;

            if (offset >= 0 && results[i * K + j].second == gt[i * K + offset].second)
            { // 并列情况
               total_correct++;
               offset--;
            }
            else
            {
               if (gt_set.find(results[i * K + j].first) != gt_set.end())
                  total_correct++;
            }
         }
      }

      // 召回率 = 正确匹配数 / 真实相关总数
      return (total_relevant > 0) ? (100.0f * total_correct / total_relevant) : 0.0f;
   }

   // fxy_add
   float calculate_recall_to_csv(const std::pair<IdxType, float> *gt,
                                 const std::pair<IdxType, float> *results,
                                 uint32_t num_queries,
                                 uint32_t K,
                                 const std::string &output_file)
   {
      float total_correct = 0;
      std::vector<float> query_recalls(num_queries, 0); // 存储每个查询的召回率

      std::ofstream file(output_file);
      if (!file.is_open())
      {
         std::cerr << "Failed to open file: " << output_file << std::endl;
         return -1; // 文件打开失败，返回错误值
      }

      file << "Query ID,Recall (%)\n"; // 写入 CSV 头部

      for (uint32_t i = 0; i < num_queries; i++)
      {
         std::set<IdxType> gt_set;
         int32_t offset = -1;
         float correct_count = 0;

         for (uint32_t j = 0; j < K; j++)
         {
            if (gt[i * K + j].first != -1)
            {
               offset = j;
               gt_set.insert(gt[i * K + j].first);
            }
         }

         for (uint32_t j = 0; j < K; j++)
         {
            if (results[i * K + j].first == -1)
               break;
            if (offset >= 0 && results[i * K + j].second == gt[i * K + offset].second)
            { // 处理 cost 相等的情况
               correct_count++;
               offset--;
            }
            else
            {
               if (gt_set.find(results[i * K + j].first) != gt_set.end())
                  correct_count++;
            }
         }

         query_recalls[i] = correct_count / K; // 计算当前查询的 recall
         total_correct += correct_count;

         file << i << "," << query_recalls[i] << "\n"; // 写入文件
      }

      file.close();

      return 100.0 * total_correct / (num_queries * K); // 返回整体召回率
   }

   // fxy_add
   void save_roaring_vector(const std::string &filename, const std::vector<roaring::Roaring> &rb_vec)
   {
      std::ofstream out(filename, std::ios::binary);

      // 写入 vector 大小
      uint64_t size = rb_vec.size();
      out.write(reinterpret_cast<const char *>(&size), sizeof(size));

      for (const auto &rb : rb_vec)
      {
         // 获取序列化大小
         size_t serialized_size = rb.getSizeInBytes();
         char *buffer = new char[serialized_size];
         rb.write(buffer); // 写入 buffer

         // 写入大小 + 数据
         out.write(reinterpret_cast<const char *>(&serialized_size), sizeof(serialized_size));
         out.write(buffer, serialized_size);

         delete[] buffer;
      }

      out.close();
      std::cout << "Saved roaring vector to " << filename << ", size = " << rb_vec.size() << std::endl;
   }

   void load_roaring_vector(const std::string &filename, std::vector<roaring::Roaring> &rb_vec)
   {
      std::ifstream in(filename, std::ios::binary);
      if (!in)
      {
         std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
         return;
      }

      // 读取 vector 大小
      uint64_t size;
      in.read(reinterpret_cast<char *>(&size), sizeof(size));
      rb_vec.resize(size);

      for (size_t i = 0; i < size; ++i)
      {
         // 读取每个 roaring bitmap 的大小
         size_t serialized_size;
         in.read(reinterpret_cast<char *>(&serialized_size), sizeof(serialized_size));

         // 分配缓冲区并读取数据
         char *buffer = new char[serialized_size];
         in.read(buffer, serialized_size);

         // 构造 roaring bitmap
         rb_vec[i] = roaring::Roaring::readSafe(buffer, serialized_size);

         delete[] buffer;
      }

      in.close();
      std::cout << "Loaded roaring vector from " << filename << ", size = " << rb_vec.size() << std::endl;
   }

   // fxy_add 辅助函数：将向量数据写入 .fvecs 文件
   void write_fvecs(const std::string &filename, const std::vector<float *> &vecs, size_t dim)
   {
      std::ofstream out(filename, std::ios::binary);
      if (!out)
      {
         throw std::runtime_error("Cannot open file for writing: " + filename);
      }
      for (const auto &vec : vecs)
      {
         out.write(reinterpret_cast<const char *>(&dim), sizeof(uint32_t));
         out.write(reinterpret_cast<const char *>(vec), dim * sizeof(float));
      }
   }

   // fxy_add辅助函数：将标签集写入 .txt 文件。格式: label1,label2,label3\n
   void write_labels_txt(const std::string &filename, const std::vector<std::vector<ANNS::LabelType>> &labels)
   {
      std::ofstream out(filename);
      if (!out)
      {
         throw std::runtime_error("Cannot open file for writing: " + filename);
      }
      for (const auto &label_set : labels)
      {
         for (size_t i = 0; i < label_set.size(); ++i)
         {
            out << label_set[i] << (i == label_set.size() - 1 ? "" : ",");
         }
         out << "\n";
      }
   }

}