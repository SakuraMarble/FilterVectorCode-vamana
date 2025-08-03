#ifndef UTILS_H
#define UTILS_H

#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "config.h"
#include <unordered_set>
#include <roaring/roaring.h>
#include <roaring/roaring.hh>

#define SEP_LINE "------------------------------------------------------------\n"

// likely and unlikely prediction
#define LIKELY(x) __builtin_expect(x, 1)
#define UNLIKELY(x) __builtin_expect(x, 0)

namespace ANNS
{
   // write and load key-value file
   void write_kv_file(const std::string &filename, const std::map<std::string, std::string> &kv_map);
   std::map<std::string, std::string> parse_kv_file(const std::string &filename);

   // write and load groundtruth file
   void write_gt_file(const std::string &filename, const std::pair<IdxType, float> *gt, uint32_t num_queries, uint32_t K);
   void load_gt_file(const std::string &filename, std::pair<IdxType, float> *gt, uint32_t num_queries, uint32_t K);

   // calculated recall
   float calculate_recall(const std::pair<IdxType, float> *gt, const std::pair<IdxType, float> *res, uint32_t num_queries, uint32_t K);
   float calculate_recall_to_csv(const std::pair<IdxType, float> *gt,
                                 const std::pair<IdxType, float> *results,
                                 uint32_t num_queries,
                                 uint32_t K,
                                 const std::string &output_file);

   // write 1个字符
   template <typename T>
   void write_one_T(const std::string &filename, const T &value)
   {
      std::ofstream out(filename);
      out << value << std::endl;
   }
   // load 1个字符
   template <typename T>
   void load_one_T(const std::string &filename, T &value)
   {
      std::ifstream in(filename);
      in >> value;
   }

   // write 1D-std::vector
   template <typename T>
   void write_1d_vector(const std::string &filename, const std::vector<T> &vec)
   {
      std::ofstream out(filename);
      for (auto &idx : vec)
         out << idx << std::endl;
   }

   // load 1D-std::vector
   template <typename T>
   void load_1d_vector(const std::string &filename, std::vector<T> &vec)
   {
      std::ifstream in(filename);
      T value;
      vec.clear();
      while (in >> value)
         vec.push_back(value);
   }

   // write 1D-std::vector , for std::pair
   template <typename T1, typename T2>
   void write_1d_pair_vector(const std::string &filename, const std::vector<std::pair<T1, T2>> &vec)
   {
      std::ofstream out(filename);
      for (auto &each : vec)
         out << each.first << " " << each.second << std::endl;
   }

   // load 1D-std::vector , for std::pair
   template <typename T1, typename T2>
   void load_1d_pair_vector(const std::string &filename, std::vector<std::pair<T1, T2>> &vec)
   {
      std::ifstream in(filename);
      std::string line;
      vec.clear();
      while (std::getline(in, line))
      {
         std::istringstream iss(line);
         T1 first;
         T2 second;
         iss >> first >> second;
         vec.push_back(std::make_pair(first, second));
      }
   }

   // write 2D-std::vector
   template <typename T>
   void write_2d_vectors(const std::string &filename, const std::vector<std::vector<T>> &vecs)
   {
      std::ofstream out(filename);
      for (auto &vec : vecs)
      {
         for (auto &idx : vec)
            out << idx << " ";
         out << std::endl;
      }
   }

   // load 2D-std::vector
   template <typename T>
   void load_2d_vectors(const std::string &filename, std::vector<std::vector<T>> &vecs)
   {
      std::ifstream in(filename);
      std::string line;
      vecs.clear();
      while (std::getline(in, line))
      {
         std::istringstream iss(line);
         std::vector<T> vec;
         T value;
         while (iss >> value)
            vec.push_back(value);
         vecs.push_back(vec);
      }
   }

   // write 2D-std::vector
   template <typename T>
   void write_2d_vectors(const std::string &filename, const std::vector<std::pair<T, T>> &vecs)
   {
      std::ofstream out(filename);
      for (auto &each : vecs)
         out << each.first << " " << each.second << std::endl;
   }

   // load 2D-std::vector
   template <typename T>
   void load_2d_vectors(const std::string &filename, std::vector<std::pair<T, T>> &vecs)
   {
      std::ifstream in(filename);
      std::string line;
      vecs.clear();
      while (std::getline(in, line))
      {
         std::istringstream iss(line);
         T first, second;
         iss >> first >> second;
         vecs.push_back(std::make_pair(first, second));
      }
   }

   // write 2D-std::vector , for std::unordered_set<IdxType>
   template <typename T>
   void write_2d_vectors(const std::string &filename, const std::vector<std::unordered_set<T>> &vecs)
   {
      std::ofstream out(filename);
      for (const auto &set : vecs)
      {
         for (const auto &elem : set)
            out << elem << " ";
         out << std::endl;
      }
   }

   // load 2D-std::vector , for std::unordered_set<IdxType>
   template <typename T>
   void load_2d_vectors(const std::string &filename, std::vector<std::unordered_set<T>> &vecs)
   {
      std::ifstream in(filename);
      std::string line;
      vecs.clear();
      while (std::getline(in, line))
      {
         std::istringstream iss(line);
         std::unordered_set<T> set;
         T value;
         while (iss >> value)
            set.insert(value);
         vecs.push_back(set);
      }
   }

   // 将 boost::dynamic_bitset 的 vector 写入文件
   template <typename BitsetType>
   void write_bitset_vector(const std::string &filename, const std::vector<BitsetType> &bitset_vec)
   {
      std::ofstream out(filename);
      if (!out.is_open())
      {
         std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
         return;
      }

      for (const auto &bs : bitset_vec)
      {
         out << bs << "\n"; // 使用 operator<< 输出
      }

      out.close();
      std::cout << "Saved bitset vector to " << filename << ", size = " << bitset_vec.size() << std::endl;
   }

   // 从文件中加载 boost::dynamic_bitset 的 vector
   template <typename BitsetType>
   void load_bitset_vector(const std::string &filename, std::vector<BitsetType> &bitset_vec)
   {
      std::ifstream in(filename);
      if (!in.is_open())
      {
         std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
         return;
      }

      bitset_vec.clear();
      std::string line;
      while (std::getline(in, line))
      {
         if (line.empty())
            continue;
         bitset_vec.emplace_back(line); // 从字符串构造 dynamic_bitset
      }

      in.close();
      std::cout << "Loaded bitset vector from " << filename << ", size = " << bitset_vec.size() << std::endl;
   }
   void save_roaring_vector(const std::string &filename, const std::vector<roaring::Roaring> &rb_vec);
   void load_roaring_vector(const std::string &filename, std::vector<roaring::Roaring> &rb_vec);
   void write_fvecs(const std::string &filename, const std::vector<float *> &vecs, size_t dim);
   void write_labels_txt(const std::string &filename, const std::vector<std::vector<ANNS::LabelType>> &labels);
}

#endif // UTILS_H