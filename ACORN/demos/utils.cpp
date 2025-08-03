#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include <sys/time.h>

#include <faiss/IndexACORN.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// added these
#include <arpa/inet.h>
#include <assert.h> /* assert */
#include <faiss/Index.h>
#include <faiss/impl/platform_macros.h>
#include <math.h>
#include <nlohmann/json.hpp>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
#include <algorithm>  // for std::sort
#include <cmath>      // for std::mean and std::stdev
#include <filesystem> // C++17 文件系统库
#include <fstream>
#include <iosfwd>
#include <iostream>
#include <numeric> // for std::accumulate
#include <set>
#include <sstream> // for ostringstream
#include <stdexcept>
#include <string>
#include <thread>
#include <utility> // for std::pair
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;
/**
 * To run this demo, please download the ANN_SIFT1M dataset from
 *
 *   http://corpus-texmex.irisa.fr/
        -> wget -r ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
        -> cd ftp.irisa.fr/local/texmex/corpus
        -> tar -xf sift.tar.gz

 * and unzip it to the sudirectory sift1M.
 **/

#include <zlib.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

// using namespace std;

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/

bool fileExists(const std::string &filePath)
{
   std::ifstream file(filePath);
   return file.good();
}

float *fvecs_read(const char *fname, size_t *d_out, size_t *n_out)
{
   FILE *f = fopen(fname, "r");
   if (!f)
   {
      fprintf(stderr, "could not open %s\n", fname);
      perror("");
      abort();
   }
   int d;
   fread(&d, 1, sizeof(int), f);
   assert((d > 0 && d < 1000000) || !"unreasonable dimension");
   fseek(f, 0, SEEK_SET);
   struct stat st;
   fstat(fileno(f), &st);
   size_t sz = st.st_size;
   assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
   size_t n = sz / ((d + 1) * 4);

   *d_out = d;
   *n_out = n;
   float *x = new float[n * (d + 1)];
   size_t nr = fread(x, sizeof(float), n * (d + 1), f);
   assert(nr == n * (d + 1) || !"could not read whole file");

   // shift array to remove row headers
   for (size_t i = 0; i < n; i++)
      memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

   fclose(f);
   return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int *ivecs_read(const char *fname, size_t *d_out, size_t *n_out)
{
   return (int *)fvecs_read(fname, d_out, n_out);
}

// get file name to load data vectors from
std::string get_file_name(std::string dataset, bool is_base, std::string BASE_DIR)
{
   if (dataset == "sift1M" || dataset == "sift1M_test")
   {
      return BASE_DIR + std::string("/datasets/sift_") +
             (is_base ? "base" : "query") + ".fvecs";
   }
   else if (dataset == "sift1B")
   {
      return BASE_DIR + std::string("/datasets/bigann_") +
             (is_base ? "base_10m" : "query") + ".fvecs";
   }
   else if (dataset == "tripclick")
   {
      return BASE_DIR + std::string("/datasets/") +
             (is_base ? "base_vecs_tripclick"
                      : "query_vecs_tripclick_min100") +
             ".fvecs";
   }
   else if (dataset == "paper" || dataset == "paper_rand2m")
   {
      return BASE_DIR + std::string("/datasets/") +
             (is_base ? "paper_base" : "paper_query") + ".fvecs";
   }
   else if (dataset == "words")
   {
      return BASE_DIR + "/" + (is_base ? "words_base" : "words_query") + ".fvecs";
   }
   else if (dataset == "MTG")
   {
      return BASE_DIR + "/" + (is_base ? "MTG_base" : "MTG_query") + ".fvecs";
   }
   else if (dataset == "arxiv")
   {
      return BASE_DIR + "/" + (is_base ? "arxiv_base" : "arxiv_query") + ".fvecs";
   }
   else if (dataset == "captcha")
   {
      return BASE_DIR + "/" + (is_base ? "captcha_base" : "captcha_query") + ".fvecs";
   }
   else if (dataset == "russian")
   {
      return BASE_DIR + "/" + (is_base ? "russian_base" : "russian_query") + ".fvecs";
   }
   else if (dataset == "bookimg")
   {
      return BASE_DIR + "/" + (is_base ? "bookimg_base" : "bookimg_query") + ".fvecs";
   }
   else if (dataset == "TimeTravel")
   {
      return BASE_DIR + "/" + (is_base ? "TimeTravel_base" : "TimeTravel_query") + ".fvecs";
   }
   else if (dataset == "app_reviews")
   {
      return BASE_DIR + "/" + (is_base ? "app_reviews_base" : "app_reviews_query") + ".fvecs";
   }
   else if (dataset == "amazing_file")
   {
      return BASE_DIR + "/" + (is_base ? "amazing_file" : "amazing_file") + ".fvecs";
   }
   else if (dataset == "celeba")
   {
      return BASE_DIR + "/" + (is_base ? "celeba" : "celeba") + ".fvecs";
   }
   else if (dataset == "VariousTaggedImages")
   {
      return BASE_DIR + "/" + (is_base ? "VariousTaggedImages" : "VariousTaggedImages") + ".fvecs";
   }
   else
   {
      std::cerr << "Invalid datset in get_file_name" << std::endl;
      return "";
   }
}

// return name is in arg file_path
void get_index_name(
    int N,
    int n_centroids,
    std::string assignment_type,
    float alpha,
    int M_beta,
    std::string &file_path)
{
   std::stringstream filepath_stream;
   filepath_stream << "./tmp/hybrid_" << (int)(N / 1000 / 1000)
                   << "m_nc=" << n_centroids
                   << "_assignment=" << assignment_type << "_alpha=" << alpha
                   << "Mb=" << M_beta << ".json";
   // copy filepath_stream to file_path
   file_path = filepath_stream.str();
}

/*******************************************************
 * Added for debugging
 *******************************************************/
const int debugFlag = 1;

void debugTime()
{
   if (debugFlag)
   {
      struct timeval tval;
      gettimeofday(&tval, NULL);
      struct tm *tm_info = localtime(&tval.tv_sec);
      char timeBuff[25] = "";
      strftime(timeBuff, 25, "%H:%M:%S", tm_info);
      char timeBuffWithMilli[50] = "";
      sprintf(timeBuffWithMilli, "%s.%06ld ", timeBuff, tval.tv_usec);
      std::string timestamp(timeBuffWithMilli);
      std::cout << timestamp << std::flush;
   }
}

// needs atleast 2 args always
//  alt debugFlag = 1 // fprintf(stderr, fmt, __VA_ARGS__);
#define debug(fmt, ...)                          \
   do                                            \
   {                                             \
      if (debugFlag == 1)                        \
      {                                          \
         fprintf(stdout, "--" fmt, __VA_ARGS__); \
      }                                          \
      if (debugFlag == 2)                        \
      {                                          \
         debugTime();                            \
         fprintf(stdout,                         \
                 "%s:%d:%s(): " fmt,             \
                 __FILE__,                       \
                 __LINE__,                       \
                 __func__,                       \
                 __VA_ARGS__);                   \
      }                                          \
   } while (0)

double elapsed()
{
   struct timeval tv;
   gettimeofday(&tv, NULL);
   return tv.tv_sec + tv.tv_usec * 1e-6;
}

/*******************************************************
 * performance testing helpers
 *******************************************************/
std::pair<float, float> get_mean_and_std(std::vector<float> &times)
{
   // compute mean
   float total = 0;
   // for (int num: times) {
   for (int i = 0; i < times.size(); i++)
   {
      // printf("%f, ", times[i]); // for debugging
      total = total + times[i];
   }
   float mean = (total / times.size());

   // compute stdev from variance, using computed mean
   float result = 0;
   for (int i = 0; i < times.size(); i++)
   {
      result = result + (times[i] - mean) * (times[i] - mean);
   }
   float variance = result / (times.size() - 1);
   // for debugging
   // printf("variance: %f\n", variance);

   float std = std::sqrt(variance);

   // return
   return std::make_pair(mean, std);
}

// ground truth labels @gt, results to evaluate @I with @nq queries, returns
// @gt_size-Recall@k where gt had max gt_size NN's per query
float compute_recall(
    std::vector<faiss::idx_t> &gt,
    int gt_size,
    std::vector<faiss::idx_t> &I,
    int nq,
    int k,
    int gamma = 1)
{
   // printf("compute_recall params: gt.size(): %ld, gt_size: %d, I.size():
   // %ld, nq: %d, k: %d, gamma: %d\n", gt.size(), gt_size, I.size(), nq, k,
   // gamma);

   int n_1 = 0, n_10 = 0, n_100 = 0;
   for (int i = 0; i < nq; i++)
   { // loop over all queries
      // int gt_nn = gt[i * k];
      std::vector<faiss::idx_t>::const_iterator first =
          gt.begin() + i * gt_size;
      std::vector<faiss::idx_t>::const_iterator last =
          gt.begin() + i * gt_size + (k / gamma);
      std::vector<faiss::idx_t> gt_nns_tmp(first, last);
      // if (gt_nns_tmp.size() > 10) {
      //     printf("gt_nns size: %ld\n", gt_nns_tmp.size());
      // }

      // gt_nns_tmp.resize(k); // truncate if gt_size > k
      std::set<faiss::idx_t> gt_nns(gt_nns_tmp.begin(), gt_nns_tmp.end());
      // if (gt_nns.size() > 10) {
      //     printf("gt_nns size: %ld\n", gt_nns.size());
      // }

      for (int j = 0; j < k; j++)
      { // iterate over returned nn results
         if (gt_nns.count(I[i * k + j]) != 0)
         {
            // if (I[i * k + j] == gt_nn) {
            if (j < 1 * gamma)
               n_1++;
            if (j < 10 * gamma)
               n_10++;
            if (j < 100 * gamma)
               n_100++;
         }
      }
   }
   // BASE ACCURACY
   // printf("* Base HNSW accuracy relative to exact search:\n");
   // printf("\tR@1 = %.4f\n", n_1 / float(nq) );
   // printf("\tR@10 = %.4f\n", n_10 / float(nq));
   // printf("\tR@100 = %.4f\n", n_100 / float(nq)); // not sure why this is
   // always same as R@10 printf("\t---Results for %ld queries, k=%d, N=%ld,
   // gt_size=%d\n", nq, k, N, gt_size);
   return (n_10 / float(nq));
}

template <typename T>
void log_values(std::string annotation, std::vector<T> &values)
{
   std::cout << annotation;
   for (int i = 0; i < values.size(); i++)
   {
      std::cout << values[i];
      if (i < values.size() - 1)
      {
         std::cout << ", ";
      }
   }
   std::cout << std::endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// FOR CORRELATION TESTING
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
std::vector<T> load_json_to_vector(std::string filepath)
{
   // Open the JSON file
   std::ifstream file(filepath);
   if (!file.is_open())
   {
      std::cerr << "Failed to open JSON file" << std::endl;
      // return 1;
   }

   // Parse the JSON data
   json data;
   try
   {
      file >> data;
   }
   catch (const std::exception &e)
   {
      std::cerr << "Failed to parse JSON data from " << filepath << ": "
                << e.what() << std::endl;
      // return 1;
   }

   // Convert data to a vector
   std::vector<T> v = data.get<std::vector<T>>();

   // print size
   std::cout << "metadata or vector loaded from json, size: " << v.size()
             << std::endl;
   return v;
}

// fxy_add
template <typename T>
std::vector<std::vector<T>> load_txt_to_vector_multi(
    const std::string &filepath)
{
   std::ifstream file(filepath);
   if (!file.is_open())
   {
      throw std::runtime_error("Failed to open TXT file: " + filepath);
   }
   std::vector<std::vector<T>> result;
   std::string line;
   while (std::getline(file, line))
   {
      std::vector<T> row;
      std::istringstream iss(line);
      std::string token;
      while (std::getline(iss, token, ','))
      {
         try
         {
            if constexpr (std::is_same_v<T, int>)
            {
               row.push_back(std::stoi(token));
            }
            else if constexpr (std::is_same_v<T, float>)
            {
               row.push_back(std::stof(token));
            }
            else if constexpr (std::is_same_v<T, double>)
            {
               row.push_back(std::stod(token));
            }
            else if constexpr (std::is_same_v<T, std::string>)
            {
               row.push_back(token);
            }
            else
            {
               throw std::runtime_error(
                   "Unsupported type for TXT parsing");
            }
         }
         catch (const std::exception &e)
         {
            throw std::runtime_error("Failed to parse value: " + token);
         }
      }
      if (!row.empty())
      {
         result.push_back(row);
      }
   }
   std::cout << "Loaded " << result.size() << " rows from TXT" << std::endl;
   return result;
}

std::vector<int> load_aq(
    std::string dataset,
    int n_centroids,
    int alpha,
    int N,
    std::string ATTR_DATA_DIR)
{
   if (dataset == "sift1M" || dataset == "sift1B")
   {
      assert((alpha == -2 || alpha == 0 || alpha == 2) ||
             !"alpha must be value in [-2, 0, 2]");

      // Compose File Name
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/query_filters_sift"
                      << (int)(N / 1000 / 1000) << "m_nc=" << n_centroids
                      << "_alpha=" << alpha << ".json";
      std::string filepath = filepath_stream.str();

      std::vector<int> v = load_json_to_vector<int>(filepath);
      printf("loaded query attributes from: %s\n", filepath.c_str());

      // // print out data for debugging
      // for (int i : v) {
      //     std::cout << i << " ";
      // }
      // std::cout << std::endl;

      return v;
   }
   else if (dataset == "tripclick")
   {
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR
                      << "/query_filters_tripclick_sample_subset_min100.json";
      std::string filepath = filepath_stream.str();
      // printf("%s\n", filepath.c_str());

      std::vector<int> v = load_json_to_vector<int>(filepath);
      printf("loaded query attributes from: %s\n", filepath.c_str());

      // // print out data for debugging
      // for (int i : v) {
      //     std::cout << i << " ";
      // }
      // std::cout << std::endl;

      return v;
   }
   else if (dataset == "sift1M_test")
   {
      // return a vector of all int 5 with lenght N
      std::vector<int> v(N, 5);
      printf("made query filters with value %d, length %ld\n",
             v[0],
             v.size());
      return v;
   }
   else if (dataset == "paper")
   {
      std::vector<int> v(N, 5);
      printf("made query filters with value %d, length %ld\n",
             v[0],
             v.size());
      return v;
   }
   else if (dataset == "paper_rand2m")
   {
      // Compose File Name
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR
                      << "/query_filters_paper_rand2m_nc=12_alpha=0.json";
      std::string filepath = filepath_stream.str();

      std::vector<int> v = load_json_to_vector<int>(filepath);
      printf("loaded query attributes from: %s\n", filepath.c_str());
      return v;
   }
   else if (dataset == "words")
   {
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/words_query_labels.txt";
      std::string filepath = filepath_stream.str();
      std::vector<int> v = load_json_to_vector<int>(filepath);
      printf("loaded query attributes from: %s\n", filepath.c_str());

      // // print out data for debugging
      // for (int i : v) {
      //     std::cout << i << " ";
      // }
      // std::cout << std::endl;

      return v;
   }
   else
   {
      std::cerr << "Invalid dataset in load_aq" << std::endl;
      return std::vector<int>();
   }
}
// fxy_add
std::vector<std::vector<int>> load_aq_multi(
    std::string dataset,
    int n_centroids,
    int alpha,
    int N,
    std::string ATTR_DATA_DIR)
{
   if (dataset == "sift1M" || dataset == "sift1B")
   {
      assert((alpha == -2 || alpha == 0 || alpha == 2) ||
             !"alpha must be value in [-2, 0, 2]");

      // Compose File Name
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/query_required_filters_sift"
                      << (int)(N / 1000 / 1000) << "m_nc=" << n_centroids
                      << "_alpha=" << alpha << ".json";
      std::string filepath = filepath_stream.str();

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      printf("loaded query attributes from: %s\n", filepath.c_str());

      // // print out data for debugging
      // for (int i : v) {
      //     std::cout << i << " ";
      // }
      // std::cout << std::endl;

      return v;
   }
   else if (dataset == "tripclick")
   {
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR
                      << "/query_filters_tripclick_sample_subset_min100.json";
      std::string filepath = filepath_stream.str();
      // printf("%s\n", filepath.c_str());

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      printf("loaded query attributes from: %s\n", filepath.c_str());

      // // print out data for debugging
      // for (int i : v) {
      //     std::cout << i << " ";
      // }
      // std::cout << std::endl;

      return v;
   }
   else if (dataset == "words")
   {
      assert((alpha == -2 || alpha == 0 || alpha == 2) ||
             !"alpha must be value in [-2, 0, 2]");

      // Compose File Name
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/words_query_labels.txt";
      std::string filepath = filepath_stream.str();

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      printf("loaded query attributes from: %s\n", filepath.c_str());
      return v;
   }
   else if (dataset == "TimeTravel")
   {
      assert((alpha == -2 || alpha == 0 || alpha == 2) ||
             !"alpha must be value in [-2, 0, 2]");

      // Compose File Name
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/TimeTravel_query_labels.txt";
      std::string filepath = filepath_stream.str();

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      printf("loaded query attributes from: %s\n", filepath.c_str());
      return v;
   }
   else if (dataset == "MTG")
   {
      assert((alpha == -2 || alpha == 0 || alpha == 2) ||
             !"alpha must be value in [-2, 0, 2]");

      // Compose File Name
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/MTG_query_labels.txt";
      std::string filepath = filepath_stream.str();

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      printf("loaded query attributes from: %s\n", filepath.c_str());
      return v;
   }
   else if (dataset == "captcha")
   {
      assert((alpha == -2 || alpha == 0 || alpha == 2) ||
             !"alpha must be value in [-2, 0, 2]");

      // Compose File Name
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/captcha_query_labels.txt";
      std::string filepath = filepath_stream.str();

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      printf("loaded query attributes from: %s\n", filepath.c_str());
      return v;
   }
   else if (dataset == "arxiv")
   {
      assert((alpha == -2 || alpha == 0 || alpha == 2) ||
             !"alpha must be value in [-2, 0, 2]");

      // Compose File Name
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/arxiv_query_labels.txt";
      std::string filepath = filepath_stream.str();

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      printf("loaded query attributes from: %s\n", filepath.c_str());
      return v;
   }
   else if (dataset == "russian")
   {
      assert((alpha == -2 || alpha == 0 || alpha == 2) ||
             !"alpha must be value in [-2, 0, 2]");

      // Compose File Name
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/russian_query_labels.txt";
      std::string filepath = filepath_stream.str();

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      printf("loaded query attributes from: %s\n", filepath.c_str());
      return v;
   }
   else if (dataset == "bookimg")
   {
      assert((alpha == -2 || alpha == 0 || alpha == 2) ||
             !"alpha must be value in [-2, 0, 2]");

      // Compose File Name
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/bookimg_query_labels.txt";
      std::string filepath = filepath_stream.str();

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      printf("loaded query attributes from: %s\n", filepath.c_str());
      return v;
   }
   else if (dataset == "TimeTravel")
   {
      assert((alpha == -2 || alpha == 0 || alpha == 2) ||
             !"alpha must be value in [-2, 0, 2]");

      // Compose File Name
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/TimeTravel_query_labels.txt";
      std::string filepath = filepath_stream.str();

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      printf("loaded query attributes from: %s\n", filepath.c_str());
      return v;
   }
   else if (dataset == "app_reviews")
   {
      assert((alpha == -2 || alpha == 0 || alpha == 2) ||
             !"alpha must be value in [-2, 0, 2]");

      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/app_reviews_query_labels.txt";
      std::string filepath = filepath_stream.str();

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      printf("loaded query attributes from: %s\n", filepath.c_str());
      return v;
   }
   else if (dataset == "amazing_file")
   {
      assert((alpha == -2 || alpha == 0 || alpha == 2) ||
             !"alpha must be value in [-2, 0, 2]");

      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/amazing_file_query_labels.txt";
      std::string filepath = filepath_stream.str();

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      printf("loaded query attributes from: %s\n", filepath.c_str());
      return v;
   }
   else if (dataset == "celeba")
   {
      assert((alpha == -2 || alpha == 0 || alpha == 2) ||
             !"alpha must be value in [-2, 0, 2]");

      // Compose File Name
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/celeba_query_labels.txt";
      std::string filepath = filepath_stream.str();

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      printf("loaded query attributes from: %s\n", filepath.c_str());
      return v;
   }
   else if (dataset == "VariousTaggedImages")
   {
      assert((alpha == -2 || alpha == 0 || alpha == 2) ||
             !"alpha must be value in [-2, 0, 2]");

      // Compose File Name
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/VariousTaggedImages_query_labels.txt";
      std::string filepath = filepath_stream.str();

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      printf("loaded query attributes from: %s\n", filepath.c_str());
      return v;
   }
   else
   {
      std::cerr << "Invalid dataset in load_aq_multi" << std::endl;
      return std::vector<std::vector<int>>();
   }
}

// assignment_type can be "rand", "soft", "soft_squared", "hard"
std::vector<int> load_ab(
    std::string dataset,
    int n_centroids,
    std::string assignment_type,
    int N,
    std::string ATTR_DATA_DIR)
{
   // Compose File Name
   if (dataset == "sift1M" || dataset == "sift1B")
   {
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/base_attrs_sift"
                      << (int)(N / 1000 / 1000) << "m_nc=" << n_centroids
                      << "_assignment=" << assignment_type << ".json";
      std::string filepath = filepath_stream.str();
      // printf("%s\n", filepath.c_str());

      std::vector<int> v = load_json_to_vector<int>(filepath);
      printf("loaded base attributes from: %s\n", filepath.c_str());

      // // print out data for debugging
      // for (int i : v) {
      //     std::cout << i << " ";
      // }
      // std::cout << std::endl;

      return v;
   }
   else if (dataset == "sift1M_test")
   {
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/sift_attr"
                      << ".json";
      std::string filepath = filepath_stream.str();
      // printf("%s\n", filepath.c_str());

      std::vector<int> v = load_json_to_vector<int>(filepath);
      printf("loaded base attributes from: %s\n", filepath.c_str());
      return v;
   }
   else if (dataset == "paper")
   {
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/paper_attr.json";
      std::string filepath = filepath_stream.str();
      // printf("%s\n", filepath.c_str());

      std::vector<int> v = load_json_to_vector<int>(filepath);
      printf("loaded base attributes from: %s\n", filepath.c_str());

      return v;
   }
   else if (dataset == "paper_rand2m")
   {
      std::stringstream filepath_stream;
      filepath_stream
          << ATTR_DATA_DIR
          << "/base_attrs_paper_rand2m_nc=12_assignment=rand.json";
      std::string filepath = filepath_stream.str();

      std::vector<int> v = load_json_to_vector<int>(filepath);
      printf("loaded base attributes from: %s\n", filepath.c_str());

      return v;
   }
   else if (dataset == "tripclick")
   {
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/base_attrs_tripclick.json";
      std::string filepath = filepath_stream.str();
      // printf("%s\n", filepath.c_str());

      std::vector<int> v = load_json_to_vector<int>(filepath);
      printf("loaded base attributes from: %s\n", filepath.c_str());

      // // print out data for debugging
      // for (int i : v) {
      //     std::cout << i << " ";
      // }
      // std::cout << std::endl;

      return v;
   }
   else if (dataset == "words")
   {
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/words_base_labels.txt";
      std::string filepath = filepath_stream.str();
      // printf("%s\n", filepath.c_str());

      std::vector<int> v = load_json_to_vector<int>(filepath);
      printf("loaded base attributes from: %s\n", filepath.c_str());
      return v;
   }
   else
   {
      std::cerr << "Invalid dataset in load_ab" << std::endl;
      return std::vector<int>();
   }
}

// fxy_add 载入多属性TXT文件
std::vector<std::vector<int>> load_ab_muti(
    std::string dataset,
    int n_centroids,
    std::string assignment_type,
    int N,
    std::string ATTR_DATA_DIR)
{
   // Compose File Name
   if (dataset == "sift1M" || dataset == "sift1B")
   {
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/base_attrs_sift"
                      << (int)(N / 1000 / 1000) << "m_nc=" << n_centroids
                      << "_assignment=" << assignment_type << ".json";
      std::string filepath = filepath_stream.str();
      // printf("%s\n", filepath.c_str());

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      printf("loaded base attributes from: %s\n", filepath.c_str());

      // // print out data for debugging
      // for (int i : v) {
      //     std::cout << i << " ";
      // }
      // std::cout << std::endl;

      return v;
   }
   else if (dataset == "sift1M_test")
   {
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/sift_attr"
                      << ".json";
      std::string filepath = filepath_stream.str();
      // printf("%s\n", filepath.c_str());

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      printf("loaded base attributes from: %s\n", filepath.c_str());
      return v;
   }
   else if (dataset == "paper")
   {
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/paper_attr.json";
      std::string filepath = filepath_stream.str();
      // printf("%s\n", filepath.c_str());

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      printf("loaded base attributes from: %s\n", filepath.c_str());

      return v;
   }
   else if (dataset == "paper_rand2m")
   {
      std::stringstream filepath_stream;
      filepath_stream
          << ATTR_DATA_DIR
          << "/base_attrs_paper_rand2m_nc=12_assignment=rand.json";
      std::string filepath = filepath_stream.str();

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      printf("loaded base attributes from: %s\n", filepath.c_str());

      return v;
   }
   else if (dataset == "tripclick")
   {
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/base_attrs_tripclick.json";
      std::string filepath = filepath_stream.str();
      // printf("%s\n", filepath.c_str());

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      printf("loaded base attributes from: %s\n", filepath.c_str());
      return v;
   }
   else if (dataset == "words")
   {
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/words_base_labels.txt";
      std::string filepath = filepath_stream.str();

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      std::cout << "loaded base attributes from:" << filepath.c_str()
                << std::endl;
      return v;
   }
   else if (dataset == "MTG")
   {
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/MTG_base_labels.txt";
      std::string filepath = filepath_stream.str();

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      std::cout << "loaded base attributes from:" << filepath.c_str()
                << std::endl;
      return v;
   }
   else if (dataset == "captcha")
   {
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/captcha_base_labels.txt";
      std::string filepath = filepath_stream.str();

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      std::cout << "loaded base attributes from:" << filepath.c_str()
                << std::endl;
      return v;
   }
   else if (dataset == "arxiv")
   {
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/arxiv_base_labels.txt";
      std::string filepath = filepath_stream.str();

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      std::cout << "loaded base attributes from:" << filepath.c_str()
                << std::endl;
      return v;
   }
   else if (dataset == "russian")
   {
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/russian_base_labels.txt";
      std::string filepath = filepath_stream.str();

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      std::cout << "loaded base attributes from:" << filepath.c_str()
                << std::endl;
      return v;
   }
   else if (dataset == "bookimg")
   {
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/bookimg_base_labels.txt";
      std::string filepath = filepath_stream.str();

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      std::cout << "loaded base attributes from:" << filepath.c_str()
                << std::endl;
      return v;
   }
   else if (dataset == "TimeTravel")
   {
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/TimeTravel_base_labels.txt";
      std::string filepath = filepath_stream.str();

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      std::cout << "loaded base attributes from:" << filepath.c_str()
                << std::endl;
      return v;
   }
   else if (dataset == "app_reviews")
   {
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/app_reviews_base_labels.txt";
      std::string filepath = filepath_stream.str();

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      std::cout << "loaded base attributes from:" << filepath.c_str()
                << std::endl;
      return v;
   }
   else if (dataset == "amazing_file")
   {
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/amazing_file_base_labels.txt";
      std::string filepath = filepath_stream.str();

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      std::cout << "loaded base attributes from:" << filepath.c_str()
                << std::endl;
      return v;
   }
   else if (dataset == "celeba")
   {
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/celeba_base_labels.txt";
      std::string filepath = filepath_stream.str();

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      std::cout << "loaded base attributes from:" << filepath.c_str()
                << std::endl;
      return v;
   }
   else if (dataset == "VariousTaggedImages")
   {
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/VariousTaggedImages_base_labels.txt";
      std::string filepath = filepath_stream.str();

      std::vector<std::vector<int>> v =
          load_txt_to_vector_multi<int>(filepath);
      std::cout << "loaded base attributes from:" << filepath.c_str()
                << std::endl;
      return v;
   }
   else
   {
      std::cerr << "Invalid dataset in load_ab_multi" << std::endl;
      return std::vector<std::vector<int>>();
   }
}

// assignment_type can be "rand", "soft", "soft_squared", "hard"
// alpha can be -2, 0, 2
std::vector<faiss::idx_t> load_gt(
    std::string dataset,
    int n_centroids,
    int alpha,
    std::string assignment_type,
    int N,
    std::string ATTR_DATA_DIR)
{
   if (dataset == "sift1M" || dataset == "sift1B")
   {
      // Compose File Name
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/gt_sift" << (int)(N / 1000 / 1000)
                      << "m_nc=" << n_centroids
                      << "_assignment=" << assignment_type
                      << "_alpha=" << alpha << ".json";
      std::string filepath = filepath_stream.str();
      // printf("%s\n", filepath.c_str());

      std::vector<int> v_tmp = load_json_to_vector<int>(filepath);
      std::vector<faiss::idx_t> v(v_tmp.begin(), v_tmp.end());
      printf("loaded gt from: %s\n", filepath.c_str());

      // // print out data for debugging
      // for (faiss::idx_t i : v) {
      //     std::cout << i << " ";
      // }
      // std::cout << std::endl;

      return v;
   }
   else if (dataset == "sift1M_test")
   {
      // Compose File Name
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/sift_gt_5.json";
      std::string filepath = filepath_stream.str();
      // printf("%s\n", filepath.c_str());

      std::vector<int> v_tmp = load_json_to_vector<int>(filepath);
      std::vector<faiss::idx_t> v(v_tmp.begin(), v_tmp.end());
      printf("loaded gt from: %s\n", filepath.c_str());

      // // print out data for debugging
      // for (faiss::idx_t i : v) {
      //     std::cout << i << " ";
      // }
      // std::cout << std::endl;

      return v;
   }
   else if (dataset == "paper")
   {
      // Compose File Name
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR << "/paper_gt_5.json";
      std::string filepath = filepath_stream.str();
      // printf("%s\n", filepath.c_str());

      std::vector<int> v_tmp = load_json_to_vector<int>(filepath);
      std::vector<faiss::idx_t> v(v_tmp.begin(), v_tmp.end());
      printf("loaded gt from: %s\n", filepath.c_str());

      // // print out data for debugging
      // for (faiss::idx_t i : v) {
      //     std::cout << i << " ";
      // }
      // std::cout << std::endl;

      return v;
   }
   else if (dataset == "paper_rand2m")
   {
      // Compose File Name
      std::stringstream filepath_stream;
      filepath_stream
          << ATTR_DATA_DIR
          << "/gt_paper_rand2m_nc=12_assignment=rand_alpha=0.json";
      std::string filepath = filepath_stream.str();
      // printf("%s\n", filepath.c_str());

      std::vector<int> v_tmp = load_json_to_vector<int>(filepath);
      std::vector<faiss::idx_t> v(v_tmp.begin(), v_tmp.end());
      printf("loaded gt from: %s\n", filepath.c_str());

      return v;
   }
   else if (dataset == "tripclick")
   {
      std::stringstream filepath_stream;
      filepath_stream << ATTR_DATA_DIR
                      << "/gt_tripclick_sample_subset_min100.json";
      std::string filepath = filepath_stream.str();
      // printf("%s\n", filepath.c_str());

      std::vector<int> v_tmp = load_json_to_vector<int>(filepath);
      std::vector<faiss::idx_t> v(v_tmp.begin(), v_tmp.end());
      printf("loaded gt from: %s\n", filepath.c_str());

      // // print out data for debugging
      // for (faiss::idx_t i : v) {
      //     std::cout << i << " ";
      // }
      // std::cout << std::endl;

      return v;
   }
   else
   {
      std::cerr << "Invalid dataset in load_gt" << std::endl;
      return std::vector<faiss::idx_t>();
   }
}

/**************************************************************
 * recall calculation
 **************************************************************/
// fxy_add
void save_single_query_to_txt(
    size_t query_idx,       // 当前查询的索引
    size_t ntotal,          // 存储向量的数量
    const float *distances, // 距离数组
    const std::string &filename_prefix,
    std::string MY_DIS_DIR)
{
   // 确保文件夹 'dis_of_every_query' 存在，如果不存在则创建
   if (!fs::exists(std::string(MY_DIS_DIR)))
   {
      fs::create_directories(std::string(MY_DIS_DIR));
      std::cout << "Directory created: " << std::string(MY_DIS_DIR)
                << std::endl;
   }

   // 生成查询的文件名，格式为 "dis_of_every_query/query_<query_idx>.txt"
   std::string filename = std::string(MY_DIS_DIR) + "/" + filename_prefix +
                          "_query_" + std::to_string(query_idx) + ".txt";

   // 打开文件并写入距离数据（每行一个距离值）
   std::ofstream output_file(filename);
   if (output_file.is_open())
   {
      for (size_t j = 0; j < ntotal; ++j)
      {
         output_file << distances[query_idx * ntotal + j] << "\n";
      }
      output_file.close();
      //   std::cout << "Distances for query " << query_idx << " saved to "
      //             << filename << std::endl;
   }
   else
   {
      std::cerr << "Failed to open file: " << filename << std::endl;
   }
}

// fxy_add
void save_distances_to_txt(
    size_t nq,              // 查询数量
    size_t ntotal,          // 存储向量数量
    const float *distances, // 存储距离的数组
    const std::string &filename_prefix,
    std::string MY_DIS_DIR)
{
   for (size_t i = 0; i < nq; ++i)
   {
      save_single_query_to_txt(i, ntotal, distances, filename_prefix,
                               MY_DIS_DIR);
   }
}

// fxy_add
float *read_all_distances_from_txt(
    const std::string &folder_path, // 文件夹路径，如 "dis_of_every_query"
    size_t nq,                      // 查询数量
    size_t N                        // 存储向量数量
)
{
   // 分配存储所有距离的数组
   float *all_distances = new float[nq * N];
   size_t index = 0;

   // 遍历所有查询文件
   for (size_t i = 0; i < nq; i++)
   {
      // 构造文件路径（假设文件名格式为 "distances_query_0.txt"）
      std::string file_path =
          folder_path + "/distances_query_" + std::to_string(i) + ".txt";

      // 打开文件
      std::ifstream file(file_path);
      if (!file.is_open())
      {
         std::cerr << "Could not open file: " << file_path << std::endl;
         continue;
      }

      // 逐行读取文件内容
      std::string line;
      size_t line_count = 0;
      while (std::getline(file, line))
      {
         if (line_count >= N)
         {
            std::cerr << "File " << file_path
                      << " has more lines than expected (" << N << ")!"
                      << std::endl;
            break;
         }

         // 将字符串转换为浮点数
         float value;
         std::istringstream iss(line);
         if (!(iss >> value))
         {
            std::cerr << "Failed to parse line " << line_count
                      << " in file " << file_path << std::endl;
            continue;
         }

         // 存储到数组
         all_distances[index++] = value;
         line_count++;
      }

      // 检查是否读取了足够的数据
      if (line_count < N)
      {
         std::cerr << "File " << file_path << " has fewer lines ("
                   << line_count << ") than expected (" << N << ")!"
                   << std::endl;
      }

      file.close();
   }

   return all_distances;
}

// fxy_add
std::vector<std::vector<std::pair<int, float>>> get_sorted_filtered_distances(
    const float *all_distances,
    const std::vector<char> &filter_ids_map,
    size_t nq,
    size_t N)
{
   std::vector<std::vector<std::pair<int, float>>> sort_filter_all_dist(nq);

   for (size_t xq = 0; xq < nq; xq++)
   {
      std::vector<std::pair<int, float>> filtered_pairs;

      // 遍历所有向量，筛选符合属性要求的
      for (size_t xb = 0; xb < N; xb++)
      {
         if (filter_ids_map[xq * N + xb])
         {
            float distance = all_distances[xq * N + xb];
            filtered_pairs.emplace_back(
                xb, distance); // 存储 (id, distance)
         }
      }

      // 按距离从小到大排序
      std::sort(
          filtered_pairs.begin(),
          filtered_pairs.end(),
          [](const auto &a, const auto &b)
          {
             return a.second < b.second; // 按距离排序
          });

      sort_filter_all_dist[xq] = std::move(filtered_pairs);
   }

   return sort_filter_all_dist;
}

// fxy_add
void save_sorted_filtered_distances_to_txt(
    const std::vector<std::vector<std::pair<int, float>>> &sorted_results,
    const std::string &output_dir,
    const std::string &filename_prefix)
{
   // 确保输出目录存在
   if (!fs::exists(output_dir))
   {
      fs::create_directory(output_dir);
      std::cout << "Created directory: " << output_dir << std::endl;
   }

   // 遍历每个查询的结果
   for (size_t xq = 0; xq < sorted_results.size(); xq++)
   {
      // 构造文件名（如 "filtered_sorted_distances/query_results_0.txt"）
      std::string filepath = output_dir + "/" + filename_prefix +
                             std::to_string(xq) + ".txt";

      // 打开文件
      std::ofstream outfile(filepath);
      if (!outfile.is_open())
      {
         std::cerr << "Failed to open file: " << filepath << std::endl;
         continue;
      }

      // 写入数据：每行格式 "id distance"
      for (const auto &[id, distance] : sorted_results[xq])
      {
         outfile << id << " " << distance << "\n";
      }

      outfile.close();
      //   std::cout << "Saved results for query " << xq << " to: " <<
      //   filepath<< std::endl;
   }
}

// fxy_add
std::vector<std::vector<std::pair<int, float>>> read_all_sorted_filtered_distances_from_txt(
    const std::string &input_dir,
    size_t nq,
    size_t N)
{
   std::vector<std::vector<std::pair<int, float>>> sorted_results(nq);

   for (size_t xq = 0; xq < nq; xq++)
   {
      // Construct the filename (e.g., "filtered_sorted_distances/query_results_0.txt")
      std::string filepath = input_dir + "/filter_sorted_dist_" + std::to_string(xq) + ".txt";

      // Open the file
      std::ifstream infile(filepath);
      if (!infile.is_open())
      {
         std::cerr << "Failed to open file: " << filepath << std::endl;
         continue;
      }

      std::string line;
      while (std::getline(infile, line))
      {
         std::istringstream iss(line);
         int id;
         float distance;

         if (!(iss >> id >> distance))
         {
            std::cerr << "Error parsing line in file: " << filepath << std::endl;
            continue;
         }

         sorted_results[xq].emplace_back(id, distance);
      }

      infile.close();
   }

   return sorted_results;
}

// fxy_add
std::vector<float> compute_recall(
    const std::vector<faiss::idx_t> nns2,
    const std::vector<std::vector<std::pair<int, float>>> &sorted_results,
    size_t nq,
    size_t k)
{
   std::vector<float> recalls(nq, 0.0f);

   for (size_t i = 0; i < nq; i++)
   {
      // 获取算法返回的有效 Top-K ID（跳过无效值，假设无效值为 -1）
      std::unordered_set<long> algo_top_k;
      size_t algo_valid_count = 0;
      for (size_t j = 0; j < k; j++)
      {
         long id = nns2[j + i * k];
         if (id != -1)
         { // 假设 -1 表示无效值
            algo_top_k.insert(id);
            algo_valid_count++;
         }
      }

      // 获取真实排序的有效 Top-K ID
      std::unordered_set<int> true_top_k;
      size_t true_valid_count = std::min(k, sorted_results[i].size());
      for (size_t j = 0; j < true_valid_count; j++)
      {
         true_top_k.insert(sorted_results[i][j].first);
      }

      // 计算交集数量
      size_t intersection = 0;
      for (long id : algo_top_k)
      {
         if (true_top_k.count(id))
         {
            intersection++;
         }
      }

      // 计算 Recall@K，分母为 min(K, algo_valid_count, true_valid_count)
      size_t denominator = std::min({k, algo_valid_count, true_valid_count});
      recalls[i] = denominator > 0
                       ? static_cast<float>(intersection) / denominator
                       : 0.0f;
   }

   return recalls;
}