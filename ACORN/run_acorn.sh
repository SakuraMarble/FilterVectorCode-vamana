#!/bin/bash

################# Script Description ##################
# This script compiles and runs the test_acorn program.
# It supports three modes for experiments:
# 1. build:  Compile the code, build indexes, and save them to a reusable, consolidated path.
# 2. search: Load existing indexes to perform search only, saving results to a unique experiment path.
# 3. all:    Execute the complete build and search workflow.
# All artifacts (Indexes, GroundTruth, Results) are stored under a single project directory.
#######################################################

# run_acorn.sh

# --- Step 1: Argument Check ---
if [ "$#" -ne 15 ]; then
    echo "Usage Error!"
    echo "Usage: $0 <mode> <dataset_name> <N> <M> <M_beta> <gamma> <query_num> <threads> <efs_list> <efs_start> <efs_end> <efs_step> <repeat_num> <if_bfs_filter> <k>"
    echo "      <mode> must be 'build', 'search', or 'all'"
    exit 1
fi

# --- Step 2: Read and Assign Arguments ---
mode=$1
dataset=$2
N=$3
M=$4
M_beta=$5
gamma=$6
query_num=$7
threads=$8
# 新增 efs_list 的赋值，并调整后续参数的索引
efs_list=$9
efs_start=${10}
efs_end=${11}
efs_step=${12}
repeat_num=${13}
if_bfs_filter=${14}
k=${15}


# Mode validation check
if [[ "$mode" != "build" && "$mode" != "search" && "$mode" != "all" ]]; then
    echo "Error: Invalid mode '$mode'. Mode must be 'build', 'search', or 'all'."
    exit 1
fi

# --- Step 3: Compile the Code ---
echo "--- Starting project compilation ---"
rm -rf build_$dataset

# Build using CMake
cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B build_$dataset
make -C build_$dataset -j faiss
make -C build_$dataset utils
make -C build_$dataset test_acorn
echo "--- Project compilation finished ---"


# --- Step 4: Set up Paths and Config File ---
now=$(date +"%Y%m%d_%H%M%S")

# 0. Define a single, consolidated base directory for all ACORN-related artifacts.
acorn_base_dir="../../FilterVectorResults/ACORN"

# 1. Define the INDEX directory path, nested within the base directory.
index_parent_dir="${acorn_base_dir}/${dataset}/Index"
index_dir="${index_parent_dir}/N${N}_M${M}_gamma${gamma}_Mb${M_beta}"

# 2. Define the GROUND TRUTH directory path, nested within the base directory.
ground_truth_parent_dir="${acorn_base_dir}/${dataset}/GroundTruth"
dis_output_path="${ground_truth_parent_dir}/"

# 3. Define the RESULTS directory path, nested within the base directory.
results_parent_dir="${acorn_base_dir}/${dataset}/Results/index_N${N}_M${M}_gamma${gamma}_Mb${M_beta}"
results_dir_name="query${query_num}_threads${threads}_k${k}_repeat${repeat_num}_ifbfs${if_bfs_filter}_efs${efs_start}-${efs_step}-${efs_end}"
final_results_dir="${results_parent_dir}/${results_dir_name}"

# Create all necessary directories
mkdir -p "$index_dir"
mkdir -p "$ground_truth_parent_dir"
mkdir -p "$final_results_dir/results"

# Define the full paths for the index files
index_path_acorn="${index_dir}/acorn.index"
index_path_acorn1="${index_dir}/acorn1.index"

# Write experiment configuration and logs to the unique results directory
config_file="${final_results_dir}/experiment_config.txt"
echo "Experiment Configuration:" > $config_file
echo "Run Mode: $mode" >> $config_file
echo "Dataset: $dataset" >> $config_file
echo "Data Size (N): $N" >> $config_file
echo "M: $M, M_beta: $M_beta, gamma: $gamma" >> $config_file
echo "k: $k" >> $config_file
echo "Query Num: $query_num, Threads: $threads, Repeat: $repeat_num" >> $config_file
echo "EFS Range: $efs_start -> $efs_end (step $efs_step)" >> $config_file
echo "Experiment Time: $now" >> $config_file
echo "---" >> $config_file
echo "Index Path (used/created): $index_dir" >> $config_file
echo "Ground Truth Path (used/created): ${dis_output_path}${dataset}_query_${query_num}/" >> $config_file
echo "Results Path: $final_results_dir" >> $config_file


# --- Step 5: Run Tests in Different Modes ---
base_path="../../FilterVectorData/${dataset}"
query_path="../../FilterVectorData/${dataset}/query_${query_num}"
base_label_path="../../FilterVectorData/${dataset}/base_${query_num}"

# All CSV output files will go into the unique results directory
csv_path="${final_results_dir}/results/"
avg_csv_path="${final_results_dir}/results/"

# --- Build Phase ---
if [[ "$mode" == "build" || "$mode" == "all" ]]; then
    echo -e "\n--- [Phase 1/2] Executing 'build' mode ---"
    echo "Building indexes and saving to: ${index_dir}"

    # Call test_acorn in build mode.
    # Logs are saved to the results directory for this specific run.
    ./build_$dataset/demos/test_acorn build \
        $N $gamma $dataset $M $M_beta \
        "$base_path" "$base_label_path" "$query_path" \
        "$csv_path" "$avg_csv_path" "$dis_output_path" \
        "$threads" "$repeat_num" "$if_bfs_filter" "$efs_list" \
        "$index_path_acorn" "$index_path_acorn1" "$k"&>> "${final_results_dir}/output_log_build.log"

    if [ $? -ne 0 ]; then
        echo "Error: Index build failed! Please check the log file: ${final_results_dir}/output_log_build.log"
        exit 1
    fi
    echo "--- Index build successful ---"
fi

# --- Search Phase ---
if [[ "$mode" == "search" || "$mode" == "all" ]]; then
    echo -e "\n--- [Phase 2/2] Executing 'search' mode ---"

    # Before searching, check if the required index files exist in their dedicated path
    if [ ! -f "$index_path_acorn" ] || [ ! -f "$index_path_acorn1" ]; then
        echo "Error: Index files not found! Please run with 'build' or 'all' mode first."
        echo "Expected index file path: ${index_dir}"
        exit 1
    fi
    echo "Loading indexes from path: ${index_dir}"

    # Call test_acorn in search mode.
    # Logs for the search phase are also saved to the unique results directory.
    ./build_$dataset/demos/test_acorn search \
        $N $gamma $dataset $M $M_beta \
        "$base_path" "$base_label_path" "$query_path" \
        "$csv_path" "$avg_csv_path" "$dis_output_path" \
        "$threads" "$repeat_num" "$if_bfs_filter" "$efs_list" \
        "$index_path_acorn" "$index_path_acorn1" "$k"&>> "${final_results_dir}/output_log_search.log"

    if [ $? -ne 0 ]; then
        echo "Error: Search execution failed! Please check the log file: ${final_results_dir}/output_log_search.log"
        exit 1
    fi
    echo "--- Search task successful ---"
fi

echo -e "\nTest finished! Results saved in: ${final_results_dir}"
echo "Configuration file: $config_file"