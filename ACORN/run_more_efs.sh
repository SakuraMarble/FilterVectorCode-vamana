#!/bin/bash

# 参数检查
if [ "$#" -ne 13 ]; then
    echo "用法: $0 <dataset_name> <N> <M> <M_beta> <gamma> <query_num> <threads> <efs_list> <efs_start> <efs_end> <efs_step> <repeat_num> <if_bfs_filter>"
    exit 1
fi

dataset=$1
N=$2
M=$3
M_beta=$4
gamma=$5
query_num=$6
threads=$7
efs_list=$8
efs_start=$9
efs_end=${10}
efs_step=${11}
repeat_num=${12}
if_bfs_filter=${13}

# 清理旧构建
rm -rf build_$dataset

# 使用CMake构建
cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B build_$dataset
make -C build_$dataset -j faiss
make -C build_$dataset utils
make -C build_$dataset test_acorn

##########################################
# 测试配置
##########################################
now=$(date +"%Y%m%d_%H%M%S")
parent_dir="../../FilterVectorResults/ACORN/${dataset}/${dataset}_query${query_num}_M${M}_gamma${gamma}_threads${threads}_repeat${repeat_num}_ifbfs${if_bfs_filter}_efs${efs_start}-${efs_step}-${efs_end}"
results="${parent_dir}/results"
mkdir -p "$results"

# 写入实验配置experiment_config.txt
config_file="${parent_dir}/experiment_config.txt"
echo "实验配置参数:" > $config_file
echo "数据集: $dataset" >> $config_file
echo "数据量(N): $N" >> $config_file
echo "M: $M" >> $config_file
echo "M_beta: $M_beta" >> $config_file
echo "gamma: $gamma" >> $config_file
echo "查询数量: $query_num" >> $config_file
echo "线程数: $threads" >> $config_file
echo "EFS列表: $efs_list" >> $config_file
echo "重复次数: $repeat_num" >> $config_file
echo "是否使用BFS过滤: $if_bfs_filter" >> $config_file
echo "实验时间: $now" >> $config_file

##########################################
# 运行测试
##########################################
base_path="../../FilterVectorData/${dataset}" # base vector dir
for i in $(seq $query_num $query_num); do
    query_path="../../FilterVectorData/${dataset}/query_${i}"
    base_label_path="../../FilterVectorData/${dataset}/base_${i}" 

    csv_path="${results}/${dataset}_query_${query_num}_M${M}_gamma${gamma}_threads${threads}_repeat${repeat_num}_ifbfs${if_bfs_filter}_efs${efs_start}-${efs_step}-${efs_end}.csv" 
    avg_csv_path="${results}/${dataset}_query_${query_num}_M${M}_gamma${gamma}_threads${threads}_repeat${repeat_num}_ifbfs${if_bfs_filter}_efs${efs_start}-${efs_step}-${efs_end}_avg.csv"
    dis_output_path="${parent_dir}/dis_output"

    echo "运行测试: 数据集=${dataset}, 查询=query${i}, gamma=${gamma}, M=${M}, 线程=${threads}, 重复次数=${repeat_num},if_bfs_filter=${if_bfs_filter}, EFS范围=${efs_start}-${efs_end}, EFS步长=${efs_step}"
    ./build_$dataset/demos/test_acorn $N $gamma $dataset $M $M_beta "$base_path" "$base_label_path" "$query_path" "$csv_path" "$avg_csv_path" "$dis_output_path" "$threads" "$repeat_num" "$if_bfs_filter" "$efs_list"&>> "${parent_dir}/output_log.log"
done

echo "测试完成! 结果保存在: ${parent_dir}"
echo "配置文件: $config_file"