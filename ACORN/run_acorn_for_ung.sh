#!/bin/bash

# ==============================================================================
#           脚本: run_acorn_for_ung.sh 
#           功能: 运行 test_acorn_in_ung 程序
# ==============================================================================

# --- 1. 参数检查 ---
if [ "$#" -ne 12 ]; then
    echo "用法: $0 <dataset> <k> <N> <M> <M_beta> <gamma> <efs> <threads> <query_vec_path> <query_attr_path> <output_dir> <compute_recall>"
    exit 1
fi

# --- 2. 为参数分配变量 ---
dataset=$1
k=$2
N=$3
M=$4
M_beta=$5
gamma=$6
efs=$7
threads=$8
query_vec_path=$9
query_attr_path=${10}
output_dir=${11}
compute_recall_flag=${12}


# ---  确定项目和脚本的绝对路径  ---
# 获取脚本文件所在的绝对目录
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ACORN_SOURCE_DIR="$SCRIPT_DIR"
PROJECT_ROOT=$(dirname "$ACORN_SOURCE_DIR")
ACORN_BUILD_DIR="${ACORN_SOURCE_DIR}/build_${dataset}"
EXECUTABLE_PATH="${ACORN_BUILD_DIR}/demos/test_acorn_in_ung"

echo "脚本已自动定位路径:"
echo " - ACORN源码目录: ${ACORN_SOURCE_DIR}"
echo " - 项目根目录:    ${PROJECT_ROOT}"
echo " - 构建目录:      ${ACORN_BUILD_DIR}"
echo " - 可执行文件:    ${EXECUTABLE_PATH}"


# --- 3. 编译 C++ 程序 (使用绝对路径) ---
echo "正在清理并编译ACORN项目..."
rm -rf "$ACORN_BUILD_DIR" # 使用变量代替硬编码的绝对路径

# 使用 -S 选项告诉cmake源码在哪里，-B 告诉在哪里构建,这样就不会依赖当前工作目录了。
cmake -S "$ACORN_SOURCE_DIR" \
      -B "$ACORN_BUILD_DIR" \
      -DFAISS_ENABLE_GPU=OFF \
      -DFAISS_ENABLE_PYTHON=OFF \
      -DBUILD_TESTING=ON \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_BUILD_TYPE=Release

#  使用绝对路径进行编译 
make -C "$ACORN_BUILD_DIR" -j "$threads" test_acorn_in_ung
if [ $? -ne 0 ]; then
    echo "编译失败! 请检查CMakeLists.txt和C++源代码。"
    exit 1
fi
echo "编译成功。"


# --- 4. 设置输出目录和配置文件 ---
now=$(date +"%Y%m%d_%H%M%S")
mkdir -p "$output_dir"
config_file="${output_dir}/run_config.txt"
echo "实验配置参数 (运行于: $now):" > "$config_file"
echo "------------------------------------" >> $config_file
echo "查询属性目录: $query_attr_path" >> $config_file
echo "输出目录: $output_dir" >> $config_file
echo "计算召回率: $compute_recall_flag" >> $config_file
echo "------------------------------------" >> $config_file


# --- 5. 运行 ACORN 搜索 ---
DATA_ROOT="/data/fxy/FilterVector/FilterVectorData"
base_path="${DATA_ROOT}/${dataset}"
base_label_path="${DATA_ROOT}/${dataset}/base_9" 
output_neighbors_file="${output_dir}/R_neighbors_with_vectors.txt"
log_file="${output_dir}/run_acorn.log"

echo ""
echo "即将运行 ACORN 搜索 (数据路径已修正)..."
echo "数据根目录: ${DATA_ROOT}"
echo "所有输出和错误信息将被重定向到: $log_file"

# 构建C++命令
CMD_ARGS=(
    --dataset "$dataset"
    --base_path "$base_path"
    --base_label_path "$base_label_path"
    --query_vec_path "$query_vec_path"
    --query_attr_path "$query_attr_path"
    --output_path "$output_neighbors_file"
    --N "$N" --M "$M" --M_beta "$M_beta" --gamma "$gamma"
    --efs "$efs" --k "$k" --threads "$threads"
)

# 根据标志添加 --compute_recall 参数
if [ "$compute_recall_flag" -eq 1 ]; then
    echo "召回率计算已启用。"
    CMD_ARGS+=(--compute_recall)
fi

#  使用绝对路径执行程序 
"$EXECUTABLE_PATH" "${CMD_ARGS[@]}" &>> "$log_file"

if [ $? -ne 0 ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!!!  错误: C++ 程序 'test_acorn_in_ung' 执行失败。"
    echo "!!!  请检查日志文件获取详细信息: $log_file"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    exit 1
fi

# --- 6. 完成  ---
echo ""
echo "✅  ACORN 搜索任务已完成!"