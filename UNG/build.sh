#!/bin/bash

# ==============================================================================
# build.sh - 负责编译代码、构建索引 (v3 - Final)
# 该脚本只关心与索引结构相关的参数。
# ==============================================================================

set -e # 如果任何命令失败，则立即退出

# --- Step 1: 解析命令行参数 ---
while [[ $# -gt 0 ]]; do
    if [[ $1 == --* ]]; then
        key=$(echo "$1" | sed 's/--//' | tr '[:lower:]-' '[:upper:]_')
        if [ -z "$2" ]; then
            echo "错误: 参数 $1 缺少值"
            exit 1
        fi
        declare "$key"="$2"
        shift 2
    else
        echo "未知参数: $1"
        exit 1
    fi
done

# --- Step 2: 编译代码 (如果构建目录或可执行文件不存在) ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
if [ ! -f "$BUILD_DIR/apps/build_UNG_index" ]; then
    echo "编译目录或可执行文件不存在，开始编译..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR" || exit
    cmake -DCMAKE_BUILD_TYPE=Release "${SCRIPT_DIR}/codes/"
    make -j
    cd .. || exit
else
    echo "编译产物已存在，跳过编译步骤。"
fi


# --- Step 3: 根据【构建参数】构造唯一的索引输出目录 ---
INDEX_DIR_NAME="Q${NUM_QUERY_SETS}_M${MAX_DEGREE}_LB${LBUILD}_alpha${ALPHA}_C${NUM_CROSS_EDGES}_EP${NUM_ENTRY_POINTS}"
INDEX_OUTPUT_DIR="${OUTPUT_DIR}/UNG/${DATASET}/Index/${INDEX_DIR_NAME}"

# --- Step 4: 如果索引已存在，则跳过，实现复用 ---
if [ -d "$INDEX_OUTPUT_DIR" ]; then
    echo "索引目录 '$INDEX_OUTPUT_DIR' 已存在，跳过构建过程。"
    exit 0
fi

echo "索引不存在，开始新的构建任务..."
mkdir -p "$INDEX_OUTPUT_DIR/index_files"
mkdir -p "$INDEX_OUTPUT_DIR/others"
mkdir -p "$INDEX_OUTPUT_DIR/acorn_output" # 即使不用也创建，保持结构一致

# --- Step 5: 构建索引 ---
echo "开始构建索引..."
"$BUILD_DIR"/apps/build_UNG_index \
   --data_type float --dist_fn L2 --num_threads 32 \
   --max_degree "$MAX_DEGREE" --Lbuild "$LBUILD" --alpha "$ALPHA" --num_cross_edges "$NUM_CROSS_EDGES" \
   --base_bin_file "$DATA_DIR/${DATASET}_base.bin" \
   --base_label_file "$DATA_DIR/base_${NUM_QUERY_SETS}/${DATASET}_base_labels.txt" \
   --base_label_info_file "$DATA_DIR/base_${NUM_QUERY_SETS}/${DATASET}_base_labels_info.log" \
   --base_label_tree_roots "$DATA_DIR/base_${NUM_QUERY_SETS}/tree_roots.txt" \
   --index_path_prefix "$INDEX_OUTPUT_DIR/index_files/" \
   --result_path_prefix "$INDEX_OUTPUT_DIR/results/" \
   --scenario general \
   --dataset "$DATASET" \
   --generate_query false \
   --generate_query_task "$GENERATE_QUERY_TASK" \
   --method1_high_coverage_p "$METHOD1_HIGH_COVERAGE_P" \
   --query_file_path "$DATA_DIR/query_${NUM_QUERY_SETS}" \
   --ung_and_acorn "$UNG_AND_ACORN"  --new_edge_policy "$NEW_EDGE_POLICY" \
   --R_in_add_new_edge "$R_IN_ADD_NEW_EDGE"  --W_in_add_new_edge "$W_IN_ADD_NEW_EDGE"  --M_in_add_new_edge "$M_IN_ADD_NEW_EDGE" \
   --layer_depth_retio "$LAYER_DEPTH_RETIO"  --query_vector_ratio "$QUERY_VECTOR_RATIO"  --root_coverage_threshold "$ROOT_COVERAGE_THRESHOLD" \
   --acorn_in_ung_output_path "$INDEX_OUTPUT_DIR/acorn_output/" \
   --M "$M"  --M_beta "$M_BETA"  --gamma "$GAMMA"  --efs "$EFS"  --compute_recall "$COMPUTE_RECALL" > "$INDEX_OUTPUT_DIR/others/${DATASET}_build_index_output.txt" 2>&1

echo "构建完成！索引已保存到: $INDEX_OUTPUT_DIR"

