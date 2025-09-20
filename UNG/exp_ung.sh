#!/bin/bash

# ==============================================================================
# exp_ung.sh - 实验流程主控脚本
# 负责读取JSON配置，并为【每个】实验配置按需调用 build, generate_gt, search
# ==============================================================================

set -e # 如果任何命令失败，则立即退出

export TMPDIR="/data/fxy/FilterVector/build"
mkdir -p "$TMPDIR"

# --- 检查jq是否安装 ---
if ! command -v jq &> /dev/null; then
    echo "错误: jq 未安装。请先安装 jq (https://stedolan.github.io/jq/)"
    exit 1
fi

CONFIG_FILE="../../FilterVectorCode/UNG/experiments.json"

# ==============================================================================
# 核心逻辑:
# 遍历【所有】实验配置, 为每个配置:
#   a. 提取【全部】参数（构建、GT、搜索）。
#   b. 调用 build.sh 创建或确认索引存在。
#   c. 调用 generate_gt.sh 创建或确认GT文件存在。
#   d. 调用 search.sh 在对应的索引和GT上执行搜索。
# ==============================================================================

cat "$CONFIG_FILE" | jq -c '.experiments[]' | while read -r experiment; do
    echo -e "\n=========================================================="
    echo "处理新的实验配置..."
    echo "=========================================================="

    # --- 【步骤1】为当前实验提取【所有】参数 ---
    
    # - 通用参数
    DATASET=$(echo "$experiment" | jq -r '.dataset')
    DATA_DIR=$(echo "$experiment" | jq -r '.data_dir')
    OUTPUT_DIR=$(echo "$experiment" | jq -r '.output_dir')

    # - 构建参数
    NUM_QUERY_SETS=$(echo "$experiment" | jq -r '.num_query_sets')
    MAX_DEGREE=$(echo "$experiment" | jq -r '.max_degree')
    LBUILD=$(echo "$experiment" | jq -r '.Lbuild')
    ALPHA=$(echo "$experiment" | jq -r '.alpha')
    NUM_CROSS_EDGES=$(echo "$experiment" | jq -r '.num_cross_edges')
    NUM_ENTRY_POINTS=$(echo "$experiment" | jq -r '.num_entry_points')
    UNG_AND_ACORN=$(echo "$experiment" | jq -r '.ung_and_acorn_param.ung_and_acorn')
    NEW_EDGE_POLICY=$(echo "$experiment" | jq -r '.ung_and_acorn_param.new_edge_policy')
    R_IN_ADD_NEW_EDGE=$(echo "$experiment" | jq -r '.ung_and_acorn_param.R_in_add_new_edge')
    W_IN_ADD_NEW_EDGE=$(echo "$experiment" | jq -r '.ung_and_acorn_param.W_in_add_new_edge')
    M_IN_ADD_NEW_EDGE=$(echo "$experiment" | jq -r '.ung_and_acorn_param.M_in_add_new_edge')
    LAYER_DEPTH_RETIO=$(echo "$experiment" | jq -r '.ung_and_acorn_param.layer_depth_retio')
    QUERY_VECTOR_RATIO=$(echo "$experiment" | jq -r '.ung_and_acorn_param.query_vector_ratio')
    ROOT_COVERAGE_THRESHOLD=$(echo "$experiment" | jq -r '.ung_and_acorn_param.root_coverage_threshold')
    M=$(echo "$experiment" | jq -r '.ung_and_acorn_param.acorn_in_ung.M')
    M_BETA=$(echo "$experiment" | jq -r '.ung_and_acorn_param.acorn_in_ung.M_beta')
    GAMMA=$(echo "$experiment" | jq -r '.ung_and_acorn_param.acorn_in_ung.gamma')
    EFS=$(echo "$experiment" | jq -r '.ung_and_acorn_param.acorn_in_ung.efs')
    COMPUTE_RECALL=$(echo "$experiment" | jq -r '.ung_and_acorn_param.acorn_in_ung.compute_recall')
    # 【修正】添加缺失的 generate_query_task 和 method1_high_coverage_p 参数
    GENERATE_QUERY_TASK=$(echo "$experiment" | jq -r '.generate_query_task')
    METHOD1_HIGH_COVERAGE_P=$(echo "$experiment" | jq -r '.method1_high_coverage_p')

    # - 搜索 & GT 参数
    K=$(echo "$experiment" | jq -r '.K')
    LSEARCH_START=$(echo "$experiment" | jq -r '.Lsearch_start')
    LSEARCH_END=$(echo "$experiment" | jq -r '.Lsearch_end')
    LSEARCH_STEP=$(echo "$experiment" | jq -r '.Lsearch_step')
    NUM_THREADS=$(echo "$experiment" | jq -r '.num_threads')
    NUM_REPEATS=$(echo "$experiment" | jq -r '.num_repeats')
    IS_NEW_TRIE_METHOD=$(echo "$experiment" | jq -r '.trie_param.is_new_trie_method')
    IS_REC_MORE_START=$(echo "$experiment" | jq -r '.trie_param.is_rec_more_start')
    IS_UNG_MORE_ENTRY=$(echo "$experiment" | jq -r '.ung_more_entry_param.is_ung_more_entry')

    # --- 【步骤2】调用 build.sh ---
    # build.sh 内部会根据构建参数生成唯一的目录名，并检查是否存在
    echo "准备索引..."
    ./build.sh \
        --dataset "$DATASET" --data_dir "$DATA_DIR" --output_dir "$OUTPUT_DIR" --build_dir "$TMPDIR" \
        --num_query_sets "$NUM_QUERY_SETS" --max_degree "$MAX_DEGREE" --Lbuild "$LBUILD" --alpha "$ALPHA" \
        --num_cross_edges "$NUM_CROSS_EDGES" --num_entry_points "$NUM_ENTRY_POINTS" \
        --ung_and_acorn "$UNG_AND_ACORN" --new_edge_policy "$NEW_EDGE_POLICY" \
        --R_in_add_new_edge "$R_IN_ADD_NEW_EDGE" --W_in_add_new_edge "$W_IN_ADD_NEW_EDGE" --M_in_add_new_edge "$M_IN_ADD_NEW_EDGE" \
        --layer_depth_retio "$LAYER_DEPTH_RETIO" --query_vector_ratio "$QUERY_VECTOR_RATIO" --root_coverage_threshold "$ROOT_COVERAGE_THRESHOLD" \
        --M "$M" --M_beta "$M_BETA" --gamma "$GAMMA" --efs "$EFS" --compute_recall "$COMPUTE_RECALL" \
        --generate_query_task "$GENERATE_QUERY_TASK" --method1_high_coverage_p "$METHOD1_HIGH_COVERAGE_P"

    # --- 【步骤3】调用 generate_gt.sh ---
    echo "准备 Ground Truth (Q=${NUM_QUERY_SETS}, K=$K)..."
    ./generate_gt.sh \
        --dataset "$DATASET" --data_dir "$DATA_DIR" --output_dir "$OUTPUT_DIR" --build_dir "$TMPDIR" \
        --num_query_sets "$NUM_QUERY_SETS" --K "$K"

    # --- 【步骤4】调用 search.sh ---
    # 需要预先计算出索引目录名，以传递给search.sh
    INDEX_DIR_NAME="Q${NUM_QUERY_SETS}_M${MAX_DEGREE}_LB${LBUILD}_alpha${ALPHA}_C${NUM_CROSS_EDGES}_EP${NUM_ENTRY_POINTS}"
    echo "开始搜索 (K=$K)..."
    ./search.sh \
        --dataset "$DATASET" --data_dir "$DATA_DIR" --output_dir "$OUTPUT_DIR" --build_dir "$TMPDIR" \
        --index_dir_name "$INDEX_DIR_NAME" \
        --num_query_sets "$NUM_QUERY_SETS" --num_entry_points "$NUM_ENTRY_POINTS" \
        --Lsearch_start "$LSEARCH_START" --Lsearch_end "$LSEARCH_END" --Lsearch_step "$LSEARCH_STEP" \
        --num_threads "$NUM_THREADS" --K "$K" --num_repeats "$NUM_REPEATS" \
        --is_new_trie_method "$IS_NEW_TRIE_METHOD" --is_rec_more_start "$IS_REC_MORE_START" --is_ung_more_entry "$IS_UNG_MORE_ENTRY"

    echo "--- 当前实验配置处理完成 ---"
done

echo -e "\n所有实验已完成！"

