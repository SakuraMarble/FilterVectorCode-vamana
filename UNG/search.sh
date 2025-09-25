#!/bin/bash

# ==============================================================================
# search.sh - 负责在已有的索引和GT上执行搜索任务
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
        echo "未知参数: $1"; exit 1
    fi
done

# --- Step 2: 根据【搜索参数】和【依赖】构造唯一的结果输出目录 ---
GT_DIR_NAME="Q${NUM_QUERY_SETS}_K${K}"
SEARCH_DIR_NAME="Ls${LSEARCH_START}-Le${LSEARCH_END}-Lp${LSEARCH_STEP}_K${K}_nT${IS_NEW_TRIE_METHOD}_rms${IS_REC_MORE_START}_th${NUM_THREADS}"
RESULT_OUTPUT_DIR="${OUTPUT_DIR}/UNG/${DATASET}/Results/Index[${INDEX_DIR_NAME}]_GT[${GT_DIR_NAME}]_Search[${SEARCH_DIR_NAME}]"

# --- Step 3: 创建结果目录 ---
mkdir -p "$RESULT_OUTPUT_DIR/results"
mkdir -p "$RESULT_OUTPUT_DIR/others"

# --- Step 4: 准备Lsearch参数序列 ---
LSEARCH_VALUES=$(seq "$LSEARCH_START" "$LSEARCH_STEP" "$LSEARCH_END" | tr '\n' ' ')
echo "将在以下Lsearch值上进行测试: $LSEARCH_VALUES"

# --- Step 5: 定义依赖文件和目录的路径 ---
INDEX_PATH="${OUTPUT_DIR}/UNG/${DATASET}/Index/${INDEX_DIR_NAME}"
GT_PATH="${OUTPUT_DIR}/UNG/${DATASET}/GroundTruth/${GT_DIR_NAME}"
QUERY_DIR="$DATA_DIR/query_${NUM_QUERY_SETS}"

echo "使用索引: $INDEX_PATH"
echo "使用GT: $GT_PATH"
echo "结果将保存到: $RESULT_OUTPUT_DIR"

# --- Step 6: 执行搜索 ---
"$BUILD_DIR"/apps/search_UNG_index \
    --data_type float --dist_fn L2 --num_threads "$NUM_THREADS" --K "$K" --num_repeats "$NUM_REPEATS" \
    --is_new_method true --is_ori_ung true \
    --is_new_trie_method "$IS_NEW_TRIE_METHOD" --is_rec_more_start "$IS_REC_MORE_START" \
    --is_ung_more_entry "$IS_UNG_MORE_ENTRY" \
    --base_bin_file "$DATA_DIR/${DATASET}_base.bin" \
    --query_bin_file "$QUERY_DIR/${DATASET}_query.bin" \
    --query_label_file "$QUERY_DIR/${DATASET}_query_labels.txt" \
    --query_group_id_file "$QUERY_DIR/${DATASET}_query_source_groups.txt" \
    --gt_file "$GT_PATH/${DATASET}_gt_labels_containment.bin" \
    --index_path_prefix "$INDEX_PATH/index_files/" \
    --result_path_prefix "$RESULT_OUTPUT_DIR/results/" \
    --selector_modle_prefix "$INDEX_PATH/SelectModels/" \
    --scenario containment \
    --num_entry_points "$NUM_ENTRY_POINTS" \
    --Lsearch $LSEARCH_VALUES > "$RESULT_OUTPUT_DIR/others/${DATASET}_search_output.txt" 2>&1

echo "搜索完成！"
