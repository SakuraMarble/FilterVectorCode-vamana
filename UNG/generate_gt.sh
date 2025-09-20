#!/bin/bash

# ==============================================================================
# generate_gt.sh - 专门负责生成Ground Truth
# 该脚本的产物由 num_query_sets 和 K 唯一确定。
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

# --- Step 2: 构造GT目录和文件名 ---
GT_DIR_NAME="Q${NUM_QUERY_SETS}_K${K}"
GT_OUTPUT_DIR="${OUTPUT_DIR}/UNG/${DATASET}/GroundTruth/${GT_DIR_NAME}"
GT_FILE_PATH="${GT_OUTPUT_DIR}/${DATASET}_gt_labels_containment.bin"

# --- Step 3: 如果GT已存在，则跳过 ---
if [ -f "$GT_FILE_PATH" ]; then
    echo "Ground Truth '$GT_FILE_PATH' 已存在，跳过生成。"
    exit 0
fi

echo "Ground Truth 不存在，开始生成: $GT_FILE_PATH"
mkdir -p "$GT_OUTPUT_DIR"

# --- Step 4: 确保查询文件为bin格式 ---
QUERY_BIN_FILE="$DATA_DIR/query_${NUM_QUERY_SETS}/${DATASET}_query.bin"
if [ ! -f "$QUERY_BIN_FILE" ]; then
    echo "转换查询文件格式..."
    "$BUILD_DIR"/tools/fvecs_to_bin --data_type float \
        --input_file "$DATA_DIR/query_${NUM_QUERY_SETS}/${DATASET}_query.fvecs" \
        --output_file "$QUERY_BIN_FILE"
fi

# --- Step 5: 执行GT计算 ---
"$BUILD_DIR"/tools/compute_groundtruth \
      --data_type float --dist_fn L2 --scenario containment --K "$K" --num_threads 32 \
      --base_bin_file "$DATA_DIR/${DATASET}_base.bin" \
      --base_label_file "$DATA_DIR/base_${NUM_QUERY_SETS}/${DATASET}_base_labels.txt" \
      --query_bin_file "$QUERY_BIN_FILE" \
      --query_label_file "$DATA_DIR/query_${NUM_QUERY_SETS}/${DATASET}_query_labels.txt" \
      --gt_file "$GT_FILE_PATH"

echo "GT 生成完成！"
