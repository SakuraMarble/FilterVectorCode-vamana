#!/bin/bash

# 检查 jq 是否安装
if ! command -v jq &> /dev/null; then
    echo "错误: jq 未安装。请先安装 jq (https://stedolan.github.io/jq/)" 
    exit 1
fi

# 读取JSON配置
CONFIG_FILE="experiments.json"

# 遍历所有实验
cat "$CONFIG_FILE" | jq -c '.experiments[]' | while read -r experiment; do
    dataset=$(echo "$experiment" | jq -r '.dataset')
    
    echo -e "============================================"
    echo "开始处理数据集: $dataset"
    echo "============================================"
    
    # 提取参数
    N=$(echo "$experiment" | jq -r '.N')
    M=$(echo "$experiment" | jq -r '.M')
    M_beta=$(echo "$experiment" | jq -r '.M_beta')
    gamma=$(echo "$experiment" | jq -r '.gamma')
    query_num=$(echo "$experiment" | jq -r '.query_num')
    threads=$(echo "$experiment" | jq -r '.threads')
    repeat_num=$(echo "$experiment" | jq -r '.repeat_num')  
    if_bfs_filter=$(echo "$experiment" | jq -r '.if_bfs_filter')

    # 新增：读取 efs_start, efs_end, efs_step，并生成 efs_list
    efs_start=$(echo "$experiment" | jq -r '.efs_start')
    efs_end=$(echo "$experiment" | jq -r '.efs_end')
    efs_step=$(echo "$experiment" | jq -r '.efs_step')

    # 使用 seq 生成逗号分隔的字符串
    efs_list=$(seq -s, $efs_start $efs_step $efs_end)

    # 打印确认生成的 efs_list
    echo "efs_list = $efs_list"

    # 运行测试脚本
    ./run_more_efs.sh "$dataset" "$N" "$M" "$M_beta" "$gamma" "$query_num" "$threads" "$efs_list" "$efs_start" "$efs_end" "$efs_step" "$repeat_num" "$if_bfs_filter"
    
    echo "数据集 $dataset 处理完成"
    echo "============================================"
done

echo "所有数据集测试已完成!"