#!/bin/bash

# 检查 jq 是否已安装
if ! command -v jq &> /dev/null; then
    echo "错误: 核心依赖 'jq' 未安装。请先安装 jq (https://stedolan.github.io/jq/)" 
    exit 1
fi

# 检查配置文件是否存在
CONFIG_FILE="experiments.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 '$CONFIG_FILE' 不存在。请先创建该文件。"
    exit 1
fi

# 遍历JSON配置文件中的所有实验
cat "$CONFIG_FILE" | jq -c '.experiments[]' | while read -r experiment; do
    # --- 提取参数 ---
    mode=$(echo "$experiment" | jq -r '.mode')
    dataset=$(echo "$experiment" | jq -r '.dataset')
    
    echo -e "\n============================================"
    echo "开始处理任务..."
    echo "数据集: $dataset"
    echo "运行模式: $mode"
    echo "============================================"
    
    N=$(echo "$experiment" | jq -r '.N')
    M=$(echo "$experiment" | jq -r '.M')
    M_beta=$(echo "$experiment" | jq -r '.M_beta')
    gamma=$(echo "$experiment" | jq -r '.gamma')
    query_num=$(echo "$experiment" | jq -r '.query_num')
    threads=$(echo "$experiment" | jq -r '.threads')
    repeat_num=$(echo "$experiment" | jq -r '.repeat_num')    
    if_bfs_filter=$(echo "$experiment" | jq -r '.if_bfs_filter')

    # 从 efs_start, efs_end, efs_step 动态生成 efs_list 字符串
    efs_start=$(echo "$experiment" | jq -r '.efs_start')
    efs_end=$(echo "$experiment" | jq -r '.efs_end')
    efs_step=$(echo "$experiment" | jq -r '.efs_step')
    efs_list=$(seq -s, $efs_start $efs_step $efs_end)

    k=$(echo "$experiment" | jq -r '.k')

    echo "从JSON生成的参数:"
    echo "  N=$N, M=$M, M_beta=$M_beta, gamma=$gamma, query_num=$query_num"
    echo "  threads=$threads, repeat_num=$repeat_num, if_bfs_filter=$if_bfs_filter"
    echo "  生成的 EFS 列表: $efs_list"

    ./run_acorn.sh "$mode" "$dataset" "$N" "$M" "$M_beta" "$gamma" "$query_num" "$threads" "$efs_list" "$efs_start" "$efs_end" "$efs_step" "$repeat_num" "$if_bfs_filter" "$k"
    
    echo -e "\n任务 ($dataset, $mode) 处理完成"
    echo "============================================"
done

echo -e "\n所有实验任务已完成!"
