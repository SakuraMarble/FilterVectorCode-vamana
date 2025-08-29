import pandas as pd
import os
import glob
import re
from collections import defaultdict

# ==============================================================================
# --- 配置区 ---
INPUT_DIR = '/data/fxy/FilterVector/FilterVectorResults/merge_results/improve2/UNG_MEP'
# (修改：更新输出文件名以反映新的逻辑)
OUTPUT_FILE = '/data/fxy/FilterVector/FilterVectorResults/merge_results/improve2/UNG_MEP/analysis_summary_recall_filtered.txt'
# ==============================================================================

def analyze_dataframe(df):
    """
    对单个数据集的完整DataFrame进行分析，并返回格式化的结果字符串。
    """
    if df.empty:
        return "数据为空，无法分析。\n"

    # --- 1. 数据清洗和扩充 ---
    df = df.rename(columns={
        'Time(ms)_U': 'Time_ms_U',
        'Time(ms)_AU': 'Time_ms_AU'
    })

    # (修改：增加了 Recall_AU 和 Recall_U 用于筛选)
    numeric_cols = [
        'DistCalcs_AU', 'DistCalcs_U', 
        'Time_ms_AU', 'Time_ms_U',
        'NumNodeVisited_U','NumNodeVisited_AU',
        'Lsearch_AU', 'Lsearch_U',
        'Recall_AU', 'Recall_U'  # 新增
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=numeric_cols, inplace=True)
    df = df[df['Lsearch_U'] != 0]

    # --- 2. 创建筛选条件 (修改：所有筛选均加入 Recall_AU >= Recall_U 条件) ---
    recall_condition = df['Recall_AU'] >= df['Recall_U']
    
   #  df['距离AU<U'] = (df['DistCalcs_AU'] < df['DistCalcs_U']) & recall_condition
   #  df['时间AU<U'] = (df['Time_ms_AU'] < df['Time_ms_U']) & recall_condition
   #  df['遍历节点数AU<U'] = (df['NumNodeVisited_AU'] < df['NumNodeVisited_U']) & recall_condition

    df['距离AU<U'] = recall_condition
    df['时间AU<U'] = recall_condition
    df['遍历节点数AU<U'] = recall_condition

    dist_true_filter = df['距离AU<U']
    time_true_filter = df['时间AU<U']
    node_true_filter = df['遍历节点数AU<U']
    # 'both_true_filter' 会自动继承上面三个条件中已包含的recall条件
    both_true_filter = df['距离AU<U'] & df['时间AU<U']

    # --- 3. 分情况计算 ---
    results = {}

    # (计算逻辑保持不变，但输入的数据已经被新的筛选条件过滤)
    # 情况一：距离AU < U 且 Recall不降低
    dist_true_df = df[dist_true_filter]
    results['dist_true_count'] = len(dist_true_df)
    if not dist_true_df.empty:
        results['dist_true_dist_ratio'] = (dist_true_df['DistCalcs_AU'] / dist_true_df['DistCalcs_U']).mean()
        results['dist_true_time_ratio'] = (dist_true_df['Time_ms_AU'] / dist_true_df['Time_ms_U']).mean()
        results['dist_true_NumNodeVisited_ratio'] = (dist_true_df['NumNodeVisited_AU'] / dist_true_df['NumNodeVisited_U']).mean()
        results['dist_true_Lsearch_ratio'] = (dist_true_df['Lsearch_AU'] / dist_true_df['Lsearch_U']).mean()
    else:
        results.update({
            'dist_true_dist_ratio': 0, 'dist_true_time_ratio': 0,
            'dist_true_NumNodeVisited_ratio': 0, 'dist_true_Lsearch_ratio': 0
        })

    # 情况二：时间AU < U 且 Recall不降低
    time_true_df = df[time_true_filter]
    results['time_true_count'] = len(time_true_df)
    if not time_true_df.empty:
        results['time_true_dist_ratio'] = (time_true_df['DistCalcs_AU'] / time_true_df['DistCalcs_U']).mean()
        results['time_true_time_ratio'] = (time_true_df['Time_ms_AU'] / time_true_df['Time_ms_U']).mean()
        results['time_true_NumNodeVisited_ratio'] = (time_true_df['NumNodeVisited_AU'] / time_true_df['NumNodeVisited_U']).mean()
        results['time_true_Lsearch_ratio'] = (time_true_df['Lsearch_AU'] / time_true_df['Lsearch_U']).mean()
    else:
        results.update({
            'time_true_dist_ratio': 0, 'time_true_time_ratio': 0,
            'time_true_NumNodeVisited_ratio': 0, 'time_true_Lsearch_ratio': 0
        })

    # 情况三：距离和时间均为AU < U 且 Recall不降低
    both_true_df = df[both_true_filter]
    results['both_true_count'] = len(both_true_df)
    if not both_true_df.empty:
        results['both_true_dist_ratio'] = (both_true_df['DistCalcs_AU'] / both_true_df['DistCalcs_U']).mean()
        results['both_true_time_ratio'] = (both_true_df['Time_ms_AU'] / both_true_df['Time_ms_U']).mean()
        results['both_true_NumNodeVisited_ratio'] = (both_true_df['NumNodeVisited_AU'] / both_true_df['NumNodeVisited_U']).mean()
        results['both_true_Lsearch_ratio'] = (both_true_df['Lsearch_AU'] / both_true_df['Lsearch_U']).mean()
    else:
        results.update({
            'both_true_dist_ratio': 0, 'both_true_time_ratio': 0,
            'both_true_NumNodeVisited_ratio': 0, 'both_true_Lsearch_ratio': 0
        })

    # 情况四：遍历节点数AU < U 且 Recall不降低
    node_true_df = df[node_true_filter]
    results['node_true_count'] = len(node_true_df)
    if not node_true_df.empty:
        results['node_true_dist_ratio'] = (node_true_df['DistCalcs_AU'] / node_true_df['DistCalcs_U']).mean()
        results['node_true_time_ratio'] = (node_true_df['Time_ms_AU'] / node_true_df['Time_ms_U']).mean()
        results['node_true_NumNodeVisited_ratio'] = (node_true_df['NumNodeVisited_AU'] / node_true_df['NumNodeVisited_U']).mean()
        results['node_true_Lsearch_ratio'] = (node_true_df['Lsearch_AU'] / node_true_df['Lsearch_U']).mean()
    else:
        results.update({
            'node_true_dist_ratio': 0, 'node_true_time_ratio': 0,
            'node_true_NumNodeVisited_ratio': 0, 'node_true_Lsearch_ratio': 0
        })
        
    # --- 4. 格式化输出字符串 (修改：注释掉所有距离相关的指标) ---
    output_str = (
        f"总行数 (Recall_AU>=Recall_U 的有效行): {len(df)}\n"
        f"--------------------------------------------------\n"
        # f"条件: 距离AU < U & Recall不降低 (共 {results['dist_true_count']} 行)\n"
        # f"  - 平均距离比值 (AU/U): {results['dist_true_dist_ratio']:.4f}\n"
        # f"  - 平均时间比值 (AU/U): {results['dist_true_time_ratio']:.4f}\n"
        # f"  - 平均遍历节点数比值 (AU/U): {results['dist_true_NumNodeVisited_ratio']:.4f}\n"
        # f"  - 平均Lsearch比值 (AU/U): {results['dist_true_Lsearch_ratio']:.4f}\n\n"
        
        f"条件: 时间AU < U & Recall不降低 (共 {results['time_true_count']} 行)\n"
        f"  - 平均距离比值 (AU/U): {results['time_true_dist_ratio']:.4f}\n"
        f"  - 平均时间比值 (AU/U): {results['time_true_time_ratio']:.4f}\n"
        f"  - 平均遍历节点数比值 (AU/U): {results['time_true_NumNodeVisited_ratio']:.4f}\n"
        f"  - 平均Lsearch比值 (AU/U): {results['time_true_Lsearch_ratio']:.4f}\n\n"

      #   f"条件: 距离和时间均为 AU < U & Recall不降低 (共 {results['both_true_count']} 行)\n"
      #   f"  - 平均距离比值 (AU/U): {results['both_true_dist_ratio']:.4f}\n"
      #   f"  - 平均时间比值 (AU/U): {results['both_true_time_ratio']:.4f}\n"
      #   f"  - 平均遍历节点数比值 (AU/U): {results['both_true_NumNodeVisited_ratio']:.4f}\n"
      #   f"  - 平均Lsearch比值 (AU/U): {results['both_true_Lsearch_ratio']:.4f}\n\n"

        f"条件: 遍历节点数 AU < U & Recall不降低 (共 {results['node_true_count']} 行)\n"
        f"  - 平均距离比值 (AU/U): {results['node_true_dist_ratio']:.4f}\n"
        f"  - 平均时间比值 (AU/U): {results['node_true_time_ratio']:.4f}\n"
        f"  - 平均遍历节点数比值 (AU/U): {results['node_true_NumNodeVisited_ratio']:.4f}\n"
        f"  - 平均Lsearch比值 (AU/U): {results['node_true_Lsearch_ratio']:.4f}\n"
    )
    return output_str


def main():
    """
    主函数，用于发现文件、按数据集分组并驱动分析。
    """
    if not os.path.isdir(INPUT_DIR):
        print(f"错误：输入目录不存在 -> {INPUT_DIR}")
        return

    csv_files = glob.glob(os.path.join(INPUT_DIR, '*.csv'))
    if not csv_files:
        print(f"错误：在目录 {INPUT_DIR} 中未找到任何CSV文件。")
        return

    datasets_data = defaultdict(list)
    pattern = re.compile(r"U_UA_(.*?)_query")

    print(f"正在从 {len(csv_files)} 个文件中读取数据...")
    for f_path in csv_files:
        filename = os.path.basename(f_path)
        match = pattern.search(filename)
        
        if match:
            dataset_name = match.group(1)
            try:
                df = pd.read_csv(f_path)
                datasets_data[dataset_name].append(df)
            except Exception as e:
                print(f"警告：读取文件 {filename} 时出错: {e}")
        else:
            print(f"警告：无法从文件名 {filename} 中解析出数据集名称。")

    if not datasets_data:
        print("未能成功解析并加载任何数据集。程序退出。")
        return

    print(f"\n数据读取完成，共找到 {len(datasets_data)} 个数据集。开始分析...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("性能分析报告 (筛选条件: Recall_AU >= Recall_U)\n")
        f.write(f"数据源目录: {INPUT_DIR}\n")
        f.write("=" * 60 + "\n\n")
        
        for i, (dataset_name, df_list) in enumerate(datasets_data.items()):
            print(f"({i+1}/{len(datasets_data)}) 正在分析数据集: {dataset_name}...")
            
            full_df = pd.concat(df_list, ignore_index=True)
            result_text = analyze_dataframe(full_df)
            
            f.write(f"数据集: {dataset_name}\n")
            f.write("=" * 60 + "\n")
            f.write(result_text)
            f.write("\n" + "=" * 60 + "\n\n")

    print(f"\n分析完成！结果已保存到文件: {os.path.abspath(OUTPUT_FILE)}")


if __name__ == '__main__':
    main()