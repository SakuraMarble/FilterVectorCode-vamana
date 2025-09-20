import pandas as pd
import os
import numpy as np

print("开始根据特定策略挑选查询任务...")

### 1. 配置 ###
# --- 输入文件 ---
full_results_csv = '/data/fxy/FilterVector/FilterVectorResults/merge_results/experiments/celeba/query4/query4_final_comparison_results.csv'

# --- 输出配置 ---
# 选择模式: 'acorn-slow' 或 'acorn-fast'
# 'acorn-slow': 在每个查询长度中，挑选ACORN-gamma耗时/UNG耗时比率最大的1000个查询
# 'acorn-fast': 在每个查询长度中，挑选ACORN-gamma耗时/UNG耗时比率最小的1000个查询
SELECTION_MODE = 'acorn-slow'

# 输出文件路径
output_dir = '/data/fxy/FilterVector/FilterVectorResults/merge_results/experiments/celeba/query4'
output_filename_prefix = 'selected_queries_by_ratio' # 修改了文件名前缀以区分

# --- 筛选参数 ---
# 要处理的查询长度列表
QUERY_SIZES_TO_PROCESS = [2, 5, 8]
# 每个长度要挑选的查询数量
NUM_QUERIES_PER_SIZE = 1000


### 2. 加载数据 ###
try:
    print(f"正在加载数据文件: {full_results_csv}")
    df = pd.read_csv(full_results_csv)
except FileNotFoundError:
    print(f"错误: 输入文件未找到 -> {full_results_csv}")
    print("请确保您已经成功运行了 process_csv.py 并生成了此文件。")
    exit()

print("数据加载成功。")


### 3. 执行挑选 ###
print(f"当前选择模式: {SELECTION_MODE}")
all_selected_ids = []

for size in QUERY_SIZES_TO_PROCESS:
    print(f"\n正在处理查询长度 (QuerySize) = {size}...")

    # 筛选出当前长度的数据
    # 使用 .copy() 避免后续操作出现 SettingWithCopyWarning
    df_size = df[df['1_QueryAttrLength'] == size].copy()

    if df_size.empty:
        print(f"警告: 未找到任何查询长度为 {size} 的数据，跳过。")
        continue


    # 计算ACORN-gamma耗时与UNG耗时的比率
    # 使用 np.divide 来处理 Time_UNG_ms 可能为0的情况, 结果会是 inf, 这在排序时是有效的
    df_size['Ratio_ACORN_UNG'] = np.divide(df_size['Time_ACORN-gamma_ms'], df_size['Time_UNG_ms'])
    
    # 替换无穷大的值为一个非常大的数（或者保持inf），并处理可能出现的NaN
    df_size.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_size.dropna(subset=['Ratio_ACORN_UNG'], inplace=True)


    # 根据模式进行排序
    if SELECTION_MODE == 'acorn-slow':
        # 降序排序，比率最大的在前
        sorted_df = df_size.sort_values(by='Ratio_ACORN_UNG', ascending=False)
        print(f"按 'Time_ACORN-gamma_ms / Time_UNG_ms' 的比率降序排序，选取比率最高的 {NUM_QUERIES_PER_SIZE} 个。")
    elif SELECTION_MODE == 'acorn-fast':
        # 升序排序，比率最小的在前
        sorted_df = df_size.sort_values(by='Ratio_ACORN_UNG', ascending=True)
        print(f"按 'Time_ACORN-gamma_ms / Time_UNG_ms' 的比率升序排序，选取比率最低的 {NUM_QUERIES_PER_SIZE} 个。")
    else:
        print(f"错误: 无效的 SELECTION_MODE: '{SELECTION_MODE}'")
        exit()
    
        
    # 选取前 N 个查询的ID
    selected_queries = sorted_df.head(NUM_QUERIES_PER_SIZE)
    selected_ids = selected_queries['QueryID'].tolist()
    
    print(f"成功为长度 {size} 挑选了 {len(selected_ids)} 个查询ID。")
    all_selected_ids.extend(selected_ids)


### 4. 保存结果 ###
if not all_selected_ids:
    print("\n处理完成，但没有挑选出任何查询ID。请检查输入文件和筛选参数。")
else:
    print(f"\n总共挑选了 {len(all_selected_ids)} 个查询ID。")
    
    # 从原始DataFrame中筛选出所有被选中的查询数据
    final_selected_df = df[df['QueryID'].isin(all_selected_ids)]
    
    # 构建输出文件名
    output_csv_path = os.path.join(output_dir, f"{output_filename_prefix}_{SELECTION_MODE}.csv")
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")
        
    # 保存到新的CSV文件
    final_selected_df.to_csv(output_csv_path, index=False, float_format='%.4f')
    print(f"挑选结果已成功保存到: {os.path.abspath(output_csv_path)}")