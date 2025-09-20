import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mticker

print("开始生成按QuerySize划分的QPS-Recall图表...")

### 1. 配置输入文件路径 ###
USE_PRE_SELECTED_QUERIES = False
PRE_SELECTED_QUERIES_FILE = '/data/fxy/FilterVector/FilterVectorResults/merge_results/experiments/celeba/query3/selected_queries_by_ratio_acorn-slow.csv'


# ACORN 详细结果文件 (每个查询的详细数据)
acorn_details_file = '/data/fxy/FilterVector/FilterVectorResults/ACORN/celeba/Results/index_N202599_M32_gamma80_Mb64/query3_threads10_k10_repeat1_ifbfstrue_efs10-5-205/results/celeba_query3_M32_gamma80_threads10_repeat1_ifbfs0_efs10-205_5.csv'
# ACORN 平均结果文件 (包含 FilterMapTime 等)
acorn_avg_file = '/data/fxy/FilterVector/FilterVectorResults/ACORN/celeba/Results/index_N202599_M32_gamma80_Mb64/query3_threads10_k10_repeat1_ifbfstrue_efs10-5-205/results/avg_celeba_query3_M32_gamma80_threads10_repeat1_ifbfs0_efs10-205_5.csv'

# --- UNG 文件路径 ---
# UNG (nT=false, 原始Trie树方法) 详细结果
ung_nt_false_details_file = '/data/fxy/FilterVector/FilterVectorResults/UNG/celeba/Results/Index[Q3_M32_LB100_alpha1.2_C6_EP16]_GT[Q3_K10]_Search[Ls1000-Le40000-Lp1000_K10_nTfalse_rmsfalse_th10]/results/query_details_repeat1.csv'
# UNG (nT=true, 新Trie树方法) 详细结果
ung_nt_true_details_file = '/data/fxy/FilterVector/FilterVectorResults/UNG/celeba/Results/Index[Q3_M32_LB100_alpha1.2_C6_EP16]_GT[Q3_K10]_Search[Ls2000-Le60000-Lp2000_K10_nTtrue_rmstrue_th10]/results/query_details_repeat1.csv'
# UNG 总结文件 (包含新方法计算Bitmap的时间)
ung_summary_file = '/data/fxy/FilterVector/FilterVectorResults/UNG/celeba/Results/Index[Q3_M32_LB100_alpha1.2_C6_EP16]_GT[Q3_K10]_Search[Ls1000-Le40000-Lp1000_K10_nTfalse_rmsfalse_th10]/results/search_time_summary.csv'

# --- 输出文件路径 ---
# 修改输出文件名以反映其内容
output_plot_path = '/data/fxy/FilterVector/FilterVectorResults/merge_results/experiments/celeba/query3/qps_recall_curve_by_querysize.png'
output_dir = '/data/fxy/FilterVector/FilterVectorResults/merge_results/experiments/celeba/query3'


def get_bitmap_time_from_summary(file_path):
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if 'Bitmap_Computation_Time_ms' in line:
                    parts = line.strip().split(',')
                    if len(parts) > 1:
                        return float(parts[1])
            return 0.0
    except (FileNotFoundError, Exception):
        return 0.0

### 2. 加载并预处理数据 ###
print("\n[步骤 1/7] 正在加载数据文件...")
try:
    df_acorn = pd.read_csv(acorn_details_file)
    df_acorn_avg = pd.read_csv(acorn_avg_file)
    df_ung_false = pd.read_csv(ung_nt_false_details_file)
    df_ung_true = pd.read_csv(ung_nt_true_details_file)
    ung_bitmap_total_time = get_bitmap_time_from_summary(ung_summary_file)
except FileNotFoundError as e:
    print(f"错误：找不到文件 {e.filename}。请检查路径配置。")
    exit()

# 参数映射
acorn_params = sorted(df_acorn['efs'].unique())
ung_params = sorted(df_ung_false['Lsearch'].unique())
if len(acorn_params) != len(ung_params):
    print("错误：ACORN和UNG的搜索参数个数不一致，无法按位置匹配。")
    exit()
param_map = pd.DataFrame({'efs': acorn_params, 'Lsearch': ung_params})
df_ung_false.rename(columns={'Lsearch': 'search_param'}, inplace=True)
df_ung_true.rename(columns={'Lsearch': 'search_param'}, inplace=True)
df_acorn = pd.merge(df_acorn, param_map, on='efs')
df_acorn.drop(columns=['efs'], inplace=True)
df_acorn.rename(columns={'Lsearch': 'search_param'}, inplace=True)
print("数据加载与参数映射完成。")

### 3. 筛选查询任务 ###
print("\n[步骤 2/7] 正在筛选查询ID...")
if USE_PRE_SELECTED_QUERIES:
    try:
        selected_df = pd.read_csv(PRE_SELECTED_QUERIES_FILE)
        selected_query_ids = selected_df['QueryID'].unique().tolist()
        num_selected_queries = len(selected_query_ids)
        print(f"成功从 {os.path.basename(PRE_SELECTED_QUERIES_FILE)} 文件中加载了 {num_selected_queries} 个查询ID。")
    except FileNotFoundError:
        print(f"错误: 预选查询文件未找到 -> {PRE_SELECTED_QUERIES_FILE}")
        exit()
else:
    queries_len_2 = df_ung_false[df_ung_false['QuerySize'] == 2]['QueryID'].unique()[:1000]
    queries_len_5 = df_ung_false[df_ung_false['QuerySize'] == 5]['QueryID'].unique()[:1000]
    queries_len_8 = df_ung_false[df_ung_false['QuerySize'] == 8]['QueryID'].unique()[:1000]
    selected_query_ids = np.concatenate([queries_len_2, queries_len_5, queries_len_8])
    num_selected_queries = len(selected_query_ids)
    print(f"成功选择了 {num_selected_queries} 个查询ID用于分析。")

# 根据选中的ID初步筛选数据
df_acorn = df_acorn[df_acorn['QueryID'].isin(selected_query_ids)]
df_ung_false = df_ung_false[df_ung_false['QueryID'].isin(selected_query_ids)]
df_ung_true = df_ung_true[df_ung_true['QueryID'].isin(selected_query_ids)]

### 4. 合并总数据 ###
print("\n[步骤 3/7] 正在合并所有算法的初步筛选数据...")
merged_df_all = pd.merge(df_acorn, df_ung_false, on=['QueryID', 'search_param'], suffixes=('_acorn', '_ung_false'))
merged_df_all = pd.merge(merged_df_all, df_ung_true, on=['QueryID', 'search_param'], suffixes=('', '_ung_true'))

# --- 创建并排的三个子图 ---
fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True) # 1行3列, 共享Y轴
fig.suptitle('QPS-Recall Analysis by Query Size', fontsize=18)
query_sizes_to_plot = [2, 5, 8]

# 用于存储图例元素
handles, labels = None, None

### 5. 循环处理每个Query Size并绘图 ###
print("\n[步骤 4/7] 开始循环处理每个Query Size...")
for i, q_size in enumerate(query_sizes_to_plot):
    ax = axes[i] # 获取当前子图的坐标轴
    print(f"\n---------- 开始处理 QuerySize = {q_size} ----------")

    # [步骤 5/7] 从已合并的数据中，根据QuerySize筛选出当前子集
    merged_df = merged_df_all[merged_df_all['QuerySize'] == q_size].copy()

    actual_queries_count = merged_df['QueryID'].nunique()
    if actual_queries_count == 0:
        print(f"警告: QuerySize={q_size} 没有有效的查询数据，跳过绘图。")
        ax.text(0.5, 0.5, f'No data for QuerySize = {q_size}', ha='center', va='center')
        ax.set_title(f'QuerySize = {q_size}')
        continue
    
    print(f"QuerySize={q_size}: 找到 {actual_queries_count} 个有效查询。")
    
    # 计算时间
    num_total_queries = merged_df['QueryID'].nunique() # 使用当前子集的查询总数
    acorn_bitmap_time = df_acorn_avg['FilterMapTime_ms'].iloc[0] / num_total_queries
    ung_bitmap_time = ung_bitmap_total_time / num_total_queries if num_total_queries > 0 else 0.0

    merged_df['Time_ACORN-1'] = merged_df['acorn_1_Time_ms'] + acorn_bitmap_time
    merged_df['Time_ACORN-gamma'] = merged_df['acorn_Time_ms'] + acorn_bitmap_time
    merged_df['Time_UNG'] = merged_df['SearchT_ms']
    merged_df['Time_Method1'] = np.minimum(merged_df['SearchT_ms'], merged_df['SearchT_ms_ung_true'])
    term_m2_acorn_part = merged_df['acorn_Time_ms'] + ung_bitmap_time
    merged_df['Time_Method2'] = np.minimum(term_m2_acorn_part, merged_df['SearchT_ms']) + merged_df['FlagT_ms']
    min_search_val = np.minimum.reduce([term_m2_acorn_part, merged_df['SearchT_ms'], merged_df['SearchT_ms_ung_true']])
    merged_df['Time_Method3'] = min_search_val + np.where(min_search_val == merged_df['SearchT_ms_ung_true'], merged_df['FlagT_ms_ung_true'], merged_df['FlagT_ms'])

    # Recall计算
    merged_df['Recall_ACORN-1'] = merged_df['acorn_1_Recall']
    merged_df['Recall_ACORN-gamma'] = merged_df['acorn_Recall']
    merged_df['Recall_UNG'] = merged_df['Recall']
    merged_df['Recall_Method1'] = np.where(merged_df['EntryGroupT_ms'] <= merged_df['EntryGroupT_ms_ung_true'], merged_df['Recall'], merged_df['Recall_ung_true'])
    merged_df['Recall_Method2'] = np.where(term_m2_acorn_part <= merged_df['SearchT_ms'], merged_df['acorn_Recall'], merged_df['Recall'])
    merged_df['Recall_Method3'] = np.where(min_search_val == term_m2_acorn_part, merged_df['acorn_Recall'], np.where(min_search_val == merged_df['SearchT_ms'], merged_df['Recall'], merged_df['Recall_ung_true']))
    
    # [步骤 6/7] 聚合数据并保存分析文件
    grouped = merged_df.groupby('search_param')
    # 为每个query size保存独立的分析文件
    output_analysis_csv = os.path.join(output_dir, f'analysis_results_qsize_{q_size}.csv')
    output_analysis_txt = os.path.join(output_dir, f'analysis_results_qsize_{q_size}.txt')
    try:
        analysis_df = grouped[['Time_ACORN-1', 'Time_ACORN-gamma', 'Time_UNG', 'Time_Method1', 'Time_Method2', 'Time_Method3']].mean()
        analysis_df.to_csv(output_analysis_csv)
        print(f"详细分析数据已保存到: {os.path.abspath(output_analysis_csv)}")
        with open(output_analysis_txt, 'w') as f:
            f.write(f"QPS-Recall 详细耗时分析 (QuerySize = {q_size})\n")
            f.write(analysis_df.to_string())
    except KeyError as e:
        print(f"发生列名错误: {e}。")

    # [步骤 7/7] 计算绘图数据并绘图
    plot_data = {}
    algorithms = ['ACORN-1', 'ACORN-gamma', 'UNG', 'Method1', 'Method2', 'Method3']
    for alg in algorithms:
        avg_time = grouped[f'Time_{alg}'].mean()
        avg_recall = grouped[f'Recall_{alg}'].mean()
        qps = 1 / (avg_time / 1000.0)
        df_plot = pd.DataFrame({'Recall': avg_recall, 'QPS': qps, 'Avg_Time_ms': avg_time}).sort_values(by='Recall').reset_index()
        
        first_reach_one_idx = df_plot.index[df_plot['Recall'] >= 0.99999].tolist()
        end_idx = first_reach_one_idx[0] if first_reach_one_idx else df_plot['Recall'].idxmax()
        plot_data[alg] = df_plot.iloc[:end_idx + 1]

    # 在当前子图(ax)上绘图
    markers = ['o', 's', '^', 'D', 'v', 'p']
    for j, (alg, data) in enumerate(plot_data.items()):
        ax.plot(data['Recall'], data['QPS'], marker=markers[j], linestyle='-', label=alg)

    ax.set_title(f'QuerySize = {q_size} ({actual_queries_count} Queries)', fontsize=14)
    ax.set_xlabel('Recall@10', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
   #  ax.legend(fontsize=10)
    ax.set_xlim(0.7, 1.0)
    
    all_qps_values = pd.concat([data['QPS'] for data in plot_data.values()]).dropna()
    if not all_qps_values.empty:
        y_max = all_qps_values.max()
        ax.set_ylim(bottom=0, top=y_max * 1.05)
    
    locator = mticker.MaxNLocator(nbins='auto', steps=[1, 2, 5, 10], integer=True)
    ax.yaxis.set_major_locator(locator)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_useOffset(False)

    if i == 0:
        handles, labels = ax.get_legend_handles_labels()

# 设置共享的Y轴标签
axes[0].set_ylabel('QPS', fontsize=12)

### 6. 调整布局并保存总图表 ###
# 在图表顶部添加一个统一的、横向的图例
# loc='upper center' - 定位在顶部中间;bbox_to_anchor - 将图例的定位点放在图表的(x=0.5, y=0.97)位置，即主标题下方;ncol=6 - 将6个图例项分为6列，实现横向排列
if handles:
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=6, fontsize=12)

# 调整布局，为顶部的主标题和图例留出空间 (rect的最后一个参数top从0.95调整为0.9)
plt.tight_layout(rect=[0, 0, 0.95, 0.95])

plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
print(f"\n图表绘制完成！已保存到：{os.path.abspath(output_plot_path)}")