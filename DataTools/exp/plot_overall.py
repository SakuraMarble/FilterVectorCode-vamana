import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mticker

print("开始生成QPS-Recall图表...")

### 1. 配置输入文件路径 ###
USE_PRE_SELECTED_QUERIES = True
PRE_SELECTED_QUERIES_FILE = '/data/fxy/FilterVector/FilterVectorResults/merge_results/experiments/celeba/query4/selected_queries_by_ratio_acorn-slow.csv'


# ACORN 详细结果文件 (每个查询的详细数据)
acorn_details_file = '/data/fxy/FilterVector/FilterVectorResults/ACORN/celeba/Results/index_N202599_M32_gamma80_Mb64/query4_threads10_k10_repeat1_ifbfstrue_efs10-5-205/results/celeba_query4_M32_gamma80_threads10_repeat1_ifbfs0_efs10-205_5.csv'
# ACORN 平均结果文件 (包含 FilterMapTime 等)
acorn_avg_file = '/data/fxy/FilterVector/FilterVectorResults/ACORN/celeba/Results/index_N202599_M32_gamma80_Mb64/query4_threads10_k10_repeat1_ifbfstrue_efs10-5-205/results/avg_celeba_query4_M32_gamma80_threads10_repeat1_ifbfs0_efs10-205_5.csv'

# --- UNG 文件路径 ---
# UNG (nT=false, 原始Trie树方法) 详细结果
ung_nt_false_details_file = '/data/fxy/FilterVector/FilterVectorResults/UNG/celeba/Results/Index[Q4_M32_LB100_alpha1.2_C6_EP16]_GT[Q4_K10]_Search[Ls1000-Le40000-Lp1000_K10_nTfalse_rmsfalse_th10]/results/query_details_repeat1.csv'
# UNG (nT=true, 新Trie树方法) 详细结果
ung_nt_true_details_file = '/data/fxy/FilterVector/FilterVectorResults/UNG/celeba/Results/Index[Q4_M32_LB100_alpha1.2_C6_EP16]_GT[Q4_K10]_Search[Ls2000-Le60000-Lp2000_K10_nTtrue_rmstrue_th10]/results/query_details_repeat1.csv'
# UNG 总结文件 (包含新方法计算Bitmap的时间)
ung_summary_file = '/data/fxy/FilterVector/FilterVectorResults/UNG/celeba/Results/Index[Q4_M32_LB100_alpha1.2_C6_EP16]_GT[Q4_K10]_Search[Ls1000-Le40000-Lp1000_K10_nTfalse_rmsfalse_th10]/results/search_time_summary.csv'
# --- 输出文件路径 ---
output_plot_path = '/data/fxy/FilterVector/FilterVectorResults/merge_results/experiments/celeba/query4/qps_recall_curve.png'
output_analysis_csv = '/data/fxy/FilterVector/FilterVectorResults/merge_results/experiments/celeba/query4/analysis_results.csv'
output_analysis_txt = '/data/fxy/FilterVector/FilterVectorResults/merge_results/experiments/celeba/query4/analysis_results.txt'


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
print("\n[步骤 1/6] 正在加载数据文件...")
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
if USE_PRE_SELECTED_QUERIES:
    print(f"\n[步骤 2/6] 检测到 USE_PRE_SELECTED_QUERIES=True，将从文件加载查询ID...")
    try:
        selected_df = pd.read_csv(PRE_SELECTED_QUERIES_FILE)
        selected_query_ids = selected_df['QueryID'].unique().tolist()
        num_selected_queries = len(selected_query_ids)
        print(f"成功从 {os.path.basename(PRE_SELECTED_QUERIES_FILE)} 文件中加载了 {num_selected_queries} 个查询ID。")
    except FileNotFoundError:
        print(f"错误: 预选查询文件未找到 -> {PRE_SELECTED_QUERIES_FILE}")
        print("请确保文件路径正确，或先运行 select_queries.py 生成该文件。")
        exit()
else:
    print("\n[步骤 2/6] USE_PRE_SELECTED_QUERIES=False，正在根据查询大小筛选前1000个查询ID...")
    queries_len_2 = df_ung_false[df_ung_false['QuerySize'] == 2]['QueryID'].unique()[:1000]
    queries_len_5 = df_ung_false[df_ung_false['QuerySize'] == 5]['QueryID'].unique()[:1000]
    queries_len_8 = df_ung_false[df_ung_false['QuerySize'] == 8]['QueryID'].unique()[:1000]
    selected_query_ids = np.concatenate([queries_len_2, queries_len_5, queries_len_8])
    num_selected_queries = len(selected_query_ids)
    print(f"成功选择了 {num_selected_queries} 个查询ID用于分析。")

df_acorn = df_acorn[df_acorn['QueryID'].isin(selected_query_ids)]
df_ung_false = df_ung_false[df_ung_false['QueryID'].isin(selected_query_ids)]
df_ung_true = df_ung_true[df_ung_true['QueryID'].isin(selected_query_ids)]


### 4. 合并并计算时间 ###
print("\n[步骤 3/6] 正在合并不同算法的数据并计算总时间...")
merged_df = pd.merge(df_acorn, df_ung_false, on=['QueryID', 'search_param'], suffixes=('_acorn', '_ung_false'))
merged_df = pd.merge(merged_df, df_ung_true, on=['QueryID', 'search_param'], suffixes=('', '_ung_true'))

actual_queries_count = merged_df['QueryID'].nunique()
print(f"数据合并完成。")
print(f"监控: 初始选择了 {num_selected_queries} 个查询ID。")
print(f"监控: 经过数据对齐和合并后，实际参与计算的有效查询数量为: {actual_queries_count} 个。")
if actual_queries_count < num_selected_queries:
    print(f"   -> 注意: 有 {num_selected_queries - actual_queries_count} 个查询ID因在某些数据文件中缺失，未被纳入最终计算。")
elif actual_queries_count == 0:
    print("错误: 没有有效的查询数据可以进行分析，请检查输入文件和筛选条件。")
    exit()

num_total_queries = df_acorn['QueryID'].nunique()
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
print("总时间与Recall计算完成。")


### 5. 按search_param分组聚合，并将详细分析输出到文件 ###
print("\n[步骤 4/6] 正在聚合数据，并将详细分析结果输出到文件...")
grouped = merged_df.groupby('search_param')
try:
    analysis_df = grouped[[
        'Time_ACORN-1', 'Time_ACORN-gamma', 'Time_UNG', 'Time_Method1', 'Time_Method2', 'Time_Method3',
        'acorn_1_Time_ms', 'acorn_Time_ms', 'SearchT_ms', 'SearchT_ms_ung_true',
        'FlagT_ms', 'FlagT_ms_ung_true'
    ]].mean()
    analysis_df['acorn_bitmap_avg'] = acorn_bitmap_time
    analysis_df['ung_bitmap_avg'] = ung_bitmap_time
    analysis_df['Diff1(UNG-M1)'] = analysis_df['Time_UNG'] - analysis_df['Time_Method1']
    analysis_df['Diff2(M2-M3)'] = analysis_df['Time_Method2'] - analysis_df['Time_Method3']
    column_order = [
        'Time_ACORN-1', 'Time_ACORN-gamma', 'Time_UNG', 'Time_Method1', 'Time_Method2', 'Time_Method3',
        'Diff1(UNG-M1)', 'Diff2(M2-M3)',
        'acorn_bitmap_avg', 'ung_bitmap_avg',
        'acorn_1_Time_ms', 'acorn_Time_ms', 'SearchT_ms', 'SearchT_ms_ung_true',
        'FlagT_ms', 'FlagT_ms_ung_true'
    ]
    analysis_df = analysis_df[column_order]
    pd.options.display.float_format = '{:,.2f}'.format
    analysis_df.to_csv(output_analysis_csv)
    print(f"详细分析数据已保存到CSV文件: {os.path.abspath(output_analysis_csv)}")
    with open(output_analysis_txt, 'w') as f:
        f.write("="*80 + "\n")
        f.write("QPS-Recall 详细耗时分析 (单位: ms)\n")
        f.write("="*80 + "\n")
        f.write(analysis_df.to_string())
    print(f"格式化分析数据已保存到文本文件: {os.path.abspath(output_analysis_txt)}")
except KeyError as e:
    print(f"发生列名错误: {e}。请检查merged_df中是否存在所有需要的列名。")


### 6. 计算绘图数据 ###
print("\n[步骤 5/6] 正在为绘图准备QPS-Recall数据...")
plot_data = {}
algorithms = ['ACORN-1', 'ACORN-gamma', 'UNG', 'Method1', 'Method2', 'Method3']
for alg in algorithms:
    # avg_time 是处理单次查询的平均耗时（单位：ms）
    avg_time = grouped[f'Time_{alg}'].mean()
    avg_recall = grouped[f'Recall_{alg}'].mean()
    
    # QPS = 1 / (单次查询的平均秒数)
    qps = 1 / (avg_time / 1000.0)
    
    # 将 search_param 也加入，方便后续查找
    # THIS IS THE FIX: Ensure 'Avg_Time_ms' is added to the DataFrame.
    df = pd.DataFrame({'Recall': avg_recall, 'QPS': qps, 'Avg_Time_ms': avg_time}).sort_values(by='Recall').reset_index()
    
    print(f"\n----- 算法 '{alg}' 的QPS计算过程 -----")
    qps_monitoring_df = pd.DataFrame({
        'Search_Param': avg_time.index,
        'Avg_Single_Query_Time_ms': avg_time.values,
        'Avg_Recall': avg_recall.values,
        'Calculated_QPS': qps.values
    })
    qps_monitoring_df['Formula'] = qps_monitoring_df.apply(
        lambda row: f"1 / ({row['Avg_Single_Query_Time_ms']:.2f} / 1000)", axis=1
    )
    print(f"数据基于 {actual_queries_count} 次独立查询的平均值计算。")
    print(qps_monitoring_df.to_string(index=False, float_format="%.2f"))
    
    first_reach_one_idx = df.index[df['Recall'] >= 0.99999].tolist()
    if first_reach_one_idx:
        end_idx = first_reach_one_idx[0]
    else:
        end_idx = df['Recall'].idxmax()
        
    plot_data[alg] = df.iloc[:end_idx + 1]


# 在绘图前，打印指定线条的第一个点坐标及相关信息
print("\n" + "="*80)
print("指定线条的第一个点详细信息:")
print("="*80)
lines_to_print = {
    "UNG (绿色)": "UNG",
    "Method1 (红色)": "Method1",
    "Method2 (紫色)": "Method2",
    "Method3 (棕色)": "Method3"
}

for line_name, alg_name in lines_to_print.items():
    if alg_name in plot_data:
        first_point = plot_data[alg_name].iloc[0]
        recall = first_point['Recall']
        qps = first_point['QPS']
        avg_time = first_point['Avg_Time_ms']
        search_param = first_point['search_param']
        print(f"{line_name}:")
        print(f"  - 坐标 (Recall, QPS): ({recall:.4f}, {qps:.2f})")
        print(f"  - 平均耗时: {avg_time:.2f} ms")
        print(f"  - 对应 Search_Param: {search_param}")
print("="*80 + "\n")


### 7. 绘图 ###
print("\n[步骤 6/6] 正在绘制图表...")
plt.figure(figsize=(12, 8))
plt.rcParams['axes.unicode_minus'] = False
markers = ['o', 's', '^', 'D', 'v', 'p']
for i, (alg, data) in enumerate(plot_data.items()):
    plt.plot(data['Recall'], data['QPS'], marker=markers[i], linestyle='-', label=alg)

plt.title(f'QPS-Recall ({actual_queries_count} Queries)', fontsize=16)
plt.xlabel('Recall@10', fontsize=12)
plt.ylabel('QPS', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=10)
#plt.yscale('log')
plt.xlim(0.7, 1.0)
# --- MODIFICATION START ---
# Y轴改为线性刻度，并自动寻找最优刻度间距

# 获取当前坐标轴
ax = plt.gca()

# 从所有绘图数据中计算Y轴的范围
all_qps_values = pd.concat([data['QPS'] for data in plot_data.values()]).dropna()
if not all_qps_values.empty:
    y_max = all_qps_values.max()

    # 设置Y轴的上下限，强制最低点为0，并增加一点上边距
    ax.set_ylim(bottom=0, top=y_max * 1.05) # 顶部增加5%的边距
    
    # 使用 MaxNLocator 自动寻找“整齐”的刻度值，并保证间距合理
    # steps=[1, 2, 5, 10] 使得刻度值倾向于以10, 20, 50, 100等结尾,integer=True 保证刻度为整数
    locator = mticker.MaxNLocator(nbins='auto', steps=[1, 2, 5, 10], integer=True)
    ax.yaxis.set_major_locator(locator)

# 设置刻度格式为普通数字
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
# 确保格式化工具不使用全局偏移量 (例如，在坐标轴顶部显示 x10^3)
ax.yaxis.get_major_formatter().set_useOffset(False)
# --- MODIFICATION END ---

plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
print(f"\n图表绘制完成！已保存到：{os.path.abspath(output_plot_path)}")

