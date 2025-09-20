import pandas as pd
import numpy as np
import os

print("开始处理实验结果文件...")

### 1. 配置输入文件路径 ###
# --- ACORN 文件路径 ---
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

# --- 输出文件 ---
output_csv_path = '/data/fxy/FilterVector/FilterVectorResults/merge_results/experiments/celeba/query4/query4_final_comparison_results.csv'


def get_bitmap_time_from_summary(file_path):
    """
    Robustly reads the Bitmap_Computation_Time_ms from the UNG summary file,
    inspired by the provided reference function.
    """
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if 'Bitmap_Computation_Time_ms' in line:
                    parts = line.strip().split(',')
                    if len(parts) > 1:
                        return float(parts[1])
        print(f"警告: 在文件 {file_path} 中未找到 'Bitmap_Computation_Time_ms'。")
        return 0.0
    except FileNotFoundError:
        print(f"错误：找不到 UNG 总结文件 {file_path}。")
        return 0.0
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return 0.0

### 2. 加载并预处理数据 ###
print("正在加载数据文件...")
try:
    # 加载ACORN数据
    df_acorn_details = pd.read_csv(acorn_details_file)
    df_acorn_avg = pd.read_csv(acorn_avg_file)
    
    # 加载UNG数据 (除了summary file)
    df_ung_false = pd.read_csv(ung_nt_false_details_file)
    df_ung_true = pd.read_csv(ung_nt_true_details_file)
    
    # **FIX**: Call the helper function to read the summary file value
    ung_bitmap_total_time = get_bitmap_time_from_summary(ung_summary_file)

except FileNotFoundError as e:
    print(f"错误：找不到文件 {e.filename}。请检查上面的文件路径配置是否正确。")
    exit()


# 获取查询总数 (以ACORN文件为准)
num_queries = df_acorn_details['QueryID'].nunique()
print(f"检测到 {num_queries} 个查询任务。")

# --- 计算每查询的平均Bitmap耗时 (这些是固定值) ---
# 2. ACORN 原始方法计算 bitmap 的耗时
acorn_bitmap_time = df_acorn_avg['FilterMapTime_ms'].iloc[0] / num_queries
# 3. ACORN 原始方法并行计算 bitmap 的耗时
acorn_para_bitmap_time = df_acorn_avg.get('FilterMapTime_para_ms', pd.Series([0])).iloc[0] / num_queries
# 4. 新方法计算 bitmap 的耗时
ung_bitmap_time = ung_bitmap_total_time / num_queries if num_queries > 0 else 0.0


### 3. 定义辅助函数，用于查找最优点 ###
def find_optimal_point(df_group, recall_col, param_col):
    """为单个查询的数据组，找到Recall首次达到1或最大值的点，并返回该行数据"""
    # 尝试找到第一个Recall >= 1.0的点
    first_reach_one = df_group[df_group[recall_col] >= 0.99999]
    if not first_reach_one.empty:
        # 找到第一个达到1的参数值，返回对应的行
        optimal_param = first_reach_one[param_col].min()
        return df_group[df_group[param_col] == optimal_param].iloc[0]

    # 如果没有达到1，则找到第一个达到最大值的点
    max_recall = df_group[recall_col].max()
    if pd.isna(max_recall): return None # Handle cases with no valid recall
    first_reach_max = df_group[df_group[recall_col] == max_recall]
    optimal_param = first_reach_max[param_col].min()
    return df_group[df_group[param_col] == optimal_param].iloc[0]


### 4. 迭代处理每个查询 ###
print("正在为每个查询计算性能指标...")
results_list = []

for query_id in range(num_queries):
    # 筛选出当前查询ID的所有数据
    q_acorn = df_acorn_details[df_acorn_details['QueryID'] == query_id].copy()
    q_ung_false = df_ung_false[df_ung_false['QueryID'] == query_id].copy()
    q_ung_true = df_ung_true[df_ung_true['QueryID'] == query_id].copy()

    if q_acorn.empty or q_ung_false.empty or q_ung_true.empty:
        print(f"警告: 查询ID {query_id} 的数据不完整，将跳过此查询。")
        continue

    # --- 找到各个方法的最优点 ---
    optimal_acorn = find_optimal_point(q_acorn, 'acorn_Recall', 'efs')
    optimal_acorn1 = find_optimal_point(q_acorn, 'acorn_1_Recall', 'efs')
    optimal_ung_false = find_optimal_point(q_ung_false, 'Recall', 'Lsearch')
    optimal_ung_true = find_optimal_point(q_ung_true, 'Recall', 'Lsearch')
    
    # 如果任何一个最优点没找到（比如数据为空），就跳过这个查询
    if any(opt is None for opt in [optimal_acorn, optimal_acorn1, optimal_ung_false, optimal_ung_true]):
        print(f"警告: 查询ID {query_id} 无法找到所有最优数据点，将跳过此查询。")
        continue

    # --- 提取基础性能数据 ---
    query_attr_length = optimal_ung_false['QuerySize']
    entry_time_orig = optimal_ung_false['EntryGroupT_ms']
    entry_time_new = optimal_ung_true['EntryGroupT_ms']
    
    # --- 计算6种算法在最优点的整体耗时 ---
    acorn_t = optimal_acorn['acorn_Time_ms']
    acorn1_t = optimal_acorn1['acorn_1_Time_ms']
    ung_false_search_t = optimal_ung_false['SearchT_ms']
    ung_true_search_t = optimal_ung_true['SearchT_ms']
    ung_false_flag_t = optimal_ung_false['FlagT_ms']
    ung_true_flag_t = optimal_ung_true['FlagT_ms']

    # ACORN-1
    time_acorn1 = acorn1_t + acorn_bitmap_time
    recall_acorn1 = optimal_acorn1['acorn_1_Recall']

    # ACORN-gamma
    time_acorn_gamma = acorn_t + acorn_bitmap_time
    recall_acorn_gamma = optimal_acorn['acorn_Recall']
    
    # UNG
    time_ung = ung_false_search_t
    recall_ung = optimal_ung_false['Recall']

    # Method1
    if entry_time_orig <= entry_time_new:
        time_method1 = ung_false_search_t + ung_false_flag_t
        flag_t_for_m1_m3 = ung_false_flag_t
    else:
        time_method1 = ung_true_search_t + ung_true_flag_t
        flag_t_for_m1_m3 = ung_true_flag_t
    recall_method1 = max(optimal_ung_false['Recall'], optimal_ung_true['Recall'])

    # Method2
    term_m2_acorn_part = acorn_t + ung_bitmap_time
    if term_m2_acorn_part <= ung_false_search_t:
        time_method2 = term_m2_acorn_part + ung_false_flag_t
        recall_method2 = optimal_acorn['acorn_Recall']
    else:
        time_method2 = ung_false_search_t + ung_false_flag_t
        recall_method2 = optimal_ung_false['Recall']

    # Method3
    min_search_val = min(term_m2_acorn_part, ung_false_search_t, ung_true_search_t)
    if min_search_val == term_m2_acorn_part:
        time_method3 = min_search_val + ung_false_flag_t 
        recall_method3 = optimal_acorn['acorn_Recall']
    elif min_search_val == ung_false_search_t:
        time_method3 = min_search_val + ung_false_flag_t
        recall_method3 = optimal_ung_false['Recall']
    else:
        time_method3 = min_search_val + ung_true_flag_t
        recall_method3 = optimal_ung_true['Recall']

    # --- 收集所有结果 ---
    results_list.append({
        'QueryID': query_id,
        '1_QueryAttrLength': query_attr_length,
        '2_BitmapTime_ACORN_Orig_ms': acorn_bitmap_time,
        '3_BitmapTime_ACORN_Para_ms': acorn_para_bitmap_time,
        '4_BitmapTime_NewMethod_ms': ung_bitmap_time,
        '5_EntryTime_UNG_Orig_ms': entry_time_orig,
        '6_EntryTime_NewMethod_ms': entry_time_new,
        '7_DecisionFlagTime_ms': flag_t_for_m1_m3,
        
        'Time_ACORN-1_ms': time_acorn1,
        'Recall_ACORN-1': recall_acorn1,
        'Time_ACORN-gamma_ms': time_acorn_gamma,
        'Recall_ACORN-gamma': recall_acorn_gamma,
        'Time_UNG_ms': time_ung,
        'Recall_UNG': recall_ung,
        'Time_Method1_ms': time_method1,
        'Recall_Method1': recall_method1,
        'Time_Method2_ms': time_method2,
        'Recall_Method2': recall_method2,
        'Time_Method3_ms': time_method3,
        'Recall_Method3': recall_method3,

        'Ratio_M3/ACORN-1': time_method3 / time_acorn1 if time_acorn1 > 0 else 0,
        'Ratio_M3/ACORN-gamma': time_method3 / time_acorn_gamma if time_acorn_gamma > 0 else 0,
        'Ratio_M3/UNG': time_method3 / time_ung if time_ung > 0 else 0,
        'Ratio_M3/Method1': time_method3 / time_method1 if time_method1 > 0 else 0,
        'Ratio_M3/Method2': time_method3 / time_method2 if time_method2 > 0 else 0,
    })

### 5. 生成最终的CSV文件 ###
if results_list:
    # 确保输出目录存在
    output_dir = os.path.dirname(output_csv_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    final_df = pd.DataFrame(results_list)
    final_df.to_csv(output_csv_path, index=False, float_format='%.4f')
    print(f"\n处理完成！结果已保存到：{os.path.abspath(output_csv_path)}")
else:
    print("\n处理未完成，没有生成任何结果。请检查输入文件和查询ID是否匹配。")