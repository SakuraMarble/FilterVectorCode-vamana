import pandas as pd
import numpy as np
import os
from tqdm import tqdm

tqdm.pandas(desc="Processing Queries")

def get_bitmap_time_from_summary(file_path):
    """
    Robustly reads the Bitmap_Computation_Time_ms from the UNG summary file.
    """
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if 'Bitmap_Computation_Time_ms' in line:
                    parts = line.strip().split(',')
                    if len(parts) > 1:
                        return float(parts[1])
        print(f"警告: 在文件 {os.path.basename(file_path)} 中未找到 'Bitmap_Computation_Time_ms'。返回 0.0。")
        return 0.0
    except FileNotFoundError:
        print(f"错误：找不到 UNG 总结文件 {file_path}。")
        raise
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return 0.0

def find_optimal_point(df_group, recall_col, param_col):
    """
    For a single query's data group, find the point where Recall first reaches 1.0 
    or its maximum value, and return that row of data.
    """
    first_reach_one = df_group[df_group[recall_col] >= 0.99999]
    if not first_reach_one.empty:
        optimal_param = first_reach_one[param_col].min()
        # Using .loc for safer indexing
        return df_group.loc[df_group[param_col] == optimal_param].iloc[0]

    max_recall = df_group[recall_col].max()
    if pd.isna(max_recall): return None
    first_reach_max = df_group[df_group[recall_col] == max_recall]
    optimal_param = first_reach_max[param_col].min()
    return df_group.loc[df_group[param_col] == optimal_param].iloc[0]

def process_query_group(q_group, acorn_bitmap_time, ung_bitmap_time, acorn_para_bitmap_time):
    """
    This function contains the logic from the original for-loop.
    It will be applied to each query group from the merged DataFrame.
    """
    # --- Find the optimal point for each method ---
    optimal_acorn = find_optimal_point(q_group, 'acorn_Recall', 'search_param')
    optimal_acorn1 = find_optimal_point(q_group, 'acorn_1_Recall', 'search_param')
    optimal_ung_false = find_optimal_point(q_group, 'Recall_ung_false', 'search_param')
    optimal_ung_true = find_optimal_point(q_group, 'Recall', 'search_param')
    
    if any(opt is None for opt in [optimal_acorn, optimal_acorn1, optimal_ung_false, optimal_ung_true]):
        return None # Return None if any optimal point is not found

    # --- Extract base performance data ---
    query_id = q_group['QueryID'].iloc[0]
    query_attr_length = optimal_ung_false['QuerySize']
    entry_time_orig = optimal_ung_false['EntryGroupT_ms_ung_false']
    entry_time_new = optimal_ung_true['EntryGroupT_ms']
    
    # --- Calculate total time for 6 algorithms at their optimal points ---
    acorn_t = optimal_acorn['acorn_Time_ms']
    acorn1_t = optimal_acorn1['acorn_1_Time_ms']
    ung_false_search_t = optimal_ung_false['SearchT_ms_ung_false']
    ung_true_search_t = optimal_ung_true['SearchT_ms']
    ung_false_flag_t = optimal_ung_false['FlagT_ms_ung_false']
    ung_true_flag_t = optimal_ung_true['FlagT_ms']

    # Calculations remain the same, just using the optimal points found
    time_acorn1 = acorn1_t + acorn_bitmap_time
    recall_acorn1 = optimal_acorn1['acorn_1_Recall']
    time_acorn_gamma = acorn_t + acorn_bitmap_time
    recall_acorn_gamma = optimal_acorn['acorn_Recall']
    time_ung = ung_false_search_t
    recall_ung = optimal_ung_false['Recall_ung_false']
    
    if entry_time_orig <= entry_time_new:
        time_method1 = ung_false_search_t
        flag_t_for_m1_m3 = ung_false_flag_t
    else:
        time_method1 = ung_true_search_t
        flag_t_for_m1_m3 = ung_true_flag_t
    recall_method1 = max(optimal_ung_false['Recall_ung_false'], optimal_ung_true['Recall'])

    term_m2_acorn_part = acorn_t + ung_bitmap_time
    if term_m2_acorn_part <= ung_false_search_t:
        time_method2 = term_m2_acorn_part + ung_false_flag_t
        recall_method2 = optimal_acorn['acorn_Recall']
    else:
        time_method2 = ung_false_search_t + ung_false_flag_t
        recall_method2 = optimal_ung_false['Recall_ung_false']

    min_search_val = min(term_m2_acorn_part, ung_false_search_t, ung_true_search_t)
    if min_search_val == term_m2_acorn_part:
        time_method3 = min_search_val + ung_false_flag_t
        recall_method3 = optimal_acorn['acorn_Recall']
    elif min_search_val == ung_false_search_t:
        time_method3 = min_search_val + ung_false_flag_t
        recall_method3 = optimal_ung_false['Recall_ung_false']
    else:
        time_method3 = min_search_val + ung_true_flag_t
        recall_method3 = optimal_ung_true['Recall']

    # --- Return results as a pandas Series ---
    return pd.Series({
        'QueryID': query_id, '1_QueryAttrLength': query_attr_length,
        '2_BitmapTime_ACORN_Orig_ms': acorn_bitmap_time, '3_BitmapTime_ACORN_Para_ms': acorn_para_bitmap_time,
        '4_BitmapTime_NewMethod_ms': ung_bitmap_time, '5_EntryTime_UNG_Orig_ms': entry_time_orig,
        '6_EntryTime_NewMethod_ms': entry_time_new, '7_DecisionFlagTime_ms': flag_t_for_m1_m3,
        'Time_ACORN-1_ms': time_acorn1, 'Recall_ACORN-1': recall_acorn1,
        'Time_ACORN-gamma_ms': time_acorn_gamma, 'Recall_ACORN-gamma': recall_acorn_gamma,
        'Time_UNG_ms': time_ung, 'Recall_UNG': recall_ung,
        'Time_Method1_ms': time_method1, 'Recall_Method1': recall_method1,
        'Time_Method2_ms': time_method2, 'Recall_Method2': recall_method2,
        'Time_Method3_ms': time_method3, 'Recall_Method3': recall_method3
    })

def run_processing(paths):
    """
    Rewritten main processing function using vectorization (groupby-apply).
    """
    print("\n[步骤 1/5] 开始合并与处理实验结果...")

    # --- 1. Load all data files ---
    print("正在加载所有数据文件...")
    df_acorn = pd.read_csv(paths['acorn_details_file'])
    df_acorn_avg = pd.read_csv(paths['acorn_avg_file'])
    df_ung_false = pd.read_csv(paths['ung_nt_false_details_file'])
    df_ung_true = pd.read_csv(paths['ung_nt_true_details_file'])
    ung_bitmap_total_time = get_bitmap_time_from_summary(paths['ung_summary_file'])
    
    # --- 2. Pre-calculation and data preparation ---
    num_queries = df_acorn['QueryID'].nunique()
    print(f"检测到 {num_queries} 个查询任务。")
    acorn_bitmap_time = ung_bitmap_total_time / num_queries if num_queries > 0 else 0.0
    ung_bitmap_time = ung_bitmap_total_time / num_queries if num_queries > 0 else 0.0
    acorn_para_bitmap_time = df_acorn_avg.get('FilterMapTime_para_ms', pd.Series([0])).iloc[0] / num_queries

    # --- 3. Merge all DataFrames into one ---
    print("正在准备并合并所有数据源...")
    
    # Map ACORN's 'efs' to UNG's 'Lsearch' to enable merging
    acorn_params = sorted(df_acorn['efs'].unique())
    ung_params = sorted(df_ung_false['Lsearch'].unique())
    if len(acorn_params) != len(ung_params):
        print("错误: ACORN 和 UNG 的搜索参数数量不匹配，无法合并。")
        return
        
    param_map = pd.DataFrame({'efs': acorn_params, 'Lsearch': ung_params})
    df_acorn = pd.merge(df_acorn, param_map, on='efs')

    df_ung_false = df_ung_false.add_suffix('_ung_false')
    df_ung_false.rename(columns={'QueryID_ung_false': 'QueryID', 'Lsearch_ung_false': 'Lsearch'}, inplace=True) # add_suffix会重命名所有列，所以我们需要把用于合并的键改回原名

    merged_df = pd.merge(df_acorn, df_ung_false, on=['QueryID', 'Lsearch'])
    merged_df = pd.merge(merged_df, df_ung_true, on=['QueryID', 'Lsearch'])
    
    merged_df.rename(columns={'Lsearch': 'search_param'}, inplace=True)# 将 'Lsearch' 重命名为通用名称


    print("正在计算所有查询的性能指标...")
    
    # Group by QueryID and apply the processing function to each group
    results_df = merged_df.groupby('QueryID').progress_apply(
        process_query_group, 
        acorn_bitmap_time=acorn_bitmap_time, 
        ung_bitmap_time=ung_bitmap_time,
        acorn_para_bitmap_time=acorn_para_bitmap_time
    )

    # Drop any rows where processing failed (process_query_group returned None)
    results_df.dropna(inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    # --- 5. Calculate final ratio columns ---
    print("正在计算最终比率...")
    results_df['Ratio_M3/ACORN-1'] = results_df['Time_Method3_ms'] / results_df['Time_ACORN-1_ms']
    results_df['Ratio_M3/ACORN-gamma'] = results_df['Time_Method3_ms'] / results_df['Time_ACORN-gamma_ms']
    results_df['Ratio_M3/UNG'] = results_df['Time_Method3_ms'] / results_df['Time_UNG_ms']
    results_df['Ratio_M3/Method1'] = results_df['Time_Method3_ms'] / results_df['Time_Method1_ms']
    results_df['Ratio_M3/Method2'] = results_df['Time_Method3_ms'] / results_df['Time_Method2_ms']
    results_df.replace([np.inf, -np.inf], 0, inplace=True)

    # --- 6. Save the final CSV file ---
    if not results_df.empty:
        output_csv_path = paths['output_csv_path']
        results_df.to_csv(output_csv_path, index=False, float_format='%.4f')
        print(f"✅ 处理完成！结果已保存到：{os.path.abspath(output_csv_path)}")
    else:
        print("\n⚠️ 处理未完成，没有生成任何结果。请检查输入文件和查询ID是否匹配。")