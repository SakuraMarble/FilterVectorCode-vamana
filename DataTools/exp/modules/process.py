import pandas as pd
import numpy as np
import os

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
    # Attempt to find the first point where Recall >= 0.99999
    first_reach_one = df_group[df_group[recall_col] >= 0.99999]
    if not first_reach_one.empty:
        # If found, return the row corresponding to the minimum parameter value that achieved it
        optimal_param = first_reach_one[param_col].min()
        return df_group[df_group[param_col] == optimal_param].iloc[0]

    # If 1.0 is not reached, find the first point that achieves the maximum recall
    max_recall = df_group[recall_col].max()
    if pd.isna(max_recall): return None # Handle cases with no valid recall
    first_reach_max = df_group[df_group[recall_col] == max_recall]
    optimal_param = first_reach_max[param_col].min()
    return df_group[df_group[param_col] == optimal_param].iloc[0]

def run_processing(paths):
    """
    Main processing function that takes a dictionary of all required file paths.
    It loads data, calculates performance metrics for various algorithms at their optimal 
    recall points, and saves the consolidated results to a new CSV file.
    """
    print("\n[步骤 1/4] 开始合并与处理实验结果...")

    # Load file paths from the paths dictionary
    acorn_details_file = paths['acorn_details_file']
    acorn_avg_file = paths['acorn_avg_file']
    ung_nt_false_details_file = paths['ung_nt_false_details_file']
    ung_nt_true_details_file = paths['ung_nt_true_details_file']
    ung_summary_file = paths['ung_summary_file']
    output_csv_path = paths['output_csv_path']

    ### 2. Load and preprocess data ###
    print("正在加载数据文件...")
    df_acorn_details = pd.read_csv(acorn_details_file)
    df_acorn_avg = pd.read_csv(acorn_avg_file)
    df_ung_false = pd.read_csv(ung_nt_false_details_file)
    df_ung_true = pd.read_csv(ung_nt_true_details_file)
    ung_bitmap_total_time = get_bitmap_time_from_summary(ung_summary_file)

    # Get the total number of unique queries
    num_queries = df_acorn_details['QueryID'].nunique()
    print(f"检测到 {num_queries} 个查询任务。")

    # --- Calculate average per-query bitmap time (these are constants) ---
    # acorn_bitmap_time = df_acorn_avg['FilterMapTime_ms'].iloc[0] / num_queries
    acorn_bitmap_time = ung_bitmap_total_time / num_queries if num_queries > 0 else 0.0
    acorn_para_bitmap_time = df_acorn_avg.get('FilterMapTime_para_ms', pd.Series([0])).iloc[0] / num_queries
    ung_bitmap_time = ung_bitmap_total_time / num_queries if num_queries > 0 else 0.0

    ### 4. Iterate through each query ###
    print("正在为每个查询计算性能指标...")
    results_list = []

    for query_id in range(num_queries):
        # Filter all data for the current query ID
        q_acorn = df_acorn_details[df_acorn_details['QueryID'] == query_id].copy()
        q_ung_false = df_ung_false[df_ung_false['QueryID'] == query_id].copy()
        q_ung_true = df_ung_true[df_ung_true['QueryID'] == query_id].copy()

        if q_acorn.empty or q_ung_false.empty or q_ung_true.empty:
            print(f"警告: 查询ID {query_id} 的数据不完整，将跳过此查询。")
            continue

        # --- Find the optimal point for each method ---
        optimal_acorn = find_optimal_point(q_acorn, 'acorn_Recall', 'efs')
        optimal_acorn1 = find_optimal_point(q_acorn, 'acorn_1_Recall', 'efs')
        optimal_ung_false = find_optimal_point(q_ung_false, 'Recall', 'Lsearch')
        optimal_ung_true = find_optimal_point(q_ung_true, 'Recall', 'Lsearch')
        
        if any(opt is None for opt in [optimal_acorn, optimal_acorn1, optimal_ung_false, optimal_ung_true]):
            print(f"警告: 查询ID {query_id} 无法找到所有最优数据点，将跳过此查询。")
            continue

        # --- Extract base performance data ---
        query_attr_length = optimal_ung_false['QuerySize']
        entry_time_orig = optimal_ung_false['EntryGroupT_ms']
        entry_time_new = optimal_ung_true['EntryGroupT_ms']
        
        # --- Calculate total time for 6 algorithms at their optimal points ---
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
            time_method1 = ung_false_search_t
            flag_t_for_m1_m3 = ung_false_flag_t
        else:
            time_method1 = ung_true_search_t
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

        # --- Collect all results ---
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

    ### 5. Generate the final CSV file ###
    if results_list:
        final_df = pd.DataFrame(results_list)
        final_df.to_csv(output_csv_path, index=False, float_format='%.4f')
        print(f"✅ 处理完成！结果已保存到：{os.path.abspath(output_csv_path)}")
    else:
        print("\n⚠️ 处理未完成，没有生成任何结果。请检查输入文件和查询ID是否匹配。")