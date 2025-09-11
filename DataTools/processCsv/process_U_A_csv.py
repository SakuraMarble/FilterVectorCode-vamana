import pandas as pd
import numpy as np
import os
import re
import glob

# ==============================================================================
BASE_RESULTS_DIR = '/data/fxy/FilterVector/FilterVectorResults'
TARGET_RECALL = 0.95
dataset_name = "arxiv"
OUTPUT_DIR = "/data/fxy/FilterVector/FilterVectorResults/merge_results/experiments/" +dataset_name
# ==============================================================================

def find_optimal_performance_per_query(df, recall_col, time_col, dist_calcs_col):
    """
    核心处理函数：对每个QueryID，根据新的筛选逻辑找到最优记录。
    """
    if df.empty or not all(c in df.columns for c in [recall_col, time_col, dist_calcs_col, 'QueryID']):
        return pd.DataFrame()
        
    for col in [recall_col, time_col, dist_calcs_col, 'QueryID']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=[recall_col, time_col, 'QueryID'], inplace=True)
    df['QueryID'] = df['QueryID'].astype(int)

    optimal_rows = []
    
    for query_id, group in df.groupby('QueryID'):
        high_recall_group = group[group[recall_col] >= TARGET_RECALL]
        
        if not high_recall_group.empty:
            optimal_row = high_recall_group.sort_values(by=time_col, ascending=True).iloc[0]
        else:
            max_recall = group[recall_col].max()
            if pd.isna(max_recall): continue
            top_recall_group = group[group[recall_col] == max_recall]
            if top_recall_group.empty: continue
            optimal_row = top_recall_group.sort_values(by=time_col, ascending=True).iloc[0]
            
        optimal_rows.append(optimal_row)
        
    if not optimal_rows:
        return pd.DataFrame()
        
    return pd.concat(optimal_rows, axis=1).T


def process_matched_pair(acorn_csv_path, ung_csv_path):
    """
    处理一对匹配好的ACORN和UNG CSV文件，并返回合并后的DataFrame。
    """
    try:
        # --- 1. 处理 UNG 文件 ---
        ung_df = pd.read_csv(ung_csv_path)
        ung_df.columns = ung_df.columns.str.strip()
        ung_optimal = find_optimal_performance_per_query(
            ung_df, 'Recall', 'SearchT_ms', 'DistCalcs'
        )
        ung_final = ung_optimal[['QueryID', 'Recall', 'SearchT_ms', 'DistCalcs']].rename(columns={
            'Recall': 'Recall_U', 'SearchT_ms': 'Time_U(ms)', 'DistCalcs': 'DistCalcs_U'
        }) if not ung_optimal.empty else pd.DataFrame()

        # --- 2. 处理 ACORN 文件 ---
        acorn_df = pd.read_csv(acorn_csv_path)
        acorn_df.columns = acorn_df.columns.str.strip()
        
        # --- 处理 'acorn' 算法 ---
        acorn_base_optimal = find_optimal_performance_per_query(
            acorn_df.copy(), 'acorn_Recall', 'acorn_Time', 'acorn_n3'
        )
        if not acorn_base_optimal.empty:
            acorn_base_final_temp = acorn_base_optimal[['QueryID', 'acorn_Recall', 'acorn_Time', 'acorn_n3']].copy()
            acorn_base_final_temp['acorn_Time'] = acorn_base_final_temp['acorn_Time'] * 1000
            acorn_base_final = acorn_base_final_temp.rename(columns={
                'acorn_Recall': 'Recall_A', 'acorn_Time': 'Time_A(ms)', 'acorn_n3': 'DistCalcs_A'
            })
        else:
            acorn_base_final = pd.DataFrame()

        # --- 处理 'ACORN_1' 算法 ---
        acorn_1_optimal = find_optimal_performance_per_query(
            acorn_df.copy(), 'ACORN_1_Recall', 'ACORN_1_Time', 'ACORN_1_n3'
        )
        if not acorn_1_optimal.empty:
            acorn_1_final_temp = acorn_1_optimal[['QueryID', 'ACORN_1_Recall', 'ACORN_1_Time', 'ACORN_1_n3']].copy()
            acorn_1_final_temp['ACORN_1_Time'] = acorn_1_final_temp['ACORN_1_Time'] * 1000
            acorn_1_final = acorn_1_final_temp.rename(columns={
                'ACORN_1_Recall': 'Recall_A1', 'ACORN_1_Time': 'Time_A1(ms)', 'ACORN_1_n3': 'DistCalcs_A1'
            })
        else:
            acorn_1_final = pd.DataFrame()

        # --- 3. 合并所有结果 ---
        final_dfs = [df for df in [ung_final, acorn_base_final, acorn_1_final] if not df.empty]
        if not final_dfs: return None

        merged_df = final_dfs[0]
        for df_to_merge in final_dfs[1:]:
            merged_df = pd.merge(merged_df, df_to_merge, on='QueryID', how='outer')
        
        # Check if 'QueryID' column exists before trying to convert its type
        if 'QueryID' in merged_df.columns:
            merged_df.dropna(subset=['QueryID'], inplace=True)
            merged_df['QueryID'] = merged_df['QueryID'].astype(int)
            merged_df.sort_values(by='QueryID', inplace=True)
        return merged_df

    except Exception as e:
        print(f"  [ERROR] 处理文件对时出错: {e}")
        return None


def main():
    """
    主函数，驱动整个匹配、处理和合并流程。
    """
    if not dataset_name:
        print("[ERROR] 未在配置区指定数据集名称，程序退出。")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n[INFO] 结果CSV文件将保存在: {os.path.abspath(OUTPUT_DIR)}")
    
    print(f"\n{'='*25}\n[INFO] 开始处理数据集: {dataset_name}\n[INFO] 匹配规则: dataset_name + query + th/threads\n{'='*25}")

    ung_base_dir = os.path.join(BASE_RESULTS_DIR, 'UNG', dataset_name)
    acorn_base_dir = os.path.join(BASE_RESULTS_DIR, 'ACORN', dataset_name)

    if not os.path.isdir(ung_base_dir):
        print(f"[ERROR] 找不到UNG数据集目录: {ung_base_dir}")
        return

    ung_pattern = re.compile(r".*_query(\d+)_(nT(?:true|false))_th(\d+)_.*")
    
    ung_exp_dirs = glob.glob(os.path.join(ung_base_dir, '*'))
    print(f"[INFO] 在UNG下找到 {len(ung_exp_dirs)} 个实验目录，开始以其为基准进行匹配...")

    success_count = 0
    for ung_dir in ung_exp_dirs:
        ung_dir_name = os.path.basename(ung_dir)
        match = ung_pattern.match(ung_dir_name)

        if not match:
            print(f"\n [SKIP] 无法从UNG目录名解析参数: {ung_dir_name}")
            continue

        query_val, nT_val,th_val = match.groups()
        print(f"\n[UNG] 找到基准实验: {ung_dir_name} (query={query_val}, th={th_val})")
        
        acorn_search_pattern = os.path.join(acorn_base_dir, f"{dataset_name}_query{query_val}_*_threads{th_val}_*")
        
        print(f"  [SEARCH] 正在查找匹配的ACORN目录...")
        matched_acorn_dirs = glob.glob(acorn_search_pattern)

        if not matched_acorn_dirs:
            print("  [FAIL] 未找到匹配的ACORN实验目录。")
            continue
        
        acorn_dir = matched_acorn_dirs[0]
        print(f"  [SUCCESS] 成功匹配ACORN目录: {os.path.basename(acorn_dir)}")

        # 步骤1: 查找所有可能的CSV文件
        all_acorn_csvs = glob.glob(os.path.join(acorn_dir, 'results', '*.csv'))
        
        # 步骤2: 过滤掉汇总文件，只保留详细结果文件
        acorn_detail_csvs = [f for f in all_acorn_csvs if not f.endswith('_avg.csv')]
        
        # 步骤3: 获取最终路径
        ung_csv_path = next(iter(glob.glob(os.path.join(ung_dir, 'results', 'query_details_repeat*.csv'))), None)
        acorn_csv_path = acorn_detail_csvs[0] if acorn_detail_csvs else None
        
        if not acorn_csv_path:
            print(f"  [FAIL] 未能找到ACORN的详细结果文件(已忽略_avg.csv)。")
            continue
        if not ung_csv_path:
            print(f"  [FAIL] 未能找到UNG的结果CSV文件。")
            continue

        print(f"  [INFO] 正在处理UNG文件: {os.path.basename(ung_csv_path)}")
        print(f"  [INFO] 正在处理ACORN文件: {os.path.basename(acorn_csv_path)}")
        
        final_merged_df = process_matched_pair(acorn_csv_path, ung_csv_path)

        if final_merged_df is not None and not final_merged_df.empty:
            output_filename = f"U_A_{dataset_name}_q{query_val}_th{th_val}_{nT_val}.csv"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            final_merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"  [SAVE] 结果已保存到: {output_path}")

            success_count += 1
        else:
            print("  [WARN] 处理结果为空，未生成文件。")


    print(f"\n[FINISH] 数据集 '{dataset_name}' 处理完成！")
    print(f"共成功处理并生成了 {success_count} 个CSV结果文件在目录 '{os.path.abspath(OUTPUT_DIR)}' 中。")


if __name__ == '__main__':
    main()