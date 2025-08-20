import pandas as pd
import numpy as np
import os
import re
import glob

# ==============================================================================
# 配置区
BASE_RESULTS_DIR = '/data/fxy/FilterVector/FilterVectorResults'
TARGET_RECALL = 0.97
dataset_name = "app_reviews"
OUTPUT_DIR = f"/data/fxy/FilterVector/FilterVectorResults/merge_results/improve2/U_nT_rms/{dataset_name}"
# ==============================================================================


def find_optimal_performance_per_query(df, recall_col, time_col, dist_calcs_col):
    """
    核心处理函数：对每个QueryID，根据筛选逻辑找到最优性能记录。（此函数无需修改）
    """
    if df.empty or not all(c in df.columns for c in [recall_col, time_col, dist_calcs_col, 'QueryID']):
        return pd.DataFrame()

    # 检查和转换所有可能用到的列
    cols_to_check = [
        recall_col, time_col, dist_calcs_col, 'QueryID', 'repeat', 'Lsearch',
        'EntryGroupT_ms', 'QuerySize', 'CandSize', 'SuccessChecks',
        'HitRatio', 'RecurCalls', 'PruneEvents', 'PruneEff','TrieNodePass'
    ]
    for col in cols_to_check:
        if col in df.columns:
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


def process_matched_triplet(nTfalse_csv_path, nTtrue_rmsfalse_csv_path, nTtrue_rmstrue_csv_path):
    """
    处理一组匹配好的三个 CSV 文件，并使用简洁的列名进行重命名。
    """
    try:
        # --- 第1部分: 读取并为每个文件找到最优行 ---
        df_false = pd.read_csv(nTfalse_csv_path)
        df_false.columns = df_false.columns.str.strip()
        optimal_false = find_optimal_performance_per_query(df_false, 'Recall', 'Time_ms', 'DistCalcs')

        df_true_rmsfalse = pd.read_csv(nTtrue_rmsfalse_csv_path)
        df_true_rmsfalse.columns = df_true_rmsfalse.columns.str.strip()
        optimal_true_rmsfalse = find_optimal_performance_per_query(df_true_rmsfalse, 'Recall', 'Time_ms', 'DistCalcs')

        df_true_rmstrue = pd.read_csv(nTtrue_rmstrue_csv_path)
        df_true_rmstrue.columns = df_true_rmstrue.columns.str.strip()
        optimal_true_rmstrue = find_optimal_performance_per_query(df_true_rmstrue, 'Recall', 'Time_ms', 'DistCalcs')

        if optimal_false.empty or optimal_true_rmsfalse.empty or optimal_true_rmstrue.empty:
            print("  [WARN] 至少一个文件处理后为空，无法比较和合并。")
            return None

        # --- 第2部分: 在所有三个文件中进行一致性检查 ---
        shared_cols = ['QuerySize', 'CandSize']
        comparison_df = pd.merge(
            optimal_false[['QueryID'] + shared_cols],
            optimal_true_rmsfalse[['QueryID'] + shared_cols],
            on='QueryID', suffixes=('_F', '_T')
        )
        comparison_df = pd.merge(
            comparison_df,
            optimal_true_rmstrue[['QueryID'] + shared_cols],
            on='QueryID'
        ).rename(columns={col: f'{col}_T_RMS' for col in shared_cols})

        for col in shared_cols:
            col_f, col_t, col_t_rms = f'{col}_F', f'{col}_T', f'{col}_T_RMS'
            if not (comparison_df[col_f].equals(comparison_df[col_t]) and comparison_df[col_f].equals(comparison_df[col_t_rms])):
                print(f"\n  [ERROR] FATAL: 列 '{col}' 在三个文件中的值不一致！")
                print("  处理已停止。")
                return None
        
        print("  [INFO] 一致性检查通过：共享列的值在三个文件中均匹配。")

        # --- 第3部分: 处理 nT=false 数据 (后缀: _F) ---
        optimal_false['Search_Only_Time'] = optimal_false['Time_ms'] - optimal_false['EntryGroupT_ms']
        cols_to_select_f = [
            'QueryID', 'repeat', 'Lsearch', 'Recall', 'Time_ms', 'DistCalcs',
            'EntryGroupT_ms', 'Search_Only_Time', 'SuccessChecks', 'HitRatio',
            'QuerySize', 'CandSize','TrieNodePass'
        ]
        rename_map_f = {
            'repeat': 'repeat_F', 'Lsearch': 'Lsearch_F', 'Recall': 'Recall_F',
            'Time_ms': 'Time_F(ms)', 'DistCalcs': 'DistCalcs_F',
            'EntryGroupT_ms': 'Get_Entry_Time_F(ms)', 'Search_Only_Time': 'Search_Only_Time_F(ms)',
            'SuccessChecks': 'SuccessChecks_F', 'HitRatio': 'HitRatio_F',
            'QuerySize': 'QuerySize', 'CandSize': 'CandSize','TrieNodePass':'TrieNodePass_F'
        }
        final_false = optimal_false[cols_to_select_f].rename(columns=rename_map_f)

        # --- 第4部分: 处理 nT=true, rms=false 数据 (后缀: _T) ---
        optimal_true_rmsfalse['Search_Only_Time'] = optimal_true_rmsfalse['Time_ms'] - optimal_true_rmsfalse['EntryGroupT_ms']
        cols_to_select_t = [
            'QueryID', 'repeat', 'Lsearch', 'Recall', 'Time_ms', 'DistCalcs',
            'EntryGroupT_ms', 'Search_Only_Time', 'RecurCalls', 'PruneEvents', 'PruneEff','TrieNodePass'
        ]
        rename_map_t = {
            'repeat': 'repeat_T', 'Lsearch': 'Lsearch_T', 'Recall': 'Recall_T',
            'Time_ms': 'Time_T(ms)', 'DistCalcs': 'DistCalcs_T',
            'EntryGroupT_ms': 'Get_Entry_Time_T(ms)', 'Search_Only_Time': 'Search_Only_Time_T(ms)',
            'RecurCalls': 'RecurCalls_T', 'PruneEvents': 'PruneEvents_T', 'PruneEff': 'PruneEff_T',
            'TrieNodePass':'TrieNodePass_T'
        }
        final_true_rmsfalse = optimal_true_rmsfalse[cols_to_select_t].rename(columns=rename_map_t)

        # --- 第5部分: 处理 nT=true, rms=true 数据 (后缀: _T_RMS) ---
        optimal_true_rmstrue['Search_Only_Time'] = optimal_true_rmstrue['Time_ms'] - optimal_true_rmstrue['EntryGroupT_ms']
        rename_map_t_rms = {
            'repeat': 'repeat_T_RMS', 'Lsearch': 'Lsearch_T_RMS', 'Recall': 'Recall_T_RMS',
            'Time_ms': 'Time_T_RMS(ms)', 'DistCalcs': 'DistCalcs_T_RMS',
            'EntryGroupT_ms': 'Get_Entry_Time_T_RMS(ms)', 'Search_Only_Time': 'Search_Only_Time_T_RMS(ms)',
            'RecurCalls': 'RecurCalls_T_RMS', 'PruneEvents': 'PruneEvents_T_RMS', 'PruneEff': 'PruneEff_T_RMS',
            'TrieNodePass':'TrieNodePass_T_RMS'
        }
        final_true_rmstrue = optimal_true_rmstrue[cols_to_select_t].rename(columns=rename_map_t_rms)

        # --- 第6部分: 合并所有三个结果 ---
        merged_df = pd.merge(final_false, final_true_rmsfalse, on='QueryID', how='outer')
        merged_df = pd.merge(merged_df, final_true_rmstrue, on='QueryID', how='outer')

        if 'QueryID' in merged_df.columns:
            first_cols = ['QueryID', 'QuerySize', 'CandSize']
            other_cols = [col for col in merged_df.columns if col not in first_cols]
            merged_df = merged_df[first_cols + other_cols]
            merged_df.sort_values(by='QueryID', inplace=True)
            merged_df['QueryID'] = merged_df['QueryID'].astype(int)

        return merged_df

    except Exception as e:
        print(f"  [ERROR] 处理文件三元组时发生意外错误: {e}")
        import traceback
        traceback.print_exc()
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

    print(f"\n{'='*25}\n[INFO] 开始处理数据集: {dataset_name}\n[INFO] 匹配规则: 查找 'nTfalse', 'nTtrue_rmsfalse', 'nTtrue_rmstrue' 实验三元组\n{'='*25}")

    ung_base_dir = os.path.join(BASE_RESULTS_DIR, 'UNG', dataset_name)
    if not os.path.isdir(ung_base_dir):
        print(f"[ERROR] 找不到UNG数据集目录: {ung_base_dir}")
        return

    nTfalse_dirs = glob.glob(os.path.join(ung_base_dir, f'*_nTfalse_*'))
    print(f"[INFO] 找到 {len(nTfalse_dirs)} 个 'nT=false' 实验，开始查找匹配的三元组...")

    success_count = 0
    for nTfalse_dir_path in nTfalse_dirs:
        dir_name = os.path.basename(nTfalse_dir_path)
        print(f"\n[BASE] 找到基准实验: {dir_name}")

        # --- 构造其他两个目录的名称 ---
        nTtrue_rmsfalse_dir_name = dir_name.replace('_nTfalse_', '_nTtrue_rmsfalse_')
        nTtrue_rmsfalse_dir_path = os.path.join(ung_base_dir, nTtrue_rmsfalse_dir_name)

        nTtrue_rmstrue_dir_name = dir_name.replace('_nTfalse_', '_nTtrue_rmstrue_')
        nTtrue_rmstrue_dir_path = os.path.join(ung_base_dir, nTtrue_rmstrue_dir_name)

        print(f"  [SEARCH] 正在查找匹配目录: {nTtrue_rmsfalse_dir_name}")
        print(f"  [SEARCH] 正在查找匹配目录: {nTtrue_rmstrue_dir_name}")

        # --- 检查所有三个目录是否存在 ---
        if not os.path.isdir(nTtrue_rmsfalse_dir_path):
            print("  [FAIL] 未找到匹配的 'nTtrue_rmsfalse' 目录。")
            continue
        if not os.path.isdir(nTtrue_rmstrue_dir_path):
            print("  [FAIL] 未找到匹配的 'nTtrue_rmstrue' 目录。")
            continue

        print(f"  [SUCCESS] 成功找到匹配的三元组。")

        # --- 在每个目录中查找CSV文件 ---
        nTfalse_csv_path = next(iter(glob.glob(os.path.join(nTfalse_dir_path, 'results', 'query_details_repeat*.csv'))), None)
        nTtrue_rmsfalse_csv_path = next(iter(glob.glob(os.path.join(nTtrue_rmsfalse_dir_path, 'results', 'query_details_repeat*.csv'))), None)
        nTtrue_rmstrue_csv_path = next(iter(glob.glob(os.path.join(nTtrue_rmstrue_dir_path, 'results', 'query_details_repeat*.csv'))), None)

        if not nTfalse_csv_path:
            print(f"  [FAIL] 在 {dir_name} 中找不到结果CSV文件。")
            continue
        if not nTtrue_rmsfalse_csv_path:
            print(f"  [FAIL] 在 {nTtrue_rmsfalse_dir_name} 中找不到结果CSV文件。")
            continue
        if not nTtrue_rmstrue_csv_path:
            print(f"  [FAIL] 在 {nTtrue_rmstrue_dir_name} 中找不到结果CSV文件。")
            continue

        print(f"  [INFO] 正在处理 'nTfalse' 文件: {os.path.basename(nTfalse_csv_path)}")
        print(f"  [INFO] 正在处理 'nTtrue_rmsfalse' 文件: {os.path.basename(nTtrue_rmsfalse_csv_path)}")
        print(f"  [INFO] 正在处理 'nTtrue_rmstrue' 文件: {os.path.basename(nTtrue_rmstrue_csv_path)}")

        # --- 调用新的处理函数 ---
        final_merged_df = process_matched_triplet(nTfalse_csv_path, nTtrue_rmsfalse_csv_path, nTtrue_rmstrue_csv_path)

        if final_merged_df is not None and not final_merged_df.empty:
            output_filename_base = os.path.basename(nTfalse_dir_path).replace('_nTfalse', '')
            output_filename = f"U_Compare_{output_filename_base}.csv" # 更新输出文件名
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            final_merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"  [SAVE] 结果已保存到: {output_path}")
            success_count += 1
        else:
            print("  [WARN] 处理结果为空或检查失败，未生成文件。")

    print(f"\n[FINISH] 数据集 '{dataset_name}' 处理完成！")
    print(f"共成功处理并生成了 {success_count} 个CSV结果文件在目录 '{os.path.abspath(OUTPUT_DIR)}' 中。")


if __name__ == '__main__':
    main()