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
OUTPUT_DIR = f"/data/fxy/FilterVector/FilterVectorResults/merge_results/improve2/U_sep/{dataset_name}"
# ==============================================================================


def find_optimal_performance_per_query(df, recall_col, time_col, dist_calcs_col):
    """
    核心处理函数：对每个QueryID，根据筛选逻辑找到最优性能记录。
    """
    if df.empty or not all(c in df.columns for c in [recall_col, time_col, dist_calcs_col, 'QueryID']):
        return pd.DataFrame()

    # 检查和转换所有可能用到的列，且只转换DataFrame中存在的列
    cols_to_check = [
        recall_col, time_col, dist_calcs_col, 'QueryID', 'repeat', 'Lsearch',
        'EntryGroupT_ms', 'QuerySize', 'CandSize', 'SuccessChecks',
        'HitRatio', 'RecurCalls', 'PruneEvents', 'PruneEff'
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


def process_matched_sep_pair(sepfalse_csv_path, septrue_csv_path):
    """
    处理一对匹配好的 CSV 文件
    """
    try:
        df_false = pd.read_csv(sepfalse_csv_path)
        df_false.columns = df_false.columns.str.strip()
        optimal_false = find_optimal_performance_per_query(df_false, 'Recall', 'Time_ms', 'DistCalcs')

        df_true = pd.read_csv(septrue_csv_path)
        df_true.columns = df_true.columns.str.strip()
        optimal_true = find_optimal_performance_per_query(df_true, 'Recall', 'Time_ms', 'DistCalcs')

        if optimal_false.empty or optimal_true.empty:
            print("  [WARN] 其中一个或两个文件处理后为空，无法比较和合并。")
            return None

        # 一致性检查
        shared_cols = ['QuerySize', 'CandSize']
        comparison_df = pd.merge(
            optimal_false[['QueryID'] + shared_cols],
            optimal_true[['QueryID'] + shared_cols],
            on='QueryID',
            suffixes=('_F', '_T')
        )

        for col in shared_cols:
            col_f, col_t = f'{col}_F', f'{col}_T'
            if not comparison_df[col_f].equals(comparison_df[col_t]):
                mismatch = comparison_df[comparison_df[col_f] != comparison_df[col_t]]
                if not mismatch.empty:
                    first_mismatch = mismatch.iloc[0]
                    error_qid = int(first_mismatch['QueryID'])
                    val_f, val_t = first_mismatch[col_f], first_mismatch[col_t]
                    print(f"\n  [ERROR] FATAL: 列 '{col}' 在 QueryID={error_qid} 处的值不一致！")
                    print(f"    - False 文件中的值: {val_f}")
                    print(f"    - True 文件中的值: {val_t}")
                    print("  处理已停止。")
                    return None
        
        print("  [INFO] 一致性检查通过：共享列的值均匹配。")

        # 处理 false 文件 (包含共享列)
        optimal_false['Search_Only_Time'] = optimal_false['Time_ms'] - optimal_false['EntryGroupT_ms']
        cols_to_select_f = [
            'QueryID', 'repeat', 'Lsearch', 'Recall', 'Time_ms', 'DistCalcs',
            'EntryGroupT_ms', 'Search_Only_Time', 'SuccessChecks', 'HitRatio',
            'QuerySize', 'CandSize'
        ]
        rename_map_f = {
            'repeat': 'repeat_sepF', 'Lsearch': 'Lsearch_sepF', 'Recall': 'Recall_sepF',
            'Time_ms': 'Time_sepF(ms)', 'DistCalcs': 'DistCalcs_sepF',
            'EntryGroupT_ms': 'Get_Entry_Time_sepF(ms)', 'Search_Only_Time': 'Search_Only_Time_sepF(ms)',
            'SuccessChecks': 'SuccessChecks_sepF', 'HitRatio': 'HitRatio_sepF',
            'QuerySize': 'QuerySize', 'CandSize': 'CandSize'
        }
        final_false = optimal_false[cols_to_select_f].rename(columns=rename_map_f)

        # 处理 true 文件 (不包含共享列)
        optimal_true['Search_Only_Time'] = optimal_true['Time_ms'] - optimal_true['EntryGroupT_ms']
        cols_to_select_t = [
            'QueryID', 'repeat', 'Lsearch', 'Recall', 'Time_ms', 'DistCalcs',
            'EntryGroupT_ms', 'Search_Only_Time', 'RecurCalls', 'PruneEvents', 'PruneEff'
        ]
        rename_map_t = {
            'repeat': 'repeat_sepT', 'Lsearch': 'Lsearch_sepT', 'Recall': 'Recall_sepT',
            'Time_ms': 'Time_sepT(ms)', 'DistCalcs': 'DistCalcs_sepT',
            'EntryGroupT_ms': 'Get_Entry_Time_sepT(ms)', 'Search_Only_Time': 'Search_Only_Time_sepT(ms)',
            'RecurCalls': 'RecurCalls_sepT', 'PruneEvents': 'PruneEvents_sepT', 'PruneEff': 'PruneEff_sepT'
        }
        final_true = optimal_true[cols_to_select_t].rename(columns=rename_map_t)

        # 合并结果
        merged_df = pd.merge(final_false, final_true, on='QueryID', how='outer')
        if 'QueryID' in merged_df.columns:
            first_cols = ['QueryID', 'QuerySize', 'CandSize']
            other_cols = [col for col in merged_df.columns if col not in first_cols]
            merged_df = merged_df[first_cols + other_cols]
            merged_df.sort_values(by='QueryID', inplace=True)
            merged_df['QueryID'] = merged_df['QueryID'].astype(int)

        return merged_df

    except Exception as e:
        print(f"  [ERROR] 处理文件对时发生意外错误: {e}")
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

    print(f"\n{'='*25}\n[INFO] 开始处理数据集: {dataset_name}\n[INFO] 匹配规则: 查找仅 'sep=true'/'sep=false' 不同的 UNG 实验对\n{'='*25}")

    ung_base_dir = os.path.join(BASE_RESULTS_DIR, 'UNG', dataset_name)
    if not os.path.isdir(ung_base_dir):
        print(f"[ERROR] 找不到UNG数据集目录: {ung_base_dir}")
        return

    sepfalse_dirs = glob.glob(os.path.join(ung_base_dir, f'*_sepfalse_*'))
    print(f"[INFO] 找到 {len(sepfalse_dirs)} 个 'sep=false' 实验，开始查找匹配的 'sep=true' 对...")

    success_count = 0
    for sepfalse_dir_path in sepfalse_dirs:
        dir_name = os.path.basename(sepfalse_dir_path)
        print(f"\n[BASE] 找到基准实验: {dir_name}")

        septrue_dir_name = dir_name.replace('_sepfalse_', '_septrue_')
        septrue_dir_path = os.path.join(ung_base_dir, septrue_dir_name)

        print(f"  [SEARCH] 正在查找匹配的目录: {septrue_dir_name}")

        if not os.path.isdir(septrue_dir_path):
            print("  [FAIL] 未找到匹配的 'sep=true' 目录。")
            continue

        print(f"  [SUCCESS] 成功找到匹配对。")

        # 注意：这里的文件名查找使用了通配符，以匹配您更新后的文件名
        sepfalse_csv_path = next(iter(glob.glob(os.path.join(sepfalse_dir_path, 'results', 'query_details_repeat*.csv'))), None)
        septrue_csv_path = next(iter(glob.glob(os.path.join(septrue_dir_path, 'results', 'query_details_repeat*.csv'))), None)

        if not sepfalse_csv_path:
            print(f"  [FAIL] 在 {dir_name} 中找不到结果CSV文件。")
            continue
        if not septrue_csv_path:
            print(f"  [FAIL] 在 {septrue_dir_name} 中找不到结果CSV文件。")
            continue

        print(f"  [INFO] 正在处理 'sep=false' 文件: {os.path.basename(sepfalse_csv_path)}")
        print(f"  [INFO] 正在处理 'sep=true' 文件: {os.path.basename(septrue_csv_path)}")

        final_merged_df = process_matched_sep_pair(sepfalse_csv_path, septrue_csv_path)

        if final_merged_df is not None and not final_merged_df.empty:
            output_filename_base = os.path.basename(sepfalse_dir_path).replace('_sepfalse', '')
            output_filename = f"U_sepF_T_{output_filename_base}.csv"
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