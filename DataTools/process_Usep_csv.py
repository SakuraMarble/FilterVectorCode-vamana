import pandas as pd
import numpy as np
import os
import re
import glob

# ==============================================================================
# 配置区
BASE_RESULTS_DIR = '/data/fxy/FilterVector/FilterVectorResults'
TARGET_RECALL = 0.97
dataset_name = "celeba"  
OUTPUT_DIR = f"/data/fxy/FilterVector/FilterVectorResults/merge_results/improve2/U_sep/{dataset_name}"
# ==============================================================================


def find_optimal_performance_per_query(df, recall_col, time_col, dist_calcs_col):
    """
    核心处理函数：对每个QueryID，根据筛选逻辑找到最优性能记录。
    """
    if df.empty or not all(c in df.columns for c in [recall_col, time_col, dist_calcs_col, 'QueryID']):
        return pd.DataFrame()

    # 确保所需列存在
    for col in [recall_col, time_col, dist_calcs_col, 'QueryID', 'repeat', 'Lsearch']:
        if col not in df.columns:
            print(f"[WARN] 列 '{col}' 在DataFrame中不存在。")
            # 可以选择返回空DataFrame或用默认值填充
            return pd.DataFrame()
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
    处理一对匹配好的 UNG CSV 文件（sep=false 和 sep=true）。
    现在会额外提取 repeat 和 Lsearch。
    """
    try:
        # --- 1. 处理 sep=false 文件 ---
        df_false = pd.read_csv(sepfalse_csv_path)
        df_false.columns = df_false.columns.str.strip()
        optimal_false = find_optimal_performance_per_query(
            df_false, 'Recall', 'UNG_time(ms)', 'DistanceCalcs'
        )
        
        # 定义需要提取的列和重命名的映射关系
        cols_to_select_f = ['QueryID', 'repeat', 'Lsearch', 'Recall', 'UNG_time(ms)', 'DistanceCalcs']
        rename_map_f = {
            'repeat': 'repeat_sepF', 'Lsearch': 'Lsearch_sepF',
            'Recall': 'Recall_sepF', 'UNG_time(ms)': 'Time_sepF(ms)', 'DistanceCalcs': 'DistCalcs_sepF'
        }
        
        # 在DataFrame非空时，提取并重命名列
        if not optimal_false.empty:
            final_false = optimal_false[cols_to_select_f].rename(columns=rename_map_f)
        else:
            final_false = pd.DataFrame(columns=['QueryID'] + list(rename_map_f.values()))

        # --- 2. 处理 sep=true 文件 ---
        df_true = pd.read_csv(septrue_csv_path)
        df_true.columns = df_true.columns.str.strip()
        optimal_true = find_optimal_performance_per_query(
            df_true, 'Recall', 'UNG_time(ms)', 'DistanceCalcs'
        )

        cols_to_select_t = ['QueryID', 'repeat', 'Lsearch', 'Recall', 'UNG_time(ms)', 'DistanceCalcs']
        rename_map_t = {
            'repeat': 'repeat_sepT', 'Lsearch': 'Lsearch_sepT',
            'Recall': 'Recall_sepT', 'UNG_time(ms)': 'Time_sepT(ms)', 'DistanceCalcs': 'DistCalcs_sepT'
        }
        
        if not optimal_true.empty:
            final_true = optimal_true[cols_to_select_t].rename(columns=rename_map_t)
        else:
            final_true = pd.DataFrame(columns=['QueryID'] + list(rename_map_t.values()))

        # --- 3. 合并结果 ---
        if final_false.empty and final_true.empty:
            return None

        # 确保 QueryID 是可用于合并的兼容类型
        if 'QueryID' in final_false.columns:
            final_false['QueryID'] = final_false['QueryID'].astype(int)
        if 'QueryID' in final_true.columns:
            final_true['QueryID'] = final_true['QueryID'].astype(int)

        merged_df = pd.merge(final_false, final_true, on='QueryID', how='outer')

        if 'QueryID' in merged_df.columns:
            # 将QueryID放到第一列，并按其排序
            cols = ['QueryID'] + [col for col in merged_df.columns if col != 'QueryID']
            merged_df = merged_df[cols]
            merged_df.sort_values(by='QueryID', inplace=True)
            merged_df['QueryID'] = merged_df['QueryID'].astype(int)


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

    print(f"\n{'='*25}\n[INFO] 开始处理数据集: {dataset_name}\n[INFO] 匹配规则: 查找仅 'sep=true'/'sep=false' 不同的 UNG 实验对\n{'='*25}")

    ung_base_dir = os.path.join(BASE_RESULTS_DIR, 'UNG', dataset_name)
    if not os.path.isdir(ung_base_dir):
        print(f"[ERROR] 找不到UNG数据集目录: {ung_base_dir}")
        return

    # 查找所有 'sepfalse' 实验目录，以此为基准进行匹配
    sepfalse_dirs = glob.glob(os.path.join(ung_base_dir, f'*_sepfalse_*'))
    print(f"[INFO] 找到 {len(sepfalse_dirs)} 个 'sep=false' 实验，开始查找匹配的 'sep=true' 对...")

    success_count = 0
    for sepfalse_dir_path in sepfalse_dirs:
        dir_name = os.path.basename(sepfalse_dir_path)
        print(f"\n[BASE] 找到基准实验: {dir_name}")

        # 构建对应的 'septrue' 目录名和路径
        septrue_dir_name = dir_name.replace('_sepfalse_', '_septrue_')
        septrue_dir_path = os.path.join(ung_base_dir, septrue_dir_name)

        print(f"  [SEARCH] 正在查找匹配的目录: {septrue_dir_name}")

        if not os.path.isdir(septrue_dir_path):
            print("  [FAIL] 未找到匹配的 'sep=true' 目录。")
            continue

        print(f"  [SUCCESS] 成功找到匹配对。")

        # 在每个目录中查找结果CSV文件
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
            # 根据基准目录名（sepfalse）创建一个描述性的输出文件名
            output_filename_base = os.path.basename(sepfalse_dir_path).replace('_sepfalse', '')
            output_filename = f"U_sepF_T_{output_filename_base}.csv"
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