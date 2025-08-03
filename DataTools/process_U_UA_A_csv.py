import pandas as pd
import numpy as np
import os
import re
import glob

# ==============================================================================
# --- 配置区 ---

# 1. U_A 文件所在目录
U_A_DIR = '/data/fxy/FilterVector/FilterVectorResults/merge_results/improve2/U_A/'

# 2. U_UA 文件所在目录
U_UA_DIR = '/data/fxy/FilterVector/FilterVectorResults/merge_results/improve2/U_UA/'

# 3. 合并后新文件的输出目录
OUTPUT_DIR = '/data/fxy/FilterVector/FilterVectorResults/merge_results/improve2/U_UA_A/'

# 4. 指定最终输出的列顺序
DESIRED_COLUMN_ORDER = [
    'QueryID',
    'Recall_U', 'Recall_A', 'Recall_AU', 'Recall_A1',
    'Time_U(ms)', 'Time_A(ms)', 'Time(ms)_AU', 'Time_A1(ms)',
    'DistCalcs_U', 'DistCalcs_A', 'DistCalcs_AU', 'DistCalcs_A1',
    'repeat_U', 'repeat_AU',
    'Lsearch_U', 'Lsearch_AU',
    'EntryPoints'
]

# ==============================================================================


def main():
    """
    主函数，驱动整个文件匹配、合并、分析和保存的流程。
    """
    # 为不同类型的输出文件创建目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    au_wins_dir = os.path.join(OUTPUT_DIR, "AU_wins")
    os.makedirs(au_wins_dir, exist_ok=True)
    
    print(f"[INFO] 合并后的CSV文件将保存在: {os.path.abspath(OUTPUT_DIR)}")
    print(f"[INFO] 'AU胜出'的条目将单独保存在: {os.path.abspath(au_wins_dir)}")

    ua_pattern = re.compile(r"U_A_(.*?)_q(\d+)_th(\d+)\.csv")
    ua_files = glob.glob(os.path.join(U_A_DIR, 'U_A_*.csv'))
    
    if not ua_files:
        print(f"[ERROR] 在目录 {U_A_DIR} 中未找到任何 'U_A_*.csv' 文件。")
        return

    print(f"\n[INFO] 找到 {len(ua_files)} 个 U_A 文件，开始逐一匹配和合并...")
    
    success_count = 0
    all_winning_entries = [] # 用于存储所有文件中“胜出”的条目

    for ua_file_path in ua_files:
        ua_filename = os.path.basename(ua_file_path)
        match = ua_pattern.match(ua_filename)

        if not match:
            print(f"\n [SKIP] 文件名格式不匹配: {ua_filename}")
            continue

        dataset, query_val, th_val = match.groups()
        print(f"\n[BASE] 正在处理文件: {ua_filename} (dataset={dataset}, q={query_val}, th={th_val})")

        u_ua_search_pattern = os.path.join(U_UA_DIR, f"U_UA_{dataset}_query{query_val}_th{th_val}_*.csv")
        
        print(f"  [SEARCH] 正在查找匹配的 U_UA 文件...")
        matched_u_ua_files = glob.glob(u_ua_search_pattern)

        if not matched_u_ua_files:
            print("  [FAIL] 未找到匹配的 U_UA 文件。")
            continue
        
        u_ua_file_path = matched_u_ua_files[0]
        print(f"  [SUCCESS] 成功匹配文件: {os.path.basename(u_ua_file_path)}")

        try:
            # --- 数据合并流程 (与之前相同) ---
            df_ua = pd.read_csv(ua_file_path)
            df_u_ua = pd.read_csv(u_ua_file_path)

            cols_to_merge = [
                'QueryID', 'repeat_U', 'Lsearch_U', 'repeat_AU', 'Lsearch_AU', 
                'Time(ms)_AU', 'DistCalcs_AU', 'Recall_AU', 'EntryPoints'
            ]
            
            if not all(col in df_u_ua.columns for col in cols_to_merge):
                print(f"  [WARN] U_UA 文件中缺少用于合并的列，跳过。")
                continue

            df_u_ua_selected = df_u_ua[cols_to_merge]
            final_merged_df = pd.merge(df_ua, df_u_ua_selected, on='QueryID', how='outer')
            
            final_ordered_cols = [col for col in DESIRED_COLUMN_ORDER if col in final_merged_df.columns]
            final_merged_df = final_merged_df.reindex(columns=final_ordered_cols)
            final_merged_df.sort_values(by='QueryID', inplace=True)

            output_filename = f"U_UA_A_{dataset}_q{query_val}_th{th_val}.csv"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            final_merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            print(f"  [SAVE] 合并成功！结果已保存到: {output_path}")
            success_count += 1

            # --- 新增：分析与统计 ---
            print("  [ANALYZE] 正在分析 'AU胜出' 的条目...")
            
            # 定义分析所需的列，检查是否存在
            analysis_cols = ['Recall_AU', 'Recall_A', 'Recall_U', 'Time(ms)_AU', 'Time_A(ms)', 'Time_U(ms)']
            if not all(col in final_merged_df.columns for col in analysis_cols):
                print("    [WARN] 缺少用于分析的必要列，跳过此文件的统计。")
                continue
            
            # 筛选前先删除相关列中包含NaN的行，确保比较的有效性
            df_for_analysis = final_merged_df.dropna(subset=analysis_cols)

            # 定义筛选条件
            condition = (
                (df_for_analysis['Recall_AU'] >= df_for_analysis['Recall_A']) &
                (df_for_analysis['Recall_AU'] >= df_for_analysis['Recall_U']) &
                (df_for_analysis['Time(ms)_AU'] < df_for_analysis['Time_A(ms)']) &
                (df_for_analysis['Time(ms)_AU'] < df_for_analysis['Time_U(ms)'])
            )
            
            winning_entries = df_for_analysis[condition]
            
            if not winning_entries.empty:
                win_count = len(winning_entries)
                print(f"    [FOUND] 在此文件中找到 {win_count} 个 'AU胜出' 的查询任务。")
                
                # 将胜出条目保存到单独的CSV
                win_filename = f"AU_wins_{dataset}_q{query_val}_th{th_val}.csv"
                win_output_path = os.path.join(au_wins_dir, win_filename)
                winning_entries.to_csv(win_output_path, index=False, encoding='utf-8-sig')
                print(f"    [SAVE] 'AU胜出' 条目已保存到: {win_output_path}")

                # 将当前文件的胜出条目添加到全局列表，用于最终统计
                all_winning_entries.append(winning_entries)
            else:
                print("    [INFO] 此文件中未找到 'AU胜出' 的查询任务。")

        except Exception as e:
            print(f"  [ERROR] 处理文件对时出错: {e}")
            continue
    
    # --- 新增：在所有文件处理完毕后，进行全局统计 ---
    print("\n" + "="*25 + " 全局统计分析报告 " + "="*25)
    if not all_winning_entries:
        print("在所有处理的文件中，未发现任何符合 'AU胜出' 条件的查询任务。")
    else:
        # 将所有胜出条目合并到一个大的DataFrame中
        overall_winners_df = pd.concat(all_winning_entries, ignore_index=True)
        total_win_count = len(overall_winners_df)
        
        print(f"全局总计: 在所有文件中，共有 {total_win_count} 个查询任务满足 'AU胜出' 条件。")
        
        # 计算性能提升比率
        # 为避免除以0的错误，将分母中的0替换为NaN
        time_u_safe = overall_winners_df['Time_U(ms)'].replace(0, np.nan)
        time_a_safe = overall_winners_df['Time_A(ms)'].replace(0, np.nan)
        dist_u_safe = overall_winners_df['DistCalcs_U'].replace(0, np.nan)

        # 计算平均值，.mean()会自动忽略NaN
        avg_time_ratio_vs_u = (overall_winners_df['Time(ms)_AU'] / time_u_safe).mean()
        avg_time_ratio_vs_a = (overall_winners_df['Time(ms)_AU'] / time_a_safe).mean()
        avg_dist_ratio_vs_u = (overall_winners_df['DistCalcs_AU'] / dist_u_safe).mean()
        
        print("\n在这些'胜出'的查询任务中，性能指标比率的平均值为:")
        print(f"  - 平均 [Time(ms)_AU / Time_U(ms)]: {avg_time_ratio_vs_u:.4f}")
        print(f"  - 平均 [Time(ms)_AU / Time_A(ms)]: {avg_time_ratio_vs_a:.4f}")
        print(f"  - 平均 [DistCalcs_AU / DistCalcs_U]: {avg_dist_ratio_vs_u:.4f}")
        
    print("="*70)

    print(f"\n[FINISH] 全部处理完成！")
    print(f"共成功合并并生成了 {success_count} 个CSV文件。")


if __name__ == '__main__':
    main()