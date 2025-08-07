import pandas as pd
import numpy as np
import os
import re
import glob
from typing import List, Tuple, Dict, Optional

# ==============================================================================
# --- 配置区 (Constants) ---

# 1. 基础目录
dataset = "app_reviews"
U_A_DIR = "/data/fxy/FilterVector/FilterVectorResults/merge_results/improve2/U_A/"+dataset+"/"
U_UA_DIR = "/data/fxy/FilterVector/FilterVectorResults/merge_results/improve2/U_UA/"+dataset+"/"
OUTPUT_DIR = "/data/fxy/FilterVector/FilterVectorResults/merge_results/improve2/U_UA_A/"+dataset+"/"

# 2. 正则表达式模式 (用于从文件名中提取信息)
U_UA_PATTERN = re.compile(r"U_UA_(.*?)_query(\d+)_(sep(?:true|false))_th(\d+)_.*\.csv")

# 3. 列配置
DESIRED_COLUMN_ORDER = [
    'QueryID',
    'Recall_U', 'Recall_A', 'Recall_AU', 'Recall_A1',
    'Time_U(ms)', 'Time_A(ms)', 'Time(ms)_AU', 'Time_A1(ms)',
    'DistCalcs_U', 'DistCalcs_A', 'DistCalcs_AU', 'DistCalcs_A1',
    'repeat_U', 'repeat_AU',
    'Lsearch_U', 'Lsearch_AU',
    'EntryPoints'
]
# "AU胜出"分析所依赖的列
ANALYSIS_COLS = ['Recall_AU', 'Recall_A', 'Recall_U', 'Time(ms)_AU', 'Time_A(ms)', 'Time_U(ms)', 'DistCalcs_AU', 'DistCalcs_U']
# 从U_UA文件合并到U_A文件的列
MERGE_COLS_FROM_U_UA = [
    'QueryID', 'repeat_U', 'Lsearch_U', 'repeat_AU', 'Lsearch_AU',
    'Time(ms)_AU', 'DistCalcs_AU', 'Recall_AU', 'EntryPoints'
]

# ==============================================================================

def setup_directories(base_dir: str, wins_sub_dir: str = "AU_wins") -> Tuple[str, str]:
    """
    创建主输出目录和用于存放"胜出"案例的子目录。
    """
    os.makedirs(base_dir, exist_ok=True)
    wins_dir = os.path.join(base_dir, wins_sub_dir)
    os.makedirs(wins_dir, exist_ok=True)
    print(f"[INFO] 合并后的CSV文件将保存在: {os.path.abspath(base_dir)}")
    print(f"[INFO] 'AU胜出'的条目将单独保存在: {os.path.abspath(wins_dir)}")
    return base_dir, wins_dir


def find_file_pairs(u_ua_dir: str, u_a_dir: str) -> List[Tuple[str, str]]:
    """
    遍历 U_UA 目录, 寻找每个文件在 U_A 目录中对应的文件。
    """
    u_ua_files = glob.glob(os.path.join(u_ua_dir, 'U_UA_*.csv'))
    if not u_ua_files:
        print(f"[ERROR] 在目录 {u_ua_dir} 中未找到任何 'U_UA_*.csv' 文件。")
        return []

    print(f"\n[INFO] 找到 {len(u_ua_files)} 个 U_UA 文件，开始进行匹配...")
    
    matched_pairs = []
    for u_ua_path in u_ua_files:
        u_ua_filename = os.path.basename(u_ua_path)
        match = U_UA_PATTERN.match(u_ua_filename)

        if not match:
            print(f"  [SKIP] 文件名格式不匹配: {u_ua_filename}")
            continue

        dataset, query_val, sep_val, th_val = match.groups()
        ua_filename_to_find = f"U_A_{dataset}_q{query_val}_th{th_val}_{sep_val}.csv"
        ua_path = os.path.join(u_a_dir, ua_filename_to_find)

        if os.path.exists(ua_path):
            matched_pairs.append((u_ua_path, ua_path))
        else:
            print(f"  [NO MATCH] {u_ua_filename} -> 未找到对应的 {ua_filename_to_find}")
            
    return matched_pairs


def process_single_pair(u_ua_path: str, ua_path: str, column_order: List[str]) -> Optional[pd.DataFrame]:
    """
    读取一对U_UA和U_A文件，将它们合并、排序并整理列顺序。
    """
    try:
        df_u_ua = pd.read_csv(u_ua_path)
        df_ua = pd.read_csv(ua_path)

        if not all(col in df_u_ua.columns for col in MERGE_COLS_FROM_U_UA):
            print(f"    [WARN] 文件 {os.path.basename(u_ua_path)} 缺少必要的合并列，跳过。")
            return None

        df_u_ua_selected = df_u_ua[MERGE_COLS_FROM_U_UA]
        final_df = pd.merge(df_ua, df_u_ua_selected, on='QueryID', how='outer')
        
        final_ordered_cols = [col for col in column_order if col in final_df.columns]
        final_df = final_df.reindex(columns=final_ordered_cols)
        final_df.sort_values(by='QueryID', inplace=True)
        
        return final_df

    except Exception as e:
        print(f"    [ERROR] 处理文件对 ({os.path.basename(u_ua_path)}, {os.path.basename(ua_path)}) 时出错: {e}")
        return None


def analyze_and_save_wins(merged_df: pd.DataFrame, base_filename: str, wins_dir: str):
    """
    分析合并后的数据，筛选出"AU胜出"条目，保存并立即打印该文件的统计报告。
    """
    print("    [ANALYZE] 正在分析 'AU胜出' 的条目...")
    if not all(col in merged_df.columns for col in ANALYSIS_COLS):
        print("      [WARN] 缺少用于分析的必要列，跳过此文件的统计。")
        return

    df_for_analysis = merged_df.dropna(subset=ANALYSIS_COLS)

    condition = (
        (df_for_analysis['Recall_AU'] >= df_for_analysis['Recall_A']) &
        (df_for_analysis['Recall_AU'] >= df_for_analysis['Recall_U']) &
        (df_for_analysis['Time(ms)_AU'] < df_for_analysis['Time_A(ms)']) &
        (df_for_analysis['Time(ms)_AU'] < df_for_analysis['Time_U(ms)'])
    )
    
    winning_entries = df_for_analysis[condition]

    if not winning_entries.empty:
        # --- 发现胜出条目，执行保存和独立分析 ---
        win_count = len(winning_entries)
        print(f"      [FOUND] 找到 {win_count} 个 'AU胜出' 的查询任务。")
        
        # 保存胜出条目到文件
        win_filename = f"AU_wins_{base_filename.replace('U_UA_', 'U_UA_A_', 1)}"
        win_output_path = os.path.join(wins_dir, win_filename)
        winning_entries.to_csv(win_output_path, index=False, encoding='utf-8-sig')
        print(f"      [SAVE] 'AU胜出' 条目已保存到: {os.path.basename(win_output_path)}")

        # --- 对当前文件的胜出条目进行即时统计分析 ---
        time_u_safe = winning_entries['Time_U(ms)'].replace(0, np.nan)
        time_a_safe = winning_entries['Time_A(ms)'].replace(0, np.nan)
        dist_u_safe = winning_entries['DistCalcs_U'].replace(0, np.nan)

        avg_time_ratio_vs_u = (winning_entries['Time(ms)_AU'] / time_u_safe).mean()
        avg_time_ratio_vs_a = (winning_entries['Time(ms)_AU'] / time_a_safe).mean()
        avg_dist_ratio_vs_u = (winning_entries['DistCalcs_AU'] / dist_u_safe).mean()
        # 新增的统计指标
        avg_time_a_vs_u = (winning_entries['Time_A(ms)'] / time_u_safe).mean()

        # 打印格式化的独立报告
        print("      " + "-"*15 + f" 'AU胜出' 统计报告 ({os.path.basename(base_filename)}) " + "-"*15)
        print(f"        - 平均 [Time(ms)_AU / Time_U(ms)]: {avg_time_ratio_vs_u:.4f}")
        print(f"        - 平均 [Time(ms)_AU / Time_A(ms)]: {avg_time_ratio_vs_a:.4f}")
        print(f"        - 平均 [Time(ms)_A / Time_U(ms)]:  {avg_time_a_vs_u:.4f}")
        print(f"        - 平均 [DistCalcs_AU / DistCalcs_U]: {avg_dist_ratio_vs_u:.4f}")
        print("      " + "-"*70)

    else:
        print("      [INFO] 此文件中未找到 'AU胜出' 的查询任务。")
    return


def main():
    """
    主函数，负责编排整个文件处理流程。
    """
    output_dir, au_wins_dir = setup_directories(OUTPUT_DIR)
    
    file_pairs = find_file_pairs(U_UA_DIR, U_A_DIR)
    if not file_pairs:
        print("\n[FINISH] 未找到任何可处理的文件对，程序结束。")
        return

    success_count = 0
    
    print("\n[INFO] 开始处理已匹配的文件对...")
    for u_ua_path, ua_path in file_pairs:
        u_ua_filename = os.path.basename(u_ua_path)
        ua_filename = os.path.basename(ua_path)
        print(f"\n[PROCESS] 正在处理对: [{u_ua_filename}] 与 [{ua_filename}]")
        
        merged_df = process_single_pair(u_ua_path, ua_path, DESIRED_COLUMN_ORDER)
        
        if merged_df is not None:
            output_filename = u_ua_filename.replace('U_UA_', 'U_UA_A_', 1)
            output_path = os.path.join(output_dir, output_filename)
            merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"    [SAVE] 合并成功！结果已保存到: {os.path.basename(output_path)}")
            success_count += 1
            
            # 分析是否存在"胜出"条目, 如果有则直接在函数内部分析并打印报告
            analyze_and_save_wins(merged_df, u_ua_filename, au_wins_dir)

    print(f"\n[FINISH] 全部处理完成！")
    print(f"共成功合并并生成了 {success_count} 个CSV文件。")


if __name__ == '__main__':
    main()