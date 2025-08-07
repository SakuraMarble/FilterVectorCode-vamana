import pandas as pd
import numpy as np
import os
import re
import glob
from typing import List, Tuple, Dict, Optional

# ==============================================================================
# --- 配置区 (Constants) ---

# 1. 基础目录
dataset = "celeba"
U_SEP_DIR = f"/data/fxy/FilterVector/FilterVectorResults/merge_results/improve2/U_sep/{dataset}/"
U_A_DIR = f"/data/fxy/FilterVector/FilterVectorResults/merge_results/improve2/U_A/{dataset}/"
OUTPUT_DIR = f"/data/fxy/FilterVector/FilterVectorResults/merge_results/improve2/U_Usep_A/{dataset}/"

# 2. 正则表达式模式 (用于从文件名中提取信息)
U_SEP_PATTERN = re.compile(r"U_sepF_T_(.*?)_query(\d+)_th(\d+)_.*\.csv")

# 3. 列配置
# 从 U_A 文件中提取的 ACORN 相关列
ACORN_COLS_TO_MERGE = [
    'QueryID', 'Recall_A', 'Time_A(ms)', 'DistCalcs_A',
    'Recall_A1', 'Time_A1(ms)', 'DistCalcs_A1'
]

# 最终输出文件期望的列顺序
DESIRED_COLUMN_ORDER = [
    'QueryID',
    'Recall_sepT', 'Recall_sepF', 'Recall_A', 'Recall_A1',
    'Time_sepT(ms)', 'Time_sepF(ms)', 'Time_A(ms)', 'Time_A1(ms)',
    'DistCalcs_sepT', 'DistCalcs_sepF', 'DistCalcs_A', 'DistCalcs_A1',
    'repeat_sepT', 'Lsearch_sepT', 'repeat_sepF', 'Lsearch_sepF'
]

# "sepT胜出"分析所依赖的列（已更新）
ANALYSIS_COLS = [
    'Recall_sepT', 'Recall_sepF', 'Recall_A',
    'Time_sepT(ms)', 'Time_sepF(ms)', 'Time_A(ms)',
    'DistCalcs_sepT', 'DistCalcs_A',
    'Lsearch_sepT', 'Lsearch_sepF' # 新增Lsearch用于比值计算
]

# ==============================================================================

def setup_directories(base_dir: str, wins_sub_dir: str = "sepT_wins_over_sepF_and_A") -> Tuple[str, str]:
    """
    创建主输出目录和用于存放"胜出"案例的子目录。
    """
    os.makedirs(base_dir, exist_ok=True)
    wins_dir = os.path.join(base_dir, wins_sub_dir)
    os.makedirs(wins_dir, exist_ok=True)
    print(f"[INFO] 合并后的CSV文件将保存在: {os.path.abspath(base_dir)}")
    print(f"[INFO] 'sepT胜出'的条目和报告将单独保存在: {os.path.abspath(wins_dir)}")
    return base_dir, wins_dir


def find_file_pairs(u_sep_dir: str, u_a_dir: str) -> List[Tuple[str, str]]:
    """
    遍历 U_sep 目录, 寻找每个文件在 U_A 目录中对应的 ACORN 数据文件。
    """
    u_sep_files = glob.glob(os.path.join(u_sep_dir, 'U_sepF_T_*.csv'))
    if not u_sep_files:
        print(f"[ERROR] 在目录 {u_sep_dir} 中未找到任何 'U_sepF_T_*.csv' 文件。")
        return []

    print(f"\n[INFO] 找到 {len(u_sep_files)} 个 U_sep 文件，开始进行匹配...")
    
    matched_pairs = []
    for u_sep_path in u_sep_files:
        u_sep_filename = os.path.basename(u_sep_path)
        match = U_SEP_PATTERN.match(u_sep_filename)

        if not match:
            print(f"  [SKIP] 文件名格式不匹配: {u_sep_filename}")
            continue
        
        dataset_name, query_val, th_val = match.groups()
        ua_filename_to_find = f"U_A_{dataset_name}_q{query_val}_th{th_val}_septrue.csv"
        ua_path = os.path.join(u_a_dir, ua_filename_to_find)

        if os.path.exists(ua_path):
            matched_pairs.append((u_sep_path, ua_path))
        else:
            print(f"  [NO MATCH] {u_sep_filename} -> 未找到对应的 {ua_filename_to_find}")
            
    return matched_pairs


def process_single_pair(u_sep_path: str, ua_path: str, column_order: List[str]) -> Optional[pd.DataFrame]:
    """
    读取一对U_sep和U_A文件，将ACORN数据合并进去，并整理列顺序。
    """
    try:
        df_sep = pd.read_csv(u_sep_path)
        df_acorn = pd.read_csv(ua_path)

        if not all(col in df_acorn.columns for col in ACORN_COLS_TO_MERGE):
            print(f"    [WARN] 文件 {os.path.basename(ua_path)} 缺少必要的ACORN列，跳过。")
            return None

        df_acorn_selected = df_acorn[ACORN_COLS_TO_MERGE]
        final_df = pd.merge(df_sep, df_acorn_selected, on='QueryID', how='outer')
        
        final_ordered_cols = [col for col in column_order if col in final_df.columns]
        final_df = final_df.reindex(columns=final_ordered_cols)
        final_df.sort_values(by='QueryID', inplace=True)
        
        return final_df

    except Exception as e:
        print(f"    [ERROR] 处理文件对 ({os.path.basename(u_sep_path)}, {os.path.basename(ua_path)}) 时出错: {e}")
        return None


def analyze_and_save_wins(merged_df: pd.DataFrame, base_filename: str, wins_dir: str):
    """
    分析合并后的数据，筛选出"sepT优于sepF和ACORN"的条目，保存并将统计报告输出到txt文件。
    """
    print("    [ANALYZE] 正在分析 'sepT 优于 sepF 和 ACORN' 的条目...")
    if not all(col in merged_df.columns for col in ANALYSIS_COLS):
        print("        [WARN] 缺少用于分析的必要列，跳过此文件的统计。")
        return

    df_for_analysis = merged_df.dropna(subset=ANALYSIS_COLS)

    # 定义"sepT胜出"的条件：召回率不低于其他两者，且时间上优于其他两者
    condition = (
        (df_for_analysis['Recall_sepT'] >= df_for_analysis['Recall_sepF']) &
        (df_for_analysis['Recall_sepT'] >= df_for_analysis['Recall_A']) &
        (df_for_analysis['Time_sepT(ms)'] < df_for_analysis['Time_sepF(ms)']) &
        (df_for_analysis['Time_sepT(ms)'] < df_for_analysis['Time_A(ms)'])
    )
    
    winning_entries = df_for_analysis[condition]

    if not winning_entries.empty:
        win_count = len(winning_entries)
        print(f"        [FOUND] 找到 {win_count} 个 'sepT胜出' 的查询任务。")
        
        # 1. 保存胜出条目的CSV文件
        win_filename = f"sepT_wins_{base_filename.replace('U_sepF_T_', 'U_Usep_A_', 1)}"
        win_output_path = os.path.join(wins_dir, win_filename)
        winning_entries.to_csv(win_output_path, index=False, encoding='utf-8-sig')
        print(f"        [SAVE] 'sepT胜出' 条目已保存到: {os.path.basename(win_output_path)}")

        # 2. 对胜出条目进行统计分析
        time_f_safe = winning_entries['Time_sepF(ms)'].replace(0, np.nan)
        time_a_safe = winning_entries['Time_A(ms)'].replace(0, np.nan)
        dist_a_safe = winning_entries['DistCalcs_A'].replace(0, np.nan)
        lsearch_f_safe = winning_entries['Lsearch_sepF'].replace(0, np.nan)

        avg_time_ratio_vs_f = (winning_entries['Time_sepT(ms)'] / time_f_safe).mean()
        avg_time_ratio_vs_a = (winning_entries['Time_sepT(ms)'] / time_a_safe).mean()
        avg_dist_ratio_vs_a = (winning_entries['DistCalcs_sepT'] / dist_a_safe).mean()
        avg_lsearch_ratio_vs_f = (winning_entries['Lsearch_sepT'] / lsearch_f_safe).mean()

        # 3. 生成报告内容字符串
        report_lines = [
            "="*10 + f" 'sepT胜出' 统计报告 ({os.path.basename(base_filename)}) " + "="*10,
            f"- 平均 [Time_sepT / Time_sepF]: {avg_time_ratio_vs_f:.4f} (sepT耗时是sepF的倍数)",
            f"- 平均 [Time_sepT / Time_A]:  {avg_time_ratio_vs_a:.4f} (sepT耗时是ACORN的倍数)",
            f"- 平均 [Lsearch_sepT / Lsearch_sepF]: {avg_lsearch_ratio_vs_f:.4f} (sepT的Lsearch是sepF的倍数)",
            f"- 平均 [DistCalcs_sepT / DistCalcs_A]:  {avg_dist_ratio_vs_a:.4f} (sepT计算量是ACORN的倍数)",
            "="*78
        ]
        report_content = "\n".join(report_lines)

        # 4. 将报告写入txt文件
        report_filename = f"stats_{base_filename.replace('U_sepF_T_', '', 1).replace('.csv', '.txt')}"
        report_output_path = os.path.join(wins_dir, report_filename)
        
        with open(report_output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"        [SAVE] 统计报告已保存到: {os.path.basename(report_output_path)}")

    else:
        print("        [INFO] 此文件中未找到 'sepT胜出' 的查询任务。")


def main():
    """
    主函数，负责编排整个文件处理流程。
    """
    output_dir, wins_dir = setup_directories(OUTPUT_DIR)
    
    file_pairs = find_file_pairs(U_SEP_DIR, U_A_DIR)
    if not file_pairs:
        print("\n[FINISH] 未找到任何可处理的文件对，程序结束。")
        return

    success_count = 0
    
    print("\n[INFO] 开始处理已匹配的文件对...")
    for u_sep_path, ua_path in file_pairs:
        u_sep_filename = os.path.basename(u_sep_path)
        ua_filename = os.path.basename(ua_path)
        print(f"\n[PROCESS] 正在处理对: [{u_sep_filename}] 与 [{ua_filename}]")
        
        merged_df = process_single_pair(u_sep_path, ua_path, DESIRED_COLUMN_ORDER)
        
        if merged_df is not None:
            output_filename = u_sep_filename.replace('U_sepF_T_', 'U_Usep_A_', 1)
            output_path = os.path.join(output_dir, output_filename)
            merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"    [SAVE] 合并成功！结果已保存到: {os.path.basename(output_path)}")
            success_count += 1
            
            # 分析是否存在"胜出"条目
            analyze_and_save_wins(merged_df, u_sep_filename, wins_dir)

    print(f"\n[FINISH] 全部处理完成！")
    print(f"共成功合并并生成了 {success_count} 个CSV文件。")


if __name__ == '__main__':
    main()