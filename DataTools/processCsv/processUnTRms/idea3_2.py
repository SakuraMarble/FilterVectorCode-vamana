# 将 FilterVectorResults/merge_results/improve2/U_nT_rms中某个数据集的几个处理好的csv文件整理，求得Ratio
import os
import pandas as pd
import numpy as np

# ==============================================================================
# 配置区
dataset="celeba"
path_to_your_csvs = f"/data/fxy/FilterVector/FilterVectorResults/merge_results/improve2/U_nT_rms/{dataset}"
output_txt_file = f"/data/fxy/FilterVector/FilterVectorResults/merge_results/improve2/U_nT_rms/{dataset}/analysis_results_full.txt"
output_csv_file = f"/data/fxy/FilterVector/FilterVectorResults/merge_results/improve2/U_nT_rms/{dataset}/analysis_summary_transposed.csv"
# ==============================================================================

print(f"开始处理文件夹: {path_to_your_csvs}")

# 检查路径是否存在
if not os.path.isdir(path_to_your_csvs):
    print(f"错误：文件夹路径 '{path_to_your_csvs}' 不存在或不是一个文件夹。")
    exit()

# 获取文件夹下所有的 CSV 文件列表
try:
    csv_files = [os.path.join(path_to_your_csvs, f) for f in os.listdir(path_to_your_csvs) if f.endswith('.csv')]
except Exception as e:
    print(f"读取文件夹时出错: {e}")
    csv_files = []

if not csv_files:
    print(f"在文件夹 '{path_to_your_csvs}' 中没有找到任何 CSV 文件。")
else:
    # 创建一个列表来存储每个文件的分析结果字典
    all_files_summary = []

    # 使用 'w' 模式打开文件，如果文件已存在则会覆盖。
    with open(output_txt_file, 'w', encoding='utf-8') as f:
        f.write("=============== 三算法全方位对比分析报告 ===============\n\n")
        
        # 遍历并处理每一个 CSV 文件
        for file_path in csv_files:
            try:
                file_name = os.path.basename(file_path)
                f.write(f"--------------------------------------------------\n")
                f.write(f"文件: {file_name}\n")
                f.write(f"--------------------------------------------------\n")

                df = pd.read_csv(file_path)
                print(df.head())

                # 筛选出满足新条件 (Recall_T_RMS >= Recall_T >= Recall_F) 的数据行
                condition = (df['Recall_T'] >= df['Recall_F']) & (df['Recall_T_RMS'] >= df['Recall_T'])
                condition_met_df = df[condition].copy()

                num_rows_condition_met = len(condition_met_df)
                
                f.write(f"满足 'Recall_T_RMS >= Recall_T >= Recall_F' 条件的总行数: {num_rows_condition_met}\n")

                if num_rows_condition_met > 0:
                    # --- 为防止除以0，准备好所有可能作为分母的列 ---
                    time_f_no_zero = condition_met_df['Time_F(ms)'].replace(0, np.nan)
                    get_entry_f_no_zero = condition_met_df['Get_Entry_Time_F(ms)'].replace(0, np.nan)
                    distcalcs_f_no_zero = condition_met_df['DistCalcs_F'].replace(0, np.nan)
                    trienode_f_no_zero = condition_met_df['TrieNodePass_F'].replace(0, np.nan)

                    time_t_no_zero = condition_met_df['Time_T(ms)'].replace(0, np.nan)
                    get_entry_t_no_zero = condition_met_df['Get_Entry_Time_T(ms)'].replace(0, np.nan)
                    distcalcs_t_no_zero = condition_met_df['DistCalcs_T'].replace(0, np.nan)
                    trienode_t_no_zero = condition_met_df['TrieNodePass_T'].replace(0, np.nan)

                    # --- 计算所有比率 ---
                    # 1. T vs. F (第一阶段提升)
                    condition_met_df['Time_Ratio_T_vs_F'] = condition_met_df['Time_T(ms)'] / time_f_no_zero
                    condition_met_df['Get_Entry_Ratio_T_vs_F'] = condition_met_df['Get_Entry_Time_T(ms)'] / get_entry_f_no_zero
                    condition_met_df['DistCalcs_Ratio_T_vs_F'] = condition_met_df['DistCalcs_T'] / distcalcs_f_no_zero
                    condition_met_df['TrieNode_Ratio_T_vs_F'] = condition_met_df['TrieNodePass_T'] / trienode_f_no_zero
                    
                    # 2. T_RMS vs. T (第二阶段提升)
                    condition_met_df['Time_Ratio_TRMS_vs_T'] = condition_met_df['Time_T_RMS(ms)'] / time_t_no_zero
                    condition_met_df['Get_Entry_Ratio_TRMS_vs_T'] = condition_met_df['Get_Entry_Time_T_RMS(ms)'] / get_entry_t_no_zero
                    condition_met_df['DistCalcs_Ratio_TRMS_vs_T'] = condition_met_df['DistCalcs_T_RMS'] / distcalcs_t_no_zero
                    condition_met_df['TrieNode_Ratio_TRMS_vs_T'] = condition_met_df['TrieNodePass_T_RMS'] / trienode_t_no_zero

                    # 3. T_RMS vs. F (总提升)
                    condition_met_df['Time_Ratio_TRMS_vs_F'] = condition_met_df['Time_T_RMS(ms)'] / time_f_no_zero
                    condition_met_df['Get_Entry_Ratio_TRMS_vs_F'] = condition_met_df['Get_Entry_Time_T_RMS(ms)'] / get_entry_f_no_zero
                    condition_met_df['DistCalcs_Ratio_TRMS_vs_F'] = condition_met_df['DistCalcs_T_RMS'] / distcalcs_f_no_zero
                    condition_met_df['TrieNode_Ratio_TRMS_vs_F'] = condition_met_df['TrieNodePass_T_RMS'] / trienode_f_no_zero

                    # 4. 其他
                    condition_met_df['M1TrieReNode_F/TrieNodePass_F'] = condition_met_df['M1TrieReNode_F'] / trienode_f_no_zero

                    # --- 计算所有平均值 ---
                    avg_metrics = {
                        'FileName': file_name,
                        'FilteredRows': num_rows_condition_met,
                        # 基础指标
                        'Avg_QuerySize': condition_met_df['QuerySize'].mean(),
                        'Avg_CandSize': condition_met_df['CandSize'].mean(),
                        # T vs F 比率
                        'Ratio_Time(T/F)': condition_met_df['Time_Ratio_T_vs_F'].mean(),
                        'Ratio_GetEntry(T/F)': condition_met_df['Get_Entry_Ratio_T_vs_F'].mean(),
                        'Ratio_TrieNode(T/F)': condition_met_df['TrieNode_Ratio_T_vs_F'].mean(),
                        # T_RMS vs T 比率
                        'Ratio_Time(TRMS/T)': condition_met_df['Time_Ratio_TRMS_vs_T'].mean(),
                        'Ratio_GetEntry(TRMS/T)': condition_met_df['Get_Entry_Ratio_TRMS_vs_T'].mean(),
                        'Ratio_TrieNode(TRMS/T)': condition_met_df['TrieNode_Ratio_TRMS_vs_T'].mean(),
                        # T_RMS vs F 比率
                        'Ratio_Time(TRMS/F)': condition_met_df['Time_Ratio_TRMS_vs_F'].mean(),
                        'Ratio_GetEntry(TRMS/F)': condition_met_df['Get_Entry_Ratio_TRMS_vs_F'].mean(),
                        'Ratio_TrieNode(TRMS/F)': condition_met_df['TrieNode_Ratio_TRMS_vs_F'].mean(),
                        # 独立指标
                        'Avg_SuccessChecks_F': condition_met_df['SuccessChecks_F'].mean(),
                        'Avg_HitRatio_F': condition_met_df['HitRatio_F'].mean(),
                        'Avg_RecurCalls_T': condition_met_df['RecurCalls_T'].mean(),
                        'Avg_PruneEvents_T': condition_met_df['PruneEvents_T'].mean(),
                        'Avg_PruneEff_T': condition_met_df['PruneEff_T'].mean(),
                        'Avg_RecurCalls_T_RMS': condition_met_df['RecurCalls_T_RMS'].mean(),
                        'Avg_PruneEvents_T_RMS': condition_met_df['PruneEvents_T_RMS'].mean(),
                        'Avg_PruneEff_T_RMS': condition_met_df['PruneEff_T_RMS'].mean()
                    }

                    # 将当前文件的结果添加到汇总列表中
                    all_files_summary.append(avg_metrics)

                    # --- 将结果写入TXT文件 ---
                    f.write("\n在满足条件的行中，各项指标的平均值为:\n")
                    f.write(f"  --- 基础指标 ---\n")
                    f.write(f"  - 平均QuerySize: {avg_metrics['Avg_QuerySize']:.2f}\n")
                    f.write(f"  - 平均CandSize: {avg_metrics['Avg_CandSize']:.2f}\n\n")

                    f.write(f"  --- 性能提升分析 (T vs. F) [第一阶段提升] ---\n")
                    f.write(f"  - Time Ratio (T/F): {avg_metrics['Ratio_Time(T/F)']:.2f}\n")
                    f.write(f"  - Get Entry Time Ratio (T/F): {avg_metrics['Ratio_GetEntry(T/F)']:.2f}\n")
                    f.write(f"  - TrieNodePass Ratio (T/F): {avg_metrics['Ratio_TrieNode(T/F)']:.2f}\n\n")

                    f.write(f"  --- 性能提升分析 (T_RMS vs. T) [第二阶段提升] ---\n")
                    f.write(f"  - Time Ratio (T_RMS/T): {avg_metrics['Ratio_Time(TRMS/T)']:.2f}\n")
                    f.write(f"  - Get Entry Time Ratio (T_RMS/T): {avg_metrics['Ratio_GetEntry(TRMS/T)']:.2f}\n")
                    f.write(f"  - TrieNodePass Ratio (T_RMS/T): {avg_metrics['Ratio_TrieNode(TRMS/T)']:.2f}\n\n")

                    f.write(f"  --- 性能提升分析 (T_RMS vs. F) [总提升] ---\n")
                    f.write(f"  - Time Ratio (T_RMS/F): {avg_metrics['Ratio_Time(TRMS/F)']:.2f}\n")
                    f.write(f"  - Get Entry Time Ratio (T_RMS/F): {avg_metrics['Ratio_GetEntry(TRMS/F)']:.2f}\n")
                    f.write(f"  - TrieNodePass Ratio (T_RMS/F): {avg_metrics['Ratio_TrieNode(TRMS/F)']:.2f}\n\n")

                    f.write(f"  --- 各算法独立指标均值 ---\n")
                    f.write(f"  - F   -> SuccessChecks: {avg_metrics['Avg_SuccessChecks_F']:.2f}, HitRatio: {avg_metrics['Avg_HitRatio_F']:.2f}\n")
                    f.write(f"  - T_RMS -> RecurCalls: {avg_metrics['Avg_RecurCalls_T_RMS']:.2f}, PruneEvents: {avg_metrics['Avg_PruneEvents_T_RMS']:.2f}, PruneEff: {avg_metrics['Avg_PruneEff_T_RMS']:.2f}\n")

                else:
                    f.write("该文件中没有找到满足条件的行。\n")

                f.write("\n\n")

            except Exception as e:
                f.write(f"处理文件 {file_name} 时发生错误: {e}\n\n")

    print(f"文本报告分析完成！结果已成功保存到文件: {output_txt_file}")

    # 在所有文件处理完毕后，创建并保存汇总的CSV文件
    if all_files_summary:
        summary_df = pd.DataFrame(all_files_summary)
        summary_df.set_index('FileName', inplace=True)
        transposed_df = summary_df.T
        transposed_df.to_csv(output_csv_file, encoding='utf-8-sig')
        print(f"汇总CSV生成完毕！转置后的结果已保存到: {output_csv_file}")
    else:
        print("没有可用于生成汇总CSV的数据。")