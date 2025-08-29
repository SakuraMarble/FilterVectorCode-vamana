# 比较两种算法——“UNG”和“UNG+ACORN”/"UNG_MEP"——在相同参数下的性能表现，并输出为文件
import pandas as pd
import os
import re
import glob

# ==============================================================================
# --- 配置区 ---
DATASETS_TO_PROCESS = ['arxiv_1_3_10']
dataset = "arxiv_1_3_10"
FILE_DIR = "UNG_MEP"
BASE_RESULTS_DIR = '/data/fxy/FilterVector/FilterVectorResults'
OUTPUT_DIR = "/data/fxy/FilterVector/FilterVectorResults/merge_results/improve2/"+FILE_DIR+"/"


# ==============================================================================


def process_dataframe(df):
    """
    核心处理函数：对于每个QueryID，找到达到最大Recall的记录，
    如果有多条，则选择Time_ms最小的。
    """
    if df.empty:
        return pd.DataFrame()

    # 确保关键列是数值类型
    for col in ['Recall', 'Time_ms', 'QueryID']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=['Recall', 'Time_ms', 'QueryID'], inplace=True)
    df['QueryID'] = df['QueryID'].astype(int)

    # 排序后，每个QueryID分组的第一行就是最优结果
    df_sorted = df.sort_values(by=['QueryID', 'Recall', 'Time_ms'], ascending=[True, False, True])
    optimal_results = df_sorted.drop_duplicates(subset=['QueryID'], keep='first')
    
    return optimal_results

def main():
    """
    主函数，驱动整个匹配、处理和合并流程。
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] 结果将保存在: {os.path.abspath(OUTPUT_DIR)}")

    for dataset in DATASETS_TO_PROCESS:
        print(f"\n{'='*25}\n[INFO] 开始处理数据集: {dataset}\n{'='*25}")
        
        # 以 FILE_DIR 为基准，查找所有实验目录
        acorn_exp_dirs = glob.glob(os.path.join(BASE_RESULTS_DIR, FILE_DIR, dataset, '*'))
        
        if not acorn_exp_dirs:
            print(f"[WARN] 在 {dataset} 数据集下未找到 'FILE_DIR' 的实验目录。")
            continue

        print(f"[INFO] 找到 {len(acorn_exp_dirs)} 个 'FILE_DIR' 实验, 开始逐一匹配...")
        
        success_count = 0
        for acorn_dir in acorn_exp_dirs:
            acorn_dir_name = os.path.basename(acorn_dir)
            
            # 从 FILE_DIR 目录名中用正则解析出通用参数
            pattern = re.compile(
               r".*?_query(\d+)_(nT(?:true|false))_(mep(?:true|false))_th(\d+)_M(\d+)_LB(\d+)_alpha([\d.]+)_C(\d+)_EP(\d+)_Ls(\d+)_Le(\d+)_Lp(\d+)_REPEATs(\d+)"
            )
            match = pattern.match(acorn_dir_name)

            if not match:
                print(f"  [SKIP] 无法从目录名中解析参数: {acorn_dir_name}")
                continue
            
            # 构建对应的 UNG 目录名
            (query, nT, mep,th, M, LB, alpha, C, EP, Ls, Le, Lp, REPEATs) = match.groups()
            ung_dir_name = f"{dataset}_query{query}_{nT}_th{th}_M{M}_LB{LB}_alpha{alpha}_C{C}_EP{EP}_Ls{Ls}_Le{Le}_Lp{Lp}_REPEATs{REPEATs}"
            
            # 查找配对的CSV文件
            acorn_csv_path = next(iter(glob.glob(os.path.join(acorn_dir, 'results', 'query_details_repeat*.csv'))), None)
            if not acorn_csv_path:
                print(f"  [SKIP] 在 {acorn_dir_name} 中未找到CSV文件。")
                continue
            
            csv_filename = os.path.basename(acorn_csv_path)
            ung_csv_path = os.path.join(BASE_RESULTS_DIR, 'UNG', dataset, ung_dir_name, 'results', csv_filename)

            if not os.path.exists(ung_csv_path):
                print(f"  [SKIP] 未找到配对的UNG文件: {ung_csv_path}")
                continue

            print(f"  [MATCH] 成功配对，正在处理: {acorn_dir_name}")

            try:
                # --- 数据处理与筛选 ---
                # 处理 UNG
                ung_df = pd.read_csv(ung_csv_path)
                ung_optimal = process_dataframe(ung_df)
                ung_final = ung_optimal[['QueryID', 'repeat', 'Lsearch', 'Time_ms', 'DistCalcs', 'Recall', 'NumEntries','NumNodeVisited']].rename(columns={
                    'repeat': 'repeat_U', 'Lsearch': 'Lsearch_U', 'Time_ms': 'Time(ms)_U', 'DistCalcs': 'DistCalcs_U', 'Recall': 'Recall_U', 'NumEntries':'NumEntries_U','NumNodeVisited':'NumNodeVisited_U'
                })

                # 处理 UNG+ACORN
                acorn_df = pd.read_csv(acorn_csv_path)
                acorn_optimal = process_dataframe(acorn_df)
                acorn_final = acorn_optimal[['QueryID', 'repeat', 'Lsearch', 'Time_ms', 'DistCalcs', 'Recall', 'NumEntries','NumNodeVisited']].rename(columns={
                    'repeat': 'repeat_AU', 'Lsearch': 'Lsearch_AU', 'Time_ms': 'Time(ms)_AU', 'DistCalcs': 'DistCalcs_AU', 'Recall': 'Recall_AU', 'NumEntries':'NumEntries_AU','NumNodeVisited':'NumNodeVisited_AU'
                })

                # --- 合并与保存 ---
                if ung_final.empty or acorn_final.empty:
                    print("  [WARN] 其中一个文件处理后无有效数据，跳过此对。")
                    continue

                merged_df = pd.merge(ung_final, acorn_final, on='QueryID', how='outer')
                
                final_columns_order = [
                    'QueryID', 'repeat_U', 'Lsearch_U', 'Time(ms)_U', 'DistCalcs_U', 'Recall_U','NumEntries_U','NumNodeVisited_U',
                    'repeat_AU', 'Lsearch_AU', 'Time(ms)_AU', 'DistCalcs_AU', 'Recall_AU','NumEntries_AU','NumNodeVisited_AU'
                ]
                merged_df = merged_df.reindex(columns=final_columns_order)
                merged_df.sort_values(by='QueryID', inplace=True)
                
                # 使用 UNG+ACORN 的文件夹名来命名输出文件，确保文件名唯一，避免结果被覆盖
                output_filename = os.path.join(OUTPUT_DIR, f"U_UA_{acorn_dir_name}.csv")
                
                merged_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
                success_count += 1

            except Exception as e:
                print(f"  [ERROR] 处理文件对时出错: {e}")

        print(f"\n[SUCCESS] 数据集 '{dataset}' 处理完成！共成功处理并生成了 {success_count} 个结果文件。")


if __name__ == '__main__':
    main()
