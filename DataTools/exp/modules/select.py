# modules/select.py (最终优化版)

import pandas as pd
import os
import numpy as np

def run_selection(paths, params):
    """
    根据定义好的策略（direct, acorn-slow, acorn-fast, percentile, center_select）挑选查询。
    """
    print("\n[步骤 2/5] 开始根据策略挑选查询任务...")

    # 从字典加载配置
    full_results_csv = paths['output_csv_path']
    output_csv_path = paths['selected_queries_file']
    output_dir = paths['output_dir']
    
    SELECTION_MODE = params['selection_mode']
    QUERY_SIZES_TO_PROCESS = params['query_sizes_to_process']
    NUM_QUERIES_PER_SIZE = params['num_queries_per_size']

    try:
        print(f"正在加载数据文件: {full_results_csv}")
        df = pd.read_csv(full_results_csv)
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 -> {full_results_csv}")
        raise

    print("数据加载成功。")
    print(f"当前选择模式: {SELECTION_MODE}")
    all_selected_ids = []

    for size in QUERY_SIZES_TO_PROCESS:
        print(f"\n正在处理查询长度 (QuerySize) = {size}...")

        df_size = df[df['1_QueryAttrLength'] == size].copy()

        if df_size.empty:
            print(f"警告: 未找到任何查询长度为 {size} 的数据，跳过。")
            continue

        # 根据模式选择查询
        if SELECTION_MODE == 'direct':
            selected_queries = df_size.head(NUM_QUERIES_PER_SIZE)
            print(f"按 'direct' 模式，直接选取前 {NUM_QUERIES_PER_SIZE} 个查询。")
        
        else: # 对于所有需要比率的模式
            df_size['Ratio_ACORN_UNG'] = np.divide(df_size['Time_ACORN-gamma_ms'], df_size['Time_UNG_ms'])
            df_size.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_size.dropna(subset=['Ratio_ACORN_UNG'], inplace=True)

            if SELECTION_MODE in ['acorn-slow', 'acorn-fast']:
                ascending_order = (SELECTION_MODE == 'acorn-fast')
                sorted_df = df_size.sort_values(by='Ratio_ACORN_UNG', ascending=ascending_order)
                selected_queries = sorted_df.head(NUM_QUERIES_PER_SIZE)
                print(f"按 '{SELECTION_MODE}' 模式，选取比率{'最低' if ascending_order else '最高'}的 {NUM_QUERIES_PER_SIZE} 个查询。")

            elif SELECTION_MODE == 'percentile':
                min_p = params.get('min_percentile', 0.0)
                max_p = params.get('max_percentile', 1.0)
                lower_bound = df_size['Ratio_ACORN_UNG'].quantile(min_p)
                upper_bound = df_size['Ratio_ACORN_UNG'].quantile(max_p)
                percentile_df = df_size[(df_size['Ratio_ACORN_UNG'] >= lower_bound) & (df_size['Ratio_ACORN_UNG'] <= upper_bound)]
                
                num_available = len(percentile_df)
                num_to_select = min(NUM_QUERIES_PER_SIZE, num_available)
                
                print(f"按 'percentile' 模式，在 {min_p*100:.0f}%-{max_p*100:.0f}% 范围内共有 {num_available} 个查询，从中随机选取 {num_to_select} 个。")
                selected_queries = percentile_df.sample(n=num_to_select, random_state=42)

            elif SELECTION_MODE == 'center_select':
                center_p = params.get('center_percentile', 0.5)
                print(f"按 'center_select' 模式，围绕 {center_p*100:.0f}% 难度中心点选取 {NUM_QUERIES_PER_SIZE} 个查询...")

                sorted_df = df_size.sort_values(by='Ratio_ACORN_UNG', ascending=True).reset_index(drop=True)
                num_available = len(sorted_df)

                if num_available < NUM_QUERIES_PER_SIZE:
                    print(f"警告: 可用查询数({num_available})少于目标数({NUM_QUERIES_PER_SIZE})，将选取所有可用查询。")
                    selected_queries = sorted_df
                else:
                    center_index = int(num_available * center_p)
                    half_window = NUM_QUERIES_PER_SIZE // 2
                    
                    start_index = center_index - half_window
                    end_index = center_index + half_window
                    
                    # 处理边界情况
                    if start_index < 0:
                        start_index = 0
                        end_index = NUM_QUERIES_PER_SIZE
                    if end_index > num_available:
                        end_index = num_available
                        start_index = end_index - NUM_QUERIES_PER_SIZE
                        
                    selected_queries = sorted_df.iloc[start_index:end_index]

            else:
                print(f"错误: 无效的 SELECTION_MODE: '{SELECTION_MODE}'")
                exit()

        selected_ids = selected_queries['QueryID'].tolist()
        print(f"成功为长度 {size} 挑选了 {len(selected_ids)} 个查询ID。")
        all_selected_ids.extend(selected_ids)

    # 保存结果
    if not all_selected_ids:
        print("\n⚠️ 处理完成，但没有挑选出任何查询ID。")
    else:
        print(f"\n总共挑选了 {len(all_selected_ids)} 个查询ID。")
        final_selected_df = df[df['QueryID'].isin(all_selected_ids)]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        final_selected_df.to_csv(output_csv_path, index=False, float_format='%.4f')
        print(f"✅ 挑选结果已成功保存到: {os.path.abspath(output_csv_path)}")