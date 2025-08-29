import pandas as pd
import numpy as np
import warnings

# 忽略一些pandas未来的警告，使输出更整洁
warnings.simplefilter(action='ignore', category=FutureWarning)

def analyze_and_model_small_sample(csv_filepath):
    """
    针对小样本数据的分析和预测方法（已修正BUG并改进）
    """
    try:
        # --- 1. 智能加载和解析复杂表头 (已改进) ---
        print(f"--- 正在加载文件: {csv_filepath} ---")
        df_full = pd.read_csv(csv_filepath, header=None, encoding='utf-8')

        if len(df_full) < 3:
            print("错误: CSV文件内容不足。必须至少包含2行表头和1行数据。")
            return

        header_top = df_full.iloc[0].ffill()
        header_bottom = df_full.iloc[1]

        new_column_names = []
        # [新增] 创建一个从新列名到原始数据集名称的映射
        column_to_dataset_map = {} 
        
        for i in range(1, len(header_top)):
            top = str(header_top[i]).strip()
            bottom = str(header_bottom[i]).strip()
            
            # 构造唯一的列名
            col_name = f"{top}_{bottom}"
            if col_name in new_column_names:
                count = 1
                base_name = col_name
                while col_name in new_column_names:
                    col_name = f"{base_name}_{count}"
                    count += 1
            
            new_column_names.append(col_name)
            # [新增] 将新列名映射到干净的数据集名称
            column_to_dataset_map[col_name] = top

        df = pd.read_csv(csv_filepath, header=None, skiprows=2, index_col=0, encoding='utf-8')
        df.columns = new_column_names
        data = df.T
        data = data.apply(pd.to_numeric, errors='coerce')
        data = data.rename(columns=lambda x: x.strip())
        
        # 兼容不同版本的列名
        if 'avg_trie_path_len_gth' in data.columns:
            data.rename(columns={'avg_trie_path_len_gth': 'trie_avg_path_length'}, inplace=True)
        if 'avg_branching_factor' in data.columns:
            data.rename(columns={'avg_branching_factor': 'trie_avg_branching_factor'}, inplace=True)

        print("--- 数据加载和表头解析成功 ---\n")

    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        return

    # --- 2. 计算增强版衍生指标 (已修复BUG) ---
    print("--- 正在计算增强版衍生指标 ---")
    
    required_cols_for_derivation = [
        'Avg_CandSize', 'trie_avg_path_length', 'Avg_RecurCalls_T', 'Avg_PruneEvents_T',
        'trie_label_cardinality', 'trie_avg_branching_factor'
    ]
    if not all(col in data.columns for col in required_cols_for_derivation):
        print("错误：CSV文件中缺少计算衍生指标所需的列。")
        print(f"需要以下列: {required_cols_for_derivation}")
        return

    data['Work_F'] = data['Avg_CandSize'] * data['trie_avg_path_length']
    
    # *** BUG修复 ***
    # 修复了因分母为0导致样本丢失的问题
    denominator_t = data['Avg_RecurCalls_T'] + data['Avg_PruneEvents_T']
    # 使用 .fillna(0) 来处理 0/0 的情况，这在逻辑上表示剪枝效率为0
    data['Pruning_Efficiency_T'] = (data['Avg_PruneEvents_T'] / denominator_t).fillna(0)
    
    data['Tree_Shape_Factor_V2'] = (data['trie_label_cardinality'] * data['trie_avg_branching_factor']) / data['trie_avg_path_length']
    
    print("衍生指标计算完成。\n")

    # --- 3. 小样本数据分析和预测 ---
    print("--- 正在进行小样本数据分析 ---")
    
    target_metric = 'Ratio_TrieNode(T/F)'
    
    features = [
        'Work_F', 'Avg_RecurCalls_T', 'Pruning_Efficiency_T', 'Tree_Shape_Factor_V2',
        'Avg_CandSize', 'Avg_QuerySize', 'trie_total_nodes', 'trie_label_cardinality',
        'trie_avg_path_length', 'trie_avg_branching_factor'
    ]
    
    if not all(f in data.columns for f in features + [target_metric]):
        print(f"错误：数据中缺少用于分析的特征或目标列。")
        print(f"需要: {features + [target_metric]}")
        return
        
    # 在这里，我们先保留所有数据进行分析，仅在计算相关性时去除NaN
    analysis_data = data.copy()
    
    # --- 4. 相关性分析 ---
    print("\n--- 正在进行相关性分析 ---")
    
    # 仅在计算相关性时丢弃空值，避免影响数据概览
    corr_analysis_data = data.dropna(subset=features + [target_metric])
    
    if corr_analysis_data.empty:
        print("错误：在去除无效值后，没有足够的数据进行相关性分析。")
        return
        
    correlations = corr_analysis_data[features + [target_metric]].corr()[target_metric].drop(target_metric)
    
    correlation_results = pd.DataFrame(correlations).reset_index()
    correlation_results.columns = ['Feature', 'Correlation']
    correlation_results['Abs_Correlation'] = correlation_results['Correlation'].abs()
    correlation_results = correlation_results.sort_values('Abs_Correlation', ascending=False).reset_index(drop=True)
    
    print("特征与目标变量的相关性 (按绝对值排序):")
    print(correlation_results)
    
    # --- 5. 简单预测分析 (已调整) ---
    print("\n--- 潜在关键特征分析 ---")
    
    top_features = correlation_results.head(3)['Feature'].tolist()
    print(f"选择相关性最高的特征: {top_features}")
    
    potential_drivers = []
    for feature in top_features:
        corr_value = correlation_results[correlation_results['Feature'] == feature]['Correlation'].iloc[0]
        if abs(corr_value) > 0.3:
            potential_drivers.append({
                'Feature': feature,
                'Correlation': corr_value,
                # [措辞调整] 结论更保守
                'Potential_Impact': 'High' if abs(corr_value) > 0.7 else 'Moderate' if abs(corr_value) > 0.3 else 'Low'
            })
    
    if potential_drivers:
        pred_df = pd.DataFrame(potential_drivers)
        print("\n潜在影响分析:")
        print(pred_df)
    else:
        print("没有发现显著的相关性特征 (绝对值 > 0.3)")

    # --- 6. 生成最终的文本分析报告 (已改进) ---
    report_filename = "analysis_report_small_sample_revised.txt"
    print(f"\n--- 正在生成文本分析报告: {report_filename} ---")
    
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write("实验数据小样本探索性分析报告\n")
        f.write("="*50 + "\n\n")

        f.write("免责声明：\n本报告基于小样本数据生成，相关性分析结果可能不稳定，\n其结论仅为初步探索性发现，需要更大规模的数据集进行验证。\n\n")

        f.write("1. 数据概览:\n")
        f.write("-" * 30 + "\n")
        # [改进] 使用修复后的 analysis_data，确保所有样本都被统计
        f.write(f"总样本数量: {len(analysis_data)}\n")
        f.write(f"用于相关性分析的有效样本数量: {len(corr_analysis_data)}\n\n")
        
        f.write("2. 关键衍生指标预览:\n")
        f.write("-" * 30 + "\n")
        # [改进] 使用更健壮的映射来获取数据集名称
        analysis_data['dataset'] = analysis_data.index.map(column_to_dataset_map)
        f.write(analysis_data[['dataset', 'Work_F', 'Pruning_Efficiency_T', 'Tree_Shape_Factor_V2']].to_string(float_format="%.2f"))
        f.write("\n\n")

        f.write("3. 特征与目标变量的相关性分析:\n")
        f.write("-" * 30 + "\n")
        f.write(f"该分析显示了各特征与目标 '{target_metric}' 的相关性强弱，用于初步识别潜在的关键影响因素。\n\n")
        f.write(correlation_results.to_string(float_format="%.4f"))
        f.write("\n\n")
        
        f.write("4. 潜在影响因素分析:\n")
        f.write("-" * 30 + "\n")
        if potential_drivers:
            f.write("基于相关性分析，以下是具有中等以上潜在影响的关键特征:\n\n")
            pred_df_to_write = pd.DataFrame(potential_drivers)
            f.write(pred_df_to_write.to_string(index=False))
        else:
            f.write("未发现具有中等以上潜在影响的特征（相关性绝对值 > 0.3）。")
        f.write("\n\n")

    print(f"报告已成功保存到 {report_filename}")
    print("\n=== 分析完成 ===")


# --- 主程序入口 ---
if __name__ == '__main__':
    # 确保 consolidated_results.csv 文件存在于此脚本的相同目录或指定路径下
    csv_file = '/data/fxy/FilterVector/FilterVectorResults/merge_results/improve2/U_nT_rms/consolidated_results.csv' 
    analyze_and_model_small_sample(csv_file)