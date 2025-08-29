# 将/data/fxy/FilterVector/FilterVectorResults/merge_results/improve2/U_nT_rms中各个数据集的结果文件进行整合
# 输出consolidated_results.csv平均值结果
import os
import re
import pandas as pd

RMS_BASE_DIR = '/data/fxy/FilterVector/FilterVectorResults/merge_results/improve2/U_nT_rms'
META_BASE_DIR = '/data/fxy/FilterVector/FilterVectorResults/UNG'
OUTPUT_FILE_DIR = '/data/fxy/FilterVector/FilterVectorResults/merge_results/improve2/U_nT_rms/consolidated_results.csv'

def parse_meta_file(file_path):
    """解析单个meta文件，返回包含其内容的字典。"""
    meta_data = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    try:
                        # 尝试将值转换为浮点数，如果失败则保留为字符串
                        meta_data[key] = float(value)
                    except ValueError:
                        meta_data[key] = value
    except FileNotFoundError:
        print(f"警告: Meta 文件未找到: {file_path}")
    return meta_data

def process_data_and_generate_csv():
    """
    主函数，用于处理所有数据集并生成最终的CSV文件。
    """
    rms_base_dir = RMS_BASE_DIR
    meta_base_dir = META_BASE_DIR
    output_filename = OUTPUT_FILE_DIR

    # 存储所有解析出的数据的字典
    # 格式: {(dataset, query): {metric1: value1, ...}}
    all_data = {}

    # 正则表达式用于从 analysis_summary_transposed.csv 的列名中提取信息
    # 例如: U_Compare_app_reviews_query12_th32_..._REPEATs3.csv
    column_pattern = re.compile(
        r'U_Compare_(?P<dataset>\w+)_query(?P<query_num>\d+)_(?P<params>.*)\.csv'
    )

    print("开始处理数据集...")

    # 1. 遍历 U_nT_rms 目录下的所有数据集文件夹
    if not os.path.isdir(rms_base_dir):
        print(f"错误: 目录不存在 -> {rms_base_dir}")
        return
        
    for dataset_name in os.listdir(rms_base_dir):
        dataset_path = os.path.join(rms_base_dir, dataset_name)
        if os.path.isdir(dataset_path):
            summary_file = os.path.join(dataset_path, 'analysis_summary_transposed.csv')
            
            if not os.path.exists(summary_file):
                print(f"跳过: 在 {dataset_path} 中未找到 analysis_summary_transposed.csv")
                continue

            print(f"正在处理数据集: {dataset_name}")
            
            # 2. 读取 analysis_summary_transposed.csv 文件
            try:
                df_summary = pd.read_csv(summary_file, index_col=0)
            except Exception as e:
                print(f"错误: 读取 {summary_file} 文件失败: {e}")
                continue

            # 3. 遍历该文件的每一列（每个查询任务）
            for col_name in df_summary.columns:
                match = column_pattern.match(col_name)
                if not match:
                    continue
                
                # 提取信息
                extracted_info = match.groupdict()
                query_num = extracted_info['query_num']
                params = extracted_info['params']
                
                current_key = (dataset_name, f"query{query_num}")
                
                # 提取summary文件中的所有指标
                query_metrics = df_summary[col_name].to_dict()

                # 4. 构建meta文件路径并解析
                meta_dir_name = f"{dataset_name}_query{query_num}_nTfalse_{params}"
                meta_file_path = os.path.join(
                    meta_base_dir, dataset_name, meta_dir_name, 'index_files/meta'
                )
                
                meta_metrics = parse_meta_file(meta_file_path)
                
                # 5. 合并来自两个文件的数据
                query_metrics.update(meta_metrics)
                all_data[current_key] = query_metrics

    if not all_data:
        print("处理完成，但未收集到任何数据。请检查目录结构和文件内容。")
        return

    print("数据收集合并完成，开始生成最终CSV文件...")

    # 6. 将字典转换为Pandas DataFrame
    final_df = pd.DataFrame(all_data)

    # 7. 对列进行排序：首先按数据集名称，然后按查询编号（数字大小）
    sorted_columns = sorted(
        final_df.columns, 
        key=lambda col: (col[0], int(re.search(r'\d+', col[1]).group()))
    )
    final_df = final_df.reindex(columns=sorted_columns)

    # 8. 定义行的顺序，以匹配提供的示例格式
    row_order = [
        'trie_total_nodes', 'trie_label_cardinality', 'trie_avg_path_length', 'trie_avg_branching_factor',
        'Avg_QuerySize', 'Avg_CandSize',
        'Ratio_Time(T/F)', 'Ratio_GetEntry(T/F)', 'Ratio_TrieNode(T/F)',
        'Ratio_Time(TRMS/T)', 'Ratio_GetEntry(TRMS/T)', 'Ratio_TrieNode(TRMS/T)',
        'Ratio_Time(TRMS/F)', 'Ratio_GetEntry(TRMS/F)', 'Ratio_TrieNode(TRMS/F)',
        'Avg_SuccessChecks_F', 'Avg_HitRatio_F',
        'Avg_RecurCalls_T', 'Avg_PruneEvents_T', 'Avg_PruneEff_T',
        'Avg_RecurCalls_T_RMS', 'Avg_PruneEvents_T_RMS', 'Avg_PruneEff_T_RMS'
    ]
    
    # 过滤出行顺序列表中真实存在的行，并按该顺序排列
    existing_rows_in_order = [row for row in row_order if row in final_df.index]
    final_df = final_df.reindex(index=existing_rows_in_order)
    
    # 9. 移除索引名称，使第一列的列标题为空
    final_df.index.name = None

    # 10. 保存到CSV文件
    try:
        final_df.to_csv(output_filename)
        print(f"成功！结果已保存到文件: {output_filename}")
    except Exception as e:
        print(f"错误: 保存文件失败: {e}")

if __name__ == '__main__':
    process_data_and_generate_csv()