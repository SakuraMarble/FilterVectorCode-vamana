import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import matplotlib.pyplot as plt
import warnings
import os
import glob
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 忽略警告
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def load_and_prepare_data(dataset_path, global_metrics_df):
    """
    加载单个数据集文件夹中的所有CSV，并根据文件名中的query编号，
    与全局trie指标进行合并。
    """
    dataset_name = os.path.basename(dataset_path)
    print(f"--- 正在处理数据集: '{dataset_name}' ---")

    csv_files = glob.glob(os.path.join(dataset_path, 'U_Compare_*.csv'))
    if not csv_files:
        print(f"⚠️ 在 '{dataset_path}' 中没有找到任何CSV文件，跳过该数据集。\n")
        return None

    df_list = []
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        match = re.search(r'U_Compare_(\w+)_query(\d+)_', filename)
        if match:
            parsed_dataset, query_num = match.group(1), match.group(2)
            query_id = f"query{query_num}"
            merge_key = f"{parsed_dataset}_{query_id}"
            temp_df = pd.read_csv(file_path)
            temp_df['query_id'] = query_id
            temp_df['merge_key'] = merge_key
            df_list.append(temp_df)
        else:
            print(f"   - 警告: 文件名 '{filename}' 格式不匹配，无法提取query编号，已跳过。")

    if not df_list: return None
    combined_data = pd.concat(df_list, ignore_index=True)
    print(f"成功解析并合并了 {len(df_list)} 个CSV文件，共 {len(combined_data)} 条记录。")
    merged_data = pd.merge(combined_data, global_metrics_df, left_on='merge_key', right_index=True, how='left')
    missing_metrics_count = merged_data['trie_total_nodes'].isnull().sum()
    if missing_metrics_count > 0:
        print(f"⚠️ 警告: 有 {missing_metrics_count} 条记录没能匹配到全局trie指标。")
        failed_keys = merged_data[merged_data['trie_total_nodes'].isnull()]['merge_key'].unique()
        print(f"   - 匹配失败的键: {list(failed_keys)}")
    else:
        print("✅ 已成功将全局trie指标精确附加到每一条记录中。\n")
    merged_data.drop(columns=['merge_key'], inplace=True)
    merged_data['dataset_source'] = dataset_name
    return merged_data


def build_decision_tree_selector(data, model_name):
    """
    使用准备好的数据训练决策树模型，并输出报告和图片。包含数据集拆分和模型评估。
    """
    try:
        num_unique_queries = data['query_id'].nunique()
        num_unique_datasets = data['dataset_source'].nunique()
        
        print(f"--- 开始为 '{model_name}' 构建和评估全局决策树 ---")
        print(f"该模型数据来源于 {num_unique_datasets} 个数据集, 共包含 {num_unique_queries} 个独特的查询任务。")
        
        # --- 【新增】详细统计数据来源 ---
        source_breakdown_str = "数据来源详情:\n"
        # 按数据集分组统计
        dataset_counts = data.groupby('dataset_source').size()
        # 按数据集和查询任务分组统计
        query_counts = data.groupby(['dataset_source', 'query_id']).size()
        for dataset, count in dataset_counts.items():
            source_breakdown_str += f"- 数据集: {dataset} (共 {count} 条)\n"
            for q_id, q_count in query_counts.loc[dataset].items():
                source_breakdown_str += f"  - {q_id}: {q_count} 条\n"
        
        print(source_breakdown_str) # 打印到控制台
        # --- 结束新增部分 ---

        # --- 1. 特征工程 ---
        print("--- 正在进行特征工程 ---")
        numeric_cols = [
            'TrieNodePass_T', 'TrieNodePass_F', 'CandSize', 'QuerySize',
            'trie_total_nodes', 'trie_label_cardinality',
            'trie_avg_path_length', 'trie_avg_branching_factor'
        ]
        for col in numeric_cols:
            if col in data.columns: data[col] = pd.to_numeric(data[col], errors='coerce')
        
        data.rename(columns={'CandSize': 'Avg_CandSize', 'QuerySize': 'Avg_QuerySize'}, inplace=True)
        data['Ratio_TrieNode(T/F)'] = data['TrieNodePass_T'] / (data['TrieNodePass_F'] + 1e-9)
        data['Work_F'] = data['Avg_CandSize'] * data['trie_avg_path_length']
        data['Tree_Shape_Factor_V2'] = (data['trie_label_cardinality'] * data['trie_avg_branching_factor']) / data['trie_avg_path_length']
        target_metric = 'Ratio_TrieNode(T/F)'
        data['choose_method_T'] = (data[target_metric] < 1).astype(int)
        
        pre_computable_features = ['Avg_QuerySize', 'Avg_CandSize', 'trie_avg_path_length', 'trie_avg_branching_factor', 'trie_label_cardinality', 'trie_total_nodes', 'Tree_Shape_Factor_V2']
        analysis_data = data.dropna(subset=pre_computable_features + ['choose_method_T'])
        if analysis_data.empty:
            print("❌ 错误：经过数据清洗后，没有可用的训练样本。\n")
            return

        # --- 2. 拆分训练集和测试集 ---
        print("--- 正在拆分训练集和测试集 ---")
        X = analysis_data[pre_computable_features]
        y = analysis_data['choose_method_T']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"数据拆分完成。训练样本数: {len(X_train)}, 测试样本数: {len(X_test)}\n")

        # --- 3. 训练决策树分类器 ---
        print("--- 正在使用训练集训练决策树模型 ---")
        dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
        dt_classifier.fit(X_train, y_train)
        print("模型训练完成。\n")

        # --- 4. 在测试集上进行预测和评估 ---
        print("--- 正在使用测试集评估模型性能 ---")
        y_pred = dt_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, target_names=['方法F (捷径法)', '方法T (递归法)'])
        print(f"模型在测试集上的准确率 (Accuracy): {accuracy:.4f}")
        print("详细分类报告 (Classification Report):")
        print(class_report)

        # --- 5. 演示与可视化 ---
        print("--- 演示：为两个假设的新查询场景进行决策 ---")
        scenario_A = pd.DataFrame([{'Avg_QuerySize': 8.0, 'Avg_CandSize': 50000, 'trie_avg_path_length': 9.5, 'trie_avg_branching_factor': 1.5, 'trie_label_cardinality': 41, 'trie_total_nodes': 300000, 'Tree_Shape_Factor_V2': (41 * 1.5) / 9.5}])
        scenario_B = pd.DataFrame([{'Avg_QuerySize': 2.0, 'Avg_CandSize': 15, 'trie_avg_path_length': 5.5, 'trie_avg_branching_factor': 1.6, 'trie_label_cardinality': 600000, 'trie_total_nodes': 1000000, 'Tree_Shape_Factor_V2': (600000 * 1.6) / 5.5}])
        prediction_A = dt_classifier.predict(scenario_A[pre_computable_features])
        prediction_B = dt_classifier.predict(scenario_B[pre_computable_features])
        decision_A = '选择方法T (递归法)' if prediction_A[0] == 1 else '选择方法F (捷径法)'
        decision_B = '选择方法T (递归法)' if prediction_B[0] == 1 else '选择方法F (捷径法)'
        print(f"场景A (窄深树,长查询,高CandSize) -> 模型决策: {decision_A}")
        print(f"场景B (宽浅树,短查询,低CandSize) -> 模型决策: {decision_B}\n")
        
        img_filename = f"decision_tree_rules_{model_name}.png"
        print(f"--- 正在将决策树规则可视化并保存到: {img_filename} ---")
        plt.figure(figsize=(20, 12))
        plot_tree(dt_classifier, feature_names=pre_computable_features, class_names=['Choose Method F', 'Choose Method T'], filled=True, rounded=True, fontsize=10)
        plt.title(f"自适应查询优化器决策树 - 全局模型 ({model_name})", fontsize=20)
        plt.savefig(img_filename)
        plt.close()
        print(f"图表已成功保存。\n")
        
        # --- 6. 生成详细的文本分析报告 ---
        report_filename = f"analysis_report_{model_name}.txt"
        print(f"--- 正在生成文本分析报告: {report_filename} ---")
        tree_rules_text = export_text(dt_classifier, feature_names=pre_computable_features)
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(f"自适应查询优化器分析报告 (全局模型) - {model_name}\n")
            f.write("="*60 + "\n\n")
            f.write("1. 训练与测试摘要\n")
            f.write("-" * 30 + "\n")
            f.write(f"模型类型: 决策树分类器 (DecisionTreeClassifier)\n")
            f.write(f"总样本数: {len(analysis_data)} (来源于 {num_unique_datasets} 个数据集, {num_unique_queries} 个查询任务)\n")
            f.write(f"训练集样本数: {len(X_train)}\n")
            f.write(f"测试集样本数: {len(X_test)}\n")
            f.write("\n" + source_breakdown_str) # 【修改】将详细来源写入报告
            f.write("\n训练目标: 预测 'choose_method_T' (1: 方法T更优, 0: 方法F更优)\n")
            f.write("\n\n2. 假设场景预测演示...\n")
            f.write("\n\n3. 模型学到的决策规则...\n")
            f.write(tree_rules_text)
            f.write("\n\n")
            f.write("4. 模型在测试集上的性能评估\n")
            f.write("-" * 30 + "\n")
            f.write(f"准确率 (Accuracy): {accuracy:.4f}\n\n")
            f.write("详细分类报告 (Classification Report):\n")
            f.write(class_report)
            f.write("\n\n")
            f.write("报告解读:\n")
            f.write(" - precision (精确率): 对某个方法的预测有多准。\n")
            f.write(" - recall (召回率): 某个方法被模型成功找出来的比例。\n")
            f.write(" - f1-score: 精确率和召回率的调和平均值，是一个综合指标。\n")

        print(f"报告已成功保存到 {report_filename}")

    except Exception as e:
        print(f"处理模型 '{model_name}' 时发生错误: {e}")
    finally:
        print(f"\n=== 全局模型 '{model_name}' 分析完成 ===")


# --- 主程序入口---
if __name__ == '__main__':
    base_dir = '/data/fxy/FilterVector/FilterVectorResults/merge_results/improve2/U_nT_rms'
    global_metrics_file = os.path.join(base_dir, 'consolidated_results.csv')
    if not os.path.exists(base_dir) or not os.path.exists(global_metrics_file):
        print("错误：基础目录或全局指标文件不存在。")
        exit()

    try:
        df_raw = pd.read_csv(global_metrics_file, header=[0, 1], index_col=0)
        cols = df_raw.columns.to_frame()
        level0 = cols[0].str.replace(r'Unnamed:.*', '', regex=True).replace('', np.nan).ffill()
        level1 = cols[1]
        df_raw.columns = pd.MultiIndex.from_arrays([level0, level1])
        df_transposed = df_raw.T
        df_transposed.index = df_transposed.index.map(lambda idx: f"{idx[0]}_{idx[1]}")
        trie_columns_to_keep = ['trie_total_nodes', 'trie_label_cardinality', 'trie_avg_path_length', 'trie_avg_branching_factor']
        global_df = df_transposed[trie_columns_to_keep]
        print("--- 全局trie指标文件加载并转换成功 ---\n")
    except Exception as e:
        print(f"加载或处理全局指标文件时出错: {e}")
        exit()

    all_prepared_data = []
    for entry in os.scandir(base_dir):
        if entry.is_dir():
            dataset_path = entry.path
            prepared_data = load_and_prepare_data(dataset_path, global_df)
            if prepared_data is not None and not prepared_data.empty:
                all_prepared_data.append(prepared_data)
    
    if not all_prepared_data:
        print("错误：未能从任何数据集中加载有效数据，程序终止。")
    else:
        print("\n--- 所有数据集处理完毕，正在合并数据以进行全局训练 ---")
        combined_dataset = pd.concat(all_prepared_data, ignore_index=True)
        print(f"数据合并完成，总样本数: {len(combined_dataset)}\n")
        build_decision_tree_selector(combined_dataset, "ALL_DATASETS")