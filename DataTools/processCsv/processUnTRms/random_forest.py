import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree, export_text
import matplotlib.pyplot as plt
import warnings
import os
import glob
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import PartialDependenceDisplay

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
    
    if not df_list: return None
    combined_data = pd.concat(df_list, ignore_index=True)
    merged_data = pd.merge(combined_data, global_metrics_df, left_on='merge_key', right_index=True, how='left')
    merged_data.drop(columns=['merge_key'], inplace=True)
    merged_data['dataset_source'] = dataset_name
    return merged_data


def build_random_forest_selector_enhanced(data, model_name):
    """
    使用随机森林训练、评估并进行深度分析。
    """
    try:
        # --- 数据准备和特征工程 ---
        print(f"--- 开始为 '{model_name}' 构建、评估并深度分析随机森林 ---")
        
        numeric_cols = [
            'TrieNodePass_T', 'TrieNodePass_F', 'CandSize', 'QuerySize',
            'trie_total_nodes', 'trie_label_cardinality',
            'trie_avg_path_length', 'trie_avg_branching_factor'
        ]
        for col in numeric_cols:
            if col in data.columns: data[col] = pd.to_numeric(data[col], errors='coerce')
        
        data.rename(columns={'CandSize': 'Avg_CandSize', 'QuerySize': 'Avg_QuerySize'}, inplace=True)
        data['Ratio_TrieNode(T/F)'] = data['TrieNodePass_T'] / (data['TrieNodePass_F'] + 1e-9)
        data['Tree_Shape_Factor_V2'] = (data['trie_label_cardinality'] * data['trie_avg_branching_factor']) / data['trie_avg_path_length']
        data['choose_method_T'] = (data['Ratio_TrieNode(T/F)'] < 1).astype(int)
        
        pre_computable_features = ['Avg_QuerySize', 'Avg_CandSize', 'trie_avg_path_length', 'trie_avg_branching_factor', 'trie_label_cardinality', 'trie_total_nodes', 'Tree_Shape_Factor_V2']
        analysis_data = data.dropna(subset=pre_computable_features + ['choose_method_T'])
        if analysis_data.empty:
            print("❌ 错误：数据清洗后无可用样本。\n")
            return

        X = analysis_data[pre_computable_features]
        y = analysis_data['choose_method_T']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        # --- 训练随机森林模型 ---
        print("--- 正在训练随机森林模型 (启用OOB评估) ---")
        rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1, oob_score=True)
        rf_classifier.fit(X_train, y_train)
        oob_score = rf_classifier.oob_score_
        print(f"模型训练完成。袋外分数 (OOB Score): {oob_score:.4f}\n")

        # --- 性能评估 ---
        print("--- 正在使用测试集评估模型性能 ---")
        y_pred = rf_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, target_names=['方法F (捷径法)', '方法T (递归法)'])
        print(f"模型在测试集上的准确率 (Accuracy): {accuracy:.4f}")
        print("详细分类报告:\n", class_report)

        # --- 特征重要性分析 ---
        print("--- 正在分析模型的全局特征重要性 ---")
        importances = rf_classifier.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': pre_computable_features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        print("模型认为的特征重要性排名:\n", feature_importance_df, "\n")
        
        # --- [新增] 部分依赖图 (Partial Dependence Plots) ---
        pdp_filename = f"partial_dependence_plots_{model_name}.png"
        print(f"--- [新增] 正在生成部分依赖图并保存到: {pdp_filename} ---")
        top_features_for_pdp = feature_importance_df['Feature'].head(3).tolist()
        
        fig, ax = plt.subplots(figsize=(15, 5), ncols=len(top_features_for_pdp))
        if len(top_features_for_pdp) == 1:
            ax = [ax]
        
        PartialDependenceDisplay.from_estimator(
            rf_classifier,
            X_train,
            features=top_features_for_pdp,
            ax=ax,
            line_kw={"color": "crimson", "linewidth": 3}
        )
        fig.suptitle('Partial Dependence Plots: Feature Value vs. Probability of Choosing Method T', fontsize=16)
        plt.tight_layout()
        plt.savefig(pdp_filename)
        plt.close()
        print(f"部分依赖图已成功保存。\n")


        # --- 生成最终的增强版报告 ---
        report_filename = f"analysis_report_RF_{model_name}.txt"
        print(f"--- 正在生成增强版文本分析报告: {report_filename} ---")
        
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(f"自适应查询优化器深度分析报告 (随机森林模型) - {model_name}\n")
            f.write("="*60 + "\n\n")
            f.write("1. 模型摘要与性能评估\n")
            f.write("-" * 30 + "\n")
            f.write(f"模型类型: 随机森林分类器 (RandomForestClassifier)\n")
            f.write(f"袋外分数 (OOB Score): {oob_score:.4f} (一个对模型泛化能力的稳健评估)\n")
            f.write(f"测试集准确率 (Accuracy): {accuracy:.4f}\n\n")
            f.write("详细分类报告 (测试集):\n")
            f.write(class_report)
            f.write("\n\n")

            f.write("2. 全局特征重要性分析\n")
            f.write("-" * 30 + "\n")
            f.write("该排名反映了模型在做决策时对每个特征的总体依赖程度。\n重要性得分越高的特征，在整个森林的决策过程中起到的平均作用越大。\n\n")
            f.write(feature_importance_df.to_string())
            f.write("\n\n")
            
            f.write("3. 核心特征决策边界分析 (解读部分依赖图)\n")
            f.write("-" * 30 + "\n")
            f.write(f"部分依赖图 ({pdp_filename}) 可视化了核心特征如何影响模型的决策概率。\n")
            f.write("例如，查看 'Avg_CandSize' 的图，可以看到随着候选集大小的变化，\n模型推荐使用方法T（递归法）的概率是如何上升或下降的，这揭示了模型学到的具体规律。\n\n")

            f.write("4. 随机森林代表树规则示例 (共3棵)\n")
            f.write("-" * 30 + "\n")
            f.write("注意：以下规则仅来自森林100棵树中的3棵代表，用于直观理解模型可能的决策逻辑。\n最终决策由所有树投票产生，其总体行为由特征重要性和部分依赖图更好地概括。\n\n")
            
            for i in range(min(3, len(rf_classifier.estimators_))):
                f.write(f"--- 代表树 #{i+1} ---\n")
                tree_rules_text = export_text(rf_classifier.estimators_[i], feature_names=pre_computable_features)
                f.write(tree_rules_text)
                f.write("\n\n")

        print(f"增强版报告已成功保存到 {report_filename}")

    except Exception as e:
        print(f"处理模型 '{model_name}' 时发生错误: {e}")
    finally:
        print(f"\n=== 全局模型 '{model_name}' 深度分析完成 ===")


# --- 主程序入口 ---
if __name__ == '__main__':
    base_dir = '/data/fxy/FilterVector/FilterVectorResults/merge_results/improve2/U_nT_rms'
    global_metrics_file = os.path.join(base_dir, 'consolidated_results.csv')
    if not os.path.exists(base_dir) or not os.path.exists(global_metrics_file):
        print("错误：基础目录或全局指标文件不存在。")
        exit()

    # 加载全局指标文件
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
    except Exception as e:
        print(f"加载或处理全局指标文件时出错: {e}")
        exit()

    # 加载所有数据集
    all_prepared_data = []
    for entry in os.scandir(base_dir):
        if entry.is_dir():
            dataset_path = entry.path
            prepared_data = load_and_prepare_data(dataset_path, global_df)
            if prepared_data is not None and not prepared_data.empty:
                all_prepared_data.append(prepared_data)
    
    # 运行增强版分析
    if not all_prepared_data:
        print("错误：未能从任何数据集中加载有效数据，程序终止。")
    else:
        print("\n--- 所有数据集处理完毕，正在合并数据以进行全局训练 ---")
        combined_dataset = pd.concat(all_prepared_data, ignore_index=True)
        print(f"数据合并完成，总样本数: {len(combined_dataset)}\n")
        build_random_forest_selector_enhanced(combined_dataset, "ALL_DATASETS_RF")