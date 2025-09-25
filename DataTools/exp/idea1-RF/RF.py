import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import export_text
from sklearn.inspection import PartialDependenceDisplay

# --- 全局配置 ---
BASE_RESULTS_DIR = "/data/fxy/FilterVector/FilterVectorResults"
DATASET_NAME = "celeba"

# 根据上述配置自动生成路径
UNG_RESULTS_BASE_DIR = os.path.join(BASE_RESULTS_DIR, "UNG", DATASET_NAME)
MODEL_OUTPUT_BASE_DIR = os.path.join(BASE_RESULTS_DIR, "UNG",DATASET_NAME,"SelectModels/idea1")


def find_result_pairs(results_dir):
   """
   扫描结果目录，查找并匹配 nT=true 和 nT=false 的成对实验结果。
   """
   print(f"--- 正在扫描目录 '{results_dir}' 以查找成对的实验结果 ---")
   
   all_dirs = glob.glob(os.path.join(results_dir, "*_nT*_*"))
   pairs_map = {}
   
   for dir_path in all_dirs:
      key = re.sub(r"_nT(true|false)", "", dir_path)
      key = re.sub(r"_rms(true|false)", "", key)
      if key not in pairs_map:
            pairs_map[key] = {'false': None, 'true': None}
      if "_nTtrue_" in dir_path:
            pairs_map[key]['true'] = dir_path
      elif "_nTfalse_" in dir_path:
            pairs_map[key]['false'] = dir_path

   pairs = []
   for key, value in pairs_map.items():
      if value['false'] and value['true']:
            pairs.append((value['false'], value['true']))
            
   if not pairs:
      print("❌ 警告: 未找到任何成对的 (nT=true vs nT=false) 实验结果目录。")
   else:
      print(f"✅ 成功找到 {len(pairs)} 对实验结果。\n")
   return pairs

def parse_build_log(index_dir):
   """
   从索引元数据文件 (meta) 中解析出Trie树的静态特征。
   """
   log_file_path = os.path.join(index_dir, "index_files", "meta")
   trie_stats = {
      'trie_total_nodes': np.nan,
      'trie_label_cardinality': np.nan,
      'trie_avg_path_length': np.nan,
      'trie_avg_branching_factor': np.nan
   }
   
   if not os.path.exists(log_file_path):
      print(f"❌ 警告: 找不到元数据文件: {log_file_path}")
      return trie_stats

   try:
      with open(log_file_path, 'r', encoding='utf-8') as f:
         for line in f:
               if '=' in line:
                  key, value = line.strip().split('=', 1)
                  if key in trie_stats:
                     try:
                           trie_stats[key] = float(value)
                     except ValueError:
                           print(f"无法将值 '{value}' 转换为浮点数 (key: {key})")
         print(f"从 '{os.path.basename(index_dir)}' 的元数据文件中提取Trie统计信息: {trie_stats}")
   except Exception as e:
      print(f"❌ 解析元数据文件 '{log_file_path}' 时出错: {e}")
   return trie_stats

def load_and_merge_data(path_false, path_true, trie_stats):
   """
   加载成对的CSV结果文件，合并它们，并添加Trie静态特征。
   """
   try:
      csv_false_path_pattern = os.path.join(glob.escape(path_false), "results", "query_details*.csv")
      csv_true_path_pattern = os.path.join(glob.escape(path_true), "results", "query_details*.csv")
      
      csv_false_list = glob.glob(csv_false_path_pattern)
      csv_true_list = glob.glob(csv_true_path_pattern)

      if not csv_false_list or not csv_true_list:
         print(f"❌ 错误: 找不到 'query_details*.csv' 文件。")
         return pd.DataFrame()

      df_false = pd.read_csv(csv_false_list[0])
      df_true = pd.read_csv(csv_true_list[0])
      
      cols_to_keep = ['Lsearch', 'QueryID', 'Time_ms', 'TrieNodePass', 'CandSize', 'QuerySize']
      df_false_filtered = df_false.get(cols_to_keep, pd.DataFrame())
      df_true_filtered = df_true.get(cols_to_keep, pd.DataFrame())
      
      merged_df = pd.merge(df_false_filtered, df_true_filtered, on=['Lsearch', 'QueryID'], suffixes=('_F', '_T'))
      
      for key, value in trie_stats.items():
         merged_df[key] = value
      return merged_df
      
   except Exception as e:
      print(f"加载和合并数据时发生未知错误: {e}")
      return pd.DataFrame()


def build_random_forest_selector_enhanced(data, model_name, output_dir):
   """
   使用随机森林训练、评估、深度分析并保存模型。
   """
   try:
      # --- 数据准备和特征工程 ---
      print(f"\n--- 开始为 '{model_name}' 构建、评估并深度分析随机森林 ---")
      
      data.rename(columns={
         'CandSize_T': 'CandSize', 
         'QuerySize_T': 'QuerySize' 
      }, inplace=True)

      numeric_cols = [
         'TrieNodePass_T', 'TrieNodePass_F', 'CandSize', 'QuerySize',
         'trie_total_nodes', 'trie_label_cardinality',
         'trie_avg_path_length', 'trie_avg_branching_factor'
      ]
      for col in numeric_cols:
         if col in data.columns: data[col] = pd.to_numeric(data[col], errors='coerce')
      
      data.rename(columns={'CandSize': 'Avg_CandSize', 'QuerySize': 'Avg_QuerySize'}, inplace=True)
      
      # --- 基础特征工程 ---
      data['Query_Depth_Ratio'] = data['Avg_QuerySize'] / (data['trie_avg_path_length'] + 1e-9)
      data['Cand_Set_Coverage_Ratio'] = data['Avg_CandSize'] / (data['trie_total_nodes'] + 1e-9)
      data['Query_Path_Density'] = data['Avg_QuerySize'] * data['trie_avg_branching_factor']

      # --- 高级特征工程 ---
      avg_nodes_per_label = data['trie_total_nodes'] / (data['trie_label_cardinality'] + 1e-9)
      data['Cand_Set_Selectivity'] = avg_nodes_per_label / (data['Avg_CandSize'] + 1e-9)
      data['Query_Cand_Ratio'] = data['Avg_QuerySize'] / (data['Avg_CandSize'] + 1e-9)
      data['Avg_CandSize_sq'] = data['Avg_CandSize'] ** 2
      data['Query_Depth_Ratio_sq'] = data['Query_Depth_Ratio'] ** 2
      
      # --- 【新增】最终特征工程：探索更深层次的交互和变换 ---
      print("--- 正在进行最终特征工程 ---")
      # 核心因素交互：候选集大小与查询长度的乘积关系
      data['Cand_x_Query_Interaction'] = data['Avg_CandSize'] * data['Avg_QuerySize']
      
      # 对数变换：捕捉候选集规模的数量级效应
      data['Log_Avg_CandSize'] = np.log1p(data['Avg_CandSize'])
      
      # 静态与动态深度交互：树的茂密度与候选集大小的交互
      data['Branching_x_CandSize'] = data['trie_avg_branching_factor'] * data['Avg_CandSize']
      print("✅ 新的最终特征已创建。\n")

      # --- 定义标签和特征集 ---
      data['choose_method_T'] = (data['Time_ms_T'] < data['Time_ms_F']).astype(int)

      # 更新 pre_computable_features 列表，包含所有可用特征
      pre_computable_features = [
         # # 基础动态特征
         # 'Avg_QuerySize', 
         # 'Avg_CandSize', 
         # 基础特征工程
         'Query_Depth_Ratio',
         'Cand_Set_Coverage_Ratio',
         'Query_Path_Density',
         # 高级特征工程
         'Cand_Set_Selectivity',
         'Query_Cand_Ratio',
         'Avg_CandSize_sq',
         'Query_Depth_Ratio_sq',
         # 最终特征工程
         'Cand_x_Query_Interaction',
         'Log_Avg_CandSize',
         'Branching_x_CandSize'
      ]

      analysis_data = data.dropna(subset=pre_computable_features + ['choose_method_T'])
      if analysis_data.empty:
         print("❌ 错误：数据清洗后无可用样本。\n")
         return

      X = analysis_data[pre_computable_features]
      y = analysis_data['choose_method_T']
      
      print(f"数据集准备完成，共有 {len(analysis_data)} 个样本。")
      print(f"使用的特征列表: {pre_computable_features}")
      print(f"类别分布: 方法F={y.value_counts().get(0, 0)} | 方法T={y.value_counts().get(1, 0)}")

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

      # --- 模型训练 ---
      print("--- 正在训练随机森林模型 (启用OOB评估) ---")
      rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, oob_score=True, class_weight='balanced')
      rf_classifier.fit(X_train, y_train)
      print(f"模型训练完成。袋外分数 (OOB Score): {rf_classifier.oob_score_:.4f}\n")

      # --- 保存训练好的模型 ---
      model_filename = os.path.join(output_dir, "idea1_selector_model.joblib")
      print(f"--- 正在将训练好的模型保存到: {model_filename} ---")
      joblib.dump(rf_classifier, model_filename)
      print("✅ 模型已成功保存。\n")

      # --- 转换为 ONNX 格式 ---
      try:
         from skl2onnx import convert_sklearn
         from skl2onnx.common.data_types import FloatTensorType
         print("--- 正在将模型转换为 ONNX 格式 ---")
         onnx_model_filename = os.path.join(output_dir, "trie_method_selector.onnx")
         initial_type = [('float_input', FloatTensorType([None, len(pre_computable_features)]))]
         # Add target_opset to export a model compatible with IR version 9
         onnx_model = convert_sklearn(rf_classifier, initial_types=initial_type, target_opset=15)
         with open(onnx_model_filename, "wb") as f:
               f.write(onnx_model.SerializeToString())
         print(f"✅ 模型已成功导出为 ONNX 格式: {onnx_model_filename}\n")
      except ImportError:
         print(" skl2onnx 未安装，跳过 ONNX 转换。")

      # --- 性能评估 ---
      print("--- 正在使用测试集评估模型性能 ---")
      y_pred = rf_classifier.predict(X_test)
      accuracy = accuracy_score(y_test, y_pred)
      class_report = classification_report(y_test, y_pred, target_names=['应选方法F (捷径法)', '应选方法T (递归法)'])
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
      
      # --- 部分依赖图 (Partial Dependence Plots) ---
      #   pdp_filename = os.path.join(output_dir, f"partial_dependence_plots_{model_name}.png")
      #   print(f"--- 正在生成部分依赖图并保存到: {pdp_filename} ---")
      #   top_features_for_pdp = feature_importance_df['Feature'].head(min(4, len(pre_computable_features))).tolist()
        
      #   try:
      #       fig, ax = plt.subplots(figsize=(20, 5), ncols=len(top_features_for_pdp))
      #       if len(top_features_for_pdp) == 1: ax = [ax]
      #       PartialDependenceDisplay.from_estimator(rf_classifier, X_train, features=top_features_for_pdp, ax=ax, line_kw={"color": "crimson", "linewidth": 3}, target=1)
      #       fig.suptitle('Partial Dependence: Feature Value vs. Probability of Choosing Method T', fontsize=16, y=1.02)
      #       plt.tight_layout()
      #       plt.savefig(pdp_filename)
      #       plt.close()
      #       print(f"✅ 部分依赖图已成功保存。\n")
      #   except Exception as plot_e:
      #       print(f"❌ 绘制部分依赖图时出错: {plot_e}")

      # --- 生成最终的增强版报告 ---
      report_filename = os.path.join(output_dir, f"analysis_report_RF_{model_name}.txt")
      print(f"--- 正在生成增强版文本分析报告: {report_filename} ---")
      
      with open(report_filename, "w", encoding="utf-8") as f:
         f.write(f"自适应查询优化器深度分析报告 (随机森林模型) - {model_name}\n")
         f.write("="*60 + "\n\n")
         f.write("1. 模型摘要与性能评估\n" + "-"*30 + "\n")
         f.write(f"模型类型: 随机森林分类器\n袋外分数 (OOB Score): {rf_classifier.oob_score_:.4f}\n测试集准确率 (Accuracy): {accuracy:.4f}\n\n详细分类报告 (测试集):\n{class_report}\n\n")
         f.write("2. 全局特征重要性分析\n" + "-"*30 + "\n")
         f.write("该排名反映了模型在做决策时对每个特征的总体依赖程度。\n\n" + feature_importance_df.to_string() + "\n\n")
         # f.write("3. 核心特征决策边界分析 (解读部分依赖图)\n" + "-"*30 + "\n")
         # f.write(f"部分依赖图 ({os.path.basename(pdp_filename)}) 可视化了核心特征如何影响模型的决策概率。\n它揭示了模型学到的具体规律和决策边界。\n\n")
         f.write("4. 随机森林代表树规则示例 (共3棵)\n" + "-"*30 + "\n")
         f.write("注意：以下规则仅为示例，最终决策由所有树投票产生。\n\n")
         for i in range(min(3, len(rf_classifier.estimators_))):
               f.write(f"--- 代表树 #{i+1} ---\n")
               f.write(export_text(rf_classifier.estimators_[i], feature_names=pre_computable_features, max_depth=3) + "\n\n")
         f.write("5. 模型部署\n" + "-"*30 + "\n")
         f.write("训练好的模型已保存到以下文件，可用于后续的推理和部署：\n")
         f.write(f"- {model_filename}\n")
         f.write(f"- {onnx_model_filename}\n\n")
         f.write("可以使用 joblib.load() (Python) 或 ONNX Runtime (C++/Python/etc.) 加载模型。\n")

      print(f"✅ 增强版报告已成功保存到 {report_filename}")

   except Exception as e:
      print(f"❌ 处理模型 '{model_name}' 时发生严重错误: {e}")
   finally:
      print(f"\n=== 全局模型 '{model_name}' 深度分析完成 ===")



def main():
   """主执行函数"""
   index_base_dir = os.path.join(UNG_RESULTS_BASE_DIR, "Index")
   results_base_dir = os.path.join(UNG_RESULTS_BASE_DIR, "Results")
   
   if not os.path.isdir(results_base_dir):
      print(f"❌ 错误: 配置的结果目录不存在: '{results_base_dir}'")
      return

   # 创建唯一的输出目录
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   model_output_dir = os.path.join(MODEL_OUTPUT_BASE_DIR, f"{DATASET_NAME}_model")
   os.makedirs(model_output_dir, exist_ok=True)
   print(f"✅ 本次训练的模型及分析报告将保存到: {model_output_dir}\n")

   result_pairs = find_result_pairs(results_base_dir)
   if not result_pairs: return
      
   all_data = []
   
   for path_f, path_t in result_pairs:
      match = re.search(r"Index\[([^\]]+)\]", os.path.basename(path_t))
      if not match:
         print(f"跳过: 无法从 '{os.path.basename(path_t)}' 中提取索引目录名。")
         continue
         
      index_dir_name = match.group(1)
      print(f"\n--- 正在处理数据对 (关联索引: {index_dir_name}) ---")

      trie_static_features = parse_build_log(os.path.join(index_base_dir, index_dir_name))
      merged_data = load_and_merge_data(path_f, path_t, trie_static_features)
      
      if not merged_data.empty:
         all_data.append(merged_data)
         
   if not all_data:
      print("\n❌ 未能成功加载任何数据，程序终止。")
      return
      
   final_dataframe = pd.concat(all_data, ignore_index=True)
   print(f"\n✅ 所有数据加载和合并完成，共获得 {len(final_dataframe)} 条记录用于建模。")
   
   # 传入输出目录
   build_random_forest_selector_enhanced(final_dataframe, "UNG_Global_Selector", model_output_dir)

if __name__ == "__main__":
   main()

