import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mticker

def get_bitmap_time_from_summary(file_path):
   """
   Robustly reads the Bitmap_Computation_Time_ms from the UNG summary file.
   """
   try:
      with open(file_path, 'r') as f:
         for line in f:
               if 'Bitmap_Computation_Time_ms' in line:
                  parts = line.strip().split(',')
                  if len(parts) > 1:
                     return float(parts[1])
      print(f"警告: 在文件 {os.path.basename(file_path)} 中未找到 'Bitmap_Computation_Time_ms'。返回 0.0。")
      return 0.0
   except (FileNotFoundError, Exception) as e:
      print(f"读取文件 {file_path} 时出错: {e}")
      # In the context of the pipeline, it's better to let the main script handle the error
      raise

def generate_overall_plot(paths):
   """
   Generates the main QPS-Recall chart by aggregating data across all selected queries.
   It also saves a detailed performance analysis CSV and text file.
   """
   print("\n[步骤 3/4] 开始生成总体的QPS-Recall图...")

   # Load file paths from the dictionary
   selected_queries_file = paths['selected_queries_file']
   acorn_details_file = paths['acorn_details_file']
   acorn_avg_file = paths['acorn_avg_file']
   ung_nt_false_details_file = paths['ung_nt_false_details_file']
   ung_nt_true_details_file = paths['ung_nt_true_details_file']
   ung_summary_file = paths['ung_summary_file']
   
   output_plot_path = paths['output_plot_overall_path']
   output_analysis_csv = os.path.join(paths['output_dir'], 'analysis_results_overall.csv')
   output_analysis_txt = os.path.join(paths['output_dir'], 'analysis_results_overall.txt')

   ### 2. Load and preprocess data ###
   print("正在加载数据文件...")
   df_acorn = pd.read_csv(acorn_details_file)
   df_acorn_avg = pd.read_csv(acorn_avg_file)
   df_ung_false = pd.read_csv(ung_nt_false_details_file)
   df_ung_true = pd.read_csv(ung_nt_true_details_file)
   ung_bitmap_total_time = get_bitmap_time_from_summary(ung_summary_file)

   # Map search parameters (efs to Lsearch) to align datasets
   acorn_params = sorted(df_acorn['efs'].unique())
   ung_params = sorted(df_ung_false['Lsearch'].unique())
   if len(acorn_params) != len(ung_params):
      print("错误：ACORN和UNG的搜索参数个数不一致，无法按位置匹配。")
      # In a real app, you might raise an error here
      return
      
   param_map = pd.DataFrame({'efs': acorn_params, 'Lsearch': ung_params})
   df_ung_false.rename(columns={'Lsearch': 'search_param'}, inplace=True)
   df_ung_true.rename(columns={'Lsearch': 'search_param'}, inplace=True)
   df_acorn = pd.merge(df_acorn, param_map, on='efs').drop(columns=['efs']).rename(columns={'Lsearch': 'search_param'})
   print("数据加载与参数映射完成。")

   ### 3. Filter queries ###
   print(f"正在从文件加载预选的查询ID: {selected_queries_file}")
   selected_df = pd.read_csv(selected_queries_file)
   selected_query_ids = selected_df['QueryID'].unique().tolist()
   num_selected_queries = len(selected_query_ids)
   print(f"成功加载了 {num_selected_queries} 个查询ID。")
   
   df_acorn = df_acorn[df_acorn['QueryID'].isin(selected_query_ids)]
   df_ung_false = df_ung_false[df_ung_false['QueryID'].isin(selected_query_ids)]
   df_ung_true = df_ung_true[df_ung_true['QueryID'].isin(selected_query_ids)]

   ### 4. Merge data and calculate times ###
   print("正在合并数据并计算总时间...")
   merged_df = pd.merge(df_acorn, df_ung_false, on=['QueryID', 'search_param'], suffixes=('_acorn', '_ung_false'))
   merged_df = pd.merge(merged_df, df_ung_true, on=['QueryID', 'search_param'], suffixes=('', '_ung_true'))

   actual_queries_count = merged_df['QueryID'].nunique()
   if actual_queries_count == 0:
      print("错误: 没有有效的查询数据可以进行分析，请检查输入文件和筛选条件。")
      return

   num_total_queries = df_acorn['QueryID'].nunique()
   # acorn_bitmap_time = df_acorn_avg['FilterMapTime_ms'].iloc[0] / num_total_queries
   acorn_bitmap_time = ung_bitmap_total_time / num_total_queries if num_total_queries > 0 else 0.0
   ung_bitmap_time = ung_bitmap_total_time / num_total_queries if num_total_queries > 0 else 0.0

   merged_df['Time_ACORN-1'] = merged_df['acorn_1_Time_ms'] + acorn_bitmap_time
   merged_df['Time_ACORN-gamma'] = merged_df['acorn_Time_ms'] + acorn_bitmap_time
   merged_df['Time_UNG'] = merged_df['SearchT_ms']
   merged_df['Time_Method1'] = np.minimum(merged_df['SearchT_ms'], merged_df['SearchT_ms_ung_true'])
   term_m2_acorn_part = merged_df['acorn_Time_ms'] + ung_bitmap_time
   merged_df['Time_Method2'] = np.minimum(term_m2_acorn_part, merged_df['SearchT_ms']) + merged_df['FlagT_ms']
   min_search_val = np.minimum.reduce([term_m2_acorn_part, merged_df['SearchT_ms'], merged_df['SearchT_ms_ung_true']])
   merged_df['Time_Method3'] = min_search_val + np.where(min_search_val == merged_df['SearchT_ms_ung_true'], merged_df['FlagT_ms_ung_true'], merged_df['FlagT_ms'])

   merged_df['Recall_ACORN-1'] = merged_df['acorn_1_Recall']
   merged_df['Recall_ACORN-gamma'] = merged_df['acorn_Recall']
   merged_df['Recall_UNG'] = merged_df['Recall']
   merged_df['Recall_Method1'] = np.where(merged_df['EntryGroupT_ms'] <= merged_df['EntryGroupT_ms_ung_true'], merged_df['Recall'], merged_df['Recall_ung_true'])
   merged_df['Recall_Method2'] = np.where(term_m2_acorn_part <= merged_df['SearchT_ms'], merged_df['acorn_Recall'], merged_df['Recall'])
   merged_df['Recall_Method3'] = np.where(min_search_val == term_m2_acorn_part, merged_df['acorn_Recall'], np.where(min_search_val == merged_df['SearchT_ms'], merged_df['Recall'], merged_df['Recall_ung_true']))
   
   ### 5. Group and aggregate ###
   grouped = merged_df.groupby('search_param')
   
   ### 6. Calculate plotting data ###
   print("正在为绘图准备QPS-Recall数据...")
   plot_data = {}
   algorithms = ['ACORN-1', 'ACORN-gamma', 'UNG', 'Method1', 'Method2', 'Method3']
   for alg in algorithms:
      avg_time = grouped[f'Time_{alg}'].mean()
      avg_recall = grouped[f'Recall_{alg}'].mean()
      qps = 1 / (avg_time / 1000.0)
      df = pd.DataFrame({'Recall': avg_recall, 'QPS': qps}).sort_values(by='Recall').reset_index()
      
      first_reach_one_idx = df.index[df['Recall'] >= 0.99999].tolist()
      end_idx = first_reach_one_idx[0] if first_reach_one_idx else df['Recall'].idxmax()
      plot_data[alg] = df.iloc[:end_idx + 1]

   ### 7. Plotting ###
   print("正在绘制图表...")
   plt.figure(figsize=(12, 8))
   plt.rcParams['axes.unicode_minus'] = False
   markers = ['o', 's', '^', 'D', 'v', 'p']
   for i, (alg, data) in enumerate(plot_data.items()):
      plt.plot(data['Recall'], data['QPS'], marker=markers[i], linestyle='-', label=alg)

   plt.title(f'QPS-Recall ({actual_queries_count} Queries)', fontsize=16)
   plt.xlabel('Recall@10', fontsize=12)
   plt.ylabel('QPS', fontsize=12)
   plt.grid(True, which='both', linestyle='--', linewidth=0.5)
   plt.legend(fontsize=10)
   plt.xlim(0.7, 1.0)
   
   ax = plt.gca()
   all_qps_values = pd.concat([data['QPS'] for data in plot_data.values()]).dropna()
   if not all_qps_values.empty:
      y_max = all_qps_values.max()
      ax.set_ylim(bottom=0, top=y_max * 1.05)
   
   locator = mticker.MaxNLocator(nbins='auto', steps=[1, 2, 5, 10], integer=True)
   ax.yaxis.set_major_locator(locator)
   ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
   ax.yaxis.get_major_formatter().set_useOffset(False)
   
   plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
   plt.close() # Close the figure to free memory
   print(f"✅ 总体图表绘制完成！已保存到：{os.path.abspath(output_plot_path)}")

def generate_querysize_plot(paths):
   """
   Generates side-by-side QPS-Recall charts, faceted by query attribute size.
   """
   print("\n[步骤 4/4] 开始生成按QuerySize划分的QPS-Recall图...")

   # Load paths
   selected_queries_file = paths['selected_queries_file']
   acorn_details_file = paths['acorn_details_file']
   acorn_avg_file = paths['acorn_avg_file']
   ung_nt_false_details_file = paths['ung_nt_false_details_file']
   ung_nt_true_details_file = paths['ung_nt_true_details_file']
   ung_summary_file = paths['ung_summary_file']
   output_plot_path = paths['output_plot_querysize_path']
   output_dir = paths['output_dir']

   # --- This section is very similar to the overall plot, with added grouping by QuerySize ---
   print("正在加载和预处理数据...")
   df_acorn = pd.read_csv(acorn_details_file)
   df_acorn_avg = pd.read_csv(acorn_avg_file)
   df_ung_false = pd.read_csv(ung_nt_false_details_file)
   df_ung_true = pd.read_csv(ung_nt_true_details_file)
   ung_bitmap_total_time = get_bitmap_time_from_summary(ung_summary_file)

   acorn_params = sorted(df_acorn['efs'].unique())
   ung_params = sorted(df_ung_false['Lsearch'].unique())
   if len(acorn_params) != len(ung_params):
      print("错误：ACORN和UNG的搜索参数个数不一致。")
      return
      
   param_map = pd.DataFrame({'efs': acorn_params, 'Lsearch': ung_params})
   df_ung_false.rename(columns={'Lsearch': 'search_param'}, inplace=True)
   df_ung_true.rename(columns={'Lsearch': 'search_param'}, inplace=True)
   df_acorn = pd.merge(df_acorn, param_map, on='efs').drop(columns=['efs']).rename(columns={'Lsearch': 'search_param'})
   
   selected_df = pd.read_csv(selected_queries_file)
   selected_query_ids = selected_df['QueryID'].unique().tolist()
   
   df_acorn = df_acorn[df_acorn['QueryID'].isin(selected_query_ids)]
   df_ung_false = df_ung_false[df_ung_false['QueryID'].isin(selected_query_ids)]
   df_ung_true = df_ung_true[df_ung_true['QueryID'].isin(selected_query_ids)]

   merged_df_all = pd.merge(df_acorn, df_ung_false, on=['QueryID', 'search_param'], suffixes=('_acorn', '_ung_false'))
   merged_df_all = pd.merge(merged_df_all, df_ung_true, on=['QueryID', 'search_param'], suffixes=('', '_ung_true'))

   # --- Create subplots ---
   fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
   fig.suptitle('QPS-Recall Analysis by Query Size', fontsize=18)
   query_sizes_to_plot = [2, 5, 8]
   handles, labels = None, None

   print("开始循环处理每个Query Size...")
   for i, q_size in enumerate(query_sizes_to_plot):
      ax = axes[i]
      
      merged_df = merged_df_all[merged_df_all['QuerySize'] == q_size].copy()
      actual_queries_count = merged_df['QueryID'].nunique()
      if actual_queries_count == 0:
         ax.text(0.5, 0.5, f'No data for QuerySize = {q_size}', ha='center', va='center')
         ax.set_title(f'QuerySize = {q_size}')
         continue
         
      num_total_queries = merged_df['QueryID'].nunique()
      # acorn_bitmap_time = df_acorn_avg['FilterMapTime_ms'].iloc[0] / num_total_queries
      acorn_bitmap_time = ung_bitmap_total_time / num_total_queries if num_total_queries > 0 else 0.0
      ung_bitmap_time = ung_bitmap_total_time / num_total_queries if num_total_queries > 0 else 0.0

      merged_df['Time_ACORN-1'] = merged_df['acorn_1_Time_ms'] + acorn_bitmap_time
      merged_df['Time_ACORN-gamma'] = merged_df['acorn_Time_ms'] + acorn_bitmap_time
      merged_df['Time_UNG'] = merged_df['SearchT_ms']
      merged_df['Time_Method1'] = np.minimum(merged_df['SearchT_ms'], merged_df['SearchT_ms_ung_true'])
      term_m2_acorn_part = merged_df['acorn_Time_ms'] + ung_bitmap_time
      merged_df['Time_Method2'] = np.minimum(term_m2_acorn_part, merged_df['SearchT_ms']) + merged_df['FlagT_ms']
      min_search_val = np.minimum.reduce([term_m2_acorn_part, merged_df['SearchT_ms'], merged_df['SearchT_ms_ung_true']])
      merged_df['Time_Method3'] = min_search_val + np.where(min_search_val == merged_df['SearchT_ms_ung_true'], merged_df['FlagT_ms_ung_true'], merged_df['FlagT_ms'])

      merged_df['Recall_ACORN-1'] = merged_df['acorn_1_Recall']
      merged_df['Recall_ACORN-gamma'] = merged_df['acorn_Recall']
      merged_df['Recall_UNG'] = merged_df['Recall']
      merged_df['Recall_Method1'] = np.where(merged_df['EntryGroupT_ms'] <= merged_df['EntryGroupT_ms_ung_true'], merged_df['Recall'], merged_df['Recall_ung_true'])
      merged_df['Recall_Method2'] = np.where(term_m2_acorn_part <= merged_df['SearchT_ms'], merged_df['acorn_Recall'], merged_df['Recall'])
      merged_df['Recall_Method3'] = np.where(min_search_val == term_m2_acorn_part, merged_df['acorn_Recall'], np.where(min_search_val == merged_df['SearchT_ms'], merged_df['Recall'], merged_df['Recall_ung_true']))
      
      grouped = merged_df.groupby('search_param')
      
      plot_data = {}
      algorithms = ['ACORN-1', 'ACORN-gamma', 'UNG', 'Method1', 'Method2', 'Method3']
      for alg in algorithms:
         avg_time = grouped[f'Time_{alg}'].mean()
         avg_recall = grouped[f'Recall_{alg}'].mean()
         qps = 1 / (avg_time / 1000.0)
         df_plot = pd.DataFrame({'Recall': avg_recall, 'QPS': qps}).sort_values(by='Recall').reset_index()
         
         first_reach_one_idx = df_plot.index[df_plot['Recall'] >= 0.99999].tolist()
         end_idx = first_reach_one_idx[0] if first_reach_one_idx else df_plot['Recall'].idxmax()
         plot_data[alg] = df_plot.iloc[:end_idx + 1]

      markers = ['o', 's', '^', 'D', 'v', 'p']
      for j, (alg, data) in enumerate(plot_data.items()):
         ax.plot(data['Recall'], data['QPS'], marker=markers[j], linestyle='-', label=alg)

      ax.set_title(f'QuerySize = {q_size} ({actual_queries_count} Queries)', fontsize=14)
      ax.set_xlabel('Recall@10', fontsize=12)
      ax.grid(True, which='both', linestyle='--', linewidth=0.5)
      ax.set_xlim(0.7, 1.0)
      
      all_qps_values = pd.concat([data['QPS'] for data in plot_data.values()]).dropna()
      if not all_qps_values.empty:
         y_max = all_qps_values.max()
         ax.set_ylim(bottom=0, top=y_max * 1.05)
      
      locator = mticker.MaxNLocator(nbins='auto', steps=[1, 2, 5, 10], integer=True)
      ax.yaxis.set_major_locator(locator)
      ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
      ax.yaxis.get_major_formatter().set_useOffset(False)

      if i == 0:
         handles, labels = ax.get_legend_handles_labels()

   axes[0].set_ylabel('QPS', fontsize=12)

   if handles:
      fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=6, fontsize=12)

   plt.tight_layout(rect=[0, 0, 1, 0.93]) # Adjust rect to make space for suptitle and legend
   
   plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
   plt.close()
   print(f"\n✅ 按QuerySize划分的图表绘制完成！已保存到：{os.path.abspath(output_plot_path)}")


def generate_comparison_plot(config, plot_config):
   """
   根据配置生成跨实验的对比图 (例如，对比不同K值)。确保每个子图都使用与独立图相同的、经过筛选的查询子集。
   """
   # 从主配置和绘图配置中提取信息
   base_dir = config['base_results_dir']
   dataset = config['dataset']
   templates = config['structure_templates']
   
   plot_type = plot_config['type']
   output_filename = plot_config['output_filename']
   title = plot_config['title']
   exp_names_to_compare = plot_config['experiment_names']

   print(f"\n[附加步骤] 开始生成对比图: {title}...")

   # 将主配置文件中的实验列表转换为一个用名字索引的字典
   experiments_by_name = {}
   for exp in config.get('experiments', []):
      if 'name' in exp and 'parameters' in exp:
         resolved_name = exp['name'].format(**exp['parameters'])
         experiments_by_name[resolved_name] = exp

   # 创建子图
   num_plots = len(exp_names_to_compare)
   fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 7), sharey=True)
   if num_plots == 1:
      axes = [axes]
   fig.suptitle(title, fontsize=18)

   handles, labels = None, None

   # 循环处理每个要对比的实验
   for i, exp_name in enumerate(exp_names_to_compare):
      ax = axes[i]
      
      if exp_name not in experiments_by_name:
         print(f"❌ 错误: 在 'experiments' 列表中未找到名为 '{exp_name}' 的实验配置。跳过...")
         ax.text(0.5, 0.5, f'Config not found for\n{exp_name}', ha='center', va='center', color='red')
         ax.set_title(exp_name)
         continue
         
      exp_details = experiments_by_name[exp_name]
      params = exp_details['parameters']
      k_val = params['K']
      
      print(f"\n---------- 正在加载实验 '{exp_name}' (K={k_val}) 的数据 ----------")
      
      try:
         # --- 数据加载与合并逻辑 ---
         acorn_search_params = templates['acorn_search_params'].format(**params)
         acorn_index_handle = templates['acorn_index_handle']
         acorn_base = os.path.join(base_dir, "ACORN", dataset, "Results", f"index_{acorn_index_handle}", f"query{params['query_num']}_{acorn_search_params}", "results")
         acorn_file = templates['acorn_file_pattern'].format(**params)
         ung_index = templates['ung_index_handle'].format(**params)
         ung_gt = templates['ung_gt_handle'].format(**params)
         ung_search_false = templates['ung_search_handle_false'].format(**params)
         ung_search_true = templates['ung_search_handle_true'].format(**params)
         ung_base_false = os.path.join(base_dir, "UNG", dataset, "Results", f"Index[{ung_index}]_GT[{ung_gt}]_Search[{ung_search_false}]", "results")
         ung_base_true = os.path.join(base_dir, "UNG", dataset, "Results", f"Index[{ung_index}]_GT[{ung_gt}]_Search[{ung_search_true}]", "results")

         df_acorn = pd.read_csv(os.path.join(acorn_base, acorn_file))
         df_acorn_avg = pd.read_csv(os.path.join(acorn_base, f"avg_{acorn_file}"))
         df_ung_false = pd.read_csv(os.path.join(ung_base_false, "query_details_repeat1.csv"))
         df_ung_true = pd.read_csv(os.path.join(ung_base_true, "query_details_repeat1.csv"))
         ung_bitmap_total_time = get_bitmap_time_from_summary(os.path.join(ung_base_false, "search_time_summary.csv"))
         
         acorn_params = sorted(df_acorn['efs'].unique())
         ung_params = sorted(df_ung_false['Lsearch'].unique())
         if len(acorn_params) != len(ung_params): continue
         
         param_map = pd.DataFrame({'efs': acorn_params, 'Lsearch': ung_params})
         df_ung_false.rename(columns={'Lsearch': 'search_param'}, inplace=True)
         df_ung_true.rename(columns={'Lsearch': 'search_param'}, inplace=True)
         df_acorn = pd.merge(df_acorn, param_map, on='efs').drop(columns=['efs']).rename(columns={'Lsearch': 'search_param'})
         
         merged_df = pd.merge(df_acorn, df_ung_false, on=['QueryID', 'search_param'], suffixes=('_acorn', '_ung_false'))
         merged_df = pd.merge(merged_df, df_ung_true, on=['QueryID', 'search_param'], suffixes=('', '_ung_true'))

         # 在绘图前，加载并应用查询筛选文件
         analysis_params = exp_details['analysis_params']
         selection_mode = analysis_params['selection_mode']
         
         # 构建筛选文件的路径
         merge_dir = os.path.join(base_dir, "merge_results/experiments",dataset, f"query{params['query_num']}", exp_name)
         # 文件名在 'direct' 和其他模式下可能不同，统一格式
         if selection_mode == 'direct':
               selected_queries_file = os.path.join(merge_dir, f"selected_queries_{selection_mode}.csv")
         else:
               selected_queries_file = os.path.join(merge_dir, f"selected_queries_{selection_mode}.csv")
         
         print(f"应用查询筛选文件: {selected_queries_file}")
         selected_df = pd.read_csv(selected_queries_file)
         selected_query_ids = selected_df['QueryID'].unique().tolist()
         
         # 使用筛选后的查询ID过滤 merged_df
         merged_df = merged_df[merged_df['QueryID'].isin(selected_query_ids)]
         
         actual_queries_count = merged_df['QueryID'].nunique()
         if actual_queries_count == 0:
               print(f"警告: 实验 '{exp_name}' 在应用筛选后没有剩余的查询数据。")
               ax.text(0.5, 0.5, f'No data after filtering for\n{exp_name}', ha='center', va='center')
               ax.set_title(exp_name)
               continue

         # --- 计算绘图数据---
         num_total_queries = merged_df['QueryID'].nunique() # 使用筛选后的查询数
         # acorn_bitmap_time = df_acorn_avg['FilterMapTime_ms'].iloc[0] / num_total_queries
         acorn_bitmap_time = ung_bitmap_total_time / num_total_queries if num_total_queries > 0 else 0.0
         ung_bitmap_time = ung_bitmap_total_time / num_total_queries if num_total_queries > 0 else 0.0

         merged_df['Time_ACORN-1'] = merged_df['acorn_1_Time_ms'] + acorn_bitmap_time
         merged_df['Time_ACORN-gamma'] = merged_df['acorn_Time_ms'] + acorn_bitmap_time
         merged_df['Time_UNG'] = merged_df['SearchT_ms']
         merged_df['Time_Method1'] = np.minimum(merged_df['SearchT_ms'], merged_df['SearchT_ms_ung_true'])
         term_m2_acorn_part = merged_df['acorn_Time_ms'] + ung_bitmap_time
         merged_df['Time_Method2'] = np.minimum(term_m2_acorn_part, merged_df['SearchT_ms']) + merged_df['FlagT_ms']
         min_search_val = np.minimum.reduce([term_m2_acorn_part, merged_df['SearchT_ms'], merged_df['SearchT_ms_ung_true']])
         merged_df['Time_Method3'] = min_search_val + np.where(min_search_val == merged_df['SearchT_ms_ung_true'], merged_df['FlagT_ms_ung_true'], merged_df['FlagT_ms'])

         merged_df['Recall_ACORN-1'] = merged_df['acorn_1_Recall']
         merged_df['Recall_ACORN-gamma'] = merged_df['acorn_Recall']
         merged_df['Recall_UNG'] = merged_df['Recall']
         merged_df['Recall_Method1'] = np.where(merged_df['EntryGroupT_ms'] <= merged_df['EntryGroupT_ms_ung_true'], merged_df['Recall'], merged_df['Recall_ung_true'])
         merged_df['Recall_Method2'] = np.where(term_m2_acorn_part <= merged_df['SearchT_ms'], merged_df['acorn_Recall'], merged_df['Recall'])
         merged_df['Recall_Method3'] = np.where(min_search_val == term_m2_acorn_part, merged_df['acorn_Recall'], np.where(min_search_val == merged_df['SearchT_ms'], merged_df['Recall'], merged_df['Recall_ung_true']))

         grouped = merged_df.groupby('search_param')
         
         plot_data = {}
         algorithms = ['ACORN-1', 'ACORN-gamma', 'UNG', 'Method1', 'Method2', 'Method3']
         for alg in algorithms:
               avg_time = grouped[f'Time_{alg}'].mean()
               avg_recall = grouped[f'Recall_{alg}'].mean()
               qps = 1 / (avg_time / 1000.0)
               df_plot = pd.DataFrame({'Recall': avg_recall, 'QPS': qps}).sort_values(by='Recall').reset_index()
               first_reach_one_idx = df_plot.index[df_plot['Recall'] >= 0.99999].tolist()
               end_idx = first_reach_one_idx[0] if first_reach_one_idx else df_plot['Recall'].idxmax()
               plot_data[alg] = df_plot.iloc[:end_idx + 1]

         markers = ['o', 's', '^', 'D', 'v', 'p']
         for j, (alg, data) in enumerate(plot_data.items()):
               ax.plot(data['Recall'], data['QPS'], marker=markers[j], linestyle='-', label=alg)
         
         ax.set_title(f'K = {k_val} ({actual_queries_count} Queries)', fontsize=14)
         ax.set_xlabel(f'Recall@{k_val}', fontsize=12)
         ax.grid(True, which='both', linestyle='--', linewidth=0.5)
         ax.set_xlim(0.7, 1.0)
         if i == 0:
               handles, labels = ax.get_legend_handles_labels()

      except FileNotFoundError as e:
         print(f"❌ 错误: 找不到实验 '{exp_name}' 的数据文件 -> {e.filename}。跳过。")
         ax.text(0.5, 0.5, f'Data not found for\n{exp_name}', ha='center', va='center', color='red')
         ax.set_title(exp_name)
         continue
   
   # --- 统一格式化和保存 ---
   if num_plots > 0 and 'axes' in locals() and len(axes) > 0:
      axes[0].set_ylabel('QPS', fontsize=12)
   if handles:
      fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=6, fontsize=12)
   plt.tight_layout(rect=[0, 0, 1, 0.93])
   
   output_path = os.path.join(config['base_results_dir'], "merge_results/experiments", dataset, f"query{params['query_num']}", output_filename)
   plt.savefig(output_path, dpi=300, bbox_inches='tight')
   plt.close()
   print(f"✅ 对比图绘制完成！已保存到：{os.path.abspath(output_path)}")
