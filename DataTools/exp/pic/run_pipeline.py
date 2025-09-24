# run_pipeline.py (全新重构版)

import json
import os
import re
import pandas as pd
from modules import process, select, plot

# ==============================================================================
# 1. 辅助函数
# ==============================================================================

def find_experiment_config(config, dataset_name, experiment_name_to_find):
    """
    在复杂的 config 结构中，根据数据集和实验名称找到完整的实验配置。
    """
    try:
        dataset_conf = config['dataset_configurations'][dataset_name]
        for exp_conf in dataset_conf['experiments']:
            # 结合 build_params 和 experiment params 来格式化 name 模板
            build_params = dataset_conf.get('build_params', {})
            all_params = {**build_params, **exp_conf['parameters']}
            current_exp_name = exp_conf['name'].format(**all_params)
            
            if current_exp_name == experiment_name_to_find:
                return exp_conf
        return None
    except KeyError:
        return None

def build_paths_for_exp(config, dataset_name, exp_config):
    """
    为特定的数据集和实验构建所有必需的文件路径。
    """
    global_settings = config['global_settings']
    dataset_conf = config['dataset_configurations'][dataset_name]
    
    base_dir = global_settings['base_results_dir']
    templates = dataset_conf['structure_templates']
    
    # 合并参数：数据集的 build_params 优先于实验自己的 params
    build_params = dataset_conf.get('build_params', {})
    params = exp_config['parameters']
    format_params = {**build_params, **params, 'dataset': dataset_name, **global_settings}
    
    exp_name = exp_config['name'].format(**format_params)

    paths = {}
    merge_dir = os.path.join(base_dir, "merge_results/experiments", dataset_name, f"query{params['query_num']}", exp_name)
    os.makedirs(merge_dir, exist_ok=True)
    
    # --- 构建所有路径 ---
    # ACORN 路径
    acorn_search_params = templates['acorn_search_params'].format(**format_params)
    acorn_index_handle = templates['acorn_index_handle'].format(**format_params)
    total_vectors = 0
    match = re.search(r'N(\d+)', acorn_index_handle)
    if match: total_vectors = int(match.group(1))
    paths['total_vectors'] = total_vectors
    acorn_base = os.path.join(base_dir, "ACORN", dataset_name, "Results", f"index_{acorn_index_handle}", f"query{params['query_num']}_{acorn_search_params}", "results")
    acorn_file = templates['acorn_file_pattern'].format(**format_params)
    paths['acorn_details_file'] = os.path.join(acorn_base, acorn_file)
    paths['acorn_avg_file'] = os.path.join(acorn_base, f"avg_{acorn_file}")

    # UNG 路径
    ung_index = templates['ung_index_handle'].format(**format_params)
    ung_gt = templates['ung_gt_handle'].format(**format_params)
    ung_search_false = templates['ung_search_handle_false'].format(**format_params)
    ung_search_true = templates['ung_search_handle_true'].format(**format_params)
    ung_base_false = os.path.join(base_dir, "UNG", dataset_name, "Results", f"Index[{ung_index}]_GT[{ung_gt}]_Search[{ung_search_false}]", "results")
    ung_base_true = os.path.join(base_dir, "UNG", dataset_name, "Results", f"Index[{ung_index}]_GT[{ung_gt}]_Search[{ung_search_true}]", "results")
    paths['ung_nt_false_details_file'] = os.path.join(ung_base_false, "query_details_repeat1.csv")
    paths['ung_nt_true_details_file'] = os.path.join(ung_base_true, "query_details_repeat1.csv")
    paths['ung_summary_file'] = os.path.join(ung_base_false, "search_time_summary.csv")

    # 输出和中间文件路径
    paths['output_csv_path'] = os.path.join(merge_dir, f"{exp_name}_final_comparison.csv")
    paths['selected_queries_file'] = os.path.join(merge_dir, f"selected_queries_{exp_config['analysis_params']['selection_mode']}.csv")
    paths['attribute_coverage_file'] = templates['attribute_coverage_file'].format(**format_params)
    paths['output_dir'] = merge_dir
    
    return paths

# ==============================================================================
# 2. 数据准备函数 - 为不同类型的图表准备数据
# ==============================================================================

def prepare_qps_recall_data(config, dataset_name, exp_name, plot_type="overall"):
    """
    为 QPS-Recall 子图准备数据。现在支持按不同类型分组。
    """
    exp_config = find_experiment_config(config, dataset_name, exp_name)
    if not exp_config:
        print(f"⚠️ 警告: 在 '{dataset_name}' 中找不到实验 '{exp_name}' 的配置，跳过。")
        return []

    try:
        print("============================================================================")
        print(f"-- 正在为 '{dataset_name}' - '{exp_name}' (类型: {plot_type}) 准备数据...")
        print("============================================================================")
        paths = build_paths_for_exp(config, dataset_name, exp_config)
        
        process.run_processing(paths)
        select.run_selection(paths, exp_config['analysis_params'])
        
        merged_df_all, ung_bitmap_time = plot._load_and_merge_data(paths)
        if merged_df_all.empty:
            print(f"     -- 警告: '{dataset_name}' - '{exp_name}' 没有有效的查询数据。")
            return []

        plot_items = []

        if plot_type == "overall":
            num_queries = merged_df_all['QueryID'].nunique()
            # 调用简化后的函数
            df_metrics = plot._calculate_performance_metrics(merged_df_all, num_queries, ung_bitmap_time)
            plot_data = plot._prepare_plot_data(df_metrics)
            plot_items.append({
                'data': plot_data,
                'title': f"{dataset_name}\n{exp_name}",
                'xlabel': f"Recall@{exp_config['parameters']['K']}"
            })

        elif plot_type == "querysize":
            query_sizes = exp_config['analysis_params']['query_sizes_to_process']
            for q_size in query_sizes:
                df_subset = merged_df_all[merged_df_all['QuerySize_ung_false'] == q_size].copy()
                num_queries = df_subset['QueryID'].nunique()
                if num_queries == 0: continue
                
                # 调用简化后的函数
                df_metrics = plot._calculate_performance_metrics(df_subset, num_queries, ung_bitmap_time)
                plot_data = plot._prepare_plot_data(df_metrics)
                plot_items.append({
                    'data': plot_data,
                    'title': f"{dataset_name}\nQuerySize={q_size}\n({num_queries} queries)",
                    'xlabel': f"Recall@{exp_config['parameters']['K']}"
                })
        
        elif plot_type == "selectivity":
            total_vectors = paths.get('total_vectors')
            if not total_vectors: raise ValueError("Total vectors not found in paths.")
            
            df_coverage = pd.read_csv(paths['attribute_coverage_file'])
            df_coverage['QueryID'] = df_coverage.index
            df_coverage['selectivity'] = df_coverage['coverage_count'] / total_vectors
            bins = [0, 0.001, 0.1, 1.0]
            bin_labels = ['High', 'Medium', 'Low']
            df_coverage['selectivity_group'] = pd.cut(df_coverage['selectivity'], bins=bins, labels=bin_labels, right=True)
            
            # --- 修正的核心：直接合并，不再使用 add_suffix ---
            merged_with_selectivity = pd.merge(merged_df_all, df_coverage[['QueryID', 'selectivity_group']], on='QueryID', how='left')
            merged_with_selectivity.dropna(subset=['selectivity_group'], inplace=True)

            for group_name in bin_labels:
                df_subset = merged_with_selectivity[merged_with_selectivity['selectivity_group'] == group_name].copy()
                num_queries = df_subset['QueryID'].nunique()
                if num_queries == 0: continue
                
                # 调用简化后的函数，不再需要 use_suffixed_cols=True
                df_metrics = plot._calculate_performance_metrics(df_subset, num_queries, ung_bitmap_time)
                plot_data = plot._prepare_plot_data(df_metrics)
                plot_items.append({
                    'data': plot_data,
                    'title': f"{dataset_name}\nSelectivity: {group_name}\n({num_queries} queries)",
                    'xlabel': f"Recall@{exp_config['parameters']['K']}"
                })
        
        return plot_items
        
    except FileNotFoundError as e:
        print(f"❌ 文件未找到错误: {e.filename}")
        print(f"     -- 跳过 '{dataset_name}' - '{exp_name}' 的处理。")
        return []
    except Exception as e:
        print(f"❌ 在为 '{dataset_name}' - '{exp_name}' 准备数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return []

def prepare_build_data(config, item_config):
    """
    为单个 Build Performance 子图准备数据。
    """
    dataset_name = item_config['dataset']
    exp_type = item_config['type']
    build_params_from_task = item_config['build_params']
    group_title = item_config['group_title'] # 用于在图上网格分组的标题
    
    # 获取数据集的完整配置
    dataset_conf = config['dataset_configurations'][dataset_name]
    base_dir = config['global_settings']['base_results_dir']
    
    # 合并参数: 任务指定的参数会覆盖数据集的默认构建参数
    base_build_params = dataset_conf.get('build_params', {})
    params = {**base_build_params, **build_params_from_task, 'dataset': dataset_name}
    
    results = []

    try:
        if exp_type == 'Hybrid':
            # --- 混合模式路径和计算 ---
            params_suffix = (f"UNG_M{params['max_degree']}_LB{params['Lbuild']}_A{params['alpha']}"
                             f"_CE{params['num_cross_edges']}_ACORN_M{params['M']}_G{params['gamma']}")
            hybrid_dir = os.path.join(base_dir, "parall/parall_build", params['dataset'], 
                                      f"query_{params['query_num']}", params_suffix)
            acorn_meta_path = os.path.join(hybrid_dir, "acorn_index_files", "acorn.index.meta")
            ung_meta_path = os.path.join(hybrid_dir, "ung_index_files", "meta")

            acorn_data = plot._parse_acorn_meta(acorn_meta_path)
            ung_data = plot._parse_ung_meta(ung_meta_path)

            if acorn_data and ung_data:
                hybrid_time = max(acorn_data['time'], ung_data['time'])
                hybrid_size = acorn_data['size'] + ung_data['size_add_rb']
                
                results.append({
                    'group_title': group_title,
                    'Algorithm': 'Hybrid',
                    'Index Time (ms)': hybrid_time,
                    'Index Size (MB)': hybrid_size
                })
            else:
                print(f"     -- 警告: 未能完全解析 '{group_title}' 的 Hybrid 构建数据，元数据文件可能缺失。")

        elif exp_type == 'Standalone':
            # --- 独立模式路径和计算 ---
            templates = dataset_conf['structure_templates']
            
            # ACORN 独立模式
            acorn_index_handle = templates['acorn_index_handle'].format(**params)
            acorn_dir = os.path.join(base_dir, "ACORN", dataset_name, "Index", acorn_index_handle)
            acorn_data = plot._parse_acorn_meta(os.path.join(acorn_dir, "acorn.index.meta"))
            if acorn_data:
                results.append({
                    'group_title': group_title, 'Algorithm': 'ACORN', 
                    'Index Time (ms)': acorn_data['time'], 'Index Size (MB)': acorn_data['size']
                })

            # ACORN-1 独立模式
            acorn1_data = plot._parse_acorn_meta(os.path.join(acorn_dir, "acorn1.index.meta"))
            if acorn1_data:
                results.append({
                    'group_title': group_title, 'Algorithm': 'ACORN-1',
                    'Index Time (ms)': acorn1_data['time'], 'Index Size (MB)': acorn1_data['size']
                })

            # UNG 独立模式
            ung_index_handle = templates['ung_index_handle'].format(**params)
            ung_index_dir = os.path.join(base_dir, "UNG", dataset_name, "Index", ung_index_handle) 
            ung_data = plot._parse_ung_meta(os.path.join(ung_index_dir, "index_files", "meta")) 
            if ung_data:
                 results.append({
                    'group_title': group_title, 'Algorithm': 'UNG',
                    'Index Time (ms)': ung_data['time'], 'Index Size (MB)': ung_data['size']
                })
        
        return results

    except Exception as e:
        print(f"❌ 解析 Build 数据时出错 ('{group_title}'): {e}")
        return []
        

# ==============================================================================
# 3. 任务处理器
# ==============================================================================

def handle_qps_recall_tasks(config):
    """处理所有 QPS-Recall 类型的对比图任务。"""
    tasks = config.get("global_comparison_tasks", {}).get("qps_recall_plots", [])
    if not tasks: return

    print("\n" + "#"*80)
    print("🔍 开始处理 QPS-Recall 对比图任务...")
    print("#"*80)

    for task in tasks:
        if not task.get("enabled", False):
            print(f"⏭️ 跳过已禁用的任务: {task.get('task_name', 'N/A')}")
            continue

        print(f"\n🚀 执行任务: {task['title']}")
        all_plot_items = []
        plot_type = task.get("type", "overall") 
        
        dataset_name_for_path = "general" # 默认文件夹
        if task["items_to_compare"]:
            dataset_name_for_path = task["items_to_compare"][0].get("dataset", "general")
        
        for item in task["items_to_compare"]:
            plot_item_data_list = prepare_qps_recall_data(config, item['dataset'], item['experiment_name'], plot_type)
            
            if plot_item_data_list:
                all_plot_items.extend(plot_item_data_list)
        
        if all_plot_items:
            output_dir = os.path.join(config['global_settings']['base_results_dir'], "merge_results", "experiments", "pic", dataset_name_for_path)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, task["output_filename"])
            plot.generate_qps_recall_grid(all_plot_items, output_path, task["title"])

def handle_build_tasks(config):
    """处理所有 Build Performance 类型的对比图任务。"""
    tasks = config.get("global_comparison_tasks", {}).get("build_plots", [])
    if not tasks: return

    print("\n" + "#"*80)
    print("📊 开始处理构建性能对比图任务...")
    print("#"*80)

    for task in tasks:
        if not task.get("enabled", False):
            print(f"⏭️ 跳过已禁用的任务: {task.get('task_name', 'N/A')}")
            continue

        print(f"\n🚀 执行任务: {task['title']}")
        all_build_data = []

        for item in task["items_to_compare"]:
            build_item_data = prepare_build_data(config, item)
            if build_item_data:
                all_build_data.extend(build_item_data)
        
        if all_build_data:
            output_dir = os.path.join(config['global_settings']['base_results_dir'], "merge_results", "experiments", "pic")
            os.makedirs(output_dir, exist_ok=True)
            
            plot.generate_build_summary_plot(
                all_build_data, 
                output_dir, 
                task["output_filename_prefix"],
                task["title"]
            )


# ==============================================================================
# 4. 主函数 - 程序入口
# ==============================================================================

def main():
    config_file = "/data/fxy/FilterVector/FilterVectorCode/DataTools/exp/pic/config.json"
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ 错误: 无法加载或解析配置文件 {config_file}。请检查文件是否存在且格式正确。\n{e}")
        return
        
    # 按顺序执行所有类型的对比任务
    handle_qps_recall_tasks(config)
    handle_build_tasks(config)

    print("\n✅ 所有任务处理完毕。")

if __name__ == "__main__":
    main()