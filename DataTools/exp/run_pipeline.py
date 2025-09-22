import json
import os
import argparse
from modules import process, select, plot
import re

def run_experiment(config, experiment_config):
    """
    为单个实验配置构建路径并按顺序执行所有处理步骤。
    """
    base_dir = config['base_results_dir']
    dataset = config['dataset']
    params = experiment_config['parameters']
    exp_name = experiment_config['name'].format(**params)
    templates = config['structure_templates']

    print("\n" + "="*80)
    print(f"🚀 开始处理实验: {dataset} - {exp_name}")
    print("="*80)

    # --- 使用模板和参数动态构建所有文件路径 ---
    paths = {}
    
    # 输出目录使用实验名称
    merge_dir = os.path.join(base_dir, "merge_results/experiments",dataset, f"query{params['query_num']}", exp_name)
    os.makedirs(merge_dir, exist_ok=True)

    # ACORN 路径
    acorn_search_params = templates['acorn_search_params'].format(**params)
    acorn_index_handle = templates['acorn_index_handle']
    # 自动从 acorn_index_handle 解析 Total Vectors
    total_vectors = 0
    match = re.search(r'N(\d+)', acorn_index_handle)
    if match:
        total_vectors = int(match.group(1))
        print(f"... 成功从 '{acorn_index_handle}' 中自动提取 Total Vectors: {total_vectors}")
    else:
        print(f"⚠️ 警告: 未能从 '{acorn_index_handle}' 中自动提取 Total Vectors。请检查 config.json。")
    paths['total_vectors'] = total_vectors

    acorn_base = os.path.join(base_dir, "ACORN", dataset, "Results", f"index_{acorn_index_handle}", f"query{params['query_num']}_{acorn_search_params}", "results")
    acorn_file = templates['acorn_file_pattern'].format(**params)
    paths['acorn_details_file'] = os.path.join(acorn_base, acorn_file)
    paths['acorn_avg_file'] = os.path.join(acorn_base, f"avg_{acorn_file}")

    # UNG 路径
    ung_index = templates['ung_index_handle'].format(**params)
    ung_gt = templates['ung_gt_handle'].format(**params)
    ung_search_false = templates['ung_search_handle_false'].format(**params)
    ung_search_true = templates['ung_search_handle_true'].format(**params)
    ung_base_false = os.path.join(base_dir, "UNG", dataset, "Results", f"Index[{ung_index}]_GT[{ung_gt}]_Search[{ung_search_false}]", "results")
    ung_base_true = os.path.join(base_dir, "UNG", dataset, "Results", f"Index[{ung_index}]_GT[{ung_gt}]_Search[{ung_search_true}]", "results")
    
    paths['ung_nt_false_details_file'] = os.path.join(ung_base_false, "query_details_repeat1.csv")
    paths['ung_nt_true_details_file'] = os.path.join(ung_base_true, "query_details_repeat1.csv")
    paths['ung_summary_file'] = os.path.join(ung_base_false, "search_time_summary.csv")

    # 中间及最终输出文件路径
    paths['output_csv_path'] = os.path.join(merge_dir, f"{exp_name}_final_comparison.csv")
    paths['selected_queries_file'] = os.path.join(merge_dir, f"selected_queries_{experiment_config['analysis_params']['selection_mode']}.csv")
    paths['output_plot_overall_path'] = os.path.join(merge_dir, "qps_recall_curve.png")
    paths['output_plot_querysize_path'] = os.path.join(merge_dir, "qps_recall_curve_by_querysize.png")
    paths['attribute_coverage_file'] = templates['attribute_coverage_file'].format(**params, dataset=dataset)
    paths['output_plot_selectivity_path'] = os.path.join(merge_dir, "qps_recall_curve_by_selectivity.png")
    paths['output_dir'] = merge_dir
    
    # 打印将要使用的关键路径以供核对
    print("... 使用以下关键输入文件:")
    print(f"  - ACORN Details: .../{'/'.join(paths['acorn_details_file'].split('/')[-5:])}")
    print(f"  - UNG False Details: .../{'/'.join(paths['ung_nt_false_details_file'].split('/')[-5:])}")
    print(f"... 结果将输出到: {paths['output_dir']}")


    # --- 按顺序执行流水线 ---
    try:
        # 步骤 1: 合并原始CSV文件
        process.run_processing(paths)
        
        # 步骤 2: 根据策略挑选查询
        select.run_selection(paths, experiment_config['analysis_params'])

        # 步骤 3: 绘制整体性能图和按查询大小划分的图
        plot.generate_overall_plot(paths)
        plot.generate_querysize_plot(paths)

        # 步骤 4: 绘制按Filter Selectivity划分的图 (可选) 
        if experiment_config.get('analysis_params', {}).get('generate_selectivity_plot', False):
            plot.generate_selectivity_plot(paths)

    except FileNotFoundError as e:
        print(f"❌ 文件未找到错误: {e.filename}")
        print("流水线处理中断。请检查 config.json 中的参数是否能正确匹配到已存在的实验结果目录。")
        return
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")
        print("流水线处理中断。")
        return

    print(f"✅ 实验 {dataset} - {exp_name} 处理完成！")
    print("="*80)


def main():

    config_file="/data/fxy/FilterVector/FilterVectorCode/DataTools/exp/config.json"

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"错误：配置文件不存在 -> {args.config_file}")
        return
    except json.JSONDecodeError:
        print(f"错误：配置文件格式无效 -> {args.config_file}")
        return

    # --- 第一部分：处理所有单个实验 ---
    for experiment_config in config.get("experiments", []):
        if experiment_config.get("enabled", False):
            run_experiment(config, experiment_config)
        else:
            print(f"⏭️ 跳过已禁用的实验: {experiment_config.get('name', 'N/A')}")
            
    # --- 第二部分：处理对比图 ---
    if "comparison_plots" in config:
        print("\n" + "#"*80)
        print("🔍 检查并处理对比图任务...")
        print("#"*80)
        for plot_config in config.get("comparison_plots", []):
            if plot_config.get("enabled", False):
                plot.generate_comparison_plot(config, plot_config)
            else:
                print(f"⏭️ 跳过已禁用的对比图: {plot_config.get('output_filename', 'N/A')}")


if __name__ == "__main__":
    main()