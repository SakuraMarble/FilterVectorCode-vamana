import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mticker
import seaborn as sns
import re

# 解决负号'-'显示为方块的问题 
plt.rcParams['axes.unicode_minus'] = False 

# ==============================================================================
# 1. 辅助函数
# ==============================================================================

def get_bitmap_time_from_summary(file_path):
    """
    健壮地从 UNG 摘要文件中读取 Bitmap_Computation_Time_ms。
    """
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if 'Bitmap_Computation_Time_ms' in line:
                    parts = line.strip().split(',')
                    if len(parts) > 1:
                        return float(parts[1])
        # 将警告信息改为英文
        print(f"Warning: 'Bitmap_Computation_Time_ms' not found in {os.path.basename(file_path)}. Returning 0.0.")
        return 0.0
    except (FileNotFoundError, Exception) as e:
        print(f"Error reading file {file_path}: {e}")
        raise

def _get_dataset_name_from_paths(paths):
    """
    从paths字典中的文件路径动态提取数据集名称。
    假定路径结构为 .../ACORN/<dataset_name>/... 或 .../UNG/<dataset_name>/...
    """
    try:
        any_path = paths.get('acorn_details_file') or paths.get('ung_nt_false_details_file')
        if not any_path:
            return "Unknown Dataset" 

        path_parts = any_path.split(os.sep)

        for i, part in enumerate(path_parts):
            if (part == 'ACORN' or part == 'UNG') and i + 1 < len(path_parts):
                return path_parts[i + 1]
        
        return "Unknown Dataset" 
    except Exception:
        return "Unknown Dataset"

def _calculate_compatible_y_limit(max_val, num_ticks=5):
    """
    根据数据的最大值，计算一个与 Matplotlib 自动刻度兼容的、“漂亮的”Y轴上限。
    这个函数会确保返回的上限值本身就是一个理想的主刻度。
    """
    if max_val <= 0:
        # MODIFIED: 同时返回步长
        return 10.0, 2.0  

    # 1. 估算一个“原始”的刻度步长
    rough_step = max_val / num_ticks

    # 2. 将原始步长“圆整”到一个漂亮的数值 (e.g., 1, 2, 5, 10, 20, 50, ...)
    magnitude = 10**np.floor(np.log10(rough_step))
    residual = rough_step / magnitude

    if residual > 5:
        nice_residual = 10
    elif residual > 2.5:
        nice_residual = 5
    elif residual > 2:
        nice_residual = 2.5
    elif residual > 1:
        nice_residual = 2
    else:
        nice_residual = 1
    
    nice_step = nice_residual * magnitude

    # 3. 基于这个“漂亮”的步长，计算最终的Y轴上限
    limit = np.ceil(max_val / nice_step) * nice_step
    
    # MODIFIED: 同时返回计算出的上限和步长
    return limit, nice_step

def _get_adaptive_xaxis_locator(data_structure):
    """
    根据绘图数据的全局范围，动态选择 x 轴的刻度间隔 (0.1 或 0.05)。
    """
    global_min_recall = 1.0
    global_max_recall = 0.0
    
    data_found = False
    # 处理字典结构 (用于 overall plot)
    if isinstance(data_structure, dict):
        for alg, data in data_structure.items():
            if not data.empty and 'Recall' in data.columns:
                global_min_recall = min(global_min_recall, data['Recall'].min())
                global_max_recall = max(global_max_recall, data['Recall'].max())
                data_found = True
    # 处理列表结构 (用于其他 plots)
    elif isinstance(data_structure, list):
        for config in data_structure:
            if config.get('data'):
                for alg, data in config['data'].items():
                    if not data.empty and 'Recall' in data.columns:
                        global_min_recall = min(global_min_recall, data['Recall'].min())
                        global_max_recall = max(global_max_recall, data['Recall'].max())
                        data_found = True

    # 如果找到了数据并计算了范围
    if data_found and global_max_recall > global_min_recall:
        data_span = global_max_recall - global_min_recall
        # 决策规则：如果数据范围很窄，使用更精细的0.05间隔
        if data_span <= 0.2:
            return mticker.MultipleLocator(0.05) 
        else:
            return mticker.MultipleLocator(0.1)
    
    # 如果没有数据或范围为0，返回默认刻度
    return mticker.MultipleLocator(0.1)

def _load_and_merge_data(paths):
    """
    加载、预处理、筛选和合并所有核心数据。(修正版)
    """
    print("Loading and merging data...")
    df_acorn = pd.read_csv(paths['acorn_details_file'])
    df_ung_false = pd.read_csv(paths['ung_nt_false_details_file'])
    df_ung_true = pd.read_csv(paths['ung_nt_true_details_file'])
    
    acorn_params = sorted(df_acorn['efs'].unique())
    ung_params = sorted(df_ung_false['Lsearch'].unique())
    if len(acorn_params) != len(ung_params):
        raise ValueError("Mismatch in the number of search parameters between ACORN and UNG.")
        
    param_map = pd.DataFrame({'efs': acorn_params, 'Lsearch': ung_params})
    df_acorn = pd.merge(df_acorn, param_map, on='efs').drop(columns=['efs'])

    # 在所有DataFrame中统一使用 'search_param' 作为合并键
    df_acorn.rename(columns={'Lsearch': 'search_param'}, inplace=True)
    df_ung_false.rename(columns={'Lsearch': 'search_param'}, inplace=True)
    df_ung_true.rename(columns={'Lsearch': 'search_param'}, inplace=True)
    
    selected_df = pd.read_csv(paths['selected_queries_file'])
    selected_query_ids = selected_df['QueryID'].unique().tolist()
    print(f"Successfully loaded {len(selected_query_ids)} query IDs for analysis.")
    
    # 1. 在合并前，主动为 ung_false 和 ung_true 的列添加后缀，确保列名唯一且符合后续计算的预期
    keys = ['QueryID', 'search_param']
    df_ung_false = df_ung_false.add_suffix('_ung_false').rename(columns={k + '_ung_false': k for k in keys})
    df_ung_true = df_ung_true.add_suffix('_ung_true').rename(columns={k + '_ung_true': k for k in keys})

    # 2. 过滤所有DataFrame，只保留被选中的查询
    df_acorn = df_acorn[df_acorn['QueryID'].isin(selected_query_ids)]
    df_ung_false = df_ung_false[df_ung_false['QueryID'].isin(selected_query_ids)]
    df_ung_true = df_ung_true[df_ung_true['QueryID'].isin(selected_query_ids)]

    # 3. 按顺序将三个准备好的数据帧合并起来
    merged_df = pd.merge(df_acorn, df_ung_false, on=keys)
    merged_df = pd.merge(merged_df, df_ung_true, on=keys)
    
    ung_bitmap_total_time = get_bitmap_time_from_summary(paths['ung_summary_file'])

    return merged_df, ung_bitmap_total_time

def _calculate_performance_metrics(df, num_queries, total_bitmap_time):
    """
    计算所有方法的性能指标 (Time 和 Recall)。
    (简化版：移除了 use_suffixed_cols 参数和相关逻辑)
    """
    if df.empty or num_queries == 0:
        return df

    acorn_bitmap_time = total_bitmap_time / num_queries
    ung_bitmap_time = total_bitmap_time / num_queries

    # 直接使用 pd.merge() 后固定的列名，不再需要条件判断
    df['Time_ACORN-1'] = df['acorn_1_Time_ms'] + acorn_bitmap_time
    df['Time_ACORN-gamma'] = df['acorn_Time_ms'] + acorn_bitmap_time
    df['Time_UNG'] = df['SearchT_ms_ung_false']
    df['Time_Method1'] = np.minimum(df['SearchT_ms_ung_false'], df['SearchT_ms_ung_true'])
    
    term_m2_acorn_part = df['acorn_Time_ms'] + ung_bitmap_time
    df['Time_Method2'] = np.minimum(term_m2_acorn_part, df['SearchT_ms_ung_false']) + df['FlagT_ms_ung_false']
    
    min_search_val = np.minimum.reduce([term_m2_acorn_part, df['SearchT_ms_ung_false'], df['SearchT_ms_ung_true']])
    df['Time_Method3'] = min_search_val + np.where(min_search_val == df['SearchT_ms_ung_true'], df['FlagT_ms_ung_true'], df['FlagT_ms_ung_false'])

    df['Recall_ACORN-1'] = df['acorn_1_Recall']
    df['Recall_ACORN-gamma'] = df['acorn_Recall']
    df['Recall_UNG'] = df['Recall_ung_false']
    df['Recall_Method1'] = np.where(df['EntryGroupT_ms_ung_false'] <= df['EntryGroupT_ms_ung_true'], df['Recall_ung_false'], df['Recall_ung_true'])
    df['Recall_Method2'] = np.where(term_m2_acorn_part <= df['SearchT_ms_ung_false'], df['acorn_Recall'], df['Recall_ung_false'])
    df['Recall_Method3'] = np.where(min_search_val == term_m2_acorn_part, df['acorn_Recall'], 
                                    np.where(min_search_val == df['SearchT_ms_ung_false'], df['Recall_ung_false'], df['Recall_ung_true']))
    
    return df

def _prepare_plot_data(df):
    """
    对数据进行分组和聚合，准备用于绘图的 QPS-Recall 数据。
    """
    if df.empty:
        return {}
        
    grouped = df.groupby('search_param')
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
        
    return plot_data

def _parse_acorn_meta(file_path):
    """
    解析ACORN的.meta文件，提取构建时间和索引大小。
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        build_time_s = float(re.search(r'build_time_s:([\d.]+)', content).group(1))
        index_size_bytes = int(re.search(r'index_only_size_bytes:(\d+)', content).group(1))
        
        # 将单位统一为毫秒(ms)和兆字节(MB)
        return {
            'time': build_time_s * 1000,
            'size': index_size_bytes / (1024 * 1024)
        }
    except (FileNotFoundError, AttributeError, ValueError) as e:
        print(f"Warning: Could not parse ACORN meta file {file_path}. Error: {e}")
        return None

def _parse_ung_meta(file_path):
    """
    增强版解析函数：解析UNG的build.log文件，提取所有需要的构建时间和索引大小指标。
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # 使用正则表达式安全地提取所需的值，如果找不到则返回 None
        def find_value(pattern, text):
            match = re.search(pattern, text)
            return float(match.group(1)) if match else None

        index_time_ms = find_value(r'index_time\(ms\)=([\d.]+)', content)
        index_size_mb = find_value(r'index_size\(MB\)=([\d.]+)', content)
        index_size_add_rb_mb = find_value(r'_index_size_add_rb\(MB\)=([\d.]+)', content)
        
        # 确保关键值存在
        if index_time_ms is None or index_size_mb is None or index_size_add_rb_mb is None:
             raise ValueError("One or more required metrics not found in log.")

        return {
            'time': index_time_ms,
            'size': index_size_mb,
            'size_add_rb': index_size_add_rb_mb
        }
    except (FileNotFoundError, AttributeError, ValueError) as e:
        print(f"Warning: Could not parse UNG meta file {file_path}. Error: {e}")
        return None

# ==============================================================================
# 2. "原子"绘图函数
# ==============================================================================

def _plot_qps_recall_on_ax(ax, plot_data, title, xlabel):
    """
    在一个给定的 ax 上绘制 QPS-Recall 曲线。
    这是从旧的绘图函数中提取出的核心绘图逻辑。
    """
    markers = ['o', 's', '^', 'D', 'v', 'p']
    ax.set_title(title, fontsize=34)
    if plot_data:
        for j, (alg, data) in enumerate(plot_data.items()):
            if not data.empty:
                ax.plot(data['Recall'], data['QPS'], marker=markers[j], linestyle='-', label=alg)
    ax.set_xlabel(xlabel, fontsize=32)

def _plot_build_bars_on_ax(ax, plot_data, y_metric, title, palette):
    """
    在一个给定的 ax 上绘制构建性能的条形图。
    这是从旧的 generate_build_comparison_plot 中提取并增强的。
    """
    sns.barplot(data=plot_data, x='Experiment', y=y_metric, hue='Algorithm', ax=ax, palette=palette)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('')
    ax.set_ylabel(y_metric.split('(')[-1].replace(')', ''), fontsize=12) # 提取单位 (ms) 或 (MB)
    ax.tick_params(axis='x', rotation=10, labelsize=10)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=10)
    
    # 只在时间图上显示图例，以保持简洁
    if 'Time' in y_metric:
        ax.legend(title='Algorithm', fontsize=10)
    else:
        # 对于大小图，如果图例存在则移除
        if ax.get_legend() is not None:
            ax.get_legend().remove()

# ==============================================================================
# 3. 网格生成
# ==============================================================================

def generate_qps_recall_grid(all_plot_items, output_path, main_title):
    """
    通用的网格生成器，用于所有 QPS-Recall 类型的图。
    :param all_plot_items: 一个列表。每个元素是一个字典，代表一个子图，
                           需包含 'data', 'title', 'xlabel' 等键。
    :param output_path: 最终组合图的保存路径。
    :param main_title: 整个图表的总标题。
    """
    if not all_plot_items:
        print("错误：没有可供绘制的 QPS-Recall 数据。")
        return

    # --- 1. 全局计算坐标轴范围 ---
    x_locator = _get_adaptive_xaxis_locator(all_plot_items)
    
    global_max_qps = 0
    for item in all_plot_items:
        if item.get('data'):
            for alg, data in item['data'].items():
                if not data.empty:
                    global_max_qps = max(global_max_qps, data['QPS'].max())
    y_upper_limit, y_step = _calculate_compatible_y_limit(global_max_qps)
    
    # --- 2. 创建图表网格 ---
    fig, axes = plt.subplots(3, 6, figsize=(36, 18), sharex=True, sharey=True)
    fig.suptitle(main_title, fontsize=36, y=0.98)

    # --- 3. 遍历数据并调用 "原子" 函数填充子图 ---
    for i, item in enumerate(all_plot_items):
        if i >= 18:  # 最多填充18个子图
             print(f"警告: 数据项数量 ({len(all_plot_items)}) 超过了 18 个子图的容量，后续数据将被忽略。")
             break
        ax = axes.flat[i]
        _plot_qps_recall_on_ax(ax, item.get('data'), item.get('title', 'N/A'), item.get('xlabel', 'Recall'))

    # --- 4. 设置统一的图例 ---
    first_ax_with_data = next((ax for ax in axes.flat if ax.has_data()), None)
    if first_ax_with_data:
        handles, labels = first_ax_with_data.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=6, fontsize=34)

    # --- 5. 格式化所有子图---
    num_filled_plots = len(all_plot_items)
    for i, ax in enumerate(axes.flat):
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='x', labelbottom=True, labelsize=32)
        ax.tick_params(axis='y', labelsize=32)
        
        ax.xaxis.set_major_locator(x_locator)
        ax.set_ylim(0, y_upper_limit)
        if y_step > 0:
            ax.set_yticks(np.arange(0, y_upper_limit + y_step, y_step))
        if i % 6 == 0: 
            ax.set_ylabel('QPS', fontsize=32)
        
        # 如果子图没有被填充，则标记为 N/A
        if i >= num_filled_plots:
            ax.set_title(f'Plot {i + 1}: N/A', fontsize=34, color='gray')
            ax.text(0.5, 0.5, 'Data Not Available', transform=ax.transAxes, ha='center', va='center', fontsize=22, color='lightgray')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ QPS-Recall 组合图已成功保存到: {os.path.abspath(output_path)}")


def generate_build_summary_plot(all_build_data, output_dir, file_prefix, main_title):
    """
    生成构建性能摘要图。
    - Y轴上限经过美化，确保顶部有刻度。
    """
    if not all_build_data:
        print("错误：没有可供绘制的构建性能数据。")
        return
        
    df = pd.DataFrame(all_build_data)

    df['Algorithm'] = df['Algorithm'].replace({
        'Hybrid': 'Our Method (Hybrid)',
        'ACORN-1': 'ACORN-1'
    })
    
    algo_order = ['UNG', 'ACORN', 'ACORN-1', 'Our Method (Hybrid)']
    palette = sns.color_palette("viridis", n_colors=len(algo_order))
    df['Algorithm'] = pd.Categorical(df['Algorithm'], categories=algo_order, ordered=True)
    
    all_group_categories = [
        'Celeba (Q7)', 'data2', 
        'data3', 'data4',
        'data5', 'data6',
        'data7', 'data8'
    ]
    df['group_title'] = pd.Categorical(df['group_title'], categories=all_group_categories, ordered=True)

    df['Index Time (s)'] = df['Index Time (ms)'] / 1000

    # (一) 生成并保存 Build Time 图表
    
    # --- 计算Y轴的动态上限和步长 ---
    max_time = df['Index Time (s)'].max()
    time_y_limit, time_y_step = _calculate_compatible_y_limit(max_time)
    
    fig_time, ax_time = plt.subplots(figsize=(18, 6), dpi=150)
    sns.barplot(data=df, x='group_title', y='Index Time (s)', hue='Algorithm', ax=ax_time, palette=palette, dodge=True)
    
    ax_time.set_title('Index Build Time', fontsize=20, pad=80)
    ax_time.set_xlabel('')
    ax_time.set_ylabel('Index Construction Time (s)', fontsize=20)
    
    # --- 应用计算好的Y轴范围和刻度 ---
    ax_time.set_ylim(0, time_y_limit)
    if time_y_step > 0:
        ax_time.set_yticks(np.arange(0, time_y_limit + time_y_step, time_y_step))
        
    ax_time.tick_params(axis='x', rotation=0, labelsize=18)
    ax_time.tick_params(axis='y', labelsize=18)
    ax_time.grid(axis='y', linestyle='--', alpha=0.7)
    
    for container in ax_time.containers:
        ax_time.bar_label(container, fmt='%.0f', fontsize=14, padding=3)

    handles, labels = ax_time.get_legend_handles_labels()
    if ax_time.get_legend() is not None:
        ax_time.get_legend().remove()
    fig_time.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.83), ncol=len(algo_order), fontsize=16)

    fig_time.tight_layout(rect=[0, 0, 1, 0.95])
    
    time_output_path = os.path.join(output_dir, f"{file_prefix}_time.png")
    fig_time.savefig(time_output_path, bbox_inches='tight')
    plt.close(fig_time)
    print(f"✅ 构建时间图表已成功保存到: {os.path.abspath(time_output_path)}")

    # (二) 生成并保存 Index Size 图表
    
    # --- 计算Y轴的动态上限和步长 ---
    max_size = df['Index Size (MB)'].max()
    size_y_limit, size_y_step = _calculate_compatible_y_limit(max_size)

    fig_size, ax_size = plt.subplots(figsize=(18, 6), dpi=150)
    
    sns.barplot(data=df, x='group_title', y='Index Size (MB)', hue='Algorithm', ax=ax_size, palette=palette, dodge=True)

    ax_size.set_title('Index Size', fontsize=20, pad=80)
    ax_size.set_xlabel('')
    ax_size.set_ylabel('Index Size (MB)', fontsize=20)
    ax_size.set_ylim(0, size_y_limit)
    if size_y_step > 0:
        ax_size.set_yticks(np.arange(0, size_y_limit + size_y_step, size_y_step))

    ax_size.tick_params(axis='x', rotation=0, labelsize=18)
    ax_size.tick_params(axis='y', labelsize=18)
    ax_size.grid(axis='y', linestyle='--', alpha=0.7)

    for container in ax_size.containers:
        ax_size.bar_label(container, fmt='%.0f', fontsize=14, padding=3)

    handles, labels = ax_size.get_legend_handles_labels()
    if ax_size.get_legend() is not None:
        ax_size.get_legend().remove()
    fig_size.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.83), ncol=len(algo_order), fontsize=16)
    
    fig_size.tight_layout(rect=[0, 0, 1, 0.95])
    
    size_output_path = os.path.join(output_dir, f"{file_prefix}_size.png")
    fig_size.savefig(size_output_path, bbox_inches='tight')
    plt.close(fig_size)
    print(f"✅ 索引大小图表已成功保存到: {os.path.abspath(size_output_path)}")