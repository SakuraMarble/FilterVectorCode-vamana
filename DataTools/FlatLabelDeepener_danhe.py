# 这份代码将扁平的标签处理成深LNG的标签
import collections
import logging
from typing import List, Set, Dict, Any, Optional, Tuple
import multiprocessing
from tqdm import tqdm

# --- 1. 参数设置 ---
MIN_SUPPORT_COUNT = 10
MIN_PATH_DEPTH = 3
COVERAGE_TARGET = 0.99
BATCH_SIZE = 1000
PRESERVE_DEPTH_THRESHOLD = 4

# --- 日志系统设置 ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler('processing_log.txt', mode='w')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# --- 2. 数据加载与预处理 ---
def load_labels(filepath: str) -> List[Set[int]]:
    logger.info(f"开始从文件加载数据: {filepath}")
    all_labels = []
    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(tqdm(f, desc="加载数据")):
                line = line.strip()
                if not line: continue
                all_labels.append(set(map(int, line.split(','))))
        logger.info(f"数据加载完成，共 {len(all_labels)} 行。")
    except FileNotFoundError:
        logger.error(f"错误：输入文件未找到 at {filepath}"); return []
    return all_labels

def build_inverted_index(all_data: List[Set[int]]) -> Dict[int, Set[int]]:
    logger.info("正在构建倒排索引，请耐心等待...")
    inverted_index = collections.defaultdict(set)
    for i, labels in enumerate(tqdm(all_data, desc="构建倒排索引")):
        for label in labels:
            inverted_index[label].add(i)
    logger.info(f"倒排索引构建完成。索引了 {len(inverted_index)} 个独立标签。")
    return inverted_index

# --- 3. 阶段一：构建“主干树”森林 ---
TreeNode = Dict[str, Any]

def build_tree_recursive_optimized(
    parent_path: List[int],
    parent_row_indices: Set[int],
    inverted_index: Dict[int, Set[int]]
) -> List[TreeNode]:
    if not parent_row_indices:
        return []

    all_possible_labels = set(inverted_index.keys())
    candidate_labels = all_possible_labels - set(parent_path)
    
    is_first_level = (len(parent_path) == 1)
    
    desc_lv1 = f"分析 根 {parent_path[0]} 的子节点" if is_first_level else None
    iterator_lv1 = tqdm(
        candidate_labels,
        desc=desc_lv1,
        disable=not is_first_level,
        leave=False, # 分析完后消失，保持界面干净
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"
    )
    
    child_candidates = []
    with iterator_lv1 as bar:
        for child_label in bar:
            child_row_indices = parent_row_indices.intersection(inverted_index.get(child_label, set()))
            support_count = len(child_row_indices)
            if support_count > 0:
                child_candidates.append({'label': child_label, 'count': support_count, 'indices': child_row_indices})

    strong_children = [child for child in child_candidates if child['count'] >= MIN_SUPPORT_COUNT]

    if strong_children:
        nodes = []
        # 按支持度降序处理，优先探索更强的分支
        strong_children.sort(key=lambda x: x['count'], reverse=True)

        iterable_strong_children = strong_children
        if is_first_level:
            logger.info(f"根 {parent_path[0]} 找到 {len(strong_children)} 个强子节点，开始构建子树...")
            # --- 第二级进度条：追踪主要子树的构建进度 ---
            desc_lv2 = f"构建根 {parent_path[0]} 的子树"
            iterable_strong_children = tqdm(strong_children, desc=desc_lv2)

        for child_data in iterable_strong_children:
            # 递归调用时，不再有任何日志打印
            children = build_tree_recursive_optimized(
                parent_path + [child_data['label']],
                child_data['indices'],
                inverted_index
            )
            nodes.append({'label': child_data['label'], 'children': children})
        return nodes
    else:
        leaf_nodes = [{'label': child['label'], 'children': []} for child in sorted(child_candidates, key=lambda x: x['count'], reverse=True)]
        return leaf_nodes

def find_single_theme_tree(data_indices: Set[int], inverted_index: Dict[int, Set[int]], all_labels_list: List[Set[int]]) -> Optional[TreeNode]:
    if not data_indices: return None
    logger.info("在当前数据子集上寻找最佳根节点...")
    
    # MODIFICATION: 优化了根节点计数，并增加了进度条。
    sub_counter = collections.Counter()
    for idx in tqdm(data_indices, desc="计数根节点候选", leave=False):
        sub_counter.update(all_labels_list[idx])

    best_root_candidate = None
    for label, count in sub_counter.most_common():
        if count >= MIN_SUPPORT_COUNT:
            best_root_candidate = (label, count); break
    
    if not best_root_candidate:
        logger.warning(f"数据池中已没有任何标签满足最小支持度 {MIN_SUPPORT_COUNT}。"); return None
    
    root_label, root_count = best_root_candidate
    logger.info(f"已选定本轮候选根: {root_label} (支持度: {root_count})")
    
    root_row_indices = data_indices.intersection(inverted_index.get(root_label, set()))
    
    # MODIFICATION: 调用优化后的树构建函数
    logger.info(f"开始为根 {root_label} 构建树...")
    children = build_tree_recursive_optimized([root_label], root_row_indices, inverted_index)
    tree = {'label': root_label, 'children': children}
    
    def get_max_depth(node: TreeNode) -> int:
        if not node.get('children'): return 1
        return 1 + max([get_max_depth(child) for child in node['children']], default=0)
        
    max_depth = get_max_depth(tree)
    logger.info(f"为根 {root_label} 构建的树，最大深度为: {max_depth}")
    
    if max_depth >= MIN_PATH_DEPTH:
        return tree
    else:
        logger.warning(f"根 {root_label} 的树不满足最小深度要求 {MIN_PATH_DEPTH}，已丢弃。"); return None

def build_theme_forest(all_data: List[Set[int]], inverted_index: Dict[int, Set[int]]) -> Tuple[List[TreeNode], List[int]]:
    logger.info("\n==================================================")
    logger.info("--- 阶段一：开始构建主干树森林 ---")
    forest = []
    root_labels = []
    remaining_indices = set(range(len(all_data)))
    total_data_count = len(all_data)
    iteration_count = 1
    
    while True:
        logger.info(f"\n--- 森林构建迭代轮次: {iteration_count} ---")
        if not remaining_indices: logger.info("所有数据已处理完毕。"); break
        
        processed_count = total_data_count - len(remaining_indices)
        coverage_ratio = processed_count / total_data_count if total_data_count > 0 else 0
        
        if coverage_ratio >= COVERAGE_TARGET:
            logger.info(f"数据覆盖率 {coverage_ratio:.2%} 已达到目标，停止挖掘。"); break
            
        logger.info(f"当前剩余数据: {len(remaining_indices)} 行。已处理覆盖率: {coverage_ratio:.2%}")
        
        # MODIFICATION: 传入 all_data 用于根节点计数
        new_tree = find_single_theme_tree(remaining_indices, inverted_index, all_data)
        
        if not new_tree: logger.warning("在剩余数据中已无法找到新树，挖掘结束。"); break
        
        forest.append(new_tree)
        root_label = new_tree['label']
        root_labels.append(root_label)
        
        covered_indices = remaining_indices.intersection(inverted_index.get(root_label, set()))
        logger.info(f"此树覆盖了 {len(covered_indices)} 个新的数据行。")
        remaining_indices -= covered_indices
        iteration_count += 1
        
    logger.info(f"\n--- 阶段一结束：共发现 {len(forest)} 棵主干树 ---")
    return forest, root_labels

# --- 4. 阶段二：匹配与重构 ---
def find_deepest_path_in_tree_recursive(node: TreeNode, original_labels: Set[int]) -> Optional[List[int]]:
    if node['label'] not in original_labels: return None
    if not node['children']: return [node['label']]
    
    best_child_path = []
    for child in node['children']:
        path_from_child = find_deepest_path_in_tree_recursive(child, original_labels)
        if path_from_child and len(path_from_child) > len(best_child_path):
            best_child_path = path_from_child
            
    return [node['label']] + best_child_path

# MODIFICATION: 增加了`process_init`函数，用于在多进程池中初始化共享变量。
def process_init(forest_data: List[TreeNode]):
    """多进程工作单元的初始化函数"""
    global forest
    forest = forest_data

def process_batch_by_mod(batch_data: List[Tuple[int, Set[int]]]) -> Tuple[List[Tuple[int, List[int]]], Dict]:
    # 注意：这里的 'forest' 变量是由 process_init 初始化的全局变量
    global forest 
    
    batch_results = []
    batch_stats = {
        "unmatched_count": 0,
        "path_length_dist": collections.Counter(),
        "tree_assignment_dist": collections.Counter()
    }
    
    for original_index, original_labels in batch_data:
        deepest_path = []
        best_tree_index = -1
        for tree_idx, tree in enumerate(forest):
            path = find_deepest_path_in_tree_recursive(tree, original_labels)
            if path and len(path) > len(deepest_path):
                deepest_path = path
                best_tree_index = tree_idx
                
        if deepest_path:
            deepest_len = len(deepest_path)
            if deepest_len > PRESERVE_DEPTH_THRESHOLD:
                final_path = deepest_path
            else:
                final_len = (original_index % deepest_len) + 1
                final_path = deepest_path[:final_len]
                
            batch_results.append((original_index, final_path))
            batch_stats["path_length_dist"][len(final_path)] += 1
            if best_tree_index != -1:
                batch_stats["tree_assignment_dist"][best_tree_index] += 1
        else:
            final_path = list(original_labels)
            batch_results.append((original_index, final_path))
            batch_stats["unmatched_count"] += 1
            batch_stats["path_length_dist"][len(final_path)] += 1
            
    return batch_results, batch_stats

def reconstruct_and_save_labels_parallel(original_data: List[Set[int]], forest: List[TreeNode], output_filepath: str, num_workers: int):
    logger.info("\n==================================================")
    logger.info(f"--- 阶段二：开始并行匹配与重构（使用 {num_workers} 个核心） ---")
    
    indexed_data = list(enumerate(original_data))
    batches = [indexed_data[i:i + BATCH_SIZE] for i in range(0, len(indexed_data), BATCH_SIZE)]
    
    final_results_with_indices = []
    final_stats = {
        "unmatched_count": 0,
        "path_length_dist": collections.Counter(),
        "tree_assignment_dist": collections.Counter()
    }
    
    # MODIFICATION: 使用 initializer 将 forest 数据传递给所有子进程
    with multiprocessing.Pool(processes=num_workers, initializer=process_init, initargs=(forest,)) as pool:
        with tqdm(total=len(batches), desc="重构标签进度") as pbar:
            for result_batch, stats_batch in pool.imap_unordered(process_batch_by_mod, batches):
                final_results_with_indices.extend(result_batch)
                final_stats["unmatched_count"] += stats_batch["unmatched_count"]
                final_stats["path_length_dist"].update(stats_batch["path_length_dist"])
                final_stats["tree_assignment_dist"].update(stats_batch["tree_assignment_dist"])
                pbar.update(1)
                
    logger.info("所有并行任务处理完成。正在整理和保存结果...")
    final_results_with_indices.sort(key=lambda x: x[0])
    final_labels = [labels for index, labels in final_results_with_indices]
    
    with open(output_filepath, 'w') as f:
        for labels in tqdm(final_labels, desc="保存结果文件"):
            f.write(','.join(map(str, labels)) + '\n')
            
    logger.info(f"文件写入完成: {output_filepath}")
    return final_labels, final_stats

# --- 5. 统计报告模块 ---
# (此部分无需修改)
def analyze_tree_structure(node: TreeNode, level: int, level_dist: collections.Counter) -> Tuple[int, int]:
    level_dist[level] += 1
    if not node.get('children'): return 1, level
    total_nodes, max_depth = 1, level
    for child in node['children']:
        child_nodes, child_depth = analyze_tree_structure(child, level + 1, level_dist)
        total_nodes += child_nodes
        max_depth = max(max_depth, child_depth)
    return total_nodes, max_depth

def log_statistics(forest: List[TreeNode], final_labels: List[List[int]], stats: Dict):
    logger.info("\n==================================================")
    logger.info("--- 最终统计报告 ---")
    logger.info("\n[森林结构统计]")
    if not forest: logger.info("未发现任何主干树。"); return
    logger.info(f"共发现 {len(forest)} 棵主干树。")
    for i, tree in enumerate(forest):
        level_dist = collections.Counter()
        total_nodes, max_depth = analyze_tree_structure(tree, 1, level_dist)
        logger.info(f"\n--- 树 {i+1} (根: {tree['label']}) ---")
        logger.info(f"  - 最大深度: {max_depth}")
        logger.info(f"  - 节点总数: {total_nodes}")
        logger.info("  - 每层节点数分布:")
        for level in sorted(level_dist.keys()):
            logger.info(f"    - 层 {level}: {level_dist[level]} 个节点")
    total_count = len(final_labels)
    if total_count == 0:
        logger.warning("最终标签列表为空，无法进行统计。")
        return
    matched_count = total_count - stats['unmatched_count']
    logger.info("\n[重构结果统计]")
    logger.info(f"总处理向量数: {total_count}")
    logger.info(f"成功匹配到主干树的数量: {matched_count} ({matched_count/total_count:.2%})")
    logger.info(f"未匹配（保留原始标签）的数量: {stats['unmatched_count']} ({stats['unmatched_count']/total_count:.2%})")
    logger.info("\n[各主干树覆盖的向量数]")
    if not stats['tree_assignment_dist']:
        logger.info("无数据匹配到任何树。")
    else:
        for tree_idx in sorted(stats['tree_assignment_dist'].keys()):
            count = stats['tree_assignment_dist'][tree_idx]
            root_label = forest[tree_idx]['label']
            logger.info(f"  - 树 {tree_idx+1} (根: {root_label}): 覆盖 {count} 个向量 ({count/total_count:.2%})")
    logger.info("\n[最终标签长度分布]")
    if not stats['path_length_dist']:
        logger.info("无有效数据。")
    else:
        total_len_sum = sum(k * v for k, v in stats['path_length_dist'].items())
        avg_len = total_len_sum / total_count if total_count > 0 else 0
        logger.info(f"  - 平均标签长度: {avg_len:.2f}")
        for length in sorted(stats['path_length_dist'].keys()):
            count = stats['path_length_dist'][length]
            logger.info(f"  - 长度为 {length} 的向量: {count} 个 ({count/total_count:.2%})")

# --- 主程序 ---
if __name__ == '__main__':
    # 请确保这里的路径是正确的
    INPUT_FILE = '/home/fengxiaoyao/Data/data/celeba/data/celeba_attributes.txt'
    OUTPUT_FILE = 'hierarchical_labels.txt'

    original_data = load_labels(INPUT_FILE)

    if not original_data:
        logger.critical("输入数据为空或文件不存在，程序退出。")
    else:
        inverted_index = build_inverted_index(original_data)
        
        # MODIFICATION: 不再需要全局变量 forest
        forest, root_labels = build_theme_forest(original_data, inverted_index)
        
        if forest:
            total_cpus = multiprocessing.cpu_count()
            # MODIFICATION: 调整worker数量，可以根据你的机器配置修改
            num_workers = max(1, total_cpus // 2) 
            logger.info(f"总CPU核心数: {total_cpus}，本次任务将使用: {num_workers} 个核心。")

            reconstructed_labels, stats = reconstruct_and_save_labels_parallel(
                original_data, forest, OUTPUT_FILE, num_workers
            )
            
            log_statistics(forest, reconstructed_labels, stats)

            # --- 为C++生成TXT格式的辅助文件 ---
            logger.info("\n==================================================")
            logger.info("--- 阶段三：开始为下游任务生成辅助文件 ---")

            # 1. 生成 tree_roots.txt
            logger.info("正在生成树根信息文件 (tree_roots.txt)...")
            output_roots_file = 'tree_roots.txt'
            try:
                with open(output_roots_file, 'w') as f:
                    for label_id in root_labels:
                        f.write(f"{label_id}\n")
                logger.info(f"成功保存树根信息文件: {output_roots_file}")
            except Exception as e:
                logger.error(f"保存树根信息文件时出错: {e}")

            logger.info("\n==================================================")
            logger.info("所有任务已成功完成！")
        else:
            logger.warning("未能发现任何主干树。程序退出，不会生成输出文件。")
            logger.info("\n==================================================")
            logger.info("处理结束。")