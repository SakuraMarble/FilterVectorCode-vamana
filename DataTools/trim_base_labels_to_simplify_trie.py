import collections
import random
import logging
import math
from itertools import combinations

def transform_by_pruning_final(input_file, output_file, log_file, target_ratio, max_prunes=200000):
    """
    通过一个两阶段混合策略进行数据修剪。
    """
    # ==============================================================================
    # 阶段 0: 初始化和数据读取
    # ==============================================================================
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()])
    logging.info(f"--- [v6-停滞恢复] 开始处理文件: {input_file} ---")
    logging.info(f"目标比值: {target_ratio:.4f}, 最大修剪次数: {max_prunes}")

    try:
        with open(input_file, 'r') as f: all_lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.error(f"错误：输入文件 {input_file} 不存在。程序终止。")
        return
    if not all_lines:
        logging.warning("输入文件为空，无需处理。")
        return
        
    total_vectors = len(all_lines)
    initial_counts = collections.Counter(all_lines)
    current_counts = initial_counts.copy()
    prunes_done = 0

    logging.info("\n--- 原始数据统计 ---")
    logging.info(f"总向量数: {total_vectors}")
    logging.info(f"独立标签集数量: {len(initial_counts)}")
    logging.info(f"初始比值: {len(initial_counts) / total_vectors:.4f}")

    if len(initial_counts) / total_vectors <= target_ratio:
        logging.warning("初始比值已达标，无需修剪。")
        with open(output_file, 'w') as f: f.writelines([line + '\n' for line in all_lines])
        return
    # ==============================================================================
    # 阶段 1: 保守模式 (带停滞检测)
    # ==============================================================================
    logging.info("\n--- [阶段 1: 保守模式 (带停滞检测)] 开始 ---")
    possible_prunes_conservative = {}
    unique_vectors_set = set(initial_counts.keys())
    for parent_str in unique_vectors_set:
        parent_tuple = tuple(parent_str.split(','));
        if len(parent_tuple) <= 1: continue
        for child_tuple in combinations(parent_tuple, len(parent_tuple) - 1):
            child_str = ','.join(child_tuple)
            if child_str in unique_vectors_set: possible_prunes_conservative[parent_str] = child_str; break
    
    if not possible_prunes_conservative:
        logging.warning("未找到任何保守的修剪路径，直接进入阶段2。")
    else:
        p1_stagnation_check_interval = 20000; p1_stagnation_threshold = 1000
        last_check_uniques_p1 = len(current_counts); last_check_prunes_p1 = prunes_done
        parents_by_length = collections.defaultdict(list)
        for p_str in possible_prunes_conservative.keys(): parents_by_length[len(p_str.split(','))].append(p_str)
        sorted_lengths = sorted(parents_by_length.keys(), reverse=True)
        
        while prunes_done < max_prunes and len(current_counts) / total_vectors > target_ratio:
            parent_to_prune = None
            for length in sorted_lengths:
                candidates = [p for p in parents_by_length[length] if current_counts.get(p, 0) > 0]
                if candidates: parent_to_prune = random.choice(candidates); break
            
            if parent_to_prune is None: logging.info("保守模式已无更多可修剪的向量。"); break

            child_to_become = possible_prunes_conservative[parent_to_prune]
            current_counts[parent_to_prune] -= 1
            if current_counts[parent_to_prune] == 0: del current_counts[parent_to_prune]
            current_counts[child_to_become] += 1; prunes_done += 1
            
            if prunes_done % 10000 == 0: logging.info(f"进度(保守): 已执行 {prunes_done} 次修剪... 当前比值: {len(current_counts) / total_vectors:.4f}")

            if prunes_done - last_check_prunes_p1 >= p1_stagnation_check_interval:
                if (last_check_uniques_p1 - len(current_counts)) < p1_stagnation_threshold:
                    logging.warning(f"保守阶段检测到停滞：在最近{p1_stagnation_check_interval}次修剪中标签减少数低于阈值。"); logging.warning("强制进入阶段2以寻求突破。"); break 
                last_check_uniques_p1 = len(current_counts); last_check_prunes_p1 = prunes_done
    
    current_ratio = len(current_counts) / total_vectors
    logging.info(f"--- 阶段 1 完成 ---"); logging.info(f"已执行 {prunes_done} 次修剪。当前比值: {current_ratio:.4f}")

    # ==============================================================================
    # 阶段 2: 自适应与自停止的多核心收敛模式
    # ==============================================================================
    if current_ratio > target_ratio and prunes_done < max_prunes:
        logging.info("\n--- [阶段 2: 自适应与自停止的多核心收敛模式] 开始 ---")
        logging.info("策略: 核心数将根据效率自动调整，若连续无法取得有意义的进展将自动终止。")

        core_recalculation_interval = 2500; status_update_interval = 5000
        stagnation_threshold = 500
        top_n_cores = []; core_scores = {}
        
        last_status_check_prunes = prunes_done; last_status_check_uniques = len(current_counts)
        last_core_recalc_prunes = prunes_done - core_recalculation_interval
        
        consecutive_no_progress_checks = 0
        max_no_progress_checks = 3
        final_stop_threshold = 10 # <--- [新增] 更宽容的最终停止阈值

        top_n_cores_count = 5

        while prunes_done < max_prunes and len(current_counts) / total_vectors > target_ratio:
            
            # --- 2.1: 停滞检查与策略调整 ---
            if prunes_done - last_status_check_prunes >= status_update_interval:
                uniques_reduced = last_status_check_uniques - len(current_counts)
                
                # --- 最终停止策略 ---
                if uniques_reduced < final_stop_threshold: # 使用更宽容的阈值
                    consecutive_no_progress_checks += 1
                    logging.warning(f"警告：检测到第 {consecutive_no_progress_checks} 个低进展周期 (减少量 < {final_stop_threshold})。")
                else:
                    consecutive_no_progress_checks = 0
                
                if consecutive_no_progress_checks >= max_no_progress_checks:
                    logging.error("算法连续多个周期无法取得有意义的进展，已陷入永久停滞。")
                    logging.error("程序将自动终止，并保存当前最佳结果。")
                    break

                # --- 自适应核心数调整 ---
                new_top_n_count = top_n_cores_count
                if uniques_reduced < 300 and top_n_cores_count > 2: new_top_n_count = 2
                elif uniques_reduced < 700 and top_n_cores_count > 3: new_top_n_count = 3
                
                if new_top_n_count != top_n_cores_count:
                    logging.info(f"效率变化：核心数从 {top_n_cores_count} 自动调整为 {new_top_n_count} 以聚焦算力。")
                    top_n_cores_count = new_top_n_count; top_n_cores = []

                # --- “强制合并”破局策略 ---
                if uniques_reduced < stagnation_threshold and len(top_n_cores) > 0:
                    logging.warning(f"检测到停滞：{status_update_interval}次修剪仅减少{uniques_reduced}个标签。执行强制合并！")
                    patriarch_core = top_n_cores[0]
                    potential_vassals = {core: score for core, score in core_scores.items() if core.startswith(patriarch_core + ',')}
                    if potential_vassals:
                        vassal_core = max(potential_vassals, key=potential_vassals.get)
                        logging.info(f"强制合并: 将核心 '{vassal_core}' 的所有后代合并到 '{patriarch_core}'")
                        vectors_to_collapse = [p for p in current_counts if p.startswith(vassal_core + ',')]
                        collapse_count = 0
                        for vector in vectors_to_collapse:
                            count = current_counts[vector]
                            current_counts[patriarch_core] = current_counts.get(patriarch_core, 0) + count
                            del current_counts[vector]; collapse_count += count
                        logging.info(f"合并完成，{len(vectors_to_collapse)}种独立标签被清除，涉及{collapse_count}个向量。")
                        prunes_done += collapse_count; top_n_cores = []
                    else:
                        logging.warning(f"核心 '{patriarch_core}' 无子核心可供合并，将尝试重新计算核心列表。"); top_n_cores = []
                
                last_status_check_prunes = prunes_done; last_status_check_uniques = len(current_counts)

            # --- 2.2: 核心识别与分层 ---
            if not top_n_cores or (prunes_done - last_core_recalc_prunes >= core_recalculation_interval):
                core_scores = collections.defaultdict(int)
                for vector_str, count in current_counts.items():
                    labels = vector_str.split(',')
                    if len(labels) > 1:
                        for i in range(1, len(labels)): core = ','.join(labels[:i]); core_scores[core] += count * len(core.split(','))
                if not core_scores: logging.warning("无法在数据中找到任何有效核心。"); break
                sorted_cores = sorted(core_scores.keys(), key=core_scores.get, reverse=True)
                selected_cores = []
                for core in sorted_cores:
                    if not any(core.startswith(sel_core + ',') or sel_core.startswith(core + ',') for sel_core in selected_cores):
                        selected_cores.append(core)
                    if len(selected_cores) >= top_n_cores_count: break
                if selected_cores != top_n_cores:
                    top_n_cores = selected_cores
                    logging.info(f"识别出新的Top-{len(top_n_cores)}个独立核心 (当前策略: {top_n_cores_count}核): {top_n_cores}")
                last_core_recalc_prunes = prunes_done

            # --- 2.3: 轮询修剪 ---
            if not top_n_cores: continue
            cores_to_remove = []
            for target_core in top_n_cores:
                if prunes_done >= max_prunes: break
                candidates_to_prune = [p for p in current_counts if p.startswith(target_core + ',') and current_counts.get(p, 0) > 0]
                if not candidates_to_prune: cores_to_remove.append(target_core); continue
                candidates_to_prune.sort(key=lambda p: (current_counts.get(p, 0), -len(p.split(',')))); parent_to_prune = candidates_to_prune[0]
                parent_list = parent_to_prune.split(','); target_core_len = len(target_core.split(','))
                child_to_become = ','.join(parent_list[:-1]); best_intermediate_child = None; max_child_count = 0
                for i in range(len(parent_list) - 1, target_core_len, -1):
                    intermediate_child_str = ','.join(parent_list[:i]); child_count = current_counts.get(intermediate_child_str, 0)
                    if child_count > max_child_count: max_child_count = child_count; best_intermediate_child = intermediate_child_str
                if best_intermediate_child: child_to_become = best_intermediate_child
                current_counts[parent_to_prune] -= 1
                if current_counts[parent_to_prune] == 0: del current_counts[parent_to_prune]
                current_counts[child_to_become] = current_counts.get(child_to_become, 0) + 1; prunes_done += 1
                if prunes_done % status_update_interval == 0: logging.info(f"[阶段 2] 已修剪: {prunes_done:<8} | 独立标签: {len(current_counts):<7} | 当前比值: {len(current_counts) / total_vectors:.4f}")
            if cores_to_remove: top_n_cores = [c for c in top_n_cores if c not in cores_to_remove]
    
    if len(current_counts) / total_vectors <= target_ratio and prunes_done < max_prunes: 
        logging.info("目标已达成！")

    # ==============================================================================
    # 阶段 3: 最终统计与输出
    # ==============================================================================
    logging.info("\n--- 正在根据最终计数结果生成输出文件... ---")
    final_lines = [];
    for vector_str in sorted(current_counts.keys()):
        count = current_counts[vector_str];
        if count > 0: final_lines.extend([vector_str] * count)
    with open(output_file, 'w') as f: f.writelines([line + '\n' for line in final_lines])
    logging.info("\n--- 最终统计 ---")
    final_unique_labels = len(current_counts); final_ratio = final_unique_labels / total_vectors if total_vectors > 0 else 0
    logging.info(f"总向量数: {total_vectors}"); logging.info(f"独立标签集数量: {final_unique_labels}"); logging.info(f"最终比值: {final_unique_labels}/{total_vectors} = {final_ratio:.4f}")
   #  logging.info("\n--- 数量变化详情 (仅显示有变化的标签集) ---")
   #  all_keys = sorted(set(initial_counts.keys()) | set(current_counts.keys())); changes_found = False
   #  for key in all_keys:
   #      initial = initial_counts.get(key, 0); final = current_counts.get(key, 0)
   #      if initial != final: changes_found = True; logging.info(f"   - 标签集 '{key}': 数量从 {initial} 变为 {final} (变化: {final - initial:+})")
   #  if not changes_found: logging.info("   (数据无任何变化)")
    logging.info(f"\n--- 处理完成 ---"); logging.info(f"共执行 {prunes_done} 次修剪。结果已保存至 {output_file}。")


if __name__ == "__main__":
    INPUT_FILE = "/data/fxy/FilterVector/FilterVectorData/app_reviews/base_11/app_reviews_base_labels.txt"
    OUTPUT_FILE = "/data/fxy/FilterVector/FilterVectorData/app_reviews/base_12/app_reviews_base_labels.txt"
    LOG_FILE = "/data/fxy/FilterVector/FilterVectorData/app_reviews/base_12/log_file.txt"
    TARGET_RATIO = 0.5
    MAX_PRUNES = 1000000

    transform_by_pruning_final(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        log_file=LOG_FILE,
        target_ratio=TARGET_RATIO,
        max_prunes=MAX_PRUNES
    )