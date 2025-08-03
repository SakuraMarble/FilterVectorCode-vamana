import os

# --- 1. 配置 ---
# 输入文件
INPUT_FILENAME = '/home/fengxiaoyao/Data/data/amazing_file/output/hierarchical_labels.txt'
# 第一个脚本生成的图结构文件
GRAPH_EDGES_FILENAME = '/home/fengxiaoyao/Data/data/amazing_file/output/graph_edges.txt' 

# --- 输出文件 ---
OUTPUT_FILENAME = 'encoded_65535_output.txt'
OUTPUT_ROOTS_FILENAME = 'encoded_tree_roots.txt' # C++最终使用的树根文件
MAPPING_FILENAME = 'encoding_65535_map.txt'

# C++ 中 uint16_t 类型的最大值
UINT16_MAX = 65535

def check_recode_and_find_root():
    """
    重编码所有标签ID，并根据图的拓扑结构，计算并写入最终的结构性根节点。
    """
    if not os.path.exists(INPUT_FILENAME):
        print(f"错误：输入文件 '{INPUT_FILENAME}' 不存在。")
        return

    print(f"步骤1: 正在读取主标签文件 '{INPUT_FILENAME}' 以收集所有ID...")
    
    unique_codes = set()
    needs_reencoding = False
    try:
        with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                codes = [int(c.strip()) for c in line.split(',') if c.strip()]
                for code in codes:
                    unique_codes.add(code)
                    if code > UINT16_MAX:
                        needs_reencoding = True
    except Exception as e:
        print(f"读取文件时发生错误: {e}"); return

    if not needs_reencoding:
        print("\n检查完成。所有编码都在 uint16_t 上限内，无需重编码。")
        print("警告：此脚本的核心功能是重编码并寻找结构根，如果无需重编码，您可能需要手动确认 tree_roots.txt 的正确性。")
        return
        
    print(f"\n发现编码超限。将进行重编码并寻找新的结构性树根...")

    # 创建新旧ID映射表: {旧ID: 新ID}
    code_mapping = {old_code: new_code for new_code, old_code in enumerate(sorted(list(unique_codes)), 1)}

    # --- 开始生成所有输出文件 ---
    try:
        # 1. 生成重编码后的主标签文件
        print(f"步骤2: 正在生成新的编码文件: {OUTPUT_FILENAME}")
        with open(INPUT_FILENAME, 'r', encoding='utf-8') as f_in, \
             open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                if not line.strip(): f_out.write('\n'); continue
                original_codes = [int(c.strip()) for c in line.split(',') if c.strip()]
                new_codes = [str(code_mapping[c]) for c in original_codes]
                f_out.write(','.join(new_codes) + '\n')

        # 2. 根据图结构计算真正的根节点
        print(f"步骤3: 正在读取图结构文件 '{GRAPH_EDGES_FILENAME}' 并计算新根节点...")
        if not os.path.exists(GRAPH_EDGES_FILENAME):
            print(f"  - 错误！未找到图结构文件 '{GRAPH_EDGES_FILENAME}'。无法确定根节点。")
        else:
            parents = set()
            children = set()
            with open(GRAPH_EDGES_FILENAME, 'r', encoding='utf-8') as f_edges:
                for line in f_edges:
                    if not line.strip(): continue
                    p_str, c_str = line.strip().split(',')
                    old_p, old_c = int(p_str), int(c_str)
                    
                    # 使用新编码的ID来记录父子关系
                    if old_p in code_mapping and old_c in code_mapping:
                        new_p = code_mapping[old_p]
                        new_c = code_mapping[old_c]
                        parents.add(new_p)
                        children.add(new_c)
            
            # 结构性的根 = 是父节点，但从未当过子节点的节点
            structural_roots = parents - children
            
            if not structural_roots:
                print("  - 警告：未能从图结构中找到任何结构性根节点！可能是图有环或结构单一。")
            else:
                print(f"  - 成功！发现 {len(structural_roots)} 个结构性根。正在写入: {OUTPUT_ROOTS_FILENAME}")
                with open(OUTPUT_ROOTS_FILENAME, 'w', encoding='utf-8') as f_roots_out:
                    for root_id in sorted(list(structural_roots)): # 排序使输出稳定
                        f_roots_out.write(f"{root_id}\n")

        # 3. 保存映射文件用于调试
        print(f"步骤4: 正在保存编码映射文件: {MAPPING_FILENAME}")
        with open(MAPPING_FILENAME, 'w', encoding='utf-8') as f_map:
            f_map.write("原始编码,新编码\n")
            for old, new in code_mapping.items(): f_map.write(f"{old},{new}\n")

        print("\n----------------------------------------")
        print("任务完成！")
        print(f"  - 新标签文件: {OUTPUT_FILENAME}")
        print(f"  - 新结构根文件: {OUTPUT_ROOTS_FILENAME}")
        print("----------------------------------------")
        print("请在C++程序中使用以上两个新生成的文件。")

    except Exception as e:
        print(f"写入文件时发生严重错误: {e}")

if __name__ == '__main__':
    check_recode_and_find_root()