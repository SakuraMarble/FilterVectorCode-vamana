# 把base属性转换为按照频率编码
import os
from collections import Counter

def recode_atomic_attributes(input_file_paths, output_dir):
    """
    读取文件列表，将逗号分隔的独立属性值作为编码单元，根据其全局频率进行编码。

    Args:
        input_file_paths (list): 包含源txt文件完整路径的列表。
        output_dir (str): 用于存放编码后文件和映射表的目录路径。
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 预处理: 检查文件是否存在 ---
    valid_file_paths = []
    for path in input_file_paths:
        if os.path.exists(path):
            valid_file_paths.append(path)
        else:
            print(f"警告：文件 '{path}' 未找到，将被跳过。")
    
    if not valid_file_paths:
        print("错误：所有提供的文件路径均无效。程序终止。")
        return

    # --- 第1步: 统计所有文件中所有独立属性值的频率 ---
    print(f"--- 开始处理 {len(valid_file_paths)} 个指定文件 ---")
    all_atomic_attributes = []
    total_lines = 0
    total_groups = 0

    for file_path in valid_file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                total_lines += 1
                # 1. 按空格分割成“属性组”
                attribute_groups = line.split()
                total_groups += len(attribute_groups)
                
                # 2. 遍历每个组，再按逗号分割成“独立属性值”
                for group in attribute_groups:
                    atomic_attributes = group.split(',')
                    all_atomic_attributes.extend(atomic_attributes)

    if not all_atomic_attributes:
        print("错误：未能在文件中找到任何可供编码的属性值。")
        return

    # 使用 collections.Counter 高效地计算频率
    frequency_counter = Counter(all_atomic_attributes)
    
    # --- 第2步: 根据频率降序创建编码映射 ---
    encoding_map = {attribute: i for i, (attribute, count) in enumerate(frequency_counter.most_common(), 1)}

    # --- 第3步: 输出统计信息并将映射表保存到文件 ---
    total_atomic_attributes_count = len(all_atomic_attributes)
    unique_atomic_attributes_count = len(encoding_map)
    
    print("\n--- 统计信息 ---")
    print(f"已处理文件数量: {len(valid_file_paths)}")
    print(f"总行数: {total_lines}")
    print(f"总属性组数 (空格分隔): {total_groups}")
    print(f"总独立属性值出现次数 (逗号分隔, 非唯一): {total_atomic_attributes_count}")
    print(f"唯一独立属性值数量: {unique_atomic_attributes_count}")
    
    map_file_path = os.path.join(output_dir, 'encoding_map.txt')
    with open(map_file_path, 'w', encoding='utf-8') as f:
        f.write("新编码 -> (独立属性值, 频率)\n")
        f.write("="*35 + "\n")
        decoding_map = {v: k for k, v in encoding_map.items()}
        for code in sorted(decoding_map.keys()):
            original_attribute = decoding_map[code]
            freq = frequency_counter[original_attribute]
            f.write(f"{code} -> ('{original_attribute}', {freq})\n")

    print(f"\n完整的编码映射表已保存至: {map_file_path}")
    print("--- 编码映射表示例 (频率最高的前10个) ---")
    print("新编码 -> (独立属性值, 频率)")
    for i, (attribute, count) in enumerate(frequency_counter.most_common(10), 1):
        print(f"{i} -> ('{attribute}', {count})")

    # --- 第4步: 重新遍历文件，进行编码并写入新文件 ---
    print("\n--- 开始重新编码文件 ---")
    for file_path in valid_file_paths:
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)
        output_file_path = os.path.join(output_dir, f"{name}_encoded{ext}")

        with open(file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                line = line.strip()
                if not line:
                    outfile.write('\n')
                    continue
                
                attribute_groups = line.split(' ')
                new_encoded_groups = []
                for group in attribute_groups:
                    atomic_attributes = group.split(',')
                    # 对每个独立的属性值进行编码
                    encoded_atomic_attrs = [str(encoding_map[attr]) for attr in atomic_attributes]
                    # 将编码后的值用逗号重新组合
                    new_encoded_groups.append(','.join(encoded_atomic_attrs))
                
                # 将重新组合好的属性组用空格连接并写入新文件
                outfile.write(' '.join(new_encoded_groups) + '\n')
        
        print(f"已生成编码文件: {output_file_path}")

    print("\n--- 所有操作已完成 ---")

if __name__ == '__main__':
    files_to_process = [
        '/data/fxy/FilterVector/FilterVectorData/app_reviews/app_reviews_base_labels.txt',
    ]

    output_directory = '/data/fxy/FilterVector/FilterVectorData/app_reviews/base_11'

    
    recode_atomic_attributes(files_to_process, output_directory)