import numpy as np
import struct
import time

def read_fvecs(filename):
    """
    读取 .fvecs 文件。
    返回一个 numpy 数组，每一行是一个向量。
    """
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            dim_data = f.read(4)
            if not dim_data:
                break
            dim = struct.unpack('i', dim_data)[0]
            
            vector_data = f.read(dim * 4)
            if len(vector_data) != dim * 4:
                break
            
            vector = struct.unpack(f'{dim}f', vector_data)
            vectors.append(list(vector))
            
    return np.array(vectors, dtype=np.float32)

def write_fvecs(filename, vectors):
    """
    将 numpy 数组写入 .fvecs 文件。
    """
    with open(filename, 'wb') as f:
        for vector in vectors:
            dim = len(vector)
            f.write(struct.pack('i', dim))
            f.write(struct.pack(f'{dim}f', *vector))

def generate_random_vectors(num_new, dim, min_vals, max_vals):
    """
    在指定的全局范围内生成完全随机的新向量。
    """
    random_vectors = np.random.uniform(low=min_vals, high=max_vals, size=(num_new, dim))
    return random_vectors.astype(np.float32)

def expand_vectors_and_attributes(
    original_fvecs_path, 
    original_txt_path, 
    new_fvecs_path, 
    new_txt_path,
    expansion_factor=10,
    crop_ratio=1.0  # 裁剪比例参数，默认为1.0（不裁剪）
):
    """
    主函数，用于读取、裁剪、扩展并写入新的向量和属性文件。
    新向量将是完全随机生成的。
    """
    start_time = time.time()
    
    # --- 1. 读取和分析原始数据 ---
    print("--- 步骤 1: 读取和分析原始数据 ---")
    
    print(f"正在从 {original_fvecs_path} 读取向量...")
    try:
        original_vectors = read_fvecs(original_fvecs_path)
    except FileNotFoundError:
        print(f"错误: 文件未找到 -> {original_fvecs_path}")
        return
        
    if original_vectors.size == 0:
        print("错误: fvecs 文件为空或格式不正确。")
        return
        
    num_original_vectors, dim = original_vectors.shape
    print(f"读取完成。共 {num_original_vectors} 个向量，每个向量维度为 {dim}。")

    print(f"正在从 {original_txt_path} 读取属性...")
    try:
        with open(original_txt_path, 'r') as f:
            original_attributes = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"错误: 文件未找到 -> {original_txt_path}")
        return
        
    num_original_attributes = len(original_attributes)
    print(f"读取完成。共 {num_original_attributes} 行属性。")

    if num_original_vectors != num_original_attributes:
        print(f"错误：向量数量 ({num_original_vectors}) 和属性行数 ({num_original_attributes}) 不匹配！")
        return

    # --- 根据 crop_ratio 裁剪数据 ---
    if 0.0 < crop_ratio < 1.0:
        print(f"\n--- 步骤 1.5: 按比例 {crop_ratio:.2%} 裁剪数据 ---")
        
        # 计算需要保留的数据量
        num_to_keep = int(num_original_vectors * crop_ratio)
        
        # 对向量和属性列表进行切片
        original_vectors = original_vectors[:num_to_keep]
        original_attributes = original_attributes[:num_to_keep]
        
        # 更新数量统计
        num_original_vectors = len(original_vectors)
        num_original_attributes = len(original_attributes)
        
        print(f"裁剪完成。将使用前 {num_original_vectors} 条数据进行后续操作。")
    elif crop_ratio > 1.0 or crop_ratio <= 0:
        print(f"警告: crop_ratio 的值 ({crop_ratio}) 无效，应在 (0, 1] 范围内。将使用全部数据。")


    # --- 2. 计算（可能已被裁剪的）数据集的全局坐标范围 ---
    print("\n--- 步骤 2: 计算向量坐标的全局范围 ---")
    min_vals = np.min(original_vectors, axis=0)
    max_vals = np.max(original_vectors, axis=0)
    print("已计算出所有维度上的最小值和最大值，将在此范围内生成新向量。")
    
    
    # --- 3. 生成新数据 ---
    print(f"\n--- 步骤 3: 开始扩展数据，每个属性集将对应 {expansion_factor} 个新随机向量 ---")
    all_new_vectors = []
    all_new_attributes = []

    for i, attribute_line in enumerate(original_attributes):
        new_vectors = generate_random_vectors(expansion_factor, dim, min_vals, max_vals)
        all_new_vectors.extend(new_vectors)
        all_new_attributes.extend([attribute_line] * expansion_factor)
        
        if (i + 1) % 10000 == 0 or (i + 1) == num_original_attributes:
            progress = (i + 1) / num_original_attributes
            print(f"  进度: {i + 1}/{num_original_attributes} ({progress:.1%}) 已处理...")

    # --- 4. 写入新文件 ---
    print("\n--- 步骤 4: 写入新文件 ---")
    
    final_vectors_np = np.array(all_new_vectors, dtype=np.float32)

    print(f"正在将 {len(final_vectors_np)} 个新向量写入 {new_fvecs_path}...")
    write_fvecs(new_fvecs_path, final_vectors_np)
    print("新向量文件写入成功。")

    print(f"正在将 {len(all_new_attributes)} 行新属性写入 {new_txt_path}...")
    with open(new_txt_path, 'w') as f:
        f.write('\n'.join(all_new_attributes) + '\n')
    print("新属性文件写入成功。")
    
    # --- 5. 最终统计信息 ---
    end_time = time.time()
    print("\n--- 所有操作完成！最终统计信息 ---")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"原始向量数 (裁剪后): {num_original_vectors}")
    print(f"扩展因子: {expansion_factor}")
    print("-" * 30)
    print(f"生成的新向量总数: {len(final_vectors_np)}")
    print(f"生成的新属性行总数: {len(all_new_attributes)}")
    print(f"新 fvecs 文件: {new_fvecs_path}")
    print(f"新 txt 文件: {new_txt_path}")
    print("-" * 30)


# --- 使用示例 ---
if __name__ == '__main__':
    original_fvecs_file = '/data/fxy/FilterVector/FilterVectorData/amazing_file/amazing_file_base.fvecs'      
    original_txt_file = '/data/fxy/FilterVector/FilterVectorData/amazing_file/base_10/amazing_file_base_labels.txt'        
    
    new_fvecs_file = '/data/fxy/FilterVector/FilterVectorData/amazing_file_1\\10_10/amazing_file_base_1\\10_10_base.fvecs'          
    new_txt_file = '/data/fxy/FilterVector/FilterVectorData/amazing_file_1\\10_10/base_0/amazing_file_1\\10_10_base_labels.txt'            
    
    
    expansion_count = 10 # 设定每个原始向量要扩展的数量
   
    crop_proportion = 0.1
    
    expand_vectors_and_attributes(
        original_fvecs_file,
        original_txt_file,
        new_fvecs_file,
        new_txt_file,
        expansion_factor=expansion_count,
        crop_ratio=crop_proportion
    )