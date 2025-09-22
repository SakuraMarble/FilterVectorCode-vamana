# -*- coding: utf-8 -*-

import os
import csv

def merge_text_files(input_files, output_file):
    """
    合并多个文本文件到一个文件中。

    :param input_files: 一个包含输入文件路径的列表 (e.g., ['file1.txt', 'file2.txt'])
    :param output_file: 输出文件的路径 (e.g., 'merged.txt')
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for filename in input_files:
                if not os.path.exists(filename):
                    print(f"警告：文件 '{filename}' 不存在，已跳过。")
                    continue
                with open(filename, 'r', encoding='utf-8') as infile:
                    # 读取输入文件的所有内容并写入输出文件
                    content = infile.read()
                    outfile.write(content)
                    # 确保文件末尾有换行符，以免下一个文件的内容紧接着上一行
                    if not content.endswith('\n'):
                        outfile.write('\n')
        print(f"成功！文本文件合并完成 -> {output_file}")
    except IOError as e:
        print(f"合并文本文件时发生错误: {e}")

def merge_binary_files(input_files, output_file):
    """
    合并多个二进制文件（如 .fvecs）到一个文件中。

    :param input_files: 一个包含输入文件路径的列表 (e.g., ['file1.fvecs', 'file2.fvecs'])
    :param output_file: 输出文件的路径 (e.g., 'merged.fvecs')
    """
    try:
        with open(output_file, 'wb') as outfile: # 'wb' 表示二进制写入模式
            for filename in input_files:
                if not os.path.exists(filename):
                    print(f"警告：文件 '{filename}' 不存在，已跳过。")
                    continue
                with open(filename, 'rb') as infile: # 'rb' 表示二进制读取模式
                    # 读取整个文件的二进制内容并直接写入输出文件
                    outfile.write(infile.read())
        print(f"成功！二进制文件合并完成 -> {output_file}")
    except IOError as e:
        print(f"合并二进制文件时发生错误: {e}")

def merge_csv_files(input_files, output_file):
    """
    合并多个 CSV 文件到一个文件中，并只保留一个表头。

    :param input_files: 一个包含输入文件路径的列表 (e.g., ['file1.csv', 'file2.csv'])
    :param output_file: 输出文件的路径 (e.g., 'merged.csv')
    """
    try:
        header_written = False
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            for filename in input_files:
                if not os.path.exists(filename):
                    print(f"警告：文件 '{filename}' 不存在，已跳过。")
                    continue
                with open(filename, 'r', encoding='utf-8') as infile:
                    reader = csv.reader(infile)
                    header = next(reader) # 读取表头

                    if not header_written:
                        writer.writerow(header)
                        header_written = True
                    
                    # 写入剩余的数据行
                    for row in reader:
                        writer.writerow(row)
        print(f"成功！CSV 文件合并完成 -> {output_file}")
    except (IOError, StopIteration) as e:
        print(f"合并 CSV 文件时发生错误: {e}")


# --- 主程序 ---
if __name__ == "__main__":
    # --- 1. 设置文件名 ---
    txt_files_to_merge = ['/data/fxy/FilterVector/FilterVectorData/celeba/query_sub_base_l=2/celeba_query_labels.txt',
                          '/data/fxy/FilterVector/FilterVectorData/celeba/query_sub_base_l=5/celeba_query_labels.txt',
                          '/data/fxy/FilterVector/FilterVectorData/celeba/query_sub_base_l=8/celeba_query_labels.txt']
    
    fvecs_files_to_merge = ['/data/fxy/FilterVector/FilterVectorData/celeba/query_sub_base_l=2/celeba_query.fvecs', 
                            '/data/fxy/FilterVector/FilterVectorData/celeba/query_sub_base_l=5/celeba_query.fvecs',
                            '/data/fxy/FilterVector/FilterVectorData/celeba/query_sub_base_l=8/celeba_query.fvecs']

    csv_files_to_merge = ['/data/fxy/FilterVector/FilterVectorData/celeba/query_sub_base_l=2/profiled_sub_base_l=2.csv',
                          '/data/fxy/FilterVector/FilterVectorData/celeba/query_sub_base_l=5/profiled_sub_base_l=5.csv',
                          '/data/fxy/FilterVector/FilterVectorData/celeba/query_sub_base_l=8/profiled_sub_base_l=8.csv']

    merged_txt_file = '/data/fxy/FilterVector/FilterVectorData/celeba/query_4/celeba_query_labels.txt'
    merged_fvecs_file = '/data/fxy/FilterVector/FilterVectorData/celeba/query_4/celeba_query.fvecs'
    merged_csv_file = '/data/fxy/FilterVector/FilterVectorData/celeba/query_4/attribute_coverage_merged.csv'

    # --- 2. 执行合并操作 ---
    print("开始合并文本文件...")
    merge_text_files(txt_files_to_merge, merged_txt_file)

    print("\n开始合并 .fvecs 文件...")
    merge_binary_files(fvecs_files_to_merge, merged_fvecs_file)
    
    print("\n开始合并 .csv 文件...")
    merge_csv_files(csv_files_to_merge, merged_csv_file)

    print("\n所有任务已完成。")
