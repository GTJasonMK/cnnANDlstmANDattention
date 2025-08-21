import pandas as pd
import os
import argparse

def remove_columns_from_csv(input_file, columns_to_remove):
    """
    从CSV文件中删除指定列并保存到同一目录
    
    参数:
    input_file (str): 输入的CSV文件路径
    columns_to_remove (list): 要删除的列名列表
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 检查要删除的列是否存在
    existing_columns = set(df.columns)
    columns_to_remove_existing = [col for col in columns_to_remove if col in existing_columns]
    columns_not_found = [col for col in columns_to_remove if col not in existing_columns]
    
    if columns_not_found:
        print(f"警告: 以下列未找到: {columns_not_found}")
    
    if not columns_to_remove_existing:
        print("没有有效的列可删除")
        return
    
    # 删除指定列
    df = df.drop(columns=columns_to_remove_existing)
    
    # 生成输出文件名
    directory = os.path.dirname(input_file)
    filename = os.path.basename(input_file)
    name, ext = os.path.splitext(filename)
    output_file = os.path.join(directory, f"{name}_removed{ext}")
    
    # 保存文件
    df.to_csv(output_file, index=False)
    print(f"处理完成! 文件已保存为: {output_file}")
    print(f"删除了以下列: {columns_to_remove_existing}")

def main():
    input_file = "E:/code/data/all_six_datasets/traffic/traffic.csv"
    columns=['date']
    # 检查输入文件是否存在
    if not os.path.isfile(input_file):
        print(f"错误: 文件 '{input_file}' 不存在")
        return
    
    remove_columns_from_csv(input_file, columns)

if __name__ == "__main__":
    main()