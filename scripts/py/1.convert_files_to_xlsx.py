# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 15:24:20 2025

@author: WANGLIANGFU
"""
import pandas as pd
import os
import re
from io import StringIO

def natural_sort_key(s):
    """
    自然排序键函数，使文件按数字顺序排序
    """
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', s)]

def convert_files_to_xlsx(working_dir, encoding='gbk'):

    level1_dirs = [d for d in os.listdir(working_dir) 
                  if os.path.isdir(os.path.join(working_dir, d))]
    level1_dirs.sort(key=natural_sort_key)
    
    print(f"找到 {len(level1_dirs)} 个一级文件夹:")
    for d in level1_dirs:
        print(f"  - {d}")
    
    # 处理转换计数
    total_converted = 0
    
    # 遍历一级子文件夹
    for level1_dir in level1_dirs:
        level1_path = os.path.join(working_dir, level1_dir)
        
        print(f"\n处理一级文件夹: {level1_dir}")
        print(f"路径: {level1_path}")
        
        # 2. 获取并排序二级子文件夹
        level2_dirs = [d for d in os.listdir(level1_path) 
                      if os.path.isdir(os.path.join(level1_path, d))]
        level2_dirs.sort(key=natural_sort_key)
        
        print(f"  找到 {len(level2_dirs)} 个二级文件夹:")
        for d in level2_dirs:
            print(f"    - {d}")
        
        # 遍历二级子文件夹
        for level2_dir in level2_dirs:
            level2_path = os.path.join(level1_path, level2_dir)
            
            print(f"\n  处理二级文件夹: {level2_dir}")
            print(f"  路径: {level2_path}")
            
            # 3. 获取并排序文件
            files = [f for f in os.listdir(level2_path) 
                    if os.path.isfile(os.path.join(level2_path, f))]
            files.sort(key=natural_sort_key)
            
            print(f"    找到 {len(files)} 个文件:")
            for f in files:
                print(f"      - {f}")
            
            # 可能的文件前缀模式
            possible_prefixes = [
                level1_dir,  # 如 "sub01"
                level2_dir,  # 如 "zjy01"
                level1_dir.lower(),  # 小写版本
                level2_dir.lower(),  # 小写版本
            ]
            
            # 遍历文件
            for filename in files:
                file_path = os.path.join(level2_path, filename)
                
                # 检查文件是否已经被处理过
                if filename.endswith('.xlsx'):
                    print(f"    跳过已处理的Excel文件: {filename}")
                    continue
                
                # 跳过.mat文件
                if filename.endswith('.mat'):
                    print(f"    跳过.mat文件: {filename}")
                    continue
                
                # 检查文件名是否以任何可能的模式开头
                starts_with_any = any(filename.startswith(prefix) for prefix in possible_prefixes)
                
                if not starts_with_any:
                    # 如果文件名不以任何前缀开头，尝试匹配部分前缀
                    for prefix in possible_prefixes:
                        # 检查文件名是否包含前缀（不一定是开头）
                        if prefix in filename:
                            print(f"    文件名 {filename} 包含前缀 {prefix}，尝试处理")
                            starts_with_any = True
                            break
                
                if starts_with_any:
                    try:
                        print(f"    尝试转换文件: {filename}")
                        
                        # 读取文件内容
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                        
                        print(f"      文件大小: {len(content)} 字符")
                        
                        # 检查文件是否为空
                        if not content.strip():
                            print(f"      警告：文件 {filename} 为空，跳过")
                            continue
                        
                        # 使用pandas读取制表符分隔的数据
                        data = pd.read_csv(
                            StringIO(content), 
                            sep='\t', 
                            header=None,
                        )
                        
                        print(f"      读取成功，数据形状: {data.shape}")
                        
                        # 生成输出文件名（原文件名 + .xlsx）
                        output_filename = f"{filename}.xlsx"
                        output_path = os.path.join(level2_path, output_filename)
                        
                        # 检查输出文件是否已存在
                        if os.path.exists(output_path):
                            print(f"      警告：输出文件已存在，覆盖: {output_filename}")
                        
                        # 保存为Excel文件
                        data.to_excel(output_path, index=False, header=False)
                        
                        total_converted += 1
                        print(f"      成功转换: {filename} -> {output_filename}")
                        
                    except UnicodeDecodeError as e:
                        print(f"      解码错误: {e}")
                        print(f"      尝试使用不同的编码...")
                        
                        # 尝试其他常见编码
                        other_encodings = ['utf-8', 'gb2312', 'big5', 'latin1']
                        for enc in other_encodings:
                            try:
                                with open(file_path, 'r', encoding=enc) as f:
                                    content = f.read()
                                
                                data = pd.read_csv(
                                    StringIO(content), 
                                    sep='\t', 
                                    header=None,
                                )
                                
                                output_filename = f"{filename}.xlsx"
                                output_path = os.path.join(level2_path, output_filename)
                                data.to_excel(output_path, index=False, header=False)
                                
                                total_converted += 1
                                print(f"      使用编码 {enc} 成功转换: {filename} -> {output_filename}")
                                break
                                
                            except Exception:
                                continue
                        else:
                            print(f"      无法使用任何编码读取文件 {filename}")
                        
                    except pd.errors.EmptyDataError:
                        print(f"      错误：文件 {filename} 没有数据")
                    except Exception as e:
                        print(f"      处理文件 {filename} 时出错: {str(e)}")
                else:
                    print(f"    跳过不匹配的文件: {filename} (前缀: {possible_prefixes})")
    
    return total_converted


def convert_all_text_files(working_dir, encoding='gbk'):
    """
    转换所有文本文件，不考虑文件名前缀
    """
    print(f"扫描目录: {working_dir}")
    
    total_converted = 0
    
    # 使用递归遍历所有文件
    for root, dirs, files in os.walk(working_dir):
        # 对目录和文件进行自然排序
        dirs.sort(key=natural_sort_key)
        files.sort(key=natural_sort_key)
        
        for filename in files:
            file_path = os.path.join(root, filename)
            
            # 只处理文本文件，跳过Excel和.mat文件
            if (filename.endswith('.txt') or 
                (not filename.endswith('.xlsx') and 
                 not filename.endswith('.mat') and 
                 '.' not in filename)):
                
                try:
                    rel_path = os.path.relpath(file_path, working_dir)
                    print(f"处理文件: {rel_path}")
                    
                    # 读取文件内容
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    
                    # 使用pandas读取制表符分隔的数据
                    data = pd.read_csv(
                        StringIO(content), 
                        sep='\t', 
                        header=None,
                    )
                    
                    # 生成输出文件名（原文件名 + .xlsx）
                    output_filename = f"{filename}.xlsx"
                    output_path = os.path.join(root, output_filename)
                    
                    # 保存为Excel文件
                    data.to_excel(output_path, index=False, header=False)
                    
                    total_converted += 1
                    print(f"  成功转换: {filename} -> {output_filename}")
                    
                except Exception as e:
                    print(f"  处理文件 {filename} 时出错: {str(e)}")
    
    return total_converted


if __name__ == "__main__":
    working_dir = 'D:/xiangmu/data/behavior_for_behavior'
    
    if not os.path.exists(working_dir):
        print(f"错误：目录不存在 - {working_dir}")
    else:
        print("=" * 60)
        print("开始转换文件...")
        print("=" * 60)
        
        # 方法1：按前缀匹配转换文件
        # total_converted = convert_files_to_xlsx(working_dir, encoding='gbk')
        
        # 方法2：转换所有文本文件（不检查前缀）
        total_converted = convert_all_text_files(working_dir, encoding='gbk')
        
        print("=" * 60)
        print(f"转换完成！共转换了 {total_converted} 个文件")
        print("=" * 60)