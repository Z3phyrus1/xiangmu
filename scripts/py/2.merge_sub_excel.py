# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 16:54:46 2025

@author: WANGLIANGFU
"""

import pandas as pd
import os
import re

def merge_excel_file(working_dir, naming_format='full_name'):
    column_names = [
        'id_sess',
        'block',
        'trial',
        'dual',
        'lines_first',
        'decision',
        'task',
        'side',
        'coherence',
        'mu',
        'meanTilt',
        'sdTilt',
        'tOn',
        'tOff',
        'rr',
        'acc',
        'tResp',
    ]
    
    # 获取工作目录下的一级子文件夹并排序
    subfolders = sorted([f for f in os.listdir(working_dir) 
                         if os.path.isdir(os.path.join(working_dir, f))])
    
    print(f"找到 {len(subfolders)} 个一级文件夹")
    
    for folder in subfolders:
        folder_path = os.path.join(working_dir, folder)
        
        print(f"\n处理文件夹: {folder}")
        
        # 根据命名格式确定输出文件名
        if naming_format == 'full_name':
            # 使用完整的文件夹名称作为文件名
            output_file = f"{folder}.xlsx"
        elif naming_format == 'number_suffix':
            match = re.search(r'(\d+)(\D*)', folder)
            if match:
                number_part = match.group(1)
                suffix_part = match.group(2) if match.group(2) else ''
                if suffix_part:
                    output_file = f"{number_part}_{suffix_part}.xlsx"
                else:
                    output_file = f"{number_part}.xlsx"
            else:
                output_file = f"{folder}.xlsx"
        elif naming_format == 'original':
            match = re.search(r'\d+', folder)
            if match:
                output_file = f"{match.group()}.xlsx"
            else:
                output_file = f"{folder}.xlsx"
        else:
            output_file = f"{folder}.xlsx"
        
        output_full_path = os.path.join(working_dir, output_file)
        all_data = []
        
        # 获取一级文件夹下的所有二级子文件夹（如 zjy01, zjy02, zjy03 等）
        level2_dirs = sorted([d for d in os.listdir(folder_path) 
                              if os.path.isdir(os.path.join(folder_path, d))])
        
        print(f"  找到 {len(level2_dirs)} 个二级文件夹: {level2_dirs}")
        
        # 遍历所有二级文件夹
        for level2_dir in level2_dirs:
            level2_path = os.path.join(folder_path, level2_dir)
            
            print(f"    处理二级文件夹: {level2_dir}")
            
            # 获取二级文件夹中的所有Excel文件
            excel_files = sorted([f for f in os.listdir(level2_path) 
                                  if f.endswith('.xlsx') and not f.startswith('~$')])
            
            print(f"      找到 {len(excel_files)} 个Excel文件")
            
            # 遍历二级文件夹中的Excel文件
            for filename in excel_files:
                # 跳过输出文件自身（避免循环读取）
                if filename == output_file:
                    print(f"      跳过输出文件自身: {filename}")
                    continue
                
                file_path = os.path.join(level2_path, filename)
                try:
                    df = pd.read_excel(file_path, header=None)
                    all_data.append(df)
                    print(f"      已读取: {filename}")
                except Exception as e:
                    print(f"      读取文件 {filename} 时出错: {str(e)}")
        
        # 如果找到数据，合并并保存
        if all_data:
            merge_df = pd.concat(all_data, ignore_index=True)
            header_df = pd.DataFrame([column_names])
            final_df = pd.concat([header_df, merge_df], ignore_index=True)
            final_df.to_excel(output_full_path, index=False, header=False)
            print(f"  已合并并保存: {output_file}")
            print(f"  合并了 {len(all_data)} 个文件，共 {len(merge_df)} 行数据")
        else:
            print(f"  警告：在 {folder} 中未找到Excel文件")


def batch_merge_excel(working_dir, naming_format='full_name'):
    if not os.path.exists(working_dir):
        print(f"错误：目录不存在 - {working_dir}")
        return
    
    print("=" * 60)
    print(f"开始处理目录: {working_dir}")
    print("=" * 60)
    
    merge_excel_file(working_dir, naming_format)
    
    print("=" * 60)
    print("批量合并完成！")
    print("=" * 60)


def merge_all_excel_files(root_dir, naming_format='full_name'):
    if not os.path.exists(root_dir):
        print(f"错误：目录不存在 - {root_dir}")
        return
    
    print(f"开始处理根目录: {root_dir}")
    
    # 遍历根目录下的一级子文件夹
    level1_dirs = sorted([d for d in os.listdir(root_dir) 
                          if os.path.isdir(os.path.join(root_dir, d))])
    
    print(f"找到 {len(level1_dirs)} 个一级文件夹")
    
    for level1_dir in level1_dirs:
        level1_path = os.path.join(root_dir, level1_dir)
        
        print(f"\n{'='*40}")
        print(f"处理一级文件夹: {level1_dir}")
        print(f"{'='*40}")
        
        # 对每个一级文件夹执行合并操作
        merge_excel_file(level1_path, naming_format)


if __name__ == "__main__":
    working_dir = r'D:/xiangmu/data/behavior_for_behavior'

    merge_all_excel_files(working_dir, naming_format='full_name')