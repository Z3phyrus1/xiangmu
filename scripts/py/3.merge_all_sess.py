# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 18:31:03 2025

@author: WANGLIANGFU
"""

import os
import pandas as pd
import re
import numpy as np


def get_seq_from_filename(filename):

    # 去掉文件扩展名
    name_without_ext = os.path.splitext(filename)[0]
    
    # 使用正则表达式查找所有数字
    numbers = re.findall(r'\d+', name_without_ext)
    
    if numbers:
        # 获取最后一个数字序列
        last_number = numbers[-1]
        if len(last_number) >= 2:
            return last_number[-2:]
        else:
            return last_number.zfill(2)
    else:
        return "00"  # 默认值


def analyze_data(df):

    print("\n数据分析:")
    print("=" * 50)
    
    # 检查总行数
    print(f"总行数: {len(df)}")
    
    # 检查每个seq的行数
    if 'seq' in df.columns:
        print("\n各seq数据行数:")
        seq_counts = df['seq'].value_counts().sort_index()
        for seq, count in seq_counts.items():
            print(f"  seq {seq}: {count} 行")
    
    # 检查每个id的数据行数
    if 'id' in df.columns:
        print("\n各id数据行数:")
        id_counts = df['id'].value_counts().sort_index()
        for id_val, count in id_counts.items():
            print(f"  id {id_val}: {count} 行")
            
    # 检查每个session的数据行数
    if 'session' in df.columns and 'id' in df.columns:
        print("\n各id+session数据行数:")
        df['id_session'] = df['id'].astype(str) + '_' + df['session'].astype(str)
        id_session_counts = df['id_session'].value_counts().sort_index()
        for id_sess, count in id_session_counts.items():
            print(f"  {id_sess}: {count} 行")
    
    # 检查每个block的数据行数
    if 'block' in df.columns:
        print("\n各block数据行数:")
        block_counts = df['block'].value_counts().sort_index()
        for block, count in block_counts.items():
            print(f"  block {block}: {count} 行")
    
    # 检查每个task的数据行数
    if 'task' in df.columns:
        print("\n各task数据行数:")
        task_counts = df['task'].value_counts()
        for task, count in task_counts.items():
            print(f"  {task}: {count} 行")
            if task in ['motion', 'orientation']:
                percentage = count / len(df) * 100
                print(f"    占比: {percentage:.2f}%")
    
    # 检查是否有缺失值
    print("\n缺失值检查:")
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if len(missing_values) > 0:
        print(missing_values)
    else:
        print("  无缺失值")
    
    # 检查是否有重复行
    duplicates = df.duplicated().sum()
    print(f"重复行数: {duplicates}")
    
    print("=" * 50)


if __name__ == "__main__":
    # 设置数据目录路径
    folder_path = r"D:/xiangmu/data/raw_behavior"
    all_data = []
    
    print(f"扫描目录: {folder_path}")
    
    # 检查目录是否存在
    if not os.path.exists(folder_path):
        print(f"错误：目录不存在 - {folder_path}")
        print("请检查路径是否正确")
        exit()
    
    # 遍历子文件夹
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        
        # 跳过非文件夹
        if not os.path.isdir(subfolder_path):
            continue
        
        print(f"处理子文件夹: {subfolder}")
        
        # 在子文件夹中查找Excel文件
        for filename in os.listdir(subfolder_path):
            if filename.endswith(".xlsx"):
                file_path = os.path.join(subfolder_path, filename)
                
                try:
                    df = pd.read_excel(file_path, sheet_name=0)
                    
                    # 从文件名中提取seq
                    seq_value = get_seq_from_filename(filename)
                    
                    # 添加seq列
                    df['seq'] = seq_value
                    
                    all_data.append(df)
                    print(f"  已读取: {filename} (seq: {seq_value}, 行数: {len(df)})")
                    
                except Exception as e:
                    print(f"  读取文件 {filename} 时出错: {str(e)}")
    
    if all_data:
        # 合并所有数据
        merge_df = pd.concat(all_data, ignore_index=True)
        print(f"合并完成，总数据行数: {len(merge_df)}")
        
        # 原始的数据处理逻辑
        merge_df['id'] = merge_df['id_sess'].astype(str).str[:2]
        merge_df['signed_mu'] = merge_df['side'] * merge_df['mu']
        merge_df['signed_coherence'] = merge_df['side'] * merge_df['coherence']
        merge_df['id'] = pd.to_numeric(merge_df['id'], errors='coerce')

        def get_session_last_digit(x):
            s = str(x)
            return float(s[-1]) if s and s[-1].isdigit() else np.nan
        
        merge_df['session'] = merge_df['id_sess'].apply(get_session_last_digit)
        def update_task(row):
            if pd.notna(row['id']) and pd.notna(row['session']):
                id_plus_session = row['id'] + row['session']
                if id_plus_session % 2 != 0:
                    if row['decision'] == 1:
                        return 'motion'
                    else:
                        return 'orientation'
                else:
                    if row['decision'] == 2:
                        return 'motion'
                    else:
                        return 'orientation'
            return row['task']
        
        merge_df['task'] = merge_df.apply(update_task, axis=1)
        
        # 调整列顺序，将seq列放在id之后
        cols = list(merge_df.columns)
        
        # 确保列的顺序：id, seq, session, id_sess, ... 
        # 先将seq列移到id之后
        if 'id' in cols and 'seq' in cols:
            # 移除seq列
            cols.remove('seq')
            # 找到id列的位置
            id_index = cols.index('id')
            # 在id列之后插入seq列
            cols.insert(id_index + 1, 'seq')
        
        # 重新排列列
        merge_df = merge_df[cols]
        
        # 修改 id 列的数据为 '原id_seq' 格式，例如 '1_01'
        merge_df['id'] = merge_df['id'].astype(str) + '_' + merge_df['seq']
        
        # 分析数据
        analyze_data(merge_df)
        
        # 计算理论行数
        total_subjects = merge_df['id'].nunique()
        total_sessions = merge_df['session'].nunique()
        total_blocks = merge_df['block'].nunique() if 'block' in merge_df.columns else 0

        
        # 假设每个block有100个trial（50 motion + 50 orientation）
        if total_blocks > 0:
            theoretical_rows = total_subjects * total_sessions * total_blocks * 100
            print(f"理论总行数: {theoretical_rows}")
            print(f"实际总行数: {len(merge_df)}")
            print(f"差异: {len(merge_df) - theoretical_rows}")
        

        
        # 保存为CSV文件
        output_path = os.path.join("D:/xiangmu/data/behavior/idsess/data.csv")
        merge_df.to_csv(output_path, index=False, na_rep="NaN")
        print(f"\n数据已保存到: {output_path}")
        print(f"总数据行数: {len(merge_df)}")
        
    else:
        print("未找到任何Excel文件")
        print(f"请检查 {folder_path} 的子文件夹中是否有.xlsx文件")