# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 19:50:45 2025

@author: WANGLIANGFU
"""


import os

def remove_last_column(working_dir):
    for subdir in os. listdir(working_dir):
        subdir_path = os.path.join(working_dir, subdir)
        if not os.path. isdir(subdir_path):
            continue
        file_prefix = subdir
        
        for folder in range(1, 5):
            folder_name = f"{folder:02d}"
            folder_path = os.path.join(subdir_path, folder_name)
            
            if not os.path.exists(folder_path):
                continue

            for filename in os.listdir(folder_path):
                if (filename.startswith(file_prefix) and 
                    not filename.endswith('. mat') and 
                    not filename. endswith('.xlsx')):
                    
                    file_path = os.path. join(folder_path, filename)
                    if os.path.isfile(file_path):
                        
                        # 读取文件
                        with open(file_path, 'rb') as f:
                            lines = f.readlines()
                        
                        # 处理每一行，删除最后一列
                        new_lines = []
                        for line in lines:
                            line = line.decode('latin-1')
                            parts = line.rstrip('\n').split('\t')
                            # 删除最后一列
                            if len(parts) > 1:
                                new_line = '\t'.join(parts[:-1]) + '\n'
                            else:
                                new_line = line
                            new_lines.append(new_line)
                        
                        # 写回原文件
                        with open(file_path, 'w', encoding='latin-1') as f:
                            f.writelines(new_lines)
                        
                        print(f"已处理: {file_path}")

if __name__ == "__main__":
    working_dir = 'D:/xiangmu/data/eeg/sub15/wym2'
    remove_last_column(working_dir)