# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 16:21:17 2025

@author: WANGLIANGFU
"""
import numpy as np
import os
import mne
import glob as glob
import shutil

def process_vmrk_files():
    # 配置被试信息
    subject = 'sub16'  # 可以根据需要修改
    working_dir = os.path.join('../../', 'data', 'eeg', subject,)
    print(f"被试文件夹: {working_dir}")

    # 获取所有.vmrk文件，使用glob.glob并转换为Python列表
    vmrk_files = glob.glob(os.path.join(working_dir, "*", '*vmrk'))
    vmrk_files.sort()  # 使用Python的sort而不是numpy的sort
    print(f"找到 {len(vmrk_files)} 个.vmrk文件:")
    for file in vmrk_files:
        print(f"  {file}")

    if not vmrk_files:  # 现在可以安全地检查空列表
        print("未找到任何.vmrk文件")
        return

    # 处理每个.vmrk文件
    for vmrk_path in vmrk_files:
        try:
            print(f"\n处理文件: {vmrk_path}")
            
            # 创建.csv文件路径
            csv_path = vmrk_path.replace('.vmrk', '.csv')
            
            # 将.vmrk文件复制为.csv文件（实际应用中可能需要更复杂的格式转换）
            # 这里假设.vmrk文件是文本格式，可以直接转换
            shutil.copy2(vmrk_path, csv_path)
            print(f"已将.vmrk文件转换为.csv: {csv_path}")

            # 读取.csv文件进行处理
            with open(csv_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            header_lines = []
            data_lines = []
            in_data_section = False

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('Mk') and '=' in line:
                    data_lines.append(line)
                else:
                    header_lines.append(line)

            # 解析数据
            parsed_data = []
            for line in data_lines:
                # 分割标记号和内容
                marker_part, content_part = line.split('=', 1)
                marker_num = marker_part[2:]  # 提取数字部分

                fields = content_part.split(',')
                type_field = fields[0] if len(fields) > 0 else ''
                description = fields[1] if len(fields) > 1 else ''
                position = fields[2] if len(fields) > 2 else ''
                size = fields[3] if len(fields) > 3 else ''
                channel = fields[4] if len(fields) > 4 else ''
                
                parsed_data.append({
                    'marker': marker_part,
                    'type': type_field,
                    'description': description,
                    'position': position,
                    'size': size,
                    'channel': channel,
                    'original_line': line
                })

            # 显示处理前的刺激类型统计
            stimulus_types = {}
            for item in parsed_data:
                desc = item['description']
                if desc:
                    stimulus_types[desc] = stimulus_types.get(desc, 0) + 1

            print("处理前的刺激类型统计:")
            for stim_type, count in sorted(stimulus_types.items()):
                print(f"  {stim_type}: {count}")

            # 第一步：过滤掉S 36的行
            filtered_data = [item for item in parsed_data if item['description'] != 'S 36']

            # 第二步：处理连续重复的S 1和S 2
            final_data = []
            i = 0
            while i < len(filtered_data):
                current_item = filtered_data[i]
                current_desc = current_item['description']

                if (i + 1 < len(filtered_data) and 
                    current_desc in ['S  1', 'S  2'] and 
                    filtered_data[i + 1]['description'] == current_desc):
                    # 保留当前行，跳过下一行
                    final_data.append(current_item)
                    i += 2  # 跳过下一行
                else:
                    # 保留当前行
                    final_data.append(current_item)
                    i += 1

            # 显示处理后的刺激类型统计
            final_stimulus_types = {}
            for item in final_data:
                desc = item['description']
                if desc:
                    final_stimulus_types[desc] = final_stimulus_types.get(desc, 0) + 1

            print("处理后的刺激类型统计:")
            for stim_type, count in sorted(final_stimulus_types.items()):
                print(f"  {stim_type}: {count}")

            # 生成输出内容
            output_lines = header_lines.copy()
            output_lines.append('')
            for item in final_data:
                output_lines.append(item['original_line'])

            # 保存处理后的.csv文件
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output_lines))  # 修复了换行符的问题
            print(f"已保存处理后的.csv文件: {csv_path}")

            # 将处理后的.csv文件转换回.vmrk文件
            # 先删除原有的.vmrk文件
            if os.path.exists(vmrk_path):
                os.remove(vmrk_path)
            
            # 将处理后的.csv文件重命名为.vmrk
            os.rename(csv_path, vmrk_path)
            print(f"已将处理后的文件改回.vmrk格式: {vmrk_path}")

            print(f"文件 {os.path.basename(vmrk_path)} 处理完成!")

        except Exception as e:
            print(f"处理文件 {vmrk_path} 时出错: {str(e)}")
            # 如果出错，尝试恢复原始文件
            if os.path.exists(csv_path) and not os.path.exists(vmrk_path):
                os.rename(csv_path, vmrk_path)
                print("已恢复原始.vmrk文件")

    print("\n所有文件处理完成!")

if __name__ == "__main__":
    process_vmrk_files()