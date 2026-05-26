# -*- coding: utf-8 -*-
"""
Created on Tue May 19 20:27:13 2026

@author: WANGLIANGFU
"""

import pandas as pd

# 读取数据
data_path = r'D:\xiangmu\data\behavior_for_behavior\data_nu.txt'
df = pd.read_csv(data_path, delimiter=';')

# 保证数据格式正确
df['acc'] = pd.to_numeric(df['acc'], errors='coerce')
df = df.dropna(subset=['acc', 'dual', 'task'])

# 分组计算准确率
grouped = df.groupby(['dual', 'task'])['acc'].mean().unstack()

# 打印结果
print("各分组正确率（acc均值）：")
print(grouped)