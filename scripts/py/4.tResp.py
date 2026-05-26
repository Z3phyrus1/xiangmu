# -*- coding: utf-8 -*-
"""
Created on Tue May 19 19:57:43 2026

@author: WANGLIANGFU
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

data_path = r'D:\xiangmu\data\behavior_for_behavior\data_nu.txt'
save_dir = r'D:\xiangmu\figures\behavior_for_behavior'
os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(data_path, delimiter=';')
df['tResp'] = pd.to_numeric(df['tResp'], errors='coerce')
df = df.dropna(subset=['tResp', 'dual'])

for dual_value in [0, 1]:
    df_sub = df[df['dual'] == dual_value]
    if df_sub.empty:
        print(f'dual={dual_value}: 没有数据，跳过绘图。')
        continue

    log_RTs = np.log(df_sub['tResp'])
    median_logRT = np.median(log_RTs)
    mean_logRT = np.mean(log_RTs)

    plt.figure(figsize=(8,5))
    plt.hist(log_RTs, bins=30, color='skyblue', edgecolor='black', alpha=0.7, label='log(RT) 分布')
    plt.axvline(median_logRT, color='red', linestyle='--', linewidth=2, label=f'median: {median_logRT:.2f}')
    plt.axvline(mean_logRT, color='blue', linestyle='-', linewidth=2, label=f'mean: {mean_logRT:.2f}')

    plt.xlabel('log(RT)')
    plt.ylabel('样本数')
    plt.title(f'dual={dual_value} log(RT) 分布直方图')
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'dual_{dual_value}.pdf')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"dual={dual_value} 图已保存到：{save_path}")