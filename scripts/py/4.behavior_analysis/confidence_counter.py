# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 10:12:09 2025

@author: WANGLIANGFU
"""
import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re 
from collections import Counter
def get_w_threshold(fit_par_path, sub_id):
    try:
        df = pd.read_csv(fit_par_path, sep='\s+')
        df.columns = df.columns.str.replace('"', '').str.strip()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.replace('"', '').str.strip()
        
        row = df[df['id'].astype(str) == str(sub_id)]
        if len(row) == 0: return None
        
        if 'model' in row.columns:
            sub2_row = row[row['model'] == 'sub2']
            if not sub2_row.empty: return float(sub2_row['w'].values[0])
        return float(row['w'].iloc[0])
    except:
        return None
    
if __name__ == '__main__':
    subject = 'sub02'
    sub_id_int = int(re.search(r'\d+', subject).group())
    behavior_path = os.path.join('../..', 'data', 'behavior', 'data_wide_wmPred.txt')
    fit_par_path = os.path.join('../..', 'data', 'behavior', 'fit_par.txt')
    df_pred = pd.read_csv(behavior_path, sep=';')
    w_val = get_w_threshold(fit_par_path, sub_id_int)
    fig,ax = plt.subplots(figsize = (10, 6))
    ax.hist(df_pred['abs1'].values)
    ax.axvline(w_val)
    confidence = np.array(df_pred['abs1'].values >= w_val, dtype = int)
    print(Counter(confidence))