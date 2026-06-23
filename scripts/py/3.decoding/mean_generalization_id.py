# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 16:24:56 2025

@author: WANGLIANGFU
"""
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import utils
from shutil import copyfile
copyfile('../common_settings.py', 'common_settings.py')
copyfile('../utils.py', 'utils.py')
if __name__ == "__main__":

    dirname = 'D:/xiangmu/results/decoding_id'
    pattern = os.path.join(dirname, '**', '*_generalization_results.npz')
    found_files = [os.path.normpath(f).replace('\\', '/') for f in glob.glob(pattern, recursive=True)]

    all_scores_acc = []
    all_scores_conf = []
    
    for file_path in found_files:
        data = np.load(file_path, allow_pickle=True)

        if 'scores_gen_acc' in data:
            all_scores_acc.append(data['scores_gen_acc'])
        if 'scores_gen_conf' in data:
            all_scores_conf.append(data['scores_gen_conf'])

        if 'times_train' in data and 'times_test' in data:
            times_train = data['times_train']
            times_test = data['times_test']
        
        data.close()
    
    if all_scores_acc:
        mean_scores_acc = np.mean(all_scores_acc, axis=0)

        utils.plot_temporal_generalization_with_contour(
            scores=mean_scores_acc,
            times_train=times_train,
            times_test=times_test,
            title=f'Average correct vs incorrect Generalization ({len(all_scores_acc)} files)',
            save_path=os.path.join(dirname, 'average_acc_generalization.png'),
            vmin=0.4, vmax=0.6
        )
    
    if all_scores_conf:
        mean_scores_conf = np.mean(all_scores_conf, axis=0)

        utils.plot_temporal_generalization_with_contour(
            scores=mean_scores_conf,
            times_train=times_train,
            times_test=times_test,
            title=f'Average high vs low Confidence Generalization ({len(all_scores_conf)} files)',
            save_path=os.path.join(dirname, 'average_conf_generalization.png'),
            vmin=0.4, vmax=0.6
        )