"""
Created on Wed Nov 19 19:23:06 2025

@author: DELL
"""

import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from shutil import copyfile

copyfile('../common_settings.py', 'common_settings.py')
copyfile('../utils.py', 'utils.py')
import common_settings as CS
import utils
import mne
from mne.decoding import GeneralizingEstimator, cross_val_multiscore
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__": 
    subject = 'sub01'
    sub_id_int = int(re.search(r'\d+', subject).group())
    #trial = '2'
    working_dir = os.path.join('../../..', 'data', 'clean_EEG_dual', subject)
    print(f"被试文件夹:  {working_dir}")
    
    # ============================================================
    # 1.读取文件
    # ============================================================
    task1_stim_files = sorted(glob(os.path.join(working_dir, 'clean_epochs_task1_ICA_*.fif')))#不分试次：'clean_epochs_task1_ICA_*.fif'
    task1_resp_files = sorted(glob(os.path.join(working_dir, 'clean_epochs_response1_ICA_*.fif')))#不分试次：'clean_epochs_response1_ICA_*.fif'
    task2_stim_files = sorted(glob(os.path.join(working_dir, 'clean_epochs_task2_ICA_*.fif')))#不分试次：'clean_epochs_task2_ICA_*.fif'
    
    # ============================================================
    # 2.读取行为数据
    # ============================================================
    behavior_path = os.path.join('../../..', 'data', 'behavior_for_behavior', 'id', 'data_wide_wmPred.txt')
    fit_par_path = os.path.join('../../..', 'data', 'behavior_for_behavior', 'id', 'fit_par.txt')
    
    df_pred = pd.read_csv(behavior_path, sep=';')
    w_val = utils.get_id_w_threshold(fit_par_path, sub_id_int)
    print(f"被试 {subject} 阈值 w = {w_val}")
    
    df_sub = df_pred[df_pred['id'] == sub_id_int].copy()
    all_id_sess = df_sub['id_sess'].unique()
    
    n_files = len(task1_resp_files)
    id_sess_to_use = all_id_sess[-n_files: ] if len(all_id_sess) >= n_files else all_id_sess
    
    beh_blocks = [df_sub[df_sub['id_sess'] == sid].copy() for sid in id_sess_to_use]
    beh_aligned = pd.concat(beh_blocks, axis=0, ignore_index=True)
    
    # ============================================================
    # 3.结果保存路径
    # ============================================================
    results_dir = os.path.join('../../..', 'results', 'decoding_id', subject,)
    figures_dir = os.path.join('../../..', 'figures', 'decoding_id', subject,)
    for d in [results_dir, figures_dir]:
        os.makedirs(d, exist_ok=True)
    
    # ============================================================
    # 4.读取 Epochs
    # ============================================================
    
    # Task1 Stimulus（用于计算 accuracy）
    epochs_t1_stim = mne.concatenate_epochs([mne.read_epochs(f, verbose=False) for f in task1_stim_files])
    
    # Task1 Response（训练集）
    epochs_t1_resp_list = [mne.read_epochs(f, verbose=False) for f in task1_resp_files]
    for ep in epochs_t1_resp_list:
        ep.apply_baseline(CS.baseline_response)
    epochs_t1_resp = mne.concatenate_epochs(epochs_t1_resp_list)
    epochs_t1_resp = epochs_t1_resp.crop(-0.5, CS.tmax_response).resample(500)
    X_train = epochs_t1_resp.get_data(copy=False)
    times_train = epochs_t1_resp.times
    
    # Task2 Stimulus（测试集）
    epochs_t2_stim_list = [mne.read_epochs(f, verbose=False) for f in task2_stim_files]
    for ep in epochs_t2_stim_list:
        ep.apply_baseline(CS.baseline_task2)
    epochs_t2_stim = mne.concatenate_epochs(epochs_t2_stim_list)
    epochs_t2_stim = epochs_t2_stim.crop(CS.tmin_task2, CS.tmax_task2).resample(500)
    X_test = epochs_t2_stim.get_data(copy=False)
    times_test = epochs_t2_stim.times
    
    print(f"\nTrain (Task1 Response): {X_train.shape}, times: {times_train[0]:.2f} ~ {times_train[-1]:.2f}s")
    print(f"Test (Task2 Stimulus):  {X_test.shape}, times: {times_test[0]:.2f} ~ {times_test[-1]:.2f}s")
    
    # ============================================================
    # 5.长度对齐
    # ============================================================
    n_min = min(len(epochs_t1_stim), len(epochs_t1_resp), len(epochs_t2_stim), len(beh_aligned))
    X_train = X_train[:n_min]
    X_test = X_test[:n_min]
    epochs_t1_stim = epochs_t1_stim[:n_min]
    epochs_t1_resp = epochs_t1_resp[:n_min]
    beh_aligned = beh_aligned.iloc[:n_min].reset_index(drop=True)
    
    # ============================================================
    # 6.构造标签
    # ============================================================
    print("\n构造标签...")
    
    # 正确/错误
    y_train_acc = utils.compute_accuracy_from_epochs(epochs_t1_stim, epochs_t1_resp)
    y_test_acc = beh_aligned['acc2'].astype(int).values
    
    # 高/低自信
    y_train_conf = (beh_aligned['abs1'] > w_val).astype(int).values
    y_test_conf = (beh_aligned['abs2'] > w_val).astype(int).values
    
    print(f"Train acc:   {np.unique(y_train_acc, return_counts=True)}")
    print(f"Test acc:   {np.unique(y_test_acc, return_counts=True)}")
    print(f"Train conf: {np.unique(y_train_conf, return_counts=True)}")
    print(f"Test conf:  {np.unique(y_test_conf, return_counts=True)}")
    
    # ============================================================
    # 7.绘图函数
    # ============================================================

    # ============================================================
    # 8.Cross-task Generalization:  正确 vs 错误
    # ============================================================
    print("\n" + "="*60)
    print("Cross-task Generalization: 正确 vs 错误")
    print("Task1 Response FIT → Task2 Stimulus SCORE")
    print("="*60)
    
    # 创建 GeneralizingEstimator
    clf_gen = GeneralizingEstimator(
        make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', class_weight='balanced')),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=True
    )
    
    # 训练
    print("\n训练中...")
    clf_gen.fit(X_train, y_train_acc)
    
    # 测试
    print("\n测试中...")
    scores_acc = clf_gen.score(X_test, y_test_acc)
    print(f"Scores shape: {scores_acc.shape}")  # (n_times_train, n_times_test)
    
    # 绘图
    utils.plot_temporal_generalization(
        scores=scores_acc,
        times_train=times_train,
        times_test=times_test,
        title=f'Correct vs Incorrect:  Task1 Response → Task2 Stimulus ({subject})',
        save_path=os.path.join(figures_dir, f'{subject}_gen_acc_t1resp_to_t2stim.png'),
        vmin=0.4,
        vmax=0.6
    )
    
    # ============================================================
    # 9.Cross-task Generalization: 高 vs 低自信
    # ============================================================
    print("\n" + "="*60)
    print("Cross-task Generalization: 高 vs 低自信")
    print("Task1 Response FIT → Task2 Stimulus SCORE")
    print("="*60)
    
    # 创建新的 GeneralizingEstimator
    clf_gen_conf = GeneralizingEstimator(
        make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', class_weight='balanced')),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=True
    )
    
    # 训练
    print("\n训练中...")
    clf_gen_conf.fit(X_train, y_train_conf)
    
    # 测试
    print("\n测试中...")
    scores_conf = clf_gen_conf.score(X_test, y_test_conf)
    print(f"Scores shape: {scores_conf.shape}")
    
    # 绘图
    utils.plot_temporal_generalization(
        scores=scores_conf,
        times_train=times_train,
        times_test=times_test,
        title=f'High vs Low Confidence: Task1 Response → Task2 Stimulus ({subject})',
        save_path=os.path.join(figures_dir, f'{subject}_gen_conf_t1resp_to_t2stim.png'),
        vmin=0.4,
        vmax=0.6
    )
    
    # ============================================================
    # 10.保存结果
    # ============================================================
    print("\n" + "="*60)
    print("保存结果")
    print("="*60)
    
    save_dict = {
        'scores_gen_acc': scores_acc,
        'scores_gen_conf':  scores_conf,
        'times_train': times_train,
        'times_test': times_test,
    }
    
    np.savez(os.path.join(results_dir, f'{subject}_generalization_results.npz'), **save_dict)
    print(f"结果已保存到 {results_dir}")
    print("\n所有处理完成！")