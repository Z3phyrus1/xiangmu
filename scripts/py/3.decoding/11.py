# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 13:48:12 2025
@author: WANGLIANGFU
"""

import os
from glob import glob
import numpy as np
import pandas as pd
import re
from shutil import copyfile

copyfile('../common_settings.py', 'common_settings.py')
copyfile('../utils.py', 'utils.py')
import common_settings as CS
import utils as utils

import mne
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from mne.decoding import SlidingEstimator, cross_val_multiscore


def read_concat(files, baseline, crop_tmin, crop_tmax):
    print(f"Loading {len(files)} files...")
    eps = [mne.read_epochs(f, verbose=False) for f in files]
    for e in eps:
        e.apply_baseline(baseline)
    epochs = mne.concatenate_epochs(eps)
    epochs.crop(crop_tmin, crop_tmax).resample(100)
    return epochs


def _sigmask_to_slice_clusters(sig_mask, p_vals):

    clusters = []
    cluster_p_vals = []
    in_cluster = False
    start = 0

    for i, flag in enumerate(sig_mask):
        if flag and not in_cluster:
            in_cluster = True
            start = i
        elif (not flag) and in_cluster:
            in_cluster = False
            stop = i
            clusters.append((slice(start, stop),))
            cluster_p_vals.append(float(np.min(p_vals[start:stop])))

    if in_cluster:
        stop = len(sig_mask)
        clusters.append((slice(start, stop),))
        cluster_p_vals.append(float(np.min(p_vals[start:stop])))

    return clusters, np.array(cluster_p_vals, dtype=float)


def run_sliding_analysis(
    X, y, times, task_name, label_name,
    subject, trial, figures_dir,
    n_splits, test_size, random_state
):
    print(f"\n--- Running: {task_name} | {label_name} ---")

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            solver='liblinear',
            class_weight='balanced',
            random_state=random_state,
            penalty='l1',
            Cs=np.logspace(-3, 3, 7),
            cv=5,
            scoring='roc_auc',
            max_iter=int(1e3),
            n_jobs=1,
            verbose=0
        )
    )

    slider = SlidingEstimator(clf, scoring='roc_auc', n_jobs=1, verbose=False)
    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    scores = cross_val_multiscore(slider, X, y, cv=cv, n_jobs=1, verbose=1)

    print("Running permutation test (TFCE)...")
    # 必须解包4个返回值
    t_vals, clusters, cluster_p_vals, H0 = mne.stats.permutation_cluster_1samp_test(
        scores - 0.5,
        tail=1,
        n_permutations=1000,
        threshold=dict(start=0, step=0.2),  # TFCE
        seed=12345,
        n_jobs=1,
        out_type='mask',
        verbose=False
    )

    # TFCE下可能无离散簇 -> 用逐点p值构造簇用于高光
    if clusters is None or len(clusters) == 0:
        # H0: permutation max-stat 分布, t_vals: 每个时间点统计量
        p_tfce = (np.sum(H0[:, None] >= t_vals[None, :], axis=0) + 1) / (len(H0) + 1)
        sig_mask = p_tfce < 0.05
        clusters, cluster_p_vals = _sigmask_to_slice_clusters(sig_mask, p_tfce)

    # 调试信息（可保留）
    print(f"n_clusters_for_plot = {len(clusters)}")
    if len(clusters) > 0:
        print(f"min_cluster_p = {np.min(cluster_p_vals):.6f}")
    else:
        print("No significant cluster under p<0.05")

    scores_mean = np.mean(scores, axis=0)
    scores_sem = np.std(scores, axis=0) / np.sqrt(scores.shape[0])

    save_path = os.path.join(figures_dir, f'{subject}_{trial}_{task_name}_{label_name}_sliding.png')

    # 传4元组（与你utils函数解包一致）
    utils.plot_with_cluster_highlight(
        times=times,
        scores_mean=scores_mean,
        scores_sem=scores_sem,
        permutation_result=(t_vals, clusters, cluster_p_vals, H0),
        title=f"{task_name}: {label_name} ({subject} Tr{trial})",
        save_path=save_path,
        chance_level=0.5
    )

    return scores


def process_subject_trial(subject, trial, test_size, n_splits, random_state, df_pred, fit_par_path):
    sub_id_int = int(re.search(r'\d+', subject).group())
    working_dir = os.path.join('../../..', 'data', 'clean_EEG', subject)

    if not os.path.exists(working_dir):
        print(f"[SKIP] {subject} 文件夹不存在: {working_dir}")
        return None

    print(f"\n{'='*60}")
    print(f"处理: {subject}, Trial: {trial}")
    print(f"{'='*60}")

    task1_resp_files = sorted(glob(os.path.join(working_dir, f'clean_epochs_response1_ICA_*{trial}_sess*.fif')))
    task2_stim_files = sorted(glob(os.path.join(working_dir, f'clean_epochs_task2_ICA_*{trial}_sess*.fif')))
    task1_stim_files = sorted(glob(os.path.join(working_dir, f'clean_epochs_task1_ICA_*{trial}_sess*.fif')))

    if len(task1_resp_files) == 0:
        print(f"[SKIP] {subject} Trial {trial}: 未找到 Task1 Response 文件")
        return None
    if len(task2_stim_files) == 0:
        print(f"[SKIP] {subject} Trial {trial}: 未找到 Task2 Stim 文件")
        return None
    if len(task1_stim_files) == 0:
        print(f"[SKIP] {subject} Trial {trial}: 未找到 Task1 Stim 文件")
        return None

    w_val = utils.get_idsess_w_threshold(fit_par_path, sub_id_int)
    if w_val is None:
        print(f"[SKIP] {subject} Trial {trial}: 无法获取 W 阈值")
        return None

    df_sub = df_pred[df_pred['id'].str.split('_').str[0].astype(int) == sub_id_int].copy()
    beh_aligned = df_sub[df_sub['seq'].astype(str) == str(trial)].reset_index(drop=True)

    if len(beh_aligned) == 0:
        print(f"[SKIP] {subject} Trial {trial}: 无行为数据")
        return None

    print("Loading Task1 Response...")
    ep_t1_r = read_concat(task1_resp_files, CS.baseline_response, -0.5, CS.tmax_response)
    X_t1 = ep_t1_r.get_data(copy=False)
    times_t1 = ep_t1_r.times

    print("Loading Task2 Stimulus...")
    ep_t2_s = read_concat(task2_stim_files, CS.baseline_task2, CS.tmin_task2, CS.tmax_task2)
    X_t2 = ep_t2_s.get_data(copy=False)
    times_t2 = ep_t2_s.times

    ep_t1_s = mne.concatenate_epochs([mne.read_epochs(f, verbose=False) for f in task1_stim_files])

    y_acc = utils.compute_accuracy_from_epochs(ep_t1_s, ep_t1_r)
    y_conf = (beh_aligned['abs1'] > w_val).astype(int).values

    print(f"Acc counts: {np.unique(y_acc, return_counts=True)}")
    print(f"Conf counts: {np.unique(y_conf, return_counts=True)}")

    if len(np.unique(y_acc)) < 2:
        print(f"[SKIP] {subject} Trial {trial}: y_acc 只有一个类别，跳过 Accuracy 解码")
        y_acc = None
    if len(np.unique(y_conf)) < 2:
        print(f"[SKIP] {subject} Trial {trial}: y_conf 只有一个类别，跳过 Confidence 解码")
        y_conf = None

    results_dir = os.path.join('../../..', 'results', 'decoding_idsess', subject, trial)
    figures_dir = os.path.join('../../..', 'figures', 'decoding_idsess', subject, trial)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    if y_acc is not None:
        s_t1_acc = run_sliding_analysis(
            X_t1, y_acc, times_t1, "Task1_Resp", "Accuracy",
            subject, trial, figures_dir, n_splits, test_size, random_state
        )
        s_t2_acc = run_sliding_analysis(
            X_t2, y_acc, times_t2, "Task2_Stim", "Accuracy",
            subject, trial, figures_dir, n_splits, test_size, random_state
        )
    else:
        s_t1_acc = None
        s_t2_acc = None

    if y_conf is not None:
        s_t1_conf = run_sliding_analysis(
            X_t1, y_conf, times_t1, "Task1_Resp", "Confidence",
            subject, trial, figures_dir, n_splits, test_size, random_state
        )
        s_t2_conf = run_sliding_analysis(
            X_t2, y_conf, times_t2, "Task2_Stim", "Confidence",
            subject, trial, figures_dir, n_splits, test_size, random_state
        )
    else:
        s_t1_conf = None
        s_t2_conf = None

    np.savez(
        os.path.join(results_dir, f'{subject}_{trial}_sliding_scores.npz'),
        scores_t1_acc=s_t1_acc,
        scores_t1_conf=s_t1_conf,
        scores_t2_acc=s_t2_acc,
        scores_t2_conf=s_t2_conf,
        times_t1=times_t1,
        times_t2=times_t2
    )

    print(f"[DONE] {subject} Trial {trial} 完成！")
    return True


if __name__ == "__main__":
    # ==========================
    # 批量参数
    # ==========================
    sub_start = 1
    sub_end = 10
    trials = ['2', '3']

    test_size = 0.2
    n_splits = 100
    random_state = 42

    behavior_path = os.path.join('../../..', 'data', 'behavior', 'idsess', 'data_wide_wmPred.txt')
    fit_par_path = os.path.join('../../..', 'data', 'behavior', 'idsess', 'fit_par.txt')
    df_pred = pd.read_csv(behavior_path, sep=';')

    success_count = 0
    skip_count = 0

    for sub_num in range(sub_start, sub_end + 1):
        subject = f'sub{sub_num:02d}'
        for trial in trials:
            result = process_subject_trial(
                subject=subject,
                trial=trial,
                test_size=test_size,
                n_splits=n_splits,
                random_state=random_state,
                df_pred=df_pred,
                fit_par_path=fit_par_path
            )
            if result is None:
                skip_count += 1
            else:
                success_count += 1

    print("\n" + "=" * 60)
    print("全部处理完成！")
    print(f"成功: {success_count} 个")
    print(f"跳过: {skip_count} 个")
    print("=" * 60)