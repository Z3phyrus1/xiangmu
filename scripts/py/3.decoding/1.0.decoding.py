# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 16:30:44 2025

@author:  WANGLIANGFU
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
from sklearn.model_selection import StratifiedShuffleSplit
from mne.decoding import cross_val_multiscore


if __name__ == "__main__":
    subject = 'sub31'
    sub_id_int = int(re.search(r'\d+', subject).group())

    working_dir = os.path.join('../../..', 'data', 'clean_EEG', subject)
    print(f"被试文件夹: {working_dir}")

    task1_response_files = sorted(glob(os.path.join(working_dir, 'clean_epochs_response1_ICA_*.fif')))
    task2_files = sorted(glob(os.path.join(working_dir, 'clean_epochs_task2_ICA_*.fif')))
    task1_files = sorted(glob(os.path.join(working_dir, 'clean_epochs_task1_ICA_*.fif')))

    print(f"找到 {len(task1_response_files)} 个 Response1 文件")
    print(f"找到 {len(task2_files)} 个 Task2 文件")
    print(f"找到 {len(task1_files)} 个 Task1 文件")

    behavior_path = os.path.join('../../..', 'data', 'behavior_for_behavior', 'data_nu.txt')
    fit_par_path = os.path.join('../../..', 'data', 'behavior_for_behavior', 'fit_par.txt')

    df_pred = pd.read_csv(behavior_path, sep=';')
    w_val = utils.get_idsess_w_threshold(fit_par_path, sub_id_int)
    if w_val is None:
        raise RuntimeError("无法获取 W 阈值，停止。")
    print(f"[INFO] 被试 {subject} 阈值 w = {w_val}")

    # 兼容 id: 18 / 18.0 / 18_xxx
    id_prefix = (
        df_pred['id'].astype(str)
        .str.split('_').str[0]
        .str.replace('.0', '', regex=False)
        .str.strip()
    )
    df_sub = df_pred[id_prefix == str(sub_id_int)].copy()
    print(f"[INFO] 该被试行为条数: {len(df_sub)}")

    # 兼容 id_sess / id_session
    sess_col = 'id_sess' if 'id_sess' in df_sub.columns else ('id_session' if 'id_session' in df_sub.columns else None)
    if sess_col is None:
        raise RuntimeError("行为文件缺少 id_sess / id_session 列")

    all_id_sess = df_sub[sess_col].astype(str).unique()
    print(f"[INFO] 所有 session id（原始顺序）: {all_id_sess}")

    # 路径
    results_dir_base = os.path.join('../..', 'results', 'decoding_id', subject)
    figures_dir_base = os.path.join('../..', 'figures', 'decoding_id', subject)
    os.makedirs(results_dir_base, exist_ok=True)
    os.makedirs(figures_dir_base, exist_ok=True)

    # ============================================================
    # Task1 Response：拼接 + 标签
    # ============================================================
    epochs_task1_response_list = []
    y_confidence_list = []

    # 用文件数裁剪 session 映射
    if len(all_id_sess) >= len(task1_response_files):
        id_sess_to_use = all_id_sess[-len(task1_response_files):]
    else:
        id_sess_to_use = all_id_sess
    print(f"[INFO] 将使用 session id: {id_sess_to_use}")

    for i, filename in enumerate(task1_response_files):
        print(f"\n[{i+1}] 处理文件: {os.path.basename(filename)}")
        epochs = mne.read_epochs(filename, verbose=False)
        epochs.apply_baseline(CS.baseline_response)
        epochs_task1_response_list.append(epochs)
        print(f"    -> Epochs 数量: {len(epochs)}")

        if i >= len(id_sess_to_use):
            print("    -> ✗ 文件索引超出可用 session id 范围")
            continue

        target_id_sess = str(id_sess_to_use[i])
        beh_data_for_file = df_sub[df_sub[sess_col].astype(str) == target_id_sess].copy().reset_index(drop=True)
        print(f"    -> 匹配到行为数据: {len(beh_data_for_file)} 条")

        if len(beh_data_for_file) == 0:
            print("    -> ✗ 此文件无匹配行为")
            continue

        # data_nu: 同一trial两行 -> 取 decision==1 做 trial-level
        beh_trial = beh_data_for_file.copy()
        if 'decision' in beh_trial.columns:
            beh_trial['decision'] = pd.to_numeric(beh_trial['decision'], errors='coerce')
            beh_trial['block'] = pd.to_numeric(beh_trial['block'], errors='coerce')
            beh_trial['trial'] = pd.to_numeric(beh_trial['trial'], errors='coerce')
            beh_trial = beh_trial[beh_trial['decision'] == 1].sort_values(['block', 'trial']).reset_index(drop=True)

        # confidence基底：abs1 > s1 > s
        if 'abs1' in beh_trial.columns:
            conf_base = pd.to_numeric(beh_trial['abs1'], errors='coerce').values
        elif 's1' in beh_trial.columns:
            conf_base = np.abs(pd.to_numeric(beh_trial['s1'], errors='coerce').values)
        elif 's' in beh_trial.columns:
            conf_base = np.abs(pd.to_numeric(beh_trial['s'], errors='coerce').values)
        else:
            print("    -> ✗ 缺少 abs1/s1/s 列")
            continue

        high_conf_all = (conf_base > w_val).astype(int)
        selection = epochs.selection

        if len(high_conf_all) == 0 or np.max(selection) >= len(high_conf_all):
            print(f"    -> ✗ 行为trial不足: beh={len(high_conf_all)}, max(selection)={np.max(selection)}")
            continue

        y_labels = high_conf_all[selection]
        print(f"    -> 最终标签数: {len(y_labels)}, Epochs数: {len(epochs)}")

        if len(y_labels) == len(epochs):
            y_confidence_list.append(y_labels)
            print("    -> ✓ 成功匹配标签")
        else:
            print("    -> ✗ 标签数不匹配")

    if len(epochs_task1_response_list) == 0:
        raise RuntimeError("没有可用的 Response1 epochs。")

    epochs_task1_response = mne.concatenate_epochs(epochs_task1_response_list)
    epochs_task1_response = epochs_task1_response.crop(-0.5, CS.tmax_response)
    epochs_task1_response.resample(100)

    X_task1 = epochs_task1_response.get_data(copy=False)
    y_task1_lr = epochs_task1_response.events[:, 2]

    y_task1_conf = None
    if len(y_confidence_list) == len(epochs_task1_response_list):
        y_task1_conf = np.concatenate(y_confidence_list)
        if len(y_task1_conf) != len(X_task1):
            print(f"[WARN] y_task1_conf长度与X不一致: {len(y_task1_conf)} vs {len(X_task1)}")
            n_min = min(len(y_task1_conf), len(X_task1))
            y_task1_conf = y_task1_conf[:n_min]
            X_task1 = X_task1[:n_min]
            y_task1_lr = y_task1_lr[:n_min]
    else:
        print(f"[WARN] confidence标签仅匹配 {len(y_confidence_list)}/{len(epochs_task1_response_list)} 个文件，跳过confidence解码")

    # ============================================================
    # 分类器
    # ============================================================
    clf_lr = utils.make_generalizing_estimator()
    clf_lr.fit(X_task1, y_task1_lr)

    epochs_task1 = mne.concatenate_epochs([mne.read_epochs(f, verbose=False) for f in task1_files])
    y_task1_acc = utils.compute_accuracy_from_epochs(epochs_task1, epochs_task1_response)

    # 保底长度对齐
    n_min_acc = min(len(X_task1), len(y_task1_acc))
    X_task1_acc = X_task1[:n_min_acc]
    y_task1_acc = y_task1_acc[:n_min_acc]

    clf_acc = utils.make_generalizing_estimator()
    clf_acc.fit(X_task1_acc, y_task1_acc)

    clf_td = utils.make_sliding_estimator()
    cv = StratifiedShuffleSplit(**CS.cv_args)

    # ============================================================
    # Task1 Accuracy 解码
    # ============================================================
    scores_task1_acc = cross_val_multiscore(clf_td, X_task1_acc, y_task1_acc, cv=cv, n_jobs=-1, verbose=1)
    permutation_t1_acc = mne.stats.permutation_cluster_1samp_test(
        scores_task1_acc - 0.5, tail=1, seed=12345, n_jobs=-1, out_type='mask'
    )

    scores_task1_acc_mean = np.mean(scores_task1_acc, axis=0)
    scores_task1_acc_sem = np.std(scores_task1_acc, axis=0) / np.sqrt(scores_task1_acc.shape[0])
    times = epochs_task1_response.times[:scores_task1_acc_mean.shape[0]]
    utils.plot_significant_window_patterns(
    X=X_task1_acc,
    y=y_task1_acc,
    times=times,
    info=epochs_task1_response.info,      # 这里一定用和X匹配的info
    permutation_result=permutation_t1_acc,
    title_prefix=f"Task1: Correct vs Incorrect Temporal Decoding ({subject})",
    save_path_prefix=os.path.join(figures_dir_base, f'{subject}_task1_acc')
)
    utils.plot_with_cluster_highlight(
        times=times,
        scores_mean=scores_task1_acc_mean,
        scores_sem=scores_task1_acc_sem,
        permutation_result=permutation_t1_acc,
        title=f"Task1: Correct vs Incorrect Temporal Decoding ({subject})",
        save_path=os.path.join(figures_dir_base, f'{subject}_task1_acc_temporal_with_highlight.png'),
        chance_level=0.5
    )

    # ============================================================
    # Task1 Confidence 解码
    # ============================================================
    scores_task1_conf = None
    if y_task1_conf is not None and len(np.unique(y_task1_conf)) > 1:
        n_min_conf = min(len(X_task1), len(y_task1_conf))
        X_task1_conf = X_task1[:n_min_conf]
        y_task1_conf_use = y_task1_conf[:n_min_conf]

        print("\n正在进行 Task1 自信度解码...")
        scores_task1_conf = cross_val_multiscore(clf_td, X_task1_conf, y_task1_conf_use, cv=cv, n_jobs=-1, verbose=1)

        permutation_t1_conf = mne.stats.permutation_cluster_1samp_test(
            scores_task1_conf - 0.5, tail=1, seed=12345, n_jobs=-1, out_type='mask'
        )

        scores_task1_conf_mean = np.mean(scores_task1_conf, axis=0)
        scores_task1_conf_sem = np.std(scores_task1_conf, axis=0) / np.sqrt(scores_task1_conf.shape[0])

        utils.plot_with_cluster_highlight(
            times=times[:scores_task1_conf_mean.shape[0]],
            scores_mean=scores_task1_conf_mean,
            scores_sem=scores_task1_conf_sem,
            permutation_result=permutation_t1_conf,
            title=f"Task1: High vs Low Confidence Temporal Decoding ({subject})",
            save_path=os.path.join(figures_dir_base, f'{subject}_task1_conf_temporal_with_highlight.png'),
            chance_level=0.5
        )
    else:
        print("[WARN] Task1 confidence 标签不可用（None或单类别）")

    print("\n" + "="*60)
    print("Task2 处理")
    print("="*60)

    # ============================================================
    # Task2
    # ============================================================
    epochs_task2 = mne.concatenate_epochs([mne.read_epochs(f, verbose=False) for f in task2_files])
    epochs_task2 = epochs_task2.crop(-0.5, CS.tmax_response)
    epochs_task2.resample(100)

    X_task2 = epochs_task2.get_data(copy=True)
    times_task2 = epochs_task2.times
    y_task2_lr = np.array([1 if e % 10 == 1 else 2 for e in epochs_task2.events[:, 2]])

    scores_task2_lr = clf_lr.score(X_task2, y_task2_lr)

    # Task2 accuracy泛化（长度对齐）
    n_min_t2_acc = min(len(X_task2), len(y_task1_acc))
    scores_task2_acc = clf_acc.score(X_task2[:n_min_t2_acc], y_task1_acc[:n_min_t2_acc])

    # Task2 confidence 时间解码
    if y_task1_conf is not None:
        n_min_t2_conf = min(len(X_task2), len(y_task1_conf))
        y_conf_t2 = y_task1_conf[:n_min_t2_conf]
        X_t2_conf = X_task2[:n_min_t2_conf]

        if len(np.unique(y_conf_t2)) > 1:
            print("\n正在进行 Task2 自信度解码...")
            scores_task2_conf = cross_val_multiscore(clf_td, X_t2_conf, y_conf_t2, cv=cv, n_jobs=-1, verbose=1)

            permutation_t2_conf = mne.stats.permutation_cluster_1samp_test(
                scores_task2_conf - 0.5, tail=1, seed=12345, n_jobs=-1, out_type='mask'
            )

            scores_task2_conf_mean = np.mean(scores_task2_conf, axis=0)
            scores_task2_conf_sem = np.std(scores_task2_conf, axis=0) / np.sqrt(scores_task2_conf.shape[0])

            utils.plot_with_cluster_highlight(
                times=times_task2[:scores_task2_conf_mean.shape[0]],
                scores_mean=scores_task2_conf_mean,
                scores_sem=scores_task2_conf_sem,
                permutation_result=permutation_t2_conf,
                title=f"Task2: High vs Low Confidence Temporal Decoding ({subject})",
                save_path=os.path.join(figures_dir_base, f'{subject}_task2_conf_temporal_with_highlight.png'),
                chance_level=0.5
            )
        else:
            print("[WARN] Task2 confidence 标签单类别，跳过")
    else:
        print("[WARN] 无 confidence 标签，跳过 Task2 confidence")

    # ============================================================
    # 保存
    # ============================================================
    save_dict = {
        'scores_task1_acc': scores_task1_acc,
        'scores_task2_lr': scores_task2_lr,
        'scores_task2_acc': scores_task2_acc,
        'times_train': times,
        'times_task2': times_task2
    }
    if scores_task1_conf is not None:
        save_dict['scores_task1_conf'] = scores_task1_conf

    np.savez(os.path.join(results_dir_base, f'{subject}_cross_task_decoding.npz'), **save_dict)
    print(f"[INFO] 结果已保存到 {results_dir_base}")
    print("[INFO] 所有处理完成！")