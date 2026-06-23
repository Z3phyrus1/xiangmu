# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:49:07 2026

@author: WANGLIANGFU
"""

import os
from glob import glob
import numpy as np
import pandas as pd
import re
from shutil import copyfile
import time

import mne
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, balanced_accuracy_score, roc_auc_score

copyfile('../common_settings.py', 'common_settings.py')
copyfile('../utils.py', 'utils.py')
import common_settings as CS
import utils as utils


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cv_eval_binary(X, y, n_splits=5, random_state=42):
    """
    返回 (auc_mean, balacc_mean)
    """
    y = np.asarray(y).astype(int)
    X = np.asarray(X, dtype=float)

    if len(y) < n_splits or len(np.unique(y)) < 2:
        return np.nan, np.nan

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver='liblinear',
            class_weight='balanced',
            random_state=random_state
        )
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "auc": "roc_auc",
        "balacc": make_scorer(balanced_accuracy_score)
    }

    res = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=1, return_train_score=False)
    return float(np.mean(res["test_auc"])), float(np.mean(res["test_balacc"]))


def majority_baseline_cv(y, n_splits=5, random_state=42):
    """
    每个fold用训练集多数类去预测测试集
    返回 (auc_mean, balacc_mean)
    """
    y = np.asarray(y).astype(int)
    if len(y) < n_splits or len(np.unique(y)) < 2:
        return np.nan, np.nan

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs, bals = [], []

    for tr, te in cv.split(np.zeros(len(y)), y):
        y_tr, y_te = y[tr], y[te]
        maj = int(np.bincount(y_tr).argmax())

        y_pred = np.full_like(y_te, maj)
        y_score = np.full_like(y_te, float(maj), dtype=float)

        if len(np.unique(y_te)) > 1:
            aucs.append(roc_auc_score(y_te, y_score))
        bals.append(balanced_accuracy_score(y_te, y_pred))

    return float(np.mean(aucs)), float(np.mean(bals))


if __name__ == "__main__":
    t0 = time.time()

    # =========================
    # 参数
    # =========================
    subject = 'sub18'
    sub_id_int = int(re.search(r'\d+', subject).group())

    # 你指定的按键前窗口
    tmin_feat = -0.42
    tmax_feat = -0.01

    n_splits = 5
    random_state = 42

    working_dir = os.path.join('../../..', 'data', 'clean_EEG_dual', subject)
    behavior_path = os.path.join('../../..', 'data', 'behavior', 'data_wide_wmPred.txt')

    results_dir = os.path.join('../..', 'results', 'behavior_eeg_joint', subject)
    figures_dir = os.path.join('../..', 'figures', 'behavior_eeg_joint', subject)
    for d in [results_dir, figures_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    log(f"被试: {subject}")
    log(f"EEG目录: {working_dir}")
    log(f"行为文件: {behavior_path}")
    log(f"特征时间窗: {tmin_feat} ~ {tmax_feat} s")

    # =========================
    # 找文件
    # =========================
    task1_response_files = sorted(glob(os.path.join(working_dir, 'clean_epochs_response1_ICA_*.fif')))
    log(f"找到 {len(task1_response_files)} 个 Response1 文件")
    if len(task1_response_files) == 0:
        raise FileNotFoundError("没有找到 clean_epochs_response1_ICA_*.fif")

    # =========================
    # 读行为数据
    # =========================
    df_pred = pd.read_csv(behavior_path, sep=';')
    if 'id' not in df_pred.columns:
        raise ValueError("behavior文件缺少 id 列")

    # 兼容 id 列可能是 int 或像 '18_xxx'
    try:
        df_sub = df_pred[df_pred['id'].astype(str).str.split('_').str[0].astype(int) == sub_id_int].copy()
    except Exception:
        df_sub = df_pred[df_pred['id'] == sub_id_int].copy()

    log(f"该被试行为数据行数: {len(df_sub)}")
    if len(df_sub) == 0:
        raise RuntimeError("该被试没有行为数据")

    if 'id_sess' not in df_sub.columns:
        raise ValueError("behavior文件缺少 id_sess 列，无法按你原逻辑做文件级对齐")

    all_id_sess = df_sub['id_sess'].unique()
    if len(all_id_sess) >= 4:
        id_sess_to_use = all_id_sess[-4:]
    else:
        id_sess_to_use = all_id_sess

    log(f"将使用 id_sess: {id_sess_to_use}")

    # =========================
    # 构建 trial-level 联合表
    # =========================
    rows_joint = []

    for i, filename in enumerate(task1_response_files):
        base = os.path.basename(filename)
        log(f"\n处理文件[{i+1}/{len(task1_response_files)}]: {base}")

        epochs = mne.read_epochs(filename, verbose=False)
        # 注意：这里只裁剪特征时间窗，不触碰标签
        ep_feat = epochs.copy().crop(tmin=tmin_feat, tmax=tmax_feat)

        X = ep_feat.get_data(copy=False)  # (n_trial_kept, n_ch, n_time)
        if X.shape[0] == 0:
            log("  -> 无trial，跳过")
            continue

        # 简单稳健的 trial-level EEG 特征
        eeg_mean = X.mean(axis=(1, 2))
        eeg_std = X.std(axis=(1, 2))
        eeg_rms = np.sqrt((X ** 2).mean(axis=(1, 2)))

        # 原始trial索引（drop后保留的映射）
        selection = epochs.selection  # len == n_trial_kept

        # 文件标识
        m = re.search(r'sub\d+_(.+?)-epo\.fif', base)
        if not m:
            log("  -> 无法提取 file_id，跳过")
            continue
        file_id = m.group(1)

        # drop_log
        log_name = f"{subject}_{file_id}_response1_drop_log.csv"
        log_path = os.path.join(working_dir, log_name)
        if not os.path.exists(log_path):
            log(f"  -> 缺少 drop_log: {log_name}，跳过")
            continue

        df_log = pd.read_csv(log_path)
        n_orig = len(df_log)

        # 关联 id_sess（沿用你原逻辑）
        if i >= len(id_sess_to_use):
            log("  -> 文件索引超出 id_sess_to_use，跳过")
            continue
        target_id_sess = id_sess_to_use[i]

        beh = df_sub[df_sub['id_sess'] == target_id_sess].copy().reset_index(drop=True)
        if len(beh) != n_orig:
            log(f"  -> 行为条数与原始trial不匹配: beh={len(beh)}, orig={n_orig}，跳过")
            continue

        # 给行为加原始trial索引
        beh['trial_index_orig'] = np.arange(n_orig, dtype=int)

        # 取保留trial对应行为行
        beh_keep = beh.iloc[selection].copy().reset_index(drop=True)

        # 必需列
        need_cols = ['abs1', 'abs2', 'r1', 'r2']
        miss = [c for c in need_cols if c not in beh_keep.columns]
        if len(miss) > 0:
            log(f"  -> 行为缺列 {miss}，跳过")
            continue

        # 清理标签
        for c in ['r1', 'r2']:
            beh_keep[c] = pd.to_numeric(beh_keep[c], errors='coerce')
        beh_keep = beh_keep.dropna(subset=['abs1', 'abs2', 'r1', 'r2']).copy()

        for c in ['r1', 'r2']:
            beh_keep[c] = beh_keep[c].astype(int)
            uniq = set(beh_keep[c].unique().tolist())
            if uniq.issubset({1, 2}):
                beh_keep[c] = (beh_keep[c] - 1).astype(int)

        # 长度再对齐一次（防NaN过滤导致长度变动）
        idx_valid = beh_keep.index.values
        # 因为 beh_keep reset_index 过，这里改用再次对齐最安全：
        n_min = min(len(beh_keep), len(eeg_mean))
        beh_keep = beh_keep.iloc[:n_min].copy()
        eeg_mean_use = eeg_mean[:n_min]
        eeg_std_use = eeg_std[:n_min]
        eeg_rms_use = eeg_rms[:n_min]

        beh_keep['subject'] = subject
        beh_keep['file'] = base
        beh_keep['file_id'] = file_id
        beh_keep['id_sess_used'] = target_id_sess
        beh_keep['eeg_mean_pre420_010'] = eeg_mean_use
        beh_keep['eeg_std_pre420_010'] = eeg_std_use
        beh_keep['eeg_rms_pre420_010'] = eeg_rms_use

        rows_joint.append(
            beh_keep[['subject', 'file', 'file_id', 'id_sess_used',
                      'abs1', 'abs2', 'r1', 'r2',
                      'eeg_mean_pre420_010', 'eeg_std_pre420_010', 'eeg_rms_pre420_010']]
        )
        log(f"  -> 成功对齐并写入 {n_min} 条trial")

    if len(rows_joint) == 0:
        raise RuntimeError("没有成功构建任何 trial-level 联合数据")

    df_joint = pd.concat(rows_joint, axis=0, ignore_index=True)
    log(f"\n联合表总行数: {len(df_joint)}")
    log(f"r1分布: {np.unique(df_joint['r1'], return_counts=True)}")
    log(f"r2分布: {np.unique(df_joint['r2'], return_counts=True)}")

    # 保存联合表
    joint_csv = os.path.join(results_dir, f'{subject}_trial_level_behavior_eeg_pre420_010.csv')
    df_joint.to_csv(joint_csv, index=False)
    log(f"已保存联合表: {joint_csv}")

    # =========================
    # 建模比较
    # =========================
    eeg_features = ['eeg_mean_pre420_010', 'eeg_std_pre420_010', 'eeg_rms_pre420_010']
    tasks = [
        ('abs1', 'r1'),
        ('abs1', 'r2'),
        ('abs2', 'r1'),
        ('abs2', 'r2'),
    ]

    result_rows = []

    for abs_col, y_col in tasks:
        for eeg_col in eeg_features:
            d = df_joint[[abs_col, eeg_col, y_col]].dropna().copy()
            y = d[y_col].values.astype(int)

            # baseline
            auc_base, bal_base = majority_baseline_cv(y, n_splits=n_splits, random_state=random_state)

            # EEG -> r
            X_eeg = d[[eeg_col]].values
            auc_eeg, bal_eeg = cv_eval_binary(X_eeg, y, n_splits=n_splits, random_state=random_state)

            # abs -> r
            X_abs = d[[abs_col]].values
            auc_abs, bal_abs = cv_eval_binary(X_abs, y, n_splits=n_splits, random_state=random_state)

            # abs + EEG -> r
            X_both = d[[abs_col, eeg_col]].values
            auc_both, bal_both = cv_eval_binary(X_both, y, n_splits=n_splits, random_state=random_state)

            result_rows.append({
                'abs_feature': abs_col,
                'eeg_feature': eeg_col,
                'target': y_col,
                'n': len(d),

                'AUC_baseline': auc_base,
                'AUC_EEG_only': auc_eeg,
                'AUC_abs_only': auc_abs,
                'AUC_abs_plus_EEG': auc_both,
                'AUC_gain_over_abs': (auc_both - auc_abs) if (not np.isnan(auc_both) and not np.isnan(auc_abs)) else np.nan,

                'BalAcc_baseline': bal_base,
                'BalAcc_EEG_only': bal_eeg,
                'BalAcc_abs_only': bal_abs,
                'BalAcc_abs_plus_EEG': bal_both,
                'BalAcc_gain_over_abs': (bal_both - bal_abs) if (not np.isnan(bal_both) and not np.isnan(bal_abs)) else np.nan,
            })

            log(f"[{abs_col}->{y_col} | {eeg_col}] "
                f"AUC: base={auc_base:.3f}, eeg={auc_eeg:.3f}, abs={auc_abs:.3f}, both={auc_both:.3f}")

    df_res = pd.DataFrame(result_rows)

    # 打印结果
    print("\n" + "="*80)
    print("联合建模比较结果（按键前 -0.42~-0.01s）")
    print("="*80)
    print(df_res.to_string(index=False))

    # 保存结果
    res_csv = os.path.join(results_dir, f'{subject}_model_compare_pre420_010.csv')
    df_res.to_csv(res_csv, index=False)
    log(f"已保存比较结果: {res_csv}")

    log(f"全部完成，用时 {time.time()-t0:.1f}s")