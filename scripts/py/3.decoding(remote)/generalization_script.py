# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:25:05 2026

@author: WANGLIANGFU
"""

import os
import re
import argparse
from glob import glob
from io import StringIO
import numpy as np
import pandas as pd
import mne

from mne.decoding import GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import common_settings as CS
import utils


def load_fit_par(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    cleaned = ''.join([ln.replace('"', '') for ln in lines])
    df_fit = pd.read_table(
        StringIO(cleaned),
        delim_whitespace=True,
        engine='python',
        dtype=str,
        keep_default_na=False
    )
    df_fit = df_fit.applymap(lambda x: x.strip())
    df_fit.columns = df_fit.columns.str.strip()
    return df_fit


def get_w_sub2(df_fit, sub_id_int, trial=None):
    id_col = df_fit['id'].astype(str).str.strip()
    model_col = df_fit['model'].astype(str).str.strip()

    row = df_fit[(id_col == str(sub_id_int)) & (model_col == 'sub2')]
    if row.empty:
        return None
    return float(row['w'].iloc[0])


def two_class(y):
    return len(np.unique(y)) == 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--trial", type=str, required=True)
    args = parser.parse_args()

    subject = args.subject
    trial = args.trial
    sub_id_int = int(re.search(r'\d+', subject).group())

    # 路径（与 bash_decoding_jobs 运行方式兼容）
    eeg_dir = os.path.join('../../..', 'data', 'clean_EEG_dual', subject)
    fit_par_path = os.path.join('../../..', 'data', 'behavior', 'fit_par.txt')
    behavior_path = os.path.join('../../..', 'data', 'behavior', 'data_wide_wmPred.txt')

    results_dir = os.path.join('../../..', 'results', 'decoding', subject, trial)
    figures_dir = os.path.join('../../..', 'figures', 'decoding', subject, trial)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Subject: {subject}, Trial: {trial}")

    # ---- 1) W threshold ----
    df_fit = load_fit_par(fit_par_path)
    w_val = get_w_sub2(df_fit, sub_id_int, trial)
    if w_val is None:
        print(f"[SKIP] no w(sub2): id={sub_id_int}")
        return
    # ---- 2) 文件 ----
    task1_stim_files = sorted(glob(os.path.join(eeg_dir, f'clean_epochs_task1_ICA_*{trial}_sess*.fif')))
    task1_resp_files = sorted(glob(os.path.join(eeg_dir, f'clean_epochs_response1_ICA_*{trial}_sess*.fif')))
    task2_stim_files = sorted(glob(os.path.join(eeg_dir, f'clean_epochs_task2_ICA_*{trial}_sess*.fif')))

    if len(task1_stim_files) == 0:
        raise FileNotFoundError("No task1 stim files")
    if len(task1_resp_files) == 0:
        raise FileNotFoundError("No task1 response files")
    if len(task2_stim_files) == 0:
        raise FileNotFoundError("No task2 stim files")

    # ---- 3) 行为 ----
    df_pred = pd.read_csv(behavior_path, sep=';')
    df_sub = df_pred[df_pred['id'] == sub_id_int].copy()
    beh = df_sub[df_sub['seq'].astype(str) == str(trial)].reset_index(drop=True)
    if len(beh) == 0:
        print(f"[SKIP] Behavior rows = 0 for {subject} trial {trial}")
        return

    # ---- 4) 读EEG并预处理（统一 100Hz，避免后续时间轴混乱）----
    ep_t1_stim = mne.concatenate_epochs([mne.read_epochs(f, verbose=False) for f in task1_stim_files])

    ep_t1_r_list = [mne.read_epochs(f, verbose=False) for f in task1_resp_files]
    for ep in ep_t1_r_list:
        ep.apply_baseline(CS.baseline_response)
    ep_t1_r = mne.concatenate_epochs(ep_t1_r_list).crop(-0.5, CS.tmax_response).resample(100)

    ep_t2_s_list = [mne.read_epochs(f, verbose=False) for f in task2_stim_files]
    for ep in ep_t2_s_list:
        ep.apply_baseline(CS.baseline_task2)
    ep_t2_s = mne.concatenate_epochs(ep_t2_s_list).crop(CS.tmin_task2, CS.tmax_task2).resample(100)

    X_train = ep_t1_r.get_data(copy=False)
    X_test = ep_t2_s.get_data(copy=False)
    times_train = ep_t1_r.times
    times_test = ep_t2_s.times

    n_min = min(len(ep_t1_stim), len(ep_t1_r), len(ep_t2_s), len(beh))
    if n_min < 20:
        raise ValueError(f"Aligned samples too small: {n_min}")

    X_train = X_train[:n_min]
    X_test = X_test[:n_min]
    ep_t1_stim = ep_t1_stim[:n_min]
    ep_t1_r = ep_t1_r[:n_min]
    beh = beh.iloc[:n_min].reset_index(drop=True)

    print(f"Aligned samples: {n_min}")

    # ---- 5) 标签 ----
    y_train_acc = utils.compute_accuracy_from_epochs(ep_t1_stim, ep_t1_r)
    y_test_acc = beh['acc2'].astype(int).values

    y_train_conf = (beh['abs1'] > w_val).astype(int).values
    y_test_conf = (beh['abs2'] > w_val).astype(int).values

    print("acc train/test:", np.unique(y_train_acc, return_counts=True), np.unique(y_test_acc, return_counts=True))
    print("conf train/test:", np.unique(y_train_conf, return_counts=True), np.unique(y_test_conf, return_counts=True))

    clf = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', class_weight='balanced'))

    # ---- 6) Generalization: ACC ----
    if two_class(y_train_acc) and two_class(y_test_acc):
        print("Running Gen: ACC (T1Resp -> T2Stim)")
        gen_acc = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=1, verbose=False)
        gen_acc.fit(X_train, y_train_acc)
        scores_acc = gen_acc.score(X_test, y_test_acc)

        utils.plot_temporal_generalization(
            scores=scores_acc,
            times_train=times_train,
            times_test=times_test,
            title=f'Acc Gen ({subject} Tr{trial})',
            save_path=os.path.join(figures_dir, f'{subject}_{trial}_gen_acc_cross_task.png'),
            vmin=0.4, vmax=0.6
        )
    else:
        print("[SKIP] ACC not two-class in train/test")
        scores_acc = None

    # ---- 7) Generalization: CONF ----
    if two_class(y_train_conf) and two_class(y_test_conf):
        print("Running Gen: CONF (T1Resp -> T2Stim)")
        gen_conf = GeneralizingEstimator(clf, scoring='roc_auc', n_jobs=1, verbose=False)
        gen_conf.fit(X_train, y_train_conf)
        scores_conf = gen_conf.score(X_test, y_test_conf)

        utils.plot_temporal_generalization(
            scores=scores_conf,
            times_train=times_train,
            times_test=times_test,
            title=f'Conf Gen ({subject} Tr{trial})',
            save_path=os.path.join(figures_dir, f'{subject}_{trial}_gen_conf_cross_task.png'),
            vmin=0.4, vmax=0.6
        )
    else:
        print("[SKIP] CONF not two-class in train/test")
        scores_conf = None

    # ---- 8) 保存 ----
    np.savez(
        os.path.join(results_dir, f'{subject}_{trial}_generalization_results.npz'),
        scores_gen_acc=scores_acc,
        scores_gen_conf=scores_conf,
        times_train=times_train,
        times_test=times_test
    )
    print("Done.")


if __name__ == "__main__":
    main()