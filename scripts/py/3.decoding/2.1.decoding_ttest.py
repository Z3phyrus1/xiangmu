# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 2026

基于 2.0.decoding_id.py 的 id 级别对齐逻辑：
1. 读取单个被试全部 clean_EEG 文件
2. 从 behavior_for_behavior/data_wide_wmPred.txt 中按 id + id_sess 对齐行为
3. 将 abs1 <= 1 标记为 low，abs1 > 1 标记为 high
4. 对对应试次的 EEG 做时间滑窗解码
5. 对每个时间点的 AUC 相对 chance=0.5 做单样本 t 检验
"""

import os
import re
from glob import glob
from shutil import copyfile

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.decoding import SlidingEstimator, cross_val_multiscore
from scipy import stats
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

copyfile('../common_settings.py', 'common_settings.py')
copyfile('../utils.py', 'utils.py')
import common_settings as CS


ABS1_THRESHOLD = 1.0
CHANCE_LEVEL = 0.5


def read_concat(files, baseline, crop_tmin, crop_tmax, resample_hz=100):
    epochs_list = [mne.read_epochs(f, verbose=False) for f in files]
    for epochs in epochs_list:
        epochs.apply_baseline(baseline)
    epochs = mne.concatenate_epochs(epochs_list)
    epochs.crop(crop_tmin, crop_tmax).resample(resample_hz)
    return epochs


def get_behavior_blocks(df_pred, sub_id_int, n_files):
    df_sub = df_pred[df_pred['id'].astype(int) == sub_id_int].copy()
    if df_sub.empty:
        raise RuntimeError(f'行为文件中未找到 id={sub_id_int} 的数据。')

    all_id_sess = df_sub['id_sess'].astype(str).unique()
    id_sess_to_use = all_id_sess[-n_files:] if len(all_id_sess) >= n_files else all_id_sess
    beh_blocks = [df_sub[df_sub['id_sess'].astype(str) == str(sid)].copy() for sid in id_sess_to_use]
    beh_aligned = pd.concat(beh_blocks, axis=0, ignore_index=True)
    return beh_aligned, id_sess_to_use


def build_abs1_labels(beh_aligned, threshold=ABS1_THRESHOLD):
    abs1 = pd.to_numeric(beh_aligned['abs1'], errors='coerce')
    valid_mask = abs1.notna().to_numpy()
    labels = (abs1[valid_mask] > threshold).astype(int).to_numpy()
    return labels, valid_mask


def run_sliding_ttest(X, y, times, n_splits=100, test_size=0.2, random_state=12345):
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
            max_iter=3000,
            n_jobs=1,
            verbose=0,
        ),
    )

    slider = SlidingEstimator(clf, scoring='roc_auc', n_jobs=1, verbose=False)
    cv = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    )

    scores = cross_val_multiscore(slider, X, y, cv=cv, n_jobs=1, verbose=1)
    scores_mean = scores.mean(axis=0)
    scores_sem = scores.std(axis=0, ddof=1) / np.sqrt(scores.shape[0])

    t_vals, p_vals = stats.ttest_1samp(scores, popmean=CHANCE_LEVEL, axis=0, nan_policy='omit')
    sig_mask = np.isfinite(p_vals) & (p_vals < 0.05)

    return {
        'scores': scores,
        'scores_mean': scores_mean,
        'scores_sem': scores_sem,
        't_vals': t_vals,
        'p_vals': p_vals,
        'sig_mask': sig_mask,
        'times': times,
    }


def save_stats_table(times, t_vals, p_vals, sig_mask, save_path):
    df = pd.DataFrame({
        'time': times,
        't_value': t_vals,
        'p_value': p_vals,
        'significant_p_lt_0_05': sig_mask.astype(int),
    })
    df.to_csv(save_path, index=False, encoding='utf-8-sig')


def plot_ttest_result(result, title, save_path):
    times = result['times']
    scores_mean = result['scores_mean']
    scores_sem = result['scores_sem']
    sig_mask = result['sig_mask']

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, scores_mean, color='C0', lw=2, label='AUC')
    ax.fill_between(
        times,
        scores_mean - scores_sem,
        scores_mean + scores_sem,
        color='C0',
        alpha=0.25,
        linewidth=0,
    )
    ax.axhline(CHANCE_LEVEL, color='k', linestyle='--', lw=1, label='Chance')
    ax.axvline(0, color='k', linestyle='-', lw=1, alpha=0.8)

    if np.any(sig_mask):
        ax.scatter(
            times[sig_mask],
            scores_mean[sig_mask],
            color='crimson',
            s=18,
            label='p < 0.05',
            zorder=3,
        )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('AUC')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    subject = 'sub18'
    n_splits = 100
    test_size = 0.2
    random_state = 12345

    sub_id_int = int(re.search(r'\d+', subject).group())
    working_dir = os.path.join('../../..', 'data', 'clean_EEG', subject)
    behavior_path = os.path.join('../../..', 'data', 'behavior_for_behavior', 'data_wide_wmPred.txt')

    print(f'===== START {subject} =====')
    print(f'EEG dir: {working_dir}')
    print(f'Behavior file: {behavior_path}')

    task1_stim_files = sorted(glob(os.path.join(working_dir, 'clean_epochs_task1_ICA_*.fif')))
    task1_resp_files = sorted(glob(os.path.join(working_dir, 'clean_epochs_response1_ICA_*.fif')))
    task2_stim_files = sorted(glob(os.path.join(working_dir, 'clean_epochs_task2_ICA_*.fif')))

    if len(task1_stim_files) == 0 or len(task1_resp_files) == 0 or len(task2_stim_files) == 0:
        raise FileNotFoundError('未找到完整的 task1/task2 EEG 文件，请检查被试目录。')

    df_pred = pd.read_csv(behavior_path, sep=';')
    beh_aligned, id_sess_to_use = get_behavior_blocks(df_pred, sub_id_int, len(task1_resp_files))

    print(f'Using id_sess blocks: {list(id_sess_to_use)}')
    print(f'Behavior rows before alignment: {len(beh_aligned)}')

    epochs_t1_stim = mne.concatenate_epochs([mne.read_epochs(f, verbose=False) for f in task1_stim_files])
    epochs_t1_resp = read_concat(task1_resp_files, CS.baseline_response, -0.5, CS.tmax_response)
    epochs_t2_stim = read_concat(task2_stim_files, CS.baseline_task2, CS.tmin_task2, CS.tmax_task2)

    n_min = min(len(epochs_t1_stim), len(epochs_t1_resp), len(epochs_t2_stim), len(beh_aligned))
    epochs_t1_stim = epochs_t1_stim[:n_min]
    epochs_t1_resp = epochs_t1_resp[:n_min]
    epochs_t2_stim = epochs_t2_stim[:n_min]
    beh_aligned = beh_aligned.iloc[:n_min].reset_index(drop=True)

    y_group, valid_mask = build_abs1_labels(beh_aligned, threshold=ABS1_THRESHOLD)
    if valid_mask.sum() < len(valid_mask):
        epochs_t1_stim = epochs_t1_stim[valid_mask]
        epochs_t1_resp = epochs_t1_resp[valid_mask]
        epochs_t2_stim = epochs_t2_stim[valid_mask]
        beh_aligned = beh_aligned.loc[valid_mask].reset_index(drop=True)

    class_counts = np.unique(y_group, return_counts=True)
    print(f'Aligned trials: {len(y_group)}')
    print(f'Low/High counts (0=low, 1=high): {class_counts}')

    if len(np.unique(y_group)) < 2:
        raise RuntimeError('abs1 分组后只有一个类别，无法继续做解码与 t 检验。')

    X_t1_resp = epochs_t1_resp.get_data(copy=False)
    X_t2_stim = epochs_t2_stim.get_data(copy=False)
    times_t1_resp = epochs_t1_resp.times
    times_t2_stim = epochs_t2_stim.times

    results_dir = os.path.join('../../..', 'results', 'decoding_ttest', subject)
    figures_dir = os.path.join('../../..', 'figures', 'decoding_ttest', subject)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print('Running Task1 Response decoding + t-test...')
    result_t1_resp = run_sliding_ttest(
        X_t1_resp,
        y_group,
        times_t1_resp,
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    )

    print('Running Task2 Stimulus decoding + t-test...')
    result_t2_stim = run_sliding_ttest(
        X_t2_stim,
        y_group,
        times_t2_stim,
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    )

    t1_csv = os.path.join(results_dir, f'{subject}_task1_response_abs1_ttest.csv')
    t2_csv = os.path.join(results_dir, f'{subject}_task2_stim_abs1_ttest.csv')
    save_stats_table(times_t1_resp, result_t1_resp['t_vals'], result_t1_resp['p_vals'], result_t1_resp['sig_mask'], t1_csv)
    save_stats_table(times_t2_stim, result_t2_stim['t_vals'], result_t2_stim['p_vals'], result_t2_stim['sig_mask'], t2_csv)

    plot_ttest_result(
        result_t1_resp,
        title=f'Task1 Response: abs1 low/high decoding ({subject})',
        save_path=os.path.join(figures_dir, f'{subject}_task1_response_abs1_ttest.png'),
    )
    plot_ttest_result(
        result_t2_stim,
        title=f'Task2 Stimulus: abs1 low/high decoding ({subject})',
        save_path=os.path.join(figures_dir, f'{subject}_task2_stim_abs1_ttest.png'),
    )

    np.savez(
        os.path.join(results_dir, f'{subject}_abs1_decoding_ttest_results.npz'),
        scores_t1_resp=result_t1_resp['scores'],
        scores_t2_stim=result_t2_stim['scores'],
        scores_t1_resp_mean=result_t1_resp['scores_mean'],
        scores_t2_stim_mean=result_t2_stim['scores_mean'],
        t_vals_t1_resp=result_t1_resp['t_vals'],
        p_vals_t1_resp=result_t1_resp['p_vals'],
        t_vals_t2_stim=result_t2_stim['t_vals'],
        p_vals_t2_stim=result_t2_stim['p_vals'],
        sig_mask_t1_resp=result_t1_resp['sig_mask'],
        sig_mask_t2_stim=result_t2_stim['sig_mask'],
        times_t1_resp=times_t1_resp,
        times_t2_stim=times_t2_stim,
        y_group=y_group,
        abs1_threshold=ABS1_THRESHOLD,
        used_id_sess=np.array(id_sess_to_use, dtype=str),
        chance_level=CHANCE_LEVEL,
    )

    print(f'Task1 stats saved to: {t1_csv}')
    print(f'Task2 stats saved to: {t2_csv}')
    print('DONE')
