# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:05:19 2026

@author: WANGLIANGFU

批量版：遍历 clean_EEG 文件夹内所有被试，运行 fixed-alignment decoding + permutation test

核心逻辑：
1. clean_eeg_root 下每个 subXX 文件夹作为一个被试
2. 行为表中读取该被试所有 id_sess
3. 根据 id_sess 末尾数字自动映射到 EEG sessX
   例如：
       54fl03 -> sess3
       54fl04 -> sess4
4. 只读取这些 session 对应的 EEG epoch 文件
5. 严格要求：
       Task1 Stim epochs == Task1 Response epochs == Task2 Stim epochs == Behavior rows
   否则跳过该被试
6. 对 Task1 Response 和 Task2 Stimulus 做 abs1 low/high decoding
7. 用标签置换 permutation test 计算每个时间点的 p 值
"""

import os
import re
import time
import traceback
from glob import glob
from shutil import copyfile

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# =========================
# 外部设置文件
# =========================
copyfile('../common_settings.py', 'common_settings.py')
copyfile('../utils.py', 'utils.py')
import common_settings as CS


# =========================
# 全局参数
# =========================
ABS1_THRESHOLD = 1.0
CHANCE_LEVEL = 0.5


# =========================
# 文件与行为 session 工具
# =========================
def get_subject_dirs(clean_eeg_root):
    """
    返回 clean_EEG 文件夹下所有 subXX 被试文件夹。
    """
    subject_dirs = sorted(glob(os.path.join(clean_eeg_root, 'sub*')))
    subject_dirs = [d for d in subject_dirs if os.path.isdir(d)]

    def sub_number(path):
        name = os.path.basename(path)
        m = re.search(r'\d+', name)
        return int(m.group()) if m else 999999

    subject_dirs = sorted(subject_dirs, key=sub_number)
    return subject_dirs


def get_behavior_sessions_for_subject(df_pred, sub_id_int):
    """
    读取某个被试在行为表中的全部 id_sess。
    """
    df_sub = df_pred[df_pred['id'].astype(int) == sub_id_int].copy()

    if df_sub.empty:
        raise RuntimeError(f'行为文件中未找到 id={sub_id_int} 的数据。')

    id_sess = df_sub['id_sess'].astype(str).unique().tolist()

    print('\nAvailable behavior id_sess:')
    for sid in id_sess:
        n = (df_sub['id_sess'].astype(str) == sid).sum()
        print(f'  {sid}: {n} rows')

    return id_sess


def behavior_sessions_to_eeg_sessions(id_sess_list):
    """
    根据行为 id_sess 自动推断 EEG sess。

    示例：
        54fl03 -> sess3
        54fl04 -> sess4
        18cl03 -> sess3
        18cl04 -> sess4
    """
    eeg_sessions = []

    for sid in id_sess_list:
        m = re.search(r'(\d+)$', str(sid))
        if m is None:
            raise RuntimeError(f'无法从 id_sess={sid} 提取 session 编号。')

        sess_num = int(m.group(1))
        eeg_sessions.append(f'sess{sess_num}')

    return eeg_sessions


def filter_files_by_sessions(files, keep_sessions, name):
    """
    只保留文件名中包含指定 sess 的 EEG 文件。
    """
    files_keep = [
        f for f in files
        if any(sess in os.path.basename(f) for sess in keep_sessions)
    ]
    files_keep = sorted(files_keep)

    print(f'\n{name}: keeping sessions {keep_sessions}')
    print(f'  before filtering: {len(files)} files')
    print(f'  after filtering:  {len(files_keep)} files')

    for f in files_keep:
        print('   ', os.path.basename(f))

    if len(files_keep) == 0:
        raise RuntimeError(f'{name}: 过滤后没有文件，请检查 keep_sessions 或文件名。')

    return files_keep


def get_behavior_blocks_by_sessions(df_pred, sub_id_int, id_sess_to_use):
    """
    明确按给定 id_sess 读取行为 block。
    """
    df_sub = df_pred[df_pred['id'].astype(int) == sub_id_int].copy()

    if df_sub.empty:
        raise RuntimeError(f'行为文件中未找到 id={sub_id_int} 的数据。')

    available_sessions = df_sub['id_sess'].astype(str).unique().tolist()

    missing = [sid for sid in id_sess_to_use if sid not in available_sessions]
    if len(missing) > 0:
        raise RuntimeError(f'行为文件中找不到这些 id_sess: {missing}')

    beh_blocks = []

    for sid in id_sess_to_use:
        block = df_sub[df_sub['id_sess'].astype(str) == str(sid)].copy()
        beh_blocks.append(block)

    beh_aligned = pd.concat(beh_blocks, axis=0, ignore_index=True)
    return beh_aligned


def read_concat(files, baseline, crop_tmin, crop_tmax, resample_hz=100):
    epochs_list = [mne.read_epochs(f, verbose=False) for f in files]

    for epochs in epochs_list:
        epochs.apply_baseline(baseline)

    epochs = mne.concatenate_epochs(epochs_list)
    epochs.crop(crop_tmin, crop_tmax).resample(resample_hz)
    return epochs


def build_abs1_labels(beh_aligned, threshold=ABS1_THRESHOLD):
    abs1 = pd.to_numeric(beh_aligned['abs1'], errors='coerce')
    valid_mask = abs1.notna().to_numpy()
    labels = (abs1[valid_mask] > threshold).astype(int).to_numpy()
    return labels, valid_mask


def get_selection_array(epochs):
    if hasattr(epochs, 'selection') and epochs.selection is not None:
        return np.asarray(epochs.selection)
    return None


def save_alignment_report(
    save_path,
    subject,
    keep_sessions,
    behavior_sessions,
    task1_stim_files,
    task1_resp_files,
    task2_stim_files,
    epochs_t1_stim,
    epochs_t1_resp,
    epochs_t2_stim,
    beh_aligned,
):
    lines = []

    def log(x=''):
        x = str(x)
        print(x)
        lines.append(x)

    log('\n================ ALIGNMENT REPORT ================')
    log(f'subject = {subject}')
    log(f'keep EEG sessions = {keep_sessions}')
    log(f'behavior sessions = {behavior_sessions}')

    log('\n[EEG files used] Task1 Stim:')
    for f in task1_stim_files:
        log(f'  {os.path.basename(f)}')

    log('\n[EEG files used] Task1 Response:')
    for f in task1_resp_files:
        log(f'  {os.path.basename(f)}')

    log('\n[EEG files used] Task2 Stimulus:')
    for f in task2_stim_files:
        log(f'  {os.path.basename(f)}')

    log('\n[Counts after filtering and reading]')
    log(f'  Task1 Stim epochs:     {len(epochs_t1_stim)}')
    log(f'  Task1 Response epochs: {len(epochs_t1_resp)}')
    log(f'  Task2 Stim epochs:     {len(epochs_t2_stim)}')
    log(f'  Behavior rows:         {len(beh_aligned)}')

    s_resp = get_selection_array(epochs_t1_resp)
    s_t2 = get_selection_array(epochs_t2_stim)

    log('\n[Selection check: Task1 Response vs Task2 Stimulus]')
    if s_resp is not None and s_t2 is not None:
        same = np.array_equal(s_resp, s_t2)
        log(f'  selections identical? {same}')
        log(f'  first 20 Task1 Resp selection = {s_resp[:20]}')
        log(f'  first 20 Task2 Stim selection = {s_t2[:20]}')
        log(f'  last 20 Task1 Resp selection = {s_resp[-20:]}')
        log(f'  last 20 Task2 Stim selection = {s_t2[-20:]}')
    else:
        log('  selection not available.')

    lengths = [
        len(epochs_t1_stim),
        len(epochs_t1_resp),
        len(epochs_t2_stim),
        len(beh_aligned),
    ]

    if len(set(lengths)) == 1:
        log('\nFINAL CHECK: OK. EEG and behavior lengths are equal.')
    else:
        log('\nFINAL CHECK: ERROR. EEG and behavior lengths are NOT equal.')

    log('================ END ALIGNMENT REPORT ================\n')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# =========================
# 解码函数
# =========================
def build_decoder(random_state=12345, C=1.0):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver='liblinear',
            class_weight='balanced',
            random_state=random_state,
            penalty='l1',
            C=C,
            max_iter=3000,
        ),
    )

    slider = SlidingEstimator(
        clf,
        scoring='roc_auc',
        n_jobs=1,
        verbose=False,
    )

    return slider


def run_decoding_scores(X, y, n_splits=10, test_size=0.2, random_state=12345, C=1.0):
    slider = build_decoder(random_state=random_state, C=C)

    cv = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    )

    scores = cross_val_multiscore(
        slider,
        X,
        y,
        cv=cv,
        n_jobs=1,
        verbose=False,
    )

    return scores


# =========================
# 显著片段工具
# =========================
def find_contiguous_segments(mask, times):
    segments = []

    if mask.sum() == 0:
        return segments

    idx = np.where(mask)[0]
    start = idx[0]
    prev = idx[0]

    for cur in idx[1:]:
        if cur == prev + 1:
            prev = cur
        else:
            segments.append((start, prev, times[start], times[prev]))
            start = cur
            prev = cur

    segments.append((start, prev, times[start], times[prev]))
    return segments


def filter_segments_by_min_duration(mask, times, min_duration_s=0.02):
    filtered = np.zeros_like(mask, dtype=bool)
    segments = find_contiguous_segments(mask, times)

    dt = np.median(np.diff(times)) if len(times) > 1 else 0.0

    for start_idx, end_idx, start_t, end_t in segments:
        duration = times[end_idx] - times[start_idx] + dt

        if duration >= min_duration_s:
            filtered[start_idx:end_idx + 1] = True

    return filtered


# =========================
# permutation test
# =========================
def run_permutation_test(
    X,
    y,
    times,
    n_splits=10,
    test_size=0.2,
    random_state=12345,
    n_permutations=50,
    p_threshold=0.05,
    min_sig_duration_s=0.02,
    C=1.0,
):
    rng = np.random.RandomState(random_state)

    t0 = time.time()
    print('Running real-label decoding...')

    real_scores = run_decoding_scores(
        X,
        y,
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
        C=C,
    )

    real_mean = real_scores.mean(axis=0)
    real_sem = real_scores.std(axis=0, ddof=1) / np.sqrt(real_scores.shape[0])

    print(f'Real decoding done in {time.time() - t0:.1f} s')

    print(f'Running permutation test: n_permutations={n_permutations}')
    perm_mean_all = np.zeros((n_permutations, len(times)), dtype=float)

    t_perm = time.time()

    for i in range(n_permutations):
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_perm
            print(f'  permutation {i + 1}/{n_permutations} | elapsed {elapsed:.1f} s')

        y_perm = rng.permutation(y)

        perm_scores = run_decoding_scores(
            X,
            y_perm,
            n_splits=n_splits,
            test_size=test_size,
            random_state=random_state + i + 1,
            C=C,
        )

        perm_mean_all[i] = perm_scores.mean(axis=0)

    p_vals = (
        np.sum(perm_mean_all >= real_mean[None, :], axis=0) + 1
    ) / (n_permutations + 1)

    sig_mask_uncorrected = (
        np.isfinite(p_vals)
        & (p_vals < p_threshold)
        & (real_mean > CHANCE_LEVEL)
    )

    sig_mask = filter_segments_by_min_duration(
        sig_mask_uncorrected,
        times,
        min_duration_s=min_sig_duration_s,
    )

    null_mean = perm_mean_all.mean(axis=0)
    null_std = perm_mean_all.std(axis=0, ddof=1)

    z_vals = np.divide(
        real_mean - null_mean,
        null_std,
        out=np.full_like(real_mean, np.nan),
        where=(null_std > 0),
    )

    return {
        'real_scores': real_scores,
        'scores_mean': real_mean,
        'scores_sem': real_sem,
        'perm_mean_all': perm_mean_all,
        'null_mean': null_mean,
        'null_std': null_std,
        'z_vals': z_vals,
        'p_vals': p_vals,
        'sig_mask_uncorrected': sig_mask_uncorrected,
        'sig_mask': sig_mask,
        'times': times,
        'n_permutations': n_permutations,
        'p_threshold': p_threshold,
        'min_sig_duration_s': min_sig_duration_s,
        'C': C,
        'n_splits': n_splits,
    }


# =========================
# 保存与绘图
# =========================
def save_stats_table(times, result, save_path):
    df = pd.DataFrame({
        'time': times,
        'auc_mean': result['scores_mean'],
        'auc_sem': result['scores_sem'],
        'null_mean': result['null_mean'],
        'null_std': result['null_std'],
        'z_value_vs_null': result['z_vals'],
        'p_value_uncorrected': result['p_vals'],
        'significant_uncorrected_p_lt_threshold': result['sig_mask_uncorrected'].astype(int),
        'significant_after_min_duration_filter': result['sig_mask'].astype(int),
    })

    df.to_csv(save_path, index=False, encoding='utf-8-sig')


def add_significance_bands(
    ax,
    times,
    sig_mask,
    color='gold',
    alpha=0.25,
    label='Permutation p < 0.05',
):
    segments = find_contiguous_segments(sig_mask, times)

    if not segments:
        return

    dt = np.median(np.diff(times)) if len(times) > 1 else 0.0
    first = True

    for _, _, start_t, end_t in segments:
        left = start_t - dt / 2
        right = end_t + dt / 2

        if first:
            ax.axvspan(
                left,
                right,
                color=color,
                alpha=alpha,
                label=label,
                zorder=0,
            )
            first = False
        else:
            ax.axvspan(
                left,
                right,
                color=color,
                alpha=alpha,
                zorder=0,
            )


def plot_permutation_result(result, title, save_path):
    times = result['times']
    scores_mean = result['scores_mean']
    scores_sem = result['scores_sem']
    sig_mask = result['sig_mask']

    fig, ax = plt.subplots(figsize=(10, 5))

    add_significance_bands(
        ax,
        times,
        sig_mask,
        color='gold',
        alpha=0.25,
        label='Permutation p < 0.05',
    )

    ax.plot(times, scores_mean, color='C0', lw=2, label='AUC')

    ax.fill_between(
        times,
        scores_mean - scores_sem,
        scores_mean + scores_sem,
        color='C0',
        alpha=0.20,
        linewidth=0,
    )

    ax.axhline(
        CHANCE_LEVEL,
        color='k',
        linestyle='--',
        lw=1,
        label='Chance',
    )

    ax.axvline(
        0,
        color='k',
        linestyle='-',
        lw=1,
        alpha=0.8,
    )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('AUC')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Figure saved to: {os.path.abspath(save_path)}')
    plt.close(fig)


def print_significant_segments(name, result):
    segments = find_contiguous_segments(result['sig_mask'], result['times'])

    print(f'\n{name} significant segments:')

    if not segments:
        print('  None')
        return

    for i, (_, _, start_t, end_t) in enumerate(segments, start=1):
        print(f'  Segment {i}: {start_t:.3f}s ~ {end_t:.3f}s')


def print_uncorrected_sig_points(name, result):
    times = result['times']
    p_vals = result['p_vals']
    mean_auc = result['scores_mean']
    sig_unc = result['sig_mask_uncorrected']

    print(f'\n{name} uncorrected significant points:')

    if sig_unc.sum() == 0:
        print('  None')
        return

    idxs = np.where(sig_unc)[0]

    for idx in idxs:
        print(
            f'  time={times[idx]:.3f}s, '
            f'mean_auc={mean_auc[idx]:.4f}, '
            f'p={p_vals[idx]:.5f}'
        )


# =========================
# 单被试运行函数
# =========================
def run_one_subject(
    subject_dir,
    df_pred,
    behavior_path,
    results_root,
    figures_root,
    n_splits=10,
    test_size=0.2,
    n_permutations=50,
    p_threshold=0.05,
    min_sig_duration_s=0.02,
    C=1.0,
    random_state=12345,
    run_task1_response=True,
    run_task2_stimulus=True,
    skip_existing=False,
):
    subject = os.path.basename(subject_dir)
    sub_id_int = int(re.search(r'\d+', subject).group())

    results_dir = os.path.join(results_root, subject)
    figures_dir = os.path.join(figures_root, subject)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    npz_path = os.path.join(
        results_dir,
        f'{subject}_abs1_decoding_permutation_results.npz',
    )

    if skip_existing and os.path.exists(npz_path):
        print(f'\n===== SKIP {subject}: existing result found =====')
        return {
            'subject': subject,
            'status': 'skipped_existing',
            'message': 'existing npz found',
        }

    print('\n\n' + '=' * 80)
    print(f'===== START {subject} =====')
    print('=' * 80)
    print(f'EEG dir: {subject_dir}')
    print(f'Behavior file: {behavior_path}')

    # ========= 根据 behavior id_sess 自动决定 EEG sessions =========
    behavior_sessions = get_behavior_sessions_for_subject(df_pred, sub_id_int)
    keep_sessions = behavior_sessions_to_eeg_sessions(behavior_sessions)

    print(f'\nBehavior sessions: {behavior_sessions}')
    print(f'EEG sessions to keep: {keep_sessions}')

    # ========= 找 EEG 文件 =========
    task1_stim_files_all = sorted(
        glob(os.path.join(subject_dir, 'clean_epochs_task1_ICA_*.fif'))
    )

    task1_resp_files_all = sorted(
        glob(os.path.join(subject_dir, 'clean_epochs_response1_ICA_*.fif'))
    )

    task2_stim_files_all = sorted(
        glob(os.path.join(subject_dir, 'clean_epochs_task2_ICA_*.fif'))
    )

    if (
        len(task1_stim_files_all) == 0
        or len(task1_resp_files_all) == 0
        or len(task2_stim_files_all) == 0
    ):
        raise FileNotFoundError('未找到完整的 task1/task2 EEG 文件，请检查被试目录。')

    # ========= 只保留与 behavior sessions 对应的 EEG sessions =========
    task1_stim_files = filter_files_by_sessions(
        task1_stim_files_all,
        keep_sessions,
        name='Task1 Stim files',
    )

    task1_resp_files = filter_files_by_sessions(
        task1_resp_files_all,
        keep_sessions,
        name='Task1 Response files',
    )

    task2_stim_files = filter_files_by_sessions(
        task2_stim_files_all,
        keep_sessions,
        name='Task2 Stimulus files',
    )

    if not (
        len(task1_stim_files)
        == len(task1_resp_files)
        == len(task2_stim_files)
        == len(behavior_sessions)
    ):
        raise RuntimeError(
            '过滤后的 EEG 文件数和 behavior session 数不一致。\n'
            f'  Task1 Stim files: {len(task1_stim_files)}\n'
            f'  Task1 Resp files: {len(task1_resp_files)}\n'
            f'  Task2 Stim files: {len(task2_stim_files)}\n'
            f'  Behavior sessions: {len(behavior_sessions)}'
        )

    # ========= 读取 behavior =========
    beh_aligned = get_behavior_blocks_by_sessions(
        df_pred,
        sub_id_int,
        id_sess_to_use=behavior_sessions,
    )

    print(f'\nBehavior rows: {len(beh_aligned)}')

    # ========= 读取 EEG =========
    epochs_t1_stim = mne.concatenate_epochs(
        [mne.read_epochs(f, verbose=False) for f in task1_stim_files]
    )

    epochs_t1_resp = read_concat(
        task1_resp_files,
        CS.baseline_response,
        -0.5,
        CS.tmax_response,
    )

    epochs_t2_stim = read_concat(
        task2_stim_files,
        CS.baseline_task2,
        CS.tmin_task2,
        CS.tmax_task2,
    )

    # ========= 保存 alignment report =========
    alignment_report_path = os.path.join(
        results_dir,
        f'{subject}_alignment_report_fixed.txt',
    )

    save_alignment_report(
        save_path=alignment_report_path,
        subject=subject,
        keep_sessions=keep_sessions,
        behavior_sessions=behavior_sessions,
        task1_stim_files=task1_stim_files,
        task1_resp_files=task1_resp_files,
        task2_stim_files=task2_stim_files,
        epochs_t1_stim=epochs_t1_stim,
        epochs_t1_resp=epochs_t1_resp,
        epochs_t2_stim=epochs_t2_stim,
        beh_aligned=beh_aligned,
    )

    # ========= 严格检查：不再静默截断 =========
    lengths = {
        'Task1 Stim epochs': len(epochs_t1_stim),
        'Task1 Response epochs': len(epochs_t1_resp),
        'Task2 Stim epochs': len(epochs_t2_stim),
        'Behavior rows': len(beh_aligned),
    }

    if len(set(lengths.values())) != 1:
        print('\nERROR: EEG/behavior lengths are not equal after filtering.')
        for k, v in lengths.items():
            print(f'  {k}: {v}')
        raise RuntimeError('长度不一致，停止该被试分析。')

    print('\nLength check passed:')
    for k, v in lengths.items():
        print(f'  {k}: {v}')

    # ========= 构造标签 =========
    y_group, valid_mask = build_abs1_labels(
        beh_aligned,
        threshold=ABS1_THRESHOLD,
    )

    if valid_mask.sum() < len(valid_mask):
        print(
            f'WARNING: abs1 has missing values. '
            f'Keeping {valid_mask.sum()} / {len(valid_mask)} trials.'
        )

        epochs_t1_stim = epochs_t1_stim[valid_mask]
        epochs_t1_resp = epochs_t1_resp[valid_mask]
        epochs_t2_stim = epochs_t2_stim[valid_mask]
        beh_aligned = beh_aligned.loc[valid_mask].reset_index(drop=True)

    class_counts = np.unique(y_group, return_counts=True)

    print(f'\nAligned trials used in decoding: {len(y_group)}')
    print(f'Low/High counts (0=low, 1=high): {class_counts}')

    if len(np.unique(y_group)) < 2:
        raise RuntimeError('abs1 分组后只有一个类别，无法继续做解码。')

    # ========= 保存最终 behavior =========
    beh_used_path = os.path.join(
        results_dir,
        f'{subject}_behavior_used_for_decoding.csv',
    )

    beh_aligned.to_csv(
        beh_used_path,
        index=False,
        encoding='utf-8-sig',
    )

    # ========= 准备解码数据 =========
    X_t1_resp = epochs_t1_resp.get_data(copy=False)
    X_t2_stim = epochs_t2_stim.get_data(copy=False)

    times_t1_resp = epochs_t1_resp.times
    times_t2_stim = epochs_t2_stim.times

    result_t1_resp = None
    result_t2_stim = None

    # ========= Task1 Response =========
    if run_task1_response:
        print('\nRunning Task1 Response decoding + permutation test...')
        t0 = time.time()

        result_t1_resp = run_permutation_test(
            X_t1_resp,
            y_group,
            times_t1_resp,
            n_splits=n_splits,
            test_size=test_size,
            random_state=random_state,
            n_permutations=n_permutations,
            p_threshold=p_threshold,
            min_sig_duration_s=min_sig_duration_s,
            C=C,
        )

        print(f'Task1 total time: {time.time() - t0:.1f} s')

        t1_csv = os.path.join(
            results_dir,
            f'{subject}_task1_response_abs1_permutation.csv',
        )

        t1_fig = os.path.join(
            figures_dir,
            f'{subject}_task1_response_abs1_permutation.png',
        )

        save_stats_table(times_t1_resp, result_t1_resp, t1_csv)

        plot_permutation_result(
            result_t1_resp,
            title=f'Task1 Response: abs1 low/high decoding ({subject})\nPermutation test, fixed alignment',
            save_path=t1_fig,
        )

        print_significant_segments('Task1 Response', result_t1_resp)
        print_uncorrected_sig_points('Task1 Response', result_t1_resp)

    # ========= Task2 Stimulus =========
    if run_task2_stimulus:
        print('\nRunning Task2 Stimulus decoding + permutation test...')
        t0 = time.time()

        result_t2_stim = run_permutation_test(
            X_t2_stim,
            y_group,
            times_t2_stim,
            n_splits=n_splits,
            test_size=test_size,
            random_state=random_state + 1000,
            n_permutations=n_permutations,
            p_threshold=p_threshold,
            min_sig_duration_s=min_sig_duration_s,
            C=C,
        )

        print(f'Task2 total time: {time.time() - t0:.1f} s')

        t2_csv = os.path.join(
            results_dir,
            f'{subject}_task2_stim_abs1_permutation.csv',
        )

        t2_fig = os.path.join(
            figures_dir,
            f'{subject}_task2_stim_abs1_permutation.png',
        )

        save_stats_table(times_t2_stim, result_t2_stim, t2_csv)

        plot_permutation_result(
            result_t2_stim,
            title=f'Task2 Stimulus: abs1 low/high decoding ({subject})\nPermutation test, fixed alignment',
            save_path=t2_fig,
        )

        print_significant_segments('Task2 Stimulus', result_t2_stim)
        print_uncorrected_sig_points('Task2 Stimulus', result_t2_stim)

    # ========= 保存 NPZ =========
    save_dict = {
        'y_group': y_group,
        'abs1_threshold': ABS1_THRESHOLD,
        'used_behavior_sessions': np.array(behavior_sessions, dtype=str),
        'used_eeg_sessions': np.array(keep_sessions, dtype=str),
        'chance_level': CHANCE_LEVEL,
        'n_permutations': n_permutations,
        'p_threshold': p_threshold,
        'min_sig_duration_s': min_sig_duration_s,
        'n_splits': n_splits,
        'test_size': test_size,
        'C': C,
    }

    if result_t1_resp is not None:
        save_dict.update({
            'real_scores_t1_resp': result_t1_resp['real_scores'],
            'scores_t1_resp_mean': result_t1_resp['scores_mean'],
            'scores_t1_resp_sem': result_t1_resp['scores_sem'],
            'p_vals_t1_resp': result_t1_resp['p_vals'],
            'sig_mask_unc_t1_resp': result_t1_resp['sig_mask_uncorrected'],
            'sig_mask_t1_resp': result_t1_resp['sig_mask'],
            'null_mean_t1_resp': result_t1_resp['null_mean'],
            'null_std_t1_resp': result_t1_resp['null_std'],
            'perm_mean_all_t1_resp': result_t1_resp['perm_mean_all'],
            'times_t1_resp': times_t1_resp,
        })

    if result_t2_stim is not None:
        save_dict.update({
            'real_scores_t2_stim': result_t2_stim['real_scores'],
            'scores_t2_stim_mean': result_t2_stim['scores_mean'],
            'scores_t2_stim_sem': result_t2_stim['scores_sem'],
            'p_vals_t2_stim': result_t2_stim['p_vals'],
            'sig_mask_unc_t2_stim': result_t2_stim['sig_mask_uncorrected'],
            'sig_mask_t2_stim': result_t2_stim['sig_mask'],
            'null_mean_t2_stim': result_t2_stim['null_mean'],
            'null_std_t2_stim': result_t2_stim['null_std'],
            'perm_mean_all_t2_stim': result_t2_stim['perm_mean_all'],
            'times_t2_stim': times_t2_stim,
        })

    np.savez(npz_path, **save_dict)

    print(f'\nNPZ saved to: {os.path.abspath(npz_path)}')
    print(f'===== DONE {subject} =====')

    return {
        'subject': subject,
        'status': 'done',
        'message': 'success',
        'n_trials': len(y_group),
        'behavior_sessions': ','.join(behavior_sessions),
        'eeg_sessions': ','.join(keep_sessions),
    }


# =========================
# 主程序：批量运行所有被试
# =========================
if __name__ == "__main__":
    # ========= 路径设置 =========
    clean_eeg_root = os.path.join('../../..', 'data', 'clean_EEG')

    behavior_path = os.path.join(
        '../../..',
        'data',
        'behavior_for_behavior',
        'data_wide_wmPred.txt',
    )

    results_root = os.path.join(
        '../../..',
        'results',
        'decoding_permutation_fixed_alignment',
    )

    figures_root = os.path.join(
        '../../..',
        'figures',
        'decoding_permutation_fixed_alignment',
    )

    os.makedirs(results_root, exist_ok=True)
    os.makedirs(figures_root, exist_ok=True)

    # ========= 解码参数 =========
    n_splits = 10
    test_size = 0.2
    n_permutations = 50
    p_threshold = 0.05
    min_sig_duration_s = 0.02
    C = 1.0
    random_state = 12345

    # ========= 运行控制 =========
    run_task1_response = True
    run_task2_stimulus = True

    # 如果已经有 npz，是否跳过
    skip_existing = False

    print('================ BATCH DECODING START ================')
    print(f'clean_eeg_root: {os.path.abspath(clean_eeg_root)}')
    print(f'behavior_path: {os.path.abspath(behavior_path)}')
    print(f'results_root: {os.path.abspath(results_root)}')
    print(f'figures_root: {os.path.abspath(figures_root)}')
    print(f'n_splits={n_splits}, n_permutations={n_permutations}, C={C}')
    print(f'skip_existing={skip_existing}')

    # ========= 读取行为表 =========
    df_pred = pd.read_csv(behavior_path, sep=';')

    # ========= 找所有被试 =========
    subject_dirs = get_subject_dirs(clean_eeg_root)

    if len(subject_dirs) == 0:
        raise RuntimeError(f'在 {clean_eeg_root} 下未找到 sub* 被试文件夹。')

    print(f'\nFound {len(subject_dirs)} subject folders:')
    for d in subject_dirs:
        print(' ', os.path.basename(d))

    batch_records = []

    # ========= 批量运行 =========
    for subject_dir in subject_dirs:
        subject = os.path.basename(subject_dir)

        try:
            record = run_one_subject(
                subject_dir=subject_dir,
                df_pred=df_pred,
                behavior_path=behavior_path,
                results_root=results_root,
                figures_root=figures_root,
                n_splits=n_splits,
                test_size=test_size,
                n_permutations=n_permutations,
                p_threshold=p_threshold,
                min_sig_duration_s=min_sig_duration_s,
                C=C,
                random_state=random_state,
                run_task1_response=run_task1_response,
                run_task2_stimulus=run_task2_stimulus,
                skip_existing=skip_existing,
            )

        except Exception as e:
            print('\n' + '!' * 80)
            print(f'ERROR in {subject}: {repr(e)}')
            print(traceback.format_exc())
            print('!' * 80)

            record = {
                'subject': subject,
                'status': 'failed',
                'message': repr(e),
                'n_trials': np.nan,
                'behavior_sessions': '',
                'eeg_sessions': '',
            }

        batch_records.append(record)

        # 每跑完一个被试就保存一次 summary，防止中途断掉丢记录
        batch_summary_path = os.path.join(
            results_root,
            'batch_summary_decoding_permutation_fixed_alignment.csv',
        )

        pd.DataFrame(batch_records).to_csv(
            batch_summary_path,
            index=False,
            encoding='utf-8-sig',
        )

    print('\n================ BATCH DECODING SUMMARY ================')
    summary_df = pd.DataFrame(batch_records)
    print(summary_df)

    batch_summary_path = os.path.join(
        results_root,
        'batch_summary_decoding_permutation_fixed_alignment.csv',
    )

    summary_df.to_csv(
        batch_summary_path,
        index=False,
        encoding='utf-8-sig',
    )

    print(f'\nBatch summary saved to: {os.path.abspath(batch_summary_path)}')
    print('================ BATCH DECODING DONE ================')