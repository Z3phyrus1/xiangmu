# -*- coding: utf-8 -*-
"""
单被试 decoding + permutation test 修复版

修复重点：
1. 只读取与行为 id_sess 对应的 EEG session
   当前 sub54 行为只有 54fl03 和 54fl04，因此 EEG 只保留 sess3 和 sess4
2. 不再允许 EEG=900、behavior=600 时继续硬截断
3. 过滤后要求 EEG 和 behavior trial 数一致
4. 对 Task1 Response 和 Task2 Stimulus 做 abs1 low/high decoding
5. 用标签置换 permutation test 检验 AUC 是否高于随机标签
"""

import os
import re
import time
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
# 基础工具函数
# =========================
def filter_files_by_sessions(files, keep_sessions, name):
    """
    只保留文件名中包含指定 sess 的 EEG 文件。
    例如 keep_sessions = ['sess3', 'sess4']
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


def read_concat(files, baseline, crop_tmin, crop_tmax, resample_hz=100):
    epochs_list = [mne.read_epochs(f, verbose=False) for f in files]

    for epochs in epochs_list:
        epochs.apply_baseline(baseline)

    epochs = mne.concatenate_epochs(epochs_list)
    epochs.crop(crop_tmin, crop_tmax).resample(resample_hz)
    return epochs


def get_behavior_blocks_by_sessions(df_pred, sub_id_int, id_sess_to_use):
    """
    明确按给定 id_sess 读取行为 block。
    这样比“自动取最后几个 session”更安全。
    """
    df_sub = df_pred[df_pred['id'].astype(int) == sub_id_int].copy()

    if df_sub.empty:
        raise RuntimeError(f'行为文件中未找到 id={sub_id_int} 的数据。')

    available_sessions = df_sub['id_sess'].astype(str).unique().tolist()
    print('\nAvailable behavior id_sess for this subject:')
    for sid in available_sessions:
        n = (df_sub['id_sess'].astype(str) == sid).sum()
        print(f'  {sid}: {n} rows')

    missing = [sid for sid in id_sess_to_use if sid not in available_sessions]
    if len(missing) > 0:
        raise RuntimeError(f'行为文件中找不到这些 id_sess: {missing}')

    beh_blocks = []
    for sid in id_sess_to_use:
        block = df_sub[df_sub['id_sess'].astype(str) == str(sid)].copy()
        beh_blocks.append(block)

    beh_aligned = pd.concat(beh_blocks, axis=0, ignore_index=True)
    return beh_aligned


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
    """
    保存本次实际使用的 EEG 文件和 behavior block。
    重点检查长度是否一致。
    """
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
        log(f'  Task1 Response selection length = {len(s_resp)}')
        log(f'  Task2 Stimulus selection length = {len(s_t2)}')
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
        log('Do not continue decoding until this is fixed.')

    log('================ END ALIGNMENT REPORT ================\n')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'Alignment report saved to: {os.path.abspath(save_path)}')


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
    """
    只保留持续时间 >= min_duration_s 的连续显著片段。

    注意：
    这里按 bin 覆盖长度计算：
        duration = end_time - start_time + dt
    这样两个相邻 100Hz 时间点，比如 0.200 和 0.210，
    会被视为大约 0.020s。
    """
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
    """
    单被试 permutation test。

    真实标签：
        real_scores: shape = (n_splits, n_times)
        real_mean: shape = (n_times,)

    置换标签：
        每次打乱 y，重新跑 decoding
        perm_mean_all: shape = (n_permutations, n_times)

    p 值：
        单尾，检验真实 AUC 是否高于随机标签下的 AUC。
    """
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
# 主程序
# =========================
if __name__ == "__main__":
    # ========= 基本参数 =========
    subject = 'sub28'
    sub_id_int = int(re.search(r'\d+', subject).group())
    random_state = 12345

    # ========= 关键修复参数 =========
    # sub54 行为只有 54fl03 和 54fl04，
    # 所以 EEG 只保留 sess3 和 sess4。
    keep_sessions = ['sess3', 'sess4']
    behavior_sessions = ['54fl03', '54fl04']

    # ========= 解码与 permutation 参数 =========
    n_splits = 10
    test_size = 0.2
    n_permutations = 50
    p_threshold = 0.05
    min_sig_duration_s = 0.02
    C = 1.0

    # ========= 是否运行两个任务 =========
    run_task1_response = True
    run_task2_stimulus = True

    # ========= 路径 =========
    working_dir = os.path.join('../../..', 'data', 'clean_EEG', subject)
    behavior_path = os.path.join(
        '../../..',
        'data',
        'behavior_for_behavior',
        'data_wide_wmPred.txt',
    )

    results_dir = os.path.join(
        '../../..',
        'results',
        'decoding_permutation',
        subject,
    )

    figures_dir = os.path.join(
        '../../..',
        'figures',
        'decoding_permutation',
        subject,
    )

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f'===== START {subject} =====')
    print(f'EEG dir: {working_dir}')
    print(f'Behavior file: {behavior_path}')
    print(f'Results dir: {os.path.abspath(results_dir)}')
    print(f'Figures dir: {os.path.abspath(figures_dir)}')
    print(f'keep_sessions = {keep_sessions}')
    print(f'behavior_sessions = {behavior_sessions}')
    print(f'n_splits={n_splits}, n_permutations={n_permutations}, C={C}')

    # ========= 找 EEG 文件 =========
    task1_stim_files_all = sorted(
        glob(os.path.join(working_dir, 'clean_epochs_task1_ICA_*.fif'))
    )

    task1_resp_files_all = sorted(
        glob(os.path.join(working_dir, 'clean_epochs_response1_ICA_*.fif'))
    )

    task2_stim_files_all = sorted(
        glob(os.path.join(working_dir, 'clean_epochs_task2_ICA_*.fif'))
    )

    if (
        len(task1_stim_files_all) == 0
        or len(task1_resp_files_all) == 0
        or len(task2_stim_files_all) == 0
    ):
        raise FileNotFoundError('未找到完整的 task1/task2 EEG 文件，请检查被试目录。')

    # ========= 修复：只保留 sess3 和 sess4 =========
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

    # 三类 EEG 文件数量必须一致
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
    df_pred = pd.read_csv(behavior_path, sep=';')

    beh_aligned = get_behavior_blocks_by_sessions(
        df_pred,
        sub_id_int,
        id_sess_to_use=behavior_sessions,
    )

    print(f'\nUsing behavior id_sess blocks: {behavior_sessions}')
    print(f'Behavior rows: {len(beh_aligned)}')

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
        print('\nERROR: EEG/behavior lengths are still not equal after filtering.')
        for k, v in lengths.items():
            print(f'  {k}: {v}')
        raise RuntimeError(
            '长度仍不一致，停止分析。请先检查 alignment_report_fixed.txt。'
        )

    print('\nLength check passed:')
    for k, v in lengths.items():
        print(f'  {k}: {v}')

    # ========= 根据 abs1 构造 low/high 标签 =========
    y_group, valid_mask = build_abs1_labels(
        beh_aligned,
        threshold=ABS1_THRESHOLD,
    )

    # 如果 abs1 有缺失值，则删除对应 trial
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

    # ========= 保存最终用于分析的 behavior =========
    beh_used_path = os.path.join(
        results_dir,
        f'{subject}_behavior_used_for_decoding.csv',
    )

    beh_aligned.to_csv(
        beh_used_path,
        index=False,
        encoding='utf-8-sig',
    )

    print(f'Behavior used for decoding saved to: {os.path.abspath(beh_used_path)}')

    # ========= 准备解码数据 =========
    X_t1_resp = epochs_t1_resp.get_data(copy=False)
    X_t2_stim = epochs_t2_stim.get_data(copy=False)

    times_t1_resp = epochs_t1_resp.times
    times_t2_stim = epochs_t2_stim.times

    result_t1_resp = None
    result_t2_stim = None

    # ========= Task1 Response 解码 =========
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

        print(f'Task1 stats saved to: {os.path.abspath(t1_csv)}')

    # ========= Task2 Stimulus 解码 =========
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

        print(f'Task2 stats saved to: {os.path.abspath(t2_csv)}')

    # ========= 保存 NPZ =========
    npz_path = os.path.join(
        results_dir,
        f'{subject}_abs1_decoding_permutation_results.npz',
    )

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
    print('DONE')