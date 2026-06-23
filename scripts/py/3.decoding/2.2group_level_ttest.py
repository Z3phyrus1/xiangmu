# -*- coding: utf-8 -*-
"""
Group-level t-test for fixed-alignment decoding results

用途：
1. 读取 fixed-alignment 批量单被试结果 npz
2. 汇总 Task1 Response / Task2 Stimulus 的 mean AUC
3. 对每个时间点做 one-sample t-test vs chance = 0.5
4. 绘制 group mean ± SEM
5. 高亮 uncorrected p < 0.05 的连续时间段
6. 保存 group csv / png / npz

输入目录默认：
../../../results/decoding_permutation_fixed_alignment/subXX/*_abs1_decoding_permutation_results.npz

输出目录：
../../../results/group_decoding_ttest_fixed_alignment
../../../figures/group_decoding_ttest_fixed_alignment
"""

import os
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


CHANCE_LEVEL = 0.5


# =========================
# 工具函数
# =========================
def extract_subject_name(npz_path):
    """
    从路径中提取 subXX。
    """
    path_norm = npz_path.replace('\\', '/')
    m = re.search(r'/(sub\d+)/', path_norm)
    if m:
        return m.group(1)

    base = os.path.basename(npz_path)
    m = re.search(r'(sub\d+)', base)
    if m:
        return m.group(1)

    return base.split('_')[0]


def sub_number(subject):
    m = re.search(r'\d+', subject)
    return int(m.group()) if m else 999999


def find_contiguous_segments(mask, times):
    """
    把 bool mask 转成连续显著时间段。
    返回：
        [(start_idx, end_idx, start_time, end_time), ...]
    """
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
    只保留覆盖时长 >= min_duration_s 的连续显著片段。

    这里按 bin 覆盖长度计算：
        duration = end_time - start_time + dt

    例如 100 Hz 下，0.200 和 0.210 两个相邻点，
    会被视为约 0.020 s。
    """
    filtered = np.zeros_like(mask, dtype=bool)
    segments = find_contiguous_segments(mask, times)

    if len(times) > 1:
        dt = np.median(np.diff(times))
    else:
        dt = 0.0

    for start_idx, end_idx, start_t, end_t in segments:
        duration = end_t - start_t + dt

        if duration >= min_duration_s:
            filtered[start_idx:end_idx + 1] = True

    return filtered


def add_significance_bands(
    ax,
    times,
    sig_mask,
    color='gold',
    alpha=0.25,
    label='uncorrected p < 0.05',
):
    """
    在图上高亮连续显著片段。
    """
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


# =========================
# 读取单被试 npz
# =========================
def load_group_npz(npz_files, score_key, time_key):
    """
    读取所有被试 npz。

    返回：
        scores_all: shape = (n_subjects, n_times)
        times: shape = (n_times,)
        subjects: list[str]
        skipped: list[dict]
    """
    rows = []
    skipped = []

    ref_times = None

    for f in sorted(npz_files):
        subject = extract_subject_name(f)

        try:
            data = np.load(f, allow_pickle=True)

            if score_key not in data:
                skipped.append({
                    'subject': subject,
                    'file': f,
                    'reason': f'missing key: {score_key}',
                })
                continue

            if time_key not in data:
                skipped.append({
                    'subject': subject,
                    'file': f,
                    'reason': f'missing key: {time_key}',
                })
                continue

            score = np.asarray(data[score_key], dtype=float)
            times = np.asarray(data[time_key], dtype=float)

            if score.ndim != 1:
                skipped.append({
                    'subject': subject,
                    'file': f,
                    'reason': f'{score_key} is not 1D, shape={score.shape}',
                })
                continue

            if ref_times is None:
                ref_times = times
            else:
                if len(times) != len(ref_times) or not np.allclose(times, ref_times):
                    skipped.append({
                        'subject': subject,
                        'file': f,
                        'reason': 'time axis mismatch',
                    })
                    continue

            rows.append({
                'subject': subject,
                'file': f,
                'score': score,
            })

        except Exception as e:
            skipped.append({
                'subject': subject,
                'file': f,
                'reason': repr(e),
            })

    if len(rows) == 0:
        raise RuntimeError(f'没有成功读取任何被试结果。score_key={score_key}, time_key={time_key}')

    rows = sorted(rows, key=lambda x: sub_number(x['subject']))

    subjects = [r['subject'] for r in rows]
    scores_all = np.vstack([r['score'] for r in rows])

    return scores_all, ref_times, subjects, skipped


# =========================
# group t 检验
# =========================
def compute_group_ttest(
    scores_all,
    times,
    chance_level=0.5,
    p_threshold=0.05,
    min_sig_duration_s=0.02,
    alternative='greater',
):
    """
    scores_all: shape = (n_subjects, n_times)

    alternative:
        'greater'：检验 AUC > chance
        'two-sided'：双尾检验
    """
    n_subjects = scores_all.shape[0]

    mean_auc = np.nanmean(scores_all, axis=0)
    sem_auc = np.nanstd(scores_all, axis=0, ddof=1) / np.sqrt(n_subjects)

    # scipy 的 ttest_1samp 在较新版本支持 alternative。
    # 为了兼容旧版本，这里手动计算单尾。
    t_vals, p_two = stats.ttest_1samp(
        scores_all,
        popmean=chance_level,
        axis=0,
        nan_policy='omit',
    )

    if alternative == 'greater':
        # 单尾：AUC > chance
        p_vals = np.where(t_vals > 0, p_two / 2, 1 - p_two / 2)
        sig_mask_uncorrected = (
            np.isfinite(p_vals)
            & (p_vals < p_threshold)
            & (mean_auc > chance_level)
        )

    elif alternative == 'two-sided':
        p_vals = p_two
        sig_mask_uncorrected = (
            np.isfinite(p_vals)
            & (p_vals < p_threshold)
        )

    else:
        raise ValueError("alternative must be 'greater' or 'two-sided'")

    sig_mask = filter_segments_by_min_duration(
        sig_mask_uncorrected,
        times,
        min_duration_s=min_sig_duration_s,
    )

    return {
        'scores_all': scores_all,
        'times': times,
        'mean_auc': mean_auc,
        'sem_auc': sem_auc,
        't_vals': t_vals,
        'p_vals': p_vals,
        'p_two_sided': p_two,
        'sig_mask_uncorrected': sig_mask_uncorrected,
        'sig_mask': sig_mask,
        'n_subjects': n_subjects,
        'chance_level': chance_level,
        'p_threshold': p_threshold,
        'min_sig_duration_s': min_sig_duration_s,
        'alternative': alternative,
    }


# =========================
# 保存结果
# =========================
def save_group_stats_table(result, subjects, save_csv):
    """
    保存每个时间点的 group 统计结果。
    """
    df = pd.DataFrame({
        'time': result['times'],
        'group_mean_auc': result['mean_auc'],
        'group_sem_auc': result['sem_auc'],
        't_value_vs_0.5': result['t_vals'],
        'p_value': result['p_vals'],
        'p_value_two_sided': result['p_two_sided'],
        'significant_uncorrected_p_lt_threshold': result['sig_mask_uncorrected'].astype(int),
        'significant_after_min_duration_filter': result['sig_mask'].astype(int),
    })

    df.to_csv(save_csv, index=False, encoding='utf-8-sig')

    subject_txt = save_csv.replace('.csv', '_subjects.txt')
    with open(subject_txt, 'w', encoding='utf-8') as f:
        for sub in subjects:
            f.write(f'{sub}\n')


def save_subject_matrix(scores_all, times, subjects, save_csv):
    """
    保存被试 x 时间点矩阵，方便之后手动检查。
    """
    df = pd.DataFrame(scores_all, columns=[f'{t:.3f}' for t in times])
    df.insert(0, 'subject', subjects)
    df.to_csv(save_csv, index=False, encoding='utf-8-sig')


def save_skipped_table(skipped, save_csv):
    if len(skipped) == 0:
        df = pd.DataFrame(columns=['subject', 'file', 'reason'])
    else:
        df = pd.DataFrame(skipped)

    df.to_csv(save_csv, index=False, encoding='utf-8-sig')


# =========================
# 绘图
# =========================
def plot_group_result(result, title, save_path):
    times = result['times']
    mean_auc = result['mean_auc']
    sem_auc = result['sem_auc']
    sig_mask = result['sig_mask']
    n_subjects = result['n_subjects']

    fig, ax = plt.subplots(figsize=(10, 5))

    add_significance_bands(
        ax,
        times,
        sig_mask,
        color='gold',
        alpha=0.25,
        label='uncorrected p < 0.05',
    )

    ax.plot(
        times,
        mean_auc,
        color='C0',
        lw=2,
        label=f'Group mean AUC (n={n_subjects})',
    )

    ax.fill_between(
        times,
        mean_auc - sem_auc,
        mean_auc + sem_auc,
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


def print_group_summary(name, result):
    """
    打印最小 p、显著点数量和显著片段。
    """
    times = result['times']
    p_vals = result['p_vals']
    mean_auc = result['mean_auc']
    sig_unc = result['sig_mask_uncorrected']
    sig_mask = result['sig_mask']

    print(f'\n===== {name} summary =====')
    print(f'n_subjects = {result["n_subjects"]}')
    print(f'alternative = {result["alternative"]}')
    print(f'min p = {np.nanmin(p_vals):.8f}')
    print(f'uncorrected significant points = {int(sig_unc.sum())}')
    print(f'significant points after duration filter = {int(sig_mask.sum())}')

    print('\nUncorrected significant points:')
    if sig_unc.sum() == 0:
        print('  None')
    else:
        idxs = np.where(sig_unc)[0]
        for idx in idxs:
            print(
                f'  time={times[idx]:.3f}s, '
                f'mean_auc={mean_auc[idx]:.4f}, '
                f't={result["t_vals"][idx]:.4f}, '
                f'p={p_vals[idx]:.6f}'
            )

    print('\nSignificant segments after duration filter:')
    segments = find_contiguous_segments(sig_mask, times)
    if len(segments) == 0:
        print('  None')
    else:
        for i, (_, _, start_t, end_t) in enumerate(segments, start=1):
            print(f'  Segment {i}: {start_t:.3f}s ~ {end_t:.3f}s')


# =========================
# 主程序
# =========================
if __name__ == '__main__':
    # ========= 路径设置 =========
    single_subject_results_root = os.path.join(
        '../../..',
        'results',
        'decoding_permutation_fixed_alignment',
    )

    group_results_root = os.path.join(
        '../../..',
        'results',
        'group_decoding_ttest_fixed_alignment',
    )

    group_figures_root = os.path.join(
        '../../..',
        'figures',
        'group_decoding_ttest_fixed_alignment',
    )

    os.makedirs(group_results_root, exist_ok=True)
    os.makedirs(group_figures_root, exist_ok=True)

    print('===== GROUP T-TEST: FIXED ALIGNMENT RESULTS =====')
    print(f'single_subject_results_root: {os.path.abspath(single_subject_results_root)}')
    print(f'group_results_root: {os.path.abspath(group_results_root)}')
    print(f'group_figures_root: {os.path.abspath(group_figures_root)}')

    # ========= 统计参数 =========
    p_threshold = 0.05
    min_sig_duration_s = 0.02

    # greater = 单尾，检验 AUC > 0.5
    # two-sided = 双尾
    alternative = 'greater'

    # ========= 搜索所有单被试 npz =========
    npz_files = sorted(
        glob(
            os.path.join(
                single_subject_results_root,
                'sub*',
                '*_abs1_decoding_permutation_results.npz',
            )
        )
    )

    if len(npz_files) == 0:
        raise FileNotFoundError(
            f'No subject npz files found in: {os.path.abspath(single_subject_results_root)}'
        )

    print(f'\nFound {len(npz_files)} subject npz files:')
    for f in npz_files:
        print(' ', f)

    # =========================
    # Task1 Response
    # =========================
    print('\n\n===== GROUP T-TEST: Task1 Response =====')

    t1_scores_all, t1_times, t1_subjects, t1_skipped = load_group_npz(
        npz_files,
        score_key='scores_t1_resp_mean',
        time_key='times_t1_resp',
    )

    t1_result = compute_group_ttest(
        t1_scores_all,
        t1_times,
        chance_level=CHANCE_LEVEL,
        p_threshold=p_threshold,
        min_sig_duration_s=min_sig_duration_s,
        alternative=alternative,
    )

    t1_csv = os.path.join(
        group_results_root,
        'group_task1_response_abs1_ttest_fixed_alignment.csv',
    )

    t1_matrix_csv = os.path.join(
        group_results_root,
        'group_task1_response_subject_by_time_fixed_alignment.csv',
    )

    t1_skipped_csv = os.path.join(
        group_results_root,
        'group_task1_response_skipped_fixed_alignment.csv',
    )

    t1_fig = os.path.join(
        group_figures_root,
        'group_task1_response_abs1_ttest_fixed_alignment.png',
    )

    save_group_stats_table(t1_result, t1_subjects, t1_csv)
    save_subject_matrix(t1_scores_all, t1_times, t1_subjects, t1_matrix_csv)
    save_skipped_table(t1_skipped, t1_skipped_csv)

    plot_group_result(
        t1_result,
        title='Group Task1 Response: abs1 low/high decoding\nOne-sample t-test vs chance, fixed alignment',
        save_path=t1_fig,
    )

    print_group_summary('Task1 Response', t1_result)

    # =========================
    # Task2 Stimulus
    # =========================
    print('\n\n===== GROUP T-TEST: Task2 Stimulus =====')

    t2_scores_all, t2_times, t2_subjects, t2_skipped = load_group_npz(
        npz_files,
        score_key='scores_t2_stim_mean',
        time_key='times_t2_stim',
    )

    t2_result = compute_group_ttest(
        t2_scores_all,
        t2_times,
        chance_level=CHANCE_LEVEL,
        p_threshold=p_threshold,
        min_sig_duration_s=min_sig_duration_s,
        alternative=alternative,
    )

    t2_csv = os.path.join(
        group_results_root,
        'group_task2_stim_abs1_ttest_fixed_alignment.csv',
    )

    t2_matrix_csv = os.path.join(
        group_results_root,
        'group_task2_stim_subject_by_time_fixed_alignment.csv',
    )

    t2_skipped_csv = os.path.join(
        group_results_root,
        'group_task2_stim_skipped_fixed_alignment.csv',
    )

    t2_fig = os.path.join(
        group_figures_root,
        'group_task2_stim_abs1_ttest_fixed_alignment.png',
    )

    save_group_stats_table(t2_result, t2_subjects, t2_csv)
    save_subject_matrix(t2_scores_all, t2_times, t2_subjects, t2_matrix_csv)
    save_skipped_table(t2_skipped, t2_skipped_csv)

    plot_group_result(
        t2_result,
        title='Group Task2 Stimulus: abs1 low/high decoding\nOne-sample t-test vs chance, fixed alignment',
        save_path=t2_fig,
    )

    print_group_summary('Task2 Stimulus', t2_result)

    # =========================
    # 保存 group npz
    # =========================
    group_npz = os.path.join(
        group_results_root,
        'group_abs1_decoding_ttest_fixed_alignment_results.npz',
    )

    np.savez(
        group_npz,

        t1_scores_all=t1_scores_all,
        t1_times=t1_times,
        t1_subjects=np.array(t1_subjects, dtype=str),
        t1_mean_auc=t1_result['mean_auc'],
        t1_sem_auc=t1_result['sem_auc'],
        t1_t_vals=t1_result['t_vals'],
        t1_p_vals=t1_result['p_vals'],
        t1_p_two_sided=t1_result['p_two_sided'],
        t1_sig_mask_unc=t1_result['sig_mask_uncorrected'],
        t1_sig_mask=t1_result['sig_mask'],

        t2_scores_all=t2_scores_all,
        t2_times=t2_times,
        t2_subjects=np.array(t2_subjects, dtype=str),
        t2_mean_auc=t2_result['mean_auc'],
        t2_sem_auc=t2_result['sem_auc'],
        t2_t_vals=t2_result['t_vals'],
        t2_p_vals=t2_result['p_vals'],
        t2_p_two_sided=t2_result['p_two_sided'],
        t2_sig_mask_unc=t2_result['sig_mask_uncorrected'],
        t2_sig_mask=t2_result['sig_mask'],

        chance_level=CHANCE_LEVEL,
        p_threshold=p_threshold,
        min_sig_duration_s=min_sig_duration_s,
        alternative=alternative,
    )

    print(f'\nGroup npz saved to: {os.path.abspath(group_npz)}')
    print('DONE')