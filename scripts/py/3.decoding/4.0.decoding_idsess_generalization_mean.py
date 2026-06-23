
# -*- coding: utf-8 -*-
"""
Group mean and cluster permutation for cross-task temporal generalization

读取单被试结果：
    results/cross_task_generalization/subXX/subXX_generalization_results.npz

每个 npz 中读取：
    scores_gen_acc
    scores_gen_conf
    scores_gen_conf_correct_trials
    times_train
    times_test

输出：
    figures/cross_task_generalization_mean/
    results/cross_task_generalization_mean/

功能：
1. 汇总所有被试的 cross-task temporal generalization 矩阵
2. 计算 group mean AUC
3. 对 AUC - 0.5 做 group-level one-sample cluster permutation test
4. 绘制 group mean temporal generalization 图
5. 黑色轮廓标记 cluster-corrected significant cluster
"""

import os
import re
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from mne.stats import permutation_cluster_1samp_test


# ============================================================
# Parameters
# ============================================================
CHANCE_LEVEL = 0.5
N_PERM = 500
ALPHA = 0.05
TAIL = 1  # 1 = one-tailed test, AUC > 0.5

VMIN = 0.4
VMAX = 0.6


# ============================================================
# Paths
# ============================================================
single_subject_results_root = os.path.join(
    '..',
    '..',
    '..',
    'results',
    'cross_task_generalization',
)

group_results_dir = os.path.join(
    '..',
    '..',
    '..',
    'results',
    'cross_task_generalization_mean',
)

figures_dir = os.path.join(
    '..',
    '..',
    '..',
    'figures',
    'cross_task_generalization_mean',
)

os.makedirs(group_results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)


# ============================================================
# Helper functions
# ============================================================
def _safe_get(data, key):
    """
    安全读取 npz key。
    """
    if key not in data:
        return None

    x = data[key]

    if isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        x = x.item()

    if x is None:
        return None

    x = np.asarray(x)

    if x.size == 0:
        return None

    return x


def _same_1d(a, b, tol=1e-12):
    """
    检查两个一维时间轴是否一致。
    """
    if a is None or b is None:
        return False

    if len(a) != len(b):
        return False

    return np.allclose(a, b, atol=tol, rtol=0)


def _subject_from_path(fp):
    """
    从路径中提取 subXX。
    """
    path_norm = fp.replace('\\', '/')
    m = re.search(r'/(sub\d+)/', path_norm)
    if m:
        return m.group(1)

    base = os.path.basename(fp)
    m = re.search(r'(sub\d+)', base)
    if m:
        return m.group(1)

    return None


def _sub_number(sub):
    """
    用于按被试编号排序。
    """
    m = re.search(r'\d+', sub)
    return int(m.group()) if m else 999999


def find_subject_npz_files(root):
    """
    搜索所有单被试 generalization npz。
    """
    pattern = os.path.join(
        root,
        'sub*',
        '*_generalization_results.npz',
    )

    files = sorted(glob(pattern))

    files = sorted(
        files,
        key=lambda fp: _sub_number(_subject_from_path(fp) or 'sub999999')
    )

    return files


def load_group_matrix(npz_files, score_key):
    """
    读取某一类 score matrix。

    参数：
        score_key:
            scores_gen_acc
            scores_gen_conf
            scores_gen_conf_correct_trials

    返回：
        mat_stack: shape = (n_subjects, n_train_times, n_test_times)
        times_train
        times_test
        used_subjects
        skipped_records
    """
    mats = []
    used_subjects = []
    skipped_records = []

    times_train_ref = None
    times_test_ref = None
    expected_shape = None

    for fp in npz_files:
        subject = _subject_from_path(fp)

        if subject is None:
            skipped_records.append({
                'subject': '',
                'file': fp,
                'reason': 'cannot extract subject name',
            })
            continue

        try:
            data = np.load(fp, allow_pickle=True)

            times_train = _safe_get(data, 'times_train')
            times_test = _safe_get(data, 'times_test')
            scores = _safe_get(data, score_key)

            data.close()

            if times_train is None or times_test is None:
                skipped_records.append({
                    'subject': subject,
                    'file': fp,
                    'reason': 'missing times_train or times_test',
                })
                continue

            if scores is None:
                skipped_records.append({
                    'subject': subject,
                    'file': fp,
                    'reason': f'missing score key: {score_key}',
                })
                continue

            if scores.ndim != 2:
                skipped_records.append({
                    'subject': subject,
                    'file': fp,
                    'reason': f'{score_key} is not 2D, shape={scores.shape}',
                })
                continue

            if times_train_ref is None:
                times_train_ref = times_train
                times_test_ref = times_test
                expected_shape = (len(times_train_ref), len(times_test_ref))

            if not _same_1d(times_train_ref, times_train):
                skipped_records.append({
                    'subject': subject,
                    'file': fp,
                    'reason': 'times_train mismatch',
                })
                continue

            if not _same_1d(times_test_ref, times_test):
                skipped_records.append({
                    'subject': subject,
                    'file': fp,
                    'reason': 'times_test mismatch',
                })
                continue

            if scores.shape != expected_shape:
                skipped_records.append({
                    'subject': subject,
                    'file': fp,
                    'reason': f'shape mismatch: {scores.shape} != {expected_shape}',
                })
                continue

            mats.append(scores)
            used_subjects.append(subject)

        except Exception as e:
            skipped_records.append({
                'subject': subject,
                'file': fp,
                'reason': repr(e),
            })

    if len(mats) == 0:
        return None, times_train_ref, times_test_ref, [], skipped_records

    order = np.argsort([_sub_number(s) for s in used_subjects])
    used_subjects = [used_subjects[i] for i in order]
    mats = [mats[i] for i in order]

    mat_stack = np.stack(mats, axis=0)

    return mat_stack, times_train_ref, times_test_ref, used_subjects, skipped_records


def cluster_perm_2d(mat_stack, chance_level=0.5, n_perm=500, alpha=0.05, tail=1):
    """
    对 2D temporal generalization matrix 做 group-level cluster permutation。

    mat_stack:
        shape = (n_subjects, n_train_times, n_test_times)

    检验：
        AUC > chance_level
    """
    n_subjects = mat_stack.shape[0]

    X = mat_stack - chance_level

    if tail == 1:
        # one-tailed threshold for AUC > chance
        threshold = stats.t.ppf(1 - alpha, df=n_subjects - 1)
    elif tail == -1:
        threshold = -stats.t.ppf(1 - alpha, df=n_subjects - 1)
    elif tail == 0:
        threshold = stats.t.ppf(1 - alpha / 2, df=n_subjects - 1)
    else:
        raise ValueError('tail must be -1, 0, or 1')

    print(f'  Cluster-forming threshold: {threshold:.4f}')

    T_obs, clusters, pvals, _ = permutation_cluster_1samp_test(
        X,
        threshold=threshold,
        n_permutations=n_perm,
        tail=tail,
        out_type='mask',
        n_jobs=1,
        verbose=False,
    )

    sig_mask = np.zeros_like(T_obs, dtype=bool)

    for cluster_mask, p in zip(clusters, pvals):
        if p <= alpha:
            sig_mask |= cluster_mask

    mean_mat = np.nanmean(mat_stack, axis=0)
    sem_mat = np.nanstd(mat_stack, axis=0, ddof=1) / np.sqrt(n_subjects)

    return {
        'mat_stack': mat_stack,
        'mean_mat': mean_mat,
        'sem_mat': sem_mat,
        'T_obs': T_obs,
        'clusters_pvals': pvals,
        'sig_mask': sig_mask,
        'n_subjects': n_subjects,
        'chance_level': chance_level,
        'n_perm': n_perm,
        'alpha': alpha,
        'tail': tail,
        'threshold': threshold,
    }


def plot_mean_with_contour(
    mean_mat,
    sig_mask,
    times_train,
    times_test,
    title,
    save_path,
    vmin=0.4,
    vmax=0.6,
):
    """
    绘制 group mean temporal generalization matrix。
    黑色实线轮廓表示 cluster-corrected significant clusters。
    黑色虚线表示 AUC = 0.5 等值线。
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(
        mean_mat,
        origin='lower',
        interpolation='lanczos',
        extent=[
            times_test[0],
            times_test[-1],
            times_train[0],
            times_train[-1],
        ],
        aspect='auto',
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
    )

    if sig_mask is not None and sig_mask.any():
        ax.contour(
            times_test,
            times_train,
            sig_mask.astype(int),
            levels=[0.5],
            colors='k',
            linewidths=1.2,
            alpha=0.95,
        )

    try:
        cs = ax.contour(
            times_test,
            times_train,
            mean_mat,
            levels=[0.5],
            colors='k',
            linewidths=0.6,
            linestyles='--',
        )
        ax.clabel(cs, fmt='%.2f', fontsize=8)
    except Exception:
        pass

    ax.axhline(0, color='k', lw=1)
    ax.axvline(0, color='k', lw=1)

    ax.set_xlabel('Testing time (s)')
    ax.set_ylabel('Training time (s)')
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('AUC')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'[SAVE] {os.path.abspath(save_path)}')


def save_subject_list(subjects, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for sub in subjects:
            f.write(f'{sub}\n')


def save_skipped_records(skipped, save_path):
    if len(skipped) == 0:
        df = pd.DataFrame(columns=['subject', 'file', 'reason'])
    else:
        df = pd.DataFrame(skipped)

    df.to_csv(save_path, index=False, encoding='utf-8-sig')


def save_matrix_summary_csv(result, times_train, times_test, save_path):
    """
    保存 group mean matrix 成长格式 csv。
    每一行是一个 train_time × test_time 点。
    """
    mean_mat = result['mean_mat']
    sem_mat = result['sem_mat']
    T_obs = result['T_obs']
    sig_mask = result['sig_mask']

    rows = []

    for i, tr_t in enumerate(times_train):
        for j, te_t in enumerate(times_test):
            rows.append({
                'train_time': tr_t,
                'test_time': te_t,
                'mean_auc': mean_mat[i, j],
                'sem_auc': sem_mat[i, j],
                't_value_vs_0.5': T_obs[i, j],
                'cluster_corrected_significant': int(sig_mask[i, j]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')


def run_one_group_analysis(
    npz_files,
    score_key,
    label_name,
    fig_name,
    result_prefix,
):
    """
    针对一种矩阵做 group 平均和 cluster permutation。
    """
    print('\n' + '=' * 80)
    print(f'GROUP ANALYSIS: {label_name}')
    print('=' * 80)

    mat_stack, times_train, times_test, used_subjects, skipped = load_group_matrix(
        npz_files=npz_files,
        score_key=score_key,
    )

    skipped_path = os.path.join(
        group_results_dir,
        f'{result_prefix}_skipped.csv',
    )
    save_skipped_records(skipped, skipped_path)

    if mat_stack is None or mat_stack.shape[0] < 2:
        print(f'[WARN] {label_name}: usable subjects < 2')
        print(f'[WARN] skipped records saved to: {os.path.abspath(skipped_path)}')
        return None

    print(f'[INFO] used subjects ({len(used_subjects)}): {used_subjects}')
    print(f'[INFO] mat_stack shape: {mat_stack.shape}')

    result = cluster_perm_2d(
        mat_stack=mat_stack,
        chance_level=CHANCE_LEVEL,
        n_perm=N_PERM,
        alpha=ALPHA,
        tail=TAIL,
    )

    print(f'[INFO] significant cluster points: {int(result["sig_mask"].sum())}')
    if len(result['clusters_pvals']) > 0:
        print(f'[INFO] min cluster p: {np.min(result["clusters_pvals"]):.6f}')
    else:
        print('[INFO] no clusters found')

    fig_path = os.path.join(figures_dir, fig_name)

    plot_mean_with_contour(
        mean_mat=result['mean_mat'],
        sig_mask=result['sig_mask'],
        times_train=times_train,
        times_test=times_test,
        title=f'{label_name} (n={result["n_subjects"]})',
        save_path=fig_path,
        vmin=VMIN,
        vmax=VMAX,
    )

    subject_txt = os.path.join(
        group_results_dir,
        f'{result_prefix}_subjects.txt',
    )
    save_subject_list(used_subjects, subject_txt)

    summary_csv = os.path.join(
        group_results_dir,
        f'{result_prefix}_matrix_summary.csv',
    )
    save_matrix_summary_csv(
        result=result,
        times_train=times_train,
        times_test=times_test,
        save_path=summary_csv,
    )

    npz_path = os.path.join(
        group_results_dir,
        f'{result_prefix}_group_results.npz',
    )

    np.savez(
        npz_path,
        mat_stack=mat_stack,
        mean_mat=result['mean_mat'],
        sem_mat=result['sem_mat'],
        T_obs=result['T_obs'],
        clusters_pvals=result['clusters_pvals'],
        sig_mask=result['sig_mask'],
        times_train=times_train,
        times_test=times_test,
        subjects=np.array(used_subjects, dtype=str),
        chance_level=CHANCE_LEVEL,
        n_perm=N_PERM,
        alpha=ALPHA,
        tail=TAIL,
        threshold=result['threshold'],
    )

    print(f'[SAVE] subjects: {os.path.abspath(subject_txt)}')
    print(f'[SAVE] summary csv: {os.path.abspath(summary_csv)}')
    print(f'[SAVE] group npz: {os.path.abspath(npz_path)}')
    print(f'[SAVE] skipped csv: {os.path.abspath(skipped_path)}')

    return result


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print('================ GROUP CROSS-TASK GENERALIZATION MEAN ================')
    print(f'single_subject_results_root: {os.path.abspath(single_subject_results_root)}')
    print(f'group_results_dir: {os.path.abspath(group_results_dir)}')
    print(f'figures_dir: {os.path.abspath(figures_dir)}')
    print(f'N_PERM = {N_PERM}')
    print(f'ALPHA = {ALPHA}')
    print(f'TAIL = {TAIL}')

    npz_files = find_subject_npz_files(single_subject_results_root)

    if len(npz_files) == 0:
        raise RuntimeError(
            f'没有找到单被试 npz 文件: {os.path.abspath(single_subject_results_root)}'
        )

    print(f'\nFound {len(npz_files)} subject npz files:')
    for fp in npz_files:
        print(' ', fp)

    # ============================================================
    # 1) Accuracy generalization
    # ============================================================
    run_one_group_analysis(
        npz_files=npz_files,
        score_key='scores_gen_acc',
        label_name='Mean Gen ACC',
        fig_name='mean_gen_acc.png',
        result_prefix='mean_gen_acc',
    )

    # ============================================================
    # 2) Confidence / abs generalization, all trials
    # ============================================================
    run_one_group_analysis(
        npz_files=npz_files,
        score_key='scores_gen_conf',
        label_name='Mean Gen CONF/ABS all trials',
        fig_name='mean_gen_conf_abs.png',
        result_prefix='mean_gen_conf_abs',
    )

    # ============================================================
    # 3) Confidence / abs generalization, correct trials only
    # ============================================================
    run_one_group_analysis(
        npz_files=npz_files,
        score_key='scores_gen_conf_correct_trials',
        label_name='Mean Gen CONF/ABS correct trials only',
        fig_name='mean_gen_conf_abs_correct_trials.png',
        result_prefix='mean_gen_conf_abs_correct_trials',
    )

    print('\n汇总完成。')

