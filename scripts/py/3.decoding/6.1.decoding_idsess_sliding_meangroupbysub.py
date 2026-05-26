# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 20:40:47 2026

@author: WANGLIANGFU
"""

import os
import glob
import re
import numpy as np
import utils
from mne.stats import permutation_cluster_1samp_test

working_dir = os.path.join('..', '..', '..', 'results', 'decoding_idsess')
figures_dir = os.path.join('..', '..', '..', 'figures', 'decoding_idsess_mean')
os.makedirs(figures_dir, exist_ok=True)

def _safe_get(data, key):
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

def _same_times(t_ref, t_new, tol=1e-12):
    if t_ref is None or t_new is None:
        return True
    return (len(t_ref) == len(t_new)) and np.allclose(t_ref, t_new, atol=tol, rtol=0)

def _subject_from_path(fp):
    m = re.search(r'(sub\d+)', fp)
    return m.group(1) if m else None

def _to_curve(x, n_times):
    x = np.asarray(x)
    if x.ndim == 2:   # (n_splits, n_times)
        if x.shape[1] != n_times:
            return None
        return x.mean(axis=0)
    if x.ndim == 1:   # (n_times,)
        if x.shape[0] != n_times:
            return None
        return x
    return None

def _build_file_list_prefer_rs100(base_dir):
    """
    对每个原始 *_sliding_scores.npz：
    - 若存在对应 *_sliding_scores_rs100.npz，则使用 rs100
    - 否则使用原始文件
    """
    raw_files = sorted(glob.glob(os.path.join(base_dir, '**', '*_sliding_scores.npz'), recursive=True))
    # 排除 rs100 本身，避免重复匹配
    raw_files = [f for f in raw_files if not f.endswith('_sliding_scores_rs100.npz')]

    chosen = []
    for rf in raw_files:
        rs = rf.replace('_sliding_scores.npz', '_sliding_scores_rs100.npz')
        if os.path.exists(rs):
            chosen.append(rs)
        else:
            chosen.append(rf)

    n_rs = sum(f.endswith('_sliding_scores_rs100.npz') for f in chosen)
    n_raw = len(chosen) - n_rs
    print(f"[INFO] 合并读取文件数: {len(chosen)}")
    print(f"[INFO] 其中 rs100 数量: {n_rs}")
    print(f"[INFO] 原始文件数量: {n_raw}")
    return sorted(chosen)

# ===== 读取文件（修正后的核心）=====
files = _build_file_list_prefer_rs100(working_dir)

# 收集：按 subject 分组
times_t1, times_t2 = None, None
by_sub = {}  # by_sub[sub]['t1_acc'/'t1_conf'/'t2_acc'/'t2_conf'] = [curve, curve, ...]

for fp in files:
    sub = _subject_from_path(fp)
    if sub is None:
        print(f"[WARN] 无法识别被试，跳过: {fp}")
        continue

    if sub not in by_sub:
        by_sub[sub] = {'t1_acc': [], 't1_conf': [], 't2_acc': [], 't2_conf': []}

    try:
        data = np.load(fp, allow_pickle=True)

        t1 = _safe_get(data, 'times_t1')
        t2 = _safe_get(data, 'times_t2')

        if t1 is not None:
            if times_t1 is None:
                times_t1 = t1
            elif not _same_times(times_t1, t1):
                print(f"[WARN] times_t1 不一致，跳过文件: {fp}")
                data.close()
                continue

        if t2 is not None:
            if times_t2 is None:
                times_t2 = t2
            elif not _same_times(times_t2, t2):
                print(f"[WARN] times_t2 不一致，跳过文件: {fp}")
                data.close()
                continue

        s_t1_acc  = _safe_get(data, 'scores_t1_acc')
        s_t1_conf = _safe_get(data, 'scores_t1_conf')
        s_t2_acc  = _safe_get(data, 'scores_t2_acc')
        s_t2_conf = _safe_get(data, 'scores_t2_conf')

        if s_t1_acc is not None and times_t1 is not None:
            c = _to_curve(s_t1_acc, len(times_t1))
            if c is not None:
                by_sub[sub]['t1_acc'].append(c)

        if s_t1_conf is not None and times_t1 is not None:
            c = _to_curve(s_t1_conf, len(times_t1))
            if c is not None:
                by_sub[sub]['t1_conf'].append(c)

        if s_t2_acc is not None and times_t2 is not None:
            c = _to_curve(s_t2_acc, len(times_t2))
            if c is not None:
                by_sub[sub]['t2_acc'].append(c)

        if s_t2_conf is not None and times_t2 is not None:
            c = _to_curve(s_t2_conf, len(times_t2))
            if c is not None:
                by_sub[sub]['t2_conf'].append(c)

        data.close()

    except Exception as e:
        print(f"[WARN] 读取失败，跳过 {fp}: {e}")

def _build_subject_matrix(metric_key):
    """每个被试内先平均多个 trial，再做组统计。"""
    subj_mat = []
    used_subs = []
    trial_count = {}
    for sub in sorted(by_sub.keys()):
        curves = by_sub[sub][metric_key]
        if len(curves) == 0:
            continue
        arr = np.vstack(curves)           # (n_trials_available, n_times)
        subj_curve = arr.mean(axis=0)     # 被试内平均
        subj_mat.append(subj_curve)
        used_subs.append(sub)
        trial_count[sub] = arr.shape[0]

    if len(subj_mat) == 0:
        return None, [], {}
    return np.vstack(subj_mat), used_subs, trial_count

def summarize_and_plot_subject_level(metric_key, times, title_stub, fname_stub):
    if times is None:
        print(f"[WARN] {title_stub} times 缺失，跳过")
        return

    arr_subj, used_subs, trial_count = _build_subject_matrix(metric_key)
    if arr_subj is None or arr_subj.shape[0] < 2:
        n = 0 if arr_subj is None else arr_subj.shape[0]
        print(f"[WARN] {title_stub} 被试数不足（n_sub={n}），跳过")
        return

    mean = arr_subj.mean(axis=0)
    sem = arr_subj.std(axis=0, ddof=1) / np.sqrt(arr_subj.shape[0])

    cluster_stats = permutation_cluster_1samp_test(
        arr_subj - 0.5,
        tail=1,
        n_permutations=1000,
        seed=12345,
        n_jobs=1,
        out_type='mask',
        verbose=False
    )

    save_path = os.path.join(figures_dir, f"{fname_stub}.png")
    utils.plot_with_cluster_highlight(
        times=times,
        scores_mean=mean,
        scores_sem=sem,
        permutation_result=cluster_stats,
        title=f"{title_stub} (n_sub={arr_subj.shape[0]})",
        save_path=save_path,
        chance_level=0.5
    )
    print(f"[SAVE] {save_path}")
    print(f"[INFO] used subjects: {used_subs}")
    print(f"[INFO] trials per subject: {trial_count}")

summarize_and_plot_subject_level('t1_acc',  times_t1, "Task1 Resp - Accuracy (subject-level)",   "group_sub_t1_acc")
summarize_and_plot_subject_level('t1_conf', times_t1, "Task1 Resp - Confidence (subject-level)", "group_sub_t1_conf")
summarize_and_plot_subject_level('t2_acc',  times_t2, "Task2 Stim - Accuracy (subject-level)",   "group_sub_t2_acc")
summarize_and_plot_subject_level('t2_conf', times_t2, "Task2 Stim - Confidence (subject-level)", "group_sub_t2_conf")

print("汇总完成（subject-level）。")