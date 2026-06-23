# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 11:56:00 2026

@author: WANGLIANGFU
"""

import os
import glob
import re
import numpy as np
import utils
from mne.stats import permutation_cluster_1samp_test

##############################
# 路径配置
##############################
working_dir = os.path.join('..', '..', 'results', 'decoding_idsess')
figures_dir = os.path.join('..', '..', 'figures', 'decoding_idsess_mean')
os.makedirs(figures_dir, exist_ok=True)

# 只统计这些被试 + trial
TARGET_SUBJECTS = {f"sub{i:02d}" for i in range(18, 29)}
TARGET_TRIAL = "1"

##############################
# 汇总容器
##############################
t1_acc_list, t1_conf_list = [], []
t2_acc_list, t2_conf_list = [], []
times_t1, times_t2 = None, None

##############################
# 工具函数
##############################
def _safe_get(data, key):
    """从npz读取并处理 object/None。"""
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
    if len(t_ref) != len(t_new):
        return False
    return np.allclose(t_ref, t_new, atol=tol, rtol=0)

def _extract_sub_trial_from_path(fp):
    """
    从路径里提取 .../decoding_idsess/subXX/trial/.../subXX_trial_sliding_scores.npz
    """
    norm = fp.replace("\\", "/")
    m = re.search(r"/decoding_idsess/(sub\d{2})/([^/]+)/", norm)
    if m:
        return m.group(1), m.group(2)
    return None, None

##############################
# 扫描文件
##############################
pattern = os.path.join(working_dir, '**', '*_sliding_scores.npz')
all_files = sorted(glob.glob(pattern, recursive=True))
print(f"找到 {len(all_files)} 个 npz 文件（全量）")

# 只留目标被试+trial
files = []
for fp in all_files:
    sub, tr = _extract_sub_trial_from_path(fp)
    if (sub in TARGET_SUBJECTS) and (str(tr) == TARGET_TRIAL):
        files.append(fp)

print(f"筛选后 {len(files)} 个 npz 文件（subjects={sorted(TARGET_SUBJECTS)}, trial={TARGET_TRIAL}）")

for fp in files:
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

        if s_t1_acc is not None:
            t1_acc_list.append(s_t1_acc)
        if s_t1_conf is not None:
            t1_conf_list.append(s_t1_conf)
        if s_t2_acc is not None:
            t2_acc_list.append(s_t2_acc)
        if s_t2_conf is not None:
            t2_conf_list.append(s_t2_conf)

        data.close()
    except Exception as e:
        print(f"[WARN] 读取失败，跳过 {fp}: {e}")

def summarize_and_plot(arr_list, times, title_stub, fname_stub):
    """
    arr_list 中每个元素可能是：
      - (n_splits, n_times): 单文件内CV重复
      - (n_times,): 已经是单条曲线
    统计策略：
      1) 每个文件先聚合成 1 条曲线（对 splits 取均值）
      2) 再在文件间做 cluster permutation
    """
    if (not arr_list) or (times is None):
        print(f"[WARN] {title_stub} 没有有效数据，跳过")
        return

    subj_curves = []
    skipped = 0

    for x in arr_list:
        try:
            x = np.asarray(x)

            if x.ndim == 2:
                if x.shape[1] != len(times):
                    skipped += 1
                    continue
                x = x.mean(axis=0)

            elif x.ndim == 1:
                if x.shape[0] != len(times):
                    skipped += 1
                    continue
            else:
                skipped += 1
                continue

            subj_curves.append(x)
        except Exception:
            skipped += 1

    if len(subj_curves) < 2:
        print(f"[WARN] {title_stub} 有效样本不足（n={len(subj_curves)}），跳过")
        return

    arr_subj = np.vstack(subj_curves)
    mean = arr_subj.mean(axis=0)
    sem  = arr_subj.std(axis=0, ddof=1) / np.sqrt(arr_subj.shape[0])

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
        title=f"{title_stub} (n_obs={arr_subj.shape[0]}, skipped={skipped}, trial={TARGET_TRIAL})",
        save_path=save_path,
        chance_level=0.5
    )
    print(f"[SAVE] {save_path}")

summarize_and_plot(t1_acc_list,  times_t1, "Task1 Resp - Accuracy",   "group_t1_acc_trial1")
summarize_and_plot(t1_conf_list, times_t1, "Task1 Resp - Confidence", "group_t1_conf_trial1")
summarize_and_plot(t2_acc_list,  times_t2, "Task2 Stim - Accuracy",   "group_t2_acc_trial1")
summarize_and_plot(t2_conf_list, times_t2, "Task2 Stim - Confidence", "group_t2_conf_trial1")

print("汇总完成。")