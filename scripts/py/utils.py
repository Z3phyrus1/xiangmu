# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 19:23:06 2025

@author: DELL
"""

from io import StringIO
import glob
import matplotlib.pyplot as plt
import os
import numpy as np
import mne
from autoreject import AutoReject
import common_settings as CS
from mne.decoding import SlidingEstimator, cross_val_multiscore, Scaler, Vectorizer, LinearModel, GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.linear_model import LogisticRegression
from mne.decoding import LinearModel, get_coef

def get_session_number(filename):
    import re
    match = re.search(r'sess(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def determine_event_id(subject, session_num):
    sub_num = int(subject.replace('sub', ''))
    if (sub_num + session_num) % 2 == 0:
        event_id_task1 = CS.event_id_line
        event_id_task2 = CS.event_id_dot
    else:
        event_id_task1 = CS.event_id_dot
        event_id_task2 = CS.event_id_line
    return {'task1': event_id_task1, 'task2': event_id_task2}


# 原eog = 0.3 muscle = 0.9
def run_ica_cleaning(epochs, picks, figures_dir, session_num, task_name,
                     threshold_eog=0.3, threshold_muscle=0.9):

    cov = mne.compute_covariance(epochs, tmin=epochs.tmin, tmax=epochs.tmax)
    ica = mne.preprocessing.ICA(noise_cov=cov, **CS.ica_args)
    ica.fit(epochs, picks=picks)

    eog_idx, _ = ica.find_bads_eog(
        epochs,
        ch_name=['Fp1', 'Fp2'],
        threshold=threshold_eog, measure='correlation'
    )
    muscle_idx, _ = ica.find_bads_muscle(epochs, threshold=threshold_muscle)

    ica.exclude.extend(eog_idx + muscle_idx)
    ica.exclude = list(set(ica.exclude))

    if len(ica.exclude) > 0:
        figures = ica.plot_properties(epochs, picks=ica.exclude)
        for fig, idx in zip(figures, ica.exclude):
            fig.savefig(os.path.join(figures_dir, f'session{session_num}_{task_name}_ICA{idx}.png'),
                        dpi=300, bbox_inches='tight')
    else:
        print(f"{task_name}: 本次没有需要排除的 ICA 成分，跳过成分可视化")

    clean_epochs = ica.apply(epochs)
    return clean_epochs


def run_autoreject(epochs, picks, figures_dir, session_num, task_name):
    ar = AutoReject(picks=picks, **CS.autoreject_args)
    ar.fit(epochs)
    ar_epochs, reject_log = ar.transform(epochs, return_log=True)

    fig_reject = reject_log.plot('horizontal')
    fig_reject.savefig(os.path.join(figures_dir, f'session{session_num}_{task_name}_reject.png'),
                       dpi=300, bbox_inches='tight')

    return ar_epochs, reject_log


def save_epochs(clean_epochs, ar_epochs, reject_log, results_dir, file_base):
    clean_epochs.save(os.path.join(results_dir, f'clean_epochs_{file_base}-epo.fif'), overwrite=True)
    ar_epochs.save(os.path.join(results_dir, f'AR_epochs_{file_base}-epo.fif'), overwrite=True)
    reject_log.save(os.path.join(results_dir, f'reject_log_{file_base}.npz'), overwrite=True)


def plot_and_save_joint(epochs, event_id, figures_dir, session_num, task_name):
    for key in event_id.keys():
        fig = epochs[key].average().plot_joint()
        fig.savefig(os.path.join(figures_dir, f'session{session_num}_{task_name}_clean_{key}.png'),
                   dpi=300, bbox_inches='tight')


def concatenate_baseline(epochs_data, baseline_data):
    # trial数不一致时截断，避免报错
    if baseline_data.shape[0] != epochs_data.shape[0]:
        n_min = min(baseline_data.shape[0], epochs_data.shape[0])
        print(f"[WARN] concatenate_baseline: baseline={baseline_data.shape[0]}, epochs={epochs_data.shape[0]} -> 截断到 {n_min}")
        baseline_data = baseline_data[:n_min]
        epochs_data = epochs_data[:n_min]

    # 通道数必须一致
    if baseline_data.shape[1] != epochs_data.shape[1]:
        raise ValueError(f"通道数不一致: baseline={baseline_data.shape[1]}, epochs={epochs_data.shape[1]}")

    return np.concatenate([baseline_data, epochs_data], axis=2)


def get_event_indices(events, events_per_trial, position, num_trials):
    indices = [trial_idx * events_per_trial + position
               for trial_idx in range(num_trials)
               if trial_idx * events_per_trial + position < len(events)]
    return indices


def compute_accuracy_labels_from_raw_events(raw_events, events_per_trial=6):
    task1_accuracy = []
    task2_accuracy = []

    # 跳过第一个 New Segment 事件
    num_trials = (len(raw_events) - 1) // events_per_trial

    for trial_idx in range(num_trials):
        base = 1 + trial_idx * events_per_trial

        if base + 5 >= len(raw_events):
            continue

        # Task1 事件和 Response1 事件
        task1_event = raw_events[base, -1]          # Task1 呈现
        response1_event = raw_events[base + 2, -1]  # Response1

        # Task2 事件和 Response2 事件
        task2_event = raw_events[base + 3, -1]      # Task2 呈现
        response2_event = raw_events[base + 5, -1]  # Response2

        # 计算正确性：事件个位数相同表示正确
        task1_correct = 1 if (task1_event % 10) == (response1_event % 10) else 0
        task2_correct = 1 if (task2_event % 10) == (response2_event % 10) else 0

        task1_accuracy.append(task1_correct)
        task2_accuracy.append(task2_correct)

    return task1_accuracy, task2_accuracy


def get_task1_accuracy_labels(events, events_per_trial):
    labels = []
    num_trials = (len(events) - 1) // events_per_trial

    for trial_idx in range(num_trials):
        base = 1 + trial_idx * events_per_trial

        if base + 2 >= len(events):
            continue

        task1_event = events[base, -1]
        response1_event = events[base + 2, -1]

        if task1_event % 10 == response1_event % 10:
            labels.append(1)
        else:
            labels.append(0)

    return np.array(labels)


# ============================================================
# 解码器创建函数
# ============================================================
def make_decoder_pipeline():
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            **CS.classifier_args
        )
    )
    return clf


def make_sliding_estimator():
    clf = make_decoder_pipeline()
    sliding_estimator = SlidingEstimator(
        clf,
        **CS.temporal_decoding_args
    )
    return sliding_estimator


def make_generalizing_estimator():
    clf = make_decoder_pipeline()
    gen_estimator = GeneralizingEstimator(
        clf,
        **CS.temporal_generalizing_args
    )
    return gen_estimator


# ============================================================
# 解码运行函数
# ============================================================
def run_generalization_decoding(X, y, epochs_info=None):
    gen_estimator = make_generalizing_estimator()
    cv = StratifiedKFold(
        n_splits=CS.cv_args['n_splits'],
        shuffle=True,
        random_state=CS.cv_args['random_state']
    )
    scores = cross_val_multiscore(
        gen_estimator, X, y,
        cv=cv,
        n_jobs=CS.temporal_generalizing_args.get('n_jobs', 1)
    )
    return scores


def compute_accuracy_from_epochs(epochs_task, epochs_response):
    task_events = epochs_task.events[:, 2]
    response_events = epochs_response.events[:, 2]

    if len(task_events) != len(response_events):
        print(f"[WARN] 事件数量不匹配! task: {len(task_events)}, response: {len(response_events)}")

    accuracy = np.array([
        1 if (t % 10) == (r % 10) else 0
        for t, r in zip(task_events, response_events)
    ])
    return accuracy


# ============================================================
# 绘图函数
# ============================================================
def plot_time_decoding(times, scores, title, save_path=None, chance_level=0.5):
    plt.figure(figsize=(10, 5))
    plt.plot(times, scores, label='Decoding score (AUC)', linewidth=2)
    plt.axhline(chance_level, color='k', linestyle='--', label='Chance level')
    plt.axvline(0, color='k', linestyle='-', label='Stimulus onset')
    plt.fill_between(times, chance_level, scores, where=(scores > chance_level),
                     alpha=0.3, color='green')
    plt.fill_between(times, chance_level, scores, where=(scores < chance_level),
                     alpha=0.3, color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('AUC')
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# === 保存试次剔除日志 ===
def save_preprocessing_log(subject, file_identifier, task_name, total_trials,
                           ica_epochs, ar_epochs, output_dir):
    current_status = []

    # 1) ICA阶段（修复了原来 for...else 误写）
    if ica_epochs.drop_log is not None:
        for d in ica_epochs.drop_log:
            # 只把真实drop记为Dropped_ICA；IGNORED不算
            if len(d) == 0 or (len(d) == 1 and d[0] == 'IGNORED'):
                current_status.append("OK")
            else:
                current_status.append("Dropped_ICA")
    else:
        # 如果没有 drop_log，默认全是 OK
        current_status = ["OK"] * len(ica_epochs)

    # 2) AutoReject阶段
    ok_indices = [i for i, s in enumerate(current_status) if s == "OK"]

    if ar_epochs.drop_log is not None:
        limit = min(len(ok_indices), len(ar_epochs.drop_log))

        for k in range(limit):
            original_idx = ok_indices[k]
            d = ar_epochs.drop_log[k]
            # 同样忽略 IGNORED
            if len(d) > 0 and not (len(d) == 1 and d[0] == 'IGNORED'):
                current_status[original_idx] = "Dropped_AutoReject"

    # 3) 保存为 CSV
    df_log = pd.DataFrame({
        'Subject': subject,
        'File': file_identifier,
        'Task': task_name,
        'Trial_ID': range(len(current_status)),
        'Status': current_status
    })

    log_name = f"{subject}_{file_identifier}_{task_name}_drop_log.csv"
    df_log.to_csv(os.path.join(output_dir, log_name), index=False)
    print(f"  --> 日志已保存: {log_name}")


def plot_pre_ica_erp(epochs, event_id, figures_dir, session_num, task_name, subject, file_stem):
    """
    在ICA处理前绘制原始ERP图
    """
    print(f"  绘制ICA前ERP图...")

    pre_ica_fig_dir = os.path.join(figures_dir, 'pre_ICA_ERPs')
    if not os.path.exists(pre_ica_fig_dir):
        os.makedirs(pre_ica_fig_dir)

    epochs_copy = epochs.copy()
    epochs_copy.apply_baseline(CS.baseline_task1)

    fig_erp = epochs_copy.plot_image(
        picks=['Fz', 'Cz', 'Pz', 'Oz'],
        combine='mean',
        show=False
    )

    erp_fig_path = os.path.join(pre_ica_fig_dir,
                                f"{subject}_{file_stem}_{task_name}_pre_ICA_ERP.png")
    fig_erp[0].savefig(erp_fig_path, dpi=300, bbox_inches='tight')
    print(f"    已保存: {erp_fig_path}")

    fig_all = epochs_copy.average().plot(spatial_colors=True, show=False)
    all_fig_path = os.path.join(pre_ica_fig_dir,
                                f"{subject}_{file_stem}_{task_name}_pre_ICA_allchannels.png")
    fig_all.savefig(all_fig_path, dpi=300, bbox_inches='tight')
    print(f"    已保存: {all_fig_path}")

    evoked = epochs_copy.average()

    fig_topo1 = evoked.plot_topomap(times='auto', show=False)
    topo_fig_path1 = os.path.join(pre_ica_fig_dir,
                                  f"{subject}_{file_stem}_{task_name}_pre_ICA_topomap_auto.png")
    fig_topo1.savefig(topo_fig_path1, dpi=300, bbox_inches='tight')
    print(f"    已保存: {topo_fig_path1}")

    tmin, tmax = epochs_copy.tmin, epochs_copy.tmax
    print(f"    实际时间范围: [{tmin:.2f}, {tmax:.2f}] 秒")

    custom_times = np.arange(max(tmin, -0.3), min(tmax, 0.3), 0.1)
    if len(custom_times) > 0:
        fig_topo2 = evoked.plot_topomap(times=custom_times, show=False)
        topo_fig_path2 = os.path.join(pre_ica_fig_dir,
                                      f"{subject}_{file_stem}_{task_name}_pre_ICA_topomap_custom.png")
        fig_topo2.savefig(topo_fig_path2, dpi=300, bbox_inches='tight')
        print(f"    已保存: {topo_fig_path2}")
    else:
        print("    警告：没有可用的自定义时间点在地形图范围内")

    n_times = 5
    times_equally_spaced = np.linspace(tmin + 0.1, tmax - 0.1, n_times)
    times_equally_spaced = times_equally_spaced[(times_equally_spaced >= tmin) & (times_equally_spaced <= tmax)]

    if len(times_equally_spaced) > 0:
        fig_topo3 = evoked.plot_topomap(times=times_equally_spaced, show=False)
        topo_fig_path3 = os.path.join(pre_ica_fig_dir,
                                      f"{subject}_{file_stem}_{task_name}_pre_ICA_topomap_equally_spaced.png")
        fig_topo3.savefig(topo_fig_path3, dpi=300, bbox_inches='tight')
        print(f"    已保存: {topo_fig_path3}")

    plt.close('all')
    return pre_ica_fig_dir


def plot_post_autoreject_erp(ar_epochs, event_id, figures_dir, session_num, task_name, subject, file_stem):
    """
    在 AutoReject 处理后绘制 ERP 图
    """
    print(f"  绘制 AutoReject 后 ERP 图...")

    post_ar_fig_dir = os.path.join(figures_dir, 'post_AutoReject_ERPs')
    if not os.path.exists(post_ar_fig_dir):
        os.makedirs(post_ar_fig_dir)

    epochs_copy = ar_epochs.copy()
    # 保持你原行为：统一 task1 baseline（不改）
    epochs_copy.apply_baseline(CS.baseline_task1)

    fig_erp = epochs_copy.plot_image(
        picks=['Fz', 'Cz', 'Pz', 'Oz'],
        combine='mean',
        show=False,
        title=f'{subject} {file_stem} Task{task_name} - AutoReject后 ERP'
    )

    erp_fig_path = os.path.join(post_ar_fig_dir,
                                f"{subject}_{file_stem}_{task_name}_POST_AR_ERP.png")
    fig_erp[0].savefig(erp_fig_path, dpi=300, bbox_inches='tight')
    print(f"    已保存:  {erp_fig_path}")

    evoked = epochs_copy.average()
    fig_all = evoked.plot(spatial_colors=True, show=False)
    fig_all.suptitle(f'{subject} {file_stem} Task{task_name} - AutoReject后 全通道 ERP',
                     fontsize=14, fontweight='bold')
    all_fig_path = os.path.join(post_ar_fig_dir,
                                f"{subject}_{file_stem}_{task_name}_POST_AR_allchannels.png")
    fig_all.savefig(all_fig_path, dpi=300, bbox_inches='tight')
    print(f"    已保存:  {all_fig_path}")

    fig_topo1 = evoked.plot_topomap(times='auto', show=False)
    fig_topo1.suptitle(f'{subject} {file_stem} Task{task_name} - AutoReject后 地形图（自动时间点）',
                       fontsize=12, fontweight='bold')
    topo_fig_path1 = os.path.join(post_ar_fig_dir,
                                  f"{subject}_{file_stem}_{task_name}_POST_AR_topomap_auto.png")
    fig_topo1.savefig(topo_fig_path1, dpi=300, bbox_inches='tight')
    print(f"    已保存: {topo_fig_path1}")

    tmin, tmax = epochs_copy.tmin, epochs_copy.tmax
    n_times = 5
    times_equally_spaced = np.linspace(tmin + 0.1, tmax - 0.1, n_times)
    times_equally_spaced = times_equally_spaced[(times_equally_spaced >= tmin) & (times_equally_spaced <= tmax)]

    if len(times_equally_spaced) > 0:
        fig_topo2 = evoked.plot_topomap(times=times_equally_spaced, show=False)
        fig_topo2.suptitle(f'{subject} {file_stem} Task{task_name} - AutoReject后 地形图（等间隔时间点）',
                           fontsize=12, fontweight='bold')
        topo_fig_path2 = os.path.join(post_ar_fig_dir,
                                      f"{subject}_{file_stem}_{task_name}_POST_AR_topomap_equally_spaced.png")
        fig_topo2.savefig(topo_fig_path2, dpi=300, bbox_inches='tight')
        print(f"    已保存: {topo_fig_path2}")

    if len(event_id) > 1:
        fig_compare, axes = plt.subplots(1, len(event_id), figsize=(15, 4))
        for idx, (key, _) in enumerate(event_id.items()):
            evoked_cond = epochs_copy[key].average()
            evoked_cond.plot(spatial_colors=True, show=False, axes=axes[idx] if len(event_id) > 1 else axes)
            axes[idx].set_title(f'{key}', fontweight='bold')

        fig_compare.suptitle(f'{subject} {file_stem} Task{task_name} - AutoReject后 条件对比',
                             fontsize=14, fontweight='bold')
        compare_fig_path = os.path.join(post_ar_fig_dir,
                                        f"{subject}_{file_stem}_{task_name}_POST_AR_conditions.png")
        fig_compare.savefig(compare_fig_path, dpi=300, bbox_inches='tight')
        print(f"    已保存:  {compare_fig_path}")

    plt.close('all')
    return post_ar_fig_dir


# ============================================================
# 获取 W 阈值（兼容 w1/w）
# ============================================================
def get_idsess_w_threshold(fit_par_path, sub_id):
    try:
        with open(fit_par_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        cleaned = ''.join([ln.replace('"', '') for ln in lines])

        df = pd.read_table(
            StringIO(cleaned),
            sep=r'\s+',
            engine='python',
            dtype=str,
            keep_default_na=False
        )

        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        df.columns = df.columns.str.strip()

        id_prefix = df['id'].astype(str).str.split('_').str[0]
        row = df[(df['model'] == 'sub2') & (id_prefix == str(int(sub_id)))]
        if row.empty:
            print(f"[DEBUG] 没有找到 model=sub2 且 id前缀={sub_id}")
            return None

        # 兼容 w1/w
        if 'w1' in row.columns:
            w_col = 'w1'
        elif 'w' in row.columns:
            w_col = 'w'
        else:
            print("[DEBUG] fit_par里没有 w1/w 列")
            return None

        w_str = row[w_col].iloc[0]
        if w_str in ['', 'NA', 'nan', 'NaN']:
            print(f"[DEBUG] 找到了行，但 {w_col} 是空/NA")
            return None

        return float(w_str)

    except Exception as e:
        print(f"[get_idsess_w_threshold ERROR] {e}")
        return None


def plot_temporal_generalization(scores, times_train, times_test, title, save_path, vmin=0.4, vmax=0.6):
    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(
        scores,
        interpolation='lanczos',
        origin='lower',
        cmap='RdBu_r',
        extent=[times_test[0], times_test[-1], times_train[0], times_train[-1]],
        vmin=vmin,
        vmax=vmax,
        aspect='auto'
    )

    ax.axhline(0, color='k', linestyle='-', linewidth=1)
    ax.axvline(0, color='k', linestyle='-', linewidth=1)

    ax.set_xlabel('Testing Time - Task2 Stimulus (s)', fontsize=12)
    ax.set_ylabel('Training Time - Task1 Response (s)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax, label='AUC')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图形已保存至: {save_path}")
    plt.show()

    return fig


def plot_temporal_generalization_with_contour(scores, times_train, times_test,
                                              title, save_path,
                                              vmin=0.4, vmax=0.6,
                                              contour_levels=None):
    if contour_levels is None:
        contour_levels = np.linspace(vmin, vmax, 6)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(scores, origin='lower', interpolation='lanczos',
                   extent=[times_test[0], times_test[-1], times_train[0], times_train[-1]],
                   aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)

    cs = ax.contour(times_test, times_train, scores, levels=contour_levels,
                    colors='k', linewidths=0.8, alpha=0.8)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

    ax.axhline(0, color='k', lw=1)
    ax.axvline(0, color='k', lw=1)
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='AUC')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVE] {save_path}")


# ============================================================
# 绘制带置换检验聚类高亮的图
# ============================================================
def plot_with_cluster_highlight(
    times,
    scores_mean,
    scores_sem,
    permutation_result,
    title,
    save_path=None,
    chance_level=0.5,
    alpha=0.05,
):
    # 解析 permutation_result
    if len(permutation_result) == 4:
        t_vals, clusters, cluster_p_vals, _ = permutation_result
    elif len(permutation_result) == 3:
        clusters, cluster_p_vals, _ = permutation_result
        t_vals = None
    else:
        raise ValueError("permutation_result 必须是长度 3 或 4 的 tuple")

    if clusters is None:
        clusters = []
    if cluster_p_vals is None:
        cluster_p_vals = np.array([])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, scores_mean, color='C0', lw=2, label='AUC')
    ax.fill_between(times, scores_mean - scores_sem, scores_mean + scores_sem,
                    color='C0', alpha=0.25, linewidth=0)
    ax.axhline(chance_level, color='k', linestyle='--', lw=1, label='Chance')
    ax.axvline(0, color='k', linestyle='-', lw=1, alpha=0.6)

    def _cluster_to_mask(clu, n_t):
        arr = np.asarray(clu)
        if arr.dtype == bool and arr.ndim == 1 and arr.shape[0] == n_t:
            return arr.copy()

        if isinstance(clu, (tuple, list)):
            if len(clu) == 1:
                first = clu[0]
                if isinstance(first, slice):
                    m = np.zeros(n_t, dtype=bool)
                    m[first] = True
                    return m
                idx = np.asarray(first)
                if idx.ndim == 1 and np.issubdtype(idx.dtype, np.integer):
                    m = np.zeros(n_t, dtype=bool)
                    idx = idx[(idx >= 0) & (idx < n_t)]
                    m[idx] = True
                    return m
            try:
                idx_last = np.asarray(clu[-1])
                if idx_last.ndim == 1 and np.issubdtype(idx_last.dtype, np.integer):
                    m = np.zeros(n_t, dtype=bool)
                    idx_last = idx_last[(idx_last >= 0) & (idx_last < n_t)]
                    m[idx_last] = True
                    return m
            except Exception:
                pass

        return np.zeros(n_t, dtype=bool)

    n_times = len(times)
    sig_count = 0

    # 只按 p<alpha 画高亮（去掉以前“p空也画”的调试分支）
    n_iter = min(len(clusters), len(cluster_p_vals))
    for i in range(n_iter):
        p_val = cluster_p_vals[i]
        if p_val < alpha:
            mask = _cluster_to_mask(clusters[i], n_times)
            if np.any(mask):
                ax.axvspan(times[mask][0], times[mask][-1], color='orange', alpha=0.25)
                sig_count += 1

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Decoding performance (AUC)")
    ax.set_title(title)
    ax.set_ylim(bottom=min(chance_level - 0.05, np.min(scores_mean - scores_sem) - 0.02))
    ax.legend(loc='best')
    ax.grid(alpha=0.2)

    print(f"[plot_with_cluster_highlight] total clusters={len(clusters)}, highlighted={sig_count}")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  图形已保存至: {save_path}")

    plt.show()


def read_concat(files, baseline, crop_tmin, crop_tmax):
    eps = [mne.read_epochs(f, verbose=False) for f in files]
    for e in eps:
        e.apply_baseline(baseline)
    epochs = mne.concatenate_epochs(eps)
    epochs.crop(crop_tmin, crop_tmax).resample(500)
    return epochs

def plot_significant_window_patterns(
    X, y, times, info, permutation_result,
    title_prefix, save_path_prefix,
    alpha=0.05
):
    """
    - X: shape (n_epochs, n_channels, n_times)
    - y: 分类标签
    - times: 时间轴
    - info: mne.Info（要与X通道对应）
    - permutation_result: mne.stats.permutation_cluster_1samp_test返回结果
    - title_prefix & save_path_prefix: 图标题和文件名前缀(不带后缀)
    """

    # cluster结果处理
    if len(permutation_result) == 4:  # mne≥1.4
        _, clusters, p_vals, _ = permutation_result
    else:                             # mne<1.4
        clusters, p_vals, _ = permutation_result

    sig_masks = []
    for clu, p in zip(clusters, p_vals):
        if p < alpha:
            mask = np.zeros(len(times), dtype=bool)
            if isinstance(clu, tuple) and len(clu) == 1:
                idx = clu[0]
                # 单向量或slice
                if isinstance(idx, slice):
                    mask[idx] = True
                else:
                    mask[np.asarray(idx)] = True
            else:
                arr = np.asarray(clu)
                if arr.dtype == bool and arr.shape[0] == len(times):
                    mask = arr.copy()
            if mask.any():
                sig_masks.append(mask)

    if len(sig_masks) == 0:
        print(f"[INFO] {title_prefix}: 没有显著时间窗，跳过拓扑图绘制")
        return

    # 合并所有显著点（你也可每个cluster单独画）
    sig_mask_all = np.any(np.vstack(sig_masks), axis=0)
    sig_idx = np.where(sig_mask_all)[0]
    t_start, t_end = times[sig_idx[0]], times[sig_idx[-1]]
    print(f"[INFO] {title_prefix}: 显著窗 {t_start:.3f} ~ {t_end:.3f} s")

    # 在显著时间窗内做平均，拟合线性模型
    X_sig = X[:, :, sig_mask_all].mean(axis=2)

    clf = make_pipeline(
        StandardScaler(),
        LinearModel(LogisticRegression(solver='liblinear'))
    )
    clf.fit(X_sig, y)
    # 提取patterns: 会自动逆变换回sensor空间
    patterns = get_coef(clf, 'patterns_', inverse_transform=True)  # (n_channels,)

    # 强制转shape
    patterns = patterns[:, np.newaxis]  # (n_channels, 1)

    evk = mne.EvokedArray(patterns, info=info, tmin=0.0)
    fig = evk.plot_topomap(
        times=[0],
        scalings=1,
        time_format=f'{t_start:.3f}~{t_end:.3f}s',
        title=f'{title_prefix} pattern',
        show=False
    )
    fig.savefig(f"{save_path_prefix}_pattern_topomap.png", dpi=300, bbox_inches="tight")
    print(f"[SAVE] {save_path_prefix}_pattern_topomap.png")