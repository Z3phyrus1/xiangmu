# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 19:23:06 2025

@author: DELL
"""


import os
import numpy as np
import mne
from autoreject import AutoReject
import common_settings as CS
from mne.decoding import SlidingEstimator, cross_val_multiscore, Scaler, Vectorizer, LinearModel, GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

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

def run_ica_cleaning(epochs, picks, figures_dir, session_num, task_name, 
                     threshold_eog=0.3, threshold_muscle=0.9):

    cov = mne.compute_covariance(epochs, tmin=epochs.tmin, tmax=epochs.tmax)
    ica = mne.preprocessing.ICA(noise_cov=cov, **CS.ica_args)
    ica.fit(epochs, picks=picks)
    
    eog_idx, _ = ica.find_bads_eog(epochs,
        ch_name=['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8'],
        threshold=threshold_eog, measure='correlation')
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


def get_accuracy_labels_for_epochs(epochs, raw_events, task='task1', events_per_trial=6):
    task1_accuracy, task2_accuracy = compute_accuracy_labels_from_raw_events(
        raw_events, events_per_trial
    )
    
    if task == 'task1':
        accuracy_list = task1_accuracy
    elif task == 'task2':
        accuracy_list = task2_accuracy
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # 确保标签数量与 epochs 数量一致
    n_epochs = len(epochs)
    if len(accuracy_list) >= n_epochs:
        labels = np.array(accuracy_list[:n_epochs])
    else:
        # 如果标签不够，用 -1 填充（后续需要过滤）
        labels = np.array(accuracy_list + [-1] * (n_epochs - len(accuracy_list)))
    
    return labels


def get_task2_lr_labels(epochs_task2):
    events = epochs_task2.events[:, -1]
    labels = np.array([0 if e % 10 == 1 else 1 for e in events])
    return labels


def get_lr_labels_from_events(epochs):
    events = epochs.events[:, -1]
    labels = np.array([0 if e % 10 == 1 else 1 for e in events])
    return labels


def filter_valid_samples(X, y):
    valid_mask = y != -1
    return X[valid_mask], y[valid_mask]


def balance_classes(X, y, random_state=12345):
    np.random.seed(random_state)
    
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    
    indices = []
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        selected = np.random.choice(cls_indices, size=min_count, replace=False)
        indices.extend(selected)
    
    indices = np.array(sorted(indices))
    return X[indices], y[indices]


# ============================================================
# 解码器创建函数
# ============================================================
def make_decoder_pipeline(epochs_info):

    clf = make_pipeline(
        Scaler(epochs_info),
        Vectorizer(),
        StandardScaler(),
        LinearModel(LogisticRegression(
            solver='liblinear',
            random_state=CS.decoding_args['random_state']
        ))
    )
    return clf


def make_sliding_estimator(epochs_info):
    clf = make_decoder_pipeline(epochs_info)
    sliding_estimator = SlidingEstimator(
        clf,
        n_jobs=CS.decoding_args['n_jobs'],
        scoring=CS.decoding_args['scoring'],
        verbose=True
    )
    return sliding_estimator


def make_generalizing_estimator(epochs_info):
    clf = make_decoder_pipeline(epochs_info)
    gen_estimator = GeneralizingEstimator(
        clf,
        n_jobs=CS.decoding_args['n_jobs'],
        scoring=CS.decoding_args['scoring'],
        verbose=True
    )
    return gen_estimator


# ============================================================
# 解码运行函数
# ============================================================
def run_cv_decoding(X, y, epochs_info):
    sliding_estimator = make_sliding_estimator(epochs_info)
    cv = StratifiedKFold(
        n_splits=CS.decoding_args['n_splits'],
        shuffle=True,
        random_state=CS.decoding_args['random_state']
    )
    scores = cross_val_multiscore(
        sliding_estimator, X, y,
        cv=cv,
        n_jobs=CS.decoding_args['n_jobs']
    )
    return scores


def run_generalization_decoding(X, y, epochs_info):

    gen_estimator = make_generalizing_estimator(epochs_info)
    cv = StratifiedKFold(
        n_splits=CS.decoding_args['n_splits'],
        shuffle=True,
        random_state=CS.decoding_args['random_state']
    )
    scores = cross_val_multiscore(
        gen_estimator, X, y,
        cv=cv,
        n_jobs=CS.decoding_args['n_jobs']
    )
    return scores


def run_cross_task_decoding(X_train, y_train, X_test, y_test, epochs_info):
    sliding_estimator = make_sliding_estimator(epochs_info)
    sliding_estimator.fit(X_train, y_train)
    scores = sliding_estimator.score(X_test, y_test)
    return scores


# ============================================================
# 绘图函数
# ============================================================
def plot_time_decoding(times, scores, title, save_path=None, chance_level=0.5):
    import matplotlib.pyplot as plt
    
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


def plot_temporal_generalization(times_train, times_test, scores, title, save_path=None, vmin=0.3, vmax=0.7):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 8))
    im = plt.imshow(scores, interpolation='lanczos', origin='lower',
                    extent=[times_test[0], times_test[-1], times_train[0], times_train[-1]],
                    cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='AUC')
    plt.xlabel('Test time (s)')
    plt.ylabel('Train time (s)')
    plt.title(title)
    plt.axhline(0, color='k', linestyle='-')
    plt.axvline(0, color='k', linestyle='-')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()