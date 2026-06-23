# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:58:44 2026

@author: WANGLIANGFU
"""

import os
import re
import argparse
from glob import glob
import numpy as np
import pandas as pd
import mne

from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from mne.decoding import SlidingEstimator, cross_val_multiscore

import common_settings as CS
import utils as utils


def read_concat(files, baseline, crop_tmin, crop_tmax):
    if len(files) == 0:
        return None
    eps = [mne.read_epochs(f, verbose=False) for f in files]
    for e in eps:
        e.apply_baseline(baseline)
    epochs = mne.concatenate_epochs(eps)
    epochs.crop(crop_tmin, crop_tmax).resample(100)
    return epochs


def get_w_sub2(df_fit, sub_id_int, trial=None):
    id_col = (
        df_fit['id'].astype(str).str.strip()
        .str.replace('"', '', regex=False)
        .str.replace("'", "", regex=False)
    )
    model_col = (
        df_fit['model'].astype(str).str.strip().str.lower()
        .str.replace('"', '', regex=False)
        .str.replace("'", "", regex=False)
    )

    mask = (id_col == str(sub_id_int)) & (model_col == 'sub2')
    row = df_fit[mask]

    if row.empty:
        mask = (id_col == str(sub_id_int)) & (model_col.str.contains('sub2', na=False))
        row = df_fit[mask]

    if row.empty:
        cand = sorted(model_col[id_col == str(sub_id_int)].dropna().unique().tolist())
        print(f"[DEBUG] models for id={sub_id_int}: {cand[:20]}")
        return None

    w_val = str(row.iloc[0]['w']).strip().replace('"', '').replace("'", "")
    return float(w_val)


def _build_clusters_from_sigmask(sig_mask, p_vals):
    clusters, cluster_p_vals, start = [], [], None
    for i, flag in enumerate(sig_mask):
        if flag and start is None:
            start = i
        end_cluster = (not flag and start is not None)
        end_last = (flag and start is not None and i == len(sig_mask) - 1)
        if end_cluster or end_last:
            end = i if end_cluster else (i + 1)
            cmask = np.zeros_like(sig_mask, dtype=bool)
            cmask[start:end] = True
            clusters.append(cmask)
            cluster_p_vals.append(float(np.min(p_vals[start:end])))
            start = None
    return clusters, np.array(cluster_p_vals, dtype=float)


def run_sliding_analysis(X, y, times, task_name, label_name, subject, trial, figures_dir, n_splits, test_size, random_state):
    print(f"\n--- Running: {task_name} | {label_name} ---")

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            solver='liblinear', class_weight='balanced', random_state=random_state,
            penalty='l1', Cs=np.logspace(-3, 3, 7), cv=5, scoring='roc_auc',
            max_iter=int(1e3), n_jobs=1, verbose=0
        )
    )

    slider = SlidingEstimator(clf, scoring='roc_auc', n_jobs=1, verbose=False)
    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    scores = cross_val_multiscore(slider, X, y, cv=cv, n_jobs=1, verbose=1)

    t_vals, clusters, cluster_p_vals, H0 = mne.stats.permutation_cluster_1samp_test(
        scores - 0.5, tail=1, n_permutations=1000,
        threshold=dict(start=0, step=0.2), seed=12345,
        n_jobs=1, out_type='mask', verbose=False
    )

    if clusters is None or len(clusters) == 0:
        p_tfce = (np.sum(H0[:, None] >= t_vals[None, :], axis=0) + 1) / (len(H0) + 1)
        sig_mask = p_tfce < 0.05
        clusters, cluster_p_vals = _build_clusters_from_sigmask(sig_mask, p_tfce)

    scores_mean = np.mean(scores, axis=0)
    scores_sem = np.std(scores, axis=0) / np.sqrt(scores.shape[0])

    save_path = os.path.join(figures_dir, f'{subject}_{trial}_{task_name}_{label_name}_sliding.png')
    utils.plot_with_cluster_highlight(
        times=times, scores_mean=scores_mean, scores_sem=scores_sem,
        permutation_result=(t_vals, clusters, cluster_p_vals, H0),
        title=f"{task_name}: {label_name} ({subject})", save_path=save_path, chance_level=0.5
    )
    print(f"Figure saved: {save_path}")
    return scores


def _get_behavior_for_subject_trial(df_pred, sub_id_int, trial):
    id_series = (
        df_pred['id'].astype(str).str.strip()
        .str.replace('"', '', regex=False)
        .str.replace("'", "", regex=False)
    )
    sub_from_id = id_series.str.extract(r'^(\d+)')[0].astype(float).astype('Int64')
    df_sub = df_pred[sub_from_id == sub_id_int].copy()

    if 'trial' in df_sub.columns:
        return df_sub[df_sub['trial'].astype(str).str.strip() == str(trial)].reset_index(drop=True)
    elif 'seq' in df_sub.columns:
        return df_sub[df_sub['seq'].astype(str).str.strip() == str(trial)].reset_index(drop=True)
    else:
        print("[WARN] no trial/seq column; use all rows")
        return df_sub.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--trial", type=str, required=True)
    args = parser.parse_args()

    subject, trial = args.subject, args.trial
    sub_id_int = int(re.search(r'\d+', subject).group())
    test_size, n_splits, random_state = 0.2, 100, 42

    working_dir = os.path.join('../../..', 'data', 'clean_EEG_dual', subject)
    print(f"Subject: {subject}, Trial: {trial}")
    print(f"Working dir: {working_dir}")

    task1_resp_files = sorted(glob(os.path.join(working_dir, f'clean_epochs_response1_ICA_*{trial}_sess*.fif')))
    task2_stim_files = sorted(glob(os.path.join(working_dir, f'clean_epochs_task2_ICA_*{trial}_sess*.fif')))
    task1_stim_files = sorted(glob(os.path.join(working_dir, f'clean_epochs_task1_ICA_*{trial}_sess*.fif')))

    if len(task1_resp_files) == 0:
        print(f"[SKIP] No task1 response files for {subject} trial {trial}"); return
    if len(task2_stim_files) == 0:
        print(f"[SKIP] No task2 stimulus files for {subject} trial {trial}"); return
    if len(task1_stim_files) == 0:
        print(f"[SKIP] No task1 stimulus files for {subject} trial {trial}"); return

    behavior_path = os.path.join('../../..', 'data', 'behavior', 'data_wide_wmPred.txt')
    fit_par_path = os.path.join('../../..', 'data', 'behavior', 'fit_par.txt')

    df_pred = pd.read_csv(behavior_path, sep=';')
    df_fit = pd.read_csv(fit_par_path, sep=r'\s+', engine='python')
    df_fit.columns = [c.strip().strip('"') for c in df_fit.columns]

    w_val = get_w_sub2(df_fit, sub_id_int, trial)
    if w_val is None:
        print(f"[SKIP] no w(sub2): id={sub_id_int}")
        return

    beh_aligned = _get_behavior_for_subject_trial(df_pred, sub_id_int, trial)
    if len(beh_aligned) == 0:
        print(f"[SKIP] Behavior rows = 0 for {subject} trial {trial}")
        return

    ep_t1_r = read_concat(task1_resp_files, CS.baseline_response, -0.5, CS.tmax_response)
    ep_t2_s = read_concat(task2_stim_files, CS.baseline_task2, CS.tmin_task2, CS.tmax_task2)
    if ep_t1_r is None or ep_t2_s is None:
        print(f"[SKIP] failed reading epochs for {subject} trial {trial}")
        return

    ep_t1_s = mne.concatenate_epochs([mne.read_epochs(f, verbose=False) for f in task1_stim_files])

    X_t1, X_t2 = ep_t1_r.get_data(copy=False), ep_t2_s.get_data(copy=False)
    times_t1, times_t2 = ep_t1_r.times, ep_t2_s.times

    y_acc = utils.compute_accuracy_from_epochs(ep_t1_s, ep_t1_r)
    y_conf = (beh_aligned['abs1'] > w_val).astype(int).values

    n_min = min(len(y_acc), len(y_conf), X_t1.shape[0], X_t2.shape[0])
    if n_min <= 1:
        print(f"[SKIP] Aligned samples too few: {n_min}")
        return

    X_t1, X_t2 = X_t1[:n_min], X_t2[:n_min]
    y_acc, y_conf = y_acc[:n_min], y_conf[:n_min]

    if len(np.unique(y_acc)) < 2:
        print("[SKIP] y_acc has only one class"); y_acc = None
    if len(np.unique(y_conf)) < 2:
        print("[SKIP] y_conf has only one class"); y_conf = None
    if y_acc is None and y_conf is None:
        print("[SKIP] both labels invalid"); return

    results_dir = os.path.join('../../..', 'results', 'decoding_idsess', subject, trial)
    figures_dir = os.path.join('../../..', 'figures', 'decoding_idsess', subject, trial)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    s_t1_acc = s_t2_acc = s_t1_conf = s_t2_conf = None

    if y_acc is not None:
        s_t1_acc = run_sliding_analysis(X_t1, y_acc, times_t1, "Task1_Resp", "Accuracy", subject, trial, figures_dir, n_splits, test_size, random_state)
        s_t2_acc = run_sliding_analysis(X_t2, y_acc, times_t2, "Task2_Stim", "Accuracy", subject, trial, figures_dir, n_splits, test_size, random_state)

    if y_conf is not None:
        s_t1_conf = run_sliding_analysis(X_t1, y_conf, times_t1, "Task1_Resp", "Confidence", subject, trial, figures_dir, n_splits, test_size, random_state)
        s_t2_conf = run_sliding_analysis(X_t2, y_conf, times_t2, "Task2_Stim", "Confidence", subject, trial, figures_dir, n_splits, test_size, random_state)

    out_npz = os.path.join(results_dir, f'{subject}_{trial}_sliding_scores.npz')
    np.savez(
        out_npz,
        scores_t1_acc=s_t1_acc, scores_t1_conf=s_t1_conf,
        scores_t2_acc=s_t2_acc, scores_t2_conf=s_t2_conf,
        times_t1=times_t1, times_t2=times_t2
    )
    print(f"Saved: {out_npz}")
    print("Done!")


if __name__ == "__main__":
    main()