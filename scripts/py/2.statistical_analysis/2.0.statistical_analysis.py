# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 15:57:11 2025

@author: WANGLIANGFU
"""

import os
from glob import glob
import mne
import numpy as np
from mne.stats import permutation_cluster_test
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_morlet
from shutil import copyfile
copyfile('../common_settings.py','common_settings.py')
copyfile('../utils.py','utils.py')
import common_settings as CS
import utils as utils


if __name__ == "__main__":
    subject = 'sub01'
    working_dir = os.path.join('../..','data','clean_EEG',subject,)
    print(f"被试文件夹: {working_dir}")
    
    working_data = np.sort(glob(os.path.join(working_dir,"*",'*fif')))
    print(f"找到 {len(working_data)} 个数据文件")
    
    results_dir = os.path.join('../..','results','first_level_stats',subject,)
    figures_dir = os.path.join('../..','figures','first_level_stats',subject,)
    for f in [results_dir, figures_dir]:
        if not os.path.exists(f):
            os.makedirs(f)
    for filename in working_data:
        clean_epochs_task1 = mne.read_epoch
#########################################
#########task1leftvsright################
########################################
    events_id_right = list(CS.event_id_task1.keys())[0]
    events_id_left = list(CS.event_id_task1.keys())[1]
    epochs_condition_right = mne.Epochs(
        CS.clean_epochs_task1,
        events = CS.events,
        events_id = events_id_right,
        tmin = CS.tmin_task1,
        tmax = CS.tmax_task1,
        picks = CS.picks,
        baseline = None,
        )
    epochs_condition_left = mne.Epochs(
        CS.clean_epochs_task1,
        events = CS.events,
        events_id = events_id_left,
        tmin = CS.tmin_task1,
        tmax = CS.tmax_task1,
        picks = CS.picks,
        baseline = None,
        )
    decim = 2
    freqs = np.arange(7, 30, 3)  # define frequencies of interest
    n_cycles = 1.5
    tfr_epochs_right = tfr_morlet(
        epochs_condition_right,
        freqs,
        n_cycles=n_cycles,
        decim=decim,
        return_itc=False,
        average=False,
        )
        
    tfr_epochs_left = tfr_morlet(
        epochs_condition_left,
        freqs,
        n_cycles=n_cycles,
        decim=decim,
        return_itc=False,
        average=False,
        )
        
    tfr_epochs_right.apply_baseline(mode="ratio", baseline=(None, 0))
    tfr_epochs_left.apply_baseline(mode="ratio", baseline=(None, 0))
    epochs_power_right = tfr_epochs_right.data[:, 0, :, :]
    epochs_power_left = tfr_epochs_left.data[:, 0, :, :]
    threshold = 6
    F_obs, clusters,cluster_p_values, H0 = permutation_cluster_test(
        [epochs_power_right, epochs_power_left],
        out_type = 'mask', 
        n_permutations = 100,
        threshold = threshold,
        tail = 0,
        seed = np.random.default_rng(seed = 12345))
    times = 1e2 * epochs_condition_right.times
    fig, (ax, ax2) = plt.subplots(2 ,1, figsize = (6, 4), layout = 'constrained')
    evoked_power_right = epochs_power_right.mean(axis = 0)
    evoked_power_left = epochs_power_left.mean(axis = 0)
    evoked_power_contrast = evoked_power_right - evoked_power_left
    signs = np.sign(evoked_power_contrast)
    F_obs_plot = np.nan*np.ones_like(F_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= 0.05:
            F_obs_plot[c] = F_obs[c] * signs[c]
    ax.imshow(
        F_obs,
        extent = [times[0], times[-1], freqs[0], freqs[-1]],
        aspect = 'auto',
        origin = 'lower',
        cmap = 'gray')
    max_F = np.nanmax(abs(F_obs_plot))
    ax.imshow(
        F_obs_plot,
        extent = [times[0], times[-1], freqs[0], freqs[-1]],
        aspect = 'auto',
        origin = 'lower',
        cmap = 'RdBu_r',
        vmin = -max_F,
        vmax = max_F)
    ax.set_xlabel('Times(ms)')
    ax.set_ylabel('Frequency(HZ)')
    evoked_condition_right = epochs_condition_right.average()
    evoked_condition_left = epochs_condition_left.average()
    evoked_contrast = mne.combine_evoked(
        [evoked_condition_right, evoked_condition_left], weights = [1, -1])
    evoked_contrast.plot(axes = ax2, time_unit = 's')
#####################################
###############task1correct vs wrong#
#####################################
    events_per_trial = 6
    num_trials = len(CS.events[1:]) // events_per_trial
    correct_idx = []
    wrong_idx = []
    for trial_idx in range(num_trials):
        response1_idx = trial_idx * events_per_trial + 3 
        stimulus1_idx = trial_idx * events_per_trial + 1
        stimulus1_events_id = CS.events[stimulus1_idx, 2]
        response1_events_id = CS.events[response1_idx, 2]
        if stimulus1_events_id % 10 == response1_events_id:
            correct_idx.append(trial_idx)
        else:
            wrong_idx.append(trial_idx)
    correct_events = CS.events[correct_idx]
    wrong_events = CS.events[wrong_idx]
    stimulus1_events_id = CS.events[stimulus1_idx, 2]
    response1_events_id = CS.events[response1_idx, 2]
    epochs_condition_correct = clean_epochs_task1[correct_idx]
        
    epochs_condition_left = clean_epochs_task1[wrong_idx]
    decim = 2
    freqs = np.arange(7, 30, 3)  # define frequencies of interest
    n_cycles = 1.5
    tfr_epochs_right = tfr_morlet(
        epochs_condition_right,
        freqs,
        n_cycles=n_cycles,
        decim=decim,
        return_itc=False,
        average=False,
        )
        
    tfr_epochs_left = tfr_morlet(
        epochs_condition_left,
        freqs,
        n_cycles=n_cycles,
        decim=decim,
        return_itc=False,
        average=False,
        )
        
    tfr_epochs_right.apply_baseline(mode="ratio", baseline=(None, 0))
    tfr_epochs_left.apply_baseline(mode="ratio", baseline=(None, 0))
    epochs_power_right = tfr_epochs_right.data[:, 0, :, :]
    epochs_power_left = tfr_epochs_left.data[:, 0, :, :]
    threshold = 6
    F_obs, clusters,cluster_p_values, H0 = permutation_cluster_test(
        [epochs_power_right, epochs_power_left],
        out_type = 'mask', 
        n_permutations = 100,
        threshold = threshold,
        tail = 0,
        seed = np.random.default_rng(seed = 12345))
    times = 1e2 * epochs_condition_right.times
    fig, (ax, ax2) = plt.subplots(2 ,1, figsize = (6, 4), layout = 'constrained')
    evoked_power_right = epochs_power_right.mean(axis = 0)
    evoked_power_left = epochs_power_left.mean(axis = 0)
    evoked_power_contrast = evoked_power_right - evoked_power_left
    signs = np.sign(evoked_power_contrast)
    F_obs_plot = np.nan*np.ones_like(F_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= 0.05:
            F_obs_plot[c] = F_obs[c] * signs[c]
    ax.imshow(
        F_obs,
        extent = [times[0], times[-1], freqs[0], freqs[-1]],
        aspect = 'auto',
        origin = 'lower',
        cmap = 'gray')
    max_F = np.nanmax(abs(F_obs_plot))
    ax.imshow(
        F_obs_plot,
        extent = [times[0], times[-1], freqs[0], freqs[-1]],
        aspect = 'auto',
        origin = 'lower',
        cmap = 'RdBu_r',
        vmin = -max_F,
        vmax = max_F)
    ax.set_xlabel('Times(ms)')
    ax.set_ylabel('Frequency(HZ)')
    evoked_condition_right = epochs_condition_right.average()
    evoked_condition_left = epochs_condition_left.average()
    evoked_contrast = mne.combine_evoked(
        [evoked_condition_right, evoked_condition_left], weights = [1, -1])
    evoked_contrast.plot(axes = ax2, time_unit = 's')