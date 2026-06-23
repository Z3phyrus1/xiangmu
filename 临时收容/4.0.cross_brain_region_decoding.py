#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 16:06:46 2025

@author: MeiNing_01
"""

import os
import mne
from glob import glob
import numpy as np
import pandas as pd

from scipy import stats
from collections import Counter

from sklearn import preprocessing as prepc
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit
from mne.decoding import GeneralizingEstimator

# copyfile('../experiment_settings.py','experiment_settings.py')
import experiment_settings as ES
# copyfile('../sub_electrodes_list.py','sub_electrodes_list.py')
import sub_electrodes_list as SEL
# copyfile('../sub_marker_list.py', 'sub_marker_list.py')
# from sub_marker_list import get_subject_markers
# copyfile('../elec_change_for_bi.py','elec_change_for_bi.py')
# from elec_change_for_bi import dict_rename_channel
# copyfile('../utils.py','utils.py')
from utils import (#parse_channels_brain_regions_ERP_accordingly,
                   #get_significant_regions,
                   load_and_concat
                   )

def prepare_for_training(epochs_list,brain_area):
    # average over all channels in one brain area
    temp_epochs = []
    temp_events = []
    temp_groups = []
    for idx_sub,e in enumerate(epochs_list):
        e = e.resample(ES.resample_for_decoding)
        data = e.get_data()
        events = e.events 
        
        temp_epochs.append(data.mean(1)[:,np.newaxis,:])
        temp_events.append(events)
        temp_groups.append([idx_sub] * events.shape[0])
        
        
    temp_epochs = np.concatenate(temp_epochs,axis = 0)
    temp_events = np.concatenate(temp_events,axis = 0)
    temp_groups = np.concatenate(temp_groups,)
    
    info = mne.create_info([ES.brain_area_folder_map[brain_area]],sfreq = e.info['sfreq'],ch_types = 'eeg',)
    temp_events[:,0] = np.arange(temp_events.shape[0])
    epochs_concated = mne.EpochsArray(temp_epochs,info,
                                    events = temp_events,
                                    event_id = ES.dict_selection['confidence'][0],
                                    tmin = ES.tmin,
                                    baseline = (ES.tmin,0))
    
    _,_,_,_,condition1,condition2 = list(ES.dict_selection['confidence'][0].keys())
    
    #
    n_unique_subjects = np.unique(temp_groups).shape[0]
    temp_dict = dict(Counter(temp_groups))
    mean_trials,std_trials = np.mean(list(temp_dict.values())),np.std(list(temp_dict.values()))
    
    return epochs_concated,temp_events,temp_groups,condition1,condition2,n_unique_subjects,mean_trials,std_trials

def make_classifier():
    pipeline = make_pipeline(prepc.StandardScaler(),
                             # LinearSVC(C = 1,
                             #           penalty='l1',
                             #           class_weight = 'balanced',
                             #           random_state = 12345,
                             #           ),)
                             LogisticRegressionCV(Cs=np.logspace(-3,3,7),
                                                  cv = 5,
                                                  scoring = 'roc_auc',
                                                  max_iter = int(3e3),
                                                  solver = 'liblinear',
                                                  penalty = 'l2',
                                                  random_state = 12345,
                                                  n_jobs = 1,
                                                  class_weight='balanced'),)
    
    slider = GeneralizingEstimator(pipeline,scoring = "roc_auc",n_jobs = 1,verbose = 1,)
    return slider

if __name__ == "__main__":
    event_type = 'cross_brain_region' # confidence, orientation
    brain_area_source = 'claustrum' 
    brain_area_target = 'frontal_area' # frontal_area, visual_area
    
    working_dir = '../../data/clean_SEEG'
    working_data = np.sort(glob(os.path.join(working_dir,'sub*','low_high_confidence-epo.fif')))
    working_beha = np.sort(glob(os.path.join(working_dir,'sub*','clean_df.csv')))
    
    
    results_dir = f'../../results/{event_type}/{brain_area_source} to {brain_area_target}'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    epochs_concat_source = load_and_concat(working_data, working_beha, brain_area_source,SEL,ES,)
    epochs_concat_target = load_and_concat(working_data, working_beha, brain_area_target,SEL,ES,)
    
    brain_region_name_source = 'claustrum'
    brain_region_name_target = 'dorsal_attention(Frontal_Mid_2 L)' # change target
    
    if not os.path.exists(os.path.join(results_dir,f'scores {brain_region_name_source} to {brain_region_name_target}.npy')):
        # train the classifiers
        ## train on 3 subjects, iteration
        epoch_list_source = epochs_concat_source[brain_region_name_source]
        (epochs_source,events_source,groups_source,
         condition1_source,condition2_source,
         n_unique_subjects_source,
         mean_trials_source,std_trials_source,
         ) = prepare_for_training(epoch_list_source,brain_area_source)
        cv = GroupShuffleSplit(n_splits = 50,train_size = 0.5,random_state = 12345)
        
        # test the classifiers
        ## test on all subjects, variation comes from the training
        epoch_list_target = epochs_concat_target[brain_region_name_target]
        (epochs_target,events_target,groups_target,
         condition1_target,condition2_target,
         n_unique_subjects_target,
         mean_trials_target,std_trials_target,
         ) = prepare_for_training(epoch_list_target,brain_area_target)
        
        (features_train,
         labels_train,
         groups_train,
         ) = epochs_source.get_data(),events_source[:,-1],groups_source
        features_test,labels_test = epochs_target.get_data(),events_target[:,-1]
        scores = []
        for idx_train,idx_no_use in cv.split(features_train,labels_train,groups_train):
            clf = make_classifier()
            clf.fit(features_train,labels_train)
            score = clf.score(features_test,labels_test)
            scores.append(score)
        # save scores
        scores = np.array(scores)
        np.save(os.path.join(results_dir,f'scores {brain_region_name_source} to {brain_region_name_target}.npy'),
                scores,allow_pickle = True, )
        # stats
        t_thresh = stats.t.ppf(1 - 0.001 / 2, df = scores.shape[0] - 1)
        t_obs,clusters,cluster_pv,H0 = mne.stats.permutation_cluster_1samp_test(scores - 0.5,
                                                                          threshold = t_thresh,
                                                                          n_permutations = int(1e4),
                                                                          tail = 1,
                                                                          stat_fun = None,
                                                                          n_jobs = 1,
                                                                          seed = 12345,
                                                                          out_type = 'mask',
                                                                          verbose = 1,
                                                                          )
        t_obs_plot = np.nan * np.ones_like(t_obs)
        for c,pval in zip(clusters,cluster_pv):
            if pval <= 0.05:
                t_obs_plot[c] = t_obs[c]
        
        np.save(os.path.join(results_dir,f't_obs {brain_region_name_source} to {brain_region_name_target}.npy'),
                t_obs,allow_pickle = True, )
        np.save(os.path.join(results_dir,f't_obs_plot {brain_region_name_source} to {brain_region_name_target}.npy'),
                t_obs_plot,allow_pickle = True, )
    