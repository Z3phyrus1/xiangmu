# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 19:23:40 2025

@author: DELL
"""
from glob import glob
import numpy as np
import os as os
import mne
import utils as utils



fmin,fmax = 1,40
tmin_task1,tmax_task1 = -1.2,0.3
tmin_task2,tmax_task2 = -0.42,0.3 # 这里最后要用和task1一样的baseline
# tmin_idi,tmax_idi = 0,0.42
tmin_response = -0.2
tmax_response = 0.3

baseline_response = (-1.2, -0)  # 拼接后的 baseline（和 task2 一致）
baseline_task1 = (-1.2,0)
baseline_task2 = (-1.2-0.42, -0.42)
# baseline_belief = (-1.2, 0)
## filter parameters
filter_args = dict(filter_length    = 'auto',    # the filter length is chosen based on the size of the transition regions (6.6 times the reciprocal of the shortest transition band for fir_window=’hamming’ and fir_design=”firwin2”, and half that for “firwin”)
method           = 'fir',     # overlap-add FIR filtering
phase            = 'zero',    # the delay of this filter is compensated for
fir_window       = 'hamming', # The window to use in FIR design
fir_design       = 'firwin',  # a time-domain design technique that generally gives improved attenuation using fewer samples than “firwin2”
n_jobs           = -1,
verbose          = False,)

## ICA parameters
ica_args = dict(n_components = .99,
                random_state = 12345,
                method = 'infomax',
                max_iter = int(3e3),
                )

## autoreject parameters
autoreject_args = dict(
    n_interpolate = np.array([1,4,32]),
    consensus = np.linspace(0, 1.0, 11),
    cv = 10,
    thresh_method = 'bayesian_optimization',
    n_jobs = -1,
    random_state = 12345,
    verbose = 1,
    )
events_per_trial = 6

##decoding parameters
classifier_args = dict(
    Cs=np.logspace(-3, 3, 7),
    solver='liblinear',
    penalty = 'l1',
    cv = 10,
    dual = False,
    class_weight = 'balanced',
    max_iter = 3000,
    random_state = 12345,
    n_jobs = 1,
    )
temporal_decoding_args = dict(
    n_jobs = 1,
    scoring = 'roc_auc',
    verbose = False,
    )
temporal_generalizing_args = dict(
    n_jobs = -1,
    scoring='roc_auc',
    verbose = False,
    )
cv_args = dict(
    n_splits = 25,
    test_size = 0.2,
    random_state = 12345
    )
# session_num+id=偶数：task1=线, task2=点
# session_num+id=奇数：task1=点, task2=线
event_id_line = {'line_right': 22, 'line_left': 21}
event_id_dot = {'dot_right': 12, 'dot_left': 11}

# Response 事件 ID
response_id = {
    'response_left': 1,
    'response_right': 2,
}