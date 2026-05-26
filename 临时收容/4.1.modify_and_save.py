#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 18:57:22 2023

@author: MeiNing_01
"""

import os
import numpy as np
import pandas as pd
from shutil import copyfile,rmtree
copyfile('../experiment_settings.py','experiment_settings.py')
copyfile('../sub_electrodes_list.py','sub_electrodes_list.py')
copyfile('../sub_marker_list.py', 'sub_marker_list.py')
copyfile('../utils.py','utils.py')
import sub_electrodes_list as SEL
import experiment_settings as ES

brain_areas = ['frontal_area','visual_area']

template = '4.0.cross_brain_region_decoding.py'
scripts_folder = '../bash_cross_decoding'
if not os.path.exists(scripts_folder):
    os.mkdir(scripts_folder)
else:
    rmtree(scripts_folder)
    os.mkdir(scripts_folder)
os.mkdir(f'{scripts_folder}/outputs')
copyfile('utils.py',f'{scripts_folder}/utils.py')
copyfile('experiment_settings.py',f'{scripts_folder}/experiment_settings.py')
copyfile('sub_electrodes_list.py',f'{scripts_folder}/sub_electrodes_list.py')
copyfile('sub_marker_list.py', f'{scripts_folder}/sub_markder_list.py')

n_jobs = 20
node = 1
mem = 5


for brain_area_target in brain_areas:
    df = dict(subject = [],
              brain_area = [],
              )
    for subj,val in SEL.subject_info.items():
        dict_channel = val[ES.brain_area_folder_map[brain_area_target]]
        unique_channels = list(set(dict_channel.values()))
        for unique_channel in unique_channels:
            df['subject'].append(subj)
            df['brain_area'].append(unique_channel)
    df = pd.DataFrame(df)
    for brain_region_name,df_sub in df.groupby('brain_area'):
        if df_sub.shape[0] > 3:
            new_script_name = os.path.join(scripts_folder,f'decode_{brain_region_name}.py'.replace(' ','-').replace('(','-').replace(')','-'))
            with open(new_script_name,'w') as new_file:
                with open(template,'r') as old_file:
                    for line in old_file:
                        if "brain_area_target = " in line:
                            print(line)
                            line = f"    brain_area_target = '{brain_area_target}'\n"
                            print(line)
                        elif "# change target" in line:
                            print(line)
                            line = f"    brain_region_name_target = '{brain_region_name}'\n"
                            print(line)
                        new_file.write(line)
                    old_file.close()
                new_file.close()
            
            new_bash_script = os.path.join(scripts_folder,f'proc_{brain_region_name}'.replace(' ','-').replace('(','-').replace(')','-'))
            content = f'''#!/bin/bash
#SBATCH --job-name=P-{brain_region_name.replace(' ','-').replace('(','-').replace(')','-')}
#SBATCH --output=outputs/out_{brain_region_name.replace(' ','-').replace('(','-').replace(')','-')}.txt
#SBATCH --error=outputs/err_{brain_region_name.replace(' ','-').replace('(','-').replace(')','-')}.txt
#SBATCH --time=100:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={n_jobs}
#SBATCH --nodes={node}
#SBATCH --account=MeiNing_01
#SBATCH --mem-per-cpu={mem}G


module load anaconda3
source activate workstation
cd $SLURM_SUBMIT_DIR
python3 {new_script_name}
'''
            with open(new_bash_script,'w') as f:
                f.write(content)
                f.close()
