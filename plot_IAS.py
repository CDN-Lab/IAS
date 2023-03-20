#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:17:11 2023

@author: raymondcn
"""

import matplotlib.pyplot as plt
import pandas as pd
import glob
import os


root_path = '/Users/raymondcn/Library/CloudStorage/OneDrive-SharedLibraries-NationalInstitutesofHealth/NIMH CDN lab - Documents/Datasets/IDM/IDM-CloudResearch'
ias_dir = os.path.join(root_path, 'IAS_Scores')

batch_csvfiles = glob.glob(os.path.join(ias_dir,'IAS*.csv'))

print(batch_csvfiles)

reverse_scores = []

for fn in batch_csvfiles:
    df = pd.read_csv(fn)
    reverse_scores = reverse_scores + list(df['Reverse Score'])
    
fig, ax = plt.subplots()
ax.hist(reverse_scores, bins = 60)


    