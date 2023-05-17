#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 18:24:32 2023

@author: raymondcn
"""

import pandas as pd
import glob
import os

root_path = '/Volumes/UCDN/datasets/IDM'
save_dir = os.path.join(root_path, 'ias_scores')


# Define the file paths of the participant CSV files to read
file_paths = glob.glob(os.path.join(root_path,'raw_csv', '*.csv'))
print(file_paths)

    
# Define the criteria for scoring the CSV files
criteria = {
    'criteria_1': {'column': 13, 'weight': 1.0, 'range': (25, 46)}
}

# Define an empty dataframe to store the scores for each participant
participant_df = pd.DataFrame(columns=['Subject', 'Total Score', 'Reverse Score'])


# Loop through each participant CSV file
for file_path in file_paths:
    # Read the CSV file into a pandas dataframe, specifying the delimiter and header=None
    df = pd.read_csv(file_path, header=None)
    
    # Extract the file name from the file path
    # file_name = file_path.split('/')[-1]
    file_name = os.path.basename(file_path).replace('.csv','')

    
 # Calculate the original and reverse scores for the criterion
    criterion_column = criteria['criteria_1']['column']
    criterion_weight = criteria['criteria_1']['weight']
    criterion_range = criteria['criteria_1']['range']
    criterion_values = pd.Series(df[criterion_column][criterion_range[0]:criterion_range[1]])
    criterion_values = pd.to_numeric(criterion_values, errors='coerce')
    criterion_reverse_values = pd.Series(
        criterion_values.replace({1: 5, 2: 4, 3: 3, 4: 2, 5: 1}).values)
    criterion_reverse_values = pd.to_numeric(criterion_reverse_values, errors='coerce').sort_values(ascending=False)
    criterion_score = criterion_values.sum() * criterion_weight
    criterion_reverse_score = criterion_reverse_values.sum() * criterion_weight
    
    # Add the participant's score to the scores dataframe
    participant_df = participant_df.append({'Subject': file_name, 'Total Score': criterion_score, 'Reverse Score': criterion_reverse_score}, ignore_index=True)

# Save the scores dataframe to a CSV file
participant_df.to_csv(os.path.join(save_dir, 'IAS_scores.csv'), index=False)
    
    
    
    