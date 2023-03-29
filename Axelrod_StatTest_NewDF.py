#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:42:57 2023

@author: raymondcn
"""

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import os
import numpy as np
import seaborn as sns


 
#load datafame
root_dir = "/Users/raymondcn/National Institutes of Health/NIMH CDN lab - Documents/Datasets/IDM/IDM-CloudResearch/"
df_aggregate = pd.read_csv(os.path.join(root_dir,"Aggregate_All_Model_Parameters.csv"))
print(list(df_aggregate))
['subject', 'cdd_meta', 'cdd_meta_negLL', 'cdd_alpha_meta', 'cdd_alpha_meta_negLL', 'crdm_meta', 'crdm_meta_negLL', 'crdm_meta_max_flag', 'cdd_meta_max_flag', 'cpdm_b1_meta', 'cpdm_b1_meta_negLL', 'cpdm_b2_meta', 'cpdm_b2_meta_negLL', 'cpdm_b3_meta', 'cpdm_b3_meta_negLL', 'cpdm_b4_meta', 'cpdm_b4_meta_negLL', 'cpdm_avg_meta', 'crdm_negLL', 'crdm_percent_lottery', 'crdm_percent_risk', 'crdm_percent_ambiguity', 'crdm_gamma', 'crdm_beta', 'crdm_alpha', 'crdm_R2', 'crdm_conf_1', 'crdm_conf_2', 'crdm_conf_3', 'crdm_conf_4', 'crdm_confidence_flag', 'cdd_negLL', 'cdd_percent_impulse', 'cdd_gamma', 'cdd_kappa', 'cdd_alpha', 'cdd_R2', 'cdd_conf_1', 'cdd_conf_2', 'cdd_conf_3', 'cdd_conf_4', 'cdd_confidence_flag', 'cdd_alpha_negLL', 'cdd_alpha_percent_impulse', 'cdd_alpha_gamma', 'cdd_alpha_kappa', 'cdd_alpha_alpha', 'cdd_alpha_R2', 'ias_Reverse Score', 'demo_Batch #', 'demo_Assigment ID', 'demo_Age', 'demo_Gender', 'demo_Race', 'demo_Hispanic or Latino?', 'demo_Income', 'demo_Diploma', 'demo_Depression and anxity', 'demo_substance abuse problems', 'demo_Smoker', 'demo_Number of cigarettes  smoke/smoked per day', 'demo_Number of alcoholic drink had in the past 30 days']



#create and plot correlation matrix
columnsDropped = ['cdd_alpha'] +['crdm_negLL'] + ['cdd_alpha_alpha'] + [c for c in list(df_aggregate) if 'crdm_conf_' in c] + [c for c in list(df_aggregate) if 'cdd_conf_' in c] + [c for c in list(df_aggregate) if 'cdd_alpha_conf_' in c] + [c for c in list(df_aggregate) if 'demo_' in c]
print(columnsDropped)
df_corr = df_aggregate.drop(columns=columnsDropped)

print("# participants pre-filters:", df_corr.shape[0])

#apply confidence variability and maxed-out meta-uncertainty filters
df_corr = df_corr[df_corr['crdm_confidence_flag'] == 0]
df_corr = df_corr[df_corr['cdd_confidence_flag'] == 0]
df_corr = df_corr[df_corr['crdm_meta_max_flag'] == 0]
df_corr = df_corr[df_corr['cdd_meta_max_flag'] == 0]

print("# participants post-filters:", df_corr.shape[0])



#create and plot correlation matrix with only columns of interest
df_corr = df_corr[['ias_Reverse Score','crdm_beta','cdd_meta','crdm_meta','cpdm_b1_meta','cpdm_b2_meta','cpdm_b3_meta','cpdm_b4_meta','cpdm_avg_meta']]
print(df_corr)

#computes pearsons coefficient by default
correlation_matrix_spearman = df_corr.corr(method = 'spearman')
fig, ax = plt.subplots(figsize=(30,15)) 
matrix = np.triu(np.ones_like(correlation_matrix_spearman))
sns.heatmap(correlation_matrix_spearman, mask=matrix, cmap="Spectral", annot=True, ax=ax, fmt='.2f')
plt.title("Spearman Correlation Matrix for all Parameters")
plt.show()



### plot distributions 

ias = df_corr['ias_Reverse Score']
CDD = df_corr['cdd_meta']
CRDM = df_corr['crdm_meta']
CPDM = df_corr['cpdm_avg_meta']
Ambiguity = df_corr['crdm_beta']

fig, ax = plt.subplots()
ax.hist(ias, bins = 30)

fig, ax = plt.subplots()
ax.hist(CDD, bins = 30)

fig, ax = plt.subplots()
ax.hist(CRDM, bins = 30)

fig, ax = plt.subplots()
ax.hist(CPDM, bins = 30)


fig, ax = plt.subplots()
ax.hist(Ambiguity, bins = 30)

### cdd spearman correlation 

x = df_corr['ias_Reverse Score']
y = df_corr['cdd_meta']

correlation, p_value = spearmanr(x, y)
print(correlation, p_value)

a, b = np.polyfit(x, y, 1)
plt.scatter(x, y)
plt.plot(x, a*x+b, 'r-')

### crdm spearman correlation

x = df_corr['ias_Reverse Score']
y = df_corr['crdm_meta']

correlation, p_value = spearmanr(x, y)
print(correlation, p_value)

a, b = np.polyfit(x, y, 1)
plt.scatter(x, y)
plt.plot(x, a*x+b, 'r-')

### cpdm spearman correlation 

x = df_corr['ias_Reverse Score']
y = df_corr['cpdm_avg_meta']

correlation, p_value = spearmanr(x, y)
print(correlation, p_value)

a, b = np.polyfit(x, y, 1)
plt.scatter(x, y)
plt.plot(x, a*x+b, 'r-')


x = df_corr['ias_Reverse Score']
y = df_corr['crdm_beta']

correlation, p_value = spearmanr(x, y)
print(correlation, p_value)

a, b = np.polyfit(x, y, 1)
plt.scatter(x, y)
plt.plot(x, a*x+b, 'r-')

### cdd pearsons correlation 

x = df_corr['ias_Reverse Score']
y = df_corr['cdd_meta']

correlation, p_value = pearsonr(x, y)
print(correlation, p_value)

### crdm pearsons correlation

x = df_corr['ias_Reverse Score']
y = df_corr['crdm_meta']

correlation, p_value = pearsonr(x, y)
print(correlation, p_value)

### cpdm pearsons correlation

x = df_corr['ias_Reverse Score']
y = df_corr['cpdm_avg_meta']

correlation, p_value = pearsonr(x, y)
print(correlation, p_value)


## linear regression 

y = df_corr['ias_Reverse Score']
x = df_corr[['cdd_meta', 'crdm_meta', 'cpdm_avg_meta', 'crdm_beta']]
X = sm.add_constant(x)


model = sm.OLS(y, X).fit()


print(model.summary())

## bar plot of t values 

t_values = model.tvalues[1:]
p_values = model.pvalues[1:]
variable_names = ['cdd_meta','crdm_meta','cpdm_avg_meta', "crdm_beta"]

alpha = 0.05

# Plot the t values as a bar plot

fig, ax = plt.subplots()
bars = ax.bar(variable_names, t_values)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_xlabel('Variable')
ax.set_ylabel('t-value')


for i in range(len(variable_names)):
    if p_values[i] < alpha:
        bars[i].set_color('green')  
        ax.text(i, t_values[i]+0.1, '*', ha='center', va='bottom', fontsize=14)  # add an asterisk to indicate significance
    else:
        bars[i].set_color('red') 

plt.show()