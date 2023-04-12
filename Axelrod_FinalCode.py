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
df_corr = df_corr[['ias_Reverse Score','cdd_alpha_meta','crdm_beta','cdd_meta','crdm_meta','cpdm_b1_meta','cpdm_b2_meta','cpdm_b3_meta','cpdm_b4_meta','cpdm_avg_meta']]
print(df_corr)

#computes pearsons coefficient by default
correlation_matrix_spearman = df_corr.corr(method = 'spearman')
fig, ax = plt.subplots(figsize=(30,15)) 
matrix = np.triu(np.ones_like(correlation_matrix_spearman))
sns.heatmap(correlation_matrix_spearman, mask=matrix, cmap="Spectral", annot=True, ax=ax, fmt='.2f')
plt.title("Spearman Correlation Matrix for all Parameters")
plt.show()

#computes pearsons coefficient by default
correlation_matrix_pearson = df_corr.corr()
fig, ax = plt.subplots(figsize=(30,15)) 
matrix = np.triu(np.ones_like(correlation_matrix_pearson))
sns.heatmap(correlation_matrix_pearson, mask=matrix, cmap="Spectral", annot=True, ax=ax, fmt='.2f')
plt.title("Pearson Correlation Matrix for all Parameters")
plt.show()



### plot distributions 

ias = df_corr['ias_Reverse Score']
CDD = df_corr['cdd_meta']
CRDM = df_corr['crdm_meta']
CPDM = df_corr['cpdm_avg_meta']
Ambiguity = df_corr['crdm_beta']


CDD_inverse = 1/CDD
CRDM_inverse = 1/CRDM
CPDM_inverse = 1/CPDM


log_ias = np.log10(ias)
log_cdd = np.log10(CDD)
log_crdm = np.log10(CRDM)
log_cpdm = np.log10(CPDM)

### IAS Histogram


fig, ax = plt.subplots()
ax.hist(log_ias, bins=30, color='lightpink')
ax.set_xlabel('Log of IAS Scores', fontsize=16)
ax.set_ylabel('Frequency', fontsize=16)
ax.set_title('Histogram of $\\bf{IAS}$ Scores', fontsize=20)

# Remove the spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()

### CDD Histogram

fig, ax = plt.subplots()
ax.hist(log_cdd, bins=30, color='lightblue')
ax.set_xlabel('Log of IAS Scores', fontsize=16)
ax.set_ylabel('Frequency', fontsize=16)
ax.set_title('Histogram of $\\bf{IAS}$ Scores', fontsize=20)

# Remove the spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()


### CRDM Histogram

fig, ax = plt.subplots()
ax.hist(log_crdm, bins=30, color='plum')
ax.set_xlabel('Log of IAS Scores', fontsize=16)
ax.set_ylabel('Frequency', fontsize=16)
ax.set_title('Histogram of $\\bf{IAS}$ Scores', fontsize=20)

# Remove the spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()


### CPDM Histogram

fig, ax = plt.subplots()
ax.hist(log_cpdm, bins=30, color='mediumaquamarine')
ax.set_xlabel('Log of IAS Scores', fontsize=16)
ax.set_ylabel('Frequency', fontsize=16)
ax.set_title('Histogram of $\\bf{IAS}$ Scores', fontsize=20)

# Remove the spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()



### cdd spearman correlation 




x = df_corr['ias_Reverse Score']
y = np.log10(CDD_inverse)

correlation, p_value = spearmanr(x, y)
print(correlation, p_value)

a, b = np.polyfit(x, y, 1)
plt.scatter(x, y, color='lightblue', edgecolors='gray', s=65)
plt.plot(x, a*x+b, 'r-', color='gray')
plt.xlabel('IAS Score', fontsize=15)
plt.ylabel('CDD Meta-Uncertainty', fontsize=15)
plt.title('Correlation between $\\bf{IAS}$ Scores and $\\bf{CDD}$ Meta-Uncertainty', fontsize=15)

# Remove the top and right spines of the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()




### crdm spearman correlation

x = df_corr['ias_Reverse Score']
y = np.log10(CRDM_inverse)

correlation, p_value = spearmanr(x, y)
print(correlation, p_value)

a, b = np.polyfit(x, y, 1)
plt.scatter(x, y, color='plum', edgecolors='gray', s=65)
plt.plot(x, a*x+b, 'r-', color='gray')
plt.xlabel('IAS Score', fontsize=15)
plt.ylabel('CDD Meta-Uncertainty', fontsize=15)
plt.title('Correlation between $\\bf{IAS}$ Scores and $\\bf{CDD}$ Meta-Uncertainty', fontsize=15)

# Remove the top and right spines of the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()


### cpdm spearman correlation 

x = df_corr['ias_Reverse Score']
y = np.log10(CPDM_inverse)

correlation, p_value = spearmanr(x, y)
print(correlation, p_value)

a, b = np.polyfit(x, y, 1)
plt.scatter(x, y, color='mediumaquamarine', edgecolors='gray', s=65)
plt.plot(x, a*x+b, 'r-', color='gray')
plt.xlabel('IAS Score', fontsize=15)
plt.ylabel('CDD Meta-Uncertainty', fontsize=15)
plt.title('Correlation between $\\bf{IAS}$ Scores and $\\bf{CDD}$ Meta-Uncertainty', fontsize=15)

# Remove the top and right spines of the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()



## linear regression 

y = df_corr['ias_Reverse Score']
x = df_corr[['cdd_meta', 'crdm_meta', 'cpdm_avg_meta']]
X = sm.add_constant(x)


model = sm.OLS(y, X).fit()


print(model.summary())

## bar plot of t values 

t_values = model.tvalues[1:]
p_values = model.pvalues[1:]
variable_names = ['cdd_meta','crdm_meta','cpdm_avg_meta']

alpha = 0.05

# Plot the t values as a bar plot

t_values = model.tvalues[1:]
p_values = model.pvalues[1:]
variable_names = ['CDD Meta','CRDM Meta','CPDM Meta']

alpha = 0.05

# Define the colors for the bars
colors = ['lightblue', 'plum', 'mediumaquamarine']

# Plot the t values as a bar plot
fig, ax = plt.subplots(figsize=(8,6)) # increase figure size
bars = ax.bar(variable_names, t_values, color=colors)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_xlabel('Meta-Uncertainty', fontsize=14) # increase font size of x-axis label
ax.set_ylabel('T Value', fontsize=14) # increase font size of y-axis label
ax.set_title('T-Values by Meta-Uncertainty Parameters', fontsize=18) # increase font size of title

plt.show()