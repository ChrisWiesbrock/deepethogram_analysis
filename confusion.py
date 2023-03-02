# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:41:05 2023

@author: wiesbrock
"""

from IPython import get_ipython
#get_ipython().magic('reset -sf')

import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.stats as stats
import seaborn as sns
from scipy.ndimage import uniform_filter1d

path=r'C:\Users\wiesbrock\Desktop\first.xlsx'

array_truth=pd.read_excel(path,sheet_name='truth')
array_pred=pd.read_excel(path,sheet_name='pred')

header=list(array_truth.columns)

confus_matrix=np.zeros((len(header),len(header)))

truth_data=np.zeros((len(array_truth),len(header)))
pred_data=np.zeros((len(array_truth),len(header)))

#truth data
for i in range(len(header)):
    truth_data[:,i]=array_truth[header[i]]


list_truth=np.zeros((len(array_truth)))

for i in range(len(header)):
    list_truth[truth_data[:,i]==1]=i
    
#truth data
for i in range(len(header)):
    pred_data[:,i]=array_pred[header[i]]


list_pred=np.zeros((len(array_pred))).astype(str)

for i in range(len(header)):
    list_pred[pred_data[:,i]==1]=i

confusion_matrix = np.zeros((len(header),len(header)))
for i in range(len(list_truth)):
    actual = int(list_truth[i])
    predicted = int(list_pred[i])
    confusion_matrix[actual, predicted] = confusion_matrix[actual, predicted]+1

norm_conf_matrix=np.zeros((len(header),len(header)))
for i in range(len(confusion_matrix)):
    for j in range(len(confusion_matrix)):
        norm_conf_matrix[i,j]=confusion_matrix[i,j]/np.sum(confusion_matrix[i])
  
from sklearn.metrics import f1_score
list_pred=list_pred.astype(int)
list_truth=list_truth.astype(int)
f1=f1_score(list_truth, list_pred, average='weighted')

fig, ax = plt.subplots()
im = ax.imshow(norm_conf_matrix)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(header)), labels=header)
ax.set_yticks(np.arange(len(header)), labels=header)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(header)):
    for j in range(len(header)):
        text = ax.text(j, i, np.round(norm_conf_matrix[i, j],2),
                       ha="center", va="center", color="b")
        
plt.title('F1-Score='+str(f1)[:5])
plt.xlabel('True')
plt.ylabel('Predicted')
fig.tight_layout()
plt.show()
