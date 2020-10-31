#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:20:50 2019

@author: sanjeevkumar
"""

import numpy as np
import pandas as pd
import seaborn as sb


dataset = pd.read_csv('Facebook_metrics/dataset_Facebook.csv',delimiter=';')
del dataset['Page total likes']
del dataset['Type']
del dataset['Paid']
del dataset['Total Interactions']
del dataset['Category']
dataset = dataset.dropna(how='any',axis=0)
#dataset = dataset.T

p = dataset.isna().sum(axis=1)

#print(dataset)
#corr_Mat = np.corrcoef(dataset)
corr_Mat = dataset.corr()
print(corr_Mat)
#print(corr_Mat.shape)
sb.heatmap(corr_Mat)

#calculate eigen values and eigen vectors

#u, s, v = np.linalg.svd(corr_Mat)
#print(u,s,v)
w, v = np.linalg.eig(corr_Mat)
#print(w,v)
w_root = np.sqrt(w)

L_complete = v * w_root
h_square = np.sum(np.square(L_complete),axis=1)
shai = 1-h_square

m = 4
L = L_complete[:,0:m] 

L_transpose = L.T


residual_error = corr_Mat - np.dot(L,L_transpose) - shai

