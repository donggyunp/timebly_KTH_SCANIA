from numba import jit, cuda
import os
import csv
import json
import sys
import datetime
import logging
import time
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.utils import to_time_series_dataset
from tslearn.svm import TimeSeriesSVC


@jit(target_backend='cuda') 
def tslearn_gpu(feature, label):
    C_grid = [0.8, 0.9, 1.0, 1.1, 1.2,]
    gamma_grid = [0.05, 0.075, 0.1, 0.125, 0.15]
    for i in C_grid:
        for j in gamma_grid:
            print(i, j)
            clf = TimeSeriesSVC(C = i, kernel = 'gak', gamma = j)
            # the other kernels are not available for different lengths of time series.
            clf.fit(ts_dataset_X, ts_dataset_y)
            scores = cross_val_score(clf, ts_dataset_X, ts_dataset_y, cv=5)
            print(scores, sum(scores)/5)

csv_path = 'csv_coords_files'
csv_files = glob.glob(csv_path + "/*.csv")

ts_dataset_X, ts_dataset_y  = [], []

for file in csv_files:
    temp = pd.read_csv(file)
    ts_dataset_X.append(temp.loc[:, temp.columns!='label'])
    ts_dataset_y.append(temp['label'].iloc[0])

ts_dataset_X = to_time_series_dataset(ts_dataset_X)
ts_Dataset_X = TimeSeriesScalerMinMax().fit_transform(ts_dataset_X)

tslearn_gpu(ts_dataset_X, ts_dataset_y)