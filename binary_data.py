import os
import csv
import json
import sys
import glob
import pandas as pd
import numpy as np

csv_path = 'csv_coords_files'
csv_files = glob.glob(csv_path + "/*.csv")

for file in csv_files:
    temp = pd.read_csv(file)
    #print(temp.loc[:, temp.columns=='label'].iloc[0], temp.loc[:, temp.columns=='label'].iloc[0,0])
    label = temp.loc[:, temp.columns=='label'].iloc[0,0]
    if label == 1:
        temp.loc[:, temp.columns=='label'] = 1
    else:
        temp.loc[:, temp.columns=='label'] = 0
    temp.to_csv('csv_coords_files_' + str(1) + '/' + file.split('/')[-1], index = False)    

for file in csv_files:
    temp = pd.read_csv(file)
    label = temp.loc[:, temp.columns=='label'].iloc[0,0]
    if label == 2:
        temp.loc[:, temp.columns=='label'] = 1 
    else:
        temp.loc[:, temp.columns=='label'] = 0
    temp.to_csv('csv_coords_files_' + str(2) + '/' + file.split('/')[-1], index = False)

for file in csv_files:
    temp = pd.read_csv(file)
    label = temp.loc[:, temp.columns=='label'].iloc[0,0]
    if label == 3:
        temp.loc[:, temp.columns=='label'] = 1
    else:
        temp.loc[:, temp.columns=='label'] = 0
    temp.to_csv('csv_coords_files_' + str(3) + '/' + file.split('/')[-1], index = False)
    
for file in csv_files:
    temp = pd.read_csv(file)
    label = temp.loc[:, temp.columns=='label'].iloc[0,0]
    if label == 4:
        temp.loc[:, temp.columns=='label'] = 1
    else:
        temp.loc[:, temp.columns=='label'] = 0
    temp.to_csv('csv_coords_files_' + str(4) + '/' + file.split('/')[-1], index = False)