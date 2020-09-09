# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 18:20:12 2020

@author: Jacob
"""

################################################################
### Slapped together implementation of UBCF ####################
# Computing everything on the fly - no storing of similarities #
################################################################

import pandas as pd
from sklearn.model_selection import train_test_split
header = ['user_id', 'item_id', 'rating', 'timestamp']
ml_data = pd.read_table('Datas/ml-100k/u.data', header=None, names=header)


def SplitData(DataSet, SplitRate = 0.25):
    TrainData, TestData = train_test_split(DataSet, test_size=SplitRate)
    return TrainData, TestData

D_train, D_test = SplitData(ml_data)

type(D_train)
D_train
D_test

import argparse
import datetime
import math
import numpy as np


startTimeLoop = datetime.datetime.now()
preds = []
K = 5
for i in range(len(D_test.index)):
    print(i)
    obs_i = D_train.iloc[i]
    obs_i    
    obs_i_item = obs_i["item_id"]
    obs_i_user = obs_i["user_id"]
    corater_df = D_train[D_train["item_id"] == obs_i_item]
    coraters = np.array(corater_df["user_id"])
    #coraters = D_train[D_train["user_id"].isin(coraters)]
    obs_i_user
    coraters = np.setdiff1d(coraters, obs_i_user)
    coraters

    obs_i_user_df = D_train[D_train["user_id"] == obs_i_user]
    sim_ij = []
    
    for j in range(len(coraters)):
        
        coraters[j]
        nhbr_j = D_train[D_train["user_id"] == coraters[j]]
        
        nhbr_j
        obs_i_user_df
        nhbr_j
        df_ij = pd.merge(obs_i_user_df, nhbr_j, how = "inner", on = "item_id")
        df_ij
        v1 = df_ij[['rating_x']]
        v2 = df_ij[['rating_y']]
   
        def pearson(v1, v2):
            v1 = np.array(v1)
            v2 = np.array(v2)
            mu_v1 = v1.mean()
            mu_v2 = v2.mean()
            pc = sum((v1 - mu_v1)*(v2 - mu_v2))/sqrt(sum(pow(v1 - mu_v1, 2)) * sum(pow(v2 - mu_v2, 2)))
            return pc
    
        pearson(v1, v2)[0]
        pc_ij = pearson(v1, v2)[0]
        pc_ij
        sim_ij.append(pc_ij)

    corater_df = corater_df[corater_df["user_id"].isin(coraters)]
    corater_df["sim"] = sim_ij
    corater_df = corater_df.sort_values(by = "sim", ascending = False)
    corater_df = corater_df[0:(K)]
    
    r_ui = corater_df["rating"].mean()
    preds.append(r_ui)    
endTimeLoop = datetime.datetime.now()
print("Total Loop time: %d seconds " % (endTimeLoop - startTimeLoop).seconds)



