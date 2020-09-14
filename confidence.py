# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 22:34:46 2020

@author: Jacob
"""
#######################################
### Confidence interval definitions ### 
#######################################
import math
from scipy.stats import t
import numpy as np


#########################
## Confidence Interval ##
#########################
def standard_ci(ratings): # input will be Train_data_matrix[neighborset]
    std_dev = np.std(ratings)
    n = len(ratings)
    if n >= 30:
        multi = 1.96
    else: 
        multi = t.interval(alpha = 0.975, df = n-1)[1] 
    return multi * std_dev/math.sqrt(n)


def knn_ci(ratings): # input will be Train_data_matrix[neighborset]
    std_dev = np.std(ratings)
    n = len(ratings)
    multi = t.interval(alpha = 0.975, df = ((n - 2)/2))[1] 
    return multi * std_dev/math.sqrt(n)

########################
## Jackknife Interval ##
########################
def jackknife_ci(ratings, sim, use_unweighted, use_weighted):

    if use_weighted == True:
        mu_weighted_ratings = sum(ratings * sim) / sum(abs(sim))
    if use_unweighted == True:
        mu_ratings = np.mean(ratings)

    mu_jk_samples, n = [] , len(ratings)
    index = np.arange(n)
    
    for i in range(n):
        if use_unweighted == True:
            jk_sample = ratings[index != i]
            mu_jk_sample = np.mean(jk_sample)
            #print(mu_jk_sample)
            mu_jk_samples.append(mu_jk_sample)
            
        if use_weighted == True:
            jk_sample = ratings[index != i] * sim[index != i]
            mu_jk_sample = sum(jk_sample)/(sum(abs(sim[index != i])))
            #print(sum(jk_sample)/sum(abs(sim[index != i])))
            #print(mu_jk_sample)
            mu_jk_samples.append(mu_jk_sample)

    if use_unweighted == True:
        se_jk = np.sqrt(sum(pow((mu_ratings - mu_jk_samples),2))*(n-1)/n)
        #print(se_jk)
    if use_weighted == True:
        se_jk = np.sqrt(sum(pow((mu_weighted_ratings - mu_jk_samples),2))*(n-1)/n)
        #print(se_jk)   
    
    if n >= 30:
        multi = 1.96
    else: 
        multi = t.interval(alpha = 0.975, df = n-1)[1] 
    return multi * se_jk



## Bootstrap Interval
