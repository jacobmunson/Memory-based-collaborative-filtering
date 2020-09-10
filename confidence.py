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



## Bootstrap Interval
