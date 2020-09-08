# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 22:34:46 2020

@author: Jacob
"""
import numpy as np
import matplotlib.pyplot as plt


#######################################
### Confidence interval definitions ### 
#######################################


## Confidence Interval

ratings  = np.array([1, 3, 1, 2, 3, 2, 5, 3, 2, 1, 3, 4])


## Jackknife Interval


## Bootstrap Interval

import numpy as np

ratings  = np.array([1, 3, 1, 2, 3, 2, 5, 3, 2, 1, 3, 4])

ratings

sample_mean = []
for _ in range(10000):  #so B=10000
    sample_n = np.random.choice(ratings, size = len(ratings))
    sample_mean.append(sample_n.mean())
    
plt.hist(sample_mean)

sample_mean
mu_rat = ratings.mean()


sample_mean - mu_rat
