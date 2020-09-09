#! python3
# -*- coding: utf-8 -*-
from DataHelper import *
from EvaluationHelper import *
import math
from confidence import *

class UBCollaborativeFilter(object):
    def __init__(self):
        self.SimilityMatrix = None
        self.truerating = []
        self.predictions = []
        self.train_data_matrix = None
        self.RMSE = dict()
        self.MAE = dict()
        self.UserMeanMatrix = None
        
        
        self.CI_upper = []
        self.CI_lower = []
        
        self.num_nhbr_actual = []
        self.sd_terms = []


    def getRating(self, Train_data_matrix, userId, simility_matrix, neighborset):
        simSums = numpy.sum(simility_matrix[neighborset])  #
        averageOfUser = self.UserMeanMatrix[userId]  #
        jiaquanAverage = (Train_data_matrix[neighborset] - self.UserMeanMatrix[neighborset]).dot(simility_matrix[neighborset]) 
        #jiaquanAverage = numpy.mean(Train_data_matrix[neighborset])
        
        
        self.num_nhbr_actual.append(len(neighborset)) # Train_data_matrix[neighborset]
        #print(np.std(Train_data_matrix[neighborset]))
        # Train_data_matrix[neighborset] # neighbor values - use these for CI, JK CI, BS CI
        if len(neighborset) > 1:
            self.sd_terms.append(np.std(Train_data_matrix[neighborset]))
        else:
            self.sd_terms.append(math.nan)
        

        # Train_data_matrix[neighborset] # neighbor values - use these for CI, JK CI, BS CI
        
        if simSums == 0 and len(neighborset) >= 2:
            self.CI_lower.append(averageOfUser - standard_ci(Train_data_matrix[neighborset]))
            self.CI_upper.append(averageOfUser + standard_ci(Train_data_matrix[neighborset]))
             # if sum of similarities is 0, then return average of the users
        elif simSums != 0 and len(neighborset) >= 2:
            pred = averageOfUser + jiaquanAverage / simSums
            self.CI_lower.append(pred - standard_ci(Train_data_matrix[neighborset]))
            self.CI_upper.append(pred + standard_ci(Train_data_matrix[neighborset]))
        elif simSums == 0 and len(neighborset) < 2:
            self.CI_lower.append(math.nan)
            self.CI_upper.append(math.nan)
        elif simSums != 0 and len(neighborset) < 2:
            self.CI_lower.append(math.nan)
            self.CI_upper.append(math.nan)
        
        
               
        if simSums == 0:
            return averageOfUser
        else:
            return averageOfUser + jiaquanAverage / simSums # jiaquanAverage # 

    def doEvaluate(self, testDataMatrix, K):
        a, b = testDataMatrix.nonzero()
        for userIndex, itemIndex in zip(a, b):
            neighborset = get_K_Neighbors(self.train_data_matrix[:, itemIndex], self.SimilityMatrix[userIndex], K)  
            prerating = self.getRating(self.train_data_matrix[:, itemIndex], userIndex, self.SimilityMatrix[userIndex], neighborset)  #
            self.truerating.append(testDataMatrix[userIndex][itemIndex])
            self.predictions.append(prerating)
            # print(len(self.predictions))
        self.RMSE[K] = RMSE(self.truerating, self.predictions)
        self.MAE[K] = MAE(self.truerating, self.predictions)
        print("UBCF Results:  K = %d, RMSE: %f, MAE: %f" % (K, self.RMSE[K], self.MAE[K]))
