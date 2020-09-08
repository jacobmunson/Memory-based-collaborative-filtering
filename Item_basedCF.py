#! python3
# -*- coding: utf-8 -*-
from DataHelper import *
from EvaluationHelper import *
import math


class IBCollaborativeFilter(object):
    def __init__(self):
        self.SimilityMatrix = None
        self.ItemMeanMatrix = None
        self.truerating = []
        self.predictions = []
        self.train_data_matrix = None
        self.RMSE = dict()
        self.MAE = dict()
        
        self.CI_upper = []
        self.CI_lower = []
        
        self.num_nhbr_actual = []
        self.sd_terms = []

                
        


    def getRating(self, Train_data_matrix, itemId, simility_matrix, knumber = 20):
        neighborset = get_K_Neighbors(Train_data_matrix, simility_matrix, knumber) # get the neighbor set
        simSums = numpy.sum(simility_matrix[neighborset])  
        # print(simSums) Sum of similarities for weighted average
        averageOfUser = self.ItemMeanMatrix[itemId]  # userId
        jiaquanAverage = (Train_data_matrix[neighborset] - self.ItemMeanMatrix[neighborset]).dot(simility_matrix[neighborset])
        # prediction function: p_ui = [ neighbor values - mean(of those neighbors - by neighbors) ] * sim(neighbors)

        #jiaquanAverage = numpy.mean(Train_data_matrix[neighborset]) # mean of neighbor values
        
        self.num_nhbr_actual.append(len(neighborset)) # Train_data_matrix[neighborset]
        
        if len(neighborset) > 1:
            self.sd_terms.append(np.std(Train_data_matrix[neighborset]))
        else:
            self.sd_terms.append(math.nan)
        

        # Train_data_matrix[neighborset] # neighbor values - use these for CI, JK CI, BS CI
        
        if simSums == 0 and len(neighborset) > 1:
            self.CI_lower.append(averageOfUser - np.std(Train_data_matrix[neighborset]))
            self.CI_upper.append(averageOfUser + np.std(Train_data_matrix[neighborset]))
             # if sum of similarities is 0, then return average of the users
        elif simSums != 0 and len(neighborset) > 1:
            pred = averageOfUser + jiaquanAverage / simSums
            self.CI_lower.append(pred - np.std(Train_data_matrix[neighborset]))
            self.CI_upper.append(pred + np.std(Train_data_matrix[neighborset]))
        elif simSums == 0 and len(neighborset) <= 1:
            self.CI_lower.append(math.nan)
            self.CI_upper.append(math.nan)
        elif simSums != 0 and len(neighborset) <= 1:
            self.CI_lower.append(math.nan)
            self.CI_upper.append(math.nan)
        


        
        #print(jiaquanAverage)
         
        if simSums == 0:
            return averageOfUser # if sum of similarities is 0, then return average of the users
        else:
            return averageOfUser + jiaquanAverage / simSums # jiaquanAverage #

    def doEvaluate(self, testDataMatrix, K):
        a, b = testDataMatrix.nonzero()
        for userIndex, itemIndex in zip(a, b):
            prerating = self.getRating(self.train_data_matrix[userIndex], itemIndex, self.SimilityMatrix[itemIndex],K)  # 基于训练集预测用户评分(用户数目<=K)
            self.truerating.append(testDataMatrix[userIndex][itemIndex])
            self.predictions.append(prerating)
            # print(len(self.predictions))
        self.RMSE[K] = RMSE(self.truerating, self.predictions)
        self.MAE[K] = MAE(self.truerating, self.predictions)
        print("IBCF Results:  K = %d, RMSE: %f, MAE: %f" % (K, self.RMSE[K], self.MAE[K]))


