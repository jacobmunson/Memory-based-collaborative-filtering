#! python3
# -*- coding: utf-8 -*-
from DataHelper import *
from EvaluationHelper import *


class UBCollaborativeFilter(object):
    def __init__(self):
        self.SimilityMatrix = None
        self.truerating = []
        self.predictions = []
        self.train_data_matrix = None
        self.RMSE = dict()
        self.MAE = dict()
        self.UserMeanMatrix = None


    def getRating(self, Train_data_matrix, userId, simility_matrix, neighborset):
        simSums = numpy.sum(simility_matrix[neighborset])  #
        averageOfUser = self.UserMeanMatrix[userId]  #
        jiaquanAverage = (Train_data_matrix[neighborset] - self.UserMeanMatrix[neighborset]).dot(simility_matrix[neighborset]) 
        #jiaquanAverage = numpy.mean(Train_data_matrix[neighborset])
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
        print("UBCF  K = %d, RMSE: %f, MAE: %f" % (K, self.RMSE[K], self.MAE[K]))
