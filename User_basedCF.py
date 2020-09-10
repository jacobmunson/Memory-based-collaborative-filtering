#! python3
# -*- coding: utf-8 -*-
from DataHelper import *
from EvaluationHelper import *
import math
from confidence import standard_ci
from confidence import knn_ci

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
        self.CI_knn_upper = []
        self.CI_knn_lower = []
        
        self.num_nhbr_actual = []
        self.sd_terms = []


    def getRating(self, Train_data_matrix, userId, simility_matrix, neighborset, pred_function = "use_intercept_weighted"):
        simSums = numpy.sum(simility_matrix[neighborset])  #
        averageOfUser = self.UserMeanMatrix[userId]  #
        
        if pred_function == "use_weighted":
            # statement
            # needs sims check

            if len(neighborset) > 0 and simSums != 0:
                pred = (Train_data_matrix[neighborset]).dot(simility_matrix[neighborset])
                pred = pred / simSums
            else:
                pred = averageOfUser   
                
            self.num_nhbr_actual.append(len(neighborset))
            if len(neighborset) > 1:
                self.sd_terms.append(np.std(Train_data_matrix[neighborset]))
            else:
                self.sd_terms.append(math.nan)

            # for CI
            if(len(neighborset) >= 2):
                ci_me = standard_ci(Train_data_matrix[neighborset])
            # for KNN CI
            if(len(neighborset > 2)):
                ci_knn_me = knn_ci(Train_data_matrix[neighborset]) # need to put in all the conditions below but for knn CI

            # for CI
            if len(neighborset) >= 2:
                self.CI_lower.append(pred - ci_me)
                self.CI_upper.append(pred + ci_me)
            elif len(neighborset) < 2:
                self.CI_lower.append(math.nan)
                self.CI_upper.append(math.nan)

            
            # for KNN CI
            if len(neighborset) > 2:
                self.CI_knn_lower.append(pred - ci_knn_me)
                self.CI_knn_upper.append(pred + ci_knn_me)
            elif len(neighborset) <= 2:
                self.CI_knn_lower.append(math.nan)
                self.CI_knn_upper.append(math.nan)
            return pred
            
            
        elif pred_function == "use_unweighted":
            # just average the neighborset
            
            
            if len(neighborset) > 0:
                pred = np.mean(Train_data_matrix[neighborset])
            else:
                pred = averageOfUser
                
            self.num_nhbr_actual.append(len(neighborset))
            if len(neighborset) > 1:
                self.sd_terms.append(np.std(Train_data_matrix[neighborset]))
            else:
                self.sd_terms.append(math.nan)

            # for CI
            if(len(neighborset) >= 2):
                ci_me = standard_ci(Train_data_matrix[neighborset])
            # for KNN CI
            if(len(neighborset > 2)):
                ci_knn_me = knn_ci(Train_data_matrix[neighborset]) # need to put in all the conditions below but for knn CI

            # for CI
            if len(neighborset) >= 2:
                self.CI_lower.append(pred - ci_me)
                self.CI_upper.append(pred + ci_me)
            elif len(neighborset) < 2:
                self.CI_lower.append(math.nan)
                self.CI_upper.append(math.nan)

            
            # for KNN CI
            if len(neighborset) > 2:
                self.CI_knn_lower.append(pred - ci_knn_me)
                self.CI_knn_upper.append(pred + ci_knn_me)
            elif len(neighborset) <= 2:
                self.CI_knn_lower.append(math.nan)
                self.CI_knn_upper.append(math.nan)

                
                
            #print("here")
            return pred
            
            print("here")
        elif pred_function == "use_intercept_weighted":
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
            
            # for CI
            if(len(neighborset) >= 2):
                ci_me = standard_ci(Train_data_matrix[neighborset])
            # for KNN CI
            if(len(neighborset > 2)):
                ci_knn_me = knn_ci(Train_data_matrix[neighborset])
                
            # for CI
            if simSums == 0 and len(neighborset) >= 2:
                self.CI_lower.append(averageOfUser - ci_me)
                self.CI_upper.append(averageOfUser + ci_me)
                 # if sum of similarities is 0, then return average of the users
            elif simSums != 0 and len(neighborset) >= 2:
                pred = averageOfUser + jiaquanAverage / simSums
                self.CI_lower.append(pred - ci_me)
                self.CI_upper.append(pred + ci_me)
            elif simSums == 0 and len(neighborset) < 2:
                self.CI_lower.append(math.nan)
                self.CI_upper.append(math.nan)
            elif simSums != 0 and len(neighborset) < 2:
                self.CI_lower.append(math.nan)
                self.CI_upper.append(math.nan)
            
            # for KNN CI
            if simSums == 0 and len(neighborset) > 2:
                self.CI_knn_lower.append(averageOfUser - ci_knn_me)
                self.CI_knn_upper.append(averageOfUser + ci_knn_me)
                 # if sum of similarities is 0, then return average of the users
            elif simSums != 0 and len(neighborset) > 2:
                pred = averageOfUser + jiaquanAverage / simSums
                self.CI_knn_lower.append(pred - ci_knn_me)
                self.CI_knn_upper.append(pred + ci_knn_me)
            elif simSums == 0 and len(neighborset) <= 2:
                self.CI_knn_lower.append(math.nan)
                self.CI_knn_upper.append(math.nan)
            elif simSums != 0 and len(neighborset) <= 2:
                self.CI_knn_lower.append(math.nan)
                self.CI_knn_upper.append(math.nan)
            
                   
            if simSums == 0:
                return averageOfUser
            else:
                return averageOfUser + jiaquanAverage / simSums # jiaquanAverage # 

    def doEvaluate(self, testDataMatrix, K, predictor):
        a, b = testDataMatrix.nonzero()
        for userIndex, itemIndex in zip(a, b):
            neighborset = get_K_Neighbors(self.train_data_matrix[:, itemIndex], self.SimilityMatrix[userIndex], K)  
            prerating = self.getRating(self.train_data_matrix[:, itemIndex], userIndex, self.SimilityMatrix[userIndex], neighborset, predictor)  #
            self.truerating.append(testDataMatrix[userIndex][itemIndex])
            self.predictions.append(prerating)
            # print(len(self.predictions))
        self.RMSE[K] = RMSE(self.truerating, self.predictions)
        self.MAE[K] = MAE(self.truerating, self.predictions)
        print("UBCF Results:  K = %d, RMSE: %f, MAE: %f" % (K, self.RMSE[K], self.MAE[K]))
