#! python3
# -*- coding: utf-8 -*-
import argparse
import datetime
import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from Item_basedCF import *
from ThreadWithReturn import *
from User_basedCF import *

MovieLensData = {
    1: 'Datas/ml-100k/u.data',
    2: 'Datas/ml-1M/ratings.dat',
    3: 'Datas/ml-10M/ratings.dat',
    4: 'Datas/ml-20m/ratings.csv'
}

use_cosine = True 
use_pearson = False
dataset = "ml-100k"
predictor_type = "use_intercept_weighted" # "use_intercept_weighted" # "use_unweighted" # "use_weighted"

def parseargs(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--ratings",
                        type = str,
                        default = dataset,
                        help = "Ratings file")

    parser.add_argument("--testsize",
                        type = float,
                        default = 0.2,
                        help = "Percentage of test data")

    return parser.parse_args()


if __name__ == '__main__':
    myparser = parseargs()
    startTime = datetime.datetime.now()
    MyData = LoadMovieLensData(myparser.ratings)
    MyUBCF = UBCollaborativeFilter()
    MyIBCF = IBCollaborativeFilter()
    train_data, test_data = train_test_split(MyData, test_size = myparser.testsize)
    print("Data snippet...")
    print(type(train_data))
    print(MyData.head())
    n_users = MyData.user_id.max()
    n_items = MyData.item_id.max()
    print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

    test1 = ThreadWithReturnValue(target=DataFrame2Matrix, args=(n_users, n_items, train_data))
    test2 = ThreadWithReturnValue(target=DataFrame2Matrix, args=(n_users, n_items, test_data))

    test1.start()
    test2.start()
 
    train_data_matrix = test1.join()
    test_data_matrix = test2.join()
    
    MyUBCF.train_data_matrix = train_data_matrix
    MyIBCF.train_data_matrix = train_data_matrix
    MyUBCF.test_data_matrix = test_data_matrix
    MyIBCF.test_data_matrix = test_data_matrix
    if predictor_type == "use_unweighted":
        print("Using unweighted averaging...")
    elif predictor_type == "use_weighted":
        print("Using weighted averaging")
    elif predictor_type == "use_intercept_weighted":
        print("Using intercept weighted predictor")
    
    if use_cosine:
        print("Using cosine similarity...")
        print("Computing UBCF similarity matrix...")
        MyUBCF.SimilityMatrix = cosine_similarity(train_data_matrix)
        print("Computing IBCF similarity matrix...")
        MyIBCF.SimilityMatrix = cosine_similarity(train_data_matrix.T)
    elif use_pearson: #### NEEDS TO BE FIXED AND TESTED
        print("Using pearson similarity...")
        print("Computing UBCF similarity matrix...")
        print("NEED TO FIX")
        MyUBCF.SimilityMatrix = cosine_similarity(train_data_matrix)
        print("Computing IBCF similarity matrix...")
        MyIBCF.SimilityMatrix = cosine_similarity(train_data_matrix.T)
        
    
    print("Computing User Mean matrix...")
    MyUBCF.UserMeanMatrix = numpy.true_divide(MyUBCF.train_data_matrix.sum(1), (MyUBCF.train_data_matrix != 0).sum(1))  
    print("Computing Item Mean matrix...")
    MyIBCF.ItemMeanMatrix = numpy.true_divide(MyUBCF.train_data_matrix.sum(0), (MyUBCF.train_data_matrix != 0).sum(0))  
    
    MyIBCF.ItemMeanMatrix[np.isnan(MyIBCF.ItemMeanMatrix)] = 0
    KList = [5, 10, 15, 25, 50, 75, 100, 125] # 5, 10, 15, 25, 50, 75, 100, 125
    df_total_ibcf = pd.DataFrame()
    df_total_ubcf = pd.DataFrame()
    
    for i in range(len(KList)):
        startTimeLoop = datetime.datetime.now()
        df_ibcf = pd.DataFrame()
        df_ubcf = pd.DataFrame()
        
        MyUBCF.truerating = []
        MyUBCF.predictions = []
        MyIBCF.truerating = []
        MyIBCF.predictions = []
        
        # Standard CI
        MyIBCF.CI_upper = []
        MyIBCF.CI_lower = []
        MyUBCF.CI_upper = []
        MyUBCF.CI_lower = []
        
        # KNN CI
        MyIBCF.CI_knn_upper = []
        MyIBCF.CI_knn_lower = []
        MyUBCF.CI_knn_upper = []
        MyUBCF.CI_knn_lower = []
        
        # JK CI 
        MyIBCF.CI_jk_upper = []
        MyIBCF.CI_jk_lower = []
        MyUBCF.CI_jk_upper = []
        MyUBCF.CI_jk_lower = []
        
        MyIBCF.num_nhbr_actual = []
        MyIBCF.sd_terms = []
        MyUBCF.num_nhbr_actual = []
        MyUBCF.sd_terms = []

        print("Starting UBCF...")
        t1 = Thread(target=MyUBCF.doEvaluate, args=(test_data_matrix, KList[i], predictor_type))
        t1.start()
        t1.join()
        
        print("Starting IBCF...")
        t2 = Thread(target=MyIBCF.doEvaluate, args=(test_data_matrix, KList[i], predictor_type))
        t2.start()
        t2.join()
        
        
                    
        #difference = []
        #zip_object = zip(list1, list2)
        #for list1_i, list2_i in zip_object:
        #    difference.append(abs(list1_i-list2_i))
        
        df_ibcf["pred"] = MyIBCF.predictions
        df_ibcf["actual"] = MyIBCF.truerating
        df_ubcf["pred"] = MyUBCF.predictions
        df_ubcf["actual"] = MyUBCF.truerating
        
        # successfully recovering RMSE and MAE
        #print(sum(difference)/len(difference))        
        #print(math.sqrt(sum(np.array(difference) ** 2)/len(difference)))
        
        #print("Attaching bounds...")
        print("Attaching CI bounds...")
        df_ibcf["ci_upper"] = MyIBCF.CI_upper
        df_ibcf["ci_lower"] = MyIBCF.CI_lower
        df_ubcf["ci_upper"] = MyUBCF.CI_upper
        df_ubcf["ci_lower"] = MyUBCF.CI_lower 

        print("Attaching KNN CI bounds...")        
        df_ibcf["ci_knn_upper"] = MyIBCF.CI_knn_upper
        df_ibcf["ci_knn_lower"] = MyIBCF.CI_knn_lower
        df_ubcf["ci_knn_upper"] = MyUBCF.CI_knn_upper
        df_ubcf["ci_knn_lower"] = MyUBCF.CI_knn_lower 
        
        print("Attaching JK CI bounds...")
        df_ibcf["ci_jk_upper"] = MyIBCF.CI_jk_upper
        df_ibcf["ci_jk_lower"] = MyIBCF.CI_jk_lower
        df_ubcf["ci_jk_upper"] = MyUBCF.CI_jk_upper
        df_ubcf["ci_jk_lower"] = MyUBCF.CI_jk_lower 
        
        
   
        #print("IBCF neighbors")
        df_ibcf["num_nbhr"] = MyIBCF.num_nhbr_actual
        df_ibcf["K"] = KList[i]
        
        #print("UBCF neighbors")
        df_ubcf["num_nbhr"] = MyUBCF.num_nhbr_actual
        df_ubcf["K"] = KList[i]
        
        #print("sd terms IBCF")
        df_ibcf["sd_terms"] = MyIBCF.sd_terms
        #print("sd terms UBCF")
        df_ubcf["sd_terms"] = MyUBCF.sd_terms

        #print("Combining data...")
        df_total_ubcf = pd.concat([df_total_ubcf, df_ubcf], ignore_index=True)
        df_total_ibcf = pd.concat([df_total_ibcf, df_ibcf], ignore_index=True)
        
        endTime = datetime.datetime.now()
        print("Total Loop time: %d seconds " % (endTime - startTimeLoop).seconds, "| Total Cost time: %d seconds " % (endTime - startTime).seconds)
        
        Savetxt("Docs/%s/UBCF %s %1.1f.txt" % (myparser.ratings, myparser.ratings, myparser.testsize),
                "UBCF  K=%d\tRMSE:%f\tMAE:%f\t" % (KList[i], MyUBCF.RMSE[KList[i]], MyUBCF.MAE[KList[i]]))
        Savetxt('Docs/%s/IBCF %s %1.1f.txt' % (myparser.ratings, myparser.ratings, myparser.testsize),
                "IBCF  K=%d\tRMSE:%f\tMAE:%f\t" % (KList[i], MyIBCF.RMSE[KList[i]], MyIBCF.MAE[KList[i]]))
    
    print('\a')
    df_total_ibcf["model"] = "IBCF"
    df_total_ubcf["model"] = "UBCF"
    df_total = pd.concat([df_total_ibcf, df_total_ubcf], ignore_index=True)
    print('\a')
    # Check performance by plotting train and test errors
    plt.plot(KList, list(MyUBCF.RMSE.values()), marker='o', label='RMSE')
    plt.plot(KList, list(MyUBCF.MAE.values()), marker='v', label='MAE')
    plt.title('The Error of UBCF in MovieLens ' + myparser.ratings)
    plt.xlabel('K')
    plt.ylabel('value')
    plt.legend()
    plt.grid()
    plt.savefig('Docs/%s/UBCF %s %1.1f.png' % (myparser.ratings, myparser.ratings, myparser.testsize))
    plt.show()
    plt.gcf().clear()
    # Check performance by plotting train and test errors
    plt.plot(KList, list(MyIBCF.RMSE.values()), marker='o', label='RMSE')
    plt.plot(KList, list(MyIBCF.MAE.values()), marker='v', label='MAE')
    plt.title('The Error of IBCF in MovieLens ' + myparser.ratings)
    plt.xlabel('K')
    plt.ylabel('value')
    plt.legend()
    plt.grid()
    plt.savefig('Docs/%s/IBCF %s %1.1f.png' % (myparser.ratings, myparser.ratings, myparser.testsize))
    plt.show()


df_total.to_csv("results_ml100k_intecept_weighted.csv")
