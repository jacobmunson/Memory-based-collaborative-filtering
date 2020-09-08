# -*- coding: utf-8 -*-
import pandas as pd
import pickle
import heapq
import numpy as np
from sklearn.model_selection import train_test_split


def SaveData2pkl(DictData, FilePath='Datas/Mydata.pkl', mode='wb'):
    pkl_file = open(FilePath, mode)
    try:
        pickle.dump(DictData, pkl_file, protocol=2)
        return True
    except:
        return False
    finally:
        pkl_file.close()


def SaveData2cvs(MatrixData, FilePath='Datas/Mydata.pkl', Thisdelimiter=','):
    try:
        np.savetxt(FilePath, MatrixData, delimiter=Thisdelimiter)
        return True
    except Exception as e:
        print(repr(e))
        return False


def LoadData4pkl(FilePath='Datas/Mydata.pkl', mode='rb'):
    pkl_file = open(FilePath, mode)
    try:
        DataDict = pickle.load(pkl_file)
        return DataDict
    except:
        return None
    finally:
        pkl_file.close()


def LoadData4cvs(FilePath='Datas/Mydata.pkl', Thisdelimiter=',', mode='rb'):
    try:
        my_matrix = np.loadtxt(open(FilePath, mode), delimiter=Thisdelimiter, skiprows=0)
        return my_matrix
    except:
        return None


def LoadDoubanData(FilePath='Datas/Mydata.pkl'):
    LineNum = 1
    UserRating = dict()
    UserIndex = LoadData4pkl('Datas/UserIndex.pkl')
    ItemIndex = LoadData4pkl('Datas/ItemIndex.pkl')
    for line in open(FilePath, 'r', encoding='UTF-8'):
        LineNum += 1
        if len(line.rstrip('\n')) == 0:
            continue
        linelist = line.split(',')
        UserID = int(linelist[0])
        MovieID = int(linelist[1])
        Rating = float(linelist[2])
        tags = str(linelist[4].rstrip('\n')).lower()
        UserRating.setdefault(UserIndex[UserID], {})
        UserRating[UserIndex[UserID]][ItemIndex[MovieID]] = Rating
        print("第%d行数据：" % LineNum)
        # if z>30000: break
    return UserRating


def LoadMovieLensData(FileType='ml-100k'):
    """
    :param FileType:
    :return: DataFrame
    """
    if FileType == 'ml-100k':
        header = ['user_id', 'item_id', 'rating', 'timestamp']
        data = pd.read_table('Datas/ml-100k/u.data', header=None, names=header)
    elif FileType == 'ml-1M':
        header = ['user_id', 'item_id', 'rating', 'timestamp']
        data = pd.read_table('Datas/ml-1M/ratings.dat', sep="::", header=None, names=header)
    elif FileType == 'ml-10M':
        header = ['user_id', 'item_id', 'rating', 'timestamp']
        data = pd.read_table('Datas/ml-10M/ratings.dat', sep="::", header=None, names=header, engine='python')
    elif FileType == 'ml-20M':
        data = pd.read_csv('Datas/ml-20m/ratings.csv')
    else:
        data = None
    return data


def SpiltData(DataSet, SpiltRate=0.25):
    TrainData, TestData = train_test_split(DataSet, test_size=SpiltRate)
    return TrainData, TestData


def get_K_Neighbors(Train_data_matrix, simility_matrix, knumber=10):
    SIM = simility_matrix.copy()
    zeroset = np.where(Train_data_matrix == 0)
    SIM[zeroset] = 0
    myresult = sparse_argsort(-SIM)[0:knumber]
    return myresult


def sparse_argsort(arr):
    indices = np.nonzero(arr)[0]
    return indices[np.argsort(arr[indices])]


# write in txt Appending mode
def Savetxt(FilePath, message='', mode='a'):
    file_object = open(FilePath, mode)
    file_object.write(message + '\n')
    file_object.close()


def DataFrame2Matrix(n_users, n_items, dataframe):
    
    #import h5py
    #hdf5_store = h5py.File("./cache.hdf5", "a")
    #hdf5_store_trans = h5py.File("./cache.hdf5", "b")
    #train_data_matrix = hdf5_store.create_dataset("train_data_matrix", (n_users, n_items), compression="gzip")
    #train_data_matrix = hdf5_store.require_dataset("train_data_matrix", (n_users, n_items), compression="gzip", dtype= np.float32,  fillvalue = 0)
    
    #train_data_matrix_transpose = hdf5_store_trans.require_dataset("train_data_matrix_transpose", (n_items, n_users), compression="gzip", dtype= np.float32,  fillvalue = 0)
    
    train_data_matrix = np.zeros((n_users, n_items))
    #from scipy import sparse
    #train_data_matrix = sparse.coo_matrix(n_user,n_items)
    for line in dataframe.itertuples():
        train_data_matrix[line[1] - 1, line[2] - 1] = line[3]
        #train_data_matrix_transpose[line[2] - 1, line[1] - 1] = line[3]
    return train_data_matrix


def get_N_Recommends(neighborset, userIndex, Train_data_matrix, simility_matrix, Nnumber=10):
    myTrain_data_matrix = Train_data_matrix.copy()
    if len(neighborset) != 0:
        # for i in neighborset:
        #    myTrain_data_matrix[i] = myTrain_data_matrix[i] * simility_matrix[userIndex][i]
        myTrain_data_matrix[neighborset] = np.multiply(myTrain_data_matrix[neighborset].T, simility_matrix[userIndex][neighborset]).T
        watched = myTrain_data_matrix[userIndex].nonzero()
        myTrain_data_matrix[:, watched] = 0
        recommendset = myTrain_data_matrix[neighborset]
        teat1 = np.where(recommendset >= heapq.nlargest(Nnumber, recommendset.flatten())[-1])
        return teat1[1]
    else:  # 冷启动处理
        watched = myTrain_data_matrix[userIndex].nonzero()
        myTrain_data_matrix[:, watched] = 0
        teat1 = np.vstack(np.unravel_index(np.argpartition(myTrain_data_matrix.flatten(), -2)[-Nnumber:], myTrain_data_matrix.shape)).T
        return teat1[:, 1]
