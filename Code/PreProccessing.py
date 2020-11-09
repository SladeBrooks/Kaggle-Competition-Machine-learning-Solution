# -*- coding: utf-8 -*-
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MaxAbsScaler
import numpy as np


"""
normalises each feature to L1 normalisation

Note: not used
"""
def norm(data):
    return np.array(normalize(data,norm = 'l1',axis = 0))

#scales the training and test data to maximum 1, scales test data using the same scale as the training
def scaler(train,test):
    min_max_scaler = MaxAbsScaler()
    train_minmax = min_max_scaler.fit_transform(train)
    test_minmax = min_max_scaler.transform(test)
    return train_minmax, test_minmax

"""
My domain adaption technique takes the traning data, and dest data, uses outputs to split the training 
data by class, the calculates the mean per feature per class. Test proportions is used to determine
the expected mean of the test data, if the mean of the test data deviates from the expected mean
then that feature is removed.

returns a reduced feature set for training and test data where  each mean feature from test data does not
deviate from the expected mean by more than mean value +- 0.5 * standard deviation
"""
def domain_adaption(train,outputs,test):
    #split data by class, use bigger and smaller than 0.5 to handle floats with low precision
    pos_data = np.array([train[i] for i in range(len(train)) if outputs[i] > 0])
    neg_data = np.array([train[i] for i in range(len(train)) if outputs[i] <= 0])
    
    #computes mean value per feature for pos and neg data
    pos_means = pos_data.mean(axis = 0)
    neg_means = neg_data.mean(axis = 0)
    test_means =test.mean(axis = 0)
    
    #the proportions of test data
    pos_prop = 0.3233
    neg_prop = 0.6767
    
    #use prior knowledge or proportions to calculate the expected mean per feature
    expected_test_means = (pos_means*pos_prop)+(neg_means*neg_prop)
    
    #calculate the difference between expected feature mean values and actual feature mean values
    feature_diss = np.array([(expected_test_means[i]-test_means[i]) for i in range(len(test_means))])
    #flips all negative values, this could be replaced by squaring values but these become to small so i do this to keep accuracy
    for i in range(len(feature_diss)):
        if feature_diss[i] < 0:
            feature_diss[i] = -feature_diss[i]
    
    keep_features_train = np.array([train[:,i] for i in range(len(feature_diss)) if feature_diss[i] < np.mean(feature_diss)+(np.std(feature_diss)*0.5) and feature_diss[i] > np.mean(feature_diss)-(np.std(feature_diss)*0.5)])
    keep_features_test = np.array([test[:,i] for i in range(len(feature_diss)) if feature_diss[i] < np.mean(feature_diss)+(np.std(feature_diss)*0.5) and feature_diss[i] > np.mean(feature_diss)-(np.std(feature_diss)*0.5)])
    print('feature size reduced from {} to {}'.format(len(train[0]),len(keep_features_train.T[0])))

    return keep_features_train.T,keep_features_test.T 
