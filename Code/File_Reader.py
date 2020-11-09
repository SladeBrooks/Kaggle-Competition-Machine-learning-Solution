import pandas as pd
import numpy as np

"""
reads the file and returns... features,outputs
where features is a concatenated list of CNN features and GIST features
and outputs is the expected class of those features

if the training data has missing values then imputation is used to insert the mean for that value for that class
"""    
def read_train_data(file_path):
    #read the file
    df = pd.read_csv(file_path)
    #seperate features
    data = [ x[1:len(x)-1]for x in df.values.tolist()]
    #seperate expected outputs
    outputs = [x[len(x)-1] for x in df.values]
    
    #activates if the data contains incomplete 'nan' values
    if np.isin('nan',data):
        
        #seperate the positive and negative data
        pos_data = np.array([row[1:len(row)-1] for row in df.values.tolist() if row[len(row)-1]> 0.5])
        neg_data = np.array([row[1:len(row)-1] for row in df.values.tolist() if row[len(row)-1] < 0.5])
        #calculate the mean values per column per positive and negative data
        neg_means = np.nanmean(neg_data, axis = 0)
        pos_means = np.nanmean(pos_data, axis = 0)
        
        #extract the missing cor-ordinates for both positive and negative data
        pos_nans = np.argwhere(np.isnan(np.array(pos_data)))
        neg_nans = np.argwhere(np.isnan(np.array(neg_data)))
        
        #replace missing negative data with the mean for that value in negative data
        for nan in neg_nans:
            neg_data[nan[0]][nan[1]] = neg_means[nan[1]]
         #replace missing positive data with the mean for that value in positive data    
        for nan in pos_nans:
            pos_data[nan[0]][nan[1]] = pos_means[nan[1]]
        
        data =  np.concatenate((pos_data,neg_data))
        outputs = np.concatenate((np.ones(len(pos_data)),np.zeros(len(neg_data))))
        return data,outputs

        return np.array(data),np.array(outputs)
    #if the data is already complete, then the data is returned
    else:
        return np.array(data),np.array(outputs)
 
#Reads in the test data       
def read_test_data(file_path):
    df = pd.read_csv(file_path)
    data = [ x[1:len(x)]for x in df.values.tolist()]
    return np.array(data)   

#Reads in the annotator confidence data
def read_confidence_data(file_path):
    df = pd.read_csv(file_path)
    data = [ x[1] for x in df.values.tolist()]
    return np.array(data)
