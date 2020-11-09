# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os


#writes the predictions to a specified file in the correct format for the kaggle competition
#path = path to be written
#filename = name of predictions file
#classes = predictions
def write_csv(path,filename,classes):
    ids = [i+1 for i in range(len(classes))]
    data = np.array([(ids[x],classes[x]) for x in range(len(ids))]).astype('int32')
    pd.DataFrame(data).to_csv(os.path.join(path,filename+'.csv'), header=['ID','prediction'], index=None)
 
#Writes an accuracy score and test name to a specified file path
#NOTE- please change 'write_file' parth to specified results file
def write_cross_val_result(accuracy,test_name):
    write_file = r'C:\Users\Slades-PC\Documents\Assignments\Machine Learning\Results\results.txt'
    with open(write_file, "a") as output:
        output.write('\n'+test_name+': '+str(accuracy))
        