# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import File_Reader
import File_Writer
from  Classifiers import create_mlp_rgr, mlp_regression,plot_var_bias
from PreProccessing import scaler,norm,domain_adaption
import matplotlib.pyplot as plt

#paths
train_path = r'C:\Users\Slades-PC\Documents\Assignments\Machine Learning\training.csv'
test_path = r'C:\Users\Slades-PC\Documents\Assignments\Machine Learning\testing.csv'
write_path = r'C:\Users\Slades-PC\Documents\Assignments\Machine Learning\Results'
confidence_path = r'C:\Users\Slades-PC\Documents\Assignments\Machine Learning\annotation_confidence.csv'
extra_train_path = r'C:\Users\Slades-PC\Documents\Assignments\Machine Learning\additional_training.csv'
print('extracting data')
#example of how to read training data
train_data, outputs = File_Reader.read_train_data(train_path)
#example of how to read test data 
test_data = File_Reader.read_test_data(test_path)
#anotator confidence 
output_confidence = File_Reader.read_confidence_data(confidence_path)
#regressor outputs
rgr_outputs = outputs
for i in range(len(outputs)):
    if outputs[i] == 0:
        rgr_outputs[i] = -output_confidence[i]
    else:
        rgr_outputs[i] = output_confidence[i]

"""
MLPR(Multilayered perceptron regressor V2) 
+annotater confidence
"""
"""
print('running mlp regressor+annotator confidence V2')
#train classifier
rgr = create_mlp_rgr(train_data,rgr_outputs)#,'sgd','adaptive',0.0001,200,(4000))#rgr = create_mlp_rgr(test[0],test[1],solver,l,lri,max_iter,hl,0.0001)
#classifying 
classes = mlp_regression(rgr, test_data,use_test_props = False)
#example printing to output csv
File_Writer.write_csv(write_path,'mlp rgr v2',classes)
"""

"""
Basic MLPR(Multilayered perceptron regressor)
+annotator confidence
+test proportions
"""
"""
print('running mlp regressor+annotator confidence V3')
#train classifier
rgr = create_mlp_rgr(train_data,rgr_outputs)#,'sgd','adaptive',0.0001,200,(4000))#rgr = create_mlp_rgr(test[0],test[1],solver,l,lri,max_iter,hl,0.0001)
#classifying 
classes = mlp_regression(rgr, test_data)
#example printing to output csv
File_Writer.write_csv(write_path,'mlp rgr v3',classes)
"""
"""
Basic MLPR(Multilayered perceptron regressor)
+annotator confidence
+test proportions
+additional training data
"""

#read extra train
extra_train_data, extra_outputs = File_Reader.read_train_data(extra_train_path)
for x in range(len(extra_outputs)):
        if extra_outputs[x] <= 0:
            extra_outputs[x] = -1
#concats all training data
all_data = np.concatenate((train_data,extra_train_data))
#concats all expected outputs for training data
all_outputs = np.concatenate((rgr_outputs,extra_outputs))

"""
print('running mlp regressor+annotator confidence V4')
#train classifier
rgr = create_mlp_rgr(all_data,all_outputs)#,'sgd','adaptive',0.0001,200,(4000))#rgr = create_mlp_rgr(test[0],test[1],solver,l,lri,max_iter,hl,0.0001)
#classifying 
classes = mlp_regression(rgr, test_data)
#example printing to output csv
File_Writer.write_csv(write_path,'mlp rgr v4',classes)
"""


"""
Basic MLPR(Multilayered perceptron regressor V5)
+annotator confidence
+test proportions
+additional training data
+scaling
"""
#scale data
all_data,test_data = scaler(all_data,test_data)

"""
print('running mlp regressor+annotator confidence V5')
#train classifier
rgr = create_mlp_rgr(all_data,all_outputs)#,'sgd','adaptive',0.0001,200,(4000))#rgr = create_mlp_rgr(test[0],test[1],solver,l,lri,max_iter,hl,0.0001)
#classifying 
classes = mlp_regression(rgr, test_data)
#example printing to output csv
File_Writer.write_csv(write_path,'mlp rgr v5',classes)
"""
"""
Basic MLPR(Multilayered perceptron regressor V6)
+annotator confidence
+test proportions
+additional training data
+scaling
+domain adaption
"""

#create domain adaption data
domain_all_data,domain_test_data = domain_adaption(all_data,all_outputs,test_data)

"""
print('running mlp regressor+annotator confidence V6')
#train classifier
rgr = create_mlp_rgr(domain_all_data,all_outputs)#,'sgd','adaptive',0.0001,200,(4000))#rgr = create_mlp_rgr(test[0],test[1],solver,l,lri,max_iter,hl,0.0001)
#classifying 
classes = mlp_regression(rgr, domain_test_data)
#example printing to output csv
File_Writer.write_csv(write_path,'mlp rgr v6',classes)
"""
"""
Basic MLPR(Multilayered perceptron regressor V7)
+annotator confidence
+test proportions
+additional training data
+scaling
+domain adaption
+hyper parameter tuning
"""

print('running mlp regressor+annotator confidence V7')
#train classifier
rgr = create_mlp_rgr(domain_all_data,all_outputs,)#,'sgd','adaptive',0.0001,200,(4000))#rgr = create_mlp_rgr(test[0],test[1],solver,l,lri,max_iter,hl,0.0001)
#classifying 
classes = mlp_regression(rgr, domain_test_data)
#example printing to output csv
File_Writer.write_csv(write_path,'mlp rgr v7',classes)


"""
Plot variance/bias
"""
print('calculating learn/train curve')
train_curve,learn_curve = plot_var_bias(domain_all_data,all_outputs)
p = plt.plot([x[1] for x in train_curve],[x[0] for x in train_curve], label = 'Training Curve')
p = plt.plot([x[1] for x in learn_curve],[x[0] for x in learn_curve], label = 'Validation Curve')
plt.xlabel('Images Used')
plt.ylabel('Error Rate')
plt.legend()  
plt.show()