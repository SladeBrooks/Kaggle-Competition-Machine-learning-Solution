# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import File_Reader
import File_Writer
from Classifiers import cross_val
from PreProccessing import scaler,norm,domain_adaption

#paths for the needed data, please change if run on a different system
train_path = r'C:\Users\Slades-PC\Documents\Assignments\Machine Learning\training.csv'
test_path = r'C:\Users\Slades-PC\Documents\Assignments\Machine Learning\testing.csv'
write_path = r'C:\Users\Slades-PC\Documents\Assignments\Machine Learning\Results'
confidence_path = r'C:\Users\Slades-PC\Documents\Assignments\Machine Learning\annotation_confidence.csv'
extra_train_path = r'C:\Users\Slades-PC\Documents\Assignments\Machine Learning\additional_training.csv'




#reads in the initial training data
train_data, outputs = File_Reader.read_train_data(train_path)
#read extra training data
extra_train_data, extra_outputs = File_Reader.read_train_data(extra_train_path)
#convert the training data to a format better suited for regression
for x in range(len(extra_outputs)):
        if extra_outputs[x] <= 0:
            extra_outputs[x] = -1

#reads in the test data
test_data = File_Reader.read_test_data(test_path)
#anotator confidence 
output_confidence = File_Reader.read_confidence_data(confidence_path)
#modifies initial training data outputs using the annotator confidence
rgr_outputs = outputs
for i in range(len(outputs)):
    if outputs[i] == 0:
        rgr_outputs[i] = -output_confidence[i]
    else:
        rgr_outputs[i] = output_confidence[i]
#concats all training data
all_data = np.concatenate((train_data,extra_train_data))
#concats all expected outputs for training data
all_outputs = np.concatenate((rgr_outputs,extra_outputs))
#scale data
all_data,test_data = scaler(all_data,test_data)
#perform domain adaption
domain_all_data,domain_test_data = domain_adaption(all_data,all_outputs,test_data)


print('Hyper parameter tests- Solver')
cross_val(5,domain_all_data,all_outputs,'mlp rgr, solver = lbfgs: ',solver = 'lbfgs')   
cross_val(5,domain_all_data,all_outputs,'mlp rgr, solver = sgd: ',solver = 'sgd')   
cross_val(5,domain_all_data,all_outputs,'mlp rgr , solver = adam: ',solver = 'adam')   

print('Hyper parameter tests- activation')
cross_val(5,domain_all_data,all_outputs,'mlp rgr, activation = identity: ',activation = 'identity')
cross_val(5,domain_all_data,all_outputs,'mlp rgr, activation = logistic: ',activation = 'logistic')
cross_val(5,domain_all_data,all_outputs,'mlp rgr, activation = tanh: ',activation = 'tanh')
cross_val(5,domain_all_data,all_outputs,'mlp rgr, activation = relu: ',activation = 'relu')

print('Hyper parameter tests- Learning Rate')
cross_val(5,domain_all_data,all_outputs,'mlp rgr, learning rate = : constant',learning_rate = 'constant')
cross_val(5,domain_all_data,all_outputs,'mlp rgr, learning rate = : invscaling',learning_rate = 'invscaling')
cross_val(5,domain_all_data,all_outputs,'mlp rgr, learning rate = : adaptive',learning_rate = 'adaptive')

print('Hyper parameter tests- Initial Learning Rate')
cross_val(5,domain_all_data,all_outputs,'mlp rgr, learning rate init = : 00001',learning_rate_init = 0.00001)
cross_val(5,domain_all_data,all_outputs,'mlp rgr, learning rate init = : 0001',learning_rate_init = 0.0001)
cross_val(5,domain_all_data,all_outputs,'mlp rgr, learning rate init = : 001',learning_rate_init = 0.001)
cross_val(5,domain_all_data,all_outputs,'mlp rgr, learning rate init = : 01',learning_rate_init = 0.01)

print('Hyper parameter tests- Max Iterations')
cross_val(5,domain_all_data,all_outputs,'mlp rgr, max_iter = 50:',max_iter = 50)
cross_val(5,domain_all_data,all_outputs,'mlp rgr, max_iter = 100:',max_iter = 100)
cross_val(5,domain_all_data,all_outputs,'mlp rgr, max_iter = 150:',max_iter = 150)
cross_val(5,domain_all_data,all_outputs,'mlp rgr, max_iter = 200:',max_iter = 200)
cross_val(5,domain_all_data,all_outputs,'mlp rgr, max_iter = 250:',max_iter = 250)
cross_val(5,domain_all_data,all_outputs,'mlp rgr, max_iter = 300:',max_iter = 300)

print('Hyper parameter tests- Data Shuffling')
cross_val(5,domain_all_data,all_outputs,'mlp rgr, shuffle = true:',shuffle = True)
cross_val(5,domain_all_data,all_outputs,'mlp rgr, shuffle = false:',shuffle = False)

print('Hyper parameter tests- Hidden Layer')
cross_val(5,domain_all_data,all_outputs,'mlp rgr, hidden_layer = 100:',hidden_layers = (100))
cross_val(5,domain_all_data,all_outputs,'mlp rgr, hidden_layer = 500:',hidden_layers = (500))
cross_val(5,domain_all_data,all_outputs,'mlp rgr, hidden_layer = 800:',hidden_layers = (800))
cross_val(5,domain_all_data,all_outputs,'mlp rgr, hidden_layer = 1000:',hidden_layers = (1000))
cross_val(5,domain_all_data,all_outputs,'mlp rgr, hidden_layer = 1500:',hidden_layers = (1500))
cross_val(5,domain_all_data,all_outputs,'mlp rgr, hidden_layer = 2000:',hidden_layers = (2000))
cross_val(5,domain_all_data,all_outputs,'mlp rgr, hidden_layer = 3000:',hidden_layers = (3000))
cross_val(5,domain_all_data,all_outputs,'mlp rgr, hidden_layer = 100,100,100,100:',hidden_layers = (100,100,100,100))