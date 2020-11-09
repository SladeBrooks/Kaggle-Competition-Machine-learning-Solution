# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
import File_Writer

#Creats a multi layered percetron, trains it using train as train_x and outputs as train_y
#The default parameters where secided using hyper parameter tuning 
def create_mlp_rgr(train,outputs,solv= 'adam',actv = 'relu',learn_rate = 'invscaling',learn_init = 0.0001,max_it = 250,shuf  = False ,rand_state = 8 ,hidden_layers = (3000)):
    rgr = MLPRegressor(solver = solv,activation = actv,learning_rate = learn_rate,learning_rate_init = learn_init,max_iter = max_it,shuffle = shuf,random_state = rand_state,hidden_layer_sizes = hidden_layers)
    rgr.fit(train, outputs)
    return rgr
#uses a multi layered perceptron to classify data
#converts proportions of classifications to that of the test data
#turns regression classifications into binary classifications
def mlp_regression(rgr,data,use_test_props = True):
    #performs classifications
    rgr_classes = rgr.predict(data)
    
    #Next 2 lines are used to ensure proportions are correct
    test_len = len(rgr_classes)
    zero_test_proportion = 0.6767#this data is supplied in the competition resources, no need to read 1 number with file reader
    
    #this while loop ensures the correct proportions of classifications are returned
    while(len([x for x in rgr_classes if x <= 0]) < (test_len * zero_test_proportion)):# and use_test_props == True:
        rgr_classes = rgr_classes-0.001
        
    #this for loop converts the regession predictions into the binary classifications needed
    for x in range(len(rgr_classes)):
        if rgr_classes[x] <= 0:
            rgr_classes[x] = 0
        else:
            rgr_classes[x] = 1
    rgr_classes = np.array(rgr_classes)
    #classes = np.rint(rgr_classes)
    
    return rgr_classes

#returns accuracy out of 1 
#works out the amount of predictions that match the actual
def calc_accuracy(predictions,actual):
    correct = 0
    total = len(predictions)
    
    for i in range(len(predictions)):
        if predictions[i] == actual[i]:
            correct += 1
    if correct > 0:
        print('{}/{}'.format(correct,total))
        return correct/total
    else:
        return 0
 
"""
Performs cross validation over train, the amount of k folds in each test is a specified parameter
"""
def cross_val(k_fold,train,outputs,test_name,solver = 'adam',activation = 'relu',learning_rate = 'invscaling',learning_rate_init = 0.0001,max_iter = 250,shuffle = False,random_state = 5,hidden_layers = (3000)):
    #shuffles the data
    shuffled = np.array([(train[i],outputs[i]) for i in range(len(train))])
    np.random.shuffle(shuffled)
    
    #splits the shuffled data back into training and expected outputs
    train = [x[0] for x in shuffled]
    outputs = [x[1] for x in shuffled]
    
    #total amount of samples
    total = len(train)
    
    """
    splits the data into k_fold samples then creates a test case for each of those splits where the k_fold
    sample is used for testing purposes and the rest of the data is used for training
    """
    splits = []
    split_ratio = int(total/k_fold)
    previous = 0
    for k in range(k_fold):
        train_split =train[:previous]+train[previous+split_ratio:]
        train_outputs =outputs[:previous]+outputs[previous+split_ratio:]
        test_split =train[previous:previous+split_ratio]
        test_output =outputs[previous:previous+split_ratio]
        for i in range(len(test_output)):
            if test_output[i] < 0:
                test_output[i] = 0
            else:
                test_output[i] = 1
                
        splits.append((train_split,train_outputs,test_split,test_output))
        previous = previous + split_ratio
    accuracies = []
    
    """
    for each of the previously defined test cases, a regressor is trained and used and the accuracy is
    recorded
    """
    for test in splits:
        #train classifier
        rgr = create_mlp_rgr(test[0],test[1],solver,activation,learning_rate,learning_rate_init,max_iter,shuffle,random_state,hidden_layers)
        #classifying 
        classes = mlp_regression(rgr, test[2]).astype('int32')
        #compute accuracy
        accuracies.append(calc_accuracy(classes,test[3]))
        
    #write results, writes accuracy and standard deviation
    File_Writer.write_cross_val_result(np.mean(np.array(accuracies)),test_name+' std:{} '.format(np.std(accuracies)))
 
"""
returns 2 lines to be plotted, the training curve and the validation curve
This method shuffled the training data, slices a reserve of test data out.
20 tests are then performed when the amount of data used to train the classifier is increased by 2 each time
this is because the most noticeable differences where in the small amounts of image datasets used.
"""
def plot_var_bias(train,outputs):
    #shuffle the training data and expected outputs
    shuffled = np.array([(train[i],outputs[i]) for i in range(len(train))])
    np.random.shuffle(shuffled)
    train = [x[0] for x in shuffled]
    outputs = [x[1] for x in shuffled]
    
    
    #partition the data
    size = len(train)
    x_train = train[int(size/10):]
    y_out = outputs[int(size/10):]
    #print(x_train)
    #print(y_out)
    reserved_test = train[:int(size/10)]#reserved for validation curve
    reserved_outputs = outputs[:int(size/10)]#reserved for validation curve
    
    #convert the reserved outputs into classification classes
    for i in range(len(reserved_outputs)):
            if reserved_outputs[i] < 0:
                reserved_outputs[i] = 0
            else:
                reserved_outputs[i] = 1
    
    #store the training curve
    train_curve = []
    #store the validation curve
    learning_curve = []
    size = len(x_train)
    partition_size = 2
    """
    for each test case train a regressor then used it to classify the seen training data 
    and the unseen test data. This is done for an increasing size of training and test data,
    with the accuracies and sizes returned 
    """
    for i in range(20):
        train_i = x_train[:int((i+1)*partition_size)]
        out_i = y_out[:int((i+1)*partition_size)]
        rgr = create_mlp_rgr(train_i,out_i,solv= 'adam',actv = 'relu',learn_rate = 'invscaling',learn_init = 0.0001,max_it = 250,shuf  = False ,rand_state = 5 ,hidden_layers = (3000))
        
        train_c_classes = mlp_regression(rgr, np.array(train_i).astype('int32'))
        learn_c_classes = mlp_regression(rgr, np.array(reserved_test[:int((i+1)*partition_size)]).astype('int32'))
        for i in range(len(out_i)):
            if out_i[i] < 0:
                out_i[i] = 0
            else:
                out_i[i] = 1
                
        train_c_acc = calc_accuracy(train_c_classes,out_i)
        learn_c_acc = calc_accuracy(learn_c_classes,reserved_outputs)
        
        train_curve.append((1-train_c_acc,len(train_i)))
        learning_curve.append((1-learn_c_acc,len(train_i)))
    
    return train_curve,learning_curve