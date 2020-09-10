#! py -3
"""
#CE807 Text Analytics - Final Assignment
##XMLC: CNN
"""
import os
import sys
import re
import nltk
import csv
import pandas as pd
import numpy as np
import time
import gc
import tensorflow as tf

from sklearn.model_selection import KFold, train_test_split
from datetime import datetime
from tensorflow import keras
import keras
from keras import backend as K
from keras.layers import Embedding, Input, Dense, \
    Flatten, Dropout, LSTM, Bidirectional, \
    MaxPooling1D,Conv1D
from keras import models
from keras.callbacks import EarlyStopping
from pre_processing import x_matrix, y_matrix, tfidf_matrix, \
                            x_matrix_med, tfidf_matrix_med
from keras.metrics import top_k_categorical_accuracy
from embedding import glove_embedding
from bootstrapping import boostrap_mean
from sklearn.metrics import f1_score

"""
Input passed when calling script
"""
if __name__ == "__main__":
    try:
        file = sys.argv[1]  #second input
        print("File: %s" % file) #control message
    except:
        file="econbiz"  #default file
        print("No file passed, default: %s" % file)   #control message
    try:
        t = int(sys.argv[2])  #second input
        print("Factor: %s" % t) #control message
    except:
        t=1 #default partition
        print("No factor passed, default: %s" % t)   #control message

#setting workinf directory
wd=os.getcwd()
data_path=wd+"\\data"+"\\"  #data storing path

folds=list(range(10))   #list of folds

#getting X and Y matrices
if file=="econbiz":
    x,word_index,max=x_matrix(file,folds,t) #X matrix
    y_label,_=y_matrix(file,folds,t)    #Y matrix
    nol=y_label.shape[1]    #number of labels

    #train/test set split 90-10
    train_s=0.9
    x_t, x_v, y_t, y_v=train_test_split(x,y_label, train_size=train_s, random_state=21)
    del x, y_label   #deleting no longer needed variables to optimize memory
    batch=256   #batch size
else:
    x_t,word_index,max,x_v=x_matrix_med(file,folds,t)   #X matrix
    y_t,y_v=y_matrix(file,folds,t) #Y matrix
    nol=y_t.shape[1]    #number of labels
    batch=1024  #batch size

n=300 #number of embedding dimsensions
embedding_matrix=glove_embedding(word_index,n) #getting GloVe codes

#Defining top k labels accuracy function
def my_acc(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

#Defining model
def create_model():
    model = models.Sequential() #defining sequential model
    #Pre-Trained Embedding layer
    model.add(Embedding(len(word_index) + 1,
                                n,
                                weights=[embedding_matrix],
                                input_length=max,
                                trainable=False))
    model.add(Conv1D(400, 3, activation='relu'))    #1D convolutional layer
    model.add(MaxPooling1D(pool_size=3))    #Maxpooling layer
    model.add(Flatten())    #flattening output
    model.add(Dense(500, activation='relu'))    #hidden fully connected layer
    model.add(Dropout(0.5)) #dropout layer
    model.add(Dense(nol, activation='softmax'))  #output layer softmax afn
    model.summary() #print model summary

    #compiling model with Adam optimizer and binary crossentropy loss fn
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[my_acc])
    return model

#defining early stopping callback
callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5)
#K-fold CV
k_fold = KFold(n_splits=10,random_state=21)  #defining folds
i=1 #iterator variables
#k-fold loop
for train_indices, test_indices in k_fold.split(x_t):
    print("Fold %s" % i)    #control message
    model=create_model()    #calling model fn
    #fitting model on train set
    history = model.fit(x_t[train_indices], y_t[train_indices],
                    epochs=30,
                    batch_size=batch,
                    validation_data=(x_t[test_indices],y_t[test_indices]),
                    callbacks=[callback])
    y_predict=model.predict(x_t[test_indices])  #predicting on val set
    #deleting model and session from memory
    del model
    gc.collect()
    K.clear_session()
    tf.compat.v1.reset_default_graph()

    theta= 0.025 #defining threshold
    f1_list=[]  #list to store results
    #estimating f1 score on dataset halfs to optimize memory
    for i in range(2):
        if i==1:
            #predicting probabilities for the first half of documents
            y_hat = model.predict(x_t[test_indices][:round(x_t[test_indices].shape[0]/2),:])
            y_hat[y_hat>theta]=1    #applying threshold to predict labels
            y_hat[y_hat!=1]=0   #setting remaining pr to zero
            #estimating sample averaged f1 score
            f1_1=f1_score(y_t[test_indices][:round(x_t[test_indices].shape[0]/2),:],y_hat,average="samples")
            del y_hat   #deleting prediction matrix to optimize memory
        else:
            #predicting probabilities for the second half of documents
            y_hat = model.predict(x_t[test_indices][round(x_t[test_indices].shape[0]/2):,])
            y_hat[y_hat>theta]=1    #applying threshold to predict labels
            y_hat[y_hat!=1]=0   #setting remaining pr to zero
            #estimating sample averaged f1 score
            f1_2=f1_score(y_t[test_indices][round(x_t[test_indices].shape[0]/2):,:],y_hat,average="samples")
            del y_hat   #deleting prediction matrix to optimize memory
    f1=(f1_1+f1_2)/2    #averaging halfs average f1 scores
    f1_list.append(f1)
    i+=1
    del y_predict, f1

f1_list=sum(f1_list)/len(f1_list)   #10-fold mean f1 score
print("Average 10-fold F1 score at theta=0.025: %0.4f" % f1_list)   #message
del f1_list #deleting object

model=create_model()    #calling model function
#fitting model
history = model.fit(x_t, y_t,
                epochs=30,
                batch_size=batch,
                validation_data=(x_v,y_v), callbacks=[callback])
del x_t, y_t    #deletting objects to optimize memory

thetas=np.linspace(0.015,0.035,3) #threshold to be evaluated
f1_list=[]  #list to store results

##loop to iterate through thresholds and compute f1 score in separate halfs
#of the documents in order to optimize memory
for theta in thetas:
    for i in range(2):
        if i==1:
            #predicting probabilities for the first half of documents
            y_hat = model.predict(x_v[:round(x_v.shape[0]/2),:])
            y_hat[y_hat>theta]=1     #applying threshold to predict labels
            y_hat[y_hat!=1]=0    #setting remaining pr to zero
            #estimating sample averaged f1 score
            f1_1=f1_score(y_v[:round(x_v.shape[0]/2),:],y_hat,average="samples")
            del y_hat   #deleting prediction matrix to optimize memory
        else:
            #predicting probabilities for the second half of documents
            y_hat = model.predict(x_v[round(x_v.shape[0]/2):,])
            y_hat[y_hat>theta]=1    #applying threshold to predict labels
            y_hat[y_hat!=1]=0   #setting remaining pr to zero
            #estimating sample averaged f1 score
            f1_2=f1_score(y_v[round(x_v.shape[0]/2):,:],y_hat,average="samples")
            del y_hat   #deleting prediction matrix to optimize memory
    f1=(f1_1+f1_2)/2     #averaging halfs average f1 scores
    f1_list.append(f1)   #appending to results list
    print("F1 at %0.3f: %0.4f" % (theta,f1))    #control message

del x_v,y_v,embedding_matrix     #deleting objects to optimize memory

#Results can be saved if wanted
#results_path=wd+"\\results"+"\\"
#results=results_path+"\\cnn_t"+str(t)+"_"+file+".txt"
#res = open(results,'w')
#res.write(str(f1_list))
#res.close()
#print("F1 score saved in: %s" % results)
