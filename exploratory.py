#! py -3
"""
#CE706 Information Retrieval-Assignment 1
##Design of an Indexing for Web Search System
"""
import os
import sys
import re
import nltk
import csv
import pandas as pd
import numpy as np
import time

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from datetime import datetime
from pre_processing import x_matrix, y_matrix, tfidf_matrix, \
                            x_matrix_med, y_matrix_med, tfidf_matrix_med
# Set CPU as available physical device
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

pd.set_option('display.max_columns', 100)
wd=os.getcwd()
data_path=wd+"\\data"+"\\"

print("Econbiz file:")
file="econbiz" #,"pubmed"]

folds=list(range(10))

x,voc,_=x_matrix(file,folds,0)
d_t=x.shape[0]
voc=len(voc)
x[x!=0]=1
w_d=np.mean(np.sum(x,axis=1))
del x,voc

y=y_matrix(file,folds,0)
l=y.shape[1]
l_hat=np.mean(np.sum(y,axis=1))
l_tilde=np.mean(np.sum(y,axis=0))
del y

print("Number of Documents: %s" % d_t)
print("Size of Vocabulary %s" % voc)
print("Average Number of Words per Document %s" % w_d)
print("Number of Labels %s" % l)
print("Average Number of Labels per Document %s" % l_hat)
print("Average Number of Documents per Label %s" % l_tilde)

print("\n")
print("PubMed file:")
file="pubmed" #,"pubmed"]

x,voc,_,_=x_matrix_med(file,folds,0)
d_t=x.shape[0]
voc=len(voc)
x[x!=0]=1
w_d=np.mean(np.sum(x,axis=1))
del x

y,_=y_matrix_med(file,folds,0)
l=y.shape[1]
l_hat=np.mean(np.sum(y,axis=1))
l_tilde=np.mean(np.sum(y,axis=0))
del y

print("Number of Documents: %s" % d_t)
print("Size of Vocabulary %s" % voc)
print("Average Number of Words per Document %s" % w_d)
print("Number of Labels %s" % l)
print("Average Number of Labels per Document %s" % l_hat)
print("Average Number of Documents per Label %s" % l_tilde)
