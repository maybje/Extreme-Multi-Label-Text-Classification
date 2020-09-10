#! py -3
"""
#CE807 Text Analytics - Final Assignment
##XMLC: Pre-Processing
"""
import os
import sys
import re
import nltk
import csv
import pandas as pd
import numpy as np
import time
import tensorflow as tf

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
from tensorflow import keras
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import MultiLabelBinarizer
# Set CPU as available physical device
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#Defining working directory
wd=os.getcwd()
data_path=wd+"\\data"+"\\"      #setting data path
results_path=wd+"\\results"+"\\"    #setting results results_path

#Create results folder if not exists
if not os.path.exists(results_path):
    os.makedirs(results_path)

#function to define the X matrix for Econbiz dataset
def x_matrix(file, folds,t):
    """X matrix generator:
        Inputs:
            filename
            folds to retrieve
            and partition to work with, either 1,2,4,8 and all docs
    """
    print("Getting X matrix...")    #control message
    features=data_path+file+"_features.csv" #filename
    x=pd.read_csv(features) #reading file

    #partitions to work with
    if t==0:
        pass
    else:
        x=x.loc[x.fold.isin(folds),:].append(x.loc[x.fold==10,:].iloc[0:x.loc[x.fold.isin(folds),:].shape[0]*(t-1),:],ignore_index=True)

    #pre-processing
    ##keeping alphabetical characters and spaces
    x["title"]=x["title"].str.replace(r'[^\sa-zA-Z]', "")
    ##joining hyphen spearated words
    x["title"]=x["title"].str.replace(r'[-]', "")
    #keeping words with counts strictly greater than 2
    x["title"]=x["title"].str.replace(r'\b\w{1,2}\b', '')
    #normalizing text
    x["title"]=x["title"].str.lower()
    max=x["title"].str.split().str.len().max()  #max number of words

    #defining tokenizer function
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x["title"]) #fitting tokenizer
    x = tokenizer.texts_to_sequences(x["title"])    #converting to sequences
    word_index = tokenizer.word_index   #getting word index

    #Zero padding sequences to the right
    x = pad_sequences(x, maxlen=max)
    return x, word_index, max

#function to define the Y matrix
def y_matrix(file,folds,t):
    """Y matrix generator:
        Inputs:
            filename
            folds to retrieve
            and partition to work with, either 1,2,4,8 and all docs
    """
    print("Getting Y matrix...")    #control message
    labels=data_path+file+"_labels.csv" #filename
    y=pd.read_csv(labels)   #reading file

    #partitions to work with
    if t==0:
        pass
    else:
        y=y.loc[y.fold.isin(folds),:].append(y.loc[y.fold==10,:].iloc[0:y.loc[y.fold.isin(folds),:].shape[0]*(t-1),:],ignore_index=True)
    mlb = MultiLabelBinarizer(sparse_output=True)   #defining binarizer
    #pre-processing depending on the dataset, as econbiz labels are numerical
    #and pubmed aphabetical
    if file=="econbiz":
        y.labels=y.labels.str.replace(r'[\t]', " ") #replacing tabs with spaces
        y.labels=y.labels.str.replace(r'[-]', "")   #joining component
        y['labels'] = y.labels.apply(word_tokenize) #tokenizing labels
        y=mlb.fit_transform(y['labels'] )   #fitting binarizer
        y_test=0
    else:
        y.labels=y.labels.str.replace(r'[^\sa-zA-Z]', "")   #alphabetical chars
        y.labels=y.labels.str.replace(r'[ ]', "")   #stripping whitespaces
        y.labels=y.labels.str.replace(r'[\t]', " ") #replacing tabs with \s
        y.labels=y.labels.str.replace(r'[-]', "")   #joining components
        y.labels=y.labels.str.lower()   #normalizing
        y['labels'] = y.labels.apply(word_tokenize) #tokeni labels
        mlb.fit(y['labels'])    #fitting binarizer
        y_test=mlb.transform(y.loc[y.fold==6,"labels"] ) #binarizing
        y=mlb.transform(y.loc[y.fold!=6,"labels"] )#binarizing
    return y,y_test

#function to define the tf.idf X matrix for Econbiz dataset
def tfidf_matrix(file, folds,t,n):
    """TF-IDF X matrix generator:
        Inputs:
            filename
            folds to retrieve
            partition to work with, either 1,2,4,8 and all docs
            and indciator of word minimum length to keep
    """
    print("Getting X matrix...")    #control message
    features=data_path+file+"_features.csv" #filename
    x=pd.read_csv(features) #reading file

    #partitions to work with
    if t==0:
        #x=x.loc[x.fold.isin(folds),:].append(x.loc[x.fold==10,:],ignore_index=True)
        pass
    else:
        x=x.loc[x.fold.isin(folds),:].append(x.loc[x.fold==10,:].iloc[0:x.loc[x.fold.isin(folds),:].shape[0]*(t-1),:],ignore_index=True)
    #pre-processing
    x["title"]=x["title"].str.replace(r'[^\sa-zA-Z]', "")   #alphabetical chars
    x["title"]=x["title"].str.replace(r'[-]', "")   #joining composed words
    #indciator of word minimum length to keep
    if n==1:
        x["title"]=x["title"].str.replace(r'\b\w{1,2}\b', '')   #length<=2
    else:
        x["title"]=x["title"].str.replace(r'\b\w{1,3}\b', '')   #length<=3
    x["title"]=x["title"].str.lower()   #normalizing
    max=x["title"].str.split().str.len().max()  #getting max title length
    stemmer = SnowballStemmer("english")    #defining stemmer
    #stemming function row by row
    def stemming(sentence):
        stemSentence = ""
        for word in sentence.split():
            stem = stemmer.stem(word)   #stem tokens
            stemSentence += stem    #joining stems into the original sentence
            stemSentence += " " #adding space between stems
        stemSentence = stemSentence.strip() #eending spaces removed
        return stemSentence
    x['title'] = x['title'].apply(stemming) #apply stemmer to titles

    #defining tfidf vectorizer
    vectorizer = TfidfVectorizer(use_idf=True, lowercase=False,ngram_range=(1,n),max_features=30000)
    x = vectorizer.fit_transform(x["title"]) #fitting vectorizer
    print("Number of tokens: %s" % x.shape[1])  #control message
    return x

#function to define the X matrix for Medpub dataset
def x_matrix_med(file, folds,t):
    """X matrix generator:
        Inputs:
            filename
            folds to retrieve
            and partition to work with, either 1,2,4,8 and all docs
    """
    print("Getting X matrix...")    #control message
    features=data_path+file+"_features.csv" #filename
    x=pd.read_csv(features) #reading file

    #partitions to work with
    if t==0:
        pass
    else:
        x=x.loc[x.fold.isin(folds),:].append(x.loc[x.fold==10,:].iloc[0:x.loc[x.fold.isin(folds),:].shape[0]*(t-1),:],ignore_index=True)

    #pre-processing
    x["title"]=x["title"].str.replace(r'[^\sa-zA-Z]', "")   #alphabetical chars
    x["title"]=x["title"].str.replace(r'[-]', "")   #joining composed words
    x["title"]=x["title"].str.replace(r'\b\w{1,3}\b', '') #deleting length<4 words
    x["title"]=x["title"].str.lower()   #normalizing
    max=x["title"].str.split().str.len().max()  #getting maximum title length

    tokenizer = Tokenizer() #defining tokenizer
    tokenizer.fit_on_texts(x["title"])  #fitting tokenizer
    #converting words to sequences
    x_test= tokenizer.texts_to_sequences(x.loc[x.fold==6,"title"])
    x = tokenizer.texts_to_sequences(x.loc[x.fold!=6,"title"])
    word_index = tokenizer.word_index #getting word index

     #padding sequences
    x_test= pad_sequences(x_test, maxlen=max)
    x= pad_sequences(x, maxlen=max)
    return x, word_index, max,x_test

#function to define the tf.idf X matrix for PubMed dataset
def tfidf_matrix_med(file, folds,t,n):
    """TF-IDF X matrix generator:
        Inputs:
            filename
            folds to retrieve
            partition to work with, either 1,2,4,8 and all docs
            and indciator of word minimum length to keep
    """
    print("Getting X matrix...")     #control message
    features=data_path+file+"_features.csv" #filename
    x=pd.read_csv(features) #reading file

    #partitions to work with
    if t==0:
        pass
    else:
        x=x.loc[x.fold.isin(folds),:].append(x.loc[x.fold==10,:].iloc[0:x.loc[x.fold.isin(folds),:].shape[0]*(t-1),:],ignore_index=True)
    #pre-processing
    x["title"]=x["title"].str.replace(r'[^\sa-zA-Z]', "")   #alphabetical chars
    x["title"]=x["title"].str.replace(r'[-]', "")   #joining composed words
    #indciator of word minimum length to keep
    if n==1:
        x["title"]=x["title"].str.replace(r'\b\w{1,2}\b', '') #length<=2
    else:
        x["title"]=x["title"].str.replace(r'\b\w{1,3}\b', '')   #length<=3
    x["title"]=x["title"].str.lower()   #normalizing
    stemmer = SnowballStemmer("english")    #defining SnowballStemmer
    #stemming function row by row
    def stemming(sentence):
        stemSentence = ""
        for word in sentence.split():
            stem = stemmer.stem(word)   #stem tokens
            stemSentence += stem    #joining stems into the original sentence
            stemSentence += " " #adding space between stems
        stemSentence = stemSentence.strip() #eending spaces removed
        return stemSentence
    x['title'] = x['title'].apply(stemming) #apply stemmer to titles

    #defining tf.idf vectorizer
    vectorizer = TfidfVectorizer(use_idf=True,ngram_range=(1,n), lowercase=False,max_features=30000)
    vectorizer.fit(x["title"])  #fitting vectorizer
    x_test = vectorizer.transform(x.loc[x.fold==6,"title"]) #vectorizing
    x = vectorizer.transform(x.loc[x.fold!=6,"title"])  #vectorizing
    print("Number of tokens: %s" % x.shape[1])  #control message
    return x, x_test
