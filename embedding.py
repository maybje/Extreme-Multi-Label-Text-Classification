#! py -3
"""
#CE807 Text Analytics - Final Assignment
##XMLC: Embedding Scores Function
"""
import os
import sys
import re
import nltk
import csv
import pandas as pd
import numpy as np
import time

#setting workinf directory
wd=os.getcwd()
glovepath=wd+"/glove/"  ##path to files

#Glove embedding scores function
def glove_embedding(word_index,n):
    """Glove embedding scores function:
        Inputs:
            word index
            and number of dimensions of the GloVe file
    """
    print("Embedding X matrix...")  #control message
    embeddings_index = {}   #empty dictionary to store scores
    file_glove='glove.6B.'+str(n)+'d.txt'   #filename
    #open file
    f = open(os.path.join(glovepath, file_glove), encoding="utf8")
    #loop to iterate thorugh file
    for line in f:
        values = line.split()   #split elements
        word = values[0]    #getting the word
        coefs = np.asarray(values[1:], dtype='float32') #getting coeficients
        embeddings_index[word] = coefs  #assigning to dict
    f.close()   #closing file

    #defining zero matrix
    embedding_matrix = np.zeros((len(word_index) + 1, n))
    #loop thorugh index of vocabulary from datset
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)   #getting words only
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector #assing score to word position
    return embedding_matrix
