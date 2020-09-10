#! py -3
"""
#CE807 Text Analytics - Final Assignment
##XMLC: Files Reader
"""
import os
import pandas as pd

#Defining the working directory
wd=os.getcwd()
data_path=wd+"\\data"+"\\"  #Path where data is stored

#create folder if does not exist yet
if not os.path.exists(data_path):
    os.makedirs(data_path)

#list of datasets to iterate through
files=["econbiz.csv","pubmed.csv"]

#loop to generate separate labels,features files
i=0     #iterator
for f in files:
    filename=data_path+f        #filename
    data=pd.read_csv(filename)  #read file in pandas DF
    print("file %s open" % f)   #Control message

    x=data.loc[:,["title","fold"]]  #Keeping only relevant columns
    #defining features file name
    name_x = data_path+os.path.splitext(files[i])[0]+"_features.csv"
    x.to_csv(name_x, index=False)   #generate CVS file
    print("File saved in: %s" % name_x) #Control message
    del x   #delete x DF to optimize memory

    y=data.loc[:,["labels","fold"]]  #Keeping only relevant columns
    #defining labels file name
    name_y=data_path+os.path.splitext(files[i])[0]+"_labels.csv"
    y.to_csv(name_y, index=False)   #generate CVS file
    del y        #delete y DF to optimize memory
    print("File saved in: %s" % name_y) #Control message
    del data     #delete y DF to optimize memory
    i+=1    #increasing iterator variable
