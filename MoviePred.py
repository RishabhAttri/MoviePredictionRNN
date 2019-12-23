# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 13:52:17 2019

@author: Rishabh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


#Import the Dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:,0].values
Y = dataset.iloc[:,1].values

#Clean the Dataset
def clean_text(text):
    text = re.sub(r"[-()~@<>+=|.?\"#;:,{}]", "", text)
    text = re.sub(r"[the,a,an]" , "", text)
    return text

clean_X = []
for sent in X:
    clean_X.append(clean_text(sent))

for i in range(len(clean_X)):
    clean_X[i] = clean_X[i].lower()

#Count the words
words2int = {}
for sent in clean_X:
    for word in sent.split():
        if words2int.get(word) is None:
            words2int[word] = 1
        else:
            words2int[word] += 1
print(len(words2int))

#Remove the words below threshold
out_string = '<OUT>'
threshold = 50
wordsAboveThres2int = {}
word_num = 0
for word, value in words2int.items():
    if value >= threshold:
        wordsAboveThres2int[word] = word_num
        word_num += 1
            

print(len(wordsAboveThres2int))

cl_clean_X = []

for sent in clean_X:
    clean_sent = []
    for word in sent.split():
        if word in wordsAboveThres2int:
            clean_sent.append(word)
        else:
            clean_sent.append(out_string)
    cl_clean_X.append(clean_sent)
    
#One Hot Y
pos2one = {'pos' : 1, 'neg' : 0}
binY = []
for y in Y:
    binY.append(pos2one[y])

#Creating the embedding dictionary
f = open('glove.6B.50d.txt', encoding='utf-8')
word_to_50dVec = {}
cnt = 0
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:],dtype='float')
    word_to_50dVec[word] = coefs
f.close()

for L in cl_clean_X:
    if L.__contains__('<OUT>'):
        L.remove('<OUT>')
cnt = 0
for L in cl_clean_X:
    if cnt < len(L):
        cnt = len(L)
print(cnt)
    
#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cl_clean_X, binY,test_size = 0.2)
#The embedding layer
dim = 50
def embedding_output(X):
    maxLen = 2200
    X = np.asarray(X)
    embedding_out = np.zeros((X.shape[0],maxLen,dim))
    for sentIndex in range(X.shape[0]):
        for wrdIndex in range(X[sentIndex].shape[0]):
            embedding_out[sentIndex][wrdIndex] = word_to_50dVec[X[sentIndex][wrdIndex]]
    return embedding_out

embedding_matrix_train = embedding_output(X_train)
embedding_matrix_test = embedding_output(X_test)

