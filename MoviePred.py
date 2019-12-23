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
threshold = 25
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
    



