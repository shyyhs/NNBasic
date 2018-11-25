from __future__ import unicode_literals, print_function, division

import os
import sys
import random
import time

import glob
import string
import unicodedata
from io import open

import torch

from globalVar import *

# assistant variable, viarable letters
all_letters = string.ascii_letters + ".,;'-"
n_letters = len(all_letters)+1 # Plus EOF
letterN = n_letters
letterLst = all_letters

# about category, number, type and dictionary
categoryN = 0
categoryType= []
categoryData = {}

# Tese four functions for readlines into [] and change it to ASCII,
# Last function getCate() be executed when call it.

def findFiles(path): return glob.glob(path)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD',s)
        if (unicodedata.category(c)!='Mn') and c in all_letters
    )

def readLines(filename):
    lines = open(filename,'r',encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

def getCate():
    global categoryN
    for filename in findFiles("./data/names/*.txt"):
        categoryTmp=os.path.basename(filename).split('.')[0]
        categoryType.append(categoryTmp)
        categoryData[categoryTmp]=readLines(filename)
    categoryN = len(categoryType)

# Data for training
def randomItem(lst):
    return lst[random.randint(0,len(lst)-1)]
def randomInputSample():
    randCate = randomItem(categoryType)
    randLine = randomItem(categoryData[randCate])
    return randCate, randLine

#One-hot
def tensorCategory(category):
    li = categoryType.index(category)
    tensor = torch.zeros(1,categoryN)
    tensor[0][li] = 1
    return tensor
    
def tensorInput(line):
    tensor = torch.zeros(len(line),1,letterN)
    for i in range(len(line)):
        c = line[i]
        tensor[i][0][letterLst.index(c)] = 1
    return tensor

def tensorTarget(line):
    target = [letterLst.find(c) for c in line[1:]]
    target.append(letterN-1)
    return torch.tensor(target, dtype = torch.int64)
         
    
        
if (__name__!="__main__"):
    #This function is used by other functions, not test
    print("Data module")
    getCate()


if (__name__=="__main__"):
    #Here's the test function
    getCate()
    print (categoryN)
    print (categoryType)
