from __future__ import unicode_literals, print_function, division

import os
import sys

import glob
import string
import unicodedata
from io import open

from globalVar import *

# assistant variable, viarable letters
all_letters = string.ascii_letters + ".,;'-"
n_letters = len(all_letters)+1 # Plus EOF

# about category, number, type and dictionary
categoryN = 0
categoryType= []
categoryData = {}

# Tese Three functions for readlines into [] and change it to ASCII
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
        
if (__name__!="__main__"):
    #This function is used by other functions, not test
    print("Data module")


if (__name__=="__main__"):
    #Here's the test function
    getCate()
    print (categoryN)
    print (categoryType)
