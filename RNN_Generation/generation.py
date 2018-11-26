from __future__ import unicode_literals, print_function, division

import os
import sys
import time
import math

import glob
import string
import unicodedata
from io import open
import random

import torch
import torch.nn as nn

from globalVar import *
from data import *
from model import *
from train import *

maxLen = 20
def sample(category=0, startLetter=0):
    if (category==0): category=randomItem(categoryType)
    if (startLetter==0): startLetter=randomItem(string.ascii_letters)
    outputString = startLetter
    outputCategory = category
    with torch.no_grad():
        category= tensorCategory(category)
        input = tensorInput(startLetter)
        hidden = rnn.initHidden()
        
        for i in range(maxLen):
            output,hidden = rnn(category,input[0],hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if (topi == letterN-1): break
            else:
                letter = letterLst[topi]
                outputString+=letter
            input = tensorInput(letter)
    return outputCategory,outputString
        
def samples(n):
    for i in range(n):
        category, name = sample()
        print ("Nation: {}, Name: {}\n".format(category,name))
        
        
        
if (__name__!="__main__"):
    print ("Generation model")
    


if (__name__ == "__main__"):
    print ("Generation begins")
    training()
    samples(20)
