from __future__ import unicode_literals, print_function, division

import os
import sys

import glob
import string
import unicodedata
from io import open

from globalVar import *
from data import *

import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize,
        categorySize=categoryN):

        super(RNN,self).__init__()
        self.hiddenSize = hiddenSize

        self.i2h = nn.Linear(categorySize+inputSize+hiddenSize, hiddenSize)
        self.i2o = nn.Linear(categorySize+inputSize+hiddenSize, outputSize)
        self.o2o = nn.Linear(outputSize+hiddenSize, outputSize)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        inputCombined = torch.cat((category,input,hidden),dim=1)
        hidden = self.i2h(inputCombined)
        output = self.i2o(inputCombined)
        output = torch.cat((output,hidden),dim=1)
        output = self.o2o(output)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1,self.hiddenSize)

if (__name__!="__main__"):
    #This function is used by other functions, not test
    print("model module")


if (__name__=="__main__"):
    r=RNN(n_letters, 128, n_letters)
    
