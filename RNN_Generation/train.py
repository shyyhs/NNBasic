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


def randomTrainingExample():
    inputCate, inputLine = randomInputSample()
    return tensorCategory(inputCate), tensorInput(inputLine), \
        tensorTarget(inputLine)

def trainStep(category, input, target):
    target.unsqueeze_(-1)
    hidden = rnn.initHidden()

    # grad    
    rnn.zero_grad()
    lossSum = 0
    for i in range(input.size(0)):
        output, hidden = rnn(category,input[i],hidden)
        lossStep = criterion(output, target[i])
        lossSum += lossStep
    lossSum.backward()

    for p in rnn.parameters():
        p.data.add_(-lr,p.grad.data)

    return output, lossSum.item()/input.size()[0]
    

#Parameters
iterN = 100000
lr = 0.005

hiddenSize = 128
rnn = RNN(letterN, hiddenSize, letterN) 
criterion = nn.NLLLoss()

printN = iterN/10 
plotN = 500

def training():
    lossLst = []
    lossTotal = 0
    startT = time.time()
    for i in range(1,iterN+1):
        output,loss = trainStep(*randomTrainingExample())
        lossTotal += loss
        if (i%printN==0):
            print ('{} {} {}'.format(i, timeSince(startT), loss))
        if (i%plotN==0):
            lossLst.append(lossTotal/plotN)
            lossTotal=0
    return rnn, lossLst 
    
    

if (__name__!="__main__"):
    print("Training module")

if (__name__=="__main__"):
    print ("Training begins")
    _, lossLst = training()
