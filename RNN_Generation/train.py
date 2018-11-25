from __future__ import unicode_literals, print_function, division

import os
import sys

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

def randomItem(lst):
    return lst[random.randint(0,len(lst)-1)]
def randomSample():
    randCate = randomItem(categoryType)
    randLine = randomItem(categoryData[randCate])
    return randCate, randLine


if (__name__!="__main__"):
    print(" Training module")

if (__name__=="__main__"):
    print ("Training begins")
    

