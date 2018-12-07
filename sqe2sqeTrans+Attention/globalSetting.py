from __future__ import unicode_literals, print_function, division

import sys
import os
import glob
import time
import math
import random

from io import open
import unicodedata
import re

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np

#import matplotlib.pyplot as plt
#plt.switch_backend('agg')
#import matplotlib.ticker as ticker

SOSToken = 0
EOSToken = 1
MAXLEN = 10
engPrefix = ("i am", "i m", "he is", "he s", "she is", "she s","you are",
          "you re", "we are", "we re","they are", "they re")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if (__name__=="__main__"):
    print("gobalSetting begins")
    print (device)

if (__name__!="__main__"):
    print ("Module: globalSetting loaded")
