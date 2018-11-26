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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if (__name__=="__main__"):
    print("gobalSetting begins")
    print (device)

if (__name__!="__main__"):
    print ("Module: globalSetting loaded")
