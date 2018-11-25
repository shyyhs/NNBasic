from __future__ import unicode_literals, print_function, division

import os
import sys
import time
import math

import glob
import string
import unicodedata
from io import open

def timeSince(begin):
    now = time.time()
    s = now - begin
    return s
