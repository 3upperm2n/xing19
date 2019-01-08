#!/usr/bin/env python

import os,sys
from sklearn import preprocessing


sys.path.append('../utils')
from cudaMetrics import *


#
# read metrics folder
#
metrics_folder='home-gpu0'
appTrace = os.listdir(metrics_folder)
#print appTrace

#
# adjust the metrics
#




