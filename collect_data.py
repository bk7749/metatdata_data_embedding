#essential libraries
import numpy as np
import pandas as pd
from scipy.cluster.vq import *
import operator
import matplotlib
reload(matplotlib)
matplotlib.use('Agg')
from matplotlib import pyplot as plt   
import pickle as pkl
import shelve
import re
from collections import Counter, defaultdict, OrderedDict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics#v_measure_score
import scipy
from sklearn.feature_extraction import DictVectorizer
from matplotlib.backends.backend_pdf import PdfPages
import csv
import sys
import math
from copy import deepcopy
import random
from datetime import datetime, timedelta
import pickle
import sys

from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import OneClassSVM
from sklearn.mixture import GMM
from sklearn.mixture import DPGMM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB

from bd_wrapper import BDWrapper

def resample_data(rawData, beginTime, endTime, sampleMethod):
	rawData = rawData[beginTime:endTime]
	if not beginTime in rawData.index:
		rawData[beginTime] = rawData.head(1)[0]
		rawData = rawData.sort_index()
	if not endTime in rawData.index:
		rawData[endTime] = rawData.tail(1)[0]
		rawData = rawData.sort_index()
	if sampleMethod== 'nextval':
		procData = rawData.resample('2Min', how='pad')
	return procData


bdm = BDWrapper()
sensors_dict = shelve.open('metadata/bacnet_devices.shelve','r')

#buildingName = 'ebu3b'
buildingName = sys.argv[1]


rawdataDir = 'rawdata/'+buildingName+'/'

naeDict = dict()
naeDict['ebu3b'] = ['505', '506']
deviceList = naeDict[buildingName]

beginTime = datetime(2016,1,18)
endTime = datetime(2016,1,25)

# Init raw set

missingDataList = list()
for nae in deviceList:
	devices = sensors_dict[nae]
	for sensor in devices['objs'][1:]:
		h_obj = sensor['props']
		srcid = nae + '_' + str(h_obj['type']) + '_' + str(h_obj['instance'])
		print srcid
		try:
			uuid = bdm.get_sensor_uuids(context={'source_identifier':srcid})[0]
			ts = bdm.get_sensor_ts(uuid, 'PresentValue', beginTime, endTime)
			resampledTs = resample_data(ts, beginTime, endTime, 'nextval')
			filename = rawdataDir + srcid + '.shelve'
#			with open(filename, 'wb') as fp:
#				pickle.dump(resampledTs, fp)
			writer = shelve.open(filename)
			writer['data'] = resampledTs
		except:
			missingDataList.append(srcid)

print missingDataList
pd.Series(missingDataList).to_csv('missingdatalist.csv')
