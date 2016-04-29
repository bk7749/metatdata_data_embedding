import shelve
from datetime import datetime
from collections import Counter
import re
import pdb
import pickle
import editdistance
import pandas as pd
import numpy as np
import sys
import os
import jellyfish as jf



class WordFilter():
	rawWordList = None
	buildingName = None
	filteredWordMap = None
	naeDict = dict()
	naeList = None
	sensorDict = None


	def __init__ (self, buildingName):
		self.buildingName = buildingName
		self.sensorDict = shelve.open('metadata/bacnet_devices.shelve','r')
		self.rawWordList = list()
		self.filteredWordMap = dict()
		self.naeDict['bonner'] = ["607", "608", "609", "557", "610"]
		self.naeDict['ap_m'] = ['514', '513','604']
		self.naeDict['bsb'] = ['519', '568', '567', '566', '564', '565']
		self.naeDict['ebu3b'] = ["505", "506"]
		self.naeList = self.naeDict[self.buildingName]
		
		srcidSet = set([])
		for nae in self.naeList:
			device = self.sensorDict[nae]
			h_dev = device['props']
			for sensor in device['objs']:
				h_obj = sensor['props']
				source_id = str(h_dev['device_id']) + '_' + str(h_obj['type']) + '_' + str(h_obj['instance'])
				if h_obj['type'] not in (0,1,2,3,4,5,13,14,19):
					continue
				else:
					srcidSet.add(source_id)
					self.rawWordList = self.rawWordList + re.findall("[a-zA-Z]+", sensor['name'])
					self.rawWordList = self.rawWordList + re.findall("[a-zA-Z]+", sensor['jci_name'])
					self.rawWordList = self.rawWordList + re.findall("[a-zA-Z]+", sensor['desc'])
		self.rawWordList = [word.lower() for word in self.rawWordList]
#		self.rawWordList = Counter(self.rawWordList).keys()
		self.rawWordList = [word.decode('unicode-escape') for word in Counter(self.rawWordList).keys()]
		print "initiated for", self.buildingName, datetime.now()

	def calc_editdistance(self, distFunc, metricName):
		distDict = dict()
		for i, word in enumerate(self.rawWordList):
			for j in range(i+1,len(self.rawWordList)):
				wordee = self.rawWordList[j].lower()
				word = word.lower()
				try:
					distDict[(word, wordee)] = distFunc(word, wordee)
				except:
					print word, wordd
					assert(False)

		wordList = [wordTuple[0] for wordTuple in distDict.keys()]
		wordeeList = [wordTuple[1] for wordTuple in distDict.keys()]
		distList = distDict.values()
		prepDF = np.transpose(np.asarray([wordList, wordeeList, distList]))
		pd.DataFrame(prepDF).to_csv('result/'+metricName+'_'+self.buildingName+'.csv')
		with open('data/'+metricName+'_'+self.buildingName+'.pkl', 'wb') as fp:
			pickle.dump(distDict, fp)
		print "finished distance calculation", datetime.now()

	def filter_editdistance(self, distDict, metricName):
		print "start filtering"
		threshold = 0.32
		distList = list()
		for i, word in enumerate(self.rawWordList):
			for j in range(i+1, len(self.rawWordList)):
				wordee = self.rawWordList[j]
				if len(word)<=2 or len(wordee)<=2:
					continue
				if len(word)>len(wordee):
					shortWord = wordee
					longWord = word
				elif len(word)<len(wordee):
					shortWord = word
					longWord = wordee
				else:
					continue
				if (shortWord, longWord) in distDict.keys():
					dist = distDict[(shortWord, longWord)]
				else:
					dist = distDict[(longWord, shortWord)]
				realDist = dist - (len(longWord) - len(shortWord))/2
				if realDist <= len(shortWord)*threshold:
				 	print shortWord, longWord, realDist
#			self.filteredWordMap[shortWord] = longWord
					self.filteredWordMap[(shortWord, longWord)] = realDist
		with open('data/'+'filteredwords_'+metricName+'_'+self.buildingName+'.pkl', 'wb') as fp:
			pickle.dump(self.filteredWordMap, fp)
		outputDF = pd.DataFrame()	
		outputDF['keyword'] = [wordTuple[0] for wordTuple in self.filteredWordMap.keys()]
		outputDF['interpretation'] = [wordTuple[1] for wordTuple in self.filteredWordMap.keys()]
		outputDF['dist'] = self.filteredWordMap.values()
		writer = pd.ExcelWriter('result/'+'filteredwords_'+self.buildingName+'.xlsx')
		outputDF.to_excel(writer, 'Sheet1')
		writer.save()
		print "finished filtering", datetime.now()
					

def main(argv):
	buildingName = 'ebu3b'
	wf = WordFilter(buildingName)
	metricNameList = ['edit', 'damerau', 'jaro', 'jarowinkler', 'hamming']
#	metricNameList = ['damerau', 'jaro', 'jarowinkler', 'hamming']
	metricList = [editdistance.eval, jf.damerau_levenshtein_distance,\
				 jf.jaro_distance, jf.jaro_winkler, jf.hamming_distance]
#	metricList = [jf.damerau_levenshtein_distance,\
	for metric, metricName in zip(metricList, metricNameList):
		print metricName
		filename = 'data/'+metricName+'_'+buildingName+'.pkl'
		if not os.path.isfile(filename):
			wf.calc_editdistance(metric, metricName)
	for metricName in metricNameList:
		filename = 'data/'+metricName+'_'+buildingName+'.pkl'
		print metricName
		with open(filename, 'rb') as fp:
			distDict = pickle.load(fp)
		wf.filter_editdistance(distDict, metricName)


if __name__ == "__main__":
	main(sys.argv)
