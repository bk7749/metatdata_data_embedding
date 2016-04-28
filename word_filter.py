import shelve
from datetime import datetime
import re
import pickle
import editdistance
import pandas as pd
import numpy as np
import sys



class WordFilter():
	rawWordList = None
	buildingName = None
	filteredWordMap = None
	naeDict = dict()
	naeList = None
	sensorDict = None
	distDict = None


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
		self.distDict = dict()
		
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
		print "initiated for", self.buildingName, datetime.now()

	def calc_editdistance(self):
		for i, word in enumerate(self.rawWordList):
			for j in range(i+1,len(self.rawWordList)):
				wordee = self.rawWordList[j]
				self.distDict[(word, wordee)] = editdistance.eval(word, wordee)
		with open('data/'+'editdist_'+self.buildingName+'.pkl', 'wb') as fp:
			pickle.dump(sefl.distDict, fp)
		print "finished distance calculation", datetime.now()

	def filter_editdistance(self):
		threshold = 0.65
		distList = list()
		for i, word in enumerate(self.rawWordList):
			for j in range(i+1, len(self.rawWordList)):
				wordee = self.rawWordList[j]
				if len(word)<=2 or len(wordee)<=2:
					continue
				if len(word)>=len(wordee):
					shortWord = word
					longWord = wordee
				else:
					shortWord = wordee
					longWord = word
				if (shortWord, longWord) in self.distDict.keys():
					dist = distDict[(shortWord, longWord)]
				else:
					dist = distDict[(longWord, shortWord)]
				realDist = dist - (len(longWord) - len(shortWord))
				if realDist >= shortWord*threshold:
				 	print shortWord, longWord
					self.filteredWordMap[shortWord] = longWord
					distList.append(realDist)
		with open('data/'+'filteredwords_'+bulidingName+'.pkl', 'wb') as fp:
			pickle.dump(filteredWordMap, fp)
		outputDF = pd.DataFrame()	
		outputDF['keyword'] = self.filteredWordMap.keys()
		outputDF['interpretation'] = self.filteredWordMap.values()
		outputDF['dist'] = realDist
		writer = pd.ExcelWriter('result/'+'filteredwords_'+self.buildingName+'.xlsx')
		outputDF.to_excel(writer, 'Sheet1')
		writer.save()
		print "finished filtering", datetime.now()
					

def main(argv):
	wf = WordFilter('ebu3b')
	wf.calc_editdistance()
	wf.filter_editdistance()


if __name__ == "__main__":
	main(sys.argv)
