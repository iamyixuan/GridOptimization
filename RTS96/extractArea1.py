import pandas as pd
import numpy as np
import os
from featureSelection import cleanData


path = '/Users/yixuansun/Documents/Research/PNNLrelated/Phase_0_RTS96'
filenames = os.listdir(path)
filenames.remove('scorepara.csv')
filenames.remove('.DS_Store')


def convertingToNum(elem):
	try:
		elem = float(elem)
	except ValueError:
		pass
	return elem


def findlines(scenarioNum):
	dir_path = path
	dir_path = os.path.join(dir_path, scenarioNum)
	txt_file = 'powersystem.raw'
	headerRow = []
	startRow = []
	endRow = []
	with open(os.path.join(dir_path, txt_file)) as file:
		for num, line in enumerate(file):
			line = line.strip()
			if '/' in line:
				headerRow.append(num)
		for start, end in zip(headerRow[:-1], headerRow[1:]):
			startRow.append(start)
			endRow.append(end)
	return	list(np.array(startRow) +1), list(np.array(endRow) - 1)



def extractCertainLines(startLine, endLine, scenarioNum):
	dir_path = path
	dir_path = os.path.join(dir_path, scenarioNum)
	txt_file = 'powersystem.raw'
	features = []
	with open(os.path.join(dir_path, txt_file)) as file:
		for i, line in enumerate(file):
			newLine = line.strip().replace(' ','').replace("'","").split(',')
			newLine = map(convertingToNum, newLine)
			if i <= endLine and i >= startLine  and newLine[0] >= 100 and newLine[0] < 200: # area one will start with 1XX
				features.append(newLine)
	oneDlist = [item for sublist in features for item in sublist]
	return oneDlist





'''
-------------
The following code is to extract contingency info
-------------
'''
def extractContingency(ContNum, scenarioNum):
	fileDir = os.path.join(path, scenarioNum)
	file = pd.read_csv(os.path.join(fileDir, 'contingency.csv'))
	return list(file.iloc[ContNum - 1])

# one contingency for all senarios
def combineContandFeat(ContNum, scenarioNum):
	start, end = findlines(scenarioNum)
	feat = []
	for i in range(len(start)):
		feat.append(extractCertainLines(start[i], end[i], scenarioNum))
	feat.append(extractContingency(ContNum, scenarioNum))
	feat = [item for i in feat for item in i]
	return feat



def ContGenDispatch(ContNum,scenarioNum):
	_path = os.path.join(path, scenarioNum)
	file_path = os.path.join(_path, 'solution2.txt')
	GenDispatch = []
	with open(file_path, 'r') as f:
		for i, line in enumerate(f):
			newLine = line.strip().replace(' ','').replace("'","").split(',')
			newLine = map(convertingToNum, newLine)
			if i >=2 and i <= 991 and newLine[0] == ContNum and newLine[2] >= 100 and newLine[2] < 200:
				GenDispatch.append(newLine[-1])
	return GenDispatch


def feat_target(scenarioNum):
	sample = []
	for i in range(10):
		feat = combineContandFeat(i+1, scenarioNum) # because contingency number starts with 1 not 0.
		target = ContGenDispatch(i+1, scenarioNum)
		sam = feat + target
		sample.append(sam)
	return sample



#creating files containing 1000 samples
# area1Data = []
# for file in filenames:
# 		area1Data.append(feat_target(file))

# area1FullData = [s for item in area1Data for s in item]

# print len(area1FullData), len(area1FullData[0])


# dataframe = pd.DataFrame(area1FullData)

# dataframe.to_csv('area1Raw.csv', index = False)

data = pd.read_csv('area1Raw.csv')
data = data.iloc[:,1:]
data = data.dropna(axis = 1)
cols = data.columns
num_cols = data._get_numeric_data().columns 
catData = list(set(cols) - set(num_cols))# detecting categorical features.
data = data.drop(catData, axis = 1) # dropping all categorical features because they are the same for all samples.

nunique = data.apply(pd.Series.nunique) # find out the repeat data.
colsToDrop = nunique[nunique == 1].index
data = data.drop(colsToDrop, axis = 1)# drop out the columns containing the same value.
print data 
data.to_csv('cleanedArea1.csv', index = False)



















