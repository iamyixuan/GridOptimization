import numpy as np
import pandas as pd
import os




path = '/Users/yixuansun/Documents/Research/PNNLrelated/14BUS_Modifited/Phase_0_Modified_IEEE14'
filenames = os.listdir(path)
filenames.remove('scorepara.csv')

def extractCertainLines(startLine, endLine, scenarioNum):
	dir_path = path
	dir_path = os.path.join(dir_path, scenarioNum)
	txt_file = 'powersystem.raw'
	features = []
	with open(os.path.join(dir_path, txt_file)) as file:
		for i, line in enumerate(file):
			if i <= endLine - 1 and i >= startLine - 1:
				newLine = line.strip().replace(' ','').replace("'","").split(',')
				features.append(newLine)
	oneDlist = [item for sublist in features for item in sublist]
	return oneDlist


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


def extractLines(startLine, endLine, scenarioNum):
	dir_path = path
	dir_path = os.path.join(dir_path, scenarioNum)
	txt_file = 'solution1.txt'
	features = []
	with open(os.path.join(dir_path, txt_file)) as file:
		for i, line in enumerate(file):
			if i <= endLine - 1 and i >= startLine - 1:
				newLine = line.strip().replace(' ','').replace("'","").split(',')
				features.append(newLine)
	oneDlist = [item for sublist in features for item in sublist]
	return oneDlist

def integrateFeat(scenarioNum):
	startLine, endLine = findlines(scenarioNum)
	feat = []
	for i in range(len(startLine)):
		feat.append(extractCertainLines(startLine[i], endLine[i], scenarioNum))
	sampleFeatures = [item for sub in feat for item in sub]
	for i, x in enumerate(sampleFeatures):
		try:
			sampleFeatures[i] = float(x)
		except ValueError:
			pass
	return sampleFeatures


def makeTargets(scenarioNum):
	target = extractLines(3,7, scenarioNum)
	indices = [0,1,4,5,8,9,12,13,16,17]
	finalTar = [i for j, i in enumerate(target) if j not in indices]
	finalTar = [float(item) for item in finalTar]
	return finalTar

def combineTogether(scenarioNum):
	feat = integrateFeat(scenarioNum)
	target = makeTargets(scenarioNum)
	tmp = [feat, target]
	final = [item for i in tmp for item in i]
	return final

dataset = []
for names in filenames:
	sample = combineTogether(names)
	dataset.append(sample)




dataframe = pd.DataFrame(dataset)

dataframe.to_csv('integratedDataset_modified.csv')









