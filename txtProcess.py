import numpy as np
import pandas as pd
import os

filenames = os.listdir('/home/yixuan/Downloads/Phase_0_IEEE14')
filenames.remove('scorepara.csv')

def extractCertainLines(startLine, endLine, scenarioNum):
	dir_path = '/home/yixuan/Downloads/Phase_0_IEEE14'
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


def extractLines(startLine, endLine, scenarioNum):
	dir_path = '/home/yixuan/Downloads/Phase_0_IEEE14'
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

def makeFeautures(scenarioNum):
	bus_data = extractCertainLines(4, 17, scenarioNum)
	load_data = extractCertainLines(19, 29, scenarioNum)
	fixed_shunt_data = extractCertainLines(31, 31, scenarioNum)
	generator_data = extractCertainLines(33, 37, scenarioNum)
	branch_data = extractCertainLines(39, 55, scenarioNum)
	transformer_data = extractCertainLines(57, 68,scenarioNum)
	feat = [bus_data, load_data, fixed_shunt_data, generator_data,
			branch_data, transformer_data]
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
	feat = makeFeautures(scenarioNum)
	target = makeTargets(scenarioNum)
	tmp = [feat, target]
	final = [item for i in tmp for item in i]
	return final

dataset = []
for names in filenames:
	sample = combineTogether(names)
	dataset.append(sample)




dataframe = pd.DataFrame(dataset)

dataframe.to_csv('integratedDataset.csv')