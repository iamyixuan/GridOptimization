import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA

def cleanData(filename):
	data = pd.read_csv(filename, header = 0)
	data = data.iloc[:,1:]
	data = data.dropna(axis = 1)
	cols = data.columns
	num_cols = data._get_numeric_data().columns 
	catData = list(set(cols) - set(num_cols))# detecting categorical features.
	data = data.drop(catData, axis = 1) # dropping all categorical features because they are the same for all samples.

	nunique = data.apply(pd.Series.nunique) # find out the repeat data.
	colsToDrop = nunique[nunique == 1].index
	data = data.drop(colsToDrop, axis = 1)# drop out the columns containing the same value.
	columnsNmae = data.columns # store the column name for later indexing
	return data, columnsNmae

	'''
	-------------------------
	now we already reduce the dimension from 800+ to 42 for features
	10 for targets by taking out repeat values sample-wise.

	We will decide if further dismension reduction is needed.
	-------------------------
	'''

def principalComponent(train, test):
	pca = PCA(n_components = 42)
	trainData = pca.fit_transform(train)
	testData = pca.transform(test)
	return trainData, pca.explained_variance_ratio_, testData

