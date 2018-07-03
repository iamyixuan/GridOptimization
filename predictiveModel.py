import numpy as np
import pandas as pd
import os
from featureSelection import cleanData, principalComponent
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

filename = 'integratedDataset.csv'
dataset, colName = cleanData(filename)
X = dataset.iloc[:,:42]
y = dataset.iloc[:,42:]




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 42)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#X_train, explainedVarianceRatio, X_test = principalComponent(X_train, X_test)



regressor = MultiOutputRegressor(linear_model.BayesianRidge(lambda_1 = 2))
regressor.fit(X_train, y_train)


y_pred = regressor.predict(X_test)
score1 = regressor.score(X_test,y_test)

print ('The R^2 score is %.4f' % score1)