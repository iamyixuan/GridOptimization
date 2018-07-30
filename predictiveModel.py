import numpy as np
import pandas as pd
import os
from featureSelection import cleanData, principalComponent
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt

'''
---------------------------
feature importance plot using 
Random forest for IEEE 14-Bus
---------------------------
'''
filename = 'integratedDataset.csv'
dataset, colName = cleanData(filename)
X = dataset.iloc[:,:42]
y = dataset.iloc[:,42:]
X.columns = ['Load-Bus02-1-P', 'Load-Bus02-1-Q',
			 'Load-Bus03-1-P','Load-Bus03-1-Q', 
			 'Load-Bus04-1-P', 'Load-Bus04-1-Q',
			 'Load-Bus05-1-P', 'Load-Bus05-1-Q',
			 'Load-Bus06-1-P', 'Load-Bus06-1-Q',
			 'Load-Bus09-1-P', 'Load-Bus09-1-Q',
			 'Load-Bus10-1-P', 'Load-Bus10-1-Q',
			 'Load-Bus11-1-P', 'Load-Bus11-1-Q',
			 'Load-Bus12-1-P', 'Load-Bus12-1-Q',
			 'Load-Bus13-1-P', 'Load-Bus13-1-Q',
			 'Load-Bus14-1-P', 'Load-Bus14-1-Q',
			 'Gen-Bus01-1-QMax', 'Gen-Bus01-1-QMin', 'Gen-Bus01-1-PMax', 'Gen-Bus01-1-PMin',
			 'Gen-Bus02-1-QMax', 'Gen-Bus02-1-QMin', 'Gen-Bus02-1-PMax', 'Gen-Bus02-1-PMin',
			 'Gen-Bus03-1-QMax', 'Gen-Bus03-1-QMin', 'Gen-Bus03-1-PMax', 'Gen-Bus03-1-PMin',
			 'Gen-Bus06-1-QMax', 'Gen-Bus06-1-QMin', 'Gen-Bus06-1-PMax', 'Gen-Bus06-1-PMin',
			 'Gen-Bus08-1-QMax', 'Gen-Bus08-1-QMin', 'Gen-Bus08-1-PMax', 'Gen-Bus08-1-PMin']


y.columns = ['Dis-Bus01-1-P', 'Dis-Bus01-1-Q',
			 'Dis-Bus06-1-P', 'Dis-Bus06-1-Q',
			 'Dis-Bus08-1-P', 'Dis-Bus08-1-Q',
			 'Dis-Bus02-1-P', 'Dis-Bus02-1-Q',
			 'Dis-Bus03-1-P', 'Dis-Bus03-1-Q']

'''X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 42)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#X_train, explainedVarianceRatio, X_test = principalComponent(X_train, X_test)


rf1 = RandomForestRegressor(n_estimators = 500, n_jobs = -1, random_state = 42)
rf1.fit(X, y.iloc[:,9])

#regressor = MultiOutputRegressor(linear_model.BayesianRidge())

#regressor.fit(X_train, y
feat_imp = pd.DataFrame({'importance': rf1.feature_importances_})
feat_imp['Feature Index'] = X.columns
#feat_imp.sort_values(by = 'importance', ascending = True, inplace = True)

plt.barh(range(len(feat_imp)), feat_imp['importance'], color = 'b', align = 'center')
plt.yticks(range(len(feat_imp)), feat_imp['Feature Index'], size = 6)
plt.xlabel('Feature importance')
plt.ylabel('Index')
plt.tight_layout()

plt.savefig('feat_imp10.jpg', format = 'jpg', dpi = 500)'''

plt.subplot(221)
plt.scatter(X['Gen-Bus08-1-PMax'], y['Dis-Bus08-1-P'])
plt.xlabel('Gen-Bus08-1-PMax')
plt.ylabel('Dis-Bus08-1-P')

plt.subplot(222)
plt.scatter(X['Gen-Bus08-1-PMin'], y['Dis-Bus08-1-P'])
plt.xlabel('Gen-Bus08-1-PMin')
plt.ylabel('Dis-Bus08-1-P')

plt.subplot(223)
plt.scatter(X['Gen-Bus08-1-QMax'], y['Dis-Bus08-1-Q'])
plt.xlabel('Gen-Bus08-1-QMax')
plt.ylabel('Dis-Bus08-1-Q')

plt.subplot(224)
plt.scatter(X['Gen-Bus08-1-QMin'], y['Dis-Bus08-1-Q'])
plt.xlabel('Gen-Bus08-1-QMin')
plt.ylabel('Dis-Bus08-1-Q')

plt.tight_layout()
plt.savefig('BUS08_scatter.jpg', format = 'jpg', dpi = 500)
plt.show()
