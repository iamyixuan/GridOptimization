import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt

filename = 'cleanedArea3.csv'
df = pd.read_csv(filename)
X = df.iloc[:,:-33]
print len(X.columns)
y = df.iloc[:,-33:]






X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 42)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#X_train, explainedVarianceRatio, X_test = principalComponent(X_train, X_test)
# rf1 = RandomForestRegressor(n_estimators = 500, n_jobs = -1, random_state = 42)
# rf1.fit(X, y.iloc[:,0])

regressor = MultiOutputRegressor(RandomForestRegressor())

regressor.fit(X_train, y_train)

score = regressor.score(X_test, y_test)
print score



# feat_imp = pd.DataFrame({'importance': rf1.feature_importances_})
# feat_imp['Feature Index'] = X.columns
# feat_imp.sort_values(by = 'importance', ascending = True, inplace = True)

'''plt.barh(range(len(feat_imp)), feat_imp['importance'], color = 'b', align = 'center')
plt.yticks(range(len(feat_imp)), feat_imp['Feature Index'], size = 6)
plt.xlabel('Feature importance')
plt.ylabel('Index')
plt.tight_layout()

plt.savefig('feat_imp1.jpg', format = 'jpg', dpi = 500)'''