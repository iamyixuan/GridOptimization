import pandas as pd
import numpy as np
import os 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings

filename = 'Area2Full.csv'

data = pd.read_csv(filename)
data = data.sample(frac=1).reset_index(drop=True) #shuffle the rows in dataframe.
X_train, y_train = data.iloc[:,:-33], data.iloc[:,-33:]

color = ['b'] * (len(X_train.columns)-1) + ['r']
'''
The implement of permutation feature importance.
http://explained.ai/rf-importance/index.html
'''

def permutation_importances(rf, X_train, y_train, metric):
    baseline = metric(rf, X_train, y_train)
    imp = []
    for col in X_train.columns:
        save = X_train[col].copy() # save the picked column.
        X_train[col] = np.random.permutation(X_train[col]) # permute the picked column.
        m = metric(rf, X_train, y_train) 
        X_train[col] = save # restore the picked column.
        imp.append(baseline - m)
    return np.array(imp)


def oob_regression_r2_score(rf, X_train, y_train):
    """
    Compute out-of-bag (OOB) R^2 for a scikit-learn random forest
    regressor. We learned the guts of scikit's RF from the BSD licensed
    code:
    https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/ensemble/forest.py#L702
    """
    X = X_train.values
    y = y_train.values

    n_samples = len(X)
    predictions = np.zeros((n_samples,33))
    n_predictions = np.zeros((n_samples,33))
    for tree in rf.estimators_:
        unsampled_indices = _generate_unsampled_indices(tree.random_state, n_samples)
        tree_preds = tree.predict(X[unsampled_indices, :])
        predictions[unsampled_indices] += tree_preds
        n_predictions[unsampled_indices] += 1

    if (n_predictions == 0).any():
        warnings.warn("Too few trees; some variables do not have OOB scores.")
        n_predictions[n_predictions == 0] = 1

    predictions /= n_predictions

    oob_score = r2_score(y, predictions)
    return oob_score


'''
creating rf model and train it.
'''
rf = RandomForestRegressor(n_estimators = 100)
rf.fit(X_train, y_train) # rf must be pre-trained
imp = permutation_importances(rf, X_train, y_train,
                              oob_regression_r2_score)



feat_imp = pd.DataFrame({'importance': imp})
area1_imp = np.mean(feat_imp['importance'].iloc[:167])
area2_imp = np.mean(feat_imp['importance'].iloc[167:334])
area3_imp = np.mean(feat_imp['importance'].iloc[334:])
area_mean_imp = [area1_imp, area2_imp, area3_imp]
print area_mean_imp
#feat_imp.sort_values(by = 'importance', ascending = True, inplace = True)
plt.subplot(121)
plt.barh(range(len(feat_imp)), feat_imp['importance'], 5, color = color, align = 'center')
plt.yticks([0, 167, 334, 501])
plt.xlabel('Feature importance for Area 2')
plt.ylabel('Index')

plt.subplot(122)
plt.barh([1,2,3], area_mean_imp,  color = 'g', align = 'center')
plt.yticks([1, 2, 3])
plt.xlabel('Area Importance Average')
plt.ylabel('Area Number')
plt.tight_layout()
plt.savefig('per_imp_2.jpg', format = 'jpg', dpi = 300)
